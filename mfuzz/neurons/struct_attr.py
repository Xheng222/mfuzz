"""结构归因：把单个失效实例定位到模型的结构位置。

Day 3 定规格、Day 4 跑通 pilot 之后的正式实现。对差分 oracle 判出的每个
失效实例做一次梯度×激活反传，给出三个读数：

- 逐结构桶的归因份额：|grad×act| 按结构桶汇总并归一化。
- 责任尺度：FPN 各层级特征的梯度绝对值和，非零最大者即该检测的责任层级，
  报成 P 层级名（按 stride 换算，跨模型可比）。
- 空间峰值落框：责任层级上 |grad×act| 的空间峰值换算回原图坐标，检查是否
  落在失效框（miss 用 consensus 代表框）内。

归因目标按失效类型选：spurious / agree 用检测分数，loc 用与代表框的 IoU（对框
坐标可微），cls 用 NMS 前分类 logit（由后处理概率经 logit 链接还原，避开高置信
处 sigmoid/softmax 的梯度饱和）。miss 没有匹配框，从未达分数阈值的候选里取：与代表框
IoU 达标的候选中取分数最高者，以它的分数为目标；一个候选都没有的记为深漏
（候选在更早阶段就没了，本模块归因不了）。

层级消融是干预式的对照证据：把 backbone 输出的某个 FPN 层级置零，重跑差分
判定，看四类失效计数相对基线怎么变。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.ops import box_iou

from mfuzz.core.det_models import AnyDetector
from mfuzz.core.types import Detection
from mfuzz.differential.det_oracle import DetRecord, judge_image

# ---------- 带计算图的前向 ----------


@dataclass
class GraphResult:
    """一次带图前向的全部产物。

    boxes_g / scores_g 覆盖模型内部阈值以上的全部输出（含未达 score_thr 的
    低分候选），dets 只含达标检测。低分候选供漏检归因取目标。
    """

    dets: list[Detection]
    kept_idx: list[int]  # dets[i] 对应 boxes_g[kept_idx[i]]
    boxes_g: Tensor  # (M, 4) 带图
    scores_g: Tensor  # (M,) 带图
    labels: list[str]  # 全部 M 个输出的类别名
    acts: dict[str, list[Tensor]]  # 层名 -> 各次调用的输出（共享 head 逐层级调用）
    feats: dict[str, Tensor]  # FPN 键 -> 带图特征

    def graph_index(self) -> dict[int, int]:
        """id(Detection) -> 在 boxes_g/scores_g 里的行号。"""
        return {id(d): self.kept_idx[i] for i, d in enumerate(self.dets)}


class GraphForward:
    """带计算图的一次前向：抓全部 Conv2d 激活，家族特定的解码与多尺度特征由
    适配器的 forward_graph 提供（torchvision 走 backbone FPN，YOLO 走 letterbox
    解码）。任意实现了 forward_graph 的适配器都能当轮换目标。"""

    def __init__(self, adapter: AnyDetector) -> None:
        self.adapter = adapter
        self.convs = adapter.conv_layers()

    def run(self, img: Tensor, score_thr: float) -> GraphResult:
        acts: dict[str, list[Tensor]] = {n: [] for n in self.convs}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        for n, m in self.convs.items():

            def hook(_m: nn.Module, _i: tuple, out: Tensor, n: str = n) -> None:
                if isinstance(out, Tensor):
                    acts[n].append(out)

            handles.append(m.register_forward_hook(hook))

        # 调用方传入已带梯度的输入时直接用（fuzzing 对输入求梯度），否则建独立叶子。
        # 卷积 hook 在适配器前向期间触发，feats 由适配器抓取（家族不同结构不同）。
        x = img if img.requires_grad else img.clone().requires_grad_(True)
        try:
            boxes_g, scores_g, labels, feats = self.adapter.forward_graph(x)
        finally:
            for h in handles:
                h.remove()

        scores_d = scores_g.detach()
        kept_idx = [i for i in range(scores_d.shape[0]) if float(scores_d[i]) >= score_thr]
        dets = [
            Detection(
                self.adapter.name,
                boxes_g[i].detach().cpu(),
                labels[i],
                float(scores_g[i].detach()),
            )
            for i in kept_idx
        ]
        return GraphResult(dets, kept_idx, boxes_g, scores_g, labels, acts, feats)


# ---------- 归因 ----------


@dataclass
class AttrResult:
    shares: dict[str, float]  # 结构桶 -> 归因份额
    level: str | None  # 责任尺度的 P 层级名
    inside: bool | None  # 空间峰值是否落在失效框内
    peak_xy: tuple[float, float] | None = None  # 空间峰值的原图坐标
    heatmap: Tensor | None = None  # 责任层级的 |grad×act| 空间图（已 detach 到 CPU），可视化用
    cand_score: float | None = None  # miss：所取候选的分数
    cand_label_match: bool | None = None  # miss：候选类别名是否等于 consensus 类别
    layer_shares: dict[str, float] | None = None  # 卷积层 -> 归因份额（桶的下钻，目标模型内定位）


def find_miss_candidate(
    g: GraphResult, rep_box: Tensor, cons_label: str, iou_thr: float, score_thr: float
) -> tuple[int, bool] | None:
    """漏检归因的候选：未达 score_thr 但与代表框 IoU 达标的输出。

    同类别名的候选优先，其中取分数最高者；没有同类候选时退回任意类别的最高
    分。返回（行号，类别是否匹配），完全没有候选时返回 None。
    """
    scores_d = g.scores_g.detach()
    low = [i for i in range(scores_d.shape[0]) if float(scores_d[i]) < score_thr]
    if not low:
        return None
    boxes = g.boxes_g[low].detach()
    ious = box_iou(boxes, rep_box.to(boxes.device)[None])[:, 0]
    hits = [(low[k], g.labels[low[k]]) for k in range(len(low)) if float(ious[k]) >= iou_thr]
    if not hits:
        return None
    same = [i for i, lab in hits if lab == cons_label]
    pool = same if same else [i for i, _ in hits]
    best = max(pool, key=lambda i: float(scores_d[i]))
    return best, bool(same)


# cls 归因目标改用 NMS 前 logit 的等价量。scores_g 是后处理概率（faster_rcnn
# softmax、retinanet/fcos sigmoid，fcos 还乘 centerness），高置信时 sigmoid/softmax
# 导数 σ(1-σ) 趋零，压低 grad×act，使误分类的责任通道与随机对照难以区分。logit
# 链接 log(p/(1-p)) 对概率单调递增，其导数 1/(p(1-p)) 恰好抵消这一饱和因子，
# 等价于把归因目标换到 NMS 前的分类 logit 尺度，且对三个家族统一、不依赖各自的
# 后处理重索引。两端做 _LOGIT_EPS 截断，避免 p→0 或 p→1 时数值发散。
_LOGIT_EPS = 1e-4


def cls_logit_target(score: Tensor) -> Tensor:
    """把带图的概率分数换成 logit 尺度目标：logit(p) = log(p) - log(1 - p)。"""
    p = score.clamp(_LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return torch.log(p) - torch.log1p(-p)


def _make_target(rec: DetRecord, idx: int, g: GraphResult, device: torch.device) -> Tensor:
    """按失效类型选归因目标：loc 用与代表框的 IoU，cls 用 NMS 前 logit（由概率
    经 logit 链接还原），其余用检测/候选分数。"""
    if rec.kind == "loc":
        rep = rec.rep_box.to(device)
        return box_iou(g.boxes_g[idx][None], rep[None])[0, 0]
    if rec.kind == "cls":
        return cls_logit_target(g.scores_g[idx])
    return g.scores_g[idx]


def attribute(
    rec: DetRecord,
    idx: int,
    g: GraphResult,
    bucket_of,
    t_size: tuple[int, int],
    orig_size: tuple[int, int],
    device: torch.device,
    keep_heatmap: bool = False,
    pad: tuple[float, float, float] | None = None,
) -> AttrResult:
    """一次反传，返回逐桶归因份额、责任 P 层级、峰值是否落框。

    idx 是归因目标在 boxes_g/scores_g 里的行号；miss 时由
    find_miss_candidate 给出，其余由 GraphResult.graph_index 给出。

    pad 是 letterbox 的 (left, top, r)：给定时空间峰值按 (padded - 偏移) / r 换算回
    原图（YOLO 居中补边）；为 None 时按 orig/t 比例换算（torchvision 左上对齐）。
    """
    target = _make_target(rec, idx, g, device)
    conv_flat = [(n, t) for n, ts in g.acts.items() for t in ts]
    inputs = [t for _, t in conv_flat] + list(g.feats.values())
    grads = torch.autograd.grad(target, inputs, retain_graph=True, allow_unused=True)

    sums: dict[str, float] = defaultdict(float)
    lsums: dict[str, float] = defaultdict(float)
    for (n, t), gr in zip(conv_flat, grads[: len(conv_flat)], strict=True):
        if gr is not None:
            v = float((gr * t.detach()).abs().sum())
            sums[bucket_of(n)] += v
            lsums[n] += v
    total = sum(sums.values())
    shares = {b: v / total for b, v in sums.items()} if total > 0 else {}
    layer_shares = {n: v / total for n, v in lsums.items() if v > 0} if total > 0 else {}

    # 责任尺度：FPN 各层级特征的梯度绝对值和，非零最大者即是。
    level_key: str | None = None
    best = 0.0
    fkeys = list(g.feats)
    for k, gr in zip(fkeys, grads[len(conv_flat) :], strict=True):
        norm = 0.0 if gr is None else float(gr.abs().sum())
        if norm > best:
            best, level_key = norm, k

    level: str | None = None
    inside: bool | None = None
    peak_xy: tuple[float, float] | None = None
    heatmap: Tensor | None = None
    if level_key is not None:
        feat = g.feats[level_key]
        stride_y = t_size[0] / feat.shape[-2]
        level = f"P{round(math.log2(stride_y))}"
        # 空间峰值 -> 原图坐标。miss 没有自身框，对照 consensus 代表框。
        box = rec.det.box if rec.det is not None else rec.rep_box
        gr = grads[len(conv_flat) + fkeys.index(level_key)]
        assert gr is not None
        ga = (gr * feat.detach())[0].abs().sum(dim=0)  # (H, W)
        flat_idx = int(ga.argmax())
        py, px = divmod(flat_idx, ga.shape[1])
        stride_x = t_size[1] / ga.shape[1]
        px_pix = (px + 0.5) * stride_x  # 责任层级上的峰值，padded 坐标系
        py_pix = (py + 0.5) * stride_y
        if pad is None:
            ix = px_pix * orig_size[1] / t_size[1]
            iy = py_pix * orig_size[0] / t_size[0]
        else:
            left, top, r = pad
            ix = (px_pix - left) / r
            iy = (py_pix - top) / r
        peak_xy = (ix, iy)
        inside = bool(box[0] <= ix <= box[2] and box[1] <= iy <= box[3])
        if keep_heatmap:
            heatmap = ga.detach().cpu()
    return AttrResult(shares, level, inside, peak_xy, heatmap, layer_shares=layer_shares)


# ---------- 层级消融 ----------


def ablate_levels(
    adapter: AnyDetector,
    paths: list,
    base_dets: dict[str, dict[str, list[Detection]]],
    load_image,
    iou_thr: float,
    loc_thr: float,
    score_thr: float,
) -> list[tuple[str, dict[str, int]]]:
    """逐 FPN 层级置零，重判差分失效，返回层级×失效计数表。

    base_dets 是各图各模型的基线检测；消融只换目标模型这一路，参考模型沿用
    基线。置零的具体机制由适配器的 level_zero_hook 提供（torchvision 置零
    backbone 输出层级，YOLO 置零 Detect 头消费层）。行名用 P 层级名。
    """
    target_name = adapter.name
    img0 = load_image(paths[0])
    fpn_map = adapter.fpn_info(img0)  # 键 -> P 层级名

    def tally(getter) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        n_cons = 0
        for path in paths:
            dets_by_model = dict(base_dets[str(path)])
            dets_by_model[target_name] = getter(path)
            records, nc = judge_image(dets_by_model, iou_thr, loc_thr)
            n_cons += nc
            for rec in records[target_name]:
                counts[rec.kind] += 1
        counts["consensus"] = n_cons
        return counts

    rows = [("baseline", tally(lambda p: base_dets[str(p)][target_name]))]
    for key, plabel in fpn_map.items():
        handle = adapter.level_zero_hook(key)
        try:
            ablated: dict[str, list[Detection]] = {}
            for path in paths:
                img = load_image(path)
                ablated[str(path)] = adapter.detect(img, score_thr)
        finally:
            handle.remove()
        rows.append((plabel, tally(lambda p, d=ablated: d[str(p)])))
    return rows
