"""定向微调修复的机制层：只解冻责任子网、做多步定向更新。

前两步（抑制 run_repair_pilot、单步权重编辑 run_repair_grad）都是无训练的，
对几何型的定位偏移（loc）和类别竞争型的误分类（cls）都修不动。这一步换成
真正的多步微调：把目标模型整体冻结，只对责任子网（回归头或分类头的全部
Conv2d）或对照层重新打开 requires_grad，建优化器只收这些参数，在 A 集失效
样本上算损失做多步更新，再到不相交的 B 集上用差分判定评测。

两类失效共用同一套训练循环与评测，只在损失与监督信号处分叉：

- loc 损失 = 1 - IoU，作用在 NMS 后的预测框 boxes_g 与共识代表框之间，复用
  struct_attr._make_target 的 loc 取框逻辑。
- cls 损失 = 对共识正类与模型当前错类两个通道的 BCEWithLogits（正类目标 1、
  错类目标 0），作用在 NMS 前的逐 anchor 分类 logit 上。NMS 前 logit 这条路径
  需要新写：归因阶段拿的是 NMS 后检测，这里要从 cls 失效实例回找它在 NMS 前
  对应的 anchor 行与 logit 向量，与归因 cls_logit_target 同一个尺度。

本模块只放机制，编排在 scripts/run_repair_finetune.py。FCOS 与 RetinaNet 的
检测头结构在 transform→backbone→head→anchor_generator→postprocess 这条链上
完全一致，cls logit 回找对两个模型共用一套代码。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.ops import box_iou

from mfuzz.core.det_models import TorchvisionDetector
from mfuzz.differential.det_oracle import DetRecord

# ---------- 责任子网 / 对照层的参数解冻 ----------


def subnet_conv_weights(
    adapter: TorchvisionDetector, prefix: str, whole_subnet: bool
) -> dict[str, Tensor]:
    """收集责任子网（或单层）里 Conv2d 的权重张量，键是模块名。

    whole_subnet=True：prefix 下的全部 Conv2d 权重（回归头/分类头的 conv 塔加
    末层）。whole_subnet=False：只取 prefix 这一层自身的 Conv2d 权重（消融变体，
    或对照层那种单层配置）。conv_layers() 已经把模型里所有 Conv2d 按名收齐，这里
    按前缀筛。
    """
    convs = adapter.conv_layers()
    if whole_subnet:
        out = {n: m.weight for n, m in convs.items() if n == prefix or n.startswith(prefix + ".")}
    else:
        out = {n: m.weight for n, m in convs.items() if n == prefix}
    if not out:
        raise ValueError(f"模块前缀 {prefix!r} 下没有 Conv2d，检查路径或 whole_subnet 开关")
    return out


def set_trainable(weights: dict[str, Tensor], flag: bool) -> None:
    """统一开关一组权重的 requires_grad。调用方负责加载后先全冻、用完再冻回去。"""
    for w in weights.values():
        w.requires_grad_(flag)


def weight_norm_scale(weights: dict[str, Tensor]) -> float:
    """这组权重当前的总范数，学习率按它归一化用。

    与 run_repair_grad.masked_step 按权重范数定步长同一思路：把同一个学习率档
    乘以这个范数，让相对扰动幅度在回归头/分类头、两个模型、责任子网/单层对照
    之间可比，而不是被各自的权重尺度带偏。返回标量。
    """
    sq = sum(float((w.detach() ** 2).sum()) for w in weights.values())
    return sq**0.5


# ---------- loc 损失：NMS 后预测框对共识代表框的 1 - IoU ----------


def loc_loss(boxes_g: Tensor, idx: int, rep_box: Tensor, device: torch.device) -> Tensor:
    """单个 loc 失效实例的损失：1 - IoU(预测框, 共识代表框)。

    boxes_g[idx] 是带图的 NMS 后预测框（GraphResult 给出的行号），rep_box 是差分
    判据记下的共识代表框。复用 struct_attr._make_target 的 loc 取框逻辑，只是这里
    取 1 - IoU 当损失、把偏移框拉回代表框。
    """
    rep = rep_box.to(device)
    iou = box_iou(boxes_g[idx][None], rep[None])[0, 0]
    return 1.0 - iou


# ---------- cls 损失：NMS 前逐 anchor 分类 logit 的回找路径 ----------


class HeadLogits:
    """对 torchvision 检测器做一次带图的"半前向"，停在 NMS 前的 head 输出。

    走 transform→backbone→head→anchor_generator 这条链，不进 postprocess/NMS，
    拿到带计算图的逐 anchor 分类 logit、解码到原图坐标的预测框、逐 anchor 的预测
    分数（按各家族后处理的算法算，FCOS 是 sqrt(sigmoid(cls)·sigmoid(ctrness))，
    RetinaNet 是 sigmoid(cls)）。cls 失效实例用 recover_cls_logit 按框 IoU 加类别
    回找到对应 anchor 行，再在这一行的 logit 向量上算损失，避开 sigmoid 饱和、与
    归因阶段 cls_logit_target 同一个尺度。

    FCOS 与 RetinaNet 这条链结构一致，只是 box_coder 解码接口与分数公式不同，这里
    按家族分支处理；faster_rcnn 不是这套单阶段 head，不支持。
    """

    def __init__(self, adapter: TorchvisionDetector) -> None:
        if adapter.name not in ("fcos", "retinanet"):
            raise ValueError(f"HeadLogits 只支持 fcos / retinanet，收到 {adapter.name!r}")
        self.adapter = adapter
        self.model = adapter.model

    def run(self, img: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """一次半前向。返回 (cls_logits, boxes_orig, scores)：

        - cls_logits: (A, num_classes) 带图，A 是全部 anchor 数（跨 FPN 层级拼接）。
        - boxes_orig: (A, 4) 每个 anchor 解码后的预测框，已 resize 回原图坐标，detach。
        - scores: (A,) 每个 anchor 取分数最高类别的预测分数（家族后处理公式），detach。

        boxes_orig 与 scores 只用来把失效实例回找到 anchor 行，不进损失；损失只读
        cls_logits。
        """
        from torchvision.models.detection.transform import resize_boxes

        model = self.model
        orig_hw = (int(img.shape[-2]), int(img.shape[-1]))
        image_list, _ = model.transform([img], None)
        t_hw = image_list.image_sizes[0]
        feats = model.backbone(image_list.tensors)
        if isinstance(feats, Tensor):
            from collections import OrderedDict

            feats = OrderedDict([("0", feats)])
        feat_list = list(feats.values())
        head_out = model.head(feat_list)
        anchors = model.anchor_generator(image_list, feat_list)[0]  # (A, 4) 单图

        cls_logits = head_out["cls_logits"][0]  # (A, num_classes) 带图
        box_reg = head_out["bbox_regression"][0]  # (A, 4)

        if self.adapter.name == "fcos":
            boxes_t = model.box_coder.decode(box_reg, anchors)
            ctr = head_out["bbox_ctrness"][0].squeeze(-1)
            prob = torch.sqrt(torch.sigmoid(cls_logits) * torch.sigmoid(ctr)[:, None])
        else:
            boxes_t = model.box_coder.decode_single(box_reg, anchors)
            prob = torch.sigmoid(cls_logits)

        scores, _ = prob.detach().max(dim=1)
        boxes_t = boxes_t.detach()
        boxes_orig = resize_boxes(boxes_t, list(t_hw), list(orig_hw))
        return cls_logits, boxes_orig, scores


def recover_cls_logit(
    cls_logits: Tensor,
    boxes_orig: Tensor,
    scores: Tensor,
    rec: DetRecord,
    categories: list[str],
    iou_thr: float,
    score_thr: float,
) -> tuple[Tensor, int, int] | None:
    """把一个 cls 失效实例回找到 NMS 前的 anchor 行，给出该行的 logit 向量与两个
    通道索引。

    cls 失效记录里有观测框 rec.det.box（原图坐标）、模型预测的错类别 rec.det.label、
    共识正确类别 rec.cons_label。回找的判据：在分数达标的 anchor 里，取预测类别名
    与错类别一致、且解码框与观测框 IoU 最高的那一行。返回 (logit_vec, cons_idx,
    wrong_idx)，logit_vec 带图、用于 BCEWithLogits；cons_idx/wrong_idx 是这两个类别
    在 categories（含占位项的 91 类）里的索引。回找不到时返回 None（PNG 量化、
    阈值边界等原因，调用方跳过并计数）。
    """
    assert rec.det is not None
    name_to_idx = {c: i for i, c in enumerate(categories)}
    wrong_idx = name_to_idx.get(rec.det.label)
    cons_idx = name_to_idx.get(rec.cons_label)
    if wrong_idx is None or cons_idx is None:
        return None

    obs = rec.det.box.to(boxes_orig.device)
    # 在分数达标且与观测框重叠达标的 anchor 里，按错类通道 logit 挑来源 anchor。
    keep = (scores >= score_thr).nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        return None
    ious = box_iou(boxes_orig[keep], obs[None])[:, 0]
    ok = (ious >= iou_thr).nonzero(as_tuple=True)[0]
    if ok.numel() == 0:
        return None
    cand = keep[ok]
    # 在与观测框重叠达标的 anchor 里，取错类通道 logit 最高者：它最可能是 NMS 后
    # 那个错类检测的来源 anchor。
    best = cand[int(cls_logits[cand, wrong_idx].argmax())]
    return cls_logits[best], cons_idx, wrong_idx


def cls_loss(
    logit_vec: Tensor,
    cons_idx: int,
    wrong_idx: int,
    push_down_wrong: bool,
) -> Tensor:
    """单个 cls 失效实例的损失：对共识正类与当前错类两个通道的 BCEWithLogits。

    正类目标 1（把 cons_label 的 logit 推高），错类目标 0（把当前错类 logit 压低）。
    只作用在这两个通道上，不动其它类别。push_down_wrong=False 时退化为只推正类
    （留作变体）。logit_vec 带图。
    """
    bce = nn.functional.binary_cross_entropy_with_logits
    pos = bce(logit_vec[cons_idx], torch.ones((), device=logit_vec.device))
    if not push_down_wrong or wrong_idx == cons_idx:
        return pos
    neg = bce(logit_vec[wrong_idx], torch.zeros((), device=logit_vec.device))
    return pos + neg


# ---------- 对照层选取 ----------


def _head_branch_prefixes() -> tuple[str, ...]:
    """两个 head 分支的模块前缀。随机对照层从可微卷积层里剔除这两支，避免抽到
    另一个 head 分支引入语义关联。"""
    return ("head.classification_head", "head.regression_head")


def pick_random_control(adapter: TorchvisionDetector, seed: int) -> str:
    """随机对照层：从 conv_layers() 里剔除两个 head 分支后，固定种子抽一层。"""
    branches = _head_branch_prefixes()
    pool = [
        n
        for n in adapter.conv_layers()
        if not any(n == b or n.startswith(b + ".") for b in branches)
    ]
    if not pool:
        raise ValueError("剔除 head 分支后没有可选的卷积层")
    pool.sort()
    gen = torch.Generator().manual_seed(seed)
    pick = int(torch.randint(len(pool), (1,), generator=gen))
    return pool[pick]


def pick_bottom_control(drilldown: list[dict] | None) -> str | None:
    """最低归因对照层：在该失效类的层级下钻表里取归因比值最低的一层。

    drilldown 是 det_analysis._layer_drilldown 对某一类失效给出的行列表，每行有
    layer 与 ratio（相对 agree 的归因比值），表按 ratio 降序。取末行即比值最低的
    层，它是归因明确指认"最不该负责"的层，构成清晰的零对照。表为空或缺失时返回
    None，由调用方决定是否跳过这条对照。
    """
    if not drilldown:
        return None
    rows = sorted(drilldown, key=lambda r: r["ratio"])
    return rows[0]["layer"]


# ---------- 责任子网模块路径 ----------

# 每个（模型, 失效类）对应的责任子网模块前缀。loc 用回归分支、cls 用分类分支。
# FCOS 与 RetinaNet 的 head 命名一致（head.regression_head / head.classification_head），
# 内部 conv 塔的 Conv2d 索引不同（FCOS 在 conv.0/3/6/9，RetinaNet 在 conv.{i}.0），
# subnet_conv_weights 按前缀收集，不依赖具体索引。faster_rcnn 的 cls 子网是 Linear，
# 不在这里。
SUBNET_PREFIX: dict[tuple[str, str], str] = {
    ("fcos", "loc"): "head.regression_head",
    ("fcos", "cls"): "head.classification_head",
    ("retinanet", "loc"): "head.regression_head",
    ("retinanet", "cls"): "head.classification_head",
}

# 主责任层（消融变体 whole_subnet=False 时用这一层）。loc 的主责任层是框架文档
# 反复指认的 conv.0；cls 的分类子网在通道级不可分、给它整子网才有意义，单层变体
# 不是 cls 试点的主看点，这里给一个对齐的入口层。
MAIN_LAYER: dict[tuple[str, str], str] = {
    ("fcos", "loc"): "head.regression_head.conv.0",
    ("fcos", "cls"): "head.classification_head.conv.0",
    ("retinanet", "loc"): "head.regression_head.conv.0.0",
    ("retinanet", "cls"): "head.classification_head.conv.0.0",
}


def subnet_target(
    adapter: TorchvisionDetector, kind: str, whole_subnet: bool
) -> tuple[str, dict[str, Tensor]]:
    """按（模型, 失效类, 解冻粒度）给出责任子网的（说明前缀, 待训练权重表）。

    whole_subnet=True 解冻整个责任子网（主试点），False 只解冻主责任层（消融变体）。
    """
    key = (adapter.name, kind)
    if key not in SUBNET_PREFIX:
        raise ValueError(f"没有为 {key} 定义责任子网，cls/loc 只支持 fcos/retinanet")
    if whole_subnet:
        prefix = SUBNET_PREFIX[key]
        return prefix, subnet_conv_weights(adapter, prefix, whole_subnet=True)
    layer = MAIN_LAYER[key]
    return layer, subnet_conv_weights(adapter, layer, whole_subnet=False)
