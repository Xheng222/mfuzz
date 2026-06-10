"""结构归因的最小可行验证（Day 4 探路脚本）。

Day 3 定下了归因目标与结构轴，本脚本在真实失效上验证两种归因手段走得通：

- 归因画像：对差分 oracle 判出的每个失效实例，以 Day 3 规格选目标（cls /
  spurious / agree 用检测分数，loc 用与 consensus 代表框的 IoU），做一次
  梯度×激活反传，把 |grad×act| 按结构桶汇总成归因份额；同时对 FPN 各层级
  特征图求梯度，非零层级即该检测的责任尺度，再取该层级 |grad×act| 的空间
  峰值，检查是否落在失效框内（空间轴的定量验证）。
- 层级消融：把 backbone 输出的某个 FPN 层级置零，重跑差分 oracle，看目标
  模型的四类失效计数相对基线怎么变，得到"层级×失效"的责任矩阵。

漏检没有匹配框，归因目标要从被抑制的候选里取，本 pilot 不做（设计见 Day 4
文档）。判失效的口袋逻辑与 Day 2 一致，仅一处简化：类别错误优先，cls 与
loc 同时成立的实例归入 cls，不再双计。

用法（服务器）：
    uv run python scripts/pilot_struct_attr.py --num-images 40 --target fcos
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from pilot_det_oracle import (
    Detection,
    _build_models,
    _cluster,
    _detect,
    _gather_images,
    _load_image_tensor,
)
from probe_det_structure import _BUCKETS, _bucket_of
from torch import Tensor, nn
from torchvision.ops import box_iou

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_IMAGE_ROOT = _PROJECT_ROOT / "datasets" / "coco" / "val2017"

_KINDS = ("spurious", "cls", "loc", "agree")
_BUCKET_ORDER = list(dict.fromkeys(label for _, label in _BUCKETS))


@dataclass
class Record:
    """目标模型在一个 consensus 对象（或单点簇）上的判定结果。"""

    kind: str  # miss / spurious / cls / loc / agree
    det: Detection | None  # miss 时为 None
    rep_box: Tensor  # consensus 代表框；spurious 用自身框
    cons_label: int


def judge_image(
    dets_by_model: dict[str, list[Detection]], iou_thr: float, loc_thr: float
) -> tuple[dict[str, list[Record]], int]:
    """对一张图的全部检测做差分判定，逐模型给出判定记录。

    逻辑与 pilot_det_oracle 的 run 一致，差别只有一处：cls 与 loc 同时成立时
    归入 cls（归因画像一个实例只挂一个失效类型）。
    """
    all_dets = [d for ds in dets_by_model.values() for d in ds]
    clusters = _cluster(all_dets, iou_thr)
    records: dict[str, list[Record]] = {m: [] for m in dets_by_model}
    n_cons = 0
    for cl in clusters:
        if cl.support >= 2:
            n_cons += 1
            lab = cl.majority_label()
            rep = cl.representative_box()
            for m in dets_by_model:
                mine = [d for d in cl.dets if d.model == m]
                if not mine:
                    records[m].append(Record("miss", None, rep, lab))
                    continue
                d = max(mine, key=lambda d: d.score)
                if d.label != lab:
                    records[m].append(Record("cls", d, rep, lab))
                    continue
                iou = float(box_iou(d.box[None], rep[None])[0, 0])
                kind = "loc" if iou < loc_thr else "agree"
                records[m].append(Record(kind, d, rep, lab))
        else:
            for d in cl.dets:
                records[d.model].append(Record("spurious", d, d.box, d.label))
    return records, n_cons


# ---------- 归因画像 ----------


class GraphForward:
    """带计算图的一次前向：抓全部 Conv2d 激活与 backbone 的 FPN 输出。"""

    def __init__(self, model: nn.Module, name: str) -> None:
        self.model = model
        self.name = name
        self.convs = {n: m for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and n}

    def run(self, img: Tensor, score_thr: float) -> _GraphResult:
        acts: dict[str, list[Tensor]] = {n: [] for n in self.convs}
        feats: dict[str, Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        for n, m in self.convs.items():

            def hook(_m: nn.Module, _i: tuple, out: Tensor, n: str = n) -> None:
                if isinstance(out, Tensor):
                    acts[n].append(out)

            handles.append(m.register_forward_hook(hook))

        def fpn_hook(_m: nn.Module, _i: tuple, out: dict[str, Tensor]) -> None:
            feats.update(out)

        handles.append(self.model.backbone.register_forward_hook(fpn_hook))  # type: ignore[operator]

        x = img.clone().requires_grad_(True)
        try:
            out = self.model([x])[0]
        finally:
            for h in handles:
                h.remove()

        keep = out["scores"] >= score_thr
        boxes_g = out["boxes"][keep]
        scores_g = out["scores"][keep]
        labels = out["labels"][keep].detach().cpu()
        dets = [
            Detection(self.name, boxes_g[i].detach().cpu(), int(labels[i]), float(scores_g[i]))
            for i in range(boxes_g.shape[0])
        ]
        return _GraphResult(dets, boxes_g, scores_g, acts, feats)


@dataclass
class _GraphResult:
    dets: list[Detection]
    boxes_g: Tensor  # (M, 4) 带图
    scores_g: Tensor  # (M,) 带图
    acts: dict[str, list[Tensor]]
    feats: dict[str, Tensor]  # FPN 键 -> 带图特征

    def index_of(self) -> dict[int, int]:
        return {id(d): i for i, d in enumerate(self.dets)}


def _make_target(rec: Record, idx: int, g: _GraphResult, device: torch.device) -> Tensor:
    """按 Day 3 规格选归因目标：loc 用与代表框的 IoU，其余用检测分数。"""
    if rec.kind == "loc":
        rep = rec.rep_box.to(device)
        return box_iou(g.boxes_g[idx][None], rep[None])[0, 0]
    return g.scores_g[idx]


def attribute(
    rec: Record,
    idx: int,
    g: _GraphResult,
    t_size: tuple[int, int],
    orig_size: tuple[int, int],
    device: torch.device,
) -> tuple[dict[str, float], str | None, bool | None]:
    """一次反传，返回（逐桶归因份额，责任 FPN 层级，峰值是否落框内）。"""
    target = _make_target(rec, idx, g, device)
    conv_flat = [(n, t) for n, ts in g.acts.items() for t in ts]
    inputs = [t for _, t in conv_flat] + list(g.feats.values())
    grads = torch.autograd.grad(target, inputs, retain_graph=True, allow_unused=True)

    sums: dict[str, float] = defaultdict(float)
    for (n, t), gr in zip(conv_flat, grads[: len(conv_flat)], strict=True):
        if gr is not None:
            sums[_bucket_of(n)] += float((gr * t).abs().sum())
    total = sum(sums.values())
    shares = {b: v / total for b, v in sums.items()} if total > 0 else {}

    # 责任尺度：FPN 各层级特征的梯度绝对值和，非零（最大）者即是。
    level: str | None = None
    inside: bool | None = None
    best = 0.0
    fkeys = list(g.feats)
    for k, gr in zip(fkeys, grads[len(conv_flat) :], strict=True):
        norm = 0.0 if gr is None else float(gr.abs().sum())
        if norm > best:
            best, level = norm, k
    if level is not None and rec.det is not None:
        gr = grads[len(conv_flat) + fkeys.index(level)]
        assert gr is not None
        ga = (gr * g.feats[level])[0].abs().sum(dim=0)  # (H, W)
        flat_idx = int(ga.argmax())
        py, px = divmod(flat_idx, ga.shape[1])
        # 特征坐标 -> 变换后图坐标 -> 原图坐标（检测框在原图坐标系）
        stride_y, stride_x = t_size[0] / ga.shape[0], t_size[1] / ga.shape[1]
        ix = (px + 0.5) * stride_x * orig_size[1] / t_size[1]
        iy = (py + 0.5) * stride_y * orig_size[0] / t_size[0]
        b = rec.det.box
        inside = bool(b[0] <= ix <= b[2] and b[1] <= iy <= b[3])
    return shares, level, inside


def run_attribution(
    models: dict[str, nn.Module],
    target_name: str,
    paths: list[Path],
    device: torch.device,
    score_thr: float,
    iou_thr: float,
    loc_thr: float,
    agree_per_image: int,
) -> dict[str, dict[str, list[Detection]]]:
    """归因画像主循环。返回各图各模型的基线检测，供消融部分复用。"""
    gf = GraphForward(models[target_name], target_name)
    others = [n for n in models if n != target_name]

    share_acc: dict[str, dict[str, list[float]]] = {k: defaultdict(list) for k in _KINDS}
    level_acc: dict[str, dict[str, int]] = {k: defaultdict(int) for k in _KINDS}
    inside_acc: dict[str, list[bool]] = {k: [] for k in _KINDS}
    base_dets: dict[str, dict[str, list[Detection]]] = {}

    for path in paths:
        img = _load_image_tensor(path, device)
        orig_size = (int(img.shape[-2]), int(img.shape[-1]))
        with torch.no_grad():
            t_hw = models[target_name].transform([img], None)[0].tensors.shape[-2:]  # type: ignore[operator]
        g = gf.run(img, score_thr)
        dets_by_model: dict[str, list[Detection]] = {target_name: g.dets}
        for n in others:
            dets_by_model[n] = _detect(models[n], n, img, score_thr)
        base_dets[str(path)] = dets_by_model

        records, _ = judge_image(dets_by_model, iou_thr, loc_thr)
        idx_of = g.index_of()
        n_agree = 0
        for rec in records[target_name]:
            if rec.kind == "miss" or rec.det is None:
                continue
            if rec.kind == "agree":
                n_agree += 1
                if n_agree > agree_per_image:
                    continue
            shares, level, inside = attribute(
                rec, idx_of[id(rec.det)], g, (int(t_hw[0]), int(t_hw[1])), orig_size, device
            )
            for b, v in shares.items():
                share_acc[rec.kind][b].append(v)
            if level is not None:
                level_acc[rec.kind][level] += 1
            if inside is not None:
                inside_acc[rec.kind].append(inside)
        del g
        torch.cuda.empty_cache()

    _report_attribution(share_acc, level_acc, inside_acc)
    return base_dets


def _report_attribution(
    share_acc: dict[str, dict[str, list[float]]],
    level_acc: dict[str, dict[str, int]],
    inside_acc: dict[str, list[bool]],
) -> None:
    buckets = [b for b in _BUCKET_ORDER if any(b in share_acc[k] for k in _KINDS)]
    n_by_kind = {k: max((len(v) for v in share_acc[k].values()), default=0) for k in _KINDS}

    print("\n" + "=" * 78)
    print("归因份额（|grad×act| 逐桶占比的均值，行为失效类型）")
    print(f"{'类型':<10}{'n':>4}" + "".join(f"{b:>13}" for b in buckets))
    for k in _KINDS:
        if n_by_kind[k] == 0:
            continue
        row = ""
        for b in buckets:
            vals = share_acc[k].get(b)
            row += f"{(sum(vals) / len(vals)):>13.3f}" if vals else f"{'-':>13}"
        print(f"{k:<10}{n_by_kind[k]:>4}{row}")

    levels = sorted({lv for d in level_acc.values() for lv in d})
    print("\n责任尺度（按 FPN 特征梯度判定的层级计数）")
    print(f"{'类型':<10}" + "".join(f"{lv:>8}" for lv in levels))
    for k in _KINDS:
        if not level_acc[k]:
            continue
        print(f"{k:<10}" + "".join(f"{level_acc[k].get(lv, 0):>8}" for lv in levels))

    print("\n梯度峰值落在检测框内的比例")
    for k in _KINDS:
        if inside_acc[k]:
            frac = sum(inside_acc[k]) / len(inside_acc[k])
            print(f"  {k:<10}{frac:.2f}  (n={len(inside_acc[k])})")
    print("=" * 78)


# ---------- 层级消融 ----------


def run_ablation(
    models: dict[str, nn.Module],
    target_name: str,
    paths: list[Path],
    base_dets: dict[str, dict[str, list[Detection]]],
    device: torch.device,
    score_thr: float,
    iou_thr: float,
    loc_thr: float,
) -> None:
    """逐 FPN 层级置零目标模型的 backbone 输出，重判失效，看计数怎么变。"""
    target = models[target_name]
    img0 = _load_image_tensor(paths[0], device)
    with torch.no_grad():
        t0 = target.transform([img0], None)[0].tensors  # type: ignore[operator]
        fkeys = list(target.backbone(t0).keys())  # type: ignore[operator]

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
    for key in fkeys:

        def zero_hook(_m: nn.Module, _i: tuple, out: dict[str, Tensor], key: str = key):
            out[key] = torch.zeros_like(out[key])
            return out

        handle = target.backbone.register_forward_hook(zero_hook)  # type: ignore[operator]
        try:
            ablated: dict[str, list[Detection]] = {}
            for path in paths:
                img = _load_image_tensor(path, device)
                ablated[str(path)] = _detect(target, target_name, img, score_thr)
        finally:
            handle.remove()
        rows.append((f"置零 {key}", tally(lambda p, d=ablated: d[str(p)])))

    print("\n" + "=" * 78)
    print(f"层级消融（{target_name}，重判 {len(paths)} 张图的差分失效）")
    cols = ("miss", "spurious", "cls", "loc", "agree", "consensus")
    print(f"{'条件':<12}" + "".join(f"{c:>10}" for c in cols))
    for name, counts in rows:
        print(f"{name:<12}" + "".join(f"{counts.get(c, 0):>10}" for c in cols))
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="结构归因 pilot：归因画像 + 层级消融")
    ap.add_argument("--image-dir", type=Path, default=_DEFAULT_IMAGE_ROOT)
    ap.add_argument("--num-images", type=int, default=40)
    ap.add_argument("--target", type=str, default="fcos", help="归因的目标模型")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--loc-thr", type=float, default=0.7)
    ap.add_argument("--agree-per-image", type=int, default=2, help="每图采样的 agree 对照数")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _build_models(device)
    assert args.target in models, f"目标模型须是 {list(models)} 之一"
    paths = _gather_images(args.image_dir, args.num_images)
    print(f"设备：{device}  目标模型：{args.target}  图像：{len(paths)} 张")
    print(f"阈值：score>={args.score_thr}  iou>={args.iou_thr}  loc<{args.loc_thr}")

    base_dets = run_attribution(
        models,
        args.target,
        paths,
        device,
        args.score_thr,
        args.iou_thr,
        args.loc_thr,
        args.agree_per_image,
    )
    if not args.skip_ablation:
        run_ablation(
            models,
            args.target,
            paths,
            base_dets,
            device,
            args.score_thr,
            args.iou_thr,
            args.loc_thr,
        )


if __name__ == "__main__":
    main()
