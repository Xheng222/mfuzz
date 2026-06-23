"""检测任务的分析阶段库：基线检测、逐实例结构归因、聚合、层级消融、生成失效归因。

检测适配器（tasks/detection.py）的 analyze 阶段调用这里。逻辑承自探路脚本
scripts/run_struct_analysis.py，参数显式传入、不依赖配置对象，便于离线复用。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torchvision.ops import box_iou
from torchvision.transforms.v2 import functional as TF

from mfuzz.core.det_models import AnyDetector
from mfuzz.core.records import FailureRecord
from mfuzz.core.types import Detection
from mfuzz.differential.det_oracle import RECORD_KINDS, DetRecord, judge_image
from mfuzz.neurons.struct_attr import (
    GraphForward,
    ablate_levels,
    attribute,
    find_miss_candidate,
)

_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.JPG", "*.PNG")


@dataclass
class DetParams:
    """检测任务的全部自有参数，从 cfg.raw["detection"] 解析。"""

    image_dir: str = "datasets/coco/val2017"
    annotations: str = "datasets/coco/annotations/instances_val2017.json"
    num_images: int = 200
    # 覆盖标定（criticality 频率统计）应取训练分布的大样本（CriticalFuzz 在完整
    # 训练集上算 cl）。profile_dir 为空时退回 image_dir 尾部取图，那是本地小数据
    # 集的兼容路径，标定样本量受限，仅供冒烟。
    profile_dir: str = ""
    profile_images: int = 0  # <=0 = 标定集全量
    cache_dir: str = "output/cache/profiles"  # 标定统计缓存；空串关闭缓存
    save_failures: int = 400
    score_thr: float = 0.5
    iou_thr: float = 0.5
    loc_thr: float = 0.7
    agree_per_image: int = 2
    viz_per_kind: int = 8
    gen_attr_per_kind: int = 50
    gt: bool = True
    ablation: bool = True

    @classmethod
    def from_raw(cls, raw: dict) -> DetParams:
        d = dict(raw.get("detection", {}))
        flat = {**d.pop("oracle", {}), **d.pop("attr", {}), **d.pop("stages", {}), **d}
        return cls(**flat)


def gather_images(root: str | Path, n: int, offset: int = 0) -> list[Path]:
    """取目录下按名排序的前 n 张图；n <= 0 表示全量。"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"找不到图像目录：{root}")
    paths: list[Path] = []
    for ext in _IMAGE_EXTS:
        paths += sorted(root.glob(ext))
    paths = sorted(paths)
    return paths[offset:] if n <= 0 else paths[offset : offset + n]


def load_image(path: Path, device: torch.device) -> Tensor:
    img = Image.open(path).convert("RGB")
    t = TF.to_image(img)
    t = TF.to_dtype(t, torch.float32, scale=True)
    return t.to(device)


def run_baseline(
    detectors: dict[str, AnyDetector],
    paths: list[Path],
    score_thr: float,
    device: torch.device,
) -> dict[str, dict[str, list[Detection]]]:
    """全图 × 全模型的基线检测，跨目标共享。"""
    base: dict[str, dict[str, list[Detection]]] = {}
    for path in paths:
        img = load_image(path, device)
        base[str(path)] = {n: a.detect(img, score_thr) for n, a in detectors.items()}
    return base


def det_dict(d: Detection) -> dict:
    return {
        "model": d.model,
        "label": d.label,
        "score": round(d.score, 4),
        "box": [round(float(v), 1) for v in d.box],
    }


def save_viz(img_path: Path, rec, res, out_file: Path) -> None:
    """单实例标注图（抽查用）。批量证据走聚合图表，不靠检测图片。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 8 * img.height / max(img.width, 1)))
    ax.imshow(img)
    if res is not None and res.heatmap is not None:
        hm = torch.nn.functional.interpolate(
            res.heatmap[None, None].float(), size=(img.height, img.width), mode="bilinear"
        )[0, 0]
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-12)
        ax.imshow(hm.numpy(), cmap="jet", alpha=(hm.numpy() * 0.65))

    def draw(box, color: str, lw: float, ls: str, label: str | None = None) -> None:
        x0, y0, x1, y1 = (float(v) for v in box)
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=lw, ls=ls)
        )
        if label:
            ax.text(x0, y0 - 3, label, color=color, fontsize=8, weight="bold")

    for d in rec.cluster or []:
        if rec.det is not None and d is rec.det:
            continue
        draw(d.box, "deepskyblue", 1.2, "-", f"{d.model}:{d.label} {d.score:.2f}")
    draw(rec.rep_box, "lime", 1.5, "--", f"consensus:{rec.cons_label}")
    if rec.det is not None:
        draw(rec.det.box, "red", 2.0, "-", f"target:{rec.det.label} {rec.det.score:.2f}")
    if res is not None and res.peak_xy is not None:
        ax.plot([res.peak_xy[0]], [res.peak_xy[1]], "x", color="white", markersize=10, mew=2)
    level = res.level if res is not None else "-"
    inside = res.inside if res is not None else "-"
    ax.set_title(f"{rec.kind}  cons={rec.cons_label}  level={level}  inside={inside}", fontsize=9)
    ax.axis("off")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", dpi=110)
    plt.close(fig)


def run_attribution(
    detectors: dict[str, AnyDetector],
    target_name: str,
    paths: list[Path],
    base_dets: dict[str, dict[str, list[Detection]]],
    p: DetParams,
    device: torch.device,
    viz_dir: Path | None,
) -> dict:
    """单目标的归因主循环：逐图带图前向、差分判定、逐实例归因。"""
    adapter = detectors[target_name]
    gf = GraphForward(adapter)
    others = [n for n in detectors if n != target_name]
    instances: list[dict] = []
    n_deep_miss = 0
    viz_count: dict[str, int] = defaultdict(int)

    for path in paths:
        img = load_image(path, device)
        orig_size = (int(img.shape[-2]), int(img.shape[-1]))
        t_size = adapter.transform_hw(img)
        pad = adapter.letterbox_pad(img)
        g = gf.run(img, p.score_thr)
        dets_by_model: dict[str, list[Detection]] = {target_name: g.dets}
        for n in others:
            dets_by_model[n] = base_dets[str(path)][n]

        records, _ = judge_image(dets_by_model, p.iou_thr, p.loc_thr)
        gidx = g.graph_index()
        n_agree = 0
        for rec in records[target_name]:
            cand_score: float | None = None
            cand_match: bool | None = None
            deep = False
            idx = -1
            if rec.kind == "miss":
                cand = find_miss_candidate(g, rec.rep_box, rec.cons_label, p.iou_thr, p.score_thr)
                if cand is None:
                    n_deep_miss += 1
                    deep = True
                else:
                    idx, cand_match = cand
                    cand_score = float(g.scores_g[idx].detach())
            else:
                assert rec.det is not None
                idx = gidx[id(rec.det)]
                if rec.kind == "agree":
                    n_agree += 1
                    if n_agree > p.agree_per_image:
                        continue

            want_viz = viz_dir is not None and viz_count[rec.kind] < p.viz_per_kind
            res = None
            if not deep:
                res = attribute(
                    rec,
                    idx,
                    g,
                    adapter.bucket_of,
                    t_size,
                    orig_size,
                    device,
                    keep_heatmap=want_viz,
                    pad=pad,
                )

            viz_rel: str | None = None
            if want_viz:
                assert viz_dir is not None
                kind_tag = "miss_deep" if deep else rec.kind
                viz_rel = f"samples/viz/{kind_tag}/{path.stem}_{len(instances):04d}.jpg"
                save_viz(path, rec, res, viz_dir / viz_rel)
                viz_count[rec.kind] += 1

            instances.append(
                {
                    "image": path.name,
                    "kind": rec.kind,
                    "cons_label": rec.cons_label,
                    "deep_miss": deep if rec.kind == "miss" else None,
                    "det": det_dict(rec.det) if rec.det is not None else None,
                    "rep_box": [round(float(v), 1) for v in rec.rep_box],
                    "cluster": [det_dict(d) for d in rec.cluster or []],
                    "level": res.level if res is not None else None,
                    "inside": res.inside if res is not None else None,
                    "peak_xy": (
                        [round(v, 1) for v in res.peak_xy]
                        if res is not None and res.peak_xy is not None
                        else None
                    ),
                    "shares": (
                        {b: round(v, 5) for b, v in res.shares.items()} if res is not None else {}
                    ),
                    "layer_shares": (
                        {n: round(v, 5) for n, v in res.layer_shares.items()}
                        if res is not None and res.layer_shares is not None
                        else {}
                    ),
                    "cand_score": cand_score,
                    "cand_label_match": cand_match,
                    "viz": viz_rel,
                }
            )
        del g
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"instances": instances, "deep_miss": n_deep_miss}


_DRILL_TOP_K = 10  # 层级下钻报表每类失效取的层数
_DRILL_MIN_N = 10  # 参与下钻的失效类最小实例数，不足时整类跳过


def _layer_drilldown(
    layer_acc: dict[str, dict[str, list[float]]], counts: dict[str, int]
) -> dict[str, list[dict]]:
    """桶级签名的层级下钻：每类失效给出相对 agree 比值最高的 top-k 卷积层。

    定位的是目标模型内部的具体层，不做跨模型对齐。份额均值对该类全部实例求，
    某层未出场的实例按零参与计入（谱缺陷定位的标准统计方式，出场条件均值会
    系统性高估稀少层）；出场次数进 n 列供读者自行判断。实例数不足 _DRILL_MIN_N
    的失效类整体跳过；agree 侧均值为零的层没有可比基线，不进表。
    """
    n_agree = counts.get("agree", 0)
    if n_agree <= 0:
        return {}
    agree_mean = {n: sum(v) / n_agree for n, v in layer_acc.get("agree", {}).items()}
    out: dict[str, list[dict]] = {}
    for k, layers in layer_acc.items():
        n_kind = counts.get(k, 0)
        if k == "agree" or n_kind < _DRILL_MIN_N:
            continue
        rows = []
        for n, vals in layers.items():
            base = agree_mean.get(n, 0.0)
            if not vals or base <= 0:
                continue
            mean = sum(vals) / n_kind
            rows.append(
                {
                    "layer": n,
                    "n": len(vals),
                    "mean_share": round(mean, 5),
                    "ratio": round(mean / base, 3),
                }
            )
        rows.sort(key=lambda r: r["ratio"], reverse=True)
        if rows:
            out[k] = rows[:_DRILL_TOP_K]
    return out


def aggregate(result: dict) -> dict:
    """逐实例记录 -> 份额均值、相对 agree 比值、责任尺度分布、落框率、计数、层级下钻。"""
    share_acc: dict[str, dict[str, list[float]]] = {k: defaultdict(list) for k in RECORD_KINDS}
    layer_acc: dict[str, dict[str, list[float]]] = {k: defaultdict(list) for k in RECORD_KINDS}
    level_acc: dict[str, dict[str, int]] = {k: defaultdict(int) for k in RECORD_KINDS}
    inside_acc: dict[str, list[bool]] = {k: [] for k in RECORD_KINDS}
    counts: dict[str, int] = dict.fromkeys(RECORD_KINDS, 0)
    for inst in result["instances"]:
        k = inst["kind"]
        counts[k] += 1
        for b, v in inst["shares"].items():
            share_acc[k][b].append(v)
        for n, v in inst.get("layer_shares", {}).items():
            layer_acc[k][n].append(v)
        if inst["level"] is not None:
            level_acc[k][inst["level"]] += 1
        if inst["inside"] is not None:
            inside_acc[k].append(inst["inside"])

    mean_shares = {k: {b: sum(v) / len(v) for b, v in d.items() if v} for k, d in share_acc.items()}
    agree = mean_shares.get("agree", {})
    ratio_vs_agree = {
        k: {b: v / agree[b] for b, v in d.items() if agree.get(b)}
        for k, d in mean_shares.items()
        if k != "agree"
    }
    inside_rate = {k: sum(v) / len(v) for k, v in inside_acc.items() if v}
    return {
        "counts": counts,
        "deep_miss": result["deep_miss"],
        "mean_shares": mean_shares,
        "ratio_vs_agree": ratio_vs_agree,
        "levels": {k: dict(d) for k, d in level_acc.items() if d},
        "inside_rate": inside_rate,
        "layer_drilldown": _layer_drilldown(layer_acc, counts),
    }


def run_ablation(
    detectors: dict[str, AnyDetector],
    target_name: str,
    paths: list[Path],
    base_dets: dict[str, dict[str, list[Detection]]],
    p: DetParams,
    device: torch.device,
) -> list[tuple[str, dict[str, int]]]:
    """FPN 层级消融：置零某层级后重判差分失效，与基线比较。"""
    return ablate_levels(
        detectors[target_name],
        paths,
        base_dets,
        lambda pa: load_image(pa, device),
        p.iou_thr,
        p.loc_thr,
        p.score_thr,
    )


def gen_layer_drilldown(
    nat_instances: list[dict], gen_instances: list[dict]
) -> dict[str, list[dict]]:
    """生成失效的层级下钻。

    生成侧没有 agree 实例（oracle 只记失效），比值基线沿用自然侧 agree 的逐层
    均值——与自然侧下钻同一把尺子，两张表逐层可比（E6 的层级版）。统计方式
    与 _layer_drilldown 相同：全实例无条件均值，未出场记零。
    """
    layer_acc: dict[str, dict[str, list[float]]] = {k: defaultdict(list) for k in RECORD_KINDS}
    counts: dict[str, int] = dict.fromkeys(RECORD_KINDS, 0)
    for inst in nat_instances:
        if inst["kind"] != "agree":
            continue
        counts["agree"] += 1
        for n, v in (inst.get("layer_shares") or {}).items():
            layer_acc["agree"][n].append(v)
    for inst in gen_instances:
        k = inst["kind"]
        counts[k] += 1
        for n, v in (inst.get("layer_shares") or {}).items():
            layer_acc[k][n].append(v)
    return _layer_drilldown(layer_acc, counts)


def unique_failures(
    failures: list[FailureRecord], iou_thr: float
) -> tuple[dict[str, int], list[FailureRecord]]:
    """评测端的缺陷身份聚类：同种子图、同失效类型、框 IoU 达标的触发记录算同一缺陷。

    对应传统 fuzzing 的 unique crashes：循环与调度只看过程信号（覆盖、触发数、
    缺口），独特缺陷数是引导质量的产出指标，只在事后评测计算，不反馈进机制层。
    漏检用共识锚框定身份，其余用观测框。返回每类独特缺陷数与每个缺陷的代表
    记录（首次触发的那条，供真值核验）。
    """
    buckets: dict[tuple[str, str], list[Tensor]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    reps: list[FailureRecord] = []
    for fr in failures:
        box = (
            fr.observed_box
            if fr.observed_box is not None
            else (fr.anchor.box if fr.anchor is not None else None)
        )
        if box is None:
            continue
        seen = buckets[str(fr.extra.get("seed_image", "")), fr.kind]
        if any(float(box_iou(box[None], b[None])[0, 0]) >= iou_thr for b in seen):
            continue
        seen.append(box)
        counts[fr.kind] += 1
        reps.append(fr)
    return dict(counts), reps


def attribute_generated(
    detectors: dict[str, AnyDetector],
    target_name: str,
    failures: list[FailureRecord],
    p: DetParams,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """对反馈循环生成的失效做结构归因（自然 vs 生成对比的输入）。

    变异图（failures[i].image_ref 指向的 PNG）重新带图前向，按记录定位归因
    目标：miss 走低分候选，其余按同名 + IoU 找回检测框。PNG 量化可能让个别
    记录找不回，跳过并计数。每类最多取 gen_attr_per_kind 个。
    """
    adapter = detectors[target_name]
    gf = GraphForward(adapter)
    per_kind: dict[str, int] = defaultdict(int)
    instances: list[dict] = []
    n_lost = 0

    for fr in failures:
        if fr.image_ref is None or per_kind[fr.kind] >= p.gen_attr_per_kind:
            continue
        img = load_image(out_dir / fr.image_ref, device)
        orig_size = (int(img.shape[-2]), int(img.shape[-1]))
        t_size = adapter.transform_hw(img)
        pad = adapter.letterbox_pad(img)
        g = gf.run(img, p.score_thr)
        cons_label = fr.anchor.label if fr.anchor is not None else (fr.observed_label or "")
        rep_box = fr.anchor.box if fr.anchor is not None else fr.observed_box
        assert rep_box is not None

        idx = -1
        det: Detection | None = None
        if fr.kind == "miss":
            cand = find_miss_candidate(g, rep_box, cons_label, p.iou_thr, p.score_thr)
            if cand is not None:
                idx = cand[0]
        else:
            assert fr.observed_box is not None
            best_iou = 0.0
            for d in g.dets:
                if d.label != fr.observed_label:
                    continue
                iou = float(box_iou(d.box[None], fr.observed_box[None])[0, 0])
                if iou > best_iou:
                    best_iou, idx, det = iou, g.graph_index()[id(d)], d
            if best_iou < p.iou_thr:
                idx = -1
        if idx < 0:
            n_lost += 1
            del g
            continue

        rec = DetRecord(fr.kind, det, rep_box, cons_label)
        res = attribute(rec, idx, g, adapter.bucket_of, t_size, orig_size, device, pad=pad)
        instances.append(
            {
                "png": fr.image_ref,
                "kind": fr.kind,
                "cons_label": cons_label,
                "round": fr.round_idx,
                "level": res.level,
                "inside": res.inside,
                "shares": {b: round(v, 5) for b, v in res.shares.items()},
                "layer_shares": (
                    {n: round(v, 5) for n, v in res.layer_shares.items()}
                    if res.layer_shares
                    else None
                ),
            }
        )
        per_kind[fr.kind] += 1
        del g
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"instances": instances, "n_lost": n_lost}
