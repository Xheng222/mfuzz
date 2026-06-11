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

from mfuzz.core.det_models import TorchvisionDetector
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
    cache_dir: str = "output/profiles"  # 标定统计缓存；空串关闭缓存
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
    detectors: dict[str, TorchvisionDetector],
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
    detectors: dict[str, TorchvisionDetector],
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
                )

            viz_rel: str | None = None
            if want_viz:
                assert viz_dir is not None
                kind_tag = "miss_deep" if deep else rec.kind
                viz_rel = f"viz/{kind_tag}/{path.stem}_{len(instances):04d}.jpg"
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
                    "cand_score": cand_score,
                    "cand_label_match": cand_match,
                    "viz": viz_rel,
                }
            )
        del g
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"instances": instances, "deep_miss": n_deep_miss}


def aggregate(result: dict) -> dict:
    """逐实例记录 -> 份额均值、相对 agree 比值、责任尺度分布、落框率、计数。"""
    share_acc: dict[str, dict[str, list[float]]] = {k: defaultdict(list) for k in RECORD_KINDS}
    level_acc: dict[str, dict[str, int]] = {k: defaultdict(int) for k in RECORD_KINDS}
    inside_acc: dict[str, list[bool]] = {k: [] for k in RECORD_KINDS}
    counts: dict[str, int] = dict.fromkeys(RECORD_KINDS, 0)
    for inst in result["instances"]:
        k = inst["kind"]
        counts[k] += 1
        for b, v in inst["shares"].items():
            share_acc[k][b].append(v)
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
    }


def run_ablation(
    detectors: dict[str, TorchvisionDetector],
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


def attribute_generated(
    detectors: dict[str, TorchvisionDetector],
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
        res = attribute(rec, idx, g, adapter.bucket_of, t_size, orig_size, device)
        instances.append(
            {
                "png": fr.image_ref,
                "kind": fr.kind,
                "cons_label": cons_label,
                "round": fr.round_idx,
                "level": res.level,
                "inside": res.inside,
                "shares": {b: round(v, 5) for b, v in res.shares.items()},
            }
        )
        per_kind[fr.kind] += 1
        del g
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"instances": instances, "n_lost": n_lost}
