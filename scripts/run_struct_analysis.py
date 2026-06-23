"""结构分析正式入口：四类失效的结构归因 + 层级消融，逐目标模型全量跑。

pilot（Day 4）验证过方法之后的成体系版本，逻辑全部来自 mfuzz 包：

- mfuzz.core.det_models   检测模型适配层
- mfuzz.differential.det_oracle   跨模型聚簇与四类失效判定
- mfuzz.neurons.struct_attr   归因画像、责任尺度、空间峰值、层级消融

相比 pilot 新增漏检归因：从未达分数阈值的候选里取目标，取不到的记为深漏。
每个目标模型输出一份 JSON 到 --out 目录，控制台同时打印聚合表。逐实例记录
带微观上下文：检测框与分数、consensus 簇内全部模型的检测、空间峰值坐标，
深漏也入档。每类失效另落标注图（目标框、代表框、其它模型的框、责任层级热
力图与峰值）到 <out>/<target>/samples/viz/<kind>/，供人工抽查失效判定与归因指向。
归因份额同时给出相对 agree 对照的比值，规避激活规模主导问题。

用法（服务器）：
    uv run python scripts/run_struct_analysis.py --num-images 200
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.v2 import functional as TF

from mfuzz.core.det_models import (
    TV_BUCKET_ORDER,
    TorchvisionDetector,
    load_detectors,
    shared_label_space,
)
from mfuzz.core.types import Detection
from mfuzz.differential.det_oracle import RECORD_KINDS, judge_image
from mfuzz.neurons.struct_attr import (
    GraphForward,
    ablate_levels,
    attribute,
    find_miss_candidate,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_IMAGE_ROOT = _PROJECT_ROOT / "datasets" / "coco" / "val2017"
_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.JPG", "*.PNG")


def _gather_images(root: Path, n: int) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"找不到图像目录：{root}")
    paths: list[Path] = []
    for ext in _IMAGE_EXTS:
        paths += sorted(root.glob(ext))
    return paths[:n]


def _load_image(path: Path, device: torch.device) -> Tensor:
    img = Image.open(path).convert("RGB")
    t = TF.to_image(img)
    t = TF.to_dtype(t, torch.float32, scale=True)
    return t.to(device)


# ---------- 微观记录与可视化 ----------


def _det_dict(d: Detection) -> dict:
    return {
        "model": d.model,
        "label": d.label,
        "score": round(d.score, 4),
        "box": [round(float(v), 1) for v in d.box],
    }


def _save_viz(img_path: Path, rec, res, out_file: Path) -> None:
    """单个实例的标注图：目标框、代表框、簇内其它模型的框、热力图与峰值。"""
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
        # 透明度随强度走，低值区域不遮挡原图
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


# ---------- 单目标模型的归因主循环 ----------


def run_target(
    adapters: dict[str, TorchvisionDetector],
    target_name: str,
    paths: list[Path],
    base_dets: dict[str, dict[str, list[Detection]]],
    device: torch.device,
    args: argparse.Namespace,
    viz_dir: Path | None,
) -> dict:
    adapter = adapters[target_name]
    gf = GraphForward(adapter)
    others = [n for n in adapters if n != target_name]
    instances: list[dict] = []
    n_deep_miss = 0
    viz_count: dict[str, int] = defaultdict(int)

    for path in paths:
        img = _load_image(path, device)
        orig_size = (int(img.shape[-2]), int(img.shape[-1]))
        t_size = adapter.transform_hw(img)
        g = gf.run(img, args.score_thr)
        dets_by_model: dict[str, list[Detection]] = {target_name: g.dets}
        for n in others:
            dets_by_model[n] = base_dets[str(path)][n]

        records, _ = judge_image(dets_by_model, args.iou_thr, args.loc_thr)
        gidx = g.graph_index()
        n_agree = 0
        for rec in records[target_name]:
            cand_score: float | None = None
            cand_match: bool | None = None
            deep = False
            idx = -1
            if rec.kind == "miss":
                cand = find_miss_candidate(
                    g, rec.rep_box, rec.cons_label, args.iou_thr, args.score_thr
                )
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
                    if n_agree > args.agree_per_image:
                        continue

            want_viz = viz_dir is not None and viz_count[rec.kind] < args.viz_per_kind
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
                viz_rel = f"samples/viz/{kind_tag}/{path.stem}_{len(instances):04d}.jpg"
                _save_viz(path, rec, res, viz_dir / viz_rel)
                viz_count[rec.kind] += 1

            instances.append(
                {
                    "image": path.name,
                    "kind": rec.kind,
                    "cons_label": rec.cons_label,
                    "deep_miss": deep if rec.kind == "miss" else None,
                    "det": _det_dict(rec.det) if rec.det is not None else None,
                    "rep_box": [round(float(v), 1) for v in rec.rep_box],
                    "cluster": [_det_dict(d) for d in rec.cluster or []],
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


# ---------- 聚合与报表 ----------


def aggregate(result: dict) -> dict:
    share_acc: dict[str, dict[str, list[float]]] = {k: defaultdict(list) for k in RECORD_KINDS}
    level_acc: dict[str, dict[str, int]] = {k: defaultdict(int) for k in RECORD_KINDS}
    inside_acc: dict[str, list[bool]] = {k: [] for k in RECORD_KINDS}
    for inst in result["instances"]:
        k = inst["kind"]
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
    counts = {k: max((len(v) for v in share_acc[k].values()), default=0) for k in RECORD_KINDS}
    inside_rate = {k: sum(v) / len(v) for k, v in inside_acc.items() if v}
    return {
        "counts": counts,
        "deep_miss": result["deep_miss"],
        "mean_shares": mean_shares,
        "ratio_vs_agree": ratio_vs_agree,
        "levels": {k: dict(d) for k, d in level_acc.items() if d},
        "inside_rate": inside_rate,
    }


def print_tables(target: str, agg: dict, ablation: list[tuple[str, dict[str, int]]]) -> None:
    buckets = [b for b in TV_BUCKET_ORDER if any(b in d for d in agg["mean_shares"].values())]
    print("\n" + "=" * 90)
    deep = agg["deep_miss"]
    print(f"目标模型 {target}    实例数 {agg['counts']}    深漏（无候选不可归因）{deep}")

    print("\n归因份额（|grad×act| 逐桶占比的均值）")
    print(f"{'类型':<10}{'n':>5}" + "".join(f"{b:>13}" for b in buckets))
    for k in RECORD_KINDS:
        d = agg["mean_shares"].get(k)
        if not d:
            continue
        row = "".join(f"{d[b]:>13.3f}" if b in d else f"{'-':>13}" for b in buckets)
        print(f"{k:<10}{agg['counts'][k]:>5}{row}")

    print("\n归因份额相对 agree 对照的比值")
    print(f"{'类型':<10}" + "".join(f"{b:>13}" for b in buckets))
    for k, d in agg["ratio_vs_agree"].items():
        if not d:
            continue
        row = "".join(f"{d[b]:>13.2f}" if b in d else f"{'-':>13}" for b in buckets)
        print(f"{k:<10}{row}")

    levels = sorted({lv for d in agg["levels"].values() for lv in d})
    print("\n责任尺度（按 FPN 特征梯度判定的 P 层级计数）")
    print(f"{'类型':<10}" + "".join(f"{lv:>8}" for lv in levels))
    for k in RECORD_KINDS:
        d = agg["levels"].get(k)
        if not d:
            continue
        print(f"{k:<10}" + "".join(f"{d.get(lv, 0):>8}" for lv in levels))

    print("\n梯度峰值落框率")
    for k, v in agg["inside_rate"].items():
        print(f"  {k:<10}{v:.2f}")

    if ablation:
        cols = ("miss", "spurious", "cls", "loc", "agree", "consensus")
        print("\n层级消融（置零该 FPN 层级后重判差分失效）")
        print(f"{'条件':<12}" + "".join(f"{c:>10}" for c in cols))
        for name, counts in ablation:
            print(f"{name:<12}" + "".join(f"{counts.get(c, 0):>10}" for c in cols))
    print("=" * 90)


def main() -> None:
    ap = argparse.ArgumentParser(description="结构分析：四类失效归因 + 层级消融")
    ap.add_argument("--image-dir", type=Path, default=_DEFAULT_IMAGE_ROOT)
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument(
        "--targets", nargs="+", default=["faster_rcnn", "retinanet", "fcos"], help="目标模型轮换"
    )
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--loc-thr", type=float, default=0.7)
    ap.add_argument("--agree-per-image", type=int, default=2)
    ap.add_argument(
        "--out", type=Path, default=_PROJECT_ROOT / "output" / "det" / "struct_analysis"
    )
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument(
        "--viz-per-kind", type=int, default=30, help="每类失效落多少张标注图，0 关闭可视化"
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapters = load_detectors(["faster_rcnn", "retinanet", "fcos"], device)
    for t in args.targets:
        assert t in adapters, f"目标模型须是 {list(adapters)} 之一"
    paths = _gather_images(args.image_dir, args.num_images)
    labels = shared_label_space(adapters.values())
    print(f"设备：{device}  图像：{len(paths)} 张  标准标签空间 {len(labels)} 类")
    print(f"阈值：score>={args.score_thr}  iou>={args.iou_thr}  loc<{args.loc_thr}")

    # 基线检测一次算好，归因时当参考、消融时当 baseline，跨目标共享。
    base_dets: dict[str, dict[str, list[Detection]]] = {}
    for path in paths:
        img = _load_image(path, device)
        base_dets[str(path)] = {n: a.detect(img, args.score_thr) for n, a in adapters.items()}
    print(f"基线检测完成：{len(paths)} 张 × {len(adapters)} 模型")

    args.out.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        viz_dir = (args.out / target) if args.viz_per_kind > 0 else None
        result = run_target(adapters, target, paths, base_dets, device, args, viz_dir)
        agg = aggregate(result)
        ablation: list[tuple[str, dict[str, int]]] = []
        if not args.skip_ablation:
            ablation = ablate_levels(
                adapters[target],
                paths,
                base_dets,
                lambda p: _load_image(p, device),
                args.iou_thr,
                args.loc_thr,
                args.score_thr,
            )
        print_tables(target, agg, ablation)

        out_file = args.out / f"{target}.json"
        payload = {
            "config": {
                "num_images": len(paths),
                "score_thr": args.score_thr,
                "iou_thr": args.iou_thr,
                "loc_thr": args.loc_thr,
                "agree_per_image": args.agree_per_image,
            },
            "aggregate": agg,
            "ablation": [{"condition": n, **c} for n, c in ablation],
            "instances": result["instances"],
        }
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已写出 {out_file}")


if __name__ == "__main__":
    main()
