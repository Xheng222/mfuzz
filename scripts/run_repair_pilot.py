"""定位引导修复的去风险试点。

目的不是调出更强的模型，而是验证结构归因指对了地方：在归因指认的责任层里，
按强度抑制责任通道，看目标失效类是不是真的随之减少，并和抑制随机通道的对照比。

流程：
  1. 在 A 集上对目标失效类做逐通道归因，挑出责任层里的责任通道（top-k）。
  2. 在与 A 不相交的 B 集上，按几档强度分别抑制责任通道、随机通道、最低责任
     通道，各自重跑差分判定，统计目标失效类（收益）与 agree 正确检测（代价）。
  3. 落 JSON 与代价收益曲线。责任通道这条若压住随机对照，定位的因果性就立住。

入口：uv run python scripts/run_repair_pilot.py --target fcos --kind loc \
        --layer head.regression_head.conv.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from loguru import logger

from mfuzz.core.det_models import load_detectors
from mfuzz.differential.det_oracle import judge_image
from mfuzz.neurons.repair import channel_scale_hook, judge_counts, layer_channel_attr
from mfuzz.neurons.struct_attr import GraphForward, find_miss_candidate
from mfuzz.tasks.det_analysis import gather_images, load_image, run_baseline


def select_channels(
    target_adapter,
    others: list[str],
    a_paths: list[Path],
    base_a: dict,
    layer: str,
    kind: str,
    iou_thr: float,
    loc_thr: float,
    score_thr: float,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """在 A 集上累加目标失效类的逐通道归因，返回 (按归因降序的通道序, 实例数)。"""
    gf = GraphForward(target_adapter)
    acc: torch.Tensor | None = None
    n_inst = 0
    for path in a_paths:
        img = load_image(path, device)
        g = gf.run(img, score_thr)
        dbm = {target_adapter.name: g.dets}
        for n in others:
            dbm[n] = base_a[str(path)][n]
        records, _ = judge_image(dbm, iou_thr, loc_thr)
        gidx = g.graph_index()
        for rec in records[target_adapter.name]:
            if rec.kind != kind:
                continue
            if kind == "miss":
                cand = find_miss_candidate(g, rec.rep_box, rec.cons_label, iou_thr, score_thr)
                if cand is None:
                    continue
                idx = cand[0]
            else:
                assert rec.det is not None
                idx = gidx[id(rec.det)]
            v = layer_channel_attr(g, rec, idx, layer, device)
            if v is None:
                continue
            acc = v if acc is None else acc + v
            n_inst += 1
    if acc is None:
        raise SystemExit(f"A 集上没有可归因的 {kind} 失效，换 kind 或加大 --num-a")
    ranked = torch.argsort(acc, descending=True)
    return ranked, n_inst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="datasets/coco/val2017")
    ap.add_argument("--models", default="faster_rcnn,retinanet,fcos")
    ap.add_argument("--target", default="fcos")
    ap.add_argument("--kind", default="loc")
    ap.add_argument("--layer", default="head.regression_head.conv.0")
    ap.add_argument("--num-a", type=int, default=300)
    ap.add_argument("--num-b", type=int, default=300)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--strengths", default="1.0,0.5,0.25,0.0")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--loc-thr", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/det/repair_pilot")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.models.split(",")
    strengths = [float(s) for s in args.strengths.split(",")]

    logger.info(f"加载模型 {names}，目标 {args.target}，责任层 {args.layer}")
    detectors = load_detectors(names, device)
    target_adapter = detectors[args.target]
    others = [n for n in names if n != args.target]
    layer_mod = target_adapter.conv_layers()[args.layer]
    n_ch = int(layer_mod.out_channels)

    a_paths = gather_images(args.image_dir, args.num_a, offset=0)
    b_paths = gather_images(args.image_dir, args.num_b, offset=args.num_a)
    logger.info(f"A 集 {len(a_paths)} 图（选通道），B 集 {len(b_paths)} 图（评测），互不相交")

    base_a = run_baseline(detectors, a_paths, args.score_thr, device)
    ranked, n_inst = select_channels(
        target_adapter,
        others,
        a_paths,
        base_a,
        args.layer,
        args.kind,
        args.iou_thr,
        args.loc_thr,
        args.score_thr,
        device,
    )
    logger.info(f"{args.kind} 失效 {n_inst} 例，{args.layer} 共 {n_ch} 通道，取 top-{args.topk}")

    gen = torch.Generator().manual_seed(args.seed)
    rand_ch = torch.randperm(n_ch, generator=gen)[: args.topk]
    channel_sets: dict[str, list[int]] = {
        "responsible": ranked[: args.topk].tolist(),
        "random": rand_ch.tolist(),
        "bottom": ranked[-args.topk :].tolist(),
    }

    base_b = run_baseline(detectors, b_paths, args.score_thr, device)
    base_counts = judge_counts(
        target_adapter,
        b_paths,
        base_b,
        load_image,
        device,
        args.iou_thr,
        args.loc_thr,
        args.score_thr,
    )
    logger.info(f"B 集基线计数 {base_counts}")

    sweep: dict[str, dict[str, dict[str, int]]] = {}
    for cfg, chans in channel_sets.items():
        sweep[cfg] = {}
        for a in strengths:
            handle = channel_scale_hook(target_adapter, args.layer, chans, a)
            try:
                counts = judge_counts(
                    target_adapter,
                    b_paths,
                    base_b,
                    load_image,
                    device,
                    args.iou_thr,
                    args.loc_thr,
                    args.score_thr,
                )
            finally:
                handle.remove()
            sweep[cfg][f"{a:g}"] = counts
            k_n = counts.get(args.kind, 0)
            logger.info(f"[{cfg}] alpha={a:g}: {args.kind}={k_n} agree={counts.get('agree', 0)}")

    result = {
        "args": vars(args),
        "n_channels": n_ch,
        "kind_instances_A": n_inst,
        "responsible_channels": channel_sets["responsible"],
        "baseline_B": base_counts,
        "sweep": sweep,
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    _plot(out, args.kind, base_counts, sweep, strengths)
    logger.info(f"产物写入 {out}")


def _plot(out: Path, kind: str, base: dict, sweep: dict, strengths: list[float]) -> None:
    """两张线图：目标失效类随强度（收益）、agree 随强度（代价）；外加代价收益前沿。"""
    colors = {"responsible": "C0", "random": "C1", "bottom": "C2"}
    xs = sorted(strengths, reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for cfg, series in sweep.items():
        ax_kind = [series[f"{a:g}"].get(kind, 0) for a in xs]
        ag = [series[f"{a:g}"].get("agree", 0) for a in xs]
        axes[0].plot(xs, ax_kind, "o-", color=colors.get(cfg), label=cfg)
        axes[1].plot(xs, ag, "o-", color=colors.get(cfg), label=cfg)
        # 前沿：横轴 agree 损失，纵轴目标失效下降
        loss = [base.get("agree", 0) - v for v in ag]
        red = [base.get(kind, 0) - series[f"{a:g}"].get(kind, 0) for a in xs]
        axes[2].plot(loss, red, "o-", color=colors.get(cfg), label=cfg)

    axes[0].axhline(base.get(kind, 0), ls="--", c="gray", lw=1, label="baseline")
    axes[0].set(xlabel="kept fraction alpha", ylabel=f"{kind} failures", title=f"benefit: {kind}")
    axes[0].invert_xaxis()
    axes[1].axhline(base.get("agree", 0), ls="--", c="gray", lw=1, label="baseline")
    axes[1].set(xlabel="kept fraction alpha", ylabel="agree count", title="cost: correct dets")
    axes[1].invert_xaxis()
    axes[2].set(xlabel="agree lost (cost)", ylabel=f"{kind} reduced", title="cost-benefit frontier")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "repair_curve.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
