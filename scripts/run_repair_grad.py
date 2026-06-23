"""方向性修复试点：对几何型失效（定位偏移）用权重梯度编辑而非抑制。

抑制（乘小系数）是减法算子，适合"多了一个响应"的虚检，却会把定位偏移这类
几何失效弄得更糟，因为它把回归输出整体往零推、把框推歪。这里换思路：归因时
已经算了目标 IoU 对责任层权重的梯度，这个方向天然指向"怎么改权重能让框更准"。
沿 +IoU 方向给责任通道对应的权重迈一步，就是一步无数据、无训练循环的方向性修复。

流程与抑制试点平行：A 集累加 loc 失效对责任层权重的梯度 G、并按逐通道归因挑
责任通道；只动责任通道对应的输出权重，按几档步长沿 +G 编辑；在不相交的 B 集
重判，看 loc（收益）与 agree（代价）。对照有沿 -G（应更糟，验证方向有意义）和
对随机通道沿其自身 +G。

入口：uv run python scripts/run_repair_grad.py --target fcos --kind loc \
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
from mfuzz.neurons.repair import judge_counts, layer_channel_attr, layer_weight_grad
from mfuzz.neurons.struct_attr import GraphForward, find_miss_candidate
from mfuzz.tasks.det_analysis import gather_images, load_image, run_baseline


def accumulate(
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
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """A 集一遍：累加责任层权重梯度 G 与逐通道归因，返回 (G, 通道归因, 实例数)。"""
    module = target_adapter.conv_layers()[layer]
    weight = module.weight
    weight.requires_grad_(True)
    gf = GraphForward(target_adapter)
    grad_acc = torch.zeros_like(weight)
    chan_acc: torch.Tensor | None = None
    n_inst = 0
    try:
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
                gw = layer_weight_grad(g, rec, idx, weight, device)
                ca = layer_channel_attr(g, rec, idx, layer, device)
                if gw is None or ca is None:
                    continue
                grad_acc += gw.detach()
                chan_acc = ca if chan_acc is None else chan_acc + ca
                n_inst += 1
    finally:
        weight.requires_grad_(False)
    if chan_acc is None:
        raise SystemExit(f"A 集上没有可归因的 {kind} 失效，换 kind 或加大 --num-a")
    return grad_acc, chan_acc, n_inst


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
    ap.add_argument("--lrs", default="0,0.1,0.3,1.0")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--loc-thr", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/det/repair_grad")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.models.split(",")
    lrs = [float(s) for s in args.lrs.split(",")]

    logger.info(f"加载模型 {names}，目标 {args.target}，责任层 {args.layer}（权重编辑式修复）")
    detectors = load_detectors(names, device)
    target_adapter = detectors[args.target]
    others = [n for n in names if n != args.target]
    module = target_adapter.conv_layers()[args.layer]
    weight = module.weight
    n_ch = int(weight.shape[0])

    a_paths = gather_images(args.image_dir, args.num_a, offset=0)
    b_paths = gather_images(args.image_dir, args.num_b, offset=args.num_a)
    logger.info(f"A 集 {len(a_paths)} 图（选向+选通道），B 集 {len(b_paths)} 图（评测），互不相交")

    base_a = run_baseline(detectors, a_paths, args.score_thr, device)
    grad_acc, chan_acc, n_inst = accumulate(
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
    ranked = torch.argsort(chan_acc, descending=True)
    resp = ranked[: args.topk]
    gen = torch.Generator().manual_seed(args.seed)
    rand = torch.randperm(n_ch, generator=gen)[: args.topk]
    logger.info(f"{args.kind} 失效 {n_inst} 例，责任层 {n_ch} 通道，取 top-{args.topk}")

    w0 = weight.data.clone()

    def masked_step(channels: torch.Tensor) -> torch.Tensor:
        """把 G 限制在 channels 这些输出通道、归一化，按责任通道权重范数定步长尺度。"""
        gm = torch.zeros_like(grad_acc)
        gm[channels] = grad_acc[channels]
        gnorm = gm.norm()
        wscale = w0[channels].norm()
        return gm / (gnorm + 1e-12) * wscale

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

    configs = {
        "resp_pos": (resp, +1.0),
        "resp_neg": (resp, -1.0),
        "rand_pos": (rand, +1.0),
    }
    sweep: dict[str, dict[str, dict[str, int]]] = {}
    for cfg, (channels, sign) in configs.items():
        step = masked_step(channels)
        sweep[cfg] = {}
        for lr in lrs:
            weight.data = w0 + sign * lr * step
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
                weight.data = w0.clone()
            sweep[cfg][f"{lr:g}"] = counts
            k_n = counts.get(args.kind, 0)
            logger.info(f"[{cfg}] lr={lr:g}: {args.kind}={k_n} agree={counts.get('agree', 0)}")

    result = {
        "args": vars(args),
        "n_channels": n_ch,
        "kind_instances_A": n_inst,
        "baseline_B": base_counts,
        "sweep": sweep,
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    _plot(out, args.kind, base_counts, sweep, lrs)
    logger.info(f"产物写入 {out}")


def _plot(out: Path, kind: str, base: dict, sweep: dict, lrs: list[float]) -> None:
    """两张线图：目标失效随步长（收益）、agree 随步长（代价）。"""
    colors = {"resp_pos": "C0", "resp_neg": "C3", "rand_pos": "C1"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for cfg, series in sweep.items():
        yk = [series[f"{lr:g}"].get(kind, 0) for lr in lrs]
        ya = [series[f"{lr:g}"].get("agree", 0) for lr in lrs]
        axes[0].plot(lrs, yk, "o-", color=colors.get(cfg), label=cfg)
        axes[1].plot(lrs, ya, "o-", color=colors.get(cfg), label=cfg)
    axes[0].axhline(base.get(kind, 0), ls="--", c="gray", lw=1, label="baseline")
    axes[0].set(xlabel="edit step lr", ylabel=f"{kind} failures", title=f"benefit: {kind}")
    axes[1].axhline(base.get("agree", 0), ls="--", c="gray", lw=1, label="baseline")
    axes[1].set(xlabel="edit step lr", ylabel="agree count", title="cost: correct dets")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "repair_grad_curve.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
