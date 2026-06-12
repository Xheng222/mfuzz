"""分类缺陷的序列化与可视化件，由分类适配器在 analyze / plot_extras 里取用。

持久化缺陷张量到 defects.pt（离线重画时加载即可拿回缺陷张量与激活向量）、写聚类
JSON、画覆盖双线 / 缺陷分布 / 缺陷流向 / 原始变异对比 / 聚类散点。统一的 result.json
与 metrics.md 由 evaluate/run_report.py 出，本模块不再负责。

图上的文字一律用英文/ASCII，避免不同机器缺中文字体时出现方块。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mfuzz.core.types import FuzzReport  # noqa: E402
from mfuzz.evaluate.metrics import cccov_round_stats  # noqa: E402
from mfuzz.neurons.cluster import ClusterResult  # noqa: E402


def save_defects(report: FuzzReport, out_dir: Path) -> Path | None:
    """持久化缺陷张量供离线重画与聚类。无缺陷则不写。"""
    if not report.defects:
        return None
    payload = [
        {
            "image": d.image,
            "source_image": d.source_image,
            "critical_activation": d.critical_activation,
            "source_label": d.source_label,
            "target_label": d.target_label,
            "target_model": d.target_model,
            "s_input": d.s_input,
            "s_path": d.s_path,
            "perturbation": d.perturbation,
        }
        for d in report.defects
    ]
    path = out_dir / "defects.pt"
    torch.save(payload, path)
    return path


def plot_coverage_curves(report: FuzzReport, out_dir: Path) -> Path | None:
    """覆盖随轮折线：全局 CNCov 一条线，类关键 CCCov 取跨类均值再叠一条线、加 min-max 带。

    CCCov 逐类的全貌仍看热力图，但热力图不直观。这里把 CCCov 跨类均值和 CNCov 画在同一张
    图、同一个 [0,1] 纵轴上，一眼比出全局覆盖和类关键覆盖的高低与涨势；阴影带给出该轮类间
    最低到最高的范围，越宽说明类别越不均衡。纯差分模式没有覆盖历史，转而画目标置信度随 PGD
    步数下降的曲线。
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if report.cncov_history:
        rounds = list(range(len(report.cncov_history)))
        ax.plot(
            rounds,
            report.cncov_history,
            "-o",
            ms=3,
            lw=1.8,
            color="#1f77b4",
            label="CNCov (global)",
        )
        if report.cccov_history:
            means, los, his = cccov_round_stats(report.cccov_history)
            cr = list(range(len(means)))
            ax.fill_between(
                cr, los, his, color="#ff7f0e", alpha=0.15, label="CCCov range (min-max)"
            )
            ax.plot(cr, means, "-s", ms=3, lw=1.6, color="#ff7f0e", label="CCCov (class mean)")
        ax.set_xlabel("round")
        ax.set_ylabel("coverage")
        ax.set_title("Critical-neuron coverage: global CNCov vs per-class CCCov")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        name = "coverage_curves.png"
    else:
        conf = report.curves.get("target_conf", [])
        if not conf:
            plt.close(fig)
            return None
        ax.plot(range(len(conf)), conf, "-o", ms=3, lw=1.8, color="#d62728")
        ax.set_xlabel("PGD step")
        ax.set_ylabel("mean target confidence on c")
        ax.set_title("Target confidence drop (pure differential)")
        ax.grid(alpha=0.3)
        name = "confidence_curve.png"
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_defect_distributions(report: FuzzReport, out_dir: Path) -> Path | None:
    """缺陷的两个连续分布：扰动 L∞ 直方、关键神经元指纹 ĉ 直方（标 [0,1] 与 OOD）。

    分布就用直方图。来源/目标类别是关系不是分布，另见缺陷流向热力图。
    """
    defs = report.defects
    if not defs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    perts = [d.perturbation for d in defs]
    hi = max(perts) if perts else 0.0
    axes[0].hist(
        perts, bins=20, range=(0.0, hi * 1.2 if hi > 0 else 1.0), color="#1f77b4", alpha=0.85
    )
    axes[0].set_title("Perturbation Linf")
    axes[0].set_xlabel("Linf")
    axes[0].set_ylabel("#defects")
    axes[0].grid(alpha=0.3)

    vecs = [
        d.critical_activation.detach().cpu().flatten().float()
        for d in defs
        if d.critical_activation is not None
    ]
    ax = axes[1]
    if vecs:
        allv = torch.cat(vecs).numpy()
        ax.hist(allv, bins=60, color="#9467bd", alpha=0.85)
        ax.axvline(0.0, color="k", ls="--", lw=1.0)
        ax.axvline(1.0, color="k", ls="--", lw=1.0)
        n_ood = int(((allv > 1.0) | (allv < 0.0)).sum())
        ax.set_title(f"Critical activation c_hat ({n_ood}/{allv.size} OOD)")
        ax.set_xlabel("normalized activation c_hat")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no fingerprint", ha="center", va="center", color="gray")
        ax.set_xticks([])
        ax.set_yticks([])

    path = out_dir / "defect_distributions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_defect_flow(report: FuzzReport, out_dir: Path) -> Path | None:
    """缺陷流向热力图：x=目标类别，y=来源类别，色=该 (源,目标) 对的缺陷数。

    源到目标是关系，热力图看出缺陷往哪些类别流、是集中还是散开。空格子（无该对）留白。
    """
    defs = report.defects
    if not defs:
        return None
    srcs = sorted({d.source_label for d in defs})
    tgts = sorted({d.target_label for d in defs})
    si = {s: i for i, s in enumerate(srcs)}
    ti = {t: i for i, t in enumerate(tgts)}
    mat = np.zeros((len(srcs), len(tgts)))
    for d in defs:
        mat[si[d.source_label], ti[d.target_label]] += 1.0
    n_pairs = int((mat > 0).sum())

    fig, ax = plt.subplots(
        figsize=(min(14.0, max(6.0, 0.16 * len(tgts))), min(11.0, max(4.0, 0.16 * len(srcs))))
    )
    cmap = plt.get_cmap("magma_r").with_extremes(bad="white")
    im = ax.imshow(
        np.ma.masked_equal(mat, 0.0),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="#defects")
    ax.set_xlabel("target class")
    ax.set_ylabel("source class")
    ax.set_title(
        f"Defect flow source->target: {len(srcs)} src x {len(tgts)} tgt, "
        f"{len(defs)} defects in {n_pairs} pairs"
    )
    xs = max(1, len(tgts) // 18)
    xt = list(range(0, len(tgts), xs))
    ax.set_xticks(xt)
    ax.set_xticklabels([str(tgts[i]) for i in xt], rotation=60, fontsize=6)
    ys = max(1, len(srcs) // 18)
    yt = list(range(0, len(srcs), ys))
    ax.set_yticks(yt)
    ax.set_yticklabels([str(srcs[i]) for i in yt], fontsize=6)
    path = out_dir / "defect_flow.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _chw_to_hwc(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def plot_defect_gallery(report: FuzzReport, out_dir: Path, n: int = 8) -> Path | None:
    """缺陷原始/变异对比图：取前 n 个缺陷，上排原始、下排变异，标注 src->tgt。"""
    with_img = [d for d in report.defects if d.source_image is not None]
    if not with_img:
        return None
    sel = with_img[:n]
    cols = len(sel)
    fig, axes = plt.subplots(2, cols, figsize=(1.7 * cols, 3.8))
    axes = np.atleast_2d(axes)
    for j, d in enumerate(sel):
        assert d.source_image is not None
        axes[0, j].imshow(_chw_to_hwc(d.source_image))
        axes[0, j].set_title(f"src {d.source_label}", fontsize=8)
        axes[1, j].imshow(_chw_to_hwc(d.image))
        axes[1, j].set_title(f"-> {d.target_label} (L∞={d.perturbation:.3f})", fontsize=7)
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=9)
    axes[1, 0].set_ylabel("mutated", fontsize=9)
    path = out_dir / "defect_gallery.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_clusters(cluster_result: ClusterResult, out_dir: Path) -> Path | None:
    """缺陷聚类散点：2D 投影按簇着色，图例标各簇样本数。"""
    if not cluster_result.embedding or cluster_result.n_defects < 2:
        return None
    emb = np.asarray(cluster_result.embedding, dtype=float)
    labels = np.asarray(cluster_result.labels, dtype=int)
    sizes = {c.cluster_id: c.size for c in cluster_result.clusters}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cmap = plt.get_cmap("tab10")
    for cid in sorted(set(labels.tolist())):
        pts = emb[labels == cid]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=22,
            color=cmap(cid % 10),
            alpha=0.75,
            label=f"cluster {cid} (n={sizes.get(cid, len(pts))})",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"Defect clusters: {cluster_result.n_clusters} groups, "
        f"silhouette={cluster_result.silhouette:.3f}"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    path = out_dir / "defect_clusters.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_clusters_json(cluster_result: ClusterResult, out_dir: Path) -> Path:
    """把聚类概况（不含坐标的精简版）写成 clusters.json，便于核对类别对。"""
    payload = {
        "n_defects": cluster_result.n_defects,
        "n_clusters": cluster_result.n_clusters,
        "silhouette": cluster_result.silhouette,
        "method": cluster_result.method,
        "n_excluded_neurons": cluster_result.n_excluded_neurons,
        "clusters": [asdict(c) for c in cluster_result.clusters],
    }
    path = out_dir / "clusters.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
