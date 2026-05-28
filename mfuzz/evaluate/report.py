"""结果序列化与可视化（实现方案 4.5）。

generate_report 是统一出口：先把五维指标算齐并并进 report.metrics，再写 result.json，
持久化缺陷张量到 defects.pt，最后写标量表与画图。run_fuzz 直接调它。离线想重画时，
加载 defects.pt 即可拿回缺陷张量与激活向量。

图只在"数字本身看不出的结构"上用。一堆单次运行的标量没有这种结构，写成 metrics.md
分维度表格，不画图。逐轮一维趋势用折线（CNCov 覆盖曲线、诊断面板里的覆盖梯度均范等）。
逐轮 × 类别的二维量用热力图（CCCov）。缺陷的源类别到目标类别是关系，用热力图（缺陷
流向）。连续分布用直方（扰动、关键神经元指纹 ĉ）。聚团用散点（缺陷聚类）。另有缺陷
原始/变异对比图。

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
from loguru import logger  # noqa: E402

from mfuzz.core.types import Config, FuzzReport  # noqa: E402
from mfuzz.evaluate.metrics import cccov_round_stats, enrich_metrics  # noqa: E402
from mfuzz.neurons.cluster import ClusterResult, cluster_defects  # noqa: E402

# metrics.md 里标量按维度分组展示的顺序。前五组对应五维评估，后三组是差分、profiling、
# 激活越界的诊断量。未列入的键统一归到「其它」，保证不漏一个标量。
_METRIC_GROUPS: list[tuple[str, list[str]]] = [
    ("缺陷", ["n_defects", "rft", "fre"]),
    (
        "覆盖",
        [
            "cncov_0",
            "cncov_final",
            "cncov_gain",
            "cccov_mean_0",
            "cccov_mean_final",
            "cccov_mean_gain",
            "n_uncovered_final",
            "mean_cov_grad_norm",
        ],
    ),
    ("语义", ["input_valid_rate", "mean_s_input"]),
    (
        "多样性",
        ["n_classes", "n_target_classes", "n_class_pairs", "oi", "n_clusters", "silhouette"],
    ),
    ("效率", ["elapsed_time", "total_iterations", "n_fuzzed", "defects_per_sec"]),
    (
        "差分与种子",
        [
            "seed_acceptance_rate",
            "n_seeds_accepted",
            "n_consensus_classes",
            "ref_consensus_hold_rate",
            "mean_target_conf_drop",
            "mean_perturbation_linf",
        ],
    ),
    ("关键神经元 profiling", ["n_neurons", "n_critical", "critical_ratio", "lambda2"]),
    ("激活越界", ["max_activation", "min_activation", "n_ood_neurons", "n_defects_ood"]),
]


def _result_dict(report: FuzzReport, target: str, config: Config) -> dict:
    """把 FuzzReport 整理成可 JSON 序列化的 dict。cccov 的 int 键转成 str。

    无 mode 字段：消融行为由 λ/feedback 旋钮决定。完整解析后的配置转储进 config，
    这样每次运行都自带全量有效参数，复现不依赖外部 TOML。
    """
    return {
        "target_model": target,
        "config": asdict(config),
        "metrics": report.metrics,
        "curves": report.curves,
        "cncov_history": report.cncov_history,
        "cccov_history": [{str(k): v for k, v in d.items()} for d in report.cccov_history],
        "rft_history": report.rft_history,
        "sem_shift_history": report.sem_shift_history,
        "lambda_history": [list(p) for p in report.lambda_history],
        "total_iterations": report.total_iterations,
        "elapsed_time": report.elapsed_time,
        # 缺陷数已在 metrics["n_defects"]，不再单列 num_defects 重复
    }


def save_result_json(report: FuzzReport, out_dir: Path, target: str, config: Config) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "result.json"
    path.write_text(
        json.dumps(_result_dict(report, target, config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


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


def plot_cccov_heatmap(report: FuzzReport, out_dir: Path) -> Path | None:
    """各类 CCCov 热力图：x=轮，y=共识类别，色=该类关键覆盖率。

    二维量用热力图，一眼看出哪些类别覆盖落后、覆盖怎么随轮填上。某轮缺某类（无该类种子）
    的格子置灰。没有覆盖历史则不出图。
    """
    hist = report.cccov_history
    classes = sorted({c for d in hist for c in d}) if hist else []
    if not classes:
        return None
    n_rounds = len(hist)
    mat = np.full((len(classes), n_rounds), np.nan)
    row = {c: i for i, c in enumerate(classes)}
    for r, d in enumerate(hist):
        for c, v in d.items():
            mat[row[c], r] = v
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.3 * n_rounds), min(12.0, max(3.0, 0.16 * len(classes))))
    )
    cmap = plt.get_cmap("viridis").with_extremes(bad="#dddddd")
    im = ax.imshow(
        np.ma.masked_invalid(mat),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="CCCov")
    ax.set_xlabel("round")
    ax.set_ylabel("consensus class")
    ax.set_title(f"Class-critical coverage: {len(classes)} classes x {n_rounds} rounds")
    step = max(1, len(classes) // 20)
    yt = list(range(0, len(classes), step))
    ax.set_yticks(yt)
    ax.set_yticklabels([str(classes[i]) for i in yt], fontsize=6)
    path = out_dir / "cccov_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_diagnostics(report: FuzzReport, out_dir: Path) -> Path | None:
    """逐轮诊断面板：覆盖梯度均范、RFT、语义偏移、λ 历史，各画一格。都没有就不出图。

    覆盖梯度均范这一格直接守住"覆盖梯度恒为零"那个旧 bug——只要存在未覆盖关键神经元，
    它就应当持续大于零。RFT、语义偏移、λ 三格依赖 Phase 3/4 的逐轮记录，缺数据时自动略过。
    """
    cov_grad = report.curves.get("cov_grad_norm", [])
    panels: list[str] = []
    if cov_grad:
        panels.append("covgrad")
    if report.rft_history:
        panels.append("rft")
    if report.sem_shift_history:
        panels.append("sem")
    if report.lambda_history:
        panels.append("lambda")
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4))
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, panels, strict=True):
        if kind == "covgrad":
            ax.plot(range(len(cov_grad)), cov_grad, "-o", ms=3, color="#8c564b")
            ax.set_title("Coverage-gradient norm over rounds")
            ax.set_xlabel("round")
            ax.set_ylabel("mean ||grad obj_cov||")
        elif kind == "rft":
            ax.plot(range(len(report.rft_history)), report.rft_history, "-o", ms=3, color="#2ca02c")
            ax.set_title("RFT over rounds")
            ax.set_xlabel("round")
            ax.set_ylabel("RFT")
        elif kind == "sem":
            ax.plot(
                range(len(report.sem_shift_history)),
                report.sem_shift_history,
                "-o",
                ms=3,
                color="#e377c2",
            )
            ax.set_title("Semantic shift over rounds")
            ax.set_xlabel("round")
            ax.set_ylabel("mean 1 - S_input")
        else:
            l2 = [p[0] for p in report.lambda_history]
            l3 = [p[1] for p in report.lambda_history]
            ax.plot(range(len(l2)), l2, "-o", ms=3, label="λ2 (cov)")
            ax.plot(range(len(l3)), l3, "-s", ms=3, label="λ3 (sem)")
            ax.set_title("Dynamic weights")
            ax.set_xlabel("round")
            ax.set_ylabel("λ")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    path = out_dir / "diagnostics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _fmt_metric(key: str, val: float) -> str:
    if key.startswith("n_") or key == "total_iterations":
        return f"{val:.0f}"
    if key == "elapsed_time":
        return f"{val:.1f}s"
    return f"{val:.4g}"


def write_metrics_table(report: FuzzReport, out_dir: Path) -> Path:
    """把所有标量指标按维度写成 metrics.md 表格。

    单次运行的标量没有可画的结构，表格才是诚实的呈现。按 _METRIC_GROUPS 分组，没列进
    分组的键收进「其它」，保证一个不漏。值的格式：计数取整，时间带 s，其余四位有效数字。
    """
    metrics = dict(report.metrics)
    metrics.setdefault("elapsed_time", report.elapsed_time)
    metrics.setdefault("total_iterations", float(report.total_iterations))

    lines = ["# 标量指标汇总", ""]
    listed: set[str] = set()
    for group, keys in _METRIC_GROUPS:
        present = [k for k in keys if k in metrics]
        if not present:
            continue
        lines += [f"## {group}", "", "| 指标 | 值 |", "| --- | --- |"]
        for k in present:
            lines.append(f"| {k} | {_fmt_metric(k, float(metrics[k]))} |")
            listed.add(k)
        lines.append("")
    leftover = sorted(k for k in metrics if k not in listed)
    if leftover:
        lines += ["## 其它", "", "| 指标 | 值 |", "| --- | --- |"]
        for k in leftover:
            lines.append(f"| {k} | {_fmt_metric(k, float(metrics[k]))} |")
        lines.append("")

    path = out_dir / "metrics.md"
    path.write_text("\n".join(lines), encoding="utf-8")
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


def generate_report(
    report: FuzzReport,
    out_dir: str | Path,
    config: Config,
    *,
    target: str,
) -> ClusterResult:
    """评估总入口：补齐五维指标、写 result.json 与 defects.pt、画全部图。

    返回聚类结果，便于调用方进一步使用。result.json 与 defects.pt 先写，确保即便
    后续绘图出问题也已落盘核心数据。产物：metrics.md 标量表、CNCov 覆盖曲线、CCCov
    热力图、逐轮诊断、缺陷分布直方、缺陷流向热力图、原始/变异对比、聚类散点。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cluster_result = cluster_defects(report.defects)
    n_consensus = int(report.metrics.get("n_consensus_classes", 0))
    extra = enrich_metrics(
        report,
        cluster_result,
        pgd_steps=config.optimize.pgd_steps,
        n_consensus_classes=n_consensus,
        gamma_input=config.semantic.gamma_input,
    )
    report.metrics.update(extra)

    n_ood = extra.get("n_ood_neurons", 0.0)
    if n_ood > 0:
        logger.info(
            f"激活越界：{n_ood:.0f} 个关键神经元 ĉ 越出 [0,1]"
            f"（ĉ∈[{extra['min_activation']:.3g}, {extra['max_activation']:.3g}]），涉及 "
            f"{extra['n_defects_ood']:.0f}/{report.num_defects} 个缺陷；"
            f"对抗把它们推出了 profiling 训练范围"
        )

    save_result_json(report, out, target, config)
    save_defects(report, out)
    save_clusters_json(cluster_result, out)

    write_metrics_table(report, out)
    plot_coverage_curves(report, out)
    plot_cccov_heatmap(report, out)
    plot_diagnostics(report, out)
    plot_defect_distributions(report, out)
    plot_defect_flow(report, out)
    plot_defect_gallery(report, out)
    plot_clusters(cluster_result, out)

    logger.info(
        f"评估完成：#Classes={extra['n_classes']:.0f}，OI={extra['oi']:.3f}，"
        f"簇数={extra['n_clusters']:.0f}（轮廓 {extra['silhouette']:.3f}），"
        f"FRE={extra['fre']:.4g}，图与产物写入 {out}"
    )
    return cluster_result
