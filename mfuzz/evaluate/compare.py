"""多组实验对比（实现方案 4.5）。

加载若干 result.json，画 CNCov 增长曲线叠加图、CCCov 跨类均值叠加图、关键标量指标的
分组柱状图、五维代表指标的归一化雷达图，并生成一张 markdown 汇总表。用来横向看不同配置
（消融、静态/动态、轮换目标）在五维指标上的差异。文字一律英文，避免字体问题。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mfuzz.evaluate.metrics import cccov_round_stats, cccov_scalars  # noqa: E402

# 雷达图每个轴取一个维度的代表性指标（都越大越好），轴名 -> result.json 键。
_RADAR_AXES: list[tuple[str, str]] = [
    ("rft", "Defect (RFT)"),
    ("cncov_gain", "Coverage (gain)"),
    ("input_valid_rate", "Semantic (valid)"),
    ("oi", "Diversity (OI)"),
    ("defects_per_sec", "Efficiency (def/s)"),
]

# 汇总表里展示的标量指标：键 -> 表头。覆盖五维里的代表量。
SUMMARY_KEYS: dict[str, str] = {
    "n_defects": "#Faults",
    "rft": "RFT",
    "fre": "FRE",
    "cncov_0": "CNCov0",
    "cncov_final": "CNCovF",
    "cncov_gain": "CNCovGain",
    "cccov_mean_final": "CCCovF",
    "cccov_mean_gain": "CCCovGain",
    "n_classes": "#Classes",
    "oi": "OI",
    "n_clusters": "#Clusters",
    "mean_cov_grad_norm": "CovGrad",
    "defects_per_sec": "Def/s",
}


@dataclass
class LoadedResult:
    name: str  # 实验名，取自所在目录名（无 mode：行为由 λ/feedback 旋钮决定）
    target: str
    metrics: dict[str, float]
    cncov_history: list[float]
    cccov_history: list[dict[str, float]] = field(default_factory=list)


def _resolve(path: str | Path) -> Path:
    """既接受 result.json 路径，也接受含 result.json 的目录。"""
    p = Path(path)
    return p / "result.json" if p.is_dir() else p


def load_result(path: str | Path) -> LoadedResult:
    p = _resolve(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    metrics = dict(data.get("metrics", {}))
    cccov_history = data.get("cccov_history", [])
    # 旧 result.json 的 metrics 里可能没有 cccov 标量，从历史现算补上，保证汇总表有这两列。
    for k, v in cccov_scalars(cccov_history).items():
        metrics.setdefault(k, v)
    return LoadedResult(
        name=p.parent.name,
        target=data.get("target_model", "?"),
        metrics=metrics,
        cncov_history=data.get("cncov_history", []),
        cccov_history=cccov_history,
    )


def load_results(paths: list[str | Path]) -> list[LoadedResult]:
    return [load_result(p) for p in paths]


def comparison_table(results: list[LoadedResult]) -> str:
    """生成 markdown 汇总表，行是实验，列是代表性标量指标。缺失的指标记 —。"""
    headers = ["experiment", "target"] + list(SUMMARY_KEYS.values())
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in results:
        cells = [r.name, r.target]
        for key in SUMMARY_KEYS:
            if key not in r.metrics:
                cells.append("—")
                continue
            v = r.metrics[key]
            cells.append(
                f"{v:.0f}" if key in ("n_defects", "n_classes", "n_clusters") else f"{v:.4g}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_cncov_overlay(results: list[LoadedResult], out_path: Path) -> Path | None:
    """各实验的 CNCov 增长曲线叠加。没有覆盖历史的实验跳过。"""
    has = [r for r in results if r.cncov_history]
    if not has:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in has:
        ax.plot(range(len(r.cncov_history)), r.cncov_history, "-o", ms=3, lw=1.6, label=r.name)
    ax.set_xlabel("round")
    ax.set_ylabel("CNCov")
    ax.set_title("CNCov growth across experiments")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_cccov_overlay(results: list[LoadedResult], out_path: Path) -> Path | None:
    """各实验 CCCov 跨类均值的增长曲线叠加，与 compare_cncov 配套看类关键覆盖。

    CCCov 逐类有几十条，跨实验再逐类叠就糊成一片，所以每组取每轮跨类均值画一条线。和
    CNCov 叠加图并排看，能区分"覆盖增益到底是全局铺开还是各类一起涨"。无 CCCov 历史的实验跳过。
    """
    has = [r for r in results if r.cccov_history]
    if not has:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in has:
        means, _, _ = cccov_round_stats(r.cccov_history)
        ax.plot(range(len(means)), means, "-o", ms=3, lw=1.6, label=r.name)
    ax.set_xlabel("round")
    ax.set_ylabel("mean CCCov (over classes)")
    ax.set_title("Mean class-critical coverage across experiments")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_metric_bars(results: list[LoadedResult], out_path: Path) -> Path | None:
    """关键标量指标的分组柱状图，每个指标一格，组内按实验并列。"""
    keys = [
        k
        for k in ("cncov_gain", "n_defects", "rft", "oi", "n_clusters")
        if any(k in r.metrics for r in results)
    ]
    if not keys:
        return None
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4))
    if n == 1:
        axes = [axes]
    names = [r.name for r in results]
    x = range(len(results))
    cmap = plt.get_cmap("tab10")
    for ax, key in zip(axes, keys, strict=True):
        vals = [r.metrics.get(key, 0.0) for r in results]
        ax.bar(x, vals, color=[cmap(i % 10) for i in x])
        ax.set_title(SUMMARY_KEYS.get(key, key))
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_metric_radar(results: list[LoadedResult], out_path: Path) -> Path | None:
    """五维代表指标的雷达图，各实验叠一层。每个轴对各实验取最大值归一化，故最优=1。

    轴都取越大越好的量，归一化后看哪组实验在五维上更"鼓"。少于三个轴或全为零时不出图。
    """
    keys = [k for k, _ in _RADAR_AXES if any(k in r.metrics for r in results)]
    labels = [lab for k, lab in _RADAR_AXES if k in keys]
    if len(keys) < 3:
        return None
    maxv = {k: max((r.metrics.get(k, 0.0) for r in results), default=0.0) for k in keys}
    angles = [2 * pi * i / len(keys) for i in range(len(keys))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.8, 6.5), subplot_kw={"polar": True})
    cmap = plt.get_cmap("tab10")
    for idx, r in enumerate(results):
        vals = [(r.metrics.get(k, 0.0) / maxv[k] if maxv[k] > 0 else 0.0) for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, "-o", ms=3, lw=1.6, color=cmap(idx % 10), label=r.name)
        ax.fill(angles, vals, color=cmap(idx % 10), alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Five-dimension comparison (per-axis max-normalized)", fontsize=11, pad=24)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.1), fontsize=8)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compare(paths: list[str | Path], out_dir: str | Path) -> Path:
    """加载多组结果，写汇总表与对比图，返回汇总表路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = load_results(paths)
    table = comparison_table(results)
    table_path = out / "comparison.md"
    table_path.write_text(table + "\n", encoding="utf-8")
    plot_cncov_overlay(results, out / "compare_cncov.png")
    plot_cccov_overlay(results, out / "compare_cccov.png")
    plot_metric_bars(results, out / "compare_metrics.png")
    plot_metric_radar(results, out / "compare_radar.png")
    return table_path
