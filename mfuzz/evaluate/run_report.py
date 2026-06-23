"""统一运行报告：result.json + metrics.md + 任务无关曲线图。

任务无关的部分只画随轮变化的量：覆盖曲线、各类新失效累计、RFT 与语义偏移、
动态权重、探针曲线。任务特有的最终值图（失效计数柱状、归因份额、消融等）由
适配器 plot_extras / plot_combined 提供。不用热力图；图上文字 ASCII。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mfuzz.core.adapter import TaskAdapter  # noqa: E402
from mfuzz.core.config import Config  # noqa: E402
from mfuzz.core.records import RunReport  # noqa: E402

KIND_COLOR = {
    "miss": "#1f77b4",
    "spurious": "#d62728",
    "cls": "#9467bd",
    "loc": "#ff7f0e",
    "agree": "#2ca02c",
}


def _result_dict(report: RunReport, cfg: Config, target: str) -> dict:
    return {
        "task": cfg.task,
        "target_model": target,
        "config": {k: v for k, v in asdict(cfg).items() if k != "raw"} | {"raw": cfg.raw},
        "metrics": report.metrics,
        "cncov_history": report.cncov_history,
        "fail_history": report.fail_history,
        "rft_history": report.rft_history,
        "sem_shift_history": report.sem_shift_history,
        "lambda_history": [list(p) for p in report.lambda_history],
        "curves": report.curves,
        "extra": {k: v for k, v in report.extra.items() if _jsonable(v)},
        "total_rounds": report.total_rounds,
        "elapsed_time": report.elapsed_time,
    }


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def plot_loop_curves(report: RunReport, out: Path, target: str) -> None:
    """主循环四联图：覆盖、各类新失效累计、RFT 与语义偏移、动态权重。"""
    if not report.rft_history:
        return
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    ax = axes[0]
    ax.plot(range(len(report.cncov_history)), report.cncov_history, "-o", ms=3, color="#1f77b4")
    ax.set_title("Critical-unit coverage")
    ax.set_xlabel("round")
    ax.set_ylabel("CNCov")
    ax.set_ylim(0, 1)

    ax = axes[1]
    for k, hist in report.fail_history.items():
        if not any(hist):
            continue
        cum = np.cumsum(hist)
        ax.plot(range(1, len(cum) + 1), cum, "-o", ms=3, label=k, color=KIND_COLOR.get(k))
    ax.set_title("Cumulative new failures")
    ax.set_xlabel("round")
    ax.set_ylabel("#failures")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(range(len(report.rft_history)), report.rft_history, "-o", ms=3, label="RFT")
    ax.plot(
        range(len(report.sem_shift_history)),
        report.sem_shift_history,
        "-s",
        ms=3,
        label="semantic shift",
    )
    ax.set_title("RFT and semantic shift")
    ax.set_xlabel("round")
    ax.legend(fontsize=8)

    ax = axes[3]
    l2 = [p[0] for p in report.lambda_history]
    l3 = [p[1] for p in report.lambda_history]
    ax.plot(range(len(l2)), l2, "-o", ms=3, label="lambda2 (cov)")
    ax.plot(range(len(l3)), l3, "-s", ms=3, label="lambda3 (sem)")
    ax.set_title("Dynamic weights")
    ax.set_xlabel("round")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{target}: feedback-driven loop over rounds", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "loop_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_probe_curves(report: RunReport, out: Path) -> None:
    """探针曲线：每条命名曲线一格折线。"""
    curves = {k: v for k, v in report.curves.items() if v}
    if not curves:
        return
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, (name, ys) in zip(axes, curves.items(), strict=True):
        ax.plot(range(len(ys)), ys, "-o", ms=3, color="#8c564b")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "probe_curves.png", dpi=130)
    plt.close(fig)


def write_metrics_md(report: RunReport, out: Path, target: str) -> None:
    lines = [f"# 运行指标：{target}", "", "| 指标 | 值 |", "|---|---|"]
    for k, v in report.metrics.items():
        s = f"{v:.0f}" if k.startswith("n_") else f"{v:.4g}"
        lines.append(f"| {k} | {s} |")
    lines.append("")
    (out / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def generate_run_report(
    report: RunReport, adapter: TaskAdapter, cfg: Config, out_dir: str | Path
) -> None:
    """单目标报告总入口：核心数据先落盘，再画通用曲线，最后交适配器画任务图。

    完整档分层：核心数据落 out/data/，通用曲线与任务图表落 out/figures/，样本图
    在 out/samples/、探针在 out/probes/（由各自的写出方负责）。out_dir 始终是该
    run/model 的目录，子目录由本函数与适配器统一拼接。
    """
    out = Path(out_dir)
    data = out / "data"
    figures = out / "figures"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (data / "result.json").write_text(
        json.dumps(_result_dict(report, cfg, adapter.target), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    write_metrics_md(report, data, adapter.target)
    plot_loop_curves(report, figures, adapter.target)
    plot_probe_curves(report, figures)
    adapter.plot_extras(report, out)
