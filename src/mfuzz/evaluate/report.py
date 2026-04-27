from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from loguru import logger

from mfuzz.core.types import FuzzReport, TestResult


def save_defect_images(
    defects: list[TestResult],
    output_dir: Path,
    max_save: int = 200,
) -> None:
    defect_dir = output_dir / "defects"
    defect_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(defects), max_save)
    for i, d in enumerate(defects[:n]):
        pair = torch.stack([d.original, d.mutated])
        path = defect_dir / f"{i:04d}_t{d.target_pred}_r{d.reference_preds[0]}.png"
        vutils.save_image(pair, str(path), nrow=2, normalize=True)

    logger.info(f"Saved {n} defect image pairs to {defect_dir}")


def save_metrics(
    report: FuzzReport,
    output_dir: Path,
    extra: dict | None = None,
) -> Path:
    result: dict = {
        "defects": report.num_defects,
        "iterations": report.total_iterations,
        "time": round(report.elapsed_time, 1),
        "rft_history": [round(v, 4) for v in report.rft_history],
    }
    if report.cncov_history:
        result["cncov_history"] = [round(v, 4) for v in report.cncov_history]
    if extra:
        result.update(extra)

    out_path = output_dir / "result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out_path


def plot_curves(
    report: FuzzReport,
    output_dir: Path,
) -> None:
    iters = list(range(1, len(report.rft_history) + 1))

    # cumulative defects
    cumulative = []
    total = 0
    for rft in report.rft_history:
        total += rft
        cumulative.append(total)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(iters, cumulative)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cumulative defect score")
    axes[0].set_title("Defect Discovery")

    if report.cncov_history:
        axes[1].plot(iters[:len(report.cncov_history)], report.cncov_history)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("CNCov")
        axes[1].set_title("Critical Neuron Coverage")
    else:
        axes[1].text(0.5, 0.5, "No coverage data", ha="center", va="center")
        axes[1].set_title("Critical Neuron Coverage")

    plt.tight_layout()
    path = output_dir / "curves.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info(f"Saved curves to {path}")
