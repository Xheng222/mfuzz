"""Entry point for mfuzz testing framework."""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import torch
from loguru import logger

from mfuzz.core.datasets import load_dataset
from mfuzz.core.hooks import ActivationExtractor
from mfuzz.differential.ensemble import ModelEnsemble
from mfuzz.neurons.profiler import NeuronProfiler
from mfuzz.neurons.coverage import CNCovTracker
from mfuzz.engine.runner import FuzzConfig, FuzzRunner
from mfuzz.engine.seed_pool import SeedPool


def main(config_path: str = "configs/default.toml") -> None:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    device = torch.device(cfg["device"]["name"])
    logger.info(f"Device: {device}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # --- Models ---
    logger.info("Loading models...")
    ensemble = ModelEnsemble(
        model_names=cfg["models"]["names"],
        dataset=cfg["dataset"]["name"],
        device=device,
        target_idx=cfg["models"]["target_idx"],
    )

    # --- Seeds ---
    logger.info("Loading seed dataset...")
    seed_loader = load_dataset(
        name=cfg["dataset"]["name"],
        split="seed",
        batch_size=cfg["dataset"]["batch_size"],
        seed_size=cfg["dataset"]["seed_size"],
    )

    pool = SeedPool()
    accepted = skipped = 0
    for images, labels in seed_loader:
        images = images.to(device)
        consensus = ensemble.consensus_labels_batch(images)
        for img, c in zip(images, consensus):
            if c is not None:
                pool.initialize([img.cpu()], [c])
                accepted += 1
            else:
                skipped += 1
    logger.info(f"Seed pool: {accepted} seeds (skipped {skipped} without consensus)")

    if len(pool) == 0:
        logger.error("No seeds with consensus — aborting")
        return

    # --- Neuron profiling (optional) ---
    neurons_cfg = cfg.get("neurons", {})
    coverage_tracker = None
    cov_extractor = None

    if neurons_cfg.get("enabled", False):
        logger.info("Profiling critical neurons...")
        extractor = ActivationExtractor(ensemble.target_model)
        train_loader = load_dataset(
            name=cfg["dataset"]["name"],
            split="train",
            batch_size=cfg["dataset"]["batch_size"],
        )
        profiler = NeuronProfiler(
            extractor=extractor,
            activation_threshold=neurons_cfg.get("activation_threshold", 0.1),
            critical_threshold=neurons_cfg.get("critical_threshold", 0.75),
            device=device,
        )
        critical_set = profiler.profile(train_loader)
        coverage_tracker = CNCovTracker(critical_set)
        cov_extractor = ActivationExtractor(ensemble.target_model)

    # --- Fuzz ---
    fuzz_cfg = cfg["fuzz"]
    config = FuzzConfig(
        max_iterations=fuzz_cfg["max_iterations"],
        step_size=fuzz_cfg["step_size"],
        pgd_steps=fuzz_cfg["pgd_steps"],
        batch_size=fuzz_cfg["batch_size"],
        lambda1=fuzz_cfg["lambda1"],
        lambda2=fuzz_cfg.get("lambda2", 0.0),
        epsilon=fuzz_cfg["epsilon"],
        log_interval=fuzz_cfg["log_interval"],
    )

    runner = FuzzRunner(
        config=config,
        ensemble=ensemble,
        seed_pool=pool,
        device=device,
        coverage_tracker=coverage_tracker,
        target_extractor=cov_extractor,
    )

    report = runner.run()

    # --- Output ---
    result = {
        "config": config_path,
        "defects": report.num_defects,
        "iterations": report.total_iterations,
        "time": round(report.elapsed_time, 1),
        "avg_rft": round(sum(report.rft_history) / len(report.rft_history), 4) if report.rft_history else 0,
        "pool_final": len(pool),
        "rft_history": [round(v, 4) for v in report.rft_history],
    }
    if coverage_tracker is not None:
        result["cncov_final"] = round(coverage_tracker.cncov, 4)
        result["covered"] = coverage_tracker.covered_count
        result["total_critical"] = len(coverage_tracker.critical_set)
        result["cncov_history"] = [round(v, 4) for v in report.cncov_history]

    out_path = output_dir / "result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"=== Results ===")
    logger.info(f"Defects: {result['defects']}")
    logger.info(f"Avg RFT: {result['avg_rft']}")
    logger.info(f"Time: {result['time']}s")
    if coverage_tracker is not None:
        logger.info(f"CNCov: {result['cncov_final']} ({result['covered']}/{result['total_critical']})")
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.toml"
    main(config)
