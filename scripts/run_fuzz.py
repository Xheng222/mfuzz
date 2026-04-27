"""Entry point for differential fuzzing (Phase 1)."""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import torch
from loguru import logger

from mfuzz.core.datasets import load_dataset
from mfuzz.differential.ensemble import ModelEnsemble
from mfuzz.engine.runner import FuzzConfig, FuzzRunner
from mfuzz.engine.seed_pool import SeedPool, Seed


def main(config_path: str = "configs/default.toml") -> None:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    device = torch.device(cfg["device"]["name"])
    logger.info(f"Device: {device}")

    logger.info("Loading models...")
    ensemble = ModelEnsemble(
        model_names=cfg["models"]["names"],
        dataset=cfg["dataset"]["name"],
        device=device,
        target_idx=cfg["models"]["target_idx"],
    )
    logger.info(f"Models loaded: {cfg['models']['names']}")

    logger.info("Loading seed dataset...")
    seed_loader = load_dataset(
        name=cfg["dataset"]["name"],
        split="seed",
        data_root=cfg["dataset"]["data_root"],
        batch_size=cfg["dataset"]["batch_size"],
        seed_size=cfg["dataset"]["seed_size"],
    )

    pool = SeedPool()
    accepted = 0
    skipped = 0
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
        logger.error("No seeds with consensus — cannot proceed")
        return

    fuzz_cfg = FuzzConfig(
        max_iterations=cfg["fuzz"]["max_iterations"],
        step_size=cfg["fuzz"]["step_size"],
        pgd_steps=cfg["fuzz"]["pgd_steps"],
        batch_size=cfg["fuzz"]["batch_size"],
        lambda1=cfg["fuzz"]["lambda1"],
        epsilon=cfg["fuzz"]["epsilon"],
        log_interval=cfg["fuzz"]["log_interval"],
    )

    runner = FuzzRunner(
        config=fuzz_cfg,
        ensemble=ensemble,
        seed_pool=pool,
        device=device,
    )

    report = runner.run()

    logger.info(f"=== Results ===")
    logger.info(f"Total defects: {report.num_defects}")
    logger.info(f"Total iterations: {report.total_iterations}")
    logger.info(f"Time: {report.elapsed_time:.1f}s")
    if report.rft_history:
        avg_rft = sum(report.rft_history) / len(report.rft_history)
        logger.info(f"Avg RFT: {avg_rft:.4f}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.toml"
    main(config)
