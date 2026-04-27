"""Smoke test: verify the full pipeline works end-to-end with synthetic data."""
from __future__ import annotations

import torch
from loguru import logger

from mfuzz.core.hooks import ActivationExtractor
from mfuzz.differential.ensemble import ModelEnsemble
from mfuzz.differential.objective import DifferentialObjective
from mfuzz.optimize.operator import ConstraintOperator
from mfuzz.engine.seed_pool import SeedPool, Seed
from mfuzz.engine.runner import FuzzConfig, FuzzRunner


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("1. Loading ImageNet pretrained models...")
    ensemble = ModelEnsemble(
        model_names=["resnet50", "vgg16_bn", "mobilenet_v2"],
        dataset="imagenet",
        device=device,
        target_idx=0,
    )
    logger.info("   Models loaded")

    logger.info("2. Testing ActivationExtractor...")
    extractor = ActivationExtractor(ensemble.target_model)
    x_dummy = torch.randn(1, 3, 224, 224, device=device)
    acts = extractor.extract(x_dummy)
    logger.info(f"   Extracted {len(acts)} layers, total neurons = {sum(a.shape[1] for a in acts.values())}")
    for name, act in list(acts.items())[:3]:
        logger.info(f"   {name}: {act.shape}")

    logger.info("3. Testing consensus detection...")
    consensus_labels = ensemble.consensus_labels_batch(x_dummy)
    logger.info(f"   Consensus: {consensus_labels}")

    logger.info("4. Testing DifferentialObjective + gradient...")
    objective = DifferentialObjective(ensemble=ensemble, lambda1=1.0)
    if consensus_labels[0] is not None:
        grad = objective.gradient(x_dummy, consensus_labels[0])
        logger.info(f"   Gradient shape: {grad.shape}, norm: {grad.norm():.4f}")
    else:
        logger.warning("   No consensus on random input — trying label 0")
        grad = objective.gradient(x_dummy, 0)
        logger.info(f"   Gradient shape: {grad.shape}, norm: {grad.norm():.4f}")

    logger.info("5. Testing ConstraintOperator...")
    op = ConstraintOperator(epsilon=0.03)
    step = op(grad, x_dummy)
    logger.info(f"   Step shape: {step.shape}, max abs: {step.abs().max():.4f}")

    logger.info("6. Running mini fuzzing loop (5 iterations)...")
    pool = SeedPool()
    for i in range(10):
        img = torch.randn(3, 224, 224)
        pool.initialize([img], [i % 1000])

    config = FuzzConfig(
        max_iterations=5,
        step_size=0.01,
        pgd_steps=3,
        batch_size=2,
        lambda1=1.0,
        epsilon=0.03,
        log_interval=1,
    )

    runner = FuzzRunner(config=config, ensemble=ensemble, seed_pool=pool, device=device)
    report = runner.run()

    logger.info(f"=== Smoke test passed ===")
    logger.info(f"   Defects: {report.num_defects}")
    logger.info(f"   Iterations: {report.total_iterations}")
    logger.info(f"   Time: {report.elapsed_time:.1f}s")


if __name__ == "__main__":
    main()
