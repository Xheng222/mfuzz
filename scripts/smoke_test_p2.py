"""Phase 2 smoke test: neuron profiling + coverage-guided fuzzing."""
from __future__ import annotations

import torch
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger

from mfuzz.core.hooks import ActivationExtractor
from mfuzz.differential.ensemble import ModelEnsemble
from mfuzz.neurons.profiler import NeuronProfiler
from mfuzz.neurons.coverage import CNCovTracker
from mfuzz.engine.seed_pool import SeedPool, Seed
from mfuzz.engine.runner import FuzzConfig, FuzzRunner


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("1. Loading models...")
    ensemble = ModelEnsemble(
        model_names=["resnet50", "vgg16_bn", "mobilenet_v2"],
        dataset="imagenet",
        device=device,
        target_idx=0,
    )

    logger.info("2. Creating synthetic training set for profiling...")
    fake_images = torch.randn(64, 3, 224, 224)
    fake_labels = torch.randint(0, 1000, (64,))
    train_ds = TensorDataset(fake_images, fake_labels)
    train_loader = DataLoader(train_ds, batch_size=16)

    logger.info("3. Profiling critical neurons...")
    extractor = ActivationExtractor(ensemble.target_model)
    profiler = NeuronProfiler(
        extractor=extractor,
        activation_threshold=0.1,
        critical_threshold=0.5,
        device=device,
    )
    critical_set = profiler.profile(train_loader)
    logger.info(f"   Critical neurons: {len(critical_set)}/{critical_set.total_neurons}")
    logger.info(f"   Layers with critical neurons: {list(critical_set.by_layer.keys())[:5]}...")

    logger.info("4. Testing CNCovTracker...")
    tracker = CNCovTracker(critical_set)
    test_input = torch.randn(1, 3, 224, 224, device=device)
    acts = extractor.extract(test_input)
    newly = tracker.update(acts)
    logger.info(f"   After 1 sample: CNCov={tracker.cncov:.3f}, newly covered={len(newly)}")

    logger.info("5. Running coverage-guided fuzzing (10 iterations)...")
    pool = SeedPool()
    for i in range(10):
        img = torch.randn(3, 224, 224)
        pool.initialize([img], [i % 1000])

    tracker.reset()

    config = FuzzConfig(
        max_iterations=10,
        step_size=0.01,
        pgd_steps=3,
        batch_size=2,
        lambda1=1.0,
        lambda2=0.5,
        epsilon=0.03,
        log_interval=5,
    )

    cov_extractor = ActivationExtractor(ensemble.target_model)

    runner = FuzzRunner(
        config=config,
        ensemble=ensemble,
        seed_pool=pool,
        device=device,
        coverage_tracker=tracker,
        target_extractor=cov_extractor,
    )
    report = runner.run()

    logger.info("=== Phase 2 smoke test passed ===")
    logger.info(f"   Defects: {report.num_defects}")
    logger.info(f"   Final CNCov: {tracker.cncov:.3f} ({tracker.covered_count}/{len(critical_set)})")
    logger.info(f"   CNCov history: {[f'{v:.3f}' for v in report.cncov_history[:5]]}...")
    logger.info(f"   Time: {report.elapsed_time:.1f}s")


if __name__ == "__main__":
    main()
