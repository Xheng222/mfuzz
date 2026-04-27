from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from mfuzz.core.hooks import ActivationExtractor
from mfuzz.core.types import TestResult, FuzzReport
from mfuzz.differential.ensemble import ModelEnsemble
from mfuzz.differential.objective import DifferentialObjective
from mfuzz.neurons.coverage import CNCovTracker
from mfuzz.neurons.objective import CoverageObjective
from mfuzz.optimize.operator import ConstraintOperator
from mfuzz.engine.seed_pool import SeedPool, Seed


@dataclass
class FuzzConfig:
    max_iterations: int = 1000
    step_size: float = 0.01
    pgd_steps: int = 10
    batch_size: int = 16
    target_model_idx: int = 0
    lambda1: float = 1.0
    lambda2: float = 0.0
    epsilon: float = 0.03
    log_interval: int = 50


class FuzzRunner:
    def __init__(
        self,
        config: FuzzConfig,
        ensemble: ModelEnsemble,
        seed_pool: SeedPool,
        device: torch.device | str = "cpu",
        coverage_tracker: CNCovTracker | None = None,
        target_extractor: ActivationExtractor | None = None,
    ):
        self.config = config
        self.ensemble = ensemble
        self.seed_pool = seed_pool
        self.device = torch.device(device)
        self.coverage_tracker = coverage_tracker
        self.target_extractor = target_extractor

        self.diff_obj = DifferentialObjective(
            ensemble=ensemble,
            lambda1=config.lambda1,
        )
        self.operator = ConstraintOperator(epsilon=config.epsilon)

        self.cov_obj: CoverageObjective | None = None
        if coverage_tracker is not None and target_extractor is not None:
            self.cov_obj = CoverageObjective(
                extractor=target_extractor,
                coverage_tracker=coverage_tracker,
            )

    def _compute_gradient(self, x: Tensor, consensus: int) -> Tensor:
        grad_diff = self.diff_obj.gradient(x, consensus)

        if self.cov_obj is not None and self.config.lambda2 > 0:
            grad_cov = self.cov_obj.gradient(x)
            return grad_diff + self.config.lambda2 * grad_cov

        return grad_diff

    def _mutate_single(self, seed: Seed) -> tuple[Tensor, TestResult] | None:
        consensus = seed.label
        x = seed.image.unsqueeze(0).to(self.device)

        x_adv = x.clone()
        for _ in range(self.config.pgd_steps):
            grad = self._compute_gradient(x_adv, consensus)
            step = self.operator(grad, x_adv)
            x_adv = x_adv + self.config.step_size * step
            x_adv = x_adv.clamp(0.0, 1.0)

        preds = self.ensemble.predict_all(x_adv)
        target_label = preds[self.ensemble.target_idx].label
        ref_labels = [p.label for i, p in enumerate(preds) if i != self.ensemble.target_idx]

        is_defect = target_label != consensus and all(r == consensus for r in ref_labels)

        new_coverage: list[str] = []
        if self.coverage_tracker is not None and self.target_extractor is not None:
            acts = self.target_extractor.extract(x_adv)
            new_coverage = self.coverage_tracker.update(acts)

        result = TestResult(
            original=seed.image.cpu(),
            mutated=x_adv.squeeze(0).cpu(),
            is_defect=is_defect,
            target_pred=target_label,
            reference_preds=ref_labels,
            new_coverage=new_coverage,
        )
        return x_adv.squeeze(0), result

    def run(self) -> FuzzReport:
        report = FuzzReport()
        start = time.time()

        has_coverage = self.coverage_tracker is not None
        mode = "diff+cov" if has_coverage and self.config.lambda2 > 0 else "diff-only"
        logger.info(
            f"Starting fuzzing ({mode}): {self.config.max_iterations} iters, "
            f"{len(self.seed_pool)} seeds, "
            f"models={self.ensemble.model_names}"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("defects={task.fields[defects]}"),
        ) as progress:
            task = progress.add_task(
                "Fuzzing", total=self.config.max_iterations, defects=0,
            )

            for iteration in range(self.config.max_iterations):
                seeds = self.seed_pool.select(self.config.batch_size)
                round_defects = 0

                for seed in seeds:
                    pair = self._mutate_single(seed)
                    if pair is None:
                        continue
                    x_new, test_result = pair

                    if test_result.is_defect:
                        report.defects.append(test_result)
                        round_defects += 1

                    if test_result.is_defect or len(test_result.new_coverage) > 0:
                        new_seed = Seed(
                            image=x_new.detach().cpu(),
                            label=seed.label,
                            coverage_gain=len(test_result.new_coverage),
                            generation=seed.generation + 1,
                        )
                        self.seed_pool.add(new_seed)

                rft = round_defects / len(seeds) if seeds else 0.0
                report.rft_history.append(rft)
                if has_coverage:
                    report.cncov_history.append(self.coverage_tracker.cncov)

                progress.update(task, advance=1, defects=report.num_defects)

                if (iteration + 1) % self.config.log_interval == 0:
                    cov_str = ""
                    if has_coverage:
                        cov_str = f", cncov={self.coverage_tracker.cncov:.3f}"
                    logger.info(
                        f"[iter {iteration + 1}] "
                        f"defects={report.num_defects}, "
                        f"pool={len(self.seed_pool)}, "
                        f"rft={rft:.3f}{cov_str}"
                    )

        report.total_iterations = self.config.max_iterations
        report.elapsed_time = time.time() - start
        cov_final = ""
        if has_coverage:
            cov_final = f", CNCov={self.coverage_tracker.cncov:.3f}"
        logger.info(
            f"Done: {report.num_defects} defects in {report.elapsed_time:.1f}s{cov_final}"
        )
        return report
