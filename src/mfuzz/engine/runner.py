from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from mfuzz.core.types import TestResult, FuzzReport
from mfuzz.differential.ensemble import ModelEnsemble
from mfuzz.differential.objective import DifferentialObjective
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
    epsilon: float = 0.03
    log_interval: int = 50


class FuzzRunner:
    def __init__(
        self,
        config: FuzzConfig,
        ensemble: ModelEnsemble,
        seed_pool: SeedPool,
        device: torch.device | str = "cpu",
    ):
        self.config = config
        self.ensemble = ensemble
        self.seed_pool = seed_pool
        self.device = torch.device(device)

        self.objective = DifferentialObjective(
            ensemble=ensemble,
            lambda1=config.lambda1,
        )
        self.operator = ConstraintOperator(epsilon=config.epsilon)

    def _mutate_single(self, seed: Seed) -> tuple[Tensor, int] | None:
        consensus = seed.label
        x = seed.image.unsqueeze(0).to(self.device)

        x_adv = x.clone()
        for _ in range(self.config.pgd_steps):
            grad = self.objective.gradient(x_adv, consensus)
            step = self.operator(grad, x_adv)
            x_adv = x_adv + self.config.step_size * step
            x_adv = x_adv.clamp(0.0, 1.0)

        preds = self.ensemble.predict_all(x_adv)
        target_label = preds[self.ensemble.target_idx].label
        ref_labels = [p.label for i, p in enumerate(preds) if i != self.ensemble.target_idx]

        is_defect = target_label != consensus and all(r == consensus for r in ref_labels)

        if is_defect:
            result = TestResult(
                original=seed.image.cpu(),
                mutated=x_adv.squeeze(0).cpu(),
                is_defect=True,
                target_pred=target_label,
                reference_preds=ref_labels,
            )
            return x_adv.squeeze(0), result
        return None

    def run(self) -> FuzzReport:
        report = FuzzReport()
        start = time.time()

        logger.info(
            f"Starting fuzzing: {self.config.max_iterations} iterations, "
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
                    result = self._mutate_single(seed)
                    if result is not None:
                        x_new, test_result = result
                        report.defects.append(test_result)
                        round_defects += 1
                        new_seed = Seed(
                            image=x_new.detach().cpu(),
                            label=seed.label,
                            generation=seed.generation + 1,
                        )
                        self.seed_pool.add(new_seed)

                rft = round_defects / len(seeds) if seeds else 0.0
                report.rft_history.append(rft)

                progress.update(task, advance=1, defects=report.num_defects)

                if (iteration + 1) % self.config.log_interval == 0:
                    logger.info(
                        f"[iter {iteration + 1}] "
                        f"defects={report.num_defects}, "
                        f"pool_size={len(self.seed_pool)}, "
                        f"rft={rft:.3f}"
                    )

        report.total_iterations = self.config.max_iterations
        report.elapsed_time = time.time() - start
        logger.info(
            f"Done: {report.num_defects} defects in {report.elapsed_time:.1f}s"
        )
        return report
