from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class LayerActivation:
    name: str
    output: Tensor


@dataclass
class ModelPrediction:
    label: int
    confidence: Tensor
    activations: dict[str, LayerActivation] = field(default_factory=dict)


@dataclass
class TestResult:
    original: Tensor
    mutated: Tensor
    is_defect: bool
    target_pred: int
    reference_preds: list[int]
    semantic_distance: float = 0.0
    new_coverage: list[str] = field(default_factory=list)


@dataclass
class FuzzReport:
    defects: list[TestResult] = field(default_factory=list)
    cncov_history: list[float] = field(default_factory=list)
    rft_history: list[float] = field(default_factory=list)
    sem_history: list[float] = field(default_factory=list)
    lambda_history: list[tuple[float, float]] = field(default_factory=list)
    total_iterations: int = 0
    elapsed_time: float = 0.0

    @property
    def num_defects(self) -> int:
        return len(self.defects)
