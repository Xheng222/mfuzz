from __future__ import annotations

import torch
from torch import Tensor

from mfuzz.core.hooks import ActivationExtractor
from mfuzz.neurons.coverage import CNCovTracker


class CoverageObjective:
    def __init__(
        self,
        extractor: ActivationExtractor,
        coverage_tracker: CNCovTracker,
    ):
        self.extractor = extractor
        self.tracker = coverage_tracker

    def __call__(self, x: Tensor) -> Tensor:
        uncovered = self.tracker.uncovered
        if not uncovered:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        target = uncovered[0]
        acts = self.extractor.extract_with_grad(x)

        if target.layer_name not in acts:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        act = acts[target.layer_name]
        return act[0, target.neuron_idx]

    def gradient(self, x: Tensor) -> Tensor:
        x_var = x.detach().clone().requires_grad_(True)
        obj = self(x_var)
        obj.backward()
        return x_var.grad.detach()
