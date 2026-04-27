from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from mfuzz.differential.ensemble import ModelEnsemble


class DifferentialObjective:
    def __init__(
        self,
        ensemble: ModelEnsemble,
        lambda1: float = 1.0,
    ):
        self.ensemble = ensemble
        self.lambda1 = lambda1

    def __call__(self, x: Tensor, consensus_label: int) -> Tensor:
        c = consensus_label
        ref_sum = torch.tensor(0.0, device=x.device)
        for i, model in enumerate(self.ensemble.models):
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            score = probs[0, c]
            if i == self.ensemble.target_idx:
                target_score = score
            else:
                ref_sum = ref_sum + score

        return ref_sum - self.lambda1 * target_score

    def gradient(self, x: Tensor, consensus_label: int) -> Tensor:
        x_var = x.detach().clone().requires_grad_(True)
        obj = self(x_var, consensus_label)
        obj.backward()
        return x_var.grad.detach()
