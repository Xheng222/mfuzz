from __future__ import annotations

import torch
from torch import Tensor


class ConstraintOperator:
    def __init__(self, epsilon: float = 0.03):
        self.epsilon = epsilon

    def __call__(self, grad: Tensor, x: Tensor) -> Tensor:
        direction = grad.sign()
        step = self.epsilon * direction
        x_new = x + step
        step = x_new.clamp(0.0, 1.0) - x
        return step
