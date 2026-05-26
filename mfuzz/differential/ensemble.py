"""多模型管理与预测。

三个模型面向同一任务、共享输入格式与标签空间。目标与参考的划分是可切换
参数，支持实验阶段轮换目标模型。预测以 softmax 概率为准（差分目标用
F_k(x)[c] 这一类别概率）。
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import Tensor


class Ensemble:
    """持有若干模型，按名字索引，维护当前目标模型。"""

    def __init__(self, models: Mapping[str, nn.Module], target: str) -> None:
        if target not in models:
            raise ValueError(f"目标模型 {target!r} 不在集合 {list(models)} 中")
        self.models = models
        self.names = list(models)
        self._target = target

    @property
    def target(self) -> str:
        return self._target

    @property
    def references(self) -> list[str]:
        return [n for n in self.names if n != self._target]

    def set_target(self, name: str) -> None:
        """轮换目标模型。"""
        if name not in self.models:
            raise ValueError(f"目标模型 {name!r} 不在集合中")
        self._target = name

    def probs(self, x: Tensor, with_grad: bool = False) -> dict[str, Tensor]:
        """各模型对 x 的 softmax 概率，形状 (B, num_classes)。

        with_grad=True 时保留计算图，供差分目标对输入求梯度；模型权重已冻结，
        反向只会落到输入上。
        """
        ctx = nullcontext() if with_grad else torch.no_grad()
        out: dict[str, Tensor] = {}
        with ctx:
            for name, model in self.models.items():
                out[name] = torch.softmax(model(x), dim=1)
        return out

    def labels(self, x: Tensor) -> dict[str, Tensor]:
        """各模型对 x 的预测标签（argmax），形状 (B,)，不保留计算图。"""
        return {name: p.argmax(dim=1) for name, p in self.probs(x, with_grad=False).items()}
