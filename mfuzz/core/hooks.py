"""中间层激活提取。

自动发现 Conv2d 与 Linear 层。Conv2d 输出做全局平均池化压成每通道一个
标量，与 Linear 输出统一为 (B, C)。提供保留与不保留计算图两个接口：
extract 用于覆盖统计等只读场景，extract_with_grad 保留计算图供归因与
梯度优化使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

LAYER_TYPES: tuple[type, ...] = (nn.Conv2d, nn.Linear)


def discover_layers(
    model: nn.Module, layer_types: tuple[type, ...] = LAYER_TYPES
) -> dict[str, nn.Module]:
    return {
        name: module for name, module in model.named_modules() if isinstance(module, layer_types)
    }


def _pool(act: Tensor) -> Tensor:
    """Conv2d 输出 (B, C, H, W) -> (B, C) 全局平均池化；其余原样返回。"""
    if act.ndim == 4:
        return act.mean(dim=(2, 3))
    return act


class ActivationExtractor:
    """对给定模型的目标层挂前向钩子，提取每神经元激活。"""

    def __init__(self, model: nn.Module, layer_names: list[str] | None = None) -> None:
        self.model = model
        if layer_names is None:
            self._layers = discover_layers(model)
        else:
            all_modules = dict(model.named_modules())
            self._layers = {}
            for name in layer_names:
                if name not in all_modules:
                    raise ValueError(f"层 {name!r} 不在模型中")
                self._layers[name] = all_modules[name]
        self._acts: dict[str, Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str, detach: bool):
        def hook(_module: nn.Module, _inp: tuple, output: Tensor) -> None:
            act = _pool(output)
            if detach:
                self._acts[name] = act.detach()
            else:
                self._acts[name] = act
                if act.requires_grad:
                    act.retain_grad()  # 保留非叶子节点梯度，供归因读取 .grad

        return hook

    def _attach(self, detach: bool) -> None:
        self._remove()
        for name, module in self._layers.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name, detach)))

    def _remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def extract(self, x: Tensor) -> dict[str, Tensor]:
        """不保留计算图，返回各层 (B, C) 激活的副本。"""
        self._acts = {}
        self._attach(detach=True)
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            self._remove()
        return dict(self._acts)

    def extract_with_grad(self, x: Tensor) -> dict[str, Tensor]:
        """保留计算图。返回的张量是计算图节点，可用于 autograd。"""
        return self.forward_with_acts(x)[1]

    def forward_with_acts(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """一次前向同时拿到模型输出与各层激活，二者共享计算图。

        覆盖目标既要目标模型的输出概率（差分项），又要中间层激活（覆盖项），
        分两次前向浪费算力，这里一次前向同时返回。
        """
        self._acts = {}
        self._attach(detach=False)
        try:
            out = self.model(x)
        finally:
            self._remove()
        return out, dict(self._acts)

    @property
    def layer_names(self) -> list[str]:
        return list(self._layers)

    def neuron_counts(self, sample: Tensor) -> dict[str, int]:
        """各层神经元数量，由一次样本前向得到的 (B, C) 推出 C。"""
        acts = self.extract(sample)
        return {name: int(act.shape[1]) for name, act in acts.items()}
