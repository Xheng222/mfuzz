from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


_LAYER_TYPES = (nn.Conv2d, nn.Linear)


def discover_layers(
    model: nn.Module,
    layer_types: tuple[type, ...] = _LAYER_TYPES,
) -> dict[str, nn.Module]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, layer_types)
    }


class ActivationExtractor:
    def __init__(
        self,
        model: nn.Module,
        layer_names: list[str] | None = None,
    ):
        self.model = model
        if layer_names is None:
            self._layers = discover_layers(model)
        else:
            all_modules = dict(model.named_modules())
            self._layers = {}
            for name in layer_names:
                if name not in all_modules:
                    raise ValueError(f"Layer {name!r} not found in model")
                self._layers[name] = all_modules[name]

        self._activations: dict[str, Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, name: str, detach: bool = True):
        def hook(_module: nn.Module, _input: tuple, output: Tensor) -> None:
            act = output.detach() if detach else output
            if act.ndim == 4:
                act = act.mean(dim=(2, 3))
            self._activations[name] = act
        return hook

    def attach(self, detach: bool = True) -> None:
        self.remove_hooks()
        for name, module in self._layers.items():
            h = module.register_forward_hook(self._make_hook(name, detach=detach))
            self._hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def extract(self, x: Tensor) -> dict[str, Tensor]:
        self._activations.clear()
        self.attach(detach=True)
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            self.remove_hooks()
        return dict(self._activations)

    def extract_with_grad(self, x: Tensor) -> dict[str, Tensor]:
        self._activations.clear()
        self.attach(detach=False)
        try:
            self.model(x)
        finally:
            self.remove_hooks()
        return dict(self._activations)

    @property
    def layer_names(self) -> list[str]:
        return list(self._layers)

    def neuron_counts(self, sample_input: Tensor) -> dict[str, int]:
        acts = self.extract(sample_input)
        return {name: act.shape[1] for name, act in acts.items()}
