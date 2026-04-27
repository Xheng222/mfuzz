from __future__ import annotations

import torch
from torch import Tensor

from mfuzz.neurons.profiler import CriticalNeuronSet, NeuronProfile


class CNCovTracker:
    def __init__(
        self,
        critical_set: CriticalNeuronSet,
        activation_threshold: float = 0.1,
    ):
        self.critical_set = critical_set
        self.activation_threshold = activation_threshold
        self._covered: set[str] = set()

    def update(self, activations: dict[str, Tensor]) -> list[str]:
        newly_covered = []
        for neuron in self.critical_set.neurons:
            nid = self.critical_set.neuron_id(neuron.layer_name, neuron.neuron_idx)
            if nid in self._covered:
                continue

            if neuron.layer_name not in activations:
                continue

            act = activations[neuron.layer_name]
            act_min = act.min(dim=1, keepdim=True).values
            act_max = act.max(dim=1, keepdim=True).values
            denom = (act_max - act_min).clamp(min=1e-8)
            scaled = (act - act_min) / denom

            if (scaled[:, neuron.neuron_idx] > self.activation_threshold).any():
                self._covered.add(nid)
                newly_covered.append(nid)

        return newly_covered

    @property
    def cncov(self) -> float:
        if len(self.critical_set) == 0:
            return 0.0
        return len(self._covered) / len(self.critical_set)

    @property
    def covered_count(self) -> int:
        return len(self._covered)

    @property
    def uncovered(self) -> list[NeuronProfile]:
        result = []
        for neuron in self.critical_set.neurons:
            nid = self.critical_set.neuron_id(neuron.layer_name, neuron.neuron_idx)
            if nid not in self._covered:
                result.append(neuron)
        return result

    def reset(self) -> None:
        self._covered.clear()
