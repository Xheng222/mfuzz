from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from loguru import logger
from rich.progress import track

from mfuzz.core.hooks import ActivationExtractor


@dataclass
class NeuronProfile:
    layer_name: str
    neuron_idx: int
    cl: float
    cl_class: dict[int, float] = field(default_factory=dict)


@dataclass
class CriticalNeuronSet:
    neurons: list[NeuronProfile]
    by_layer: dict[str, list[int]]
    by_class: dict[int, list[NeuronProfile]]
    threshold: float
    total_neurons: int

    def __len__(self) -> int:
        return len(self.neurons)

    def neuron_id(self, layer_name: str, neuron_idx: int) -> str:
        return f"{layer_name}:{neuron_idx}"


def _cache_key(
    model_name: str,
    dataset: str,
    activation_threshold: float,
    critical_threshold: float,
) -> str:
    raw = f"{model_name}|{dataset}|{activation_threshold}|{critical_threshold}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _serialize(cs: CriticalNeuronSet) -> dict:
    return {
        "neurons": [
            {"layer_name": n.layer_name, "neuron_idx": n.neuron_idx,
             "cl": n.cl, "cl_class": n.cl_class}
            for n in cs.neurons
        ],
        "by_layer": cs.by_layer,
        "by_class_keys": {
            k: [(n.layer_name, n.neuron_idx) for n in profiles]
            for k, profiles in cs.by_class.items()
        },
        "threshold": cs.threshold,
        "total_neurons": cs.total_neurons,
    }


def _deserialize(data: dict) -> CriticalNeuronSet:
    neurons = [
        NeuronProfile(**n) for n in data["neurons"]
    ]
    neuron_map = {(n.layer_name, n.neuron_idx): n for n in neurons}
    by_class: dict[int, list[NeuronProfile]] = {}
    for k, keys in data["by_class_keys"].items():
        by_class[int(k)] = [neuron_map[tuple(key)] for key in keys]
    return CriticalNeuronSet(
        neurons=neurons,
        by_layer=data["by_layer"],
        by_class=by_class,
        threshold=data["threshold"],
        total_neurons=data["total_neurons"],
    )


class NeuronProfiler:
    def __init__(
        self,
        extractor: ActivationExtractor,
        activation_threshold: float = 0.1,
        critical_threshold: float = 0.75,
        device: torch.device | str = "cpu",
    ):
        self.extractor = extractor
        self.activation_threshold = activation_threshold
        self.critical_threshold = critical_threshold
        self.device = torch.device(device)

    def profile(
        self,
        train_loader: DataLoader,
        *,
        cache_dir: Path | None = None,
        model_name: str = "",
        dataset: str = "",
    ) -> CriticalNeuronSet:
        if cache_dir is not None and model_name and dataset:
            key = _cache_key(model_name, dataset,
                             self.activation_threshold, self.critical_threshold)
            cache_path = cache_dir / f"profile_{key}.pt"
            if cache_path.exists():
                logger.info(f"Loading cached profile from {cache_path}")
                data = torch.load(cache_path, weights_only=False)
                result = _deserialize(data)
                logger.info(
                    f"Loaded: {len(result)}/{result.total_neurons} critical neurons "
                    f"({len(result)/result.total_neurons*100:.1f}%)"
                )
                return result
        else:
            cache_path = None

        result = self._run_profiling(train_loader)

        if cache_path is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(_serialize(result), cache_path)
            logger.info(f"Saved profile cache to {cache_path}")

        return result

    def _run_profiling(self, train_loader: DataLoader) -> CriticalNeuronSet:
        layer_names = self.extractor.layer_names
        activation_counts: dict[str, Tensor] = {}
        class_activation_counts: dict[str, dict[int, Tensor]] = {}
        class_sample_counts: dict[int, int] = {}
        total_samples = 0
        layer_sizes: dict[str, int] | None = None

        for images, labels in track(train_loader, description="Profiling neurons"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            acts = self.extractor.extract(images)

            if layer_sizes is None:
                layer_sizes = {name: act.shape[1] for name, act in acts.items()}
                for name, size in layer_sizes.items():
                    activation_counts[name] = torch.zeros(size, device=self.device)
                    class_activation_counts[name] = {}

            batch_size = images.shape[0]
            total_samples += batch_size

            for name, act in acts.items():
                act_min = act.min(dim=1, keepdim=True).values
                act_max = act.max(dim=1, keepdim=True).values
                denom = (act_max - act_min).clamp(min=1e-8)
                scaled = (act - act_min) / denom

                activated = (scaled > self.activation_threshold).float()
                activation_counts[name] += activated.sum(dim=0)

                unique_classes = labels.unique()
                for c in unique_classes:
                    c_val = c.item()
                    mask = labels == c
                    if c_val not in class_activation_counts[name]:
                        class_activation_counts[name][c_val] = torch.zeros(
                            layer_sizes[name], device=self.device
                        )
                    class_activation_counts[name][c_val] += activated[mask].sum(dim=0)
                    if name == layer_names[0]:
                        class_sample_counts[c_val] = class_sample_counts.get(c_val, 0) + mask.sum().item()

        neurons: list[NeuronProfile] = []
        by_layer: dict[str, list[int]] = {}
        by_class: dict[int, list[NeuronProfile]] = {}
        total_neurons = 0

        for name, counts in activation_counts.items():
            cl_values = counts / total_samples
            size = layer_sizes[name]
            total_neurons += size

            for idx in range(size):
                cl_val = cl_values[idx].item()
                if cl_val > self.critical_threshold:
                    cl_class = {}
                    for c, c_counts in class_activation_counts[name].items():
                        n_c = class_sample_counts[c]
                        if n_c > 0:
                            cl_class[c] = (c_counts[idx] / n_c).item()

                    profile = NeuronProfile(
                        layer_name=name,
                        neuron_idx=idx,
                        cl=cl_val,
                        cl_class=cl_class,
                    )
                    neurons.append(profile)
                    by_layer.setdefault(name, []).append(idx)

                    for c, cl_c in cl_class.items():
                        if cl_c > self.critical_threshold:
                            by_class.setdefault(c, []).append(profile)

        result = CriticalNeuronSet(
            neurons=neurons,
            by_layer=by_layer,
            by_class=by_class,
            threshold=self.critical_threshold,
            total_neurons=total_neurons,
        )

        logger.info(
            f"Profiling done: {len(neurons)}/{total_neurons} critical neurons "
            f"({len(neurons)/total_neurons*100:.1f}%) across {len(by_layer)} layers"
        )
        return result
