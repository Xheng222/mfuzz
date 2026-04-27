from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mfuzz.core.models import load_model
from mfuzz.core.types import ModelPrediction


class ModelEnsemble:
    def __init__(
        self,
        model_names: list[str],
        dataset: str = "imagenet",
        device: torch.device | str = "cpu",
        target_idx: int = 0,
    ):
        self.device = torch.device(device)
        self.target_idx = target_idx
        self.model_names = model_names
        self.models: list[nn.Module] = []
        for name in model_names:
            m = load_model(name, dataset=dataset, device=self.device)
            self.models.append(m)

    @property
    def target_model(self) -> nn.Module:
        return self.models[self.target_idx]

    @property
    def reference_models(self) -> list[nn.Module]:
        return [m for i, m in enumerate(self.models) if i != self.target_idx]

    def predict_all(self, x: Tensor) -> list[ModelPrediction]:
        results = []
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
            probs = F.softmax(logits, dim=1)
            label = probs.argmax(dim=1).item()
            results.append(ModelPrediction(label=label, confidence=probs.squeeze(0)))
        return results

    def consensus_label(self, x: Tensor) -> int | None:
        preds = self.predict_all(x)
        labels = [p.label for p in preds]
        if len(set(labels)) == 1:
            return labels[0]
        return None

    def has_disagreement(self, predictions: list[ModelPrediction]) -> bool:
        target_label = predictions[self.target_idx].label
        ref_labels = [p.label for i, p in enumerate(predictions) if i != self.target_idx]
        return target_label not in ref_labels or len(set(ref_labels)) > 1

    def consensus_labels_batch(self, x: Tensor) -> list[int | None]:
        all_labels: list[list[int]] = []
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
            labels = logits.argmax(dim=1).tolist()
            all_labels.append(labels)

        batch_size = x.shape[0]
        result = []
        for i in range(batch_size):
            sample_labels = [all_labels[m][i] for m in range(len(self.models))]
            if len(set(sample_labels)) == 1:
                result.append(sample_labels[0])
            else:
                result.append(None)
        return result
