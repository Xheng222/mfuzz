from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

_IMAGENET_REGISTRY: dict[str, type] = {
    "resnet50": tvm.resnet50,
    "vgg16_bn": tvm.vgg16_bn,
    "mobilenet_v2": tvm.mobilenet_v2,
}

_IMAGENET_WEIGHTS: dict[str, str] = {
    "resnet50": "IMAGENET1K_V2",
    "vgg16_bn": "IMAGENET1K_V1",
    "mobilenet_v2": "IMAGENET1K_V2",
}


def list_models() -> list[str]:
    return list(_IMAGENET_REGISTRY)


def load_model(
    name: str,
    dataset: str = "imagenet",
    device: torch.device | str = "cpu",
    weights_path: str | None = None,
) -> nn.Module:
    name = name.lower()
    device = torch.device(device)

    if dataset == "imagenet":
        if name not in _IMAGENET_REGISTRY:
            raise ValueError(f"Unknown model: {name}. Available: {list_models()}")
        factory = _IMAGENET_REGISTRY[name]
        weights = _IMAGENET_WEIGHTS[name]
        model = factory(weights=weights)
    elif dataset == "cifar10":
        if weights_path is None:
            raise ValueError("CIFAR-10 models require a weights_path")
        model = _IMAGENET_REGISTRY[name](num_classes=10)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    model.to(device)
    model.eval()
    return model
