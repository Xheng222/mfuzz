"""模型注册与加载。

三个模型面向同一 ImageNet 1000 类任务、共享输入格式与标签空间，满足
差分测试前提。权重冻结：框架只对输入求梯度，不更新权重。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

IMAGENET_NUM_CLASSES = 1000

# 名称 -> (工厂函数, 预训练权重枚举名)
_IMAGENET_FACTORIES: dict[str, tuple[object, str]] = {
    "resnet50": (tvm.resnet50, "IMAGENET1K_V2"),
    "vgg16_bn": (tvm.vgg16_bn, "IMAGENET1K_V1"),
    "mobilenet_v2": (tvm.mobilenet_v2, "IMAGENET1K_V2"),
}


def list_models() -> list[str]:
    return list(_IMAGENET_FACTORIES)


def load_model(name: str, device: torch.device | str = "cpu") -> nn.Module:
    name = name.lower()
    if name not in _IMAGENET_FACTORIES:
        raise ValueError(f"未知模型 {name!r}，可用：{list_models()}")
    factory, weights = _IMAGENET_FACTORIES[name]
    model: nn.Module = factory(weights=weights)  # type: ignore[operator]
    model.to(torch.device(device)).eval()
    for p in model.parameters():
        p.requires_grad_(False)  # 只需输入梯度，冻结权重省显存
    return model


def load_ensemble(names: list[str], device: torch.device | str = "cpu") -> dict[str, nn.Module]:
    return {name: load_model(name, device) for name in names}
