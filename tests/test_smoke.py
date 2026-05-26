"""Phase 0 冒烟测试：加载三模型、提取激活、构建种子集。

验证基础设施可用与各张量形状符合预期。需要本地有 mini-ImageNet 数据集
与预训练权重缓存；缺失时跳过。
"""

from __future__ import annotations

import pytest
import torch

from mfuzz.core.datasets import imagenet_denormalize, load_imagenet
from mfuzz.core.hooks import ActivationExtractor
from mfuzz.core.models import IMAGENET_NUM_CLASSES, list_models, load_ensemble
from mfuzz.core.seed import build_seed_pool


@pytest.fixture(scope="module")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_smoke(device: str) -> None:
    # 1. 加载三个模型
    models = load_ensemble(list_models(), device=device)
    assert set(models) == {"resnet50", "vgg16_bn", "mobilenet_v2"}

    # 2. 构建种子集
    dataset = load_imagenet("val")
    seeds = build_seed_pool(dataset, size=4, device=device)
    assert len(seeds) == 4
    for s in seeds:
        assert s.image.shape == (3, 224, 224)
        assert 0 <= s.true_label < IMAGENET_NUM_CLASSES
        assert s.consensus_label == -1  # 共识过滤在 Phase 1

    batch = torch.stack([s.image for s in seeds]).to(device)

    # 3. 预测形状
    with torch.no_grad():
        logits = models["resnet50"](batch)
    assert logits.shape == (4, IMAGENET_NUM_CLASSES)

    # 4. 不保留计算图的激活提取，形状 (B, C)
    extractor = ActivationExtractor(models["resnet50"])
    acts = extractor.extract(batch)
    assert len(acts) > 0
    for value in acts.values():
        assert value.ndim == 2 and value.shape[0] == 4
        assert not value.requires_grad

    # 5. 保留计算图的接口，激活在计算图上
    grad_batch = batch.clone().requires_grad_(True)
    grad_acts = extractor.extract_with_grad(grad_batch)
    assert next(iter(grad_acts.values())).requires_grad

    # 6. 反归一化回到 [0, 1] 像素空间
    pixel = imagenet_denormalize(seeds[0].image)
    assert pixel.min() >= 0.0 and pixel.max() <= 1.0


def test_neuron_counts(device: str) -> None:
    model = load_ensemble(["mobilenet_v2"], device=device)["mobilenet_v2"]
    extractor = ActivationExtractor(model)
    sample = torch.randn(1, 3, 224, 224, device=device)
    counts = extractor.neuron_counts(sample)
    assert len(counts) == len(extractor.layer_names)
    assert all(c > 0 for c in counts.values())
