"""Phase 3 语义模块单元测试。

验证特征层选取、S_input / S_path 余弦、obj_sem 数值与可微，全程 CPU、不依赖模型。
"""

from __future__ import annotations

import torch

from mfuzz.semantic.feature import feature_layer, s_input
from mfuzz.semantic.objective import semantic_objective
from mfuzz.semantic.path import s_path


def test_feature_layer_picks_penultimate() -> None:
    layers = ["conv1", "layer4.2.conv3", "fc"]
    assert feature_layer(layers) == "layer4.2.conv3"


def test_feature_layer_needs_two_layers() -> None:
    import pytest

    with pytest.raises(ValueError):
        feature_layer(["only_logits"])


def test_s_input_identical_and_orthogonal() -> None:
    v0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    # 同向 -> 1
    same = s_input(v0.clone(), v0)
    assert same.shape == (2,)
    assert torch.allclose(same, torch.ones(2), atol=1e-6)
    # 正交 -> 0
    v_orth = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    orth = s_input(v_orth, v0)
    assert torch.allclose(orth, torch.zeros(2), atol=1e-6)


def test_s_path_identical_and_orthogonal() -> None:
    u0 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert torch.allclose(s_path(u0.clone(), u0), torch.ones(1), atol=1e-6)
    u_orth = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert torch.allclose(s_path(u_orth, u0), torch.zeros(1), atol=1e-6)


def test_semantic_objective_is_one_minus_cosine() -> None:
    v0 = torch.tensor([[1.0, 0.0]])
    v = torch.tensor([[1.0, 0.0]])
    obj = semantic_objective(v, v0)
    assert torch.allclose(obj, torch.zeros(1), atol=1e-6)  # 同向偏移为 0
    v_orth = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(semantic_objective(v_orth, v0), torch.ones(1), atol=1e-6)


def test_semantic_objective_differentiable() -> None:
    # obj_sem 要能对输入求梯度，供 PGD 用。
    v0 = torch.tensor([[1.0, 0.0, 0.0]])
    x = torch.tensor([[0.5, 0.5, 0.2]], requires_grad=True)
    obj = semantic_objective(x, v0).sum()
    (grad,) = torch.autograd.grad(obj, x)
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
