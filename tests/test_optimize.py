"""联合优化与变异算子单元测试。

验证梯度 L2 归一化数值、obj_total 梯度合并（含 λ=0 消融）、投影梯度上升算子的投影边界、
动态反馈的四指标规则与权重夹界，以及腐蚀算子的值域与强度单调性，全程 CPU、不依赖模型。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mfuzz.core.config import OptimizeConfig
from mfuzz.core.types import FeedbackConfig, FeedbackState
from mfuzz.optimize.feedback import FeedbackController
from mfuzz.optimize.joint import combine_gradients, normalize_grad
from mfuzz.optimize.mutator import CORRUPTION_OPS, build_mutator, corrupt
from mfuzz.optimize.operator import pgd_step


def test_normalize_grad_unit_norm() -> None:
    g = torch.randn(4, 3, 5, 5)
    out = normalize_grad(g)
    norms = out.flatten(1).norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_normalize_grad_zero_safe() -> None:
    # 零梯度不应除出 NaN/Inf。
    out = normalize_grad(torch.zeros(2, 3, 4, 4))
    assert torch.isfinite(out).all()
    assert out.abs().sum() == 0.0


def test_combine_gradients_numeric() -> None:
    # g1=[3,4]→归一化[0.6,0.8]；gc=[0,1]→[0,1]；gs=[1,0]→[1,0]。
    # combined = [0.6,0.8] + 0.5·[0,1] − 0.25·[1,0] = [0.35, 1.3]。
    g1 = torch.tensor([[[[3.0, 4.0]]]])
    gc = torch.tensor([[[[0.0, 1.0]]]])
    gs = torch.tensor([[[[1.0, 0.0]]]])
    out = combine_gradients(g1, gc, gs, lambda2=0.5, lambda3=0.25)
    assert torch.allclose(out, torch.tensor([[[[0.35, 1.3]]]]), atol=1e-6)


def test_combine_gradients_none_is_ablation() -> None:
    g1 = torch.randn(3, 3, 4, 4)
    # gc=gs=None：只剩归一化的 g1，等价于覆盖与语义都消融。
    out = combine_gradients(g1, None, None, lambda2=0.7, lambda3=0.7)
    assert torch.allclose(out, normalize_grad(g1), atol=1e-6)
    # 只给 gc：语义项缺席，只加覆盖项。
    gc = torch.randn(3, 3, 4, 4)
    out_cov = combine_gradients(g1, gc, None, lambda2=0.5, lambda3=0.9)
    expect = normalize_grad(g1) + 0.5 * normalize_grad(gc)
    assert torch.allclose(out_cov, expect, atol=1e-6)


def test_pgd_step_within_feasible_region() -> None:
    x0 = torch.full((2, 3, 4, 4), 0.5)
    grad = torch.randn(2, 3, 4, 4)
    eps = 0.03
    x = x0.clone()
    for _ in range(20):  # 多步累积，验证始终被投影回可行域
        x = pgd_step(x, x0, grad, step_size=0.01, epsilon=eps)
    assert (x - x0).abs().max() <= eps + 1e-6  # L∞ 约束
    assert x.min() >= 0.0 and x.max() <= 1.0  # 像素范围


def test_pgd_step_sign_direction() -> None:
    x0 = torch.full((1, 1, 2, 2), 0.5)
    up = pgd_step(x0.clone(), x0, torch.ones_like(x0), step_size=0.01, epsilon=0.03)
    assert torch.allclose(up, x0 + 0.01, atol=1e-6)  # 正梯度上移
    down = pgd_step(x0.clone(), x0, -torch.ones_like(x0), step_size=0.01, epsilon=0.03)
    assert torch.allclose(down, x0 - 0.01, atol=1e-6)  # 负梯度下移


def _state(delta_cncov: float, mean_sem_shift: float) -> FeedbackState:
    return FeedbackState(
        delta_cncov=delta_cncov, rft=0.1, mean_sem_shift=mean_sem_shift, path_novel_ratio=0.0
    )


def _fb(config: FeedbackConfig, l2: float, l3: float, sem_thr: float = 0.1) -> FeedbackController:
    return FeedbackController(
        config,
        l2,
        l3,
        sem_shift_threshold=sem_thr,
        lambda2_bounds=[0.1, 2.0],
        lambda3_bounds=[0.1, 2.0],
    )


def test_feedback_disabled_keeps_lambda() -> None:
    fb = _fb(FeedbackConfig(enabled=False), 0.5, 0.5)
    for _ in range(5):
        l2, l3 = fb.step(_state(delta_cncov=0.0, mean_sem_shift=0.9))
        assert l2 == 0.5 and l3 == 0.5
    assert fb.lambda2 == 0.5 and fb.lambda3 == 0.5


def test_feedback_coverage_stall_raises_lambda2() -> None:
    fb = _fb(FeedbackConfig(), 0.5, 0.5)
    # ΔCNCov=0 持续停滞 → λ2 升；语义偏移很小 → λ3 回落。
    for _ in range(3):
        fb.step(_state(delta_cncov=0.0, mean_sem_shift=0.0))
    assert fb.lambda2 > 0.5
    assert fb.lambda3 < 0.5
    assert fb.coverage_stalled


def test_feedback_sem_shift_raises_lambda3() -> None:
    fb = _fb(FeedbackConfig(), 0.5, 0.5)
    # 语义偏移持续高于阈 → λ3 升；覆盖一直在涨 → λ2 回落。
    for _ in range(3):
        fb.step(_state(delta_cncov=0.05, mean_sem_shift=0.5))
    assert fb.lambda3 > 0.5
    assert fb.lambda2 < 0.5
    assert not fb.coverage_stalled


def test_feedback_clamps_to_bounds() -> None:
    fb = _fb(FeedbackConfig(), 0.5, 0.5)  # 默认夹界 [0.1, 2.0]
    for _ in range(50):  # 一直停滞且语义良好 → λ2 顶到上界、λ3 跌到下界
        fb.step(_state(delta_cncov=0.0, mean_sem_shift=0.0))
    assert fb.lambda2 == 2.0
    assert fb.lambda3 == 0.1


def test_feedback_zero_lambda_stays_zero() -> None:
    # 初值为 0 的模块视作关闭：即便覆盖停滞，λ2 也保持 0，不被下界夹到 0.1。
    fb = _fb(FeedbackConfig(), 0.0, 0.5)
    for _ in range(10):
        fb.step(_state(delta_cncov=0.0, mean_sem_shift=0.0))
    assert fb.lambda2 == 0.0


def test_corrupt_ops_range_and_change() -> None:
    torch.manual_seed(0)
    img = torch.rand(3, 32, 48)
    for op in CORRUPTION_OPS:
        out = corrupt(op, img, severity=3)
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert not torch.allclose(out, img)  # 腐蚀必须真的改了图


def test_corrupt_severity_monotonic() -> None:
    # 噪声与对比度的偏离量应随强度单调增大（模糊与亮度走相同参数表，不重复验）。
    torch.manual_seed(0)
    img = torch.rand(3, 32, 32)
    for op in ("gaussian_noise", "contrast"):
        devs = [float((corrupt(op, img, s) - img).abs().mean()) for s in (1, 3, 5)]
        assert devs[0] < devs[1] < devs[2]


def test_corrupt_unknown_op_raises() -> None:
    with pytest.raises(KeyError, match="未知腐蚀算子"):
        corrupt("motion_blur", torch.rand(3, 8, 8), 3)


def test_corruption_mutator_batch() -> None:
    torch.manual_seed(0)
    m = build_mutator("corruption")
    x0 = torch.rand(2, 3, 16, 16)
    ctx = SimpleNamespace(batch=SimpleNamespace(x0=x0), opt=OptimizeConfig())
    out = m.mutate(ctx)  # type: ignore[arg-type]
    assert out.shape == x0.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not torch.allclose(out, x0)
