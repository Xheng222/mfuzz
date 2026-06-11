"""单元覆盖账本的 CPU 单元测试：布局、标定、覆盖更新、U 选择与可微目标。

布局与 GAP 向量函数在检测适配器（层×调用×通道是检测的单元语义），账本本身
任务无关（neurons/unit_coverage）。
"""

from __future__ import annotations

import torch

from mfuzz.neurons.unit_coverage import UnitCoverageTracker, build_frequency_profile
from mfuzz.tasks.detection import _build_layout, _gap_vector

_CPU = torch.device("cpu")


def _fake_acts(scale: float) -> dict[str, list[torch.Tensor]]:
    """两层、其中一层调用两次（共享 head 的多尺度调用），通道数 3/2。"""
    torch.manual_seed(0)
    return {
        "conv_a": [torch.rand(1, 3, 4, 4) * scale],
        "conv_b": [torch.rand(1, 2, 4, 4) * scale, torch.rand(1, 2, 2, 2) * scale],
    }


def test_layout_and_gap_vector_shape():
    acts = _fake_acts(1.0)
    layout = _build_layout(acts)
    assert [(n, ci) for n, ci, _ in layout] == [("conv_a", 0), ("conv_b", 0), ("conv_b", 1)]
    v = _gap_vector(acts, layout, _CPU)
    assert v.shape == (3 + 2 + 2,)


def test_gap_vector_missing_call_padded_zero():
    acts = _fake_acts(1.0)
    layout = _build_layout(acts)
    acts_short = {"conv_a": acts["conv_a"], "conv_b": acts["conv_b"][:1]}
    v = _gap_vector(acts_short, layout, _CPU)
    assert torch.all(v[-2:] == 0)


def test_unit_name_mapping():
    layout = _build_layout(_fake_acts(1.0))
    vectors = [_gap_vector(_fake_acts(s), layout, _CPU) for s in (0.2, 1.0)]
    profile = build_frequency_profile(vectors, layout, t_freq=0.5, critical_tau=0.0)
    assert profile.unit_name(0) == "conv_a[0]:0"
    assert profile.unit_name(3) == "conv_b[0]:0"
    assert profile.unit_name(5) == "conv_b[1]:0"


def test_profile_and_tracker_coverage():
    layout = _build_layout(_fake_acts(1.0))
    vectors = [_gap_vector(_fake_acts(s), layout, _CPU) for s in (0.2, 0.5, 1.0, 1.5)]
    profile = build_frequency_profile(vectors, layout, t_freq=0.5, critical_tau=0.5)
    assert profile.num_units == 7
    assert 0 < profile.num_critical <= 7

    tracker = UnitCoverageTracker(profile, t_cov=0.85, device=_CPU)
    assert tracker.cncov == 0.0
    # 标定最大值那张图必然把全部关键单元推到归一化 1.0 > t_cov
    new = tracker.update(_gap_vector(_fake_acts(1.5), layout, _CPU)[None])
    assert new == profile.num_critical
    assert tracker.cncov == 1.0


def test_select_u_and_objective_differentiable():
    layout = _build_layout(_fake_acts(1.0))
    vectors = [_gap_vector(_fake_acts(s), layout, _CPU) for s in (0.2, 0.5, 1.0, 1.5)]
    profile = build_frequency_profile(vectors, layout, t_freq=0.5, critical_tau=0.5)
    tracker = UnitCoverageTracker(profile, t_cov=0.85, device=_CPU)

    acts = {k: [t.clone().requires_grad_(True) for t in v] for k, v in _fake_acts(0.6).items()}
    v = _gap_vector(acts, layout, _CPU)[None]
    u = tracker.select_u([], v, u_size=2)
    assert u is not None and u.numel() == 2
    obj = tracker.objective(v, u)
    obj.backward()
    grads = [t.grad for ts in acts.values() for t in ts]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
