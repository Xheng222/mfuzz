"""Phase 2 关键神经元单元测试。

用一个小模型快速验证 cl 计算、覆盖跟踪、U 选取与覆盖目标梯度，不依赖
ImageNet 与 GPU。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mfuzz.core.hooks import ActivationExtractor
from mfuzz.neurons.coverage import CoverageTracker
from mfuzz.neurons.objective import coverage_objective, select_u
from mfuzz.neurons.profiler import build_profile, flatten_acts, normalize_acts


class Tiny(nn.Module):
    """conv(3->8) + fc(8->4)，池化后共 12 个神经元。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 4)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x)).mean(dim=(2, 3))
        return self.fc(h)


def _fake_loaders(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    lbl = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    profile_loader = [
        (torch.rand(8, 3, 8, 8, generator=g), lbl),
        (torch.rand(8, 3, 8, 8, generator=g), lbl),
    ]
    class_loaders = {
        0: [(torch.rand(6, 3, 8, 8, generator=g), torch.zeros(6, dtype=torch.long))],
        1: [(torch.rand(6, 3, 8, 8, generator=g), torch.ones(6, dtype=torch.long))],
    }
    return profile_loader, class_loaders


def _build(tmp_path, tau: float, alpha: float = 0.5, tau_class: float = 0.5):
    torch.manual_seed(0)
    model = Tiny()
    profile_loader, class_loaders = _fake_loaders()
    return model, build_profile(
        model,
        "tiny",
        profile_loader,
        class_loaders,
        t=0.5,
        tau=tau,
        tau_class=tau_class,
        alpha=alpha,
        dataset_name="fake",
        val_fraction=0.2,
        cache_dir=str(tmp_path),
        device=torch.device("cpu"),
    )


def test_profile_shapes_and_ratio(tmp_path) -> None:
    _, profile = _build(tmp_path, tau=0.3)
    assert profile.num_neurons == 12
    assert profile.low.shape == (12,) and profile.high.shape == (12,)
    assert torch.all(profile.high >= profile.low)
    assert profile.freq.min() >= 0.0 and profile.freq.max() <= 1.0
    assert 0.0 < profile.critical_ratio <= 1.0
    assert set(profile.critical_per_class) == {0, 1}
    assert set(profile.cl_per_class) == {0, 1}
    assert all(v.shape == (12,) for v in profile.cl_per_class.values())


def test_class_threshold_shrinks_sets(tmp_path) -> None:
    # τ_class 越严，D_en^c 越小：同一 cl_c 下分位阈值对集合大小单调。
    _, profile = _build(tmp_path, tau=0.1, tau_class=0.8)
    for c in (0, 1):
        strict = int(profile.class_critical_mask_at(c, 0.8).sum())
        loose = int(profile.class_critical_mask_at(c, 0.3).sum())
        assert strict <= loose
        assert int(profile.critical_per_class[c].sum()) == strict  # 实际用 τ_class=0.8


def test_alpha1_global_cl_is_freq(tmp_path) -> None:
    _, profile = _build(tmp_path, tau=0.2, alpha=1.0)
    # α=1 纯频率：全局 cl 等于全局频率；类关键集合由各类自己的频率定，键齐全。
    assert torch.allclose(profile.cl, profile.freq)
    assert set(profile.critical_per_class) == {0, 1}


def test_cache_hit(tmp_path) -> None:
    _build(tmp_path, tau=0.3)
    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 1  # 第二次同参构建命中缓存，不新增文件
    _build(tmp_path, tau=0.3)
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_coverage_grows(tmp_path) -> None:
    model, profile = _build(tmp_path, tau=0.3)
    tracker = CoverageTracker(profile, "cpu", t_cov=0.5)
    extractor = ActivationExtractor(model)
    before = tracker.cncov
    with torch.no_grad():
        tracker.update(extractor.extract(torch.rand(8, 3, 8, 8)))
    assert tracker.cncov >= before
    cccov = tracker.cccov()
    assert set(cccov) == {0, 1}
    assert all(0.0 <= v <= 1.0 for v in cccov.values())


def test_select_u_respects_size_and_uncovered(tmp_path) -> None:
    model, profile = _build(tmp_path, tau=0.1)  # 低阈值保证有关键神经元
    tracker = CoverageTracker(profile, "cpu", t_cov=0.5)
    k = tracker.den_idx.numel()
    extractor = ActivationExtractor(model)
    x = torch.rand(2, 3, 8, 8)
    acts = extractor.extract(x)
    crit_flat = flatten_acts(acts, profile.layers)[:, tracker.den_idx]
    crit = normalize_acts(crit_flat, tracker.low_den, tracker.high_den)
    mask = select_u(tracker, crit, [0, 1], u_size=3)
    assert mask.shape == (2, k)
    assert mask.sum(dim=1).max() <= 3
    # 选中的神经元必须未覆盖
    assert torch.all(~tracker.covered_den()[mask.any(dim=0)])


def test_coverage_objective_gradient_nonzero(tmp_path) -> None:
    model, profile = _build(tmp_path, tau=0.1)
    tracker = CoverageTracker(profile, "cpu", t_cov=0.5)
    extractor = ActivationExtractor(model)
    x = torch.rand(2, 3, 8, 8).requires_grad_(True)
    acts = extractor.extract_with_grad(x)
    crit_flat = flatten_acts(acts, profile.layers)[:, tracker.den_idx]
    crit = normalize_acts(crit_flat, tracker.low_den, tracker.high_den)
    with torch.no_grad():
        crit_det = crit.detach()
    mask = select_u(tracker, crit_det, [0, 1], u_size=3)
    objcov = coverage_objective(crit, mask)
    (g,) = torch.autograd.grad(objcov, x)
    assert float(g.norm()) > 0.0
