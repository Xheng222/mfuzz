"""关键神经元覆盖跟踪。

按研究方案 2.2：覆盖只有一个全局概念——存在某个测试样本使关键神经元归一化
激活超过阈值 t，即认为该神经元被覆盖。CNCov 是已覆盖关键神经元占 D_en 的比例。
CCCov[c] 是同一本全局覆盖账在类关键集合 D_en^c 上的比例。类别之间的差异来自
各类 D_en^c 是不同的神经元集合（profiler 用各类自己的数据算出），而不是分类别
记账。CCCov 揭示哪些类别的关键结构被探索得少，指导种子调度向其倾斜。

覆盖向量沿全局神经元轴 (N,) 维护，与 profiling 同序。激活先按 profiling 的
per-neuron 区间 [low, high] 做 min-max 归一化（ĉ），再和覆盖阈值 t_cov 比较。

覆盖判定用的 t_cov 与 profiler 频率项里的 t_freq 是两个独立阈值（见实现方案第10章）。
两者标度相同，都作用在同一套 ĉ 上，但职责不同：t_freq 决定一个神经元在训练数据上算
不算"激活"，喂给关键度的频率项；t_cov 决定一个测试样本算不算"覆盖"了某关键神经元。
逐神经元 min-max 下 ĉ>0.5 太容易满足（多数样本落在区间中下段，几百个种子里总有一个
越过中点），CNCov 会在初始就饱和；覆盖要的是把神经元推到区间高位，故 t_cov 取得比
t_freq 高。这套分离借自 CriticalFuzz：它的 profiling 激活阈值 t 与运行时覆盖阈值 k
本就是两个参数，且数值不同。
"""

from __future__ import annotations

import torch
from torch import Tensor

from mfuzz.neurons.profiler import NeuronProfile, flatten_acts, normalize_acts


class CoverageTracker:
    """维护一个 (N,) 覆盖布尔向量，按需读出 CNCov 与 CCCov。"""

    def __init__(self, profile: NeuronProfile, device: torch.device | str, t_cov: float) -> None:
        self.profile = profile
        self.device = torch.device(device)
        self.low = profile.low.to(self.device)  # (N,)
        self.high = profile.high.to(self.device)  # (N,)
        # 覆盖判定阈值，与 profiling 频率阈值 profile.t（t_freq）解耦，由 config 直接传入。
        # 不挂在被缓存的 NeuronProfile 上：t_cov 不影响 profiling，改它不该让 profile 缓存失效。
        self.t_cov = t_cov
        self.layers = profile.layers
        self.critical = profile.critical.to(self.device)  # (N,) bool
        self.critical_per_class = {
            c: m.to(self.device) for c, m in profile.critical_per_class.items()
        }
        self.covered = torch.zeros(profile.num_neurons, dtype=torch.bool, device=self.device)

        # 供覆盖目标使用：全局关键神经元的扁平下标、归一化区间、逐类归属。
        self.den_idx = torch.nonzero(self.critical, as_tuple=False).squeeze(1)  # (K,)
        self.low_den = self.low[self.den_idx]  # (K,)
        self.high_den = self.high[self.den_idx]  # (K,)
        self.den_in_class: dict[int, Tensor] = {
            c: m[self.den_idx] for c, m in self.critical_per_class.items()
        }

    def update_flat(self, acts_flat: Tensor) -> None:
        """用一批原始扁平激活 (B, N) 更新全局覆盖。"""
        fired = normalize_acts(acts_flat.detach(), self.low, self.high) > self.t_cov  # (B, N) bool
        self.covered |= fired.any(dim=0)

    def update(self, acts: dict[str, Tensor]) -> None:
        self.update_flat(flatten_acts(acts, self.layers))

    @property
    def cncov(self) -> float:
        return float(self.covered[self.critical].float().mean())

    def cccov(self) -> dict[int, float]:
        """全局覆盖账分别投影到各类关键集合 D_en^c 上的比例。"""
        out: dict[int, float] = {}
        for c, mask in self.critical_per_class.items():
            n = int(mask.sum())
            out[c] = float(self.covered[mask].float().mean()) if n else 0.0
        return out

    def covered_den(self) -> Tensor:
        """全局关键神经元当前覆盖状态 (K,) bool。"""
        return self.covered[self.den_idx]

    @property
    def num_uncovered(self) -> int:
        return int((~self.covered[self.critical]).sum())
