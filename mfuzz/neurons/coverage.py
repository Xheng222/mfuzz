"""关键神经元覆盖跟踪。

按研究方案 2.2：覆盖只有一个全局概念——存在某个测试样本使关键神经元归一化
激活超过阈值 t，即认为该神经元被覆盖。CNCov 是已覆盖关键神经元占 D_en 的比例。
CCCov[c] 是同一本全局覆盖账在类关键集合 D_en^c 上的比例。类别之间的差异来自
各类 D_en^c 是不同的神经元集合（profiler 用各类自己的数据算出），而不是分类别
记账。CCCov 揭示哪些类别的关键结构被探索得少，指导种子调度向其倾斜。

覆盖向量沿全局神经元轴 (N,) 维护，与 profiling 同序。激活先按 profiling 的
per-neuron scale 归一化再和 t 比较，与 profiler 的频率口径一致。
"""

from __future__ import annotations

import torch
from torch import Tensor

from mfuzz.neurons.profiler import NeuronProfile, flatten_acts


class CoverageTracker:
    """维护一个 (N,) 覆盖布尔向量，按需读出 CNCov 与 CCCov。"""

    def __init__(self, profile: NeuronProfile, device: torch.device | str) -> None:
        self.profile = profile
        self.device = torch.device(device)
        self.scale = profile.scale.to(self.device)  # (N,)
        self.t = profile.t
        self.layers = profile.layers
        self.critical = profile.critical.to(self.device)  # (N,) bool
        self.critical_per_class = {
            c: m.to(self.device) for c, m in profile.critical_per_class.items()
        }
        self.covered = torch.zeros(profile.num_neurons, dtype=torch.bool, device=self.device)

        # 供覆盖目标使用：全局关键神经元的扁平下标、尺度、逐类归属。
        self.den_idx = torch.nonzero(self.critical, as_tuple=False).squeeze(1)  # (K,)
        self.scale_den = self.scale[self.den_idx]  # (K,)
        self.den_in_class: dict[int, Tensor] = {
            c: m[self.den_idx] for c, m in self.critical_per_class.items()
        }

    def update_flat(self, acts_flat: Tensor) -> None:
        """用一批原始扁平激活 (B, N) 更新全局覆盖。"""
        fired = (acts_flat.detach() / self.scale) > self.t  # (B, N) bool
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
