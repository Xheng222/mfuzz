"""任务无关的单元覆盖账本。

覆盖单位统一抽象成"单元"：分类的（层，通道）神经元、检测的（层，第几次调用，
通道）都是单元，区别只在 layout 怎么建、关键度怎么算——那是各任务适配器
build_tracker 的事。账本只做三件事：min-max 归一化、t_cov 判覆盖（只增不减）、
给覆盖目标提供可微的归一化激活。

UnitProfile.layout 是强制项：单元下标 -> (层名, 调用序, 通道数) 的映射，
观测探针靠它把异常单元翻译回结构位置。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mfuzz.core.records import Seed

_EPS = 1e-12

Layout = list[tuple[str, int, int]]  # (层名, 调用序, 通道数)，顺序即单元在向量里的位置


@dataclass
class UnitProfile:
    """标定结果：单元布局、归一化界、关键单元下标。freq 供频率版关键度诊断。"""

    layout: Layout
    low: Tensor  # (N,)
    high: Tensor  # (N,)
    critical_idx: Tensor  # 关键单元在全向量里的下标
    freq: Tensor | None = None

    @property
    def num_units(self) -> int:
        return int(self.low.shape[0])

    @property
    def num_critical(self) -> int:
        return int(self.critical_idx.shape[0])

    def unit_name(self, i: int) -> str:
        """单元下标 -> "层名[调用序]:通道"。探针输出用。"""
        off = 0
        for name, ci, ch in self.layout:
            if i < off + ch:
                return f"{name}[{ci}]:{i - off}"
            off += ch
        raise IndexError(f"单元下标 {i} 超出布局（共 {off}）")


def profile_from_stats(
    layout: Layout, low: Tensor, high: Tensor, freq: Tensor, critical_tau: float
) -> UnitProfile:
    """由统计量组装标定结果：频率达 critical_tau 分位、归一化界非退化的单元
    为关键。统计量可以来自一次性堆叠，也可以来自两遍流式扫描（全量标定集）。"""
    q = torch.quantile(freq, critical_tau)
    mask = (freq >= q) & (high > low)
    return UnitProfile(layout, low, high, mask.nonzero(as_tuple=True)[0], freq)


def build_frequency_profile(
    vectors: list[Tensor], layout: Layout, t_freq: float, critical_tau: float
) -> UnitProfile:
    """频率版关键度的标定（小样本一次性堆叠版）。分类侧的频率+归因融合关键度
    在其适配器里自行构造 UnitProfile，不走这里。"""
    mat = torch.stack(vectors)  # (M, N)
    low = mat.amin(dim=0)
    high = mat.amax(dim=0)
    norm = (mat - low) / (high - low + _EPS)
    freq = (norm > t_freq).float().mean(dim=0)
    return profile_from_stats(layout, low, high, freq, critical_tau)


class UnitCoverageTracker:
    """关键单元覆盖账本。子类可覆写 select_u / gap_for 引入任务特有策略。"""

    def __init__(self, profile: UnitProfile, t_cov: float, device: torch.device) -> None:
        self.profile = profile
        self.t_cov = t_cov
        self.device = device
        self.low = profile.low.to(device)
        self.high = profile.high.to(device)
        self.crit_idx = profile.critical_idx.to(device)
        self.covered = torch.zeros(profile.num_critical, dtype=torch.bool, device=device)

    def norm_critical(self, v: Tensor) -> Tensor:
        """(B, N) 全单元激活 -> (B, K) 关键单元归一化激活。可微（v 带图时）。"""
        vc = v[:, self.crit_idx]
        return (vc - self.low[self.crit_idx]) / (
            self.high[self.crit_idx] - self.low[self.crit_idx] + _EPS
        )

    def norm_all(self, v: Tensor) -> Tensor:
        return (v - self.low) / (self.high - self.low + _EPS)

    def update(self, v: Tensor) -> int:
        """记入一批激活 (B, N)，返回新覆盖的关键单元数。"""
        hit = (self.norm_critical(v.detach()) > self.t_cov).any(dim=0)
        new = hit & ~self.covered
        self.covered |= hit
        return int(new.sum())

    @property
    def cncov(self) -> float:
        n = self.covered.shape[0]
        return float(self.covered.sum()) / n if n else 0.0

    @property
    def num_uncovered(self) -> int:
        return int((~self.covered).sum())

    def select_u(self, seeds: list[Seed], v0: Tensor, u_size: int) -> Tensor | None:
        """默认 U 选择：未覆盖关键单元里当前归一化激活（批内最大）最高的 U 个。

        返回关键向量里的下标；全覆盖时 None。分类子类按类关键缺口覆写。
        """
        norm = self.norm_critical(v0.detach()).amax(dim=0)  # (K,)
        cand = (~self.covered).nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            return None
        k = min(u_size, cand.numel())
        order = norm[cand].argsort(descending=True)
        return cand[order[:k]]

    def objective(self, v: Tensor, u_idx: Tensor) -> Tensor:
        """覆盖目标：U 个目标单元归一化激活之和（可微）。v 形状 (B, N)。"""
        return self.norm_critical(v)[:, u_idx].sum()

    def gap_for(self, seed: Seed) -> float:
        """调度的覆盖缺口项。默认 0（中性）；分类子类按共识类的 CCCov 缺口覆写。"""
        return 0.0
