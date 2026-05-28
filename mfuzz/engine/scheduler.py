"""种子调度：多维优先级（实现方案 4.4）。

每轮从种子池里按优先级选一批种子变异。优先级综合输入有效性、覆盖增量、路径新颖性、
历史选择次数、CCCov 类别缺口：

- 输入有效性是前提。最近一次变异的 S_input 低于 γ_input 的种子直接压到最低，不给高优先级。
- 满足有效性后，覆盖增量和路径新颖性权重高于缺陷触发历史（缺陷触发有随机性）。
- CCCov 显示某类覆盖不足（缺口大）时向该类种子倾斜。
- 多次变异（fuzz_count 高）扣分，抑制对同一种子反复变异；覆盖停滞时这项惩罚加倍。

打分高的先选。同分按池中原序，保证可复现。
"""

from __future__ import annotations

from mfuzz.core.types import SchedulerConfig, Seed


def seed_priority(
    seed: Seed,
    cccov_gap: float,
    weights: SchedulerConfig,
    gamma_input: float,
    coverage_stalled: bool,
) -> float:
    """单个种子的调度优先级。cccov_gap 是该种子共识类的覆盖缺口（1−CCCov[c]，越大越缺）。"""
    if seed.last_s_input < gamma_input:
        return float("-inf")  # 输入有效性门：语义已失效的种子不给高优先级
    fuzz_penalty = weights.w_fuzz_penalty * seed.fuzz_count
    if coverage_stalled:
        fuzz_penalty *= 2.0  # 覆盖停滞时更狠地压低重复种子
    return (
        weights.w_coverage * seed.recent_gain
        + weights.w_novelty * (1.0 if seed.path_novel else 0.0)
        + weights.w_defect * seed.defect_count
        + weights.w_cccov_gap * cccov_gap
        - fuzz_penalty
    )


def select_seeds(
    seeds: list[Seed],
    k: int,
    cccov: dict[int, float],
    weights: SchedulerConfig,
    gamma_input: float,
    coverage_stalled: bool,
) -> list[Seed]:
    """按优先级选前 k 个种子。cccov 缺某类时缺口记 0（中性）。"""
    if k >= len(seeds):
        return list(seeds)
    scored: list[tuple[float, int]] = []
    for i, s in enumerate(seeds):
        gap = 1.0 - cccov.get(s.consensus_label, 1.0)
        score = seed_priority(s, gap, weights, gamma_input, coverage_stalled)
        scored.append((score, i))
    scored.sort(key=lambda t: (-t[0], t[1]))  # 高分优先，同分按原序
    return [seeds[i] for _, i in scored[:k]]
