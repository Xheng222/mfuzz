"""Phase 4 调度与种子池单元测试。

验证多维种子优先级（输入有效性门、覆盖增量、CCCov 缺口、变异次数惩罚）、按优先级选取，
以及种子池的统计回写与无产出退役，全程 CPU、不依赖模型。
"""

from __future__ import annotations

import torch

from mfuzz.core.types import SchedulerConfig, Seed
from mfuzz.engine.scheduler import seed_priority, select_seeds
from mfuzz.engine.seed_pool import SeedOutcome, SeedPool

_W = SchedulerConfig()
_GAMMA = 0.9


def _seed(cls: int, **kw) -> Seed:
    return Seed(image=torch.rand(3, 4, 4), true_label=cls, consensus_label=cls, **kw)


def test_priority_validity_gate() -> None:
    # 最近 S_input 低于 γ_input 的种子直接压到最低。
    bad = _seed(10, last_s_input=0.5)
    assert seed_priority(
        bad, cccov_gap=0.9, weights=_W, gamma_input=_GAMMA, coverage_stalled=False
    ) == float("-inf")
    ok = _seed(10, last_s_input=0.95)
    assert seed_priority(
        ok, cccov_gap=0.9, weights=_W, gamma_input=_GAMMA, coverage_stalled=False
    ) > float("-inf")


def test_priority_fuzz_penalty_and_stall() -> None:
    fresh = _seed(10, fuzz_count=0)
    fuzzed = _seed(10, fuzz_count=10)
    p_fresh = seed_priority(fresh, 0.0, _W, _GAMMA, coverage_stalled=False)
    p_fuzzed = seed_priority(fuzzed, 0.0, _W, _GAMMA, coverage_stalled=False)
    assert p_fresh > p_fuzzed  # 变异多的扣分更多
    # 覆盖停滞时惩罚加倍，差距进一步拉大。
    p_fuzzed_stall = seed_priority(fuzzed, 0.0, _W, _GAMMA, coverage_stalled=True)
    assert p_fuzzed_stall < p_fuzzed


def test_select_prefers_recent_gain() -> None:
    a = _seed(10, recent_gain=5.0)
    b = _seed(10, recent_gain=0.0)
    sel = select_seeds(
        [b, a], k=1, cccov={10: 1.0}, weights=_W, gamma_input=_GAMMA, coverage_stalled=False
    )
    assert sel == [a]


def test_select_favors_low_cccov_class() -> None:
    # 类 11 覆盖低（缺口大），其种子优先。
    s10 = _seed(10)
    s11 = _seed(11)
    sel = select_seeds(
        [s10, s11],
        k=1,
        cccov={10: 0.9, 11: 0.1},
        weights=_W,
        gamma_input=_GAMMA,
        coverage_stalled=False,
    )
    assert sel == [s11]


def test_select_returns_all_when_k_ge_len() -> None:
    seeds = [_seed(10), _seed(11)]
    sel = select_seeds(seeds, k=5, cccov={}, weights=_W, gamma_input=_GAMMA, coverage_stalled=False)
    assert sel == seeds


def test_pool_capacity_truncates() -> None:
    seeds = [_seed(i) for i in range(5)]
    pool = SeedPool(seeds, capacity=3, retire_patience=6)
    assert len(pool.seeds) == 3
    assert pool.n_active == 3


def test_pool_update_writes_stats() -> None:
    s = _seed(10)
    pool = SeedPool([s], capacity=10, retire_patience=6)
    pool.update_after_round(
        [
            SeedOutcome(
                seed=s, new_coverage=3.0, path_novel=False, s_input=0.95, produced_defect=True
            )
        ]
    )
    assert s.fuzz_count == 1
    assert s.recent_gain == 3.0
    assert s.coverage_gain == 3.0
    assert s.defect_count == 1
    assert s.last_s_input == 0.95
    pool.update_after_round(
        [
            SeedOutcome(
                seed=s, new_coverage=2.0, path_novel=False, s_input=0.9, produced_defect=False
            )
        ]
    )
    assert s.fuzz_count == 2
    assert s.recent_gain == 2.0
    assert s.coverage_gain == 5.0  # 累计
    assert s.defect_count == 1


def test_pool_retires_after_no_gain() -> None:
    s = _seed(10)
    pool = SeedPool([s], capacity=10, retire_patience=2)
    no_gain = SeedOutcome(
        seed=s, new_coverage=0.0, path_novel=False, s_input=0.95, produced_defect=False
    )
    pool.update_after_round([no_gain])
    assert pool.n_active == 1  # 1 轮无产出，还没到 patience
    pool.update_after_round([no_gain])
    assert pool.n_active == 0  # 连续 2 轮无产出，退役
    assert pool.n_retired == 1
    assert pool.select(1, {}, _W, _GAMMA, False) == []  # 退役后选不出


def test_pool_productive_resets_no_gain() -> None:
    s = _seed(10)
    pool = SeedPool([s], capacity=10, retire_patience=2)
    no_gain = SeedOutcome(
        seed=s, new_coverage=0.0, path_novel=False, s_input=0.95, produced_defect=False
    )
    gain = SeedOutcome(
        seed=s, new_coverage=4.0, path_novel=False, s_input=0.95, produced_defect=False
    )
    pool.update_after_round([no_gain])
    pool.update_after_round([gain])  # 有产出，重置计数
    pool.update_after_round([no_gain])
    assert pool.n_active == 1  # 不该退役
