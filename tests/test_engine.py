"""统一种子池单元测试。

验证多维优先级（输入有效性门、覆盖增量、类缺口、变异次数惩罚与停滞加倍）、
按优先级选取、统计回写与无产出退役。全程 CPU、不依赖模型；覆盖账本用只提供
gap_for 的桩。
"""

from __future__ import annotations

from mfuzz.core.config import Config
from mfuzz.core.records import ConsensusAnchor, Seed
from mfuzz.engine.loop import GenericSeedPool


class _StubTracker:
    """只提供调度需要的 gap_for。"""

    def __init__(self, gaps: dict[str, float] | None = None) -> None:
        self.gaps = gaps or {}

    def gap_for(self, seed: Seed) -> float:
        return self.gaps.get(seed.anchors[0].label, 0.0)


def _seed(label: str = "10", **kw) -> Seed:
    return Seed(anchors=[ConsensusAnchor(label=label)], **kw)


def _pool(seeds, gaps=None, retire_patience=6, capacity=256) -> GenericSeedPool:
    cfg = Config()
    cfg.loop.retire_patience = retire_patience
    cfg.loop.pool_capacity = capacity
    return GenericSeedPool(seeds, cfg, _StubTracker(gaps))  # type: ignore[arg-type]


def test_priority_validity_gate() -> None:
    bad = _seed(last_s_input=0.5)
    ok = _seed(last_s_input=0.95)
    pool = _pool([bad, ok])
    assert pool._priority(bad, stalled=False) == float("-inf")
    assert pool._priority(ok, stalled=False) > float("-inf")


def test_priority_fuzz_penalty_and_stall() -> None:
    fresh = _seed(fuzz_count=0)
    fuzzed = _seed(fuzz_count=10)
    pool = _pool([fresh, fuzzed])
    assert pool._priority(fresh, stalled=False) > pool._priority(fuzzed, stalled=False)
    # 覆盖停滞时惩罚加倍，差距进一步拉大
    assert pool._priority(fuzzed, stalled=True) < pool._priority(fuzzed, stalled=False)


def test_select_prefers_recent_gain() -> None:
    a = _seed(recent_gain=5.0)
    b = _seed(recent_gain=0.0)
    pool = _pool([b, a])
    assert pool.select(1, stalled=False) == [a]


def test_select_favors_gap_class() -> None:
    s10 = _seed("10")
    s11 = _seed("11")
    pool = _pool([s10, s11], gaps={"10": 0.1, "11": 0.9})
    assert pool.select(1, stalled=False) == [s11]


def test_select_returns_all_when_k_ge_len() -> None:
    seeds = [_seed("10"), _seed("11")]
    pool = _pool(seeds)
    assert pool.select(5, stalled=False) == seeds


def test_pool_capacity_truncates() -> None:
    pool = _pool([_seed(str(i)) for i in range(5)], capacity=3)
    assert pool.n_active == 3


def test_pool_update_writes_stats_and_retires() -> None:
    s = _seed()
    pool = _pool([s], retire_patience=2)
    pool.update(s, new_cov=3.0, produced=1, path_novel=False, s_in=0.95)
    assert s.fuzz_count == 1
    assert s.recent_gain == 3.0
    assert s.coverage_gain == 3.0
    assert s.defect_count == 1
    assert s.last_s_input == 0.95

    pool.update(s, new_cov=0.0, produced=0, path_novel=False, s_in=0.95)
    assert pool.n_active == 1  # 1 轮无产出，未到 patience
    pool.update(s, new_cov=0.0, produced=0, path_novel=False, s_in=0.95)
    assert pool.n_active == 0  # 连续 2 轮无产出，退役
    assert pool.n_retired == 1
    assert pool.select(1, stalled=False) == []


def test_pool_productive_resets_no_gain() -> None:
    s = _seed()
    pool = _pool([s], retire_patience=2)
    pool.update(s, new_cov=0.0, produced=0, path_novel=False, s_in=0.95)
    pool.update(s, new_cov=4.0, produced=0, path_novel=False, s_in=0.95)  # 有产出，重置
    pool.update(s, new_cov=0.0, produced=0, path_novel=False, s_in=0.95)
    assert pool.n_active == 1
