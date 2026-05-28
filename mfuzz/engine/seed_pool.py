"""种子池管理（实现方案 4.4）。

池里是共识过滤后的原始种子，成员固定。每轮调度器从中选一批变异，变异结果回写到对应
种子的统计（fuzz_count、最近新覆盖、路径新颖、最近 S_input、缺陷计数），供下一轮调度。
连续多轮既无新覆盖、又无新路径、也不触发缺陷的种子退役，不再参与选取，对应「三者皆无
的丢弃」与「池配合调度做采样」。

本阶段把变异候选的统计回写到来源种子，不把候选本身当新种子加入池：这样 S_input 始终
相对真正的原始 x0 度量，避免候选当种子后语义偏移逐代累积绕过 γ_input。重选同一种子仍
有产出，因为每轮目标神经元集合 U 随覆盖推进而变、λ 也可能被反馈改动。见实现方案第十章。
"""

from __future__ import annotations

from dataclasses import dataclass

from mfuzz.core.types import SchedulerConfig, Seed
from mfuzz.engine.scheduler import select_seeds


@dataclass
class SeedOutcome:
    """一个种子在某轮变异后的结果，用于回写统计。"""

    seed: Seed
    new_coverage: float  # 本轮带来的新覆盖关键神经元数
    path_novel: bool  # 是否探索了新路径（S_path < θ_path）
    s_input: float  # 本轮变异的 S_input
    produced_defect: bool  # 是否触发缺陷


class SeedPool:
    """成员固定的种子池，按调度器优先级选取，按结果回写统计并退役无产出的种子。"""

    def __init__(self, seeds: list[Seed], capacity: int, retire_patience: int) -> None:
        self.seeds = seeds[:capacity]  # 超容量取前 capacity 个（种子已随机抽样、可复现）
        self.retire_patience = retire_patience
        self._idx = {id(s): i for i, s in enumerate(self.seeds)}
        self._no_gain = [0] * len(self.seeds)
        self._retired = [False] * len(self.seeds)

    def active(self) -> list[Seed]:
        return [s for i, s in enumerate(self.seeds) if not self._retired[i]]

    @property
    def n_active(self) -> int:
        return sum(1 for r in self._retired if not r)

    @property
    def n_retired(self) -> int:
        return sum(1 for r in self._retired if r)

    def select(
        self,
        k: int,
        cccov: dict[int, float],
        weights: SchedulerConfig,
        gamma_input: float,
        coverage_stalled: bool,
    ) -> list[Seed]:
        return select_seeds(self.active(), k, cccov, weights, gamma_input, coverage_stalled)

    def update_after_round(self, outcomes: list[SeedOutcome]) -> None:
        for o in outcomes:
            i = self._idx[id(o.seed)]
            s = o.seed
            s.fuzz_count += 1
            s.recent_gain = o.new_coverage
            s.coverage_gain += o.new_coverage
            s.path_novel = o.path_novel
            s.last_s_input = o.s_input
            if o.produced_defect:
                s.defect_count += 1
            productive = o.new_coverage > 0 or o.path_novel or o.produced_defect
            self._no_gain[i] = 0 if productive else self._no_gain[i] + 1
            if self._no_gain[i] >= self.retire_patience:
                self._retired[i] = True
