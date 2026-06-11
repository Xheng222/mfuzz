"""任务无关的反馈驱动主循环。

每轮：调度选种 → 适配器组批 → 参考态前向 → 变异算子（联合目标）→ 终态前向
→ 覆盖更新 → 失效判定 → 回写种子统计 → 动态反馈调 λ → 终止判定。任务语义
全部在 TaskAdapter 后面，观测走 ProbeBus 事件，本模块不含任何任务分支。

终止条件：达最大轮数、种子池全部退役、覆盖与新失效连续多轮双低。
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from loguru import logger

from mfuzz.core.adapter import TaskAdapter
from mfuzz.core.config import Config
from mfuzz.core.probe import ProbeBus, ProbeContext, build_probes
from mfuzz.core.records import FAILURE_KINDS, RunReport, Seed
from mfuzz.core.types import FeedbackState
from mfuzz.neurons.unit_coverage import UnitCoverageTracker
from mfuzz.optimize.feedback import FeedbackController
from mfuzz.optimize.mutator import MutationContext, build_mutator

_GROWTH_EPS = 1e-4


class GenericSeedPool:
    """统一种子池：多维优先级选取，按结果回写，连续无产出退役。"""

    def __init__(self, seeds: list[Seed], cfg: Config, tracker: UnitCoverageTracker) -> None:
        self.seeds = seeds[: cfg.loop.pool_capacity]
        self.cfg = cfg
        self.tracker = tracker
        self._no_gain = dict.fromkeys((id(s) for s in self.seeds), 0)
        self._retired: set[int] = set()

    @property
    def n_active(self) -> int:
        return len(self.seeds) - len(self._retired)

    @property
    def n_retired(self) -> int:
        return len(self._retired)

    def _priority(self, s: Seed, stalled: bool) -> float:
        w = self.cfg.scheduler
        if s.last_s_input < self.cfg.semantic.gamma_input:
            return float("-inf")  # 输入有效性门
        penalty = w.w_fuzz_penalty * s.fuzz_count * (2.0 if stalled else 1.0)
        return (
            w.w_coverage * s.recent_gain
            + w.w_novelty * (1.0 if s.path_novel else 0.0)
            + w.w_defect * s.defect_count
            + w.w_cccov_gap * self.tracker.gap_for(s)
            - penalty
        )

    def select(self, k: int, stalled: bool) -> list[Seed]:
        alive = [s for s in self.seeds if id(s) not in self._retired]
        if k >= len(alive):
            return alive
        scored = sorted(
            ((self._priority(s, stalled), i) for i, s in enumerate(alive)),
            key=lambda t: (-t[0], t[1]),
        )
        return [alive[i] for _, i in scored[:k]]

    def update(self, seed: Seed, new_cov: float, produced: int, path_novel: bool, s_in: float):
        seed.fuzz_count += 1
        seed.recent_gain = new_cov
        seed.coverage_gain += new_cov
        seed.path_novel = path_novel
        seed.last_s_input = s_in
        seed.defect_count += produced
        key = id(seed)
        if new_cov > 0 or produced > 0 or path_novel:
            self._no_gain[key] = 0
        else:
            self._no_gain[key] += 1
            if self._no_gain[key] >= self.cfg.loop.retire_patience:
                self._retired.add(key)


def run_loop(
    adapter: TaskAdapter,
    tracker: UnitCoverageTracker,
    seeds: list[Seed],
    cfg: Config,
    out_dir: Path,
) -> RunReport:
    """单目标模型的一次完整 fuzzing 运行。返回统一 RunReport。"""
    torch.manual_seed(cfg.random_seed)
    probes = ProbeBus(build_probes(cfg.probes.enabled))
    probes.emit("run_start", ProbeContext(cfg, adapter, tracker, out_dir))
    mutator = build_mutator(cfg.optimize.mutator)

    report = RunReport()
    report.fail_history = {k: [] for k in FAILURE_KINDS}

    # 初始覆盖：种子参考态过一遍账本
    with torch.no_grad():
        for batch in adapter.make_batches(seeds):
            fw0 = adapter.forward(batch.x0, batch)
            tracker.update(fw0.unit_acts)
            del fw0
    cncov0 = tracker.cncov
    report.cncov_history.append(cncov0)
    logger.info(
        f"[{adapter.target}] 种子 {len(seeds)}，关键单元 {tracker.profile.num_critical}，"
        f"CNCov_0 = {cncov0:.3f}"
    )

    sem_thr = cfg.feedback.sem_shift_threshold
    if sem_thr <= 0.0:
        sem_thr = 1.0 - cfg.semantic.gamma_input
    feedback = FeedbackController(
        cfg.feedback,
        cfg.coverage.lambda2,
        cfg.semantic.lambda3,
        sem_thr,
        cfg.coverage.lambda2_bounds,
        cfg.semantic.lambda3_bounds,
    )
    pool = GenericSeedPool(seeds, cfg, tracker)

    n_fuzzed = 0
    n_new_failures = 0
    n_sem_valid = 0
    s_input_sum = 0.0
    double_low = 0
    rounds_run = 0
    t0 = time.perf_counter()

    for rnd in range(cfg.loop.max_iterations):
        lam2, lam3 = feedback.lambda2, feedback.lambda3
        selected = pool.select(cfg.loop.seeds_per_round, feedback.coverage_stalled)
        if not selected:
            logger.info(f"[{adapter.target}] 第 {rnd} 轮：种子池已空，终止")
            break
        rounds_run += 1

        round_new = dict.fromkeys(FAILURE_KINDS, 0)
        round_sem_sum = 0.0
        round_novel = 0
        n_round = 0
        for batch in adapter.make_batches(selected):
            with torch.no_grad():
                fw0 = adapter.forward(batch.x0, batch)
            probes.emit("forward", rnd, -1, batch, fw0)
            u_idx = (
                tracker.select_u(batch.seeds, fw0.unit_acts, cfg.coverage.u_size)
                if lam2 != 0.0
                else None
            )
            ctx = MutationContext(
                adapter=adapter,
                batch=batch,
                tracker=tracker,
                u_idx=u_idx,
                v_sem0=fw0.v_sem.detach(),
                lam2=lam2,
                lam3=lam3,
                opt=cfg.optimize,
                rnd=rnd,
                emit=probes.emit,
            )
            x_adv = mutator.mutate(ctx)

            with torch.no_grad():
                fw = adapter.forward(x_adv, batch)
                new_cov_total = tracker.update(fw.unit_acts)
                s_in = torch.cosine_similarity(fw.v_sem, fw0.v_sem, dim=1)  # (B,)
            probes.emit("mutation_done", rnd, batch, x_adv, fw)

            records, stats = adapter.judge(x_adv, fw0, fw, batch, s_in, rnd)
            probes.emit("judged", rnd, batch, records)
            report.failures.extend(records)
            for r in records:
                round_new[r.kind] += 1
            del fw0, fw

            # 新覆盖按"是否整批带来"回写：批内逐种子归因在检测（B=1）下自然成立，
            # 分类大批下均摊（与调度只看相对优先级一致）。
            per_seed_cov = new_cov_total / max(len(batch.seeds), 1)
            for i, seed in enumerate(batch.seeds):
                si = float(s_in[i])
                st = stats[i]
                n_fuzzed += 1
                n_round += 1
                s_input_sum += si
                round_sem_sum += si
                if si >= cfg.semantic.gamma_input:
                    n_sem_valid += 1
                if st.path_novel:
                    round_novel += 1
                n_new_failures += st.produced
                pool.update(seed, per_seed_cov, st.produced, st.path_novel, si)
        if adapter.device.type == "cuda":
            torch.cuda.empty_cache()

        cncov_now = tracker.cncov
        delta = cncov_now - report.cncov_history[-1]
        round_total = sum(round_new.values())
        report.cncov_history.append(cncov_now)
        for k, v in round_new.items():
            report.fail_history[k].append(v)
        report.rft_history.append(round_total / n_round if n_round else 0.0)
        report.sem_shift_history.append(1.0 - round_sem_sum / n_round if n_round else 0.0)
        report.lambda_history.append((lam2, lam3))
        stats_round = {
            "cncov": cncov_now,
            "delta_cncov": delta,
            "new_failures": float(round_total),
            "lambda2": lam2,
            "lambda3": lam3,
            "n_active": float(pool.n_active),
        }
        probes.emit("round_end", rnd, stats_round)
        feedback.step(
            FeedbackState(
                delta_cncov=delta,
                rft=report.rft_history[-1],
                mean_sem_shift=report.sem_shift_history[-1],
                path_novel_ratio=round_novel / n_round if n_round else 0.0,
            )
        )

        if rnd % max(1, cfg.loop.log_interval) == 0:
            logger.info(
                f"[{adapter.target}] 轮 {rnd}: CNCov {cncov_now:.3f}(Δ{delta:+.4f}) "
                f"新失效 {round_total} λ2={lam2:.3f} λ3={lam3:.3f} 活跃 {pool.n_active}"
            )
        double_low = double_low + 1 if delta < _GROWTH_EPS and round_total == 0 else 0
        if double_low >= cfg.loop.growth_patience:
            logger.info(f"[{adapter.target}] 覆盖与新失效连续 {double_low} 轮双低，终止")
            break

    elapsed = time.perf_counter() - t0
    report.total_rounds = rounds_run
    report.elapsed_time = elapsed
    report.metrics = {
        "n_seeds": float(len(seeds)),
        "n_rounds": float(rounds_run),
        "n_fuzzed": float(n_fuzzed),
        "n_new_failures": float(n_new_failures),
        "rft": n_new_failures / n_fuzzed if n_fuzzed else 0.0,
        "cncov_0": cncov0,
        "cncov_final": tracker.cncov,
        "cncov_gain": tracker.cncov - cncov0,
        "n_units": float(tracker.profile.num_units),
        "n_critical": float(tracker.profile.num_critical),
        "input_valid_rate": n_sem_valid / n_fuzzed if n_fuzzed else 0.0,
        "mean_s_input": s_input_sum / n_fuzzed if n_fuzzed else 0.0,
        "lambda2_final": feedback.lambda2,
        "lambda3_final": feedback.lambda3,
        "n_seeds_retired": float(pool.n_retired),
        "elapsed_time": elapsed,
    }
    probes.collect(report)
    report.metrics.update(adapter.enrich_metrics(report))
    logger.info(
        f"[{adapter.target}] 循环完成：{rounds_run} 轮，新失效 {n_new_failures}，"
        f"CNCov {cncov0:.3f}->{tracker.cncov:.3f}，耗时 {elapsed:.1f}s"
    )
    return report
