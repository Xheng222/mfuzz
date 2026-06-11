"""观测探针协议与注册表。

核心循环在固定关口发事件，探针订阅、只读上下文、互相独立。事件携带未消化
的对象（TaskForward、记录列表、账本状态），探针自己积累状态：

- on_run_start(ctx)            ctx: config / adapter / tracker / out_dir
- on_forward(rnd, step, batch, fw)        变异内逐步（含 step=-1 的参考态前向）
- on_mutation_done(rnd, batch, x_adv, fw)
- on_judged(rnd, batch, records)
- on_round_end(rnd, stats)     stats: 覆盖 / λ / 本轮新失效等标量
- on_run_end(report) -> dict | None       返回 {"curves": {名: 序列}, "metrics": {名: 标量}}
  曲线由通用报告画折线、标量进 metrics 表；探针也可向 out_dir/probes/<名>/ 自写产物。

探针异常不中断主循环：发射器捕获并记日志。配置 [probes] enabled 按注册名启用。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from mfuzz.core.adapter import Batch, TaskAdapter, TaskForward
    from mfuzz.core.config import Config
    from mfuzz.core.records import FailureRecord, RunReport
    from mfuzz.neurons.unit_coverage import UnitCoverageTracker


@dataclass
class ProbeContext:
    cfg: Config
    adapter: TaskAdapter
    tracker: UnitCoverageTracker
    out_dir: Path


class Probe:
    """探针基类。全部钩子默认空实现，子类只覆写关心的关口。"""

    name: str = "probe"

    def on_run_start(self, ctx: ProbeContext) -> None: ...

    def on_forward(self, rnd: int, step: int, batch: Batch, fw: TaskForward) -> None: ...

    def on_mutation_done(self, rnd: int, batch: Batch, x_adv, fw: TaskForward) -> None: ...

    def on_judged(self, rnd: int, batch: Batch, records: list[FailureRecord]) -> None: ...

    def on_round_end(self, rnd: int, stats: dict[str, float]) -> None: ...

    def on_run_end(self, report: RunReport) -> dict[str, Any] | None:
        return None


_REGISTRY: dict[str, Callable[[], Probe]] = {}


def register_probe(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def build_probes(names: list[str]) -> list[Probe]:
    # 导入内置探针包触发注册
    import mfuzz.probes  # noqa: F401

    out: list[Probe] = []
    for n in names:
        if n not in _REGISTRY:
            raise KeyError(f"未注册的探针：{n}（可用：{sorted(_REGISTRY)}）")
        out.append(_REGISTRY[n]())
    return out


@dataclass
class ProbeBus:
    """事件发射器。逐探针调用，异常记日志不中断。"""

    probes: list[Probe] = field(default_factory=list)

    def emit(self, event: str, *args, **kwargs) -> None:
        for p in self.probes:
            try:
                getattr(p, f"on_{event}")(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 探针不允许带崩主循环
                logger.warning(f"探针 {p.name} 在 {event} 抛异常：{e}")

    def collect(self, report: RunReport) -> None:
        """run_end：把各探针返回的曲线与标量并进 report。"""
        for p in self.probes:
            try:
                out = p.on_run_end(report)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"探针 {p.name} 在 run_end 抛异常：{e}")
                continue
            if not out:
                continue
            for k, v in out.get("curves", {}).items():
                report.curves[f"{p.name}.{k}"] = list(v)
            for k, v in out.get("metrics", {}).items():
                report.metrics[f"{p.name}.{k}"] = float(v)
