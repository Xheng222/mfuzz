"""任务无关的统一数据结构：共识锚点、失效记录、种子、运行报告。

框架核心（engine/loop、evaluate/run_report）只认这里的类型。任务差异收进
适配层（mfuzz/tasks/*）：分类的标签翻转是 kind="cls" 的退化情形，检测产生
四类失效。core/types.py 里的旧分类结构（Seed/DefectRecord/FuzzReport）降级
为分类适配器的内部表示，不再被框架核心引用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor

FAILURE_KINDS = ("miss", "spurious", "cls", "loc")


@dataclass
class ConsensusAnchor:
    """共识锚点。分类 = 共识标签（box 为 None，锚是整图）；检测 = 共识对象。"""

    label: str  # 统一类别名空间，分类侧由适配器转换
    box: Tensor | None = None
    support: dict[str, float] = field(default_factory=dict)  # 模型名 -> 支持强度


@dataclass
class FailureRecord:
    """一条失效记录。kind 是 FAILURE_KINDS 之一，分类只产生 cls。"""

    kind: str
    anchor: ConsensusAnchor | None  # spurious 无锚（不被任何共识支持）
    observed_label: str | None = None  # 目标模型实际输出；miss 为 None
    observed_box: Tensor | None = None
    observed_score: float | None = None
    s_input: float = 1.0
    image_ref: str | None = None  # 来源图名或落盘变异图的相对路径
    round_idx: int = -1  # 产生于第几轮；-1 = 基线（自然失效）
    extra: dict = field(default_factory=dict)  # 任务自有上下文，核心不解释


@dataclass
class Seed:
    """统一种子：样本 + 共识锚点 + 基线失效 + 调度统计。

    样本用 image 或 path 之一承载（分类是预处理后的张量，检测按路径惰性加载，
    由适配器自行解释）。调度统计字段沿用分类版语义，scheduler 通用使用。
    """

    anchors: list[ConsensusAnchor]
    image: Tensor | None = None
    path: Path | None = None
    baseline_failures: list[FailureRecord] = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # 适配器自用（真实标签、语义参考等）
    # ---- 调度统计 ----
    fuzz_count: int = 0
    recent_gain: float = 0.0  # 最近一轮带来的新覆盖单元数
    coverage_gain: float = 0.0
    path_novel: bool = False
    last_s_input: float = 1.0
    defect_count: int = 0


@dataclass
class RunReport:
    """一次完整运行的统一输出。逐轮历史供曲线，failures 供下游分析。"""

    failures: list[FailureRecord] = field(default_factory=list)
    cncov_history: list[float] = field(default_factory=list)
    fail_history: dict[str, list[int]] = field(default_factory=dict)  # kind -> 每轮新失效数
    rft_history: list[float] = field(default_factory=list)
    sem_shift_history: list[float] = field(default_factory=list)
    lambda_history: list[tuple[float, float]] = field(default_factory=list)
    curves: dict[str, list[float]] = field(default_factory=dict)  # 探针与诊断曲线
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)  # 任务自有历史（分类 CCCov 等）
    total_rounds: int = 0
    elapsed_time: float = 0.0

    @property
    def num_failures(self) -> int:
        return len(self.failures)
