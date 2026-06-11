"""任务适配层协议。框架核心（engine/loop）只持有 TaskAdapter 引用。

任务差异全部收在适配器后面：模型加载、种子构建（共识过滤/基线判定）、覆盖
单元标定、批策略、带图前向、差分目标、失效判定、任务特有分析与图。模型家族
适配（torchvision/YOLO）和数据集适配挂在各任务适配器内部。

核心循环对一个批的调用序列：

    fw0 = adapter.forward(batch.x0, batch)          # no_grad，参考态
    u   = tracker.select_u(batch.seeds, fw0.unit_acts, u_size)
    x_adv = mutator.mutate(ctx)                      # 内部反复调 forward/objective1
    fw  = adapter.forward(x_adv, batch)              # no_grad，终态
    records, stats = adapter.judge(x_adv, fw0, fw, batch, s_input, rnd)

s_input 由核心从 v_sem 余弦得出；语义有效性门（γ_input）由 judge 自行应用，
语义失效样本不产记录（对应分类 triage 的 SEMANTIC_FAIL）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from mfuzz.core.config import Config
from mfuzz.core.records import FailureRecord, RunReport, Seed
from mfuzz.neurons.unit_coverage import UnitCoverageTracker


@dataclass
class Batch:
    """一个变异批。分类把同尺寸样本堆成大批；检测每图一批（B=1）。"""

    seeds: list[Seed]
    x0: Tensor  # (B, C, H, W) 像素域 [0,1]，变异在此域进行
    meta: dict = field(default_factory=dict)  # 适配器自用


@dataclass
class TaskForward:
    """一次带图前向的产物。核心只碰 unit_acts 与 v_sem，raw 对核心不透明。"""

    unit_acts: Tensor  # (B, N) 覆盖单元激活，可微
    v_sem: Tensor  # (B, C) 语义特征，可微
    raw: Any = None  # 任务自用：检测的 GraphResult、分类的 logits/acts


@dataclass
class SeedStats:
    """judge 返回的逐种子统计，核心回写调度。"""

    produced: int = 0  # 本轮该种子触发的新失效数
    path_novel: bool = False  # 分类的路径新颖；检测暂无此概念，恒 False


class TaskAdapter(ABC):
    """任务适配器基类。有状态：模型、数据、目标模型名都在实例上。"""

    task: str  # 注册名

    def __init__(self, cfg: Config, target: str, device: torch.device, out_dir: Path) -> None:
        self.cfg = cfg
        self.target = target
        self.device = device
        self.out_dir = out_dir

    # ---- 构建阶段 ----

    @abstractmethod
    def setup(self) -> None:
        """加载模型集合与数据。"""

    @abstractmethod
    def build_seeds(self) -> list[Seed]:
        """共识过滤 / 基线判定，锚点与基线失效写进 Seed。"""

    @abstractmethod
    def build_tracker(self) -> UnitCoverageTracker:
        """覆盖单元标定，返回账本（子类可带任务特有策略）。"""

    # ---- 循环内 ----

    @abstractmethod
    def make_batches(self, seeds: list[Seed]) -> list[Batch]:
        """批策略：分类堆大批，检测逐图。"""

    @abstractmethod
    def forward(self, x: Tensor, batch: Batch) -> TaskForward:
        """带图前向（调用方控制 no_grad 与否）。x 为像素域，归一化在适配器内。"""

    @abstractmethod
    def objective1(self, fw: TaskForward, batch: Batch) -> Tensor:
        """差分目标（标量，批内求和）：锚点上的偏离，越大越偏。"""

    @abstractmethod
    def judge(
        self,
        x_adv: Tensor,
        fw0: TaskForward,
        fw: TaskForward,
        batch: Batch,
        s_input: Tensor,
        round_idx: int,
    ) -> tuple[list[FailureRecord], list[SeedStats]]:
        """变异后判定。返回新失效记录（已过语义门与新颖性比对）与逐种子统计。"""

    # ---- 评估与报告钩子 ----

    def enrich_metrics(self, report: RunReport) -> dict[str, float]:
        """任务特有标量指标，并进 report.metrics。"""
        return {}

    def analyze(self, report: RunReport, out_dir) -> None:
        """任务特有深度分析（检测：结构归因/消融/真值核验；分类：聚类）。"""
        return None

    def plot_extras(self, report: RunReport, out_dir) -> None:
        """任务特有图。核心只画通用曲线。"""
        return None

    @classmethod
    def plot_combined(cls, per_target: dict[str, RunReport], cfg: Config, out_dir) -> None:
        """跨目标模型的汇总图（目标轮换时由入口调用一次）。"""
        return None
