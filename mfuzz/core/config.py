"""统一实验配置。一份 TOML 对应一个完整实验，task 字段选择任务适配器。

公共节（run/models/coverage/semantic/optimize/scheduler/loop/feedback/probes）
由框架核心消费；任务自有节（[classification] / [detection]）由对应适配器从
raw 里解析，框架核心不解释。extends 链式继承沿用 types._load_raw。

消融照旧走旋钮：coverage.lambda2 / semantic.lambda3 置零即该项不进梯度，
feedback.enabled=false 即静态权重，loop.max_iterations=0 即跳过 fuzzing
循环、只跑适配器的分析阶段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mfuzz.core.types import FeedbackConfig, SchedulerConfig, _load_raw


@dataclass
class RunConfig:
    out: str = "output/run"


@dataclass
class ModelsConfig:
    names: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)  # 轮换的目标模型；空 = names 全部轮换


@dataclass
class CoverageConfig:
    """单元覆盖的公共旋钮。单元的含义由适配器定（神经元 / 层×尺度×通道）。"""

    t_freq: float = 0.5  # profiling 频率项激活阈值
    t_cov: float = 0.85  # 覆盖判定阈值
    critical_tau: float = 0.5  # 关键度分位阈
    u_size: int = 16  # 每轮覆盖目标单元集合 U 的大小
    lambda2: float = 0.5  # 覆盖目标权重（=0 自动消融）
    lambda2_bounds: list[float] = field(default_factory=lambda: [0.1, 2.0])


@dataclass
class SemanticConfig:
    gamma_input: float = 0.9  # S_input 有效下界
    lambda3: float = 0.5  # 语义保持权重（=0 自动消融）
    lambda3_bounds: list[float] = field(default_factory=lambda: [0.1, 2.0])


@dataclass
class OptimizeConfig:
    mutator: str = "pgd"  # 变异算子注册名：pgd | corruption
    pgd_steps: int = 10
    step_size: float = 0.01
    epsilon: float = 0.03
    # corruption 算子的旋钮（pgd 不读）：每图随机选一种腐蚀，强度 1-5 对照 ImageNet-C
    corruption_ops: list[str] = field(
        default_factory=lambda: ["gaussian_noise", "gaussian_blur", "brightness", "contrast"]
    )
    corruption_severity: int = 3


@dataclass
class LoopConfig:
    max_iterations: int = 100  # 0 = 跳过 fuzzing 循环
    seeds_per_round: int = 8
    log_interval: int = 10
    growth_patience: int = 8  # 覆盖与新失效连续多少轮双低即终止
    retire_patience: int = 6  # 种子连续多少轮无产出即退役
    pool_capacity: int = 256


@dataclass
class ProbesConfig:
    enabled: list[str] = field(default_factory=list)  # 探针注册名列表


@dataclass
class Config:
    task: str = "classification"  # classification | detection
    random_seed: int = 42
    device: str = "cuda"
    run: RunConfig = field(default_factory=RunConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    probes: ProbesConfig = field(default_factory=ProbesConfig)
    raw: dict = field(default_factory=dict)  # 完整原始 TOML，适配器读自己的节

    def target_names(self) -> list[str]:
        return self.models.targets or list(self.models.names)


def load_config(path: str | Path) -> Config:
    raw = _load_raw(Path(path))
    raw.pop("extends", None)
    return Config(
        task=raw.get("task", "classification"),
        random_seed=raw.get("random_seed", 42),
        device=raw.get("device", "cuda"),
        run=RunConfig(**raw.get("run", {})),
        models=ModelsConfig(**raw.get("models", {})),
        coverage=CoverageConfig(**raw.get("coverage", {})),
        semantic=SemanticConfig(**raw.get("semantic", {})),
        optimize=OptimizeConfig(**raw.get("optimize", {})),
        scheduler=SchedulerConfig(**raw.get("scheduler", {})),
        loop=LoopConfig(**raw.get("loop", {})),
        feedback=FeedbackConfig(**raw.get("feedback", {})),
        probes=ProbesConfig(**raw.get("probes", {})),
        raw=raw,
    )
