"""公共数据结构。集中定义，避免各模块重复。

字段按 docs/实现方案.md 3.3 设计。部分字段（s_input、s_path、
critical_activation 等）在后续阶段才被填充，这里先给出稳定契约。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from torch import Tensor


@dataclass
class LayerActivation:
    """单层激活。value 为每神经元一个标量的激活向量。

    Conv2d 输出经全局平均池化后形状为 (C,)（单样本）或 (B, C)（批）。
    Linear 输出形状一致。
    """

    name: str
    value: Tensor


@dataclass
class ModelPrediction:
    """单个模型对单个样本的预测。

    activations 默认空：差分预测只需 label 与 probs，仅目标模型在需要
    覆盖/语义计算时才填激活，避免为所有模型保存激活。
    """

    label: int  # argmax 类别，模型 1000 类输出空间
    probs: Tensor  # (num_classes,) softmax 概率
    activations: dict[str, Tensor] = field(default_factory=dict)  # 层名 -> (C,) 激活


@dataclass
class Seed:
    """一个种子样本。

    consensus_label 在 Phase 1 共识过滤后填入并据此筛选；Phase 0 构建时
    仅置 true_label，consensus_label 留 -1 占位。
    """

    image: Tensor  # 归一化后图像 (C, H, W)
    true_label: int  # 数据集真实标签，模型 1000 类空间
    consensus_label: int = -1  # 各模型一致预测；-1 表示未确定
    model_confidences: dict[str, float] = field(default_factory=dict)  # 各模型对共识标签的置信度
    fuzz_count: int = 0  # 已变异次数
    coverage_gain: float = 0.0  # 累计带来的覆盖增益
    path_novel: bool = False  # 最近一次变异是否探索了新路径


@dataclass
class Candidate:
    """一次变异产生的候选样本。"""

    image: Tensor  # 变异后图像 (C, H, W)
    source: Seed  # 来源种子
    predictions: dict[str, ModelPrediction]  # 模型名 -> 预测
    s_input: float = 1.0  # 输入语义相似度 S_input
    s_path: float = 1.0  # 路径相似度 S_path


@dataclass
class DefectRecord:
    """候选缺陷记录。在候选信息外额外存关键神经元激活向量，供聚类。"""

    image: Tensor  # 变异后图像 (C, H, W)
    source_label: int  # 来源共识标签 c
    target_label: int  # 目标模型偏离后的预测
    target_model: str  # 触发缺陷时的目标模型名
    s_input: float  # 输入语义相似度
    s_path: float  # 路径相似度
    perturbation: float = 0.0  # 相对原始种子的扰动大小
    critical_activation: Tensor | None = None  # 在 D_en 上的激活向量，路径指纹；Phase 2 起填充


@dataclass
class FeedbackState:
    """动态反馈的滑动窗口四指标。"""

    delta_cncov: float = 0.0  # 覆盖增长率 ΔCNCov
    rft: float = 0.0  # 候选缺陷触发率
    mean_sem_shift: float = 0.0  # 输入语义偏移均值 d̄_sem
    path_novel_ratio: float = 0.0  # 路径新颖样本比例


@dataclass
class FuzzReport:
    """一次完整运行的输出。"""

    defects: list[DefectRecord] = field(default_factory=list)
    cncov_history: list[float] = field(default_factory=list)  # 每轮 CNCov
    cccov_history: list[dict[int, float]] = field(default_factory=list)  # 每轮各类 CCCov
    rft_history: list[float] = field(default_factory=list)
    sem_shift_history: list[float] = field(default_factory=list)
    lambda_history: list[tuple[float, float]] = field(default_factory=list)  # (λ2, λ3)
    metrics: dict[str, float] = field(default_factory=dict)  # 标量汇总指标
    curves: dict[str, list[float]] = field(default_factory=dict)  # 诊断曲线，如逐步目标置信度
    total_iterations: int = 0
    elapsed_time: float = 0.0

    @property
    def num_defects(self) -> int:
        return len(self.defects)


# ---- 配置 ----
# 与 configs/base.toml 各节对应。经 tomllib 读入填进 dataclass，不裸 dict 跨模块传。


@dataclass
class DatasetConfig:
    name: str = "imagenet"
    seed_split: str = "val"
    profile_split: str = "train"
    seed_size: int = 200
    profile_subset_size: int = 2000
    batch_size: int = 32


@dataclass
class ModelsConfig:
    names: list[str] = field(default_factory=lambda: ["resnet50", "vgg16_bn", "mobilenet_v2"])
    target_idx: int = 0


@dataclass
class DifferentialConfig:
    lambda1: float = 1.0


@dataclass
class NeuronsConfig:
    activation_threshold: float = 0.5
    critical_threshold: float = 0.9
    alpha: float = 0.5
    mode: str = "fusion"  # fusion | frequency | attribution
    u_size: int = 16


@dataclass
class SemanticConfig:
    gamma_input: float = 0.9
    theta_path: float = 0.0


@dataclass
class FuzzConfig:
    max_iterations: int = 100
    pgd_steps: int = 10
    step_size: float = 0.01
    epsilon: float = 0.03
    lambda2: float = 0.5
    lambda3: float = 0.5
    batch_size: int = 8
    log_interval: int = 20


@dataclass
class FeedbackConfig:
    enabled: bool = True
    window: int = 10
    lambda2_bounds: list[float] = field(default_factory=lambda: [0.1, 2.0])
    lambda3_bounds: list[float] = field(default_factory=lambda: [0.1, 2.0])


@dataclass
class Config:
    random_seed: int = 42
    device: str = "cuda"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    differential: DifferentialConfig = field(default_factory=DifferentialConfig)
    neurons: NeuronsConfig = field(default_factory=NeuronsConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    fuzz: FuzzConfig = field(default_factory=FuzzConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config(
        random_seed=raw.get("random_seed", 42),
        device=raw.get("device", "cuda"),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        models=ModelsConfig(**raw.get("models", {})),
        differential=DifferentialConfig(**raw.get("differential", {})),
        neurons=NeuronsConfig(**raw.get("neurons", {})),
        semantic=SemanticConfig(**raw.get("semantic", {})),
        fuzz=FuzzConfig(**raw.get("fuzz", {})),
        feedback=FeedbackConfig(**raw.get("feedback", {})),
    )
