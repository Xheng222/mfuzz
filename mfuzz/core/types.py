"""公共数据结构。集中定义，避免各模块重复。

字段按 docs/实现方案.md 3.3 设计。部分字段（s_input、s_path、
critical_activation 等）在后续阶段才被填充，这里先给出稳定契约。
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    critical_activation: Tensor  # 在 D_en 上的激活向量，路径指纹
    perturbation: float = 0.0  # 相对原始种子的扰动大小


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
    total_iterations: int = 0
    elapsed_time: float = 0.0

    @property
    def num_defects(self) -> int:
        return len(self.defects)
