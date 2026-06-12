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
class Detection:
    """单个检测框（检测任务公共结构）。

    label 存类别名而不是索引：各模型家族的索引约定不同（torchvision 91 类含
    占位项、YOLO 80 类无占位），跨模型差分必须在类别名空间比较。
    box 为 xyxy，原图坐标系。
    """

    model: str
    box: Tensor  # (4,) xyxy
    label: str
    score: float


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
    coverage_gain: float = 0.0  # 累计带来的新覆盖关键神经元数
    recent_gain: float = 0.0  # 最近一次变异带来的新覆盖数，调度看产出用
    last_s_input: float = 1.0  # 最近一次变异的 S_input，调度的输入有效性门
    path_novel: bool = False  # 最近一次变异是否探索了新路径
    defect_count: int = 0  # 累计触发缺陷数，调度的缺陷历史信号（权重低，含随机性）


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
    source_image: Tensor | None = None  # 原始种子像素图 (C,H,W)，供原始/变异对比图；可空


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
    # 每轮累计的缺陷多样性，单调不减，套用覆盖增长曲线画「多样性增长」。
    pair_history: list[int] = field(default_factory=list)  # 累计不同 (源,目标) 类别对数
    target_history: list[int] = field(default_factory=list)  # 累计不同目标类别数
    metrics: dict[str, float] = field(default_factory=dict)  # 标量汇总指标
    curves: dict[str, list[float]] = field(default_factory=dict)  # 诊断曲线，如逐步目标置信度
    total_iterations: int = 0
    elapsed_time: float = 0.0

    @property
    def num_defects(self) -> int:
        return len(self.defects)


# ---- 公共配置节与 TOML 加载工具 ----
# SchedulerConfig 与 FeedbackConfig 由新旧两套代码共用；_load_raw / _deep_merge 是
# extends 链 TOML 加载工具，core/config.py 的统一 Config 也用它。


@dataclass
class SchedulerConfig:
    pool_capacity: int = 256  # 种子池容量上限，超出则采样
    retire_patience: int = 6  # 种子连续多少轮无收益即退役
    w_coverage: float = 1.0  # 最近覆盖增量权重
    w_novelty: float = 0.8  # 路径新颖权重
    w_defect: float = 0.3  # 缺陷历史权重（含随机性，压低）
    w_fuzz_penalty: float = 0.5  # 变异次数惩罚，抑制过度变异
    w_cccov_gap: float = 1.0  # 类关键覆盖缺口权重，向覆盖不足的类倾斜


@dataclass
class FeedbackConfig:
    """动态反馈控制器的机制参数。λ 的夹界在各自模块（neurons.lambda2_bounds 等）。"""

    enabled: bool = True  # 关闭即静态权重基线
    window: int = 10  # 滑动窗口轮数
    step_up: float = 1.15  # 触发时的乘法上调因子
    step_down: float = 0.95  # 不触发时的乘法回落因子
    cov_stall_eps: float = 0.005  # 窗口内 ΔCNCov 均值低于此判覆盖停滞
    sem_shift_threshold: float = -1.0  # 语义偏移阈，<=0 表示自动取 1-γ_input


def _deep_merge(base: dict, over: dict) -> dict:
    """子表覆盖父表：同名子节递归合并，标量/数组整体替换。'extends' 键不参与合并。"""
    out = dict(base)
    for k, v in over.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: Path, seen: set[Path] | None = None) -> dict:
    """读 TOML 并解析 extends 链：父表先加载，子表覆盖父表（后者覆盖前者）。"""
    seen = seen if seen is not None else set()
    rp = path.resolve()
    if rp in seen:
        raise ValueError(f"配置 extends 形成环：{rp}")
    seen.add(rp)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    parent = raw.get("extends")
    if parent:
        base_raw = _load_raw(path.parent / parent, seen)
        raw = _deep_merge(base_raw, raw)
    return raw
