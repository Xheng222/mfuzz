"""五维指标（实现方案 4.5）。

五个维度：缺陷、覆盖、语义、多样性、效率。五维一起报告，用来区分某个模块到底是
真带来收益，还是只改变了覆盖曲线的形状。

缺陷维度的 #Faults、RFT 与覆盖维度的 CNCov、CCCov、效率维度的运行时间、单位时间
缺陷数，已经在 runner 里随 FuzzReport 算好。这里补齐三类此前没有的量：

- FRE（单位代价缺陷数）：研究材料只给了名字没给公式。这里定义为每一步像素变异换来
  的缺陷数，即 FRE = #Faults / (被变异种子数 × PGD 步数)。它和 RFT 的区别是把每个
  种子内部的多步迭代也算进代价，反映"单位生成开销"而非"单位种子"的产出。
- #Classes：缺陷覆盖的原始类别数，即缺陷里出现过的不同共识标签个数。
- OI（类别覆盖均衡度 / 输出公正性）：缺陷在各原始类别上的分布是否均衡。取缺陷的
  按类计数分布的香农熵，除以机会集合大小的对数做归一化。机会集合是种子里出现过的
  共识类别数 n_consensus_classes；缺这个数时退回缺陷自身覆盖到的类别数。OI=1 表示
  缺陷在所有有机会的类别上完全均匀，0 表示全挤在一个类别或没有可分散的余地。

语义维度（输入有效率、平均 S_input）由 Phase 3 的 diff_cov_sem 真正接进来：runner 按
全候选统计算好 input_valid_rate 与 mean_s_input，enrich_metrics 直接沿用。diff / diff_cov
没算语义（s_input 占位 1.0），回落到 semantic_metrics 的缺陷统计汇总，数值先当占位看。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import torch

from mfuzz.core.types import FuzzReport
from mfuzz.neurons.cluster import ClusterResult

# 五维指标各自包含的 result.json 指标键，供 report / compare 分组展示。
FIVE_DIMENSIONS: dict[str, list[str]] = {
    "缺陷": ["n_defects", "rft", "fre"],
    "覆盖": [
        "cncov_0",
        "cncov_final",
        "cncov_gain",
        "cccov_mean_0",
        "cccov_mean_final",
        "cccov_mean_gain",
        "n_uncovered_final",
        "mean_cov_grad_norm",
    ],
    "语义": ["input_valid_rate", "mean_s_input"],
    "多样性": ["n_classes", "oi", "n_clusters", "silhouette"],
    "效率": ["elapsed_time", "defects_per_sec", "n_fuzzed"],
}


@dataclass
class DiversityMetrics:
    n_classes: int  # 缺陷覆盖的原始类别数
    n_target_classes: int  # 缺陷偏离到的目标类别数
    n_class_pairs: int  # 不同 (原始, 目标) 类别对数
    oi: float  # 类别覆盖均衡度
    n_clusters: int
    silhouette: float


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return h


def output_impartiality(source_labels: list[int], n_opportunity_classes: int) -> float:
    """缺陷按原始类别分布的归一化熵。n_opportunity_classes 是可能出缺陷的类别总数。"""
    if not source_labels:
        return 0.0
    counts = list(Counter(source_labels).values())
    denom_classes = max(n_opportunity_classes, len(counts))
    if denom_classes <= 1:
        return 0.0
    return _entropy(counts) / math.log(denom_classes)


def diversity_metrics(
    report: FuzzReport,
    cluster_result: ClusterResult,
    n_consensus_classes: int,
) -> DiversityMetrics:
    src = [d.source_label for d in report.defects]
    tgt = [d.target_label for d in report.defects]
    pairs = {(d.source_label, d.target_label) for d in report.defects}
    return DiversityMetrics(
        n_classes=len(set(src)),
        n_target_classes=len(set(tgt)),
        n_class_pairs=len(pairs),
        oi=output_impartiality(src, n_consensus_classes),
        n_clusters=cluster_result.n_clusters,
        silhouette=cluster_result.silhouette,
    )


def semantic_metrics(report: FuzzReport, gamma_input: float) -> dict[str, float]:
    """语义维度的缺陷统计汇总，供 diff / diff_cov 回落。diff_cov_sem 用 runner 的全候选统计。"""
    s = [d.s_input for d in report.defects]
    if not s:
        return {"input_valid_rate": 0.0, "mean_s_input": 0.0}
    valid = sum(1 for v in s if v >= gamma_input)
    return {"input_valid_rate": valid / len(s), "mean_s_input": sum(s) / len(s)}


def cccov_round_stats(
    cccov_history: list[dict[int, float]] | list[dict[str, float]],
) -> tuple[list[float], list[float], list[float]]:
    """逐轮把各类 CCCov 汇成三条序列：跨类均值、最小、最大。

    CCCov 是逐轮逐类的二维量，热力图保留类别这一维看冷热不均。这里另给一个一维概括，
    供折线图把"类关键覆盖整体涨到哪、类间差多大"和 CNCov 画在一起。某轮没有任何类别
    （那一轮没出现对应种子）时三者都记 NaN，画图自然留缺口。键是 int 还是 str 都行，只取值。
    """
    means: list[float] = []
    los: list[float] = []
    his: list[float] = []
    for d in cccov_history:
        vals = list(d.values())
        if vals:
            means.append(sum(vals) / len(vals))
            los.append(min(vals))
            his.append(max(vals))
        else:
            means.append(math.nan)
            los.append(math.nan)
            his.append(math.nan)
    return means, los, his


def cccov_scalars(
    cccov_history: list[dict[int, float]] | list[dict[str, float]],
) -> dict[str, float]:
    """跨类 CCCov 均值的首轮、末轮与增益，三个可比标量。

    CNCov 有 cncov_0/final/gain 作横向比较，CCCov 类别多、平时按类画热力图，缺一个能进
    汇总表的标量。取每轮跨类均值后报首轮、末轮、增益，与 CNCov 那三个对齐。无历史返回空。
    """
    means, _, _ = cccov_round_stats(cccov_history)
    means = [m for m in means if not math.isnan(m)]
    if not means:
        return {}
    c0, cf = means[0], means[-1]
    return {"cccov_mean_0": c0, "cccov_mean_final": cf, "cccov_mean_gain": cf - c0}


def fault_rate_per_effort(n_defects: float, n_fuzzed: float, pgd_steps: int) -> float:
    """单位代价缺陷数：每一步像素变异换来的缺陷数。"""
    cost = n_fuzzed * pgd_steps
    return n_defects / cost if cost > 0 else 0.0


def activation_anomalies(
    report: FuzzReport, *, lo: float = 0.0, hi: float = 1.0
) -> dict[str, float]:
    """统计缺陷指纹 ĉ 的越界情况，作为评估输出。

    ĉ 是各关键神经元按 profiling 区间 min-max 归一化后的激活，正常落在 [0,1]。ĉ>1 说明
    对抗把神经元推过 profiling 上界，ĉ<0 推过下界，两头都是越界信号——回答"是不是有大量
    神经元被推出训练范围"。报越界神经元数、受影响缺陷数与 ĉ 的极值。profiling 上区间恒定
    的神经元会除出极大 ĉ、也落进这个计数，不单独伺候（详见实验报告的恒定神经元说明）。
    """
    vecs = [
        d.critical_activation.detach().cpu().float()
        for d in report.defects
        if d.critical_activation is not None
    ]
    if not vecs:
        return {
            "max_activation": 0.0,
            "min_activation": 0.0,
            "n_ood_neurons": 0.0,
            "n_defects_ood": 0.0,
        }
    mat = torch.stack(vecs)  # (M, K)，ĉ
    ood = (mat > hi) | (mat < lo)
    return {
        "max_activation": float(mat.max()),
        "min_activation": float(mat.min()),
        "n_ood_neurons": float(int(ood.any(dim=0).sum())),
        "n_defects_ood": float(int(ood.any(dim=1).sum())),
    }


def enrich_metrics(
    report: FuzzReport,
    cluster_result: ClusterResult,
    *,
    pgd_steps: int,
    n_consensus_classes: int,
    gamma_input: float,
) -> dict[str, float]:
    """算出 runner 没算的指标，返回要并进 report.metrics 的键值。"""
    m = report.metrics
    div = diversity_metrics(report, cluster_result, n_consensus_classes)
    sem = semantic_metrics(report, gamma_input)
    extra: dict[str, float] = {
        "fre": fault_rate_per_effort(
            m.get("n_defects", float(report.num_defects)),
            m.get("n_fuzzed", 0.0),
            pgd_steps,
        ),
        "n_classes": float(div.n_classes),
        "n_target_classes": float(div.n_target_classes),
        "n_class_pairs": float(div.n_class_pairs),
        "oi": div.oi,
        "n_clusters": float(div.n_clusters),
        "silhouette": div.silhouette,
        # diff_cov_sem 的 runner 已按全候选统计算好 input_valid_rate / mean_s_input，沿用；
        # diff / diff_cov 没算（s_input 占位 1.0），回落到缺陷统计的 semantic_metrics。
        "input_valid_rate": m.get("input_valid_rate", sem["input_valid_rate"]),
        "mean_s_input": m.get("mean_s_input", sem["mean_s_input"]),
    }
    extra.update(activation_anomalies(report))
    extra.update(cccov_scalars(report.cccov_history))
    return extra
