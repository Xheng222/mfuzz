"""缺陷激活向量聚类（研究内容 2 的缺陷分类部分）。

每个候选缺陷在关键神经元集合 D_en 上的激活向量是它的内部路径指纹，Phase 2 的
runner 已把它存进 DefectRecord.critical_activation。这里对这些向量聚类：簇数反映
缺陷多样性，结合各簇的原始类别与错误类别描述每类缺陷模式。

按研究方案 2.4：缺陷少用层次聚类，缺陷多按轮廓系数自动选簇数。实现走凝聚层次聚类，
按轮廓系数在候选簇数里挑最好的一个，距离用余弦。

距离选余弦有两个理由。一是与项目自身一致：实现方案 4.3 把路径相似度 $S_{path}$ 定义为
关键神经元激活向量的余弦，缺陷指纹与它共享同一表示，聚类这些指纹自然也该用余弦。二是
鲁棒：激活向量按各神经元自身区间 min-max 归一化，个别神经元仍可能因区间近恒定而出现
尺度异常，欧氏距离会被这种点带偏，余弦对每个样本整体归一、天然不怕。不再做 StandardScaler
——向量已是 min-max 归一化的可比量，再标准化等于把近乎不变的噪声神经元抬到和有区分力的
神经元同等权重，实测压低分离度。凝聚聚类确定性，结果可复现。聚类是多样性指标，不是根因
诊断，同簇不代表同根因。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from mfuzz.core.types import DefectRecord


@dataclass
class ClusterSummary:
    """单个缺陷簇的概况。"""

    cluster_id: int
    size: int
    source_labels: dict[int, int]  # 原始共识类别 -> 该簇内样本数
    target_labels: dict[int, int]  # 偏离后的目标类别 -> 样本数
    dominant_pair: tuple[int, int]  # 簇内最常见的 (原始类别, 错误类别)
    dominant_pair_count: int


@dataclass
class ClusterResult:
    """一次缺陷聚类的结果。embedding 是 2D 投影，仅供散点可视化。"""

    n_defects: int  # 参与聚类的缺陷数（带激活向量的）
    n_clusters: int
    labels: list[int]  # 每个缺陷的簇号，与参与聚类的缺陷同序
    silhouette: float  # 轮廓系数，[-1,1]，越大簇越分得开；无法定义时记 0
    method: str
    n_excluded_neurons: int = 0  # 聚类前排除的疑似死神经元维数（爆值，归一化产物）
    embedding: list[list[float]] = field(default_factory=list)  # 每缺陷一个 2D 坐标
    clusters: list[ClusterSummary] = field(default_factory=list)


def _stack_vectors(defects: list[DefectRecord]) -> tuple[np.ndarray, list[int]]:
    """取出带激活向量的缺陷，堆成 (M, K) 矩阵，返回矩阵与它们在原列表中的下标。"""
    vecs: list[np.ndarray] = []
    keep: list[int] = []
    for i, d in enumerate(defects):
        if d.critical_activation is None:
            continue
        vecs.append(d.critical_activation.detach().cpu().float().numpy())
        keep.append(i)
    if not vecs:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(vecs).astype(np.float32), keep


def _summaries(
    defects: list[DefectRecord], keep: list[int], labels: np.ndarray
) -> list[ClusterSummary]:
    out: list[ClusterSummary] = []
    for cid in sorted(set(int(v) for v in labels)):
        members = [defects[keep[j]] for j in range(len(keep)) if int(labels[j]) == cid]
        src = Counter(d.source_label for d in members)
        tgt = Counter(d.target_label for d in members)
        pair = Counter((d.source_label, d.target_label) for d in members)
        (dom_pair, dom_count) = pair.most_common(1)[0]
        out.append(
            ClusterSummary(
                cluster_id=cid,
                size=len(members),
                source_labels=dict(src),
                target_labels=dict(tgt),
                dominant_pair=dom_pair,
                dominant_pair_count=dom_count,
            )
        )
    return out


def _embed_2d(vectors: np.ndarray) -> list[list[float]]:
    """L2 归一化后投到 2D 画散点，使投影几何与余弦聚类一致。"""
    m, dim = vectors.shape
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.where(norms == 0.0, 1.0, norms)
    if dim >= 2 and m >= 2:
        coords = PCA(n_components=2, random_state=0).fit_transform(normed)
    elif dim == 1:
        coords = np.concatenate([normed, np.zeros((m, 1), dtype=normed.dtype)], axis=1)
    else:
        coords = np.zeros((m, 2), dtype=np.float32)
    return coords.astype(float).tolist()


def _single(defects: list[DefectRecord], keep: list[int], m: int, method: str) -> ClusterResult:
    labels = np.zeros(m, dtype=int)
    vectors, _ = _stack_vectors(defects)
    return ClusterResult(
        n_defects=m,
        n_clusters=1,
        labels=labels.tolist(),
        silhouette=0.0,
        method=method,
        embedding=_embed_2d(vectors) if m else [],
        clusters=_summaries(defects, keep, labels) if m else [],
    )


def cluster_defects(
    defects: list[DefectRecord],
    *,
    max_k: int = 10,
    min_defects: int = 4,
    extreme: float = 1e3,
) -> ClusterResult:
    """对缺陷激活向量聚类，自动选簇数。

    缺陷不足 min_defects 时不强行分簇，按整体当 1 簇返回（仍给出类别统计）。否则在
    [2, min(max_k, M-1)] 范围内逐个簇数跑余弦距离的凝聚聚类，取轮廓系数最高者。不做
    PCA 降维：余弦在原始高维上直接算，几百到几千个缺陷的成对余弦开销可接受，且实测降维
    会重排到高方差方向、扭曲余弦结构、压低分离度。

    聚类前保留一条排除"爆值维"的兜底：任一缺陷上 |归一化激活| 超过 extreme（默认 1e3）的维
    就剔除。它本是针对早期"除以正向峰值"归一化的——那种归一化会让正向峰值近零的神经元爆到
    $10^9$、用符号牵着余弦二分走。换成区间 min-max 归一化后 ĉ 有界（实测最大约 1.7 到 3.1），
    无维触发这条、当前排除 0 维，逻辑留作兜底。被排除维数记进结果，由 metrics 的越界统计一并报出。
    """
    vectors, keep = _stack_vectors(defects)
    m = vectors.shape[0]
    if m == 0:
        return ClusterResult(n_defects=0, n_clusters=0, labels=[], silhouette=0.0, method="none")
    if m < min_defects:
        return _single(defects, keep, m, "single (too few defects)")

    valid = (np.abs(vectors) <= extreme).all(axis=0)
    n_excluded = int((~valid).sum())
    vclean = vectors[:, valid] if n_excluded else vectors

    # 排除后无有效维、全零或方差塌缩时余弦无意义，按 1 簇处理。
    norms = np.linalg.norm(vclean, axis=1)
    if vclean.shape[1] == 0 or np.all(norms == 0.0) or float(np.var(vclean)) < 1e-12:
        return _single(defects, keep, m, "single (degenerate)")

    best_labels: np.ndarray | None = None
    best_score = -2.0
    best_k = 1
    upper = int(min(max_k, m - 1))
    for k in range(2, upper + 1):
        labels_k = AgglomerativeClustering(
            n_clusters=k, linkage="average", metric="cosine"
        ).fit_predict(vclean)
        if len(set(labels_k.tolist())) < 2:
            continue
        score = float(silhouette_score(vclean, labels_k, metric="cosine"))
        if score > best_score:
            best_score, best_labels, best_k = score, labels_k, k

    if best_labels is None:  # 所有候选簇数都退化，落回单簇
        return _single(defects, keep, m, "single (no valid k)")

    logger.info(
        f"缺陷聚类：{m} 个缺陷分为 {best_k} 簇，余弦轮廓系数 {best_score:.3f}"
        f"（排除 {n_excluded} 个疑似死神经元维）"
    )
    return ClusterResult(
        n_defects=m,
        n_clusters=best_k,
        labels=best_labels.tolist(),
        silhouette=best_score,
        method="agglomerative-average (cosine, silhouette-selected)",
        n_excluded_neurons=n_excluded,
        embedding=_embed_2d(vclean),
        clusters=_summaries(defects, keep, best_labels),
    )
