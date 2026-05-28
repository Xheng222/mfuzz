"""路径相似度 S_path（研究方案 3.1）。

    S_path(x, x0) = cos(u(x), u(x0))

u(x) 取变异样本在关键神经元集合 D_en 上的激活向量，与缺陷分类用的激活指纹共享同一
表示（见 neurons 一系）。S_path 衡量变异样本是否走了与原始样本不同的内部决策路径，
越低越有价值：低于阈值 θ_path 说明路径足够新颖，该优先保留。

S_path 不进梯度公式。它衡量的是候选样本与已有测试之间的整体多样性，作用于每轮结束
的去向评估与种子调度，而非单步变异的方向约束（研究方案 3.2）。把它放进梯度会让内层
优化同时担两个职责。本模块只负责计算，调度的使用在 optimize（Phase 4）。
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def s_path(u_x: Tensor, u_x0: Tensor) -> Tensor:
    """逐样本路径相似度 cos(u(x), u(x0))。

    u_x、u_x0 形状 (B, K)，K 是关键神经元数，返回 (B,)。u_x0 是原始种子在 D_en 上的
    激活，固定参考。值越低路径越新颖。
    """
    return F.cosine_similarity(u_x, u_x0, dim=1)
