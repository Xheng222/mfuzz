"""覆盖目标 obj_cov 与目标神经元集合 U 的选取。

每轮迭代前从未覆盖的关键神经元里选一组 U，优先选接近激活阈值且与当前种子
类别相关的神经元（离阈值远的梯度信号弱）：

    obj_cov(x) = Σ_{n in U} out(n, x)

U 取一组神经元求和，不是只取第一个未覆盖神经元；后者在覆盖饱和时梯度恒为零，
是旧实现覆盖引导失效的直接原因。U 轮内固定，轮末按新覆盖重选。
"""

from __future__ import annotations

import torch
from torch import Tensor

from mfuzz.neurons.coverage import CoverageTracker

_CLASS_PENALTY = 1e4  # 类无关神经元的打分惩罚，使其仅在类相关候选不足时入选


def select_u(
    tracker: CoverageTracker,
    crit_acts_norm: Tensor,
    classes: list[int],
    u_size: int,
) -> Tensor:
    """为一批样本各选一组目标神经元 U，返回 (B, K) bool 掩码。

    crit_acts_norm: (B, K) 全局关键神经元的归一化激活，已 detach，用于排序。
    classes: 每个样本的共识标签，决定类相关偏好。
    """
    device = crit_acts_norm.device
    b, k = crit_acts_norm.shape
    uncovered = ~tracker.covered_den()  # (K,)

    class_mask = torch.stack(
        [
            tracker.den_in_class.get(c, torch.ones(k, dtype=torch.bool, device=device))
            for c in classes
        ],
        dim=0,
    )  # (B, K)

    near = -(crit_acts_norm - tracker.t).abs()  # 越接近阈值分越高
    tier1 = uncovered.unsqueeze(0) & class_mask  # 未覆盖且类相关
    tier2 = uncovered.unsqueeze(0) & ~class_mask  # 未覆盖但类无关，备选

    score = near.clone()
    score[tier2] -= _CLASS_PENALTY
    score[~(tier1 | tier2)] = float("-inf")  # 已覆盖的排除

    n_take = min(u_size, k)
    top = score.topk(n_take, dim=1)
    mask = torch.zeros(b, k, dtype=torch.bool, device=device)
    valid = top.values > float("-inf")
    mask.scatter_(1, top.indices, valid)
    return mask


def coverage_objective(crit_acts_norm: Tensor, mask_u: Tensor) -> Tensor:
    """obj_cov，对 U 内归一化激活求和（批内再求和供一次反传）。

    crit_acts_norm: (B, K) 带计算图的归一化关键激活。
    mask_u: (B, K) bool，select_u 的输出。
    """
    return (crit_acts_norm * mask_u).sum()
