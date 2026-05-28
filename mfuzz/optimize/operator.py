"""投影梯度上升算子（实现方案 4.4）。

    x_{t+1} = Proj_C( x_t + η · sign(∇_x obj_total) )

取梯度符号，每像素以统一步长 η 移动，再裁剪回可行域 C：先做 L∞ 约束
‖x − x_0‖_∞ ≤ ε，再裁回像素范围 [0, 1]。
"""

from __future__ import annotations

import torch
from torch import Tensor


def pgd_step(
    x: Tensor,
    x0_pixel: Tensor,
    grad: Tensor,
    step_size: float,
    epsilon: float,
) -> Tensor:
    """投影梯度上升一步，返回 detach 后的新像素图。

    x: 当前像素图。x0_pixel: 原始种子像素图，L∞ 投影的中心。grad: 联合目标对 x 的梯度。
    """
    with torch.no_grad():
        x = x + step_size * grad.sign()
        x = torch.clamp(x, x0_pixel - epsilon, x0_pixel + epsilon)  # L∞ 投影
        x = torch.clamp(x, 0.0, 1.0)  # 像素范围
    return x.detach()
