"""联合目标的梯度组装（实现方案 4.4）。

联合目标：

    obj_total(x) = obj_1(x) + λ2 · obj_cov(x) − λ3 · obj_sem(x)

三个目标分别基于概率、激活值、余弦距离，量纲不同。若直接把三个梯度相加，范数大的
分量会主导更新、权重失去意义（旧实现覆盖引导失效的原因之一）。这里把每个目标的梯度
分量先除以自身 L2 范数再加权合并。

λ=0 即该项不进梯度：调用方在 λ2=0 时根本不算 obj_cov 的梯度、传 gc=None，λ3=0 同理。
这正是「某模块 λ=0 自动消融」在梯度侧的落地——覆盖与语义的度量照常算，只是不驱动更新。
"""

from __future__ import annotations

from torch import Tensor

_EPS = 1e-12


def normalize_grad(grad: Tensor) -> Tensor:
    """按样本把梯度除以自身 L2 范数，输出每个样本的梯度范数为 1（零梯度除外）。"""
    norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return grad / (norm + _EPS)


def combine_gradients(
    g1: Tensor,
    gc: Tensor | None,
    gs: Tensor | None,
    lambda2: float,
    lambda3: float,
) -> Tensor:
    """合并 obj_total 的梯度：normalize(g1) + λ2·normalize(gc) − λ3·normalize(gs)。

    g1 是差分项梯度（恒在）。gc 是覆盖项梯度，gs 是语义偏移项梯度，为 None 表示对应
    λ=0、该项未参与（自动消融）。语义项取负号，对应 obj_total 里的 −λ3·obj_sem，把变异
    往保持 S_input 高的方向拉。
    """
    combined = normalize_grad(g1)
    if gc is not None:
        combined = combined + lambda2 * normalize_grad(gc)
    if gs is not None:
        combined = combined - lambda3 * normalize_grad(gs)
    return combined
