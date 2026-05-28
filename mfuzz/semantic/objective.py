"""语义偏移目标 obj_sem（研究方案 3.2）。

    obj_sem(x) = 1 - S_input(x, x0)

并进联合目标 obj_total = obj_1 + λ2·obj_cov - λ3·obj_sem。负号表示压低语义偏移：
联合目标按上升方向走时，-λ3·obj_sem 这一项把变异往保持 S_input 高（语义不漂）的方向
拉。它是连续、近似的梯度惩罚，配合 triage 里 S_input ≥ γ_input 的离散后验过滤，
两层一起用，λ3 不必设得过大。

与 differential_objective 同风格：返回逐样本目标值 (B,)，由 runner 求和后一次反传。
"""

from __future__ import annotations

from torch import Tensor

from mfuzz.semantic.feature import s_input


def semantic_objective(v_x: Tensor, v_x0: Tensor) -> Tensor:
    """逐样本 obj_sem = 1 - S_input。v_x、v_x0 形状 (B, C)，返回 (B,)。

    v_x 带计算图（来自覆盖前向的中间层激活），v_x0 是原始种子特征的固定参考。
    """
    return 1.0 - s_input(v_x, v_x0)
