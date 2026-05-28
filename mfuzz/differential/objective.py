"""差分目标函数 obj_1。

设网络集合 F = {F_1, ..., F_n}，目标模型 F_j，共识标签 c：

    obj_1(x) = Σ_{k≠j} F_k(x)[c] - F_j(x)[c]

前半维持参考模型对 c 的置信度，后半降低目标模型对 c 的置信度。函数值增大
即目标模型偏离共识而参考模型维持。这里只算目标值（逐样本），梯度由调用方
对输入做 autograd，因为梯度依赖于具体的前向计算图。

obj_1 是联合目标的锚：其梯度在 combine_gradients 里先被 L2 归一化，任何正标量
权重（旧 λ1）都会被归一化抹掉、不影响合成方向，故不再设此权重。
"""

from __future__ import annotations

from torch import Tensor


def differential_objective(probs: dict[str, Tensor], target: str, c: Tensor) -> Tensor:
    """逐样本 obj_1。

    probs: 模型名 -> (B, num_classes) softmax 概率。
    c: (B,) 各样本共识标签。
    返回: (B,) 逐样本目标值。
    """
    if target not in probs:
        raise ValueError(f"目标模型 {target!r} 不在 probs 中")
    index = c.view(-1, 1)  # (B, 1)
    target_term = probs[target].gather(1, index).squeeze(1)  # (B,)
    ref_term = None
    for name, p in probs.items():
        if name == target:
            continue
        term = p.gather(1, index).squeeze(1)
        ref_term = term if ref_term is None else ref_term + term
    if ref_term is None:
        raise ValueError("差分目标至少需要一个参考模型")
    return ref_term - target_term
