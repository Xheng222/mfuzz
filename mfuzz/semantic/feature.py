"""输入语义相似度 S_input（研究方案 3.1）。

    S_input(x, x0) = cos(v(x), v(x0))

v(x) 取目标模型一个预选中间层的特征向量。中间层特征比像素更接近模型实际使用的
内部表示，对光照、几何这类无关变化更稳。低于阈值 γ_input 判定变异样本已偏离原始
输入的有效范围，触发后验过滤（见 differential/triage）。

预选层取「倒数第二个被发现的层」layers[-2]：ActivationExtractor 按定义顺序发现
Conv2d 与 Linear，最后一个总是分类头 logits，倒数第二个就是分类头前的特征嵌入。
三个注册模型都成立——resnet50 末层卷积池化 2048 维、vgg16_bn 的 classifier 中间
4096 维、mobilenet_v2 末层 1x1 卷积池化 1280 维。这层已在覆盖前向的 acts 字典里带
梯度，obj_sem 因此零额外前向。研究方案只说「预选中间层」，这里给出统一的具体落地。
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def feature_layer(layers: list[str]) -> str:
    """预选特征层名：倒数第二个被发现的层（分类头 logits 前的特征嵌入）。"""
    if len(layers) < 2:
        raise ValueError(f"层数 {len(layers)} 不足，至少需要特征层与分类头两层")
    return layers[-2]


def s_input(v_x: Tensor, v_x0: Tensor) -> Tensor:
    """逐样本输入语义相似度 cos(v(x), v(x0))。

    v_x、v_x0 形状 (B, C)，返回 (B,)。v_x0 应为原始种子的特征，固定参考，调用方
    传入前已 detach。值落在 [-1, 1]，正常变异下接近 1。
    """
    return F.cosine_similarity(v_x, v_x0, dim=1)
