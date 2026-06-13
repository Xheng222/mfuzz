"""定位引导的修复探针：在归因指认的责任层里抑制特定通道，验证定位的因果性。

整个项目做的是缺陷检测，这一步不是为了调出更强的模型，而是反过来证明归因
真的指对了地方。做法是训练无关的通道编辑：把某类失效在责任层里的责任通道按
强度缩放，重跑差分判定，看那一类失效是不是真的随之减少。纯抑制没有微调那种
"网络自己学会绕开"的后门，唯一变的就是我们碰的那几个通道，所以失效下降只能
归到这个位置，这是比层级置零更细的一路干预证据。

代价是另一根轴。抑制通道很可能连带损失正确检测，所以重判时一并统计 agree
(与共识一致的正确检测) 的变化和其它失效类的变化。收益是目标失效类下降，代价是
agree 下降，两者一起刻画出一条代价收益曲线。责任通道这条曲线若压住随机通道的
对照，就说明起作用的是"定位准"，不是"随便抑制都有效"。

本模块只放机制，编排在 scripts/run_repair_pilot.py。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import torch
from torch import Tensor

from mfuzz.core.det_models import AnyDetector
from mfuzz.core.types import Detection
from mfuzz.differential.det_oracle import DetRecord, judge_image
from mfuzz.neurons.struct_attr import GraphResult, _make_target


def layer_channel_attr(
    g: GraphResult,
    rec: DetRecord,
    idx: int,
    layer: str,
    device: torch.device,
) -> Tensor | None:
    """单个失效实例在某层的逐通道归因：对该层各次调用做 |grad×act|，对空间求和、
    保留通道维，跨调用累加。返回 (C,) 张量；该层未参与目标计算图时返回 None。

    与 struct_attr.attribute 同一套归因目标（loc 用 IoU，其余用分数），只是这里
    保留通道维以便挑责任通道。
    """
    target = _make_target(rec, idx, g, device)
    ts = g.acts.get(layer)
    if not ts:
        return None
    grads = torch.autograd.grad(target, ts, retain_graph=True, allow_unused=True)
    acc: Tensor | None = None
    for t, gr in zip(ts, grads, strict=True):
        if gr is None:
            continue
        v = (gr * t.detach()).abs().sum(dim=(0, 2, 3))  # (C,)
        acc = v if acc is None else acc + v
    return acc


def layer_weight_grad(
    g: GraphResult,
    rec: DetRecord,
    idx: int,
    weight: Tensor,
    device: torch.device,
) -> Tensor | None:
    """单个失效实例对责任层卷积权重的梯度 ∂目标/∂W（loc 用 IoU，其余用分数）。

    权重编辑式修复用：沿 +IoU 方向给权重迈一步，等于做一步"无数据的方向性修复"。
    weight 需在前向前置 requires_grad=True，调用方负责开关。该层未进计算图时返回
    None。
    """
    target = _make_target(rec, idx, g, device)
    grads = torch.autograd.grad(target, weight, retain_graph=True, allow_unused=True)
    return grads[0]


def channel_scale_hook(
    adapter: AnyDetector,
    layer: str,
    channels: list[int],
    alpha: float,
) -> torch.utils.hooks.RemovableHandle:
    """在 layer 的卷积输出上，把 channels 这些通道乘以 alpha 的前向 hook。

    alpha=0 即完全抑制，alpha=1 即不变。hook 直接挂在卷积模块上，detect 与
    forward_graph 都会触发；共享 head 逐层级多次调用时每次都抑制，强度一致。
    """
    module = adapter.conv_layers()[layer]
    idx = torch.as_tensor(sorted({int(c) for c in channels}), dtype=torch.long)

    def hook(_m, _i, out: Tensor) -> Tensor:
        out = out.clone()
        sel = idx.to(out.device)
        out[:, sel] = out[:, sel] * alpha
        return out

    return module.register_forward_hook(hook)


def judge_counts(
    adapter: AnyDetector,
    paths: list,
    base_dets: dict[str, dict[str, list[Detection]]],
    load_image: Callable,
    device: torch.device,
    iou_thr: float,
    loc_thr: float,
    score_thr: float,
) -> dict[str, int]:
    """在当前模型状态下重跑目标模型 detect、按现成的差分判定统计各类失效计数。

    参考模型沿用基线检测，只换目标模型这一路。调用方负责挂/摘通道抑制 hook，
    hook 已挂时这里跑出来就是抑制后的计数；不挂即基线。返回各 RECORD_KIND 的计数。
    """
    counts: dict[str, int] = defaultdict(int)
    for path in paths:
        img = load_image(path, device)
        dets = adapter.detect(img, score_thr)
        dbm = dict(base_dets[str(path)])
        dbm[adapter.name] = dets
        records, _ = judge_image(dbm, iou_thr, loc_thr)
        for rec in records[adapter.name]:
            counts[rec.kind] += 1
    return dict(counts)
