"""变异算子协议。两个实现：PGD（梯度驱动）与腐蚀（COCO-C 风格随机变换，E7）。

算子拿到一个 MutationContext，返回变异后的像素图。PGD 在预算内反复调适配器
的 forward/objective1，联合目标在算子内构造：

    obj_total = obj_1 + λ2 · obj_cov − λ3 · obj_sem

obj_1 来自适配器（任务语义），obj_cov 来自覆盖账本（U 个目标单元的归一化
激活和），obj_sem = 1 − S_input（v_sem 与参考态的余弦）。三路梯度按范数归一
加权合并（optimize/joint），λ=0 的项不算梯度、自动消融。

腐蚀算子不用梯度与联合目标，λ 旋钮对它无效；候选评估（覆盖更新、语义门）
由循环统一做，对算子透明。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from mfuzz.core.adapter import Batch, TaskAdapter
from mfuzz.core.config import OptimizeConfig
from mfuzz.neurons.unit_coverage import UnitCoverageTracker
from mfuzz.optimize.joint import combine_gradients
from mfuzz.optimize.operator import pgd_step
from mfuzz.semantic.objective import semantic_objective


@dataclass
class MutationContext:
    adapter: TaskAdapter
    batch: Batch
    tracker: UnitCoverageTracker
    u_idx: Tensor | None  # 覆盖目标单元；None = 本轮不推覆盖
    v_sem0: Tensor  # (B, C) 参考语义特征，已 detach
    lam2: float
    lam3: float
    opt: OptimizeConfig
    rnd: int
    emit: Callable  # ProbeBus.emit


class Mutator(ABC):
    name: str = "mutator"

    @abstractmethod
    def mutate(self, ctx: MutationContext) -> Tensor:
        """返回变异后的像素图，形状同 batch.x0。"""


class PgdMutator(Mutator):
    """投影梯度上升：联合目标逐步上升，L∞ 投影回 ε 球与像素范围。"""

    name = "pgd"

    def mutate(self, ctx: MutationContext) -> Tensor:
        x0 = ctx.batch.x0
        x = x0.clone()
        need_cov = ctx.lam2 != 0.0 and ctx.u_idx is not None
        need_sem = ctx.lam3 != 0.0
        for step in range(ctx.opt.pgd_steps):
            x_g = x.detach().requires_grad_(True)
            fw = ctx.adapter.forward(x_g, ctx.batch)
            ctx.emit("forward", ctx.rnd, step, ctx.batch, fw)

            obj1 = ctx.adapter.objective1(fw, ctx.batch)
            g1: Tensor | None = None
            if obj1 is not None and obj1.requires_grad:
                (g1,) = torch.autograd.grad(
                    obj1, x_g, retain_graph=(need_cov or need_sem), allow_unused=True
                )
            gc: Tensor | None = None
            if need_cov:
                assert ctx.u_idx is not None
                objcov = ctx.tracker.objective(fw.unit_acts, ctx.u_idx)
                (gc,) = torch.autograd.grad(objcov, x_g, retain_graph=need_sem, allow_unused=True)
            gs: Tensor | None = None
            if need_sem:
                objsem = semantic_objective(fw.v_sem, ctx.v_sem0).sum()
                (gs,) = torch.autograd.grad(objsem, x_g, allow_unused=True)
            del fw
            if g1 is None:
                g1 = torch.zeros_like(x_g)
            combined = combine_gradients(g1, gc, gs, ctx.lam2, ctx.lam3)
            x = pgd_step(x, x0, combined, ctx.opt.step_size, ctx.opt.epsilon)
        return x


# ---- 腐蚀（E7 正向外部验证的诱发手段） ----
# 强度 1-5 的参数表对照 ImageNet-C（[0,1] 像素空间）。亮度为加性偏移，
# 对比度为压向逐通道均值的系数（越小越强）。

_NOISE_SIGMA = (0.08, 0.12, 0.18, 0.26, 0.38)
_BLUR_SIGMA = (1.0, 2.0, 3.0, 4.0, 6.0)
_BRIGHT_DELTA = (0.1, 0.2, 0.3, 0.4, 0.5)
_CONTRAST_FACTOR = (0.4, 0.3, 0.2, 0.1, 0.05)

CORRUPTION_OPS = ("gaussian_noise", "gaussian_blur", "brightness", "contrast")


def corrupt(op: str, img: Tensor, severity: int) -> Tensor:
    """对单张 (C,H,W) [0,1] 图施加一种腐蚀。severity 取 1-5。"""
    s = severity - 1
    if op == "gaussian_noise":
        out = img + _NOISE_SIGMA[s] * torch.randn_like(img)
    elif op == "gaussian_blur":
        from torchvision.transforms.v2.functional import gaussian_blur

        sigma = _BLUR_SIGMA[s]
        k = 2 * math.ceil(3 * sigma) + 1
        out = gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])
    elif op == "brightness":
        sign = 1.0 if torch.rand(()) < 0.5 else -1.0
        out = img + sign * _BRIGHT_DELTA[s]
    elif op == "contrast":
        mean = img.mean(dim=(-2, -1), keepdim=True)
        out = (img - mean) * _CONTRAST_FACTOR[s] + mean
    else:
        raise KeyError(f"未知腐蚀算子 {op!r}（可用：{CORRUPTION_OPS}）")
    return out.clamp(0.0, 1.0)


class CorruptionMutator(Mutator):
    """COCO-C 风格腐蚀：每图随机选一种腐蚀作用在原图上，强度由配置固定。

    与 PGD 同协议、同管线、同裁判（差分 + 真值核验），用作 E7 的换诱发手段
    对照。破坏过度的样本由循环的语义门（S_input ≥ γ）过滤。
    """

    name = "corruption"

    def mutate(self, ctx: MutationContext) -> Tensor:
        x0 = ctx.batch.x0
        ops = ctx.opt.corruption_ops
        sev = ctx.opt.corruption_severity
        out = [
            corrupt(ops[int(torch.randint(len(ops), ()))], x0[i], sev) for i in range(x0.shape[0])
        ]
        return torch.stack(out)


_MUTATORS: dict[str, type[Mutator]] = {"pgd": PgdMutator, "corruption": CorruptionMutator}


def build_mutator(name: str) -> Mutator:
    if name not in _MUTATORS:
        raise KeyError(f"未注册的变异算子：{name}（可用：{sorted(_MUTATORS)}）")
    return _MUTATORS[name]()
