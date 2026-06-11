"""变异算子协议。现阶段只有 PGD 实现，腐蚀/几何变换留作后续平替。

算子拿到一个 MutationContext，在预算内反复调适配器的 forward/objective1，
返回变异后的像素图。联合目标的构造在算子内完成：

    obj_total = obj_1 + λ2 · obj_cov − λ3 · obj_sem

obj_1 来自适配器（任务语义），obj_cov 来自覆盖账本（U 个目标单元的归一化
激活和），obj_sem = 1 − S_input（v_sem 与参考态的余弦）。三路梯度按范数归一
加权合并（optimize/joint），λ=0 的项不算梯度、自动消融。
"""

from __future__ import annotations

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


_MUTATORS: dict[str, type[Mutator]] = {"pgd": PgdMutator}


def build_mutator(name: str) -> Mutator:
    if name not in _MUTATORS:
        raise KeyError(f"未注册的变异算子：{name}（可用：{sorted(_MUTATORS)}）")
    return _MUTATORS[name]()
