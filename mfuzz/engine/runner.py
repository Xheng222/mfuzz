"""迭代主循环。

Phase 1 实现纯差分驱动：共识过滤得到种子后，对每个种子在像素空间做投影
梯度上升，沿 obj_1 增大的方向变异，再对结果做四类分流，收集候选缺陷并统计
五项差分指标。

这里内置的 _pgd_ascent 是最小投影梯度上升，运行在 [0,1] 像素空间、约束
L∞ 扰动。Phase 4 会用 optimize/operator.py 与 joint.py 替换它，把覆盖与
语义目标并入联合目标。
"""

from __future__ import annotations

import time

import torch
from loguru import logger
from torch import Tensor

from mfuzz.core.datasets import imagenet_denormalize, imagenet_normalize, load_imagenet
from mfuzz.core.models import load_ensemble
from mfuzz.core.seed import build_seed_pool
from mfuzz.core.types import Config, DefectRecord, FuzzReport
from mfuzz.differential.consensus import filter_consensus
from mfuzz.differential.ensemble import Ensemble
from mfuzz.differential.objective import differential_objective
from mfuzz.differential.triage import Verdict, triage


def _pgd_ascent(
    ensemble: Ensemble,
    x0_pixel: Tensor,
    c: Tensor,
    target: str,
    steps: int,
    step_size: float,
    epsilon: float,
    lambda1: float,
) -> tuple[Tensor, list[Tensor]]:
    """在像素空间沿 obj_1 增大方向做投影梯度上升。

    返回变异后像素图与每步更新前的目标模型对 c 的置信度（用于诊断置信度下降）。
    """
    x = x0_pixel.clone()
    conf_per_step: list[Tensor] = []
    index = c.view(-1, 1)
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        probs = ensemble.probs(imagenet_normalize(x), with_grad=True)
        conf_per_step.append(probs[target].gather(1, index).squeeze(1).detach())
        obj = differential_objective(probs, target, c, lambda1).sum()
        (grad,) = torch.autograd.grad(obj, x)
        with torch.no_grad():
            x = x + step_size * grad.sign()
            x = torch.clamp(x, x0_pixel - epsilon, x0_pixel + epsilon)  # L∞ 投影
            x = torch.clamp(x, 0.0, 1.0)  # 像素范围
    return x.detach(), conf_per_step


def run_differential(config: Config, device: torch.device) -> FuzzReport:
    torch.manual_seed(config.random_seed)
    names = config.models.names
    target = names[config.models.target_idx]
    ensemble = Ensemble(load_ensemble(names, device), target)
    logger.info(f"目标模型 {target}，参考模型 {ensemble.references}")

    dataset = load_imagenet(config.dataset.seed_split)
    raw_seeds = build_seed_pool(dataset, config.dataset.seed_size, device, config.random_seed)
    cons = filter_consensus(ensemble, raw_seeds, batch_size=config.dataset.batch_size)
    logger.info(
        f"共识过滤：{cons.accepted}/{cons.total} 通过（接受率 {cons.acceptance_rate:.3f}），"
        f"低置信度 {cons.low_confidence}"
    )
    seeds = cons.seeds

    steps = config.fuzz.pgd_steps
    bs = config.fuzz.batch_size
    report = FuzzReport()
    t0 = time.perf_counter()

    n_fuzzed = 0
    refs_hold = 0
    conf_drop_sum = 0.0
    pert_sum = 0.0
    conf_curve_sum = [0.0] * steps
    conf_curve_n = 0

    for start in range(0, len(seeds), bs):
        chunk = seeds[start : start + bs]
        x_norm0 = torch.stack([s.image for s in chunk]).to(device)
        x0_pixel = imagenet_denormalize(x_norm0)
        c = torch.tensor([s.consensus_label for s in chunk], device=device)
        index = c.view(-1, 1)

        with torch.no_grad():
            orig_conf = ensemble.probs(x_norm0)[target].gather(1, index).squeeze(1)

        x_adv, conf_per_step = _pgd_ascent(
            ensemble,
            x0_pixel,
            c,
            target,
            steps,
            config.fuzz.step_size,
            config.fuzz.epsilon,
            config.differential.lambda1,
        )
        for t, ct in enumerate(conf_per_step):
            conf_curve_sum[t] += float(ct.sum())
        conf_curve_n += len(chunk)

        with torch.no_grad():
            final_probs = ensemble.probs(imagenet_normalize(x_adv))
        final_labels = {name: p.argmax(dim=1) for name, p in final_probs.items()}
        final_conf = final_probs[target].gather(1, index).squeeze(1)
        pert = (x_adv - x0_pixel).abs().flatten(1).amax(dim=1)

        for i in range(len(chunk)):
            ci = int(c[i])
            tgt_label = int(final_labels[target][i])
            ref_labels = [int(final_labels[name][i]) for name in ensemble.references]
            verdict = triage(tgt_label, ref_labels, ci, semantic_ok=True)

            n_fuzzed += 1
            if all(r == ci for r in ref_labels):
                refs_hold += 1
            conf_drop_sum += float(orig_conf[i] - final_conf[i])
            pert_sum += float(pert[i])

            if verdict is Verdict.DEFECT:
                report.defects.append(
                    DefectRecord(
                        image=x_adv[i].detach().cpu(),
                        source_label=ci,
                        target_label=tgt_label,
                        target_model=target,
                        s_input=1.0,
                        s_path=1.0,
                        perturbation=float(pert[i]),
                    )
                )

    elapsed = time.perf_counter() - t0
    report.total_iterations = steps
    report.elapsed_time = elapsed
    report.metrics = {
        "seed_acceptance_rate": cons.acceptance_rate,
        "n_seeds_total": float(cons.total),
        "n_seeds_accepted": float(cons.accepted),
        "n_fuzzed": float(n_fuzzed),
        "n_defects": float(report.num_defects),
        "rft": report.num_defects / n_fuzzed if n_fuzzed else 0.0,
        "ref_consensus_hold_rate": refs_hold / n_fuzzed if n_fuzzed else 0.0,
        "mean_target_conf_drop": conf_drop_sum / n_fuzzed if n_fuzzed else 0.0,
        "mean_perturbation_linf": pert_sum / n_fuzzed if n_fuzzed else 0.0,
        "defects_per_sec": report.num_defects / elapsed if elapsed else 0.0,
    }
    report.curves = {
        "target_conf": [s / conf_curve_n for s in conf_curve_sum] if conf_curve_n else []
    }
    logger.info(
        f"纯差分完成：缺陷 {report.num_defects}，RFT {report.metrics['rft']:.3f}，"
        f"参考共识保持率 {report.metrics['ref_consensus_hold_rate']:.3f}，耗时 {elapsed:.1f}s"
    )
    return report
