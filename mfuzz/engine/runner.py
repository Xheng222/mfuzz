"""迭代主循环。

run_differential 是 Phase 1 的纯差分驱动：共识过滤得到种子后，对每个种子在
像素空间做投影梯度上升，沿 obj_1 增大的方向变异，再做四类分流。

run_diff_cov 是 Phase 2 的差分加覆盖驱动：先对目标模型做关键神经元 profiling
得 D_en 与 CNCov_0，再在差分目标外并入覆盖目标 obj_cov。两个目标的梯度各自按
L2 范数归一化后加权合并，避免量纲不一致时覆盖项被差分项淹没（旧实现直接相加
梯度是覆盖引导失效的原因之一）。Phase 4 会把这套合并逻辑搬进 optimize/。
"""

from __future__ import annotations

import time

import torch
from loguru import logger
from torch import Tensor

from mfuzz.core.datasets import (
    build_dataset,
    imagenet_denormalize,
    imagenet_normalize,
    make_loader,
)
from mfuzz.core.hooks import ActivationExtractor
from mfuzz.core.models import load_ensemble
from mfuzz.core.seed import build_seed_pool
from mfuzz.core.types import Config, DefectRecord, FuzzReport
from mfuzz.differential.consensus import filter_consensus
from mfuzz.differential.ensemble import Ensemble
from mfuzz.differential.objective import differential_objective
from mfuzz.differential.triage import Verdict, triage
from mfuzz.neurons.coverage import CoverageTracker
from mfuzz.neurons.objective import coverage_objective, select_u
from mfuzz.neurons.profiler import build_profile, flatten_acts

_EPS = 1e-12


def _l2_normalize(grad: Tensor) -> Tensor:
    """按样本把梯度除以自身 L2 范数，统一各目标梯度量纲。"""
    norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return grad / (norm + _EPS)


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

    bundle = build_dataset(config.dataset.name, config.dataset.val_fraction, config.random_seed)
    raw_seeds = build_seed_pool(
        bundle.seed_set, config.dataset.seed_size, device, config.random_seed
    )
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


def run_diff_cov(config: Config, device: torch.device) -> FuzzReport:
    """差分加关键神经元覆盖驱动。λ2 控制覆盖目标权重，λ2=0 退化为带覆盖跟踪的纯差分。"""
    torch.manual_seed(config.random_seed)
    names = config.models.names
    target = names[config.models.target_idx]
    ensemble = Ensemble(load_ensemble(names, device), target)
    target_model = ensemble.models[target]
    ref_models = {n: ensemble.models[n] for n in ensemble.references}
    logger.info(f"目标模型 {target}，参考模型 {ensemble.references}")

    bundle = build_dataset(config.dataset.name, config.dataset.val_fraction, config.random_seed)
    raw_seeds = build_seed_pool(
        bundle.seed_set, config.dataset.seed_size, device, config.random_seed
    )
    cons = filter_consensus(ensemble, raw_seeds, batch_size=config.dataset.batch_size)
    logger.info(f"共识过滤：{cons.accepted}/{cons.total} 通过（接受率 {cons.acceptance_rate:.3f}）")
    seeds = cons.seeds

    # 关键神经元 profiling：全局频率走完整 profiling 集，每个共识类的频率与归因走
    # 该类自己的 profiling 数据（研究方案 T -> T_c），只对种子里出现的共识类别算。
    # 共识标签是模型一致预测，可能落在 profiling 类别之外（模型一致误判），这类
    # 没有训练数据、给不出 D_en^c，跳过即可——对应种子仍可变异，覆盖目标退回全局 D_en。
    consensus_classes = sorted({s.consensus_label for s in seeds})
    profile_classes = [c for c in consensus_classes if c in bundle.class_to_indices]
    skipped = len(consensus_classes) - len(profile_classes)
    if skipped:
        logger.info(f"共识类别 {len(consensus_classes)} 个，{skipped} 个无 profiling 数据已跳过")
    bs_prof = config.dataset.batch_size
    profile_loader = make_loader(bundle.profile_set, batch_size=bs_prof, shuffle=False)
    class_loaders = {
        c: make_loader(bundle.class_subset(c), batch_size=bs_prof, shuffle=False)
        for c in profile_classes
    }
    profile = build_profile(
        target_model,
        target,
        profile_loader,
        class_loaders,
        t=config.neurons.activation_threshold,
        tau=config.neurons.critical_threshold,
        tau_class=config.neurons.class_critical_threshold,
        alpha=config.neurons.alpha,
        dataset_name=config.dataset.name,
        val_fraction=config.dataset.val_fraction,
        cache_dir=config.neurons.cache_dir,
        device=device,
    )
    pct = profile.cl_percentiles()
    logger.info(
        f"关键度分位 p50={pct[0.5]:.3f} p75={pct[0.75]:.3f} p90={pct[0.9]:.3f}；"
        f"关键占比 {profile.critical_ratio:.3f}（共 {profile.num_critical}/{profile.num_neurons}）"
    )

    extractor = ActivationExtractor(target_model)
    layers = profile.layers
    tracker = CoverageTracker(profile, device)
    den_idx = tracker.den_idx
    scale_den = tracker.scale_den

    # CNCov_0：初始种子覆盖率。
    with torch.no_grad():
        for start in range(0, len(seeds), config.dataset.batch_size):
            chunk = seeds[start : start + config.dataset.batch_size]
            xb = torch.stack([s.image for s in chunk]).to(device)
            tracker.update(extractor.extract(xb))
    cncov0 = tracker.cncov
    report = FuzzReport()
    report.cncov_history.append(cncov0)
    report.cccov_history.append(tracker.cccov())
    logger.info(f"CNCov_0 = {cncov0:.3f}（未覆盖关键神经元 {tracker.num_uncovered}）")

    steps = config.fuzz.pgd_steps
    bs = config.fuzz.batch_size
    lam1 = config.differential.lambda1
    lam2 = config.fuzz.lambda2
    u_size = config.neurons.u_size
    eps = config.fuzz.epsilon
    step_size = config.fuzz.step_size
    t0 = time.perf_counter()

    n_fuzzed = 0
    refs_hold = 0
    conf_drop_sum = 0.0
    pert_sum = 0.0
    cov_grad_norm_sum = 0.0
    cov_grad_norm_n = 0

    for start in range(0, len(seeds), bs):
        chunk = seeds[start : start + bs]
        x_norm0 = torch.stack([s.image for s in chunk]).to(device)
        x0_pixel = imagenet_denormalize(x_norm0)
        c = torch.tensor([s.consensus_label for s in chunk], device=device)
        classes = [int(v) for v in c]
        index = c.view(-1, 1)

        with torch.no_grad():
            orig_conf = ensemble.probs(x_norm0)[target].gather(1, index).squeeze(1)
            acts0 = extractor.extract(x_norm0)
            crit0 = flatten_acts(acts0, layers)[:, den_idx] / scale_den  # (B, K)
        mask_u = select_u(tracker, crit0, classes, u_size)  # 轮内固定

        x = x0_pixel.clone()
        for _ in range(steps):
            x = x.detach().requires_grad_(True)
            xn = imagenet_normalize(x)
            target_out, acts = extractor.forward_with_acts(xn)
            probs = {target: torch.softmax(target_out, dim=1)}
            for name, m in ref_models.items():
                probs[name] = torch.softmax(m(xn), dim=1)

            obj1 = differential_objective(probs, target, c, lam1).sum()
            crit_norm = flatten_acts(acts, layers)[:, den_idx] / scale_den  # (B, K) 带图
            objcov = coverage_objective(crit_norm, mask_u)

            (g1,) = torch.autograd.grad(obj1, x, retain_graph=True)
            (gc,) = torch.autograd.grad(objcov, x)
            cov_grad_norm_sum += float(gc.flatten(1).norm(dim=1).mean())
            cov_grad_norm_n += 1

            combined = _l2_normalize(g1) + lam2 * _l2_normalize(gc)
            with torch.no_grad():
                x = x + step_size * combined.sign()
                x = torch.clamp(x, x0_pixel - eps, x0_pixel + eps)
                x = torch.clamp(x, 0.0, 1.0)
        x_adv = x.detach()

        with torch.no_grad():
            adv_acts = extractor.extract(imagenet_normalize(x_adv))
            tracker.update(adv_acts)
            adv_crit = flatten_acts(adv_acts, layers)[:, den_idx] / scale_den  # (B, K)
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
                        critical_activation=adv_crit[i].detach().cpu(),
                    )
                )

        report.cncov_history.append(tracker.cncov)
        report.cccov_history.append(tracker.cccov())

    elapsed = time.perf_counter() - t0
    report.total_iterations = steps
    report.elapsed_time = elapsed
    mean_cov_grad = cov_grad_norm_sum / cov_grad_norm_n if cov_grad_norm_n else 0.0
    report.metrics = {
        "seed_acceptance_rate": cons.acceptance_rate,
        "n_seeds_accepted": float(cons.accepted),
        "n_fuzzed": float(n_fuzzed),
        "n_defects": float(report.num_defects),
        "rft": report.num_defects / n_fuzzed if n_fuzzed else 0.0,
        "ref_consensus_hold_rate": refs_hold / n_fuzzed if n_fuzzed else 0.0,
        "mean_target_conf_drop": conf_drop_sum / n_fuzzed if n_fuzzed else 0.0,
        "mean_perturbation_linf": pert_sum / n_fuzzed if n_fuzzed else 0.0,
        "n_neurons": float(profile.num_neurons),
        "n_critical": float(profile.num_critical),
        "critical_ratio": profile.critical_ratio,
        "cncov_0": cncov0,
        "cncov_final": tracker.cncov,
        "cncov_gain": tracker.cncov - cncov0,
        "n_uncovered_final": float(tracker.num_uncovered),
        "mean_cov_grad_norm": mean_cov_grad,
        "lambda2": lam2,
        "defects_per_sec": report.num_defects / elapsed if elapsed else 0.0,
    }
    report.curves = {"cncov": report.cncov_history}
    logger.info(
        f"diff+cov 完成：缺陷 {report.num_defects}，RFT {report.metrics['rft']:.3f}，"
        f"CNCov {cncov0:.3f}->{tracker.cncov:.3f}，"
        f"覆盖梯度均范 {mean_cov_grad:.4g}，耗时 {elapsed:.1f}s"
    )
    return report


def run_fuzz(config: Config, device: torch.device) -> FuzzReport:
    """按 config.run.mode 选择运行模式。"""
    mode = config.run.mode
    if mode == "diff":
        return run_differential(config, device)
    if mode == "diff_cov":
        return run_diff_cov(config, device)
    raise ValueError(f"模式 {mode!r} 尚未实现")
