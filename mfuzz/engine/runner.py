"""迭代主循环（实现方案 4.4）。

单一 λ 驱动主循环把四条线索拢到一起：差分提供候选缺陷信号，关键神经元覆盖引导探索，
语义约束限制偏离，动态反馈逐轮协调权重。不再按模块拆成多个入口——运行模式只是 λ 的
预设：λ2=0 即覆盖不进梯度、λ3=0 即语义不进梯度，自动消融。无论 λ 如何，profiling、覆盖、
S_input、S_path 等度量照常计算并写入结果，λ 只决定是否进梯度、不决定是否被测量。

每轮：调度器从种子池按多维优先级选一批种子 → 联合目标投影梯度上升变异 → 四类分流 →
变异结果回写种子统计 → 动态反馈按四指标调 λ2/λ3。终止条件：CNCov 达预设值、覆盖与缺陷
增长连续多轮双低、达最大轮次、连续多轮输入有效率过低或路径新颖为零（后两条识别退化）。
"""

from __future__ import annotations

import time

import torch
from loguru import logger

from mfuzz.core.datasets import (
    build_dataset,
    imagenet_denormalize,
    imagenet_normalize,
    make_loader,
)
from mfuzz.core.hooks import ActivationExtractor
from mfuzz.core.models import load_ensemble
from mfuzz.core.seed import build_seed_pool
from mfuzz.core.types import Config, DefectRecord, FeedbackState, FuzzReport
from mfuzz.differential.consensus import filter_consensus
from mfuzz.differential.ensemble import Ensemble
from mfuzz.differential.objective import differential_objective
from mfuzz.differential.triage import Verdict, triage
from mfuzz.engine.seed_pool import SeedOutcome, SeedPool
from mfuzz.neurons.coverage import CoverageTracker
from mfuzz.neurons.objective import coverage_objective, select_u
from mfuzz.neurons.profiler import build_profile, flatten_acts, normalize_acts
from mfuzz.optimize.feedback import FeedbackController
from mfuzz.optimize.joint import combine_gradients
from mfuzz.optimize.operator import pgd_step
from mfuzz.semantic.feature import feature_layer, s_input
from mfuzz.semantic.objective import semantic_objective
from mfuzz.semantic.path import s_path

_GROWTH_EPS = 1e-4  # 覆盖几乎不增长的阈，配合缺陷为零判定增长双低
_DEGEN_VALID_RATE = 0.3  # 某轮输入有效率低于此算退化信号


def run_fuzz(config: Config, device: torch.device) -> FuzzReport:
    """单一 λ 驱动迭代主循环。消融靠 config 把 λ 置 0，无 mode 标签。"""
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

    # profiling 总是跑：覆盖与类关键集合恒被测量，不因 λ2=0 而跳过。
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
        p_low=config.neurons.p_low,
        p_high=config.neurons.p_high,
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
    feat = feature_layer(layers)  # S_input 的预选特征层：分类头前的特征嵌入
    tracker = CoverageTracker(profile, device, config.neurons.coverage_threshold)
    den_idx = tracker.den_idx
    low_den = tracker.low_den
    high_den = tracker.high_den
    t_cov = tracker.t_cov

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

    steps = config.optimize.pgd_steps
    bs = config.loop.seeds_per_round
    gamma = config.semantic.gamma_input
    theta = config.semantic.theta_path
    u_size = config.neurons.u_size
    eps = config.optimize.epsilon
    step_size = config.optimize.step_size
    max_rounds = config.loop.max_iterations
    cncov_target = config.loop.cncov_target
    growth_patience = config.loop.growth_patience
    log_interval = max(1, config.loop.log_interval)

    sem_thr = config.feedback.sem_shift_threshold
    if sem_thr <= 0.0:
        sem_thr = 1.0 - gamma  # 自动取 1-γ_input
    feedback = FeedbackController(
        config.feedback,
        config.neurons.lambda2,
        config.semantic.lambda3,
        sem_thr,
        config.neurons.lambda2_bounds,
        config.semantic.lambda3_bounds,
    )
    pool = SeedPool(seeds, config.scheduler.pool_capacity, config.scheduler.retire_patience)

    n_fuzzed = 0
    refs_hold = 0
    conf_drop_sum = 0.0
    pert_sum = 0.0
    cov_grad_norm_sum = 0.0
    cov_grad_norm_n = 0
    cov_grad_history: list[float] = []
    sem_valid = 0
    s_input_sum = 0.0
    defect_path_novel = 0
    low_valid_streak = 0
    zero_novel_streak = 0
    double_low_streak = 0
    rounds_run = 0
    t0 = time.perf_counter()

    for rnd in range(max_rounds):
        lam2 = feedback.lambda2
        lam3 = feedback.lambda3
        need_cov = lam2 != 0.0
        need_sem = lam3 != 0.0
        selected = pool.select(
            bs, tracker.cccov(), config.scheduler, gamma, feedback.coverage_stalled
        )
        if not selected:
            logger.info(f"第 {rnd} 轮：种子池已空（全部退役），终止")
            break
        rounds_run += 1

        x_norm0 = torch.stack([s.image for s in selected]).to(device)
        x0_pixel = imagenet_denormalize(x_norm0)
        c = torch.tensor([s.consensus_label for s in selected], device=device)
        classes = [int(v) for v in c]
        index = c.view(-1, 1)

        with torch.no_grad():
            orig_conf = ensemble.probs(x_norm0)[target].gather(1, index).squeeze(1)
            acts0 = extractor.extract(x_norm0)
            crit0 = normalize_acts(flatten_acts(acts0, layers)[:, den_idx], low_den, high_den)
            v_x0 = acts0[feat].detach()
        mask_u = select_u(tracker, crit0, classes, u_size) if need_cov else None

        x = x0_pixel.clone()
        batch_grad_sum = 0.0
        for _ in range(steps):
            x = x.detach().requires_grad_(True)
            xn = imagenet_normalize(x)
            target_out, acts = extractor.forward_with_acts(xn)
            probs = {target: torch.softmax(target_out, dim=1)}
            for name, m in ref_models.items():
                probs[name] = torch.softmax(m(xn), dim=1)
            obj1 = differential_objective(probs, target, c).sum()

            gc = None
            gs = None
            (g1,) = torch.autograd.grad(obj1, x, retain_graph=(need_cov or need_sem))
            if need_cov:
                assert mask_u is not None  # need_cov 时 mask_u 已选好
                crit_norm = normalize_acts(
                    flatten_acts(acts, layers)[:, den_idx], low_den, high_den
                )
                objcov = coverage_objective(crit_norm, mask_u)
                (gc,) = torch.autograd.grad(objcov, x, retain_graph=need_sem)
                gnorm = float(gc.flatten(1).norm(dim=1).mean())
                cov_grad_norm_sum += gnorm
                cov_grad_norm_n += 1
                batch_grad_sum += gnorm
            if need_sem:
                objsem = semantic_objective(acts[feat], v_x0).sum()
                (gs,) = torch.autograd.grad(objsem, x)

            combined = combine_gradients(g1, gc, gs, lam2, lam3)
            x = pgd_step(x, x0_pixel, combined, step_size, eps)
        x_adv = x
        cov_grad_history.append(batch_grad_sum / steps if need_cov else 0.0)

        with torch.no_grad():
            before = tracker.covered_den().clone()  # 轮前 D_en 覆盖状态，逐种子归因用
            adv_acts = extractor.extract(imagenet_normalize(x_adv))
            tracker.update(adv_acts)
            adv_crit = normalize_acts(flatten_acts(adv_acts, layers)[:, den_idx], low_den, high_den)
            final_probs = ensemble.probs(imagenet_normalize(x_adv))
            s_in = s_input(adv_acts[feat], v_x0)
            s_pa = s_path(adv_crit, crit0)
        final_labels = {name: p.argmax(dim=1) for name, p in final_probs.items()}
        final_conf = final_probs[target].gather(1, index).squeeze(1)
        pert = (x_adv - x0_pixel).abs().flatten(1).amax(dim=1)
        # 逐种子新覆盖：该样本把哪些轮前未覆盖的关键神经元推过 t_cov。
        new_cov = ((adv_crit > t_cov) & ~before.unsqueeze(0)).sum(dim=1)

        cncov_now = tracker.cncov
        delta_cncov = cncov_now - report.cncov_history[-1]

        round_defects = 0
        round_valid = 0
        round_novel = 0
        s_in_round_sum = 0.0
        outcomes: list[SeedOutcome] = []
        for i in range(len(selected)):
            ci = int(c[i])
            tgt_label = int(final_labels[target][i])
            ref_labels = [int(final_labels[name][i]) for name in ensemble.references]
            si = float(s_in[i])
            sp = float(s_pa[i])
            semantic_ok = si >= gamma
            verdict = triage(tgt_label, ref_labels, ci, semantic_ok=semantic_ok)

            n_fuzzed += 1
            if all(r == ci for r in ref_labels):
                refs_hold += 1
            conf_drop_sum += float(orig_conf[i] - final_conf[i])
            pert_sum += float(pert[i])
            s_input_sum += si
            s_in_round_sum += si
            if semantic_ok:
                sem_valid += 1
                round_valid += 1
            path_novel = sp < theta
            if path_novel:
                round_novel += 1

            is_defect = verdict is Verdict.DEFECT
            if is_defect:
                round_defects += 1
                if path_novel:
                    defect_path_novel += 1
                report.defects.append(
                    DefectRecord(
                        image=x_adv[i].detach().cpu(),
                        source_label=ci,
                        target_label=tgt_label,
                        target_model=target,
                        s_input=si,
                        s_path=sp,
                        perturbation=float(pert[i]),
                        critical_activation=adv_crit[i].detach().cpu(),
                        source_image=x0_pixel[i].detach().cpu(),
                    )
                )
            outcomes.append(
                SeedOutcome(
                    seed=selected[i],
                    new_coverage=float(new_cov[i]),
                    path_novel=path_novel,
                    s_input=si,
                    produced_defect=is_defect,
                )
            )

        pool.update_after_round(outcomes)

        n_sel = len(selected)
        round_rft = round_defects / n_sel
        round_sem_shift = 1.0 - s_in_round_sum / n_sel
        round_valid_rate = round_valid / n_sel
        report.cncov_history.append(cncov_now)
        report.cccov_history.append(tracker.cccov())
        report.rft_history.append(round_rft)
        report.sem_shift_history.append(round_sem_shift)
        report.lambda_history.append((lam2, lam3))
        # 累计多样性：到本轮为止不同的 (源,目标) 对数与目标类别数，单调不减，供多样性增长曲线
        report.pair_history.append(len({(d.source_label, d.target_label) for d in report.defects}))
        report.target_history.append(len({d.target_label for d in report.defects}))

        feedback.step(
            FeedbackState(
                delta_cncov=delta_cncov,
                rft=round_rft,
                mean_sem_shift=round_sem_shift,
                path_novel_ratio=round_novel / n_sel,
            )
        )

        if rnd % log_interval == 0:
            logger.info(
                f"轮 {rnd}: CNCov {cncov_now:.3f}(Δ{delta_cncov:+.4f}) 缺陷 {round_defects} "
                f"有效率 {round_valid_rate:.3f} λ2={lam2:.3f} λ3={lam3:.3f} 活跃 {pool.n_active}"
            )

        # 终止条件
        if cncov_now >= cncov_target:
            logger.info(f"CNCov 达标 {cncov_now:.3f} ≥ {cncov_target}，终止于第 {rnd} 轮")
            break
        double_low_streak = (
            double_low_streak + 1 if delta_cncov < _GROWTH_EPS and round_defects == 0 else 0
        )
        if double_low_streak >= growth_patience:
            logger.info(f"覆盖与缺陷增长连续 {growth_patience} 轮双低，终止于第 {rnd} 轮")
            break
        low_valid_streak = low_valid_streak + 1 if round_valid_rate < _DEGEN_VALID_RATE else 0
        if low_valid_streak >= growth_patience:
            logger.info(f"输入有效率连续 {growth_patience} 轮过低（退化），终止于第 {rnd} 轮")
            break
        if theta > 0.0:
            zero_novel_streak = zero_novel_streak + 1 if round_novel == 0 else 0
            if zero_novel_streak >= growth_patience:
                logger.info(f"路径新颖样本连续 {growth_patience} 轮为零（退化），终止于第 {rnd} 轮")
                break

    elapsed = time.perf_counter() - t0
    report.total_iterations = rounds_run
    report.elapsed_time = elapsed
    mean_cov_grad = cov_grad_norm_sum / cov_grad_norm_n if cov_grad_norm_n else 0.0
    report.metrics = {
        "seed_acceptance_rate": cons.acceptance_rate,
        "n_seeds_accepted": float(cons.accepted),
        "n_consensus_classes": float(len(consensus_classes)),
        "n_rounds": float(rounds_run),
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
        "lambda2": feedback.lambda2,
        "lambda3": feedback.lambda3,
        "lambda2_init": config.neurons.lambda2,
        "lambda3_init": config.semantic.lambda3,
        "input_valid_rate": sem_valid / n_fuzzed if n_fuzzed else 0.0,
        "mean_s_input": s_input_sum / n_fuzzed if n_fuzzed else 0.0,
        "path_novel_ratio": defect_path_novel / report.num_defects if report.num_defects else 0.0,
        "n_seeds_retired": float(pool.n_retired),
        "feedback_enabled": 1.0 if config.feedback.enabled else 0.0,
        "defects_per_sec": report.num_defects / elapsed if elapsed else 0.0,
    }
    report.curves = {"cov_grad_norm": cov_grad_history}
    logger.info(
        f"主循环完成：{rounds_run} 轮，缺陷 {report.num_defects}，"
        f"RFT {report.metrics['rft']:.3f}，CNCov {cncov0:.3f}->{tracker.cncov:.3f}，"
        f"输入有效率 {report.metrics['input_valid_rate']:.3f}，"
        f"λ2 {config.neurons.lambda2:.2f}->{feedback.lambda2:.3f}，"
        f"λ3 {config.semantic.lambda3:.2f}->{feedback.lambda3:.3f}，耗时 {elapsed:.1f}s"
    )
    return report
