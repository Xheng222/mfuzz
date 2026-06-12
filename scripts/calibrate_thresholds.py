"""语义阈值校准（实现方案 4.3）。

校准两个语义阈值，只报告分布与建议值，不自动改配置——γ_input 还要人工抽检确认。

- γ_input：对代表性种子施加不同量级的像素变异，统计每档 S_input 分布。变异越大语义
  偏移越大、S_input 越低。在实验用的 ε 档取一个低分位（默认 5%）作建议 γ_input：让正常
  小变异基本通过、把明显破坏语义的样本挡在外面。
- θ_path：按共识类抽同类样本，在关键神经元集合 D_en 上两两算 S_path，取下四分位数（Q1）
  作建议 θ_path（参考 NSGen）。低于该值说明路径比同类常态更分散，算新颖、该优先保留。

用法：
    uv run python scripts/calibrate_thresholds.py --config configs/cls/base.toml
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from mfuzz.core.config import Config, load_config
from mfuzz.core.datasets import (
    build_dataset,
    imagenet_denormalize,
    imagenet_normalize,
    make_loader,
)
from mfuzz.core.hooks import ActivationExtractor
from mfuzz.core.models import load_ensemble
from mfuzz.core.seed import build_seed_pool
from mfuzz.differential.consensus import filter_consensus
from mfuzz.differential.ensemble import Ensemble
from mfuzz.neurons.coverage import CoverageTracker
from mfuzz.neurons.profiler import build_profile, flatten_acts, normalize_acts
from mfuzz.semantic.feature import feature_layer, s_input
from mfuzz.tasks.classification import ClsParams

_EPS_GRID = [0.01, 0.03, 0.05, 0.1, 0.2]  # 像素 L∞ 变异档位
_PCTS = [5, 25, 50, 75, 95]
_MAX_CLASSES = 20  # θ_path 抽样的类别数上限
_MAX_PER_CLASS = 12  # 每类样本数上限


def _percentiles(x: Tensor, pcts: list[int]) -> dict[int, float]:
    qs = torch.tensor([p / 100.0 for p in pcts], device=x.device)
    vals = torch.quantile(x, qs)
    return {p: float(v) for p, v in zip(pcts, vals, strict=True)}


def _fmt(d: dict[int, float]) -> str:
    return "  ".join(f"p{p}={v:.4f}" for p, v in d.items())


def calibrate_gamma(
    target_model: torch.nn.Module,
    feat: str,
    seeds_norm: Tensor,
    op_epsilon: float,
    device: torch.device,
    random_seed: int,
) -> float:
    """逐 ε 档统计 S_input 分布，返回实验 ε 档低分位作建议 γ_input。"""
    extractor = ActivationExtractor(target_model)
    gen = torch.Generator(device=device).manual_seed(random_seed)
    x0_pixel = imagenet_denormalize(seeds_norm)
    with torch.no_grad():
        v0 = extractor.extract(seeds_norm)[feat]
    logger.info(f"γ_input 校准：{seeds_norm.shape[0]} 个种子，特征层 {feat}")
    suggestion = 0.9
    for eps in _EPS_GRID:
        sign = torch.randint(0, 2, x0_pixel.shape, generator=gen, device=device).float() * 2 - 1
        perturbed = torch.clamp(x0_pixel + eps * sign, 0.0, 1.0)
        with torch.no_grad():
            v = extractor.extract(imagenet_normalize(perturbed))[feat]
            si = s_input(v, v0)
        pct = _percentiles(si, _PCTS)
        mark = " <- 实验档" if abs(eps - op_epsilon) < 1e-9 else ""
        logger.info(f"  ε={eps:<5}: mean={float(si.mean()):.4f}  {_fmt(pct)}{mark}")
        if abs(eps - op_epsilon) < 1e-9:
            suggestion = pct[5]
    logger.info(
        f"建议 γ_input ≈ {suggestion:.3f}（实验 ε={op_epsilon} 档的 5% 分位，需人工抽检确认）"
    )
    return suggestion


def _class_path_vectors(
    extractor: ActivationExtractor,
    layers: list[str],
    den_idx: Tensor,
    low_den: Tensor,
    high_den: Tensor,
    loader: object,
    device: torch.device,
    cap: int,
) -> Tensor:
    """取一类样本在 D_en 上的归一化激活向量 (m, K)，最多 cap 个。"""
    vecs: list[Tensor] = []
    seen = 0
    with torch.no_grad():
        for x, _ in loader:  # type: ignore[attr-defined]
            flat = flatten_acts(extractor.extract(x.to(device)), layers)[:, den_idx]
            vecs.append(normalize_acts(flat, low_den, high_den))
            seen += x.shape[0]
            if seen >= cap:
                break
    return torch.cat(vecs)[:cap]


def calibrate_theta(
    target_model: torch.nn.Module,
    profile: object,
    bundle: object,
    classes: list[int],
    device: torch.device,
    t_cov: float,
) -> float:
    """同类样本两两 S_path 的下四分位数作建议 θ_path。"""
    tracker = CoverageTracker(profile, device, t_cov)  # type: ignore[arg-type]
    extractor = ActivationExtractor(target_model)
    layers = profile.layers  # type: ignore[attr-defined]
    den_idx, low_den, high_den = tracker.den_idx, tracker.low_den, tracker.high_den
    sims: list[Tensor] = []
    used = 0
    for c in classes:
        if used >= _MAX_CLASSES:
            break
        loader = make_loader(bundle.class_subset(c), batch_size=_MAX_PER_CLASS, shuffle=False)  # type: ignore[attr-defined]
        u = _class_path_vectors(
            extractor, layers, den_idx, low_den, high_den, loader, device, _MAX_PER_CLASS
        )
        if u.shape[0] < 2:
            continue
        un = F.normalize(u, dim=1)
        pair = un @ un.t()  # (m, m) 两两余弦
        iu = torch.triu_indices(u.shape[0], u.shape[0], offset=1)
        sims.append(pair[iu[0], iu[1]])
        used += 1
    if not sims:
        logger.warning("θ_path 校准：无足够同类样本，保持默认 0.0")
        return 0.0
    allsim = torch.cat(sims)
    pct = _percentiles(allsim, _PCTS)
    logger.info(f"θ_path 校准：{used} 类、{allsim.numel()} 对同类样本  {_fmt(pct)}")
    logger.info(f"建议 θ_path ≈ {pct[25]:.3f}（同类 S_path 分布的下四分位数 Q1）")
    return pct[25]


def main() -> None:
    parser = argparse.ArgumentParser(description="mfuzz 语义阈值校准")
    parser.add_argument("--config", default="configs/cls/base.toml", help="实验配置 TOML 路径")
    args = parser.parse_args()

    config: Config = load_config(args.config)
    p = ClsParams.from_raw(config.raw)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    names = config.models.names
    target = config.target_names()[0]
    ensemble = Ensemble(load_ensemble(names, device), target)
    target_model = ensemble.models[target]
    logger.info(f"config={args.config}，device={device}，target={target}")

    bundle = build_dataset(p.dataset, p.val_fraction, config.random_seed)
    raw_seeds = build_seed_pool(bundle.seed_set, p.seed_size, device, config.random_seed)
    cons = filter_consensus(ensemble, raw_seeds, batch_size=p.batch_size)
    seeds = cons.seeds
    consensus_classes = sorted({s.consensus_label for s in seeds})
    profile_classes = [c for c in consensus_classes if c in bundle.class_to_indices]

    bs_prof = p.batch_size
    profile = build_profile(
        target_model,
        target,
        make_loader(bundle.profile_set, batch_size=bs_prof, shuffle=False),
        {
            c: make_loader(bundle.class_subset(c), batch_size=bs_prof, shuffle=False)
            for c in profile_classes
        },
        t=config.coverage.t_freq,
        tau=config.coverage.critical_tau,
        tau_class=p.class_critical_threshold,
        alpha=p.alpha,
        p_low=p.p_low,
        p_high=p.p_high,
        dataset_name=p.dataset,
        val_fraction=p.val_fraction,
        cache_dir=p.cache_dir,
        device=device,
    )

    feat = feature_layer(profile.layers)
    seeds_norm = torch.stack([s.image for s in seeds]).to(device)
    gamma = calibrate_gamma(
        target_model, feat, seeds_norm, config.optimize.epsilon, device, config.random_seed
    )
    theta = calibrate_theta(
        target_model, profile, bundle, profile_classes, device, config.coverage.t_cov
    )
    logger.info(
        f"校准结束。把 [semantic] gamma_input={gamma:.3f}、theta_path={theta:.3f} "
        f"按需手填进 {args.config}（建议先人工抽检几张变异图确认 γ_input）。"
    )


if __name__ == "__main__":
    main()
