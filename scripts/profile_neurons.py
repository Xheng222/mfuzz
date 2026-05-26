"""关键神经元 profiling 与阈值校准辅助脚本。

按 TOML 配置对目标模型做一次 profiling，打印关键度分位数与若干候选 τ 下的
关键神经元占比，用于把占比校准到合理区间（实现方案第七章参考 30%-82%）。
profiling 结果写入缓存，正式 fuzzing 会直接复用。

用法：
    uv run python scripts/profile_neurons.py --config configs/diff_cov.toml
"""

from __future__ import annotations

import argparse

import torch
from loguru import logger

from mfuzz.core.datasets import build_dataset, make_loader
from mfuzz.core.models import load_ensemble
from mfuzz.core.seed import build_seed_pool
from mfuzz.core.types import load_config
from mfuzz.differential.consensus import filter_consensus
from mfuzz.differential.ensemble import Ensemble
from mfuzz.neurons.profiler import build_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="关键神经元 profiling / 阈值校准")
    parser.add_argument("--config", default="configs/diff_cov.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    names = config.models.names
    target = names[config.models.target_idx]
    ensemble = Ensemble(load_ensemble(names, device), target)

    bundle = build_dataset(config.dataset.name, config.dataset.val_fraction, config.random_seed)
    raw_seeds = build_seed_pool(
        bundle.seed_set, config.dataset.seed_size, device, config.random_seed
    )
    cons = filter_consensus(ensemble, raw_seeds, batch_size=config.dataset.batch_size)
    seeds = cons.seeds

    consensus_classes = sorted({s.consensus_label for s in seeds})
    profile_classes = [c for c in consensus_classes if c in bundle.class_to_indices]
    logger.info(
        f"共识类别 {len(consensus_classes)} 个，种子 {len(seeds)} 个，"
        f"有 profiling 数据的 {len(profile_classes)} 个"
    )

    bs = config.dataset.batch_size
    profile_loader = make_loader(bundle.profile_set, batch_size=bs, shuffle=False)
    class_loaders = {
        c: make_loader(bundle.class_subset(c), batch_size=bs, shuffle=False)
        for c in profile_classes
    }
    profile = build_profile(
        ensemble.models[target],
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

    logger.info(f"神经元总数 {profile.num_neurons}，α={profile.alpha}，t={profile.t}")
    pct = profile.cl_percentiles((0.1, 0.25, 0.5, 0.75, 0.9, 0.95))
    logger.info(
        "关键度 cl 分位数：" + ", ".join(f"p{int(q * 100)}={v:.3f}" for q, v in pct.items())
    )
    logger.info(f"全局 τ_global={profile.tau} 下关键占比 {profile.critical_ratio:.3f}")
    for tau in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        logger.info(f"  若 τ_global={tau:.1f}：关键占比 {profile.ratio_at(tau):.3f}")
    cls_ratios = profile.class_critical_ratios()
    if cls_ratios:
        vals = sorted(cls_ratios.values())
        mean = sum(vals) / len(vals)
        logger.info(
            f"类关键 τ_class={profile.tau_class} 下各类占比："
            f"min={vals[0]:.3f} mean={mean:.3f} max={vals[-1]:.3f}（{len(vals)} 类）"
        )


if __name__ == "__main__":
    main()
