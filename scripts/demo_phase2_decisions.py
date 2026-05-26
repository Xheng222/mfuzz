"""把 Phase 2 几处工程决定的依据在真实模型与数据上打印出来，可重复运行。

依次演示：
  A  激活按神经元峰值归一化的必要性——固定阈值 t 直接卡原始激活，各层激活
     比例天差地别，有的层整层不激活；归一化后每层都进得来、比例可比。
  B  关键集合用分位阈值而非绝对 cl>τ——绝对阈值不可移植，同一 τ 选出的占比
     随模型与数据漂移；分位阈值让占比≈1-τ，可控且落在合理区间。
  C  全局 D_en 用宽阈值 τ_global、类关键 D_en^c 用严阈值 τ_class，两套分位分开。
     D_en^c 更小更类专属、两两重合度低，同一本全局覆盖账投到各类 D_en^c 上，
     CCCov 跨类差异明显——不需要分类别记账。对照：若 D_en^c 也用宽的 τ_global，
     各类集合高度重合，CCCov 投到任何一类都≈全局 CNCov，区分不出类别。

用法：
    uv run python scripts/demo_phase2_decisions.py --config configs/diff_cov.toml
"""

from __future__ import annotations

import argparse
import itertools
import sys

import torch
from torch import Tensor

from mfuzz.core.datasets import build_dataset, make_loader
from mfuzz.core.hooks import ActivationExtractor
from mfuzz.core.models import load_ensemble
from mfuzz.core.seed import build_seed_pool
from mfuzz.core.types import load_config
from mfuzz.differential.consensus import filter_consensus
from mfuzz.differential.ensemble import Ensemble
from mfuzz.neurons.profiler import build_profile, flatten_acts


def _layer_slices(counts: dict[str, int]) -> dict[str, slice]:
    slices, off = {}, 0
    for name, c in counts.items():
        slices[name] = slice(off, off + c)
        off += c
    return slices


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description="Phase 2 工程决定演示")
    parser.add_argument("--config", default="configs/diff_cov.toml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    target = cfg.models.names[cfg.models.target_idx]
    ens = Ensemble(load_ensemble(cfg.models.names, dev), target)
    model = ens.models[target]
    ext = ActivationExtractor(model)
    layers = ext.layer_names
    t = cfg.neurons.activation_threshold

    bundle = build_dataset(cfg.dataset.name, cfg.dataset.val_fraction, cfg.random_seed)
    raw = build_seed_pool(bundle.seed_set, cfg.dataset.seed_size, dev, cfg.random_seed)
    cons = filter_consensus(ens, raw, batch_size=cfg.dataset.batch_size)
    seeds = cons.seeds
    classes = [
        c for c in sorted({s.consensus_label for s in seeds}) if c in bundle.class_to_indices
    ]
    bs = cfg.dataset.batch_size
    profile_loader = make_loader(bundle.profile_set, batch_size=bs, shuffle=False)
    class_loaders = {
        c: make_loader(bundle.class_subset(c), batch_size=bs, shuffle=False) for c in classes
    }

    profile = build_profile(
        model,
        target,
        profile_loader,
        class_loaders,
        t=t,
        tau=cfg.neurons.critical_threshold,
        tau_class=cfg.neurons.class_critical_threshold,
        alpha=cfg.neurons.alpha,
        dataset_name=cfg.dataset.name,
        val_fraction=cfg.dataset.val_fraction,
        cache_dir=cfg.neurons.cache_dir,
        device=dev,
    )
    scale = profile.scale
    print(f"\n目标模型 {target}，t={t}，神经元 {profile.num_neurons}，共识类别 {len(classes)}")

    # ---------- A 归一化必要性 ----------
    # 全局频率 profile.freq 就是归一化后的激活比例；再流式数一遍原始 act>t 作对照。
    raw_count: Tensor | None = None
    total = 0
    with torch.no_grad():
        for x, _ in profile_loader:
            fired = (flatten_acts(ext.extract(x.to(dev)), layers) > t).sum(dim=0)
            raw_count = fired if raw_count is None else raw_count + fired
            total += x.shape[0]
    assert raw_count is not None
    raw_active = raw_count.float() / total
    norm_active = profile.freq

    print("\n========== A  固定阈值 t 直接卡原始激活 vs 按神经元峰值归一化 ==========")
    slices = _layer_slices(profile.counts)
    sel = list(slices)
    pick = [sel[0], sel[len(sel) // 3], sel[2 * len(sel) // 3], sel[-1]]
    print(f"{'层':28}{'原始>t 激活比例':>16}{'归一化>t 激活比例':>18}")
    for ln in pick:
        sl = slices[ln]
        print(
            f"{ln[:26]:28}{float(raw_active[sl].mean()):>16.3f}{float(norm_active[sl].mean()):>18.3f}"
        )
    lr = torch.tensor([float(raw_active[slices[ln]].mean()) for ln in sel])
    ln_ = torch.tensor([float(norm_active[slices[ln]].mean()) for ln in sel])

    def _spread(v: Tensor) -> str:
        return f"min={float(v.min()):.3f} max={float(v.max()):.3f} std={float(v.std()):.3f}"

    print(f"\n跨 {len(sel)} 层层均激活比例：")
    print(f"  原始阈值 : {_spread(lr)}")
    print(f"  归一化后 : {_spread(ln_)}")
    print("结论：原始阈值下有层整层不激活、层间标准差大，单一 t 无意义；归一化后各层可比。")

    # ---------- B 分位 vs 绝对阈值 ----------
    cl = profile.cl
    print("\n========== B  关键度 cl 分布与两种阈值下的关键占比 ==========")
    qs = (0.5, 0.75, 0.9, 0.95, 0.99)
    print(
        "cl 分位数: " + ", ".join(f"p{int(q * 100)}={float(torch.quantile(cl, q)):.4f}" for q in qs)
    )
    print(f"{'τ':>6}{'绝对阈值 cl>τ 占比':>20}{'分位阈值占比(≈1-τ)':>22}")
    for tau in (0.05, 0.1, 0.2, 0.5, 0.9):
        abs_r = float((cl > tau).float().mean())
        q_r = float((cl > torch.quantile(cl, tau)).float().mean())
        print(f"{tau:>6.2f}{abs_r:>20.3f}{q_r:>22.3f}")
    print("结论：绝对阈值占比随 τ 漂移、难落进 30%-82%；分位阈值占比≈1-τ，可控。")

    # ---------- C 严 τ_class 让 D_en^c 拉开 + 全局账下 CCCov 跨类有差异 ----------
    # 对照：把 D_en^c 也按宽的 τ_global 取一遍，看集合是否高度重合、CCCov 是否塌缩。
    masks = {c: profile.critical_per_class[c] for c in classes}
    wide = {c: profile.class_critical_mask_at(c, cfg.neurons.critical_threshold) for c in classes}

    def _jaccard(ms: dict[int, Tensor]) -> list[float]:
        return [
            float((ms[a] & ms[b]).sum()) / float((ms[a] | ms[b]).sum())
            for a, b in itertools.combinations(classes, 2)
        ]

    covered = torch.zeros(profile.num_neurons, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for start in range(0, len(seeds), bs):
            xb = torch.stack([s.image for s in seeds[start : start + bs]]).to(dev)
            fired = (flatten_acts(ext.extract(xb), layers) / scale) > t
            covered |= fired.any(dim=0)

    def _cccov(ms: dict[int, Tensor]) -> Tensor:
        return torch.tensor([float(covered[ms[c]].float().mean()) for c in classes])

    jac, jac_w = _jaccard(masks), _jaccard(wide)
    cccov, cccov_w = _cccov(masks), _cccov(wide)
    size = sum(int(masks[c].sum()) for c in classes) / len(classes)
    size_w = sum(int(wide[c].sum()) for c in classes) / len(classes)
    print("\n========== C  τ_class（严）vs τ_global（宽）下 D_en^c 与全局账 CCCov ==========")
    print(f"{'方案':28}{'每类集合均值':>12}{'两两Jaccard均值':>16}{'CCCov std':>12}")
    print(
        f"{f'τ_class={profile.tau_class} (严, 实际用)':28}"
        f"{size:>12.0f}{sum(jac) / len(jac):>16.3f}{float(cccov.std()):>12.4f}"
    )
    print(
        f"{f'τ_global={profile.tau} (宽, 对照)':28}"
        f"{size_w:>12.0f}{sum(jac_w) / len(jac_w):>16.3f}{float(cccov_w.std()):>12.4f}"
    )
    print(
        f"严阈值下 CCCov: min={float(cccov.min()):.3f} "
        f"max={float(cccov.max()):.3f} std={float(cccov.std()):.4f}"
    )
    print("结论：宽阈值下 D_en^c 大而高度重合，CCCov 投到各类都≈全局 CNCov，std 趋零、区分不出")
    print("     类别；严的 τ_class 让集合小而类专属、重合度降，CCCov 跨类 std 明显拉大。")


if __name__ == "__main__":
    main()
