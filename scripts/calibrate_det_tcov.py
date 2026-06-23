"""检测覆盖参数标定：在配置指定的标定集上建（或复用）两遍流式统计缓存，
然后对参数网格离线扫描——t_freq 取缓存里的各档，critical_tau 与 t_cov 是
事后参数——报告每个组合下种子集的初始覆盖 CNCov_0 与关键单元数。

用法：
  uv run python scripts/calibrate_det_tcov.py --config configs/det/base.toml
  uv run python scripts/calibrate_det_tcov.py --target fcos   # 单目标，便于多卡并行

首跑会做全量两遍前向（小时级），之后走 output/cache/profiles 缓存、秒级出全网格。
结果同时落 JSON 到 output/det/tcov_scan/。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

from mfuzz.core.config import load_config
from mfuzz.core.det_models import load_detectors
from mfuzz.neurons.struct_attr import GraphForward
from mfuzz.neurons.unit_coverage import profile_from_stats
from mfuzz.tasks.det_analysis import DetParams, gather_images, load_image
from mfuzz.tasks.detection import _gap_vector, build_det_profile

GRID_TCOV = (0.85, 0.90, 0.95, 0.98, 1.00, 1.05, 1.10)
GRID_TAU = (0.3, 0.5, 0.7)
_EPS = 1e-12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/det/base.toml")
    ap.add_argument("--target", default="", help="只跑一个目标模型（多卡并行用）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = DetParams.from_raw(cfg.raw)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if p.profile_dir:
        prof_paths = gather_images(p.profile_dir, p.profile_images)
    else:
        prof_paths = gather_images(p.image_dir, p.profile_images, offset=p.num_images)
    seed_paths = gather_images(p.image_dir, p.num_images)
    targets = [args.target] if args.target else cfg.target_names()
    logger.info(f"标定集 {len(prof_paths)} 张，种子集 {len(seed_paths)} 张，目标 {targets}")

    out_dir = Path("output/det/tcov_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        det = load_detectors([target], device)[target]
        gf = GraphForward(det)
        # 建/载缓存（含多档 t_freq 的频率统计）
        build_det_profile(
            target,
            gf,
            prof_paths,
            cfg.coverage.t_freq,
            cfg.coverage.critical_tau,
            p.score_thr,
            device,
            p.cache_dir,
        )
        tag = prof_paths[0].parent.name
        cache = Path(p.cache_dir) / f"det_{target}_{tag}_n{len(prof_paths)}.pt"
        stats = torch.load(cache, map_location="cpu", weights_only=False)

        seed_vecs = []
        with torch.no_grad():
            for pa in seed_paths:
                g = gf.run(load_image(pa, device), p.score_thr)
                seed_vecs.append(_gap_vector(g.acts, stats["layout"], device).cpu())
                del g
        mat = torch.stack(seed_vecs)

        rows = []
        for t_freq in sorted(stats["freqs"]):
            freq = stats["freqs"][t_freq]
            for tau in GRID_TAU:
                profile = profile_from_stats(
                    stats["layout"], stats["low"], stats["high"], freq, tau
                )
                crit = profile.critical_idx
                lo, hi = profile.low[crit], profile.high[crit]
                mx = ((mat[:, crit] - lo) / (hi - lo + _EPS)).amax(dim=0)
                for t_cov in GRID_TCOV:
                    cov0 = float((mx > t_cov).float().mean())
                    rows.append(
                        {
                            "t_freq": t_freq,
                            "critical_tau": tau,
                            "t_cov": t_cov,
                            "n_critical": profile.num_critical,
                            "cncov_0": round(cov0, 4),
                        }
                    )
                    logger.info(
                        f"[{target}] t_freq={t_freq:.1f} tau={tau:.1f} "
                        f"t_cov={t_cov:.2f}  关键 {profile.num_critical}  CNCov_0={cov0:.3f}"
                    )
        with open(out_dir / f"{target}.json", "w", encoding="utf-8") as f:
            json.dump({"target": target, "n_profile": len(prof_paths), "rows": rows}, f, indent=1)
        del det, gf, seed_vecs, mat
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
