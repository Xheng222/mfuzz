"""fuzzing 入口脚本。

Phase 1 支持 --mode diff（纯差分）。后续阶段接入 diff+cov、diff+cov+sem
与完整动态反馈模式。

用法：
    uv run python scripts/run_fuzz.py --config configs/base.toml --mode diff
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

from mfuzz.core.types import load_config
from mfuzz.engine.runner import run_differential


def main() -> None:
    parser = argparse.ArgumentParser(description="mfuzz fuzzing 入口")
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--mode", default="diff", choices=["diff"])
    parser.add_argument("--seed-size", type=int, default=None, help="覆盖配置的种子数")
    parser.add_argument("--target", default=None, help="覆盖目标模型名")
    parser.add_argument("--out", default="output/diff")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed_size is not None:
        config.dataset.seed_size = args.seed_size
    if args.target is not None:
        config.models.target_idx = config.models.names.index(args.target)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    target = config.models.names[config.models.target_idx]
    logger.info(f"device={device}，mode={args.mode}，target={target}")

    if args.mode == "diff":
        report = run_differential(config, device)
    else:  # pragma: no cover - 仅 diff 在 Phase 1 可用
        raise ValueError(f"模式 {args.mode} 尚未实现")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": args.mode,
        "target_model": target,
        "metrics": report.metrics,
        "curves": report.curves,
        "total_iterations": report.total_iterations,
        "elapsed_time": report.elapsed_time,
        "num_defects": report.num_defects,
    }
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("metrics:\n" + json.dumps(report.metrics, ensure_ascii=False, indent=2))
    logger.info(f"结果已写入 {result_path}")


if __name__ == "__main__":
    main()
