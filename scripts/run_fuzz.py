"""fuzzing 入口脚本。

所有参数走 TOML 配置，命令行只选择配置文件。一份 TOML 完整定义一个实验，
要调参就改 TOML 或复制一份再用 --config 指定。

用法：
    uv run python scripts/run_fuzz.py --config configs/base.toml
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
    parser.add_argument("--config", default="configs/base.toml", help="实验配置 TOML 路径")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    target = config.models.names[config.models.target_idx]
    mode = config.run.mode
    logger.info(f"config={args.config}，device={device}，mode={mode}，target={target}")

    if mode == "diff":
        report = run_differential(config, device)
    else:  # pragma: no cover - 仅 diff 在 Phase 1 可用
        raise ValueError(f"模式 {mode!r} 尚未实现")

    out_dir = Path(config.run.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": mode,
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
