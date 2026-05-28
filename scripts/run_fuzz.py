"""fuzzing 入口脚本。

所有参数走 TOML 配置，命令行只选择配置文件。一份 TOML 完整定义一个实验，
要调参就改 TOML 或复制一份再用 --config 指定。

用法：
    uv run python scripts/run_fuzz.py --config configs/base.toml
"""

from __future__ import annotations

import argparse
import json

import torch
from loguru import logger

from mfuzz.core.types import load_config
from mfuzz.engine.runner import run_fuzz
from mfuzz.evaluate.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="mfuzz fuzzing 入口")
    parser.add_argument("--config", default="configs/base.toml", help="实验配置 TOML 路径")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    target = config.models.names[config.models.target_idx]
    logger.info(f"config={args.config}，device={device}，target={target}，out={config.run.out}")

    report = run_fuzz(config, device)

    # 评估总入口：补齐五维指标、写 result.json 与 defects.pt、画全部图。
    generate_report(report, config.run.out, config, target=target)
    logger.info("metrics:\n" + json.dumps(report.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
