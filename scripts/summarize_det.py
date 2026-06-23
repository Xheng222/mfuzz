"""检测实验跨实验汇总入口。

按 output 重整后的新结构发现 output/det/<run>/<model>/data/result.json，写出一张跨
run/model 的对照表与对比图（CNCov 增长、累计失效、RFT 与语义偏移、差分失效计数）。

用法：
    # 扫整个 output/det 下所有实验
    uv run python scripts/summarize_det.py --root output/det --out output/det/_summary

    # 只扫单个 run 的各 model
    uv run python scripts/summarize_det.py --root output/det/base --out output/det/base/_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from mfuzz.evaluate.det_summary import summarize


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总 output/det 下各实验的 result.json")
    ap.add_argument(
        "--root", default="output/det", help="实验根目录，扫 <run>/<model>/data/result.json"
    )
    ap.add_argument("--out", default="output/det/_summary", help="汇总产物输出目录")
    args = ap.parse_args()

    table_path = summarize(args.root, args.out)
    logger.info(f"对照表写入 {table_path}")
    logger.info(f"对比图写入 {Path(args.out)}")
    print("\n" + table_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
