"""多组实验对比脚本（实现方案 4.5）。

加载若干 result.json，写出 markdown 汇总表与对比图（CNCov 叠加、关键指标柱状图）到
输出目录。

用法：
    uv run python scripts/compare_experiments.py output/diff output/diff_cov \
        --out output/compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from mfuzz.evaluate.compare import compare


def main() -> None:
    parser = argparse.ArgumentParser(description="对比多组 result.json")
    parser.add_argument("results", nargs="+", help="result.json 路径或含它的目录，可多个")
    parser.add_argument("--out", default="output/compare", help="对比产物输出目录")
    args = parser.parse_args()

    table_path = compare(args.results, args.out)
    logger.info(f"汇总表写入 {table_path}")
    logger.info(f"对比图写入 {Path(args.out)}")
    print("\n" + table_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
