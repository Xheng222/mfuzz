"""统一实验入口：一份 TOML 配置对应一个完整实验，task 字段选择任务适配器。

流程（逐目标模型轮换）：适配器构建（模型/数据/种子/覆盖标定）→ 反馈驱动主
循环（loop.max_iterations=0 时跳过，只跑分析）→ 适配器深度分析（检测：结构
归因/层级消融/真值核验/生成失效归因；分类：缺陷聚类与五维指标）→ 统一报告
（曲线 + 任务图表）→ 跨目标汇总图。

用法（服务器）：
    uv run python scripts/run_fuzz.py --config configs/det/base.toml
    uv run python scripts/run_fuzz.py --config configs/cls/base.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from mfuzz.core.config import load_config
from mfuzz.core.records import RunReport
from mfuzz.engine.loop import run_loop
from mfuzz.evaluate.run_report import generate_run_report
from mfuzz.tasks import adapter_class, build_adapter


def main() -> None:
    ap = argparse.ArgumentParser(description="mfuzz 统一实验入口")
    ap.add_argument("--config", default="configs/det/base.toml", help="实验配置 TOML 路径")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = Path(cfg.run.out)
    targets = cfg.target_names()
    logger.info(f"task={cfg.task} config={args.config} device={device} targets={targets} out={out}")

    per_target: dict[str, RunReport] = {}
    for target in targets:
        out_t = out / target
        if (out_t / "data" / "result.json").exists():
            logger.info(f"[{target}] 已有 result.json，跳过（断点续跑）")
            continue
        out_t.mkdir(parents=True, exist_ok=True)
        adapter = build_adapter(cfg, target, device, out_t)
        adapter.setup()
        seeds = adapter.build_seeds()
        tracker = adapter.build_tracker()
        logger.info(
            f"[{target}] 构建完成：种子 {len(seeds)}，单元 {tracker.profile.num_units}"
            f"（关键 {tracker.profile.num_critical}）"
        )
        if cfg.loop.max_iterations > 0:
            report = run_loop(adapter, tracker, seeds, cfg, out_t)
        else:
            report = RunReport()
            logger.info(f"[{target}] loop.max_iterations=0，跳过反馈循环、只跑分析")
        adapter.analyze(report, out_t)
        generate_run_report(report, adapter, cfg, out_t)
        per_target[target] = report

    adapter_class(cfg.task).plot_combined(per_target, cfg, out)
    logger.info(f"全部完成，产物写入 {out}")


if __name__ == "__main__":
    main()
