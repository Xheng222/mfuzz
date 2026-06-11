"""任务适配器注册。框架核心按 cfg.task 取适配器类，任务细节对核心透明。"""

from __future__ import annotations

from pathlib import Path

import torch

from mfuzz.core.adapter import TaskAdapter
from mfuzz.core.config import Config


def adapter_class(task: str) -> type[TaskAdapter]:
    if task == "detection":
        from mfuzz.tasks.detection import DetectionAdapter

        return DetectionAdapter
    if task == "classification":
        from mfuzz.tasks.classification import ClassificationAdapter

        return ClassificationAdapter
    raise KeyError(f"未知任务：{task}（可用：classification | detection）")


def build_adapter(cfg: Config, target: str, device: torch.device, out_dir: Path) -> TaskAdapter:
    return adapter_class(cfg.task)(cfg, target, device, out_dir)
