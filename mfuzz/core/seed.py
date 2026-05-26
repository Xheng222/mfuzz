"""种子集构建。

从数据集抽取样本构建 Seed 列表。Phase 0 只填 image 与 true_label，
consensus_label 留 -1 占位。Phase 1 的共识过滤（differential/consensus.py）
会送入所有模型，把一致预测写入 consensus_label、填各模型置信度，并据此
筛掉不一致的样本。
"""

from __future__ import annotations

import random

import torch
from torch import Tensor
from torch.utils.data import Dataset

from mfuzz.core.types import Seed


def build_seed_pool(
    dataset: Dataset[tuple[Tensor, int]],
    size: int,
    device: torch.device | str = "cpu",
    random_seed: int = 42,
) -> list[Seed]:
    """从数据集随机抽 size 个样本构建初始种子池（未做共识过滤）。"""
    rng = random.Random(random_seed)
    n = len(dataset)  # type: ignore[arg-type]
    indices = rng.sample(range(n), min(size, n))
    pool: list[Seed] = []
    for i in indices:
        img, true_label = dataset[i]
        pool.append(Seed(image=img.to(device), true_label=int(true_label)))
    return pool
