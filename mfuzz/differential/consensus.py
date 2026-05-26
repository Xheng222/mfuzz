"""共识过滤与置信度记录。

候选种子送进所有模型，只保留全部一致的样本，把一致预测记为初始共识标签
c（c 不一定是真实标签）。同时记录各模型对 c 的置信度。所有模型都低置信度
的种子标低优先级，由调度阶段决定是否当主要变异起点。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mfuzz.core.types import Seed
from mfuzz.differential.ensemble import Ensemble


@dataclass
class ConsensusResult:
    seeds: list[Seed]  # 通过共识过滤的种子，consensus_label 与置信度已填
    total: int  # 送检种子总数
    accepted: int  # 通过数
    low_confidence: int  # 通过但所有模型对 c 都低于阈值的数量

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.total if self.total else 0.0


def filter_consensus(
    ensemble: Ensemble,
    seeds: list[Seed],
    batch_size: int = 32,
    low_conf_threshold: float = 0.1,
) -> ConsensusResult:
    """对种子做共识过滤，返回通过的种子及统计。"""
    device = seeds[0].image.device if seeds else torch.device("cpu")
    accepted: list[Seed] = []
    low_conf = 0

    for start in range(0, len(seeds), batch_size):
        chunk = seeds[start : start + batch_size]
        batch = torch.stack([s.image for s in chunk]).to(device)
        probs = ensemble.probs(batch, with_grad=False)  # name -> (B, C)
        labels = {name: p.argmax(dim=1) for name, p in probs.items()}

        stacked = torch.stack([labels[name] for name in ensemble.names], dim=0)  # (M, B)
        agree = (stacked == stacked[0]).all(dim=0)  # (B,) 全模型一致

        for i, seed in enumerate(chunk):
            if not bool(agree[i]):
                continue
            c = int(stacked[0, i])
            confidences = {name: float(probs[name][i, c]) for name in ensemble.names}
            seed.consensus_label = c
            seed.model_confidences = confidences
            if max(confidences.values()) < low_conf_threshold:
                low_conf += 1
            accepted.append(seed)

    return ConsensusResult(
        seeds=accepted, total=len(seeds), accepted=len(accepted), low_confidence=low_conf
    )
