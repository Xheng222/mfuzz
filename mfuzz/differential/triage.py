"""候选样本四类分流。

按目标模型是否偏离共识、参考模型是否仍保持共识，把候选分四类：

- A DEFECT：目标偏离 c 且所有参考仍预测 c，进候选缺陷集合。
- B CONSENSUS：所有模型仍保持 c，丢弃。
- C GRAY：参考模型未全部保持 c（彼此分歧或整体漂移），进待核验集合，不归因
  于目标模型。
- D SEMANTIC_FAIL：语义已失效，由语义检查过滤（Phase 3 起生效；本阶段
  semantic_ok 恒 True，不产生 D）。
"""

from __future__ import annotations

from enum import Enum


class Verdict(Enum):
    DEFECT = "A"
    CONSENSUS = "B"
    GRAY = "C"
    SEMANTIC_FAIL = "D"


def triage(
    target_label: int,
    reference_labels: list[int],
    consensus_label: int,
    semantic_ok: bool = True,
) -> Verdict:
    """对单个候选样本分流。"""
    if not semantic_ok:
        return Verdict.SEMANTIC_FAIL
    refs_hold_c = all(label == consensus_label for label in reference_labels)
    if not refs_hold_c:
        # 参考模型没有全部保持 c，缺陷不可归因于目标模型
        return Verdict.GRAY
    if target_label != consensus_label:
        return Verdict.DEFECT
    return Verdict.CONSENSUS
