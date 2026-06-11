"""检测差分 oracle：跨模型 IoU 聚簇与四类失效判定。

分类版 consensus.py 与 triage.py 的检测对应物。没有真值标注，oracle 完全靠
跨模型一致性：把各模型的检测框按 IoU 跨模型聚成簇，被至少两个模型支持的簇
算一个 consensus 对象，再按每个模型相对 consensus 的偏离判四类失效：

- 漏检 miss：consensus 对象缺这个模型的框。
- 虚检 spurious：这个模型的框不被其它任何模型支持。
- 类别错误 cls：在 consensus 簇里有框，但类别名与多数票不一致。
- 定位偏移 loc：类别一致，但与代表框的 IoU 低于 loc 阈值。

类别比较在类别名空间进行（见 core/types.py 的 Detection）。cls 与 loc 同时
成立时归入 cls，一个实例只挂一个失效类型，方便逐实例归因。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torchvision.ops import box_iou

from mfuzz.core.types import Detection

FAILURE_KINDS = ("miss", "spurious", "cls", "loc")
RECORD_KINDS = (*FAILURE_KINDS, "agree")


@dataclass
class Cluster:
    """跨模型聚成的一簇框，指向同一物理对象。"""

    dets: list[Detection] = field(default_factory=list)

    @property
    def models(self) -> set[str]:
        return {d.model for d in self.dets}

    @property
    def support(self) -> int:
        return len(self.models)

    def majority_label(self) -> str:
        votes = Counter(d.label for d in self.dets)
        top = max(votes.values())
        cand = [lab for lab, c in votes.items() if c == top]
        if len(cand) == 1:
            return cand[0]
        # 票数并列时取分数最高的那个
        best = max((d for d in self.dets if d.label in cand), key=lambda d: d.score)
        return best.label

    def representative_box(self) -> Tensor:
        """consensus 代表框：多数类别里分数最高的框。"""
        lab = self.majority_label()
        return max((d for d in self.dets if d.label == lab), key=lambda d: d.score).box


def cluster_detections(dets: list[Detection], iou_thr: float) -> list[Cluster]:
    """跨模型按 IoU 聚簇：同模型的框不互相合并。并查集求连通分量。"""
    n = len(dets)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    if n > 1:
        boxes = torch.stack([d.box for d in dets])
        iou = box_iou(boxes, boxes)
        for i in range(n):
            for j in range(i + 1, n):
                if dets[i].model != dets[j].model and iou[i, j] >= iou_thr:
                    union(i, j)

    groups: dict[int, Cluster] = {}
    for i, d in enumerate(dets):
        groups.setdefault(find(i), Cluster()).dets.append(d)
    return list(groups.values())


@dataclass
class DetRecord:
    """某模型在一个 consensus 对象（或单点簇）上的判定结果。"""

    kind: str  # RECORD_KINDS 之一
    det: Detection | None  # miss 时为 None
    rep_box: Tensor  # consensus 代表框；spurious 用自身框
    cons_label: str
    cluster: list[Detection] | None = None  # 所在簇的全部检测，微观分析的上下文


def judge_image(
    dets_by_model: dict[str, list[Detection]], iou_thr: float, loc_thr: float
) -> tuple[dict[str, list[DetRecord]], int]:
    """对一张图的全部检测做差分判定，逐模型给出判定记录与 consensus 对象数。"""
    all_dets = [d for ds in dets_by_model.values() for d in ds]
    clusters = cluster_detections(all_dets, iou_thr)
    records: dict[str, list[DetRecord]] = {m: [] for m in dets_by_model}
    n_cons = 0
    for cl in clusters:
        if cl.support >= 2:
            n_cons += 1
            lab = cl.majority_label()
            rep = cl.representative_box()
            for m in dets_by_model:
                mine = [d for d in cl.dets if d.model == m]
                if not mine:
                    records[m].append(DetRecord("miss", None, rep, lab, cl.dets))
                    continue
                d = max(mine, key=lambda d: d.score)
                if d.label != lab:
                    records[m].append(DetRecord("cls", d, rep, lab, cl.dets))
                    continue
                iou = float(box_iou(d.box[None], rep[None])[0, 0])
                kind = "loc" if iou < loc_thr else "agree"
                records[m].append(DetRecord(kind, d, rep, lab, cl.dets))
        else:
            for d in cl.dets:
                records[d.model].append(DetRecord("spurious", d, d.box, d.label, cl.dets))
    return records, n_cons
