"""检测差分 oracle 与漏检候选选取的单元测试。纯 CPU、合成框，不加载模型。"""

from __future__ import annotations

import torch

from mfuzz.core.types import Detection
from mfuzz.differential.det_oracle import cluster_detections, judge_image
from mfuzz.neurons.struct_attr import GraphResult, find_miss_candidate


def _det(model: str, box: list[float], label: str, score: float) -> Detection:
    return Detection(model, torch.tensor(box, dtype=torch.float32), label, score)


BOX = [10.0, 10.0, 100.0, 100.0]
BOX_NEAR = [12.0, 12.0, 102.0, 102.0]  # 与 BOX 的 IoU 约 0.91
BOX_SHIFT = [40.0, 10.0, 130.0, 100.0]  # 与 BOX 的 IoU 约 0.5，落在 loc 带内
BOX_FAR = [300.0, 300.0, 400.0, 400.0]


def test_cluster_merges_across_models_only() -> None:
    dets = [
        _det("a", BOX, "person", 0.9),
        _det("b", BOX_NEAR, "person", 0.8),
        _det("a", BOX_NEAR, "dog", 0.7),  # 同模型不与第一个合并，但经 b 连通
        _det("c", BOX_FAR, "cat", 0.9),
    ]
    clusters = cluster_detections(dets, iou_thr=0.5)
    sizes = sorted(len(c.dets) for c in clusters)
    assert sizes == [1, 3]
    big = max(clusters, key=lambda c: len(c.dets))
    assert big.support == 2  # 三个框但只来自 a、b 两个模型

    assert big.majority_label() == "person"


def test_judge_four_kinds() -> None:
    dets_by_model = {
        "a": [_det("a", BOX, "person", 0.9), _det("a", BOX_FAR, "cat", 0.8)],
        "b": [_det("b", BOX_NEAR, "person", 0.8)],
        "c": [_det("c", BOX_SHIFT, "person", 0.7)],
    }
    records, n_cons = judge_image(dets_by_model, iou_thr=0.4, loc_thr=0.7)
    assert n_cons == 1
    kinds = {m: [r.kind for r in rs] for m, rs in records.items()}
    assert kinds["a"] == ["agree", "spurious"]
    assert kinds["b"] == ["agree"]
    assert kinds["c"] == ["loc"]  # IoU 约 0.5，低于 loc_thr


def test_judge_cls_takes_precedence() -> None:
    dets_by_model = {
        "a": [_det("a", BOX, "person", 0.9)],
        "b": [_det("b", BOX_NEAR, "person", 0.8)],
        "c": [_det("c", BOX_SHIFT, "dog", 0.7)],  # 类别错且定位偏，只记 cls
    }
    records, _ = judge_image(dets_by_model, iou_thr=0.4, loc_thr=0.7)
    assert [r.kind for r in records["c"]] == ["cls"]


def test_judge_miss() -> None:
    dets_by_model = {
        "a": [_det("a", BOX, "person", 0.9)],
        "b": [_det("b", BOX_NEAR, "person", 0.8)],
        "c": [],
    }
    records, _ = judge_image(dets_by_model, iou_thr=0.4, loc_thr=0.7)
    assert [r.kind for r in records["c"]] == ["miss"]
    assert records["c"][0].det is None
    assert records["c"][0].cons_label == "person"


def _graph(boxes: list[list[float]], scores: list[float], labels: list[str]) -> GraphResult:
    return GraphResult(
        dets=[],
        kept_idx=[],
        boxes_g=torch.tensor(boxes, dtype=torch.float32),
        scores_g=torch.tensor(scores, dtype=torch.float32),
        labels=labels,
        acts={},
        feats={},
    )


def test_find_miss_candidate_prefers_same_label() -> None:
    g = _graph(
        [BOX_NEAR, BOX, BOX_FAR],
        [0.4, 0.3, 0.45],
        ["dog", "person", "person"],
    )
    rep = torch.tensor(BOX, dtype=torch.float32)
    cand = find_miss_candidate(g, rep, "person", iou_thr=0.5, score_thr=0.5)
    assert cand is not None
    idx, match = cand
    assert idx == 1  # 同类别名的候选优先于更高分的异类候选
    assert match is True


def test_find_miss_candidate_falls_back_and_deep_miss() -> None:
    g = _graph([BOX_NEAR, BOX_FAR], [0.4, 0.45], ["dog", "person"])
    rep = torch.tensor(BOX, dtype=torch.float32)
    cand = find_miss_candidate(g, rep, "person", iou_thr=0.5, score_thr=0.5)
    assert cand is not None and cand == (0, False)  # 没有同类候选，退回最高分

    deep = find_miss_candidate(g, torch.tensor(BOX_FAR) + 500.0, "person", 0.5, 0.5)
    assert deep is None
