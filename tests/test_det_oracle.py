"""检测差分 oracle、漏检候选选取与归因聚合的单元测试。纯 CPU、合成框，不加载模型。"""

from __future__ import annotations

import torch

from mfuzz.core.records import ConsensusAnchor, FailureRecord
from mfuzz.core.types import Detection
from mfuzz.differential.det_oracle import cluster_detections, judge_image
from mfuzz.evaluate.det_gt import validate_instances
from mfuzz.neurons.struct_attr import GraphResult, find_miss_candidate
from mfuzz.tasks.det_analysis import aggregate
from mfuzz.tasks.detection import _gt_failure_view


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


def _inst(kind: str, shares: dict, layer_shares: dict) -> dict:
    return {
        "kind": kind,
        "shares": shares,
        "layer_shares": layer_shares,
        "level": None,
        "inside": None,
    }


def test_gt_failure_view_and_gen_verification() -> None:
    # 生成失效记录转真值核验视图：种子图名取自 extra，坐标系与种子一致。
    fr = FailureRecord(
        kind="spurious",
        anchor=None,
        observed_label="dog",
        observed_box=torch.tensor(BOX, dtype=torch.float32),
        observed_score=0.8,
        extra={"seed_image": "000000000139.jpg"},
    )
    view = _gt_failure_view(fr)
    assert view is not None
    assert view["image"] == "000000000139.jpg"
    assert view["det"] == {"box": BOX, "label": "dog", "score": 0.8}

    # 同名真值框重叠 -> spurious 被真值反驳（supported_by_gt）。
    gt = {"000000000139.jpg": [{"box": BOX_NEAR, "name": "dog", "crowd": False}]}
    res = validate_instances([view], gt, iou_thr=0.5, loc_thr=0.7)
    assert res["counts"]["spurious"] == {"supported_by_gt": 1}

    # 旧记录没有 seed_image：跳过、不计入。
    old = FailureRecord(kind="miss", anchor=ConsensusAnchor(label="cat"))
    assert _gt_failure_view(old) is None


def test_aggregate_layer_drilldown_ranks_target_layer() -> None:
    # loc 在 head.b 层的份额高于 agree 基线 4 倍，应排在下钻表首位；
    # cls 只有 2 个样本，低于最小样本量、整类跳过。
    instances = (
        [_inst("agree", {"head": 0.2}, {"head.a": 0.1, "head.b": 0.1}) for _ in range(12)]
        + [_inst("loc", {"head": 0.5}, {"head.a": 0.1, "head.b": 0.4}) for _ in range(12)]
        + [_inst("cls", {"head": 0.9}, {"head.a": 0.9}) for _ in range(2)]
    )
    agg = aggregate({"instances": instances, "deep_miss": 0})
    drill = agg["layer_drilldown"]
    assert "cls" not in drill  # 样本量不足
    rows = drill["loc"]
    assert rows[0]["layer"] == "head.b"
    assert rows[0]["ratio"] == 4.0
    assert rows[0]["n"] == 12
    assert {r["layer"] for r in rows} == {"head.a", "head.b"}
