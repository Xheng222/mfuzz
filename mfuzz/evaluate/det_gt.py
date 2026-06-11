"""真值核验：差分 oracle 的逐实例判定对照 COCO 真值标注。

差分判定完全来自跨模型一致性，没有用到真值。这里把逐实例记录逐条对照
instances_val2017.json，检查每类判定相对真值是否成立，产出"oracle 判定 ×
真值裁决"的混淆计数。裁决标准（IoU 阈值与差分判定时一致）：

- spurious：同名真值框 IoU 达标记 supported_by_gt（物体真实存在）；异名达标
  记 gt_label_differs；无真值重叠记 confirmed（真值确认虚检）。
- miss：consensus 代表框与同名真值框达标记 confirmed；异名达标记
  consensus_label_differs；无真值重叠记 consensus_unsupported（consensus
  本身不被真值支持，混合裁决：含真实未标注物体与集体误检）。
- cls：检测框先匹配真值框（任意类别），看真值类别站在哪边：model_right /
  consensus_right / both_wrong / no_gt_object。
- loc：物体身份由 consensus 定（代表框匹配同名真值框），再量检测框相对该
  真值框的 IoU：低于 loc 阈值记 confirmed，达标记 gt_loc_ok，匹配不到记
  no_gt。
- agree（抽样对照组）：correct / label_differs / unsupported。

iscrowd 的真值框参与匹配，命中时实例裁决带 crowd 标志。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torchvision.ops import box_iou

VERDICTS: dict[str, tuple[str, ...]] = {
    "spurious": ("confirmed", "supported_by_gt", "gt_label_differs"),
    "miss": ("confirmed", "consensus_label_differs", "consensus_unsupported"),
    "cls": ("consensus_right", "model_right", "both_wrong", "no_gt_object"),
    "loc": ("confirmed", "gt_loc_ok", "no_gt"),
    "agree": ("correct", "label_differs", "unsupported"),
}


def load_gt(ann_path: str | Path) -> dict[str, list[dict]]:
    """file_name -> [{box(xyxy), name, crowd}]。box 从 COCO 的 xywh 转 xyxy。"""
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)
    cat_name = {c["id"]: c["name"] for c in ann["categories"]}
    file_of = {im["id"]: im["file_name"] for im in ann["images"]}
    gt: dict[str, list[dict]] = defaultdict(list)
    for a in ann["annotations"]:
        x, y, w, h = a["bbox"]
        gt[file_of[a["image_id"]]].append(
            {
                "box": [x, y, x + w, y + h],
                "name": cat_name[a["category_id"]],
                "crowd": bool(a.get("iscrowd", 0)),
            }
        )
    return gt


def best_match(
    box: list[float], gts: list[dict], label: str | None = None
) -> tuple[float, dict | None]:
    """box 与真值框的最佳 IoU 匹配。label 给定时只在同名真值里找。"""
    pool = [g for g in gts if label is None or g["name"] == label]
    if not pool:
        return 0.0, None
    ious = box_iou(torch.tensor([box]), torch.tensor([g["box"] for g in pool]))[0]
    k = int(ious.argmax())
    return float(ious[k]), pool[k]


def judge_instance(inst: dict, gts: list[dict], iou_thr: float, loc_thr: float) -> dict:
    """单条实例的真值裁决。返回 verdict 与裁决依据（IoU、真值类别、crowd）。"""
    kind = inst["kind"]
    out: dict = {"verdict": "", "gt_iou": 0.0, "gt_label": None, "crowd": False}

    def fill(verdict: str, iou: float, g: dict | None) -> dict:
        out["verdict"] = verdict
        out["gt_iou"] = round(iou, 3)
        if g is not None:
            out["gt_label"] = g["name"]
            out["crowd"] = g["crowd"]
        return out

    if kind == "spurious":
        det = inst["det"]
        iou_s, g_s = best_match(det["box"], gts, det["label"])
        if iou_s >= iou_thr:
            return fill("supported_by_gt", iou_s, g_s)
        iou_a, g_a = best_match(det["box"], gts)
        if iou_a >= iou_thr:
            return fill("gt_label_differs", iou_a, g_a)
        return fill("confirmed", iou_a, None)

    if kind == "miss":
        iou_s, g_s = best_match(inst["rep_box"], gts, inst["cons_label"])
        if iou_s >= iou_thr:
            return fill("confirmed", iou_s, g_s)
        iou_a, g_a = best_match(inst["rep_box"], gts)
        if iou_a >= iou_thr:
            return fill("consensus_label_differs", iou_a, g_a)
        return fill("consensus_unsupported", iou_a, None)

    if kind == "cls":
        det = inst["det"]
        iou_a, g_a = best_match(det["box"], gts)
        if iou_a < iou_thr or g_a is None:
            return fill("no_gt_object", iou_a, None)
        if g_a["name"] == det["label"]:
            return fill("model_right", iou_a, g_a)
        if g_a["name"] == inst["cons_label"]:
            return fill("consensus_right", iou_a, g_a)
        return fill("both_wrong", iou_a, g_a)

    if kind == "loc":
        det = inst["det"]
        iou_rep, g_rep = best_match(inst["rep_box"], gts, inst["cons_label"])
        if iou_rep < iou_thr or g_rep is None:
            return fill("no_gt", iou_rep, None)
        iou_det = float(box_iou(torch.tensor([det["box"]]), torch.tensor([g_rep["box"]]))[0, 0])
        verdict = "confirmed" if iou_det < loc_thr else "gt_loc_ok"
        return fill(verdict, iou_det, g_rep)

    # agree 对照组
    det = inst["det"]
    iou_s, g_s = best_match(det["box"], gts, det["label"])
    if iou_s >= iou_thr:
        return fill("correct", iou_s, g_s)
    iou_a, g_a = best_match(det["box"], gts)
    if iou_a >= iou_thr:
        return fill("label_differs", iou_a, g_a)
    return fill("unsupported", iou_a, None)


def validate_instances(
    instances: list[dict], gt: dict[str, list[dict]], iou_thr: float, loc_thr: float
) -> dict:
    """对一组逐实例记录做真值核验，返回 counts（kind -> verdict -> n）与逐实例裁决。"""
    counts: dict[str, dict[str, int]] = {k: defaultdict(int) for k in VERDICTS}
    rows: list[dict] = []
    for inst in instances:
        gts = gt.get(inst["image"], [])
        res = judge_instance(inst, gts, iou_thr, loc_thr)
        counts[inst["kind"]][res["verdict"]] += 1
        rows.append(
            {
                "image": inst["image"],
                "kind": inst["kind"],
                "deep_miss": inst.get("deep_miss"),
                "cons_label": inst["cons_label"],
                "det_label": inst["det"]["label"] if inst.get("det") else None,
                "det_score": inst["det"]["score"] if inst.get("det") else None,
                "viz": inst.get("viz"),
                **res,
            }
        )
    return {"counts": {k: dict(v) for k, v in counts.items()}, "instances": rows}
