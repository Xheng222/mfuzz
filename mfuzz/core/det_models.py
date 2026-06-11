"""检测模型注册与家族适配层。

每个模型家族一个适配器，统一负责四件事：加载与权重冻结、前向出框（把本地
标签索引翻成类别名）、暴露可挂 hook 的结构（结构桶映射、FPN 尺度）、给出
变换后的输入尺寸。oracle 与归因层只依赖适配器接口，不接触家族差异。

当前实现 torchvision 家族（Faster R-CNN / RetinaNet / FCOS，共享 COCO 91 类
含占位项的类别表）。YOLO 家族的适配器待实现，层名风格与置信度语义都不同。
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from mfuzz.core.types import Detection

# 层名前缀 -> 结构桶。顺序即匹配优先级，第一个命中的生效。
# 三个 torchvision 模型共享 backbone/fpn 命名；rpn/roi 只在 Faster R-CNN，
# head.* 只在 RetinaNet / FCOS。来自 Day 3 的结构探查（scripts/probe_det_structure.py）。
TV_BUCKETS: list[tuple[str, str]] = [
    ("backbone.body.conv1", "backbone.stem"),
    ("backbone.body.layer1", "backbone.C2"),
    ("backbone.body.layer2", "backbone.C3"),
    ("backbone.body.layer3", "backbone.C4"),
    ("backbone.body.layer4", "backbone.C5"),
    ("backbone.fpn.inner_blocks", "fpn.inner"),
    ("backbone.fpn.layer_blocks", "fpn.layer"),
    ("backbone.fpn.extra_blocks", "fpn.extra"),
    ("rpn.head.conv", "rpn.shared"),
    ("rpn.head.cls_logits", "rpn.cls"),
    ("rpn.head.bbox_pred", "rpn.reg"),
    ("roi_heads.box_head", "roi.box_head"),
    ("roi_heads.box_predictor.cls_score", "roi.cls"),
    ("roi_heads.box_predictor.bbox_pred", "roi.reg"),
    ("head.classification_head", "head.cls"),
    ("head.regression_head", "head.reg"),
]

# 结构桶的展示顺序（聚合报表用）。
TV_BUCKET_ORDER: list[str] = list(dict.fromkeys(label for _, label in TV_BUCKETS))

_PLACEHOLDERS = {"__background__", "N/A"}


class TorchvisionDetector:
    """torchvision 检测模型的适配器。"""

    family = "torchvision"

    def __init__(self, name: str, device: torch.device | str = "cpu") -> None:
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            FCOS_ResNet50_FPN_Weights,
            RetinaNet_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
            fcos_resnet50_fpn,
            retinanet_resnet50_fpn_v2,
        )

        specs = {
            "faster_rcnn": (
                fasterrcnn_resnet50_fpn_v2,
                FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            ),
            "retinanet": (retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT),
            "fcos": (fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights.DEFAULT),
        }
        if name not in specs:
            raise ValueError(f"未知 torchvision 检测模型 {name!r}，可用：{list(specs)}")
        factory, weights = specs[name]
        self.name = name
        self.device = torch.device(device)
        self.model: nn.Module = factory(weights=weights)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.categories: list[str] = list(weights.meta["categories"])

    # ---- 标签空间 ----

    def label_names(self) -> set[str]:
        """本模型能预测的类别名集合，剔除占位项。"""
        return {c for c in self.categories if c not in _PLACEHOLDERS}

    def to_name(self, idx: int) -> str:
        return self.categories[idx]

    # ---- 前向 ----

    @torch.no_grad()
    def detect(self, img: Tensor, score_thr: float) -> list[Detection]:
        """单图前向，返回分数达标的检测，标签已翻成类别名。"""
        out = self.model([img])[0]
        keep = out["scores"] >= score_thr
        boxes = out["boxes"][keep].cpu()
        labels = out["labels"][keep].cpu()
        scores = out["scores"][keep].cpu()
        return [
            Detection(self.name, boxes[i], self.to_name(int(labels[i])), float(scores[i]))
            for i in range(boxes.shape[0])
        ]

    # ---- 结构 ----

    def bucket_of(self, layer_name: str) -> str:
        for prefix, label in TV_BUCKETS:
            if layer_name.startswith(prefix):
                return label
        return "other"

    def conv_layers(self) -> dict[str, nn.Conv2d]:
        return {n: m for n, m in self.model.named_modules() if isinstance(m, nn.Conv2d) and n}

    def transform_hw(self, img: Tensor) -> tuple[int, int]:
        """模型内部 transform 之后的输入尺寸，空间归因换算坐标用。"""
        with torch.no_grad():
            t = self.model.transform([img], None)[0].tensors  # type: ignore[operator]
        return int(t.shape[-2]), int(t.shape[-1])

    def fpn_info(self, img: Tensor) -> dict[str, str]:
        """backbone 输出键 -> P 层级名（按 stride 推断，跨模型可比）。

        Faster R-CNN 输出 P2 到 P6，RetinaNet / FCOS 输出 P3 到 P7，键名不同，
        尺度对齐必须按 stride，不能按键名。
        """
        t_h, t_w = self.transform_hw(img)
        with torch.no_grad():
            t = self.model.transform([img], None)[0].tensors  # type: ignore[operator]
            feats = self.model.backbone(t)  # type: ignore[operator]
        info: dict[str, str] = {}
        for key, f in feats.items():
            stride = round(t_h / f.shape[-2])
            info[str(key)] = f"P{round(math.log2(stride))}"
        return info


def load_detectors(
    names: Iterable[str], device: torch.device | str = "cpu"
) -> dict[str, TorchvisionDetector]:
    return {name: TorchvisionDetector(name, device) for name in names}


def shared_label_space(adapters: Iterable[TorchvisionDetector]) -> set[str]:
    """一次实验的标准标签空间：参与模型类别名的交集（占位项已剔除）。"""
    spaces = [a.label_names() for a in adapters]
    if not spaces:
        return set()
    out = spaces[0]
    for s in spaces[1:]:
        out = out & s
    return out
