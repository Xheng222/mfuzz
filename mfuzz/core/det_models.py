"""检测模型注册与家族适配层。

每个模型家族一个适配器，统一负责四件事：加载与权重冻结、前向出框（把本地
标签索引翻成类别名）、暴露可挂 hook 的结构（结构桶映射、FPN 尺度）、给出
变换后的输入尺寸。oracle 与归因层只依赖适配器接口，不接触家族差异。

两个家族：torchvision（Faster R-CNN / RetinaNet / FCOS，COCO 91 类含占位项）
与 ultralytics YOLO（yolo11 系列，COCO 80 类无占位）。跨家族差分必须按类别
名比，索引约定不同。两个家族都实现了 forward_graph（带图前向）与 level_zero_hook
（层级消融），都能当轮换目标做覆盖标定与结构归因；detect 则供黑盒差分投票。
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms

from mfuzz.core.types import Detection

# YOLO 带图前向的解码常量：letterbox 到 _YOLO_IMGSZ（保持长宽比、右下补 _YOLO_PAD
# 灰边、补到 stride 整数倍），候选先按 _YOLO_CONF_FLOOR 粗筛再按 _YOLO_NMS_IOU 做
# 类内 NMS，最多保留 _YOLO_MAX_DET 个。floor 取低值（与 torchvision 内部 0.05 一致），
# 保留低分候选供漏检归因。
_YOLO_IMGSZ = 640
_YOLO_PAD = 114 / 255
_YOLO_CONF_FLOOR = 0.05
_YOLO_NMS_IOU = 0.7
_YOLO_MAX_DET = 300

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

    # ---- 带图前向 ----

    def forward_graph(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, list[str], dict[str, Tensor]]:
        """一次带计算图的前向，抓 FPN 各层级特征，返回全部输出（含低分候选）。

        x 已带梯度。boxes/scores 是模型内部阈值（约 0.05）以上的全部输出，已还原
        到原图坐标系；卷积激活由调用方 GraphForward 的通用 hook 抓取。
        """
        feats: dict[str, Tensor] = {}

        def fpn_hook(_m: nn.Module, _i: tuple, out: dict[str, Tensor]) -> None:
            feats.update(out)

        handle = self.model.backbone.register_forward_hook(fpn_hook)  # type: ignore[operator]
        try:
            out = self.model([x])[0]
        finally:
            handle.remove()
        boxes_g = out["boxes"]
        scores_g = out["scores"]
        labels = [self.to_name(int(i)) for i in out["labels"].detach().cpu()]
        return boxes_g, scores_g, labels, feats

    def level_zero_hook(self, key: str) -> torch.utils.hooks.RemovableHandle:
        """层级消融：把 backbone 输出的 FPN 层级 key 置零的前向 hook。"""

        def zero_hook(_m: nn.Module, _i: tuple, out: dict[str, Tensor], key: str = key):
            out[key] = torch.zeros_like(out[key])
            return out

        return self.model.backbone.register_forward_hook(zero_hook)  # type: ignore[operator]

    def letterbox_pad(self, img: Tensor) -> None:
        """transform 左上对齐、补边在右下，归因按 orig/t 比例换算即可，无需偏移。"""
        del img
        return None


# yolo11 的层序号 -> 结构桶。来自服务器上对 yolo11n 的结构探查（2026-06-12）：
# backbone 0-10（stride 翻倍点在 1/3/5/7），neck 自顶向下 11-16、自底向上 17-22，
# Detect 头在 23。neck 两段语义上对应 torchvision 的 fpn.inner/fpn.layer，但融合
# 方向不同，单独命名不混桶。
_YOLO_IDX_BUCKETS: dict[int, str] = {
    0: "backbone.stem",
    **dict.fromkeys((1, 2), "backbone.C2"),
    **dict.fromkeys((3, 4), "backbone.C3"),
    **dict.fromkeys((5, 6), "backbone.C4"),
    **dict.fromkeys((7, 8, 9, 10), "backbone.C5"),
    **dict.fromkeys(range(11, 17), "neck.td"),
    **dict.fromkeys(range(17, 23), "neck.bu"),
}


def yolo_bucket(layer_name: str) -> str:
    """yolo11 层名 -> 结构桶（model.<idx>.… 风格）。"""
    parts = layer_name.split(".")
    if len(parts) < 2 or not parts[1].isdigit():
        return "other"
    idx = int(parts[1])
    if idx == 23:  # Detect 头：cv2 回归分支、cv3 分类分支、dfl 框分布解码
        if len(parts) > 2 and parts[2] == "cv3":
            return "head.cls"
        return "head.reg"
    return _YOLO_IDX_BUCKETS.get(idx, "other")


class YoloDetector:
    """ultralytics YOLO 的适配器。

    detect 与 forward_graph 共用同一套张量管线：可微的居中 letterbox、图内解码、NMS。
    detect 在 no_grad 下跑、按阈值筛，当差分投票者；forward_graph 保留计算图，让 YOLO
    当轮换目标做覆盖标定与结构归因。两者前处理与解码完全一致，框架里不再调用 ultralytics
    的 predict。结构接口（conv_layers / bucket_of / fpn_info）按层序号映射结构桶（见
    yolo_bucket）。

    权重找 weights/<name>.pt（git 忽略），不存在时由 ultralytics 下载到该处。
    """

    family = "ultralytics"

    def __init__(self, name: str, device: torch.device | str = "cpu") -> None:
        from pathlib import Path

        from ultralytics import YOLO

        weights = Path("weights") / f"{name}.pt"
        weights.parent.mkdir(exist_ok=True)
        self.name = name
        self.device = torch.device(device)
        self.yolo = YOLO(str(weights))
        self.yolo.to(self.device)
        self.model: nn.Module = self.yolo.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.names: dict[int, str] = dict(self.yolo.names)

    # ---- 标签空间 ----

    def label_names(self) -> set[str]:
        return set(self.names.values())

    def to_name(self, idx: int) -> str:
        return self.names[idx]

    # ---- 前向 ----

    @torch.no_grad()
    def detect(self, img: Tensor, score_thr: float) -> list[Detection]:
        """单图前向。与 forward_graph 同一套张量管线（可微 letterbox、图内解码、NMS），
        只是不留计算图、按 score_thr 筛。YOLO 在框架里任何角色都走这一条，不再调用
        ultralytics 的 predict，保证投票、当目标、归因前后用的是同一套预处理与解码。"""
        boxes_g, scores_g, labels, _ = self.forward_graph(img)
        keep = (scores_g >= score_thr).nonzero(as_tuple=True)[0]
        return [
            Detection(self.name, boxes_g[int(i)].cpu(), labels[int(i)], float(scores_g[int(i)]))
            for i in keep
        ]

    # ---- 结构 ----

    def bucket_of(self, layer_name: str) -> str:
        return yolo_bucket(layer_name)

    def conv_layers(self) -> dict[str, nn.Conv2d]:
        return {n: m for n, m in self.model.named_modules() if isinstance(m, nn.Conv2d) and n}

    def fpn_info(self, img: Tensor) -> dict[str, str]:
        """Detect 头消费层 16/19/22，stride 8/16/32 即 P3/P4/P5（探查确认）。"""
        del img
        head = self.model.model[-1]  # type: ignore[index]
        levels = [f"P{round(math.log2(float(s)))}" for s in head.stride]
        return {str(f): lv for f, lv in zip(head.f, levels, strict=True)}

    # ---- 带图前向 ----

    def _letterbox(self, h: int, w: int) -> tuple[float, int, int, int, int, int, int]:
        """letterbox 尺寸：缩放比 r、缩放后 (nh, nw)、补到 32 整数倍后的 (ph, pw)、
        以及居中补边的上、左偏移 (top, left)。

        保持长宽比，长边缩到 _YOLO_IMGSZ，灰边居中补到 32 整数倍，几何上与 ultralytics
        官方 letterbox 一致（YOLO 训练时见的就是居中补边）。框还原需先减偏移再除以 r。
        """
        r = _YOLO_IMGSZ / max(h, w)
        nh, nw = round(h * r), round(w * r)
        ph = math.ceil(nh / 32) * 32
        pw = math.ceil(nw / 32) * 32
        top = round((ph - nh) / 2 - 0.1)
        left = round((pw - nw) / 2 - 0.1)
        return r, nh, nw, ph, pw, top, left

    def transform_hw(self, img: Tensor) -> tuple[int, int]:
        """letterbox 补边后的输入尺寸，归因换算尺度用（与 forward_graph 一致）。"""
        _, _, _, ph, pw, _, _ = self._letterbox(int(img.shape[-2]), int(img.shape[-1]))
        return ph, pw

    def letterbox_pad(self, img: Tensor) -> tuple[float, float, float]:
        """归因把空间峰值换算回原图用：返回 (left, top, r)，原图坐标 = (padded - 偏移) / r。"""
        r, _, _, _, _, top, left = self._letterbox(int(img.shape[-2]), int(img.shape[-1]))
        return float(left), float(top), r

    def forward_graph(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, list[str], dict[str, Tensor]]:
        """一次带计算图的前向。letterbox 预处理可微，模型 eval 解码出 xywh 与 sigmoid
        类分，换算回原图坐标系后按类内 NMS 选框，返回的 boxes/scores 含低分候选。

        feats 为 Detect 头消费的多尺度特征（层 head.f，即 P3/P4/P5），键与 fpn_info
        对齐，供责任尺度归因。卷积激活由调用方 GraphForward 的通用 hook 抓取。
        """
        h0, w0 = int(x.shape[-2]), int(x.shape[-1])
        r, nh, nw, ph, pw, top, left = self._letterbox(h0, w0)
        resized = F.interpolate(x[None], size=(nh, nw), mode="bilinear", align_corners=False)
        padded = F.pad(resized, (left, pw - nw - left, top, ph - nh - top), value=_YOLO_PAD)

        head = self.model.model[-1]  # type: ignore[index]
        feats: dict[str, Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
        for i in [int(f) for f in head.f]:

            def feat_hook(_m: nn.Module, _i: tuple, out: Tensor, key: str = str(i)) -> None:
                if isinstance(out, Tensor):
                    feats[key] = out

            handles.append(self.model.model[i].register_forward_hook(feat_hook))  # type: ignore[index]
        try:
            out = self.model(padded)
        finally:
            for hd in handles:
                hd.remove()

        preds = out[0] if isinstance(out, (tuple, list)) else out  # (1, 4+nc, N)
        p = preds[0]  # (4+nc, N)
        box = p[:4].transpose(0, 1)  # (N, 4) xywh，letterbox 像素
        cls = p[4:].transpose(0, 1)  # (N, nc) sigmoid 类分
        conf, cls_idx = cls.max(dim=1)
        cx, cy, bw, bh = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        x1 = (((cx - bw / 2) - left) / r).clamp(0, w0)
        y1 = (((cy - bh / 2) - top) / r).clamp(0, h0)
        x2 = (((cx + bw / 2) - left) / r).clamp(0, w0)
        y2 = (((cy + bh / 2) - top) / r).clamp(0, h0)
        xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # (N, 4) 原图坐标

        floor = (conf >= _YOLO_CONF_FLOOR).nonzero(as_tuple=True)[0]
        if floor.numel():
            keep = batched_nms(
                xyxy[floor].detach(), conf[floor].detach(), cls_idx[floor], _YOLO_NMS_IOU
            )
            keep = floor[keep[:_YOLO_MAX_DET]]
        else:
            keep = floor
        boxes_g = xyxy[keep]
        scores_g = conf[keep]
        labels = [self.to_name(int(c)) for c in cls_idx[keep].detach().cpu()]
        return boxes_g, scores_g, labels, feats

    def level_zero_hook(self, key: str) -> torch.utils.hooks.RemovableHandle:
        """层级消融：把 Detect 头消费层 key 的输出置零的前向 hook。"""

        def zero_hook(_m: nn.Module, _i: tuple, out):
            return torch.zeros_like(out) if isinstance(out, Tensor) else out

        return self.model.model[int(key)].register_forward_hook(zero_hook)  # type: ignore[index]


AnyDetector = TorchvisionDetector | YoloDetector

_TV_NAMES = {"faster_rcnn", "retinanet", "fcos"}


def load_detectors(
    names: Iterable[str], device: torch.device | str = "cpu"
) -> dict[str, AnyDetector]:
    out: dict[str, AnyDetector] = {}
    for name in names:
        if name in _TV_NAMES:
            out[name] = TorchvisionDetector(name, device)
        elif name.startswith("yolo"):
            out[name] = YoloDetector(name, device)
        else:
            raise ValueError(
                f"未知检测模型 {name!r}：torchvision 可用 {sorted(_TV_NAMES)}，YOLO 用 yolo* 命名"
            )
    return out


def shared_label_space(adapters: Iterable[AnyDetector]) -> set[str]:
    """一次实验的标准标签空间：参与模型类别名的交集（占位项已剔除）。"""
    spaces = [a.label_names() for a in adapters]
    if not spaces:
        return set()
    out = spaces[0]
    for s in spaces[1:]:
        out = out & s
    return out
