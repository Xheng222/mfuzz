"""检测差分 oracle 的最小可行验证（Day 2 探路脚本）。

加载三个异构的 torchvision COCO 检测模型，对同一张图做差分：

- Faster R-CNN（两阶段、anchor-based）
- RetinaNet（一阶段、anchor-based）
- FCOS（一阶段、anchor-free）

三者共享 COCO 91 类标签空间，标签可直接互比，满足差分前提。没有真值标注，
oracle 完全靠跨模型一致性：把各模型的检测框按 IoU 跨模型聚成簇，被多数模型
（>=2/3）以一致类别支持的簇算一个 consensus 对象，再按每个模型相对 consensus
的偏离判四类失效。

四类失效（相对 consensus，针对单个模型）：
- 漏检 miss：consensus 对象缺这个模型的框。
- 虚检 spurious：这个模型的框不被其它任何模型支持（只在自身的单点簇里）。
- 类别错误 cls：这个模型在 consensus 簇里有框，但类别和 consensus 多数票不一致。
- 定位偏移 loc：类别一致，但框与 consensus 代表框的 IoU 偏低（落在 [iou, loc) 带内）。

cls 与 loc 是两个独立标志，可同时为真（即 TIDE 的 both），不单列第五类。

用法：
    uv run python scripts/pilot_det_oracle.py --num-images 20
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torchvision.ops import box_iou
from torchvision.transforms.v2 import functional as TF

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_IMAGE_ROOT = _PROJECT_ROOT / "datasets" / "ImageNet" / "test"


# 名称 -> (工厂函数模块路径, 权重枚举)。延迟到加载时再取，避免顶部一次性导入。
def _build_models(device: torch.device) -> dict[str, torch.nn.Module]:
    from torchvision.models.detection import (
        FCOS_ResNet50_FPN_Weights,
        FasterRCNN_ResNet50_FPN_V2_Weights,
        RetinaNet_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
        fcos_resnet50_fpn,
        retinanet_resnet50_fpn_v2,
    )

    specs = {
        "faster_rcnn": (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
        "retinanet": (retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT),
        "fcos": (fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights.DEFAULT),
    }
    models: dict[str, torch.nn.Module] = {}
    categories: list[str] | None = None
    for name, (factory, weights) in specs.items():
        model = factory(weights=weights)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        models[name] = model
        cats = weights.meta["categories"]
        if categories is None:
            categories = cats
        elif cats != categories:
            raise RuntimeError(f"{name} 的类别空间与其它模型不一致，差分前提不成立")
    return models


@dataclass
class Detection:
    model: str
    box: Tensor  # (4,) xyxy
    label: int
    score: float


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

    def majority_label(self) -> int:
        votes = Counter(d.label for d in self.dets)
        top = max(votes.values())
        # 票数并列时取分数最高的那个
        cand = [lab for lab, c in votes.items() if c == top]
        if len(cand) == 1:
            return cand[0]
        best = max((d for d in self.dets if d.label in cand), key=lambda d: d.score)
        return best.label

    def representative_box(self) -> Tensor:
        """consensus 代表框：取多数类别里分数最高的框。"""
        lab = self.majority_label()
        return max((d for d in self.dets if d.label == lab), key=lambda d: d.score).box


def _load_image_tensor(path: Path, device: torch.device) -> Tensor:
    img = Image.open(path).convert("RGB")
    t = TF.to_image(img)
    t = TF.to_dtype(t, torch.float32, scale=True)  # [0,1] CxHxW
    return t.to(device)


_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.JPG", "*.PNG")


def _gather_images(root: Path, n: int) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"找不到图像目录：{root}")
    paths: list[Path] = []
    # 兼容两种布局：root 下直接是图片，或 root/<子目录>/图片
    for ext in _IMAGE_EXTS:
        paths += sorted(root.glob(ext))
    if not paths:
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for ext in _IMAGE_EXTS:
                paths += sorted(sub.glob(ext))
            if len(paths) >= n:
                break
    return paths[:n]


@torch.no_grad()
def _detect(model: torch.nn.Module, name: str, img: Tensor, score_thr: float) -> list[Detection]:
    out = model([img])[0]
    keep = out["scores"] >= score_thr
    boxes = out["boxes"][keep].cpu()
    labels = out["labels"][keep].cpu()
    scores = out["scores"][keep].cpu()
    return [
        Detection(name, boxes[i], int(labels[i]), float(scores[i])) for i in range(boxes.shape[0])
    ]


def _cluster(dets: list[Detection], iou_thr: float) -> list[Cluster]:
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
class ModelTally:
    miss: int = 0
    spurious: int = 0
    cls: int = 0
    loc: int = 0
    agree: int = 0  # 在 consensus 簇里且类别一致、定位也好


def run(image_dir: Path, num_images: int, score_thr: float, iou_thr: float, loc_thr: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    models = _build_models(device)
    names = list(models)
    paths = _gather_images(image_dir, num_images)
    print(f"图像：{len(paths)} 张，来自 {image_dir}")
    print(f"阈值：score>={score_thr}  iou>={iou_thr}  loc<{loc_thr}\n")

    tally = {n: ModelTally() for n in names}
    n_consensus = 0
    n_unanimous = 0  # 三模型都在、类别一致、定位都好

    for k, path in enumerate(paths):
        img = _load_image_tensor(path, device)
        dets: list[Detection] = []
        for name in names:
            dets += _detect(models[name], name, img, score_thr)
        clusters = _cluster(dets, iou_thr)

        for cl in clusters:
            if cl.support >= 2:
                # consensus 对象
                n_consensus += 1
                cons_label = cl.majority_label()
                rep = cl.representative_box()
                unanimous = True
                for name in names:
                    mine = [d for d in cl.dets if d.model == name]
                    if not mine:
                        tally[name].miss += 1
                        unanimous = False
                        continue
                    d = max(mine, key=lambda d: d.score)
                    if d.label != cons_label:
                        tally[name].cls += 1
                        unanimous = False
                    iou = float(box_iou(d.box[None], rep[None])[0, 0])
                    if d.label == cons_label and iou < loc_thr:
                        tally[name].loc += 1
                        unanimous = False
                    if d.label == cons_label and iou >= loc_thr:
                        tally[name].agree += 1
                if unanimous:
                    n_unanimous += 1
            else:
                # 单模型独有的框 -> 该模型虚检
                for d in cl.dets:
                    tally[d.model].spurious += 1

    _report(names, tally, n_consensus, n_unanimous, len(paths))


def _report(
    names: list[str], tally: dict[str, ModelTally], n_consensus: int, n_unanimous: int, n_img: int
) -> None:
    print("=" * 64)
    print(f"图像数 {n_img}    consensus 对象 {n_consensus}    三模型完全一致 {n_unanimous}")
    if n_consensus:
        print(f"完全一致占比 {n_unanimous / n_consensus:.1%}")
    print("-" * 64)
    print(f"{'模型':<14}{'漏检':>8}{'虚检':>8}{'类别错':>8}{'定位偏':>8}{'一致':>8}")
    for n in names:
        t = tally[n]
        print(f"{n:<14}{t.miss:>8}{t.spurious:>8}{t.cls:>8}{t.loc:>8}{t.agree:>8}")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="检测差分 oracle 最小验证")
    ap.add_argument(
        "--image-dir", type=Path, default=_DEFAULT_IMAGE_ROOT, help="图像目录（服务器上须显式指定）"
    )
    ap.add_argument("--num-images", type=int, default=20)
    ap.add_argument("--score-thr", type=float, default=0.5, help="单模型保留检测的分数下限")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="跨模型聚簇的 IoU 下限")
    ap.add_argument("--loc-thr", type=float, default=0.7, help="类别一致时判定位偏移的 IoU 下限")
    args = ap.parse_args()
    run(args.image_dir, args.num_images, args.score_thr, args.iou_thr, args.loc_thr)


if __name__ == "__main__":
    main()
