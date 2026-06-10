"""探明检测模型的结构轴与归因梯度路径（Day 3 探路脚本）。

结构归因要把四类失效定位到 backbone 阶段、FPN 层级、head 分支、空间区域，
前提是知道每个模型里这些结构轴各自对应哪些层、有多少神经元，以及从检测输出
（类别分数 / 框坐标）反传的梯度实际能流回哪些轴。本脚本对三个 torchvision
模型和 YOLO 做三件事：

- 枚举 Conv2d / Linear 层并按结构轴分桶，报每桶的层数与通道规模。
- 跑一张真实图，报 FPN 各层级特征图的键名与空间形状（空间区域轴的载体）。
- 梯度连通性探测：挂 hook 取各桶代表层的激活，分别以最高检测分数和该框
  坐标为目标反传，看梯度流回了哪些桶。归因目标选类别分数还是 objectness、
  分支归因能不能分开，由这个结果直接回答。

用法：
    uv run python scripts/probe_det_structure.py --image datasets/coco/val2017/<某张>.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"

# 层名前缀 -> 结构轴。顺序即匹配优先级，第一个命中的生效。
# torchvision 的 ResNet-FPN 检测模型三者共享 backbone/fpn 的命名；
# rpn/roi 只在 Faster R-CNN 出现，head.* 只在 RetinaNet / FCOS 出现。
_BUCKETS: list[tuple[str, str]] = [
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

_LAYER_TYPES: tuple[type, ...] = (nn.Conv2d, nn.Linear)


def _bucket_of(name: str) -> str:
    for prefix, label in _BUCKETS:
        if name.startswith(prefix):
            return label
    return "other"


def _out_channels(m: nn.Module) -> int:
    if isinstance(m, nn.Conv2d):
        return m.out_channels
    if isinstance(m, nn.Linear):
        return m.out_features
    return 0


def _build_models(device: torch.device) -> dict[str, nn.Module]:
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        FCOS_ResNet50_FPN_Weights,
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
    models: dict[str, nn.Module] = {}
    for name, (factory, weights) in specs.items():
        model = factory(weights=weights)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        models[name] = model
    return models


def _resolve_image(image: Path | None) -> Path:
    if image is not None:
        return image
    val = _PROJECT_ROOT / "datasets" / "coco" / "val2017"
    for p in sorted(val.glob("*.jpg")):
        return p
    raise FileNotFoundError("没有可用测试图，请用 --image 指定")


def _load_image_tensor(path: Path, device: torch.device) -> Tensor:
    from PIL import Image
    from torchvision.transforms.v2 import functional as TF

    img = Image.open(path).convert("RGB")
    t = TF.to_image(img)
    t = TF.to_dtype(t, torch.float32, scale=True)
    return t.to(device)


# ---------- 第一部分：torchvision 模型的结构枚举与 FPN 层级 ----------


def report_structure(name: str, model: nn.Module) -> dict[str, list[str]]:
    """按结构轴分桶报层数与通道规模，返回每桶的层名清单供梯度探测取代表层。"""
    grouped: dict[str, list[str]] = {}
    modules = dict(model.named_modules())
    for lname, m in modules.items():
        if isinstance(m, _LAYER_TYPES) and lname:
            grouped.setdefault(_bucket_of(lname), []).append(lname)

    print(f"\n--- {name} 结构轴 ---")
    print(f"{'结构轴':<16}{'层数':>6}{'通道和':>10}  示例层名")
    for _, label in _BUCKETS:
        names = grouped.get(label)
        if not names:
            continue
        ch = sum(_out_channels(modules[n]) for n in names)
        span = names[0] if len(names) == 1 else f"{names[0]} .. {names[-1]}"
        print(f"{label:<16}{len(names):>6}{ch:>10}  {span}")
    if "other" in grouped:
        print(f"{'other':<16}{len(grouped['other']):>6}{'':>10}  {grouped['other'][:3]}")
    return grouped


def report_fpn_levels(name: str, model: nn.Module, img: Tensor) -> None:
    """跑 transform + backbone，报 FPN 输出各层级的键名与特征图形状。"""
    with torch.no_grad():
        images, _ = model.transform([img], None)  # type: ignore[operator]
        feats = model.backbone(images.tensors)  # type: ignore[operator]
    shapes = ", ".join(f"{k}:{tuple(v.shape)}" for k, v in feats.items())
    print(f"FPN 层级（输入 {tuple(images.tensors.shape)}）：{shapes}")


# ---------- 第二部分：梯度连通性探测 ----------


class _Capture:
    """每个代表层把所有前向调用的输出都记下来（head 卷积逐 FPN 层级调用多次）。"""

    def __init__(self) -> None:
        self.acts: dict[str, list[Tensor]] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, modules: dict[str, nn.Module]) -> None:
        for lname, m in modules.items():
            self.acts[lname] = []

            def hook(_m: nn.Module, _i: tuple, out: Tensor, lname: str = lname) -> None:
                if isinstance(out, Tensor):
                    self.acts[lname].append(out)

            self.handles.append(m.register_forward_hook(hook))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _grad_status(target: Tensor, acts: dict[str, list[Tensor]]) -> dict[str, str]:
    """对每个代表层报梯度状态：断（None）/ 零 / 范数。多次调用取任意非 None。"""
    flat: list[Tensor] = [t for ts in acts.values() for t in ts]
    grads = torch.autograd.grad(target, flat, allow_unused=True, retain_graph=True)
    status: dict[str, str] = {}
    idx = 0
    for lname, ts in acts.items():
        norms = []
        for _ in ts:
            g = grads[idx]
            idx += 1
            if g is not None:
                norms.append(float(g.norm()))
        if not norms:
            status[lname] = "断"
        elif max(norms) == 0.0:
            status[lname] = "零"
        else:
            status[lname] = f"{max(norms):.2e}"
    return status


def probe_gradients(
    name: str, model: nn.Module, grouped: dict[str, list[str]], img: Tensor
) -> None:
    """以最高分检测的类别分数与框坐标为目标反传，看梯度流回哪些结构轴。"""
    modules = dict(model.named_modules())
    # 每个结构轴取最深一层当代表（stem/fpn.inner 这类单层桶就是它自己）
    reps: dict[str, nn.Module] = {}
    for label, names in grouped.items():
        if label != "other":
            reps[f"{label}::{names[-1]}"] = modules[names[-1]]
    cap = _Capture()
    cap.attach(reps)
    x = img.clone().requires_grad_(True)
    try:
        out = model([x])[0]
    finally:
        cap.remove()

    scores: Tensor = out["scores"]
    boxes: Tensor = out["boxes"]
    if scores.numel() == 0:
        print(f"{name}: 这张图没有检出，换图重试")
        return
    top = int(scores.argmax())
    print(f"\n--- {name} 梯度连通性（最高分检测 score={float(scores[top].detach()):.3f}）---")

    targets: dict[str, Tensor] = {"<-类别分数": scores[top], "<-框坐标": boxes[top].sum()}
    # Faster R-CNN 的 proposal 在 RPN 出口被 detach，最终分数的梯度回不到 RPN。
    # RPN 的归因要用它自己的 objectness 当目标，这里直接拿 hook 抓到的
    # cls_logits 输出（逐 FPN 层级各一份）验证这条路是否连通。
    rpn_key = next((k for k in cap.acts if k.startswith("rpn.cls::")), None)
    if rpn_key and cap.acts[rpn_key]:
        targets["<-objectness"] = torch.cat([t.flatten() for t in cap.acts[rpn_key]]).max()

    columns: dict[str, dict[str, str]] = {}
    for tname, t in targets.items():
        try:
            columns[tname] = _grad_status(t, cap.acts)
        except RuntimeError as e:
            print(f"{tname} 反传失败：{e}")
            columns[tname] = {}
    print(f"{'结构轴':<16}" + "".join(f"{tname:>14}" for tname in columns))
    for key in cap.acts:
        label = key.split("::")[0]
        row = "".join(f"{col.get(key, '-'):>14}" for col in columns.values())
        print(f"{label:<16}{row}")


# ---------- 第三部分：YOLO 结构与梯度 ----------


def _resolve_weights(name: str) -> str:
    given = Path(name)
    if given.exists():
        return str(given)
    local = _WEIGHTS_DIR / given.name
    if local.exists():
        return str(local)
    _WEIGHTS_DIR.mkdir(exist_ok=True)
    return str(local)


def probe_yolo(weights_name: str, device: torch.device) -> None:
    from ultralytics import YOLO

    yolo = YOLO(_resolve_weights(weights_name))
    net = yolo.model.to(device).eval()  # DetectionModel
    for p in net.parameters():
        p.requires_grad_(False)

    n_backbone = len(net.yaml["backbone"])  # yaml 里 backbone 与 head（neck+检测头）的分界
    n_top = len(net.model)
    print(f"\n--- yolo（{weights_name}）结构轴 ---")
    print(f"顶层模块 {n_top} 个，yaml backbone 0..{n_backbone - 1}，head {n_backbone}..{n_top - 1}")
    print(f"{'idx':<5}{'区段':<10}{'类型':<16}{'Conv2d数':>8}{'通道和':>8}")
    detect = None
    for m in net.model:
        convs = [mm for mm in m.modules() if isinstance(mm, nn.Conv2d)]
        seg = "backbone" if m.i < n_backbone else "neck"
        ty = m.type.rsplit(".", 1)[-1]
        if ty == "Detect":
            seg = "head"
            detect = m
        ch = sum(c.out_channels for c in convs)
        print(f"{m.i:<5}{seg:<10}{ty:<16}{len(convs):>8}{ch:>8}")

    assert detect is not None, "没找到 Detect 头"
    strides = [int(s) for s in detect.stride]
    levels = [s.bit_length() - 1 for s in strides]
    print(f"Detect 头：{len(strides)} 个尺度，stride {strides}（即 P{levels[0]}..P{levels[-1]}）")
    for bname, branch, role in (("cv2", detect.cv2, "回归分支"), ("cv3", detect.cv3, "分类分支")):
        for si, seq in enumerate(branch):
            convs = [mm for mm in seq.modules() if isinstance(mm, nn.Conv2d)]
            tag = f"model.{detect.i}.{bname}[{si}]"
            print(f"  {tag}（{role}，stride {strides[si]}）：Conv2d {len(convs)} 个")

    # 梯度连通性：backbone 末层、neck 中段、Detect 两分支各取一个代表 Conv2d。
    # 直接调 net 前向（不走 predict，避免 no_grad 与 Conv+BN 融合），eval 下
    # Detect 返回 (拼接预测 (1, 4+nc, anchors), 各尺度特征)。
    def last_conv(m: nn.Module) -> nn.Conv2d:
        return [mm for mm in m.modules() if isinstance(mm, nn.Conv2d)][-1]

    reps = {
        f"backbone(model.{n_backbone - 1})": last_conv(net.model[n_backbone - 1]),
        f"neck(model.{detect.i - 1})": last_conv(net.model[detect.i - 1]),
        f"head.reg(model.{detect.i}.cv2[0])": last_conv(detect.cv2[0]),
        f"head.cls(model.{detect.i}.cv3[0])": last_conv(detect.cv3[0]),
    }
    cap = _Capture()
    cap.attach(reps)
    x = torch.rand(1, 3, 640, 640, device=device, requires_grad=True)
    try:
        preds = net(x)
    finally:
        cap.remove()
    pred = preds[0] if isinstance(preds, (tuple, list)) else preds  # (1, 4+nc, N)
    nc = pred.shape[1] - 4
    print(f"eval 前向输出：{tuple(pred.shape)}（4 框坐标 + {nc} 类分数 × anchor 数）")
    t_cls = pred[:, 4:, :].max()
    t_box = pred[:, :4, :].sum()
    s_cls = _grad_status(t_cls, cap.acts)
    s_box = _grad_status(t_box, cap.acts)
    print(f"{'结构轴':<28}{'<-类别分数':>14}{'<-框坐标':>14}")
    for key in cap.acts:
        print(f"{key:<28}{s_cls.get(key, '-'):>14}{s_box.get(key, '-'):>14}")


def main() -> None:
    ap = argparse.ArgumentParser(description="探明检测模型结构轴与归因梯度路径")
    ap.add_argument("--image", type=Path, default=None, help="测试图；默认取 COCO val2017 第一张")
    ap.add_argument("--yolo-weights", type=str, default="yolo11n.pt")
    ap.add_argument("--skip-yolo", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = _resolve_image(args.image)
    print(f"设备：{device}  测试图：{img_path.name}")

    models = _build_models(device)
    img = _load_image_tensor(img_path, device)
    for name, model in models.items():
        grouped = report_structure(name, model)
        report_fpn_levels(name, model, img)
        probe_gradients(name, model, grouped, img)

    if not args.skip_yolo:
        probe_yolo(args.yolo_weights, device)


if __name__ == "__main__":
    main()
