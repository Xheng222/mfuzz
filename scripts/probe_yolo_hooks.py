"""验证 Ultralytics YOLO 的中间层是否可被 forward hook 访问（Day 2 探路脚本）。

差分 oracle 只需要模型的最终检测框，任何检测器都能当黑盒投票。但研究内容 2
（关键神经元 / 中间层覆盖）要求能拿到中间层激活，这就要求模型暴露可挂 hook 的
子模块。torchvision 的检测模型是标准 nn.Module，天然可挂；YOLO 经过 Ultralytics
封装，需要单独确认。

本脚本做三件事：
- 找到 YOLO 封装底下真正的 nn.Module（ultralytics 的 YOLO.model 是 DetectionModel）。
- 枚举其子模块，在 backbone / neck / head 各取若干层注册 forward hook。
- 跑一次前向，确认 hook 真的触发、能取到中间激活的形状。

结论用于判定 YOLO 能否进入结构层分析，还是只能当纯黑盒 oracle。

用法：
    uv run python scripts/probe_yolo_hooks.py --weights yolo11n.pt --image datasets/coco/val2017/<某张>.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"


def _resolve_weights(name: str) -> str:
    """权重集中放在项目 weights/ 下（git 忽略）。给的就是现成路径就直接用；
    否则在 weights/ 里按文件名找，找到用本地、避免重复下载；都没有就返回
    weights/ 下的目标路径，交给 ultralytics 下载到那里。"""
    given = Path(name)
    if given.exists():
        return str(given)
    local = _WEIGHTS_DIR / given.name
    if local.exists():
        return str(local)
    _WEIGHTS_DIR.mkdir(exist_ok=True)
    return str(local)


def _resolve_image(image: Path | None) -> str | None:
    """没给图就从 COCO val2017 里随手取一张；取不到就返回 None（让 YOLO 用自带样图）。"""
    if image is not None:
        return str(image)
    val = _PROJECT_ROOT / "datasets" / "coco" / "val2017"
    if val.exists():
        for p in sorted(val.glob("*.jpg")):
            return str(p)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="探测 Ultralytics YOLO 中间层是否可挂 hook")
    ap.add_argument("--weights", type=str, default="yolo11n.pt", help="YOLO 权重名或路径")
    ap.add_argument("--image", type=Path, default=None, help="测试图；默认从 COCO val2017 取一张")
    ap.add_argument("--max-layers", type=int, default=8, help="采样注册 hook 的中间层数量")
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics import __version__ as ul_ver

    print(f"ultralytics 版本：{ul_ver}")
    print(f"torch：{torch.__version__}  cuda 可用：{torch.cuda.is_available()}")

    weights = _resolve_weights(args.weights)
    yolo = YOLO(weights)
    print(f"加载权重：{weights}  封装类型：{type(yolo).__name__}")

    # 封装底下的真实网络。ultralytics 的 YOLO.model 即 DetectionModel(nn.Module)。
    net = yolo.model
    print(f"底层网络类型：{type(net).__name__}  是 nn.Module：{isinstance(net, torch.nn.Module)}")

    # 枚举叶子模块（无子模块的实际算子层）。注意 ultralytics 推理时会把 Conv+BN 融合，
    # 融合后 BatchNorm 被旁路、其 forward 不再触发，所以挂点取 Conv2d——它既是神经元覆盖
    # 真正关心的激活载体，也不受融合影响。
    leaves = [
        (name, m)
        for name, m in net.named_modules()
        if isinstance(m, torch.nn.Conv2d) and name
    ]
    total_modules = sum(1 for _ in net.named_modules())
    print(f"子模块总数：{total_modules}  Conv2d 叶子数：{len(leaves)}")

    # 在叶子里均匀采样若干层挂 hook，覆盖从浅到深。
    step = max(1, len(leaves) // args.max_layers)
    picked = leaves[::step][: args.max_layers]

    captured: dict[str, tuple[int, ...]] = {}

    def make_hook(layer_name: str):
        def hook(_module: torch.nn.Module, _inp, out) -> None:
            t = out[0] if isinstance(out, (tuple, list)) and out else out
            if isinstance(t, Tensor):
                captured[layer_name] = tuple(t.shape)

        return hook

    handles = [m.register_forward_hook(make_hook(name)) for name, m in picked]
    print(f"\n注册了 {len(handles)} 个 forward hook，准备前向：")
    for name, m in picked:
        print(f"  - {name}  ({type(m).__name__})")

    img = _resolve_image(args.image)
    print(f"\n测试图：{img if img else 'ultralytics 自带样图'}")

    # 跑一次预测触发前向。verbose=False 关掉逐张日志。
    yolo.predict(source=img, verbose=False, save=False)

    for h in handles:
        h.remove()

    print("\n" + "=" * 60)
    print(f"触发并捕获到激活的层：{len(captured)} / {len(picked)}")
    for name in (n for n, _ in picked):
        shape = captured.get(name)
        mark = "OK" if shape else "未触发"
        print(f"  [{mark}] {name:<40} {shape if shape else ''}")
    print("=" * 60)
    if len(captured) == len(picked) and picked:
        print("结论：YOLO 中间层可被 forward hook 访问，可进入结构层分析。")
    elif captured:
        print("结论：部分层可挂 hook（动态分支可能跳过个别层），中间层访问基本可行。")
    else:
        print("结论：未捕获到任何中间激活，需进一步排查或当作纯黑盒 oracle。")


if __name__ == "__main__":
    main()
