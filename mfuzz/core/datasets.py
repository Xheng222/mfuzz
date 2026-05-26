"""数据集加载。

当前数据集是 mini-ImageNet 子集（train/val/test，每个 split 是若干
synset 文件夹的 ImageFolder 布局）。关键约束：ImageFolder 给出的标签是
子集内按字母序的局部下标（0..N-1），与 torchvision 预训练模型的 1000 类
输出空间不一致。必须用 ImageNetLabel2Index.json 把 synset 映射到 1000 类
索引，否则真实标签与模型 argmax 永远对不上。本模块返回的标签一律是 1000
类索引。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision.datasets as tvd
import torchvision.transforms.v2 as T
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # mfuzz/core/datasets.py -> 项目根
_DATA_ROOT = _PROJECT_ROOT / "datasets"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_IMAGENET_TRANSFORM = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
)


def imagenet_denormalize(x: Tensor) -> Tensor:
    """把归一化张量还原回 [0, 1] 像素空间，供可视化与像素域约束使用。"""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def _load_label2index(root: Path) -> dict[str, int]:
    path = root / "ImageNetLabel2Index.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 synset->索引映射：{path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class ImageNetSubset(Dataset[tuple[Tensor, int]]):
    """mini-ImageNet 单个 split。__getitem__ 返回 (image, label)，label 为
    torchvision 1000 类索引。"""

    def __init__(self, root: Path, split: str) -> None:
        path = root / split
        if not path.exists():
            raise FileNotFoundError(
                f"未找到 {split} split：{path}（期望 ImageFolder 布局 <synset>/*.JPEG）"
            )
        self._folder = tvd.ImageFolder(str(path), transform=_IMAGENET_TRANSFORM)
        label_map = _load_label2index(root)
        idx_to_synset = {v: k for k, v in self._folder.class_to_idx.items()}
        # 子集局部下标 -> 1000 类索引
        self._local_to_model = {local: label_map[synset] for local, synset in idx_to_synset.items()}

    def __len__(self) -> int:
        return len(self._folder)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        img, local = self._folder[index]
        return img, self._local_to_model[local]


def load_imagenet(split: str = "val", data_root: str | None = None) -> ImageNetSubset:
    root = Path(data_root) if data_root else _DATA_ROOT / "imagenet"
    return ImageNetSubset(root, split)


def make_loader(
    dataset: Dataset[tuple[Tensor, int]],
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader[tuple[Tensor, int]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
