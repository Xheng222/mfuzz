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
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.datasets as tvd
import torchvision.transforms.v2 as T
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

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


def _mean_std(x: Tensor) -> tuple[Tensor, Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return mean, std


def imagenet_denormalize(x: Tensor) -> Tensor:
    """把归一化张量还原回 [0, 1] 像素空间，供可视化与像素域约束使用。"""
    mean, std = _mean_std(x)
    return (x * std + mean).clamp(0.0, 1.0)


def imagenet_normalize(x: Tensor) -> Tensor:
    """把 [0, 1] 像素张量归一化到模型输入空间。可微，供像素域 PGD 使用。"""
    mean, std = _mean_std(x)
    return (x - mean) / std


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

    def label_at(self, index: int) -> int:
        """不加载图像，直接给出该样本的 1000 类标签，供分组与切分。"""
        return self._local_to_model[self._folder.targets[index]]


def load_imagenet(split: str = "val", data_root: str | None = None) -> ImageNetSubset:
    root = Path(data_root) if data_root else _DATA_ROOT / "imagenet"
    return ImageNetSubset(root, split)


def _per_class_split(
    base: ImageNetSubset, val_fraction: float, random_seed: int
) -> tuple[list[int], list[int]]:
    """把一个 split 按类别切成 profiling 部分与种子部分。

    few-shot 的 mini-ImageNet 里 train/val/test 类别不相交，没法用 val 的类对上
    train 算出的类关键神经元。所以 mini 模式只用 train，在 train 内部按类切一刀，
    人为造一个 val：每类留出 val_fraction 比例当种子池，其余做 profiling。切分按
    类别分别进行、确定性（受 random_seed 控制），保证两边类别空间一致。
    """
    by_class: dict[int, list[int]] = {}
    for idx, local in enumerate(base._folder.targets):
        by_class.setdefault(local, []).append(idx)
    gen = torch.Generator().manual_seed(random_seed)
    profile_idx: list[int] = []
    seed_idx: list[int] = []
    for local in sorted(by_class):
        idxs = by_class[local]
        perm = torch.randperm(len(idxs), generator=gen).tolist()
        shuffled = [idxs[p] for p in perm]
        n_val = max(1, int(len(shuffled) * val_fraction))
        seed_idx.extend(shuffled[:n_val])
        profile_idx.extend(shuffled[n_val:])
    return profile_idx, seed_idx


def _group_by_label(base: ImageNetSubset, positions: list[int]) -> dict[int, list[int]]:
    """按 1000 类标签分组。返回 标签 -> 在 positions 列表里的下标（即子集内位置）。"""
    out: dict[int, list[int]] = {}
    for j, idx in enumerate(positions):
        out.setdefault(base.label_at(idx), []).append(j)
    return out


@dataclass
class DatasetBundle:
    """一次实验用到的数据划分。profile_set 做关键神经元 profiling，seed_set 出种子。"""

    profile_set: Dataset[tuple[Tensor, int]]  # profiling 全量数据
    seed_set: Dataset[tuple[Tensor, int]]  # 种子来源
    class_to_indices: dict[int, list[int]]  # 1000 类标签 -> profile_set 内位置

    def class_subset(self, label: int) -> Subset[tuple[Tensor, int]]:
        """取 profile_set 中某一类的全部样本，供逐类频率与归因统计。"""
        return Subset(self.profile_set, self.class_to_indices[label])


def build_dataset(
    name: str, val_fraction: float = 0.2, random_seed: int = 42, data_root: str | None = None
) -> DatasetBundle:
    """按数据集模式构建划分。

    mini-imagenet：只用 train，按类切出 profiling 与种子两部分，类别空间一致。
    imagenet：profiling 用完整 train，种子来自 val（两者共享 1000 类）。
    """
    root = Path(data_root) if data_root else _DATA_ROOT / "imagenet"
    if name == "mini-imagenet":
        train = ImageNetSubset(root, "train")
        profile_pos, seed_pos = _per_class_split(train, val_fraction, random_seed)
        profile_set = Subset(train, profile_pos)
        seed_set: Dataset[tuple[Tensor, int]] = Subset(train, seed_pos)
        class_to_indices = _group_by_label(train, profile_pos)
    elif name == "imagenet":
        train = ImageNetSubset(root, "train")
        profile_pos = list(range(len(train)))
        profile_set = Subset(train, profile_pos)
        seed_set = ImageNetSubset(root, "val")
        class_to_indices = _group_by_label(train, profile_pos)
    else:
        raise ValueError(f"未知数据集模式 {name!r}，可选 mini-imagenet | imagenet")
    return DatasetBundle(profile_set, seed_set, class_to_indices)


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
