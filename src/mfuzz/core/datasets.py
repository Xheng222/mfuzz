from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as tvd
import torchvision.transforms.v2 as T


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT = _PROJECT_ROOT / "datasets"

_IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_CIFAR10_TRANSFORM = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])


def _imagenet_inverse_normalize() -> T.Normalize:
    return T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )


def load_dataset(
    name: str,
    split: str = "test",
    data_root: str | None = None,
    batch_size: int = 32,
    seed_size: int = 200,
    num_workers: int = 0,
) -> DataLoader:
    root = Path(data_root) if data_root else _DATA_ROOT / name

    if name == "cifar10":
        ds = _load_cifar10(root, split, seed_size)
    elif name == "imagenet":
        ds = _load_imagenet(root, split, seed_size)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )


def _load_cifar10(root: Path, split: str, seed_size: int):
    split_dir = "train" if split == "train" else "test"
    path = root / split_dir
    if not path.exists():
        raise FileNotFoundError(
            f"CIFAR-10 {split_dir} not found at {path}. "
            f"Expected ImageFolder layout: {path}/<class_name>/*.png"
        )
    ds = tvd.ImageFolder(str(path), transform=_CIFAR10_TRANSFORM)
    if split == "seed":
        ds = Subset(ds, list(range(min(seed_size, len(ds)))))
    return ds


def _load_imagenet(root: Path, split: str, seed_size: int):
    split_dir = "train" if split == "train" else "val"
    path = root / split_dir
    if not path.exists():
        raise FileNotFoundError(
            f"ImageNet {split_dir} not found at {path}. "
            f"Expected ImageFolder layout: {path}/<synset_id>/*.JPEG"
        )
    ds = tvd.ImageFolder(str(path), transform=_IMAGENET_TRANSFORM)
    if split == "seed":
        ds = Subset(ds, list(range(min(seed_size, len(ds)))))
    return ds
