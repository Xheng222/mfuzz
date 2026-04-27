from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as tvd
import torchvision.transforms.v2 as T


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
    data_root: str = "./data",
    batch_size: int = 32,
    seed_size: int = 200,
    num_workers: int = 0,
) -> DataLoader:
    if name == "cifar10":
        is_train = split == "train"
        ds = tvd.CIFAR10(
            root=data_root,
            train=is_train,
            download=True,
            transform=_CIFAR10_TRANSFORM,
        )
        if split == "seed":
            ds = tvd.CIFAR10(
                root=data_root, train=False, download=True,
                transform=_CIFAR10_TRANSFORM,
            )
            indices = list(range(min(seed_size, len(ds))))
            ds = Subset(ds, indices)
    elif name == "imagenet":
        if split == "train":
            sp = "train"
        else:
            sp = "val"
        ds = tvd.ImageNet(root=data_root, split=sp, transform=_IMAGENET_TRANSFORM)
        if split == "seed":
            indices = list(range(min(seed_size, len(ds))))
            ds = Subset(ds, indices)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
