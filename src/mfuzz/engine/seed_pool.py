from __future__ import annotations

import random
from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class Seed:
    image: Tensor
    label: int
    coverage_gain: int = 0
    fuzz_count: int = 0
    generation: int = 0


class SeedPool:
    def __init__(self, max_size: int = 5000):
        self.seeds: list[Seed] = []
        self.max_size = max_size

    def initialize(self, images: list[Tensor], labels: list[int]) -> None:
        for img, label in zip(images, labels):
            self.seeds.append(Seed(image=img, label=label))

    def select(self, batch_size: int) -> list[Seed]:
        if not self.seeds:
            raise RuntimeError("Seed pool is empty")
        n = min(batch_size, len(self.seeds))
        weights = [1.0 / (s.fuzz_count + 1) for s in self.seeds]
        selected = random.choices(self.seeds, weights=weights, k=n)
        for s in selected:
            s.fuzz_count += 1
        return selected

    def add(self, seed: Seed) -> None:
        if len(self.seeds) < self.max_size:
            self.seeds.append(seed)

    def __len__(self) -> int:
        return len(self.seeds)
