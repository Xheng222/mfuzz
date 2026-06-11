"""异常单元探针：观测变异过程中归一化激活越出 [0, 1] 标定范围的覆盖单元。

变异把单元激活推出 profiling 标定范围，说明对抗输入到达了训练数据没覆盖的
激活区域（分类版报告里的"激活越界"诊断，这里成为独立探针）。逐轮记录越界
单元数当曲线，按越界幅度留 top-k 实例（含单元的结构位置），run_end 落
out/probes/ood_units/ood_units.json。
"""

from __future__ import annotations

import json
from typing import Any

from mfuzz.core.adapter import Batch, TaskForward
from mfuzz.core.probe import Probe, ProbeContext, register_probe
from mfuzz.core.records import RunReport

_TOP_K = 50


@register_probe("ood_units")
class OodUnitsProbe(Probe):
    def __init__(self) -> None:
        self._ctx: ProbeContext | None = None
        self._round_counts: list[int] = []
        self._current = 0
        self._tops: list[dict] = []

    def on_run_start(self, ctx: ProbeContext) -> None:
        self._ctx = ctx

    def on_mutation_done(self, rnd: int, batch: Batch, x_adv, fw: TaskForward) -> None:
        assert self._ctx is not None
        tracker = self._ctx.tracker
        norm = tracker.norm_all(fw.unit_acts.detach())  # (B, N)
        ood = (norm > 1.0) | (norm < 0.0)
        self._current += int(ood.any(dim=0).sum())
        # 按越界幅度留 top 实例
        excess = (norm - 1.0).clamp(min=0) + (-norm).clamp(min=0)
        flat = excess.amax(dim=0)
        k = min(5, int((flat > 0).sum()))
        if k > 0:
            top = flat.topk(k)
            profile = tracker.profile
            for val, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True):
                self._tops.append(
                    {
                        "round": rnd,
                        "unit": profile.unit_name(idx),
                        "excess": round(val, 4),
                        "image": str(batch.seeds[0].path or ""),
                    }
                )

    def on_round_end(self, rnd: int, stats: dict[str, float]) -> None:
        self._round_counts.append(self._current)
        self._current = 0

    def on_run_end(self, report: RunReport) -> dict[str, Any] | None:
        assert self._ctx is not None
        self._tops.sort(key=lambda d: -d["excess"])
        out = self._ctx.out_dir / "probes" / "ood_units"
        out.mkdir(parents=True, exist_ok=True)
        (out / "ood_units.json").write_text(
            json.dumps(self._tops[:_TOP_K], ensure_ascii=False, indent=1), encoding="utf-8"
        )
        total = sum(self._round_counts)
        return {
            "curves": {"n_ood_per_round": self._round_counts},
            "metrics": {"n_ood_total": float(total)},
        }
