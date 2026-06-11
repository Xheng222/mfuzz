"""分类任务适配器。

把原 engine/runner.py 的分类语义装配到统一协议后面，机制模块原样复用：
共识过滤（differential/consensus）、关键神经元 profiling（neurons/profiler，
频率 + 归因融合关键度）、四类分流（differential/triage）、缺陷聚类与分类版
图表（neurons/cluster、evaluate/report、evaluate/metrics）。

统一表示下的退化形态：锚点是共识标签（box 为 None，锚是整图），失效只产生
kind="cls" 的标签翻转记录；批策略是把同尺寸样本堆成一个大批。类感知的
U 选择与 CCCov 留在分类覆盖子类（ClsCoverageTracker）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from mfuzz.core.adapter import Batch, SeedStats, TaskAdapter, TaskForward
from mfuzz.core.datasets import build_dataset, imagenet_denormalize, imagenet_normalize, make_loader
from mfuzz.core.hooks import ActivationExtractor
from mfuzz.core.models import load_ensemble
from mfuzz.core.records import ConsensusAnchor, FailureRecord, RunReport, Seed
from mfuzz.core.seed import build_seed_pool
from mfuzz.core.types import DefectRecord, FuzzReport
from mfuzz.differential.consensus import filter_consensus
from mfuzz.differential.ensemble import Ensemble
from mfuzz.differential.objective import differential_objective
from mfuzz.differential.triage import Verdict, triage
from mfuzz.neurons.profiler import build_profile, flatten_acts
from mfuzz.neurons.unit_coverage import Layout, UnitCoverageTracker, UnitProfile
from mfuzz.semantic.feature import feature_layer
from mfuzz.semantic.path import s_path

_CLASS_PENALTY = 1e4  # 类无关单元的打分惩罚，类相关候选不足时才入选


@dataclass
class ClsParams:
    """分类任务自有参数，从 cfg.raw["classification"] 解析。"""

    dataset: str = "mini-imagenet"
    seed_size: int = 200
    val_fraction: float = 0.2
    batch_size: int = 32
    alpha: float = 0.5  # 关键度融合权重（频率 vs 归因）
    p_low: float = 0.0
    p_high: float = 100.0
    class_critical_threshold: float = 0.9
    cache_dir: str = "output/profiles"
    theta_path: float = 0.0  # 路径新颖阈值；0 = 关闭

    @classmethod
    def from_raw(cls, raw: dict) -> ClsParams:
        return cls(**raw.get("classification", {}))


class ClsCoverageTracker(UnitCoverageTracker):
    """分类覆盖子类：全 N 覆盖账（供 CCCov）、类感知 U 选择、类缺口调度项。"""

    def __init__(self, profile: UnitProfile, t_cov, device, critical_per_class) -> None:
        super().__init__(profile, t_cov, device)
        self.covered_all = torch.zeros(profile.num_units, dtype=torch.bool, device=device)
        self.per_class = {c: m.to(device) for c, m in critical_per_class.items()}
        # 类关键掩码投影到全局关键下标轴，U 选择用
        self.class_on_crit = {c: m[self.crit_idx] for c, m in self.per_class.items()}

    def update(self, v: Tensor) -> int:
        fired = (self.norm_all(v.detach()) > self.t_cov).any(dim=0)
        self.covered_all |= fired
        return super().update(v)

    def cccov(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for c, mask in self.per_class.items():
            n = int(mask.sum())
            out[c] = float(self.covered_all[mask].float().mean()) if n else 0.0
        return out

    def gap_for(self, seed: Seed) -> float:
        c = int(seed.extra.get("consensus_label", -1))
        return 1.0 - self.cccov().get(c, 1.0)

    def select_u(self, seeds: list[Seed], v0: Tensor, u_size: int) -> Tensor | None:
        """类感知 U 选择：未覆盖且类相关、接近阈值的优先。返回 (B, K) bool 掩码。"""
        norm = self.norm_critical(v0.detach())  # (B, K)
        b, k = norm.shape
        uncovered = ~self.covered
        ones = torch.ones(k, dtype=torch.bool, device=self.device)
        class_mask = torch.stack(
            [self.class_on_crit.get(int(s.extra.get("consensus_label", -1)), ones) for s in seeds]
        )
        near = -(norm - self.t_cov).abs()
        tier1 = uncovered.unsqueeze(0) & class_mask
        tier2 = uncovered.unsqueeze(0) & ~class_mask
        score = near.clone()
        score[tier2] -= _CLASS_PENALTY
        score[~(tier1 | tier2)] = float("-inf")
        top = score.topk(min(u_size, k), dim=1)
        mask = torch.zeros(b, k, dtype=torch.bool, device=self.device)
        mask.scatter_(1, top.indices, top.values > float("-inf"))
        return mask if bool(mask.any()) else None

    def objective(self, v: Tensor, u_idx: Tensor) -> Tensor:
        """obj_cov：U 掩码内归一化激活求和。u_idx 是 select_u 的 (B, K) 掩码。"""
        return (self.norm_critical(v) * u_idx).sum()


class ClassificationAdapter(TaskAdapter):
    task = "classification"

    def __init__(self, cfg, target, device, out_dir) -> None:
        super().__init__(cfg, target, device, out_dir)
        self.p = ClsParams.from_raw(cfg.raw)
        self._cccov_hist: list[dict[int, float]] = []
        self._refs_hold = 0
        self._n_judged = 0

    # ---- 构建 ----

    def setup(self) -> None:
        names = self.cfg.models.names
        self.ensemble = Ensemble(load_ensemble(names, self.device), self.target)
        self.target_model = self.ensemble.models[self.target]
        self.ref_models = {n: self.ensemble.models[n] for n in self.ensemble.references}
        self.extractor = ActivationExtractor(self.target_model)
        self.bundle = build_dataset(self.p.dataset, self.p.val_fraction, self.cfg.random_seed)

    def build_seeds(self) -> list[Seed]:
        raw_seeds = build_seed_pool(
            self.bundle.seed_set, self.p.seed_size, self.device, self.cfg.random_seed
        )
        cons = filter_consensus(self.ensemble, raw_seeds, batch_size=self.p.batch_size)
        self._acceptance = cons.acceptance_rate
        seeds = []
        for s in cons.seeds:
            seeds.append(
                Seed(
                    anchors=[
                        ConsensusAnchor(label=str(s.consensus_label), support=s.model_confidences)
                    ],
                    image=s.image,
                    extra={"consensus_label": s.consensus_label, "true_label": s.true_label},
                )
            )
        self._consensus_classes = sorted({int(s.extra["consensus_label"]) for s in seeds})
        return seeds

    def build_tracker(self) -> ClsCoverageTracker:
        classes = [c for c in self._consensus_classes if c in self.bundle.class_to_indices]
        bs = self.p.batch_size
        profile_loader = make_loader(self.bundle.profile_set, batch_size=bs, shuffle=False)
        class_loaders = {
            c: make_loader(self.bundle.class_subset(c), batch_size=bs, shuffle=False)
            for c in classes
        }
        profile = build_profile(
            self.target_model,
            self.target,
            profile_loader,
            class_loaders,
            t=self.cfg.coverage.t_freq,
            tau=self.cfg.coverage.critical_tau,
            tau_class=self.p.class_critical_threshold,
            alpha=self.p.alpha,
            p_low=self.p.p_low,
            p_high=self.p.p_high,
            dataset_name=self.p.dataset,
            val_fraction=self.p.val_fraction,
            cache_dir=self.p.cache_dir,
            device=self.device,
        )
        self.layers = profile.layers
        self.feat = feature_layer(self.layers)
        # 布局：profiling 的扁平顺序按 layers 序拼接，通道数从一次提取读出
        with torch.no_grad():
            sample = self.bundle.profile_set[0][0][None].to(self.device)
            acts = self.extractor.extract(sample)
        layout: Layout = [(name, 0, int(acts[name].shape[1])) for name in self.layers]
        uprofile = UnitProfile(
            layout=layout,
            low=profile.low,
            high=profile.high,
            critical_idx=profile.critical.nonzero(as_tuple=True)[0],
        )
        self.tracker = ClsCoverageTracker(
            uprofile, self.cfg.coverage.t_cov, self.device, profile.critical_per_class
        )
        return self.tracker

    # ---- 循环内 ----

    def make_batches(self, seeds: list[Seed]) -> list[Batch]:
        # 同尺寸样本堆成一个大批；x0 是像素域（变异域），归一化在 forward 内
        x_norm = torch.stack([s.image for s in seeds if s.image is not None]).to(self.device)
        c = torch.tensor([int(s.extra["consensus_label"]) for s in seeds], device=self.device)
        return [Batch(seeds=seeds, x0=imagenet_denormalize(x_norm), meta={"c": c})]

    def forward(self, x: Tensor, batch: Batch) -> TaskForward:
        xn = imagenet_normalize(x)
        logits, acts = self.extractor.forward_with_acts(xn)
        return TaskForward(
            unit_acts=flatten_acts(acts, self.layers),
            v_sem=acts[self.feat],
            raw={"logits": logits, "xn": xn},
        )

    def objective1(self, fw: TaskForward, batch: Batch) -> Tensor:
        probs = {self.target: torch.softmax(fw.raw["logits"], dim=1)}
        for name, m in self.ref_models.items():
            probs[name] = torch.softmax(m(fw.raw["xn"]), dim=1)
        return differential_objective(probs, self.target, batch.meta["c"]).sum()

    def judge(
        self,
        x_adv: Tensor,
        fw0: TaskForward,
        fw: TaskForward,
        batch: Batch,
        s_input: Tensor,
        round_idx: int,
    ) -> tuple[list[FailureRecord], list[SeedStats]]:
        with torch.no_grad():
            labels = {self.target: fw.raw["logits"].argmax(dim=1)}
            for name, m in self.ref_models.items():
                labels[name] = m(fw.raw["xn"]).argmax(dim=1)
            sp = s_path(
                self.tracker.norm_critical(fw.unit_acts), self.tracker.norm_critical(fw0.unit_acts)
            )
            pert = (x_adv - batch.x0).abs().flatten(1).amax(dim=1)
            adv_crit = self.tracker.norm_critical(fw.unit_acts)

        records: list[FailureRecord] = []
        stats: list[SeedStats] = []
        gamma = self.cfg.semantic.gamma_input
        for i, seed in enumerate(batch.seeds):
            c = int(batch.meta["c"][i])
            tgt = int(labels[self.target][i])
            refs = [int(labels[n][i]) for n in self.ref_models]
            si = float(s_input[i])
            novel = self.p.theta_path > 0.0 and float(sp[i]) < self.p.theta_path
            verdict = triage(tgt, refs, c, semantic_ok=si >= gamma)
            self._n_judged += 1
            if all(r == c for r in refs):
                self._refs_hold += 1
            produced = 0
            if verdict is Verdict.DEFECT:
                produced = 1
                legacy = DefectRecord(
                    image=x_adv[i].detach().cpu(),
                    source_label=c,
                    target_label=tgt,
                    target_model=self.target,
                    s_input=si,
                    s_path=float(sp[i]),
                    perturbation=float(pert[i]),
                    critical_activation=adv_crit[i].detach().cpu(),
                    source_image=batch.x0[i].detach().cpu(),
                )
                records.append(
                    FailureRecord(
                        kind="cls",
                        anchor=ConsensusAnchor(label=str(c), support=dict(seed.anchors[0].support)),
                        observed_label=str(tgt),
                        s_input=si,
                        round_idx=round_idx,
                        extra={"legacy": legacy},
                    )
                )
            stats.append(SeedStats(produced=produced, path_novel=novel))
        self._cccov_hist.append(self.tracker.cccov())
        return records, stats

    # ---- 分析与报告 ----

    def _legacy_report(self, report: RunReport) -> FuzzReport:
        legacy = FuzzReport()
        legacy.defects = [r.extra["legacy"] for r in report.failures]
        legacy.cncov_history = report.cncov_history
        legacy.cccov_history = self._cccov_hist
        legacy.rft_history = report.rft_history
        legacy.sem_shift_history = report.sem_shift_history
        legacy.lambda_history = report.lambda_history
        legacy.metrics = dict(report.metrics)
        legacy.total_iterations = report.total_rounds
        legacy.elapsed_time = report.elapsed_time
        return legacy

    def enrich_metrics(self, report: RunReport) -> dict[str, float]:
        return {
            "seed_acceptance_rate": self._acceptance,
            "n_consensus_classes": float(len(self._consensus_classes)),
            "ref_consensus_hold_rate": self._refs_hold / self._n_judged if self._n_judged else 0.0,
        }

    def analyze(self, report: RunReport, out_dir) -> None:
        """缺陷聚类与五维指标（neurons/cluster、evaluate/metrics 原样复用）。"""
        from mfuzz.evaluate.metrics import enrich_metrics as legacy_enrich
        from mfuzz.evaluate.report import save_clusters_json, save_defects
        from mfuzz.neurons.cluster import cluster_defects

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        legacy = self._legacy_report(report)
        cluster_result = cluster_defects(legacy.defects)
        extra = legacy_enrich(
            legacy,
            cluster_result,
            pgd_steps=self.cfg.optimize.pgd_steps,
            n_consensus_classes=len(self._consensus_classes),
            gamma_input=self.cfg.semantic.gamma_input,
        )
        report.metrics.update(extra)
        report.extra["cccov_history"] = [
            {str(k): v for k, v in d.items()} for d in self._cccov_hist
        ]
        save_defects(legacy, out)
        save_clusters_json(cluster_result, out)
        (out / "cluster_summary.json").write_text(
            json.dumps({"n_clusters": cluster_result.n_clusters}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._cluster_result = cluster_result

    def plot_extras(self, report: RunReport, out_dir) -> None:
        """分类版图表原样复用：覆盖双线、缺陷分布、流向、对比图、聚类散点。"""
        from mfuzz.evaluate.report import (
            plot_clusters,
            plot_coverage_curves,
            plot_defect_distributions,
            plot_defect_flow,
            plot_defect_gallery,
        )

        out = Path(out_dir)
        legacy = self._legacy_report(report)
        plot_coverage_curves(legacy, out)
        plot_defect_distributions(legacy, out)
        plot_defect_flow(legacy, out)
        plot_defect_gallery(legacy, out)
        if getattr(self, "_cluster_result", None) is not None:
            plot_clusters(self._cluster_result, out)
