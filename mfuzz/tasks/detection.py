"""检测任务适配器。

把检测任务的全部语义装配到统一协议后面：torchvision 检测器族当模型层
（core/det_models），跨模型 IoU 聚簇与四类失效判定当 oracle
（differential/det_oracle），结构归因与层级消融当分析阶段（tasks/det_analysis、
neurons/struct_attr），真值核验当外部裁决（evaluate/det_gt）。

- 种子 = 一张图 + 目标模型在基线上 agree 的 consensus 对象（锚点）。
- obj_1 = 锚点当前匹配检测分数之和取负：把目标模型推向漏检/分歧，参考模型
  只当裁判、不进梯度。
- 覆盖单元 = 层 × 调用（尺度）× 通道，GAP 激活，频率版关键度。
- v(x) = FPN 各层级特征的全局平均池化拼接。
- 失效判定：变异图三模型重判，目标失效与基线失效按同类 IoU 去重，语义门
  γ_input 由本层应用；新失效变异图落 PNG 供生成失效归因。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from loguru import logger
from torch import Tensor
from torchvision.ops import box_iou

from mfuzz.core.adapter import Batch, SeedStats, TaskAdapter, TaskForward
from mfuzz.core.det_models import TorchvisionDetector, load_detectors
from mfuzz.core.records import ConsensusAnchor, FailureRecord, RunReport, Seed
from mfuzz.core.types import Detection
from mfuzz.differential.det_oracle import DetRecord, judge_image
from mfuzz.evaluate import det_report
from mfuzz.evaluate.det_gt import load_gt, validate_instances
from mfuzz.neurons.struct_attr import GraphForward, GraphResult
from mfuzz.neurons.unit_coverage import (
    Layout,
    UnitCoverageTracker,
    UnitProfile,
    profile_from_stats,
)
from mfuzz.tasks.det_analysis import (
    DetParams,
    aggregate,
    attribute_generated,
    det_dict,
    gather_images,
    load_image,
    run_ablation,
    run_attribution,
    run_baseline,
)


def _build_layout(acts: dict[str, list[Tensor]]) -> Layout:
    """从一次前向的 acts 建单元布局：层按字典序、调用按发生序。"""
    layout: Layout = []
    for name in sorted(acts):
        for ci, t in enumerate(acts[name]):
            layout.append((name, ci, int(t.shape[1])))
    return layout


# 第二遍扫描同时统计的 t_freq 档位：缓存一次、参数网格离线可扫
_T_FREQ_GRID = (0.3, 0.5, 0.7)


def _freq_pass(
    gf: GraphForward,
    paths: list,
    layout: Layout,
    low: Tensor,
    high: Tensor,
    t_list: tuple[float, ...],
    score_thr: float,
    device: torch.device,
) -> dict[float, Tensor]:
    """流式频率统计：归一化激活超各档 t 的图占比。"""
    span = (high - low + 1e-12).to(device)
    lo = low.to(device)
    cnts = {t: torch.zeros_like(lo) for t in t_list}
    with torch.no_grad():
        for i, pa in enumerate(paths):
            g = gf.run(load_image(pa, device), score_thr)
            norm = (_gap_vector(g.acts, layout, device) - lo) / span
            for t in t_list:
                cnts[t] += (norm > t).float()
            del g
            if (i + 1) % 2000 == 0:
                logger.info(f"频率统计 {i + 1}/{len(paths)}")
    return {t: (c / len(paths)).cpu() for t, c in cnts.items()}


def build_det_profile(
    target: str,
    gf: GraphForward,
    paths: list,
    t_freq: float,
    critical_tau: float,
    score_thr: float,
    device: torch.device,
    cache_dir: str = "",
) -> tuple[Layout, UnitProfile]:
    """两遍流式标定 + 落盘缓存。

    第一遍逐图统计 min/max，第二遍按 _T_FREQ_GRID 各档同时统计激活频率。缓存键
    为（目标模型、标定集目录名、图数），存 layout/low/high/多档 freq；t_cov 与
    critical_tau 是事后参数，不进缓存。配置里的 t_freq 不在网格中时单独补一遍。
    """
    stats = None
    cache_path = None
    if cache_dir:
        tag = paths[0].parent.name if paths else "none"
        cache_path = Path(cache_dir) / f"det_{target}_{tag}_n{len(paths)}.pt"
        if cache_path.exists():
            stats = torch.load(cache_path, map_location="cpu", weights_only=False)
            logger.info(f"[{target}] 加载标定缓存 {cache_path}")
    if stats is None:
        layout: Layout | None = None
        low: Tensor | None = None
        high: Tensor | None = None
        with torch.no_grad():
            for i, pa in enumerate(paths):
                g = gf.run(load_image(pa, device), score_thr)
                if layout is None:
                    layout = _build_layout(g.acts)
                v = _gap_vector(g.acts, layout, device)
                low = v.clone() if low is None else torch.minimum(low, v)
                high = v.clone() if high is None else torch.maximum(high, v)
                del g
                if (i + 1) % 2000 == 0:
                    logger.info(f"区间统计 {i + 1}/{len(paths)}")
        assert layout is not None and low is not None and high is not None, "覆盖标定图集为空"
        freqs = _freq_pass(gf, paths, layout, low, high, _T_FREQ_GRID, score_thr, device)
        stats = {
            "layout": layout,
            "low": low.cpu(),
            "high": high.cpu(),
            "freqs": freqs,
            "n": len(paths),
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(stats, cache_path)
            logger.info(f"[{target}] 标定缓存写入 {cache_path}")
    if t_freq not in stats["freqs"]:
        stats["freqs"].update(
            _freq_pass(
                gf,
                paths,
                stats["layout"],
                stats["low"],
                stats["high"],
                (t_freq,),
                score_thr,
                device,
            )
        )
        if cache_path is not None:
            torch.save(stats, cache_path)
    profile = profile_from_stats(
        stats["layout"], stats["low"], stats["high"], stats["freqs"][t_freq], critical_tau
    )
    return stats["layout"], profile


def _gap_vector(acts: dict[str, list[Tensor]], layout: Layout, device: torch.device) -> Tensor:
    """按布局把 acts 拼成单元激活向量（可微）。缺失的调用补零。"""
    parts: list[Tensor] = []
    for name, ci, ch in layout:
        calls = acts.get(name, [])
        if ci < len(calls):
            parts.append(calls[ci].mean(dim=(0, 2, 3)))
        else:
            parts.append(torch.zeros(ch, device=device))
    return torch.cat(parts)


def _sem_vector(g: GraphResult) -> Tensor:
    return torch.cat([f.mean(dim=(0, 2, 3)) for f in g.feats.values()])


def _record_box(fr: FailureRecord) -> Tensor:
    box = (
        fr.observed_box
        if fr.observed_box is not None
        else (fr.anchor.box if fr.anchor is not None else None)
    )
    assert box is not None
    return box


def _to_failure(rec: DetRecord, s_in: float, image_ref: str | None, rnd: int) -> FailureRecord:
    anchor = None
    if rec.kind != "spurious":
        anchor = ConsensusAnchor(
            label=rec.cons_label,
            box=rec.rep_box,
            support={d.model: d.score for d in rec.cluster or []},
        )
    return FailureRecord(
        kind=rec.kind,
        anchor=anchor,
        observed_label=rec.det.label if rec.det is not None else None,
        observed_box=rec.det.box if rec.det is not None else None,
        observed_score=rec.det.score if rec.det is not None else None,
        s_input=round(s_in, 4),
        image_ref=image_ref,
        round_idx=rnd,
        extra={"cluster": [det_dict(d) for d in rec.cluster or []]},
    )


def _save_png(img: Tensor, path: Path) -> None:
    from PIL import Image as PILImage

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (img.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    PILImage.fromarray(arr.permute(1, 2, 0).numpy()).save(path)


class DetectionAdapter(TaskAdapter):
    task = "detection"

    def __init__(self, cfg, target, device, out_dir) -> None:
        super().__init__(cfg, target, device, out_dir)
        self.p = DetParams.from_raw(cfg.raw)
        self.detectors: dict[str, TorchvisionDetector] = {}
        self.layout: Layout | None = None
        self._n_saved = 0

    # ---- 构建 ----

    def setup(self) -> None:
        self.detectors = load_detectors(self.cfg.models.names, self.device)
        self.tv = self.detectors[self.target]
        self.gf = GraphForward(self.tv)
        self.paths = gather_images(self.p.image_dir, self.p.num_images)
        self.base_dets = run_baseline(self.detectors, self.paths, self.p.score_thr, self.device)

    def build_seeds(self) -> list[Seed]:
        seeds: list[Seed] = []
        for path in self.paths:
            records, _ = judge_image(self.base_dets[str(path)], self.p.iou_thr, self.p.loc_thr)
            mine = records[self.target]
            anchors = [
                ConsensusAnchor(
                    label=r.cons_label,
                    box=r.rep_box,
                    support={d.model: d.score for d in r.cluster or []},
                )
                for r in mine
                if r.kind == "agree"
            ]
            if not anchors:
                continue
            baseline = [_to_failure(r, 1.0, None, -1) for r in mine if r.kind != "agree"]
            seeds.append(Seed(anchors=anchors, path=path, baseline_failures=baseline))
        return seeds

    def build_tracker(self) -> UnitCoverageTracker:
        if self.p.profile_dir:
            profile_paths = gather_images(self.p.profile_dir, self.p.profile_images)
        else:
            # 兼容路径：无独立标定集时取 image_dir 尾部、与种子图不重叠
            profile_paths = gather_images(
                self.p.image_dir, self.p.profile_images, offset=self.p.num_images
            )
        self.layout, profile = build_det_profile(
            self.target,
            self.gf,
            profile_paths,
            self.cfg.coverage.t_freq,
            self.cfg.coverage.critical_tau,
            self.p.score_thr,
            self.device,
            self.p.cache_dir,
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return UnitCoverageTracker(profile, self.cfg.coverage.t_cov, self.device)

    # ---- 循环内 ----

    def make_batches(self, seeds: list[Seed]) -> list[Batch]:
        # 检测图尺寸不一，逐图一批
        out = []
        for s in seeds:
            assert s.path is not None
            out.append(Batch(seeds=[s], x0=load_image(s.path, self.device)[None]))
        return out

    def forward(self, x: Tensor, batch: Batch) -> TaskForward:
        assert self.layout is not None
        g = self.gf.run(x[0], self.p.score_thr)
        return TaskForward(
            unit_acts=_gap_vector(g.acts, self.layout, self.device)[None],
            v_sem=_sem_vector(g)[None],
            raw=g,
        )

    def objective1(self, fw: TaskForward, batch: Batch) -> Tensor:
        g: GraphResult = fw.raw
        scores: list[Tensor] = []
        for anchor in batch.seeds[0].anchors:
            s = self._match_score(g, anchor)
            if s is not None:
                scores.append(s)
        if scores:
            return -torch.stack(scores).sum()
        # 锚点全部丢失：退而压低全部输出分数，保持梯度有定义
        return -g.scores_g.sum() if g.scores_g.numel() else g.scores_g.sum()

    def _match_score(self, g: GraphResult, anchor: ConsensusAnchor) -> Tensor | None:
        """锚点当前匹配分数（带图）。同名候选优先，否则任意类别，再无则 None。"""
        if g.boxes_g.shape[0] == 0 or anchor.box is None:
            return None
        ious = box_iou(g.boxes_g.detach(), anchor.box.to(g.boxes_g.device)[None])[:, 0]
        hit = (ious >= self.p.iou_thr).nonzero(as_tuple=True)[0]
        if hit.numel() == 0:
            return None
        same = [int(i) for i in hit if g.labels[int(i)] == anchor.label]
        pool = same if same else [int(i) for i in hit]
        scores_d = g.scores_g.detach()
        best = max(pool, key=lambda i: float(scores_d[i]))
        return g.scores_g[best]

    def judge(
        self,
        x_adv: Tensor,
        fw0: TaskForward,
        fw: TaskForward,
        batch: Batch,
        s_input: Tensor,
        round_idx: int,
    ) -> tuple[list[FailureRecord], list[SeedStats]]:
        seed = batch.seeds[0]
        s_in = float(s_input[0])
        g: GraphResult = fw.raw
        with torch.no_grad():
            dets_by_model: dict[str, list[Detection]] = {self.target: g.dets}
            for n, a in self.detectors.items():
                if n != self.target:
                    dets_by_model[n] = a.detect(x_adv[0], self.p.score_thr)
        records, _ = judge_image(dets_by_model, self.p.iou_thr, self.p.loc_thr)

        out: list[FailureRecord] = []
        if s_in >= self.cfg.semantic.gamma_input:  # 语义门：失效必须在有效输入上
            png_rel: str | None = None
            for rec in records[self.target]:
                if rec.kind == "agree" or not self._is_new(rec, seed.baseline_failures):
                    continue
                if self._n_saved < self.p.save_failures:
                    if png_rel is None:
                        assert seed.path is not None
                        png_rel = f"gen/r{round_idx:03d}_{seed.path.stem}.png"
                        _save_png(x_adv[0], self.out_dir / png_rel)
                    self._n_saved += 1
                out.append(_to_failure(rec, s_in, png_rel, round_idx))
        return out, [SeedStats(produced=len(out))]

    def _is_new(self, rec: DetRecord, baseline: list[FailureRecord]) -> bool:
        box = rec.det.box if rec.det is not None else rec.rep_box
        for b in baseline:
            if b.kind != rec.kind:
                continue
            if float(box_iou(box[None], _record_box(b)[None])[0, 0]) >= self.p.iou_thr:
                return False
        return True

    # ---- 分析与报告 ----

    def enrich_metrics(self, report: RunReport) -> dict[str, float]:
        return {"n_gen_saved": float(self._n_saved)}

    def analyze(self, report: RunReport, out_dir) -> None:
        """检测的深度分析：自然失效归因 + 真值核验 + 层级消融 + 生成失效归因。

        结果存进 report.extra["det"]（plot_extras / plot_combined 取用），逐实例
        微观记录落 detail.json。
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        viz_dir = out if self.p.viz_per_kind > 0 else None
        result = run_attribution(
            self.detectors, self.target, self.paths, self.base_dets, self.p, self.device, viz_dir
        )
        data: dict = {"aggregate": aggregate(result)}

        gt_instances = None
        if self.p.gt:
            gt = load_gt(self.p.annotations)
            gt_res = validate_instances(result["instances"], gt, self.p.iou_thr, self.p.loc_thr)
            data["gt"] = {"counts": gt_res["counts"]}
            gt_instances = gt_res["instances"]

        if self.p.ablation:
            data["ablation"] = run_ablation(
                self.detectors, self.target, self.paths, self.base_dets, self.p, self.device
            )

        gen_attr = None
        if report.failures:
            gen_attr = attribute_generated(
                self.detectors, self.target, report.failures, self.p, self.device, out
            )
            data["gen_attr"] = gen_attr

        report.extra["det"] = data
        detail = {
            "instances": result["instances"],
            "gt_instances": gt_instances,
            "gen_attr": gen_attr,
        }
        (out / "detail.json").write_text(
            json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def plot_extras(self, report: RunReport, out_dir) -> None:
        data = report.extra.get("det")
        if not data:
            return
        out = Path(out_dir)
        buckets = det_report.buckets_present({self.target: data})
        det_report.plot_share_ratios(self.target, data["aggregate"], buckets, out)
        det_report.plot_ablation(self.target, data.get("ablation", []), out)
        if data.get("gen_attr"):
            det_report.plot_nat_vs_gen(
                self.target, data["aggregate"], data["gen_attr"]["instances"], buckets, out
            )

    @classmethod
    def plot_combined(cls, per_target: dict[str, RunReport], cfg, out_dir) -> None:
        data = {t: r.extra["det"] for t, r in per_target.items() if "det" in r.extra}
        if not data:
            return
        out = Path(out_dir)
        det_report.plot_failure_counts(data, out)
        det_report.plot_level_distribution(data, out)
        det_report.plot_gt_verdicts(data, out)
        det_report.write_det_metrics_md(data, out)
