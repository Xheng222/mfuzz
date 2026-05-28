"""Phase 5 评估模块单元测试。

合成缺陷向量与 FuzzReport，验证聚类、五维指标、结果校验、多组对比与可视化总入口，
全程 CPU、不依赖 ImageNet 与模型。
"""

from __future__ import annotations

import json

import torch

from mfuzz.core.types import Config, DefectRecord, FuzzReport
from mfuzz.evaluate.compare import compare, comparison_table, load_results
from mfuzz.evaluate.metrics import (
    activation_anomalies,
    cccov_round_stats,
    cccov_scalars,
    diversity_metrics,
    enrich_metrics,
    fault_rate_per_effort,
    output_impartiality,
)
from mfuzz.evaluate.report import generate_report
from mfuzz.evaluate.validate import has_failures, validate_result
from mfuzz.neurons.cluster import cluster_defects


def _defect(src: int, tgt: int, vec: torch.Tensor) -> DefectRecord:
    return DefectRecord(
        image=torch.rand(3, 8, 8),
        source_label=src,
        target_label=tgt,
        target_model="resnet50",
        s_input=1.0,
        s_path=1.0,
        perturbation=0.03,
        critical_activation=vec,
        source_image=torch.rand(3, 8, 8),
    )


def _two_blob_defects(k: int = 30, n_each: int = 12, seed: int = 0) -> list[DefectRecord]:
    # 两团指向正交方向，余弦能清楚分开：A 集中在前半维，B 集中在后半维。
    g = torch.Generator().manual_seed(seed)
    center_a = torch.zeros(k)
    center_a[: k // 2] = 5.0
    center_b = torch.zeros(k)
    center_b[k // 2 :] = 5.0
    out: list[DefectRecord] = []
    for _ in range(n_each):
        out.append(_defect(10, 20, center_a + 0.3 * torch.randn(k, generator=g)))
    for _ in range(n_each):
        out.append(_defect(11, 21, center_b + 0.3 * torch.randn(k, generator=g)))
    return out


def test_cluster_two_blobs_finds_two() -> None:
    res = cluster_defects(_two_blob_defects())
    assert res.n_clusters >= 2
    assert res.silhouette > 0.5  # 两团方向正交，余弦轮廓应很高
    assert len(res.labels) == res.n_defects == 24
    # 干净两分：每簇内只来自同一个 (src, tgt) 对
    assert all(len(c.source_labels) == 1 for c in res.clusters)
    assert len(res.embedding) == 24
    # 每簇都有主导类别对统计
    assert all(c.dominant_pair_count >= 1 for c in res.clusters)


def test_cluster_too_few_defects_single() -> None:
    res = cluster_defects(_two_blob_defects(n_each=1))
    assert res.n_clusters == 1
    assert res.n_defects == 2


def test_cluster_no_vectors() -> None:
    d = _defect(1, 2, torch.zeros(5))
    d.critical_activation = None
    res = cluster_defects([d])
    assert res.n_defects == 0
    assert res.n_clusters == 0


def test_output_impartiality_bounds() -> None:
    # 全挤在一个类别 -> 0；在所有机会类别上均匀 -> 1。
    assert output_impartiality([3, 3, 3, 3], n_opportunity_classes=4) == 0.0
    even = output_impartiality([1, 2, 3, 4], n_opportunity_classes=4)
    assert abs(even - 1.0) < 1e-9
    partial = output_impartiality([1, 1, 2], n_opportunity_classes=4)
    assert 0.0 < partial < 1.0


def test_fre_and_diversity() -> None:
    assert fault_rate_per_effort(10.0, 100.0, 10) == 10.0 / 1000.0
    report = FuzzReport(defects=_two_blob_defects(n_each=3))
    res = cluster_defects(report.defects)
    div = diversity_metrics(report, res, n_consensus_classes=5)
    assert div.n_classes == 2
    assert div.n_target_classes == 2
    assert div.n_class_pairs == 2
    assert 0.0 <= div.oi <= 1.0


def test_activation_anomalies_flags_ood() -> None:
    # ĉ 正常落在 [0,1]；越界是 ĉ>1（推过上界）或 ĉ<0（推过下界）。
    defs = [
        _defect(10, 20, torch.tensor([0.2, 0.5, 0.9, 0.3])),  # 全在范围内
        _defect(11, 21, torch.tensor([0.4, 1.8, 0.1, -0.5])),  # 维1>1、维3<0
    ]
    a = activation_anomalies(FuzzReport(defects=defs))
    assert a["n_ood_neurons"] == 2  # 维1 与 维3
    assert a["n_defects_ood"] == 1  # 只有第二个缺陷越界
    assert a["max_activation"] > 1.0
    assert a["min_activation"] < 0.0


def test_enrich_keeps_runner_semantic_values() -> None:
    # diff_cov_sem 的 runner 已按全候选统计算好语义两项，enrich 不能用缺陷统计覆盖。
    report = FuzzReport(defects=_two_blob_defects(n_each=3))
    report.metrics = {"n_fuzzed": 50.0, "input_valid_rate": 0.7, "mean_s_input": 0.95}
    res = cluster_defects(report.defects)
    extra = enrich_metrics(report, res, pgd_steps=10, n_consensus_classes=5, gamma_input=0.9)
    assert extra["input_valid_rate"] == 0.7
    assert extra["mean_s_input"] == 0.95


def test_enrich_falls_back_to_defect_semantics() -> None:
    # diff / diff_cov 没算语义，回落到缺陷统计（_defect 造的 s_input=1.0）。
    report = FuzzReport(defects=_two_blob_defects(n_each=3))
    report.metrics = {"n_fuzzed": 50.0}
    res = cluster_defects(report.defects)
    extra = enrich_metrics(report, res, pgd_steps=10, n_consensus_classes=5, gamma_input=0.9)
    assert extra["mean_s_input"] == 1.0
    assert extra["input_valid_rate"] == 1.0


def test_cccov_round_stats_and_scalars() -> None:
    # 两类逐轮上涨，str 键也要能算；空轮记 NaN 不进标量。
    hist = [{"10": 0.2, "11": 0.4}, {"10": 0.5, "11": 0.9}, {}]
    means, los, his = cccov_round_stats(hist)
    assert abs(means[0] - 0.3) < 1e-9 and los[0] == 0.2 and his[0] == 0.4
    assert abs(means[1] - 0.7) < 1e-9 and his[1] == 0.9
    assert means[2] != means[2]  # NaN
    sc = cccov_scalars(hist)
    assert abs(sc["cccov_mean_0"] - 0.3) < 1e-9
    assert abs(sc["cccov_mean_final"] - 0.7) < 1e-9  # 末个非空轮
    assert abs(sc["cccov_mean_gain"] - 0.4) < 1e-9
    assert cccov_scalars([]) == {}


def _good_result() -> dict:
    return {
        "mode": "diff_cov",
        "metrics": {
            "critical_ratio": 0.5,
            "cncov_gain": 0.24,
            "n_uncovered_final": 5000.0,
            "mean_cov_grad_norm": 14.6,
            "rft": 0.94,
            "ref_consensus_hold_rate": 0.99,
            "n_clusters": 3.0,
            "mean_s_input": 1.0,
            "input_valid_rate": 1.0,
        },
        "cncov_history": [0.34 + 0.02 * i for i in range(10)],
        "cccov_history": [{"10": 0.30 + 0.02 * i, "11": 0.26 + 0.02 * i} for i in range(10)],
        "curves": {},
    }


def test_validate_passes_good() -> None:
    checks = validate_result(_good_result())
    assert not has_failures(checks)


def test_validate_flags_round1_saturation() -> None:
    bad = _good_result()
    bad["cncov_history"] = [0.34, 0.999, 0.999]  # 第一轮饱和
    assert has_failures(validate_result(bad))


def test_validate_flags_zero_cov_grad() -> None:
    bad = _good_result()
    bad["metrics"]["mean_cov_grad_norm"] = 0.0  # 有未覆盖却梯度为零
    assert has_failures(validate_result(bad))


def test_compare_table_and_load(tmp_path) -> None:
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "result.json").write_text(json.dumps(_good_result()), encoding="utf-8")
    results = load_results([tmp_path / "a", tmp_path / "b"])
    assert len(results) == 2
    table = comparison_table(results)
    assert "experiment" in table and "RFT" in table
    assert "CCCovF" in table  # CCCov 跨类均值从历史现算，进汇总表
    # compare 总入口：写汇总表与全套对比图（含 CCCov 叠加、五维雷达）。
    compare([tmp_path / "a", tmp_path / "b"], tmp_path / "cmp")
    assert (tmp_path / "cmp" / "comparison.md").exists()
    assert (tmp_path / "cmp" / "compare_cncov.png").exists()
    assert (tmp_path / "cmp" / "compare_cccov.png").exists()
    assert (tmp_path / "cmp" / "compare_metrics.png").exists()
    assert (tmp_path / "cmp" / "compare_radar.png").exists()


def test_generate_report_end_to_end(tmp_path) -> None:
    report = FuzzReport(defects=_two_blob_defects())
    report.metrics = {
        "n_fuzzed": 100.0,
        "n_defects": float(report.num_defects),
        "n_consensus_classes": 5.0,
        "rft": 0.24,
    }
    report.cncov_history = [0.3 + 0.05 * i for i in range(6)]
    report.cccov_history = [{10: 0.3 + 0.05 * i, 11: 0.28 + 0.05 * i} for i in range(6)]
    report.total_iterations = 5

    res = generate_report(report, tmp_path, Config(), mode="diff_cov", target="resnet50")
    assert res.n_clusters >= 2
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "defects.pt").exists()
    assert (tmp_path / "clusters.json").exists()
    assert (tmp_path / "metrics.md").exists()
    assert (tmp_path / "coverage_curves.png").exists()
    assert (tmp_path / "cccov_heatmap.png").exists()
    assert (tmp_path / "defect_distributions.png").exists()
    assert (tmp_path / "defect_flow.png").exists()
    assert (tmp_path / "defect_gallery.png").exists()
    assert (tmp_path / "defect_clusters.png").exists()

    data = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert data["metrics"]["n_clusters"] >= 2
    assert "oi" in data["metrics"] and "fre" in data["metrics"]
    assert data["metrics"]["n_classes"] == 2
    # CCCov 跨类均值标量随 enrich 落进 metrics，末轮 > 首轮。
    assert data["metrics"]["cccov_mean_final"] > data["metrics"]["cccov_mean_0"]
