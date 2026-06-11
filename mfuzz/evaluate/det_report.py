"""检测任务的图表库。被检测适配器的 plot_extras / plot_combined 调用。

输入是按目标模型组织的数据字典：{目标名: {"aggregate", "gt", "ablation",
"gen_attr"}}（见 tasks/detection.py 的 analyze）。只画最终值的柱状/堆叠柱状
（计数、份额比值、责任尺度、消融变化、真值裁决构成）；随轮变化的曲线在通用
报告（evaluate/run_report）。不用热力图；图上文字 ASCII。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mfuzz.core.det_models import TV_BUCKET_ORDER  # noqa: E402
from mfuzz.evaluate.det_gt import VERDICTS  # noqa: E402

FAIL_KINDS = ("miss", "spurious", "cls", "loc")
_KIND_COLOR = {
    "miss": "#1f77b4",
    "spurious": "#d62728",
    "cls": "#9467bd",
    "loc": "#ff7f0e",
    "agree": "#2ca02c",
}
_LEVEL_CMAP = plt.get_cmap("viridis")


def buckets_present(per_target: dict) -> list[str]:
    seen: set[str] = set()
    for data in per_target.values():
        for d in data["aggregate"]["mean_shares"].values():
            seen.update(d)
    return [b for b in TV_BUCKET_ORDER if b in seen]


def plot_failure_counts(per_target: dict, out: Path) -> None:
    """失效计数分组柱状：x=失效类型，每目标模型一组。"""
    targets = list(per_target)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(len(targets), 1)
    xs = np.arange(len(FAIL_KINDS))
    for i, t in enumerate(targets):
        counts = per_target[t]["aggregate"]["counts"]
        vals = [counts.get(k, 0) for k in FAIL_KINDS]
        bars = ax.bar(xs + i * width, vals, width, label=t)
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(xs + width * (len(targets) - 1) / 2)
    ax.set_xticklabels(FAIL_KINDS)
    ax.set_ylabel("#instances")
    ax.set_title("Differential failure counts by kind and target")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "failure_counts.png", dpi=130)
    plt.close(fig)


def plot_share_ratios(target: str, agg: dict, buckets: list[str], out: Path) -> None:
    """归因份额相对 agree 比值的分组柱状：x=结构桶，每失效类型一组，基线 1.0。"""
    ratios = agg["ratio_vs_agree"]
    kinds = [k for k in FAIL_KINDS if ratios.get(k)]
    if not kinds:
        return
    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(buckets)), 4))
    width = 0.8 / len(kinds)
    xs = np.arange(len(buckets))
    for i, k in enumerate(kinds):
        vals = [ratios[k].get(b, np.nan) for b in buckets]
        ax.bar(xs + i * width, vals, width, label=k, color=_KIND_COLOR[k])
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    ax.set_xticks(xs + width * (len(kinds) - 1) / 2)
    ax.set_xticklabels(buckets, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("share ratio vs agree")
    ax.set_title(f"{target}: attribution share ratio vs agree baseline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "share_ratio.png", dpi=130)
    plt.close(fig)


def plot_level_distribution(per_target: dict, out: Path) -> None:
    """责任尺度堆叠柱状：每目标一个子图，x=失效类型，堆叠=P 层级占比。"""
    targets = list(per_target)
    levels_all = sorted(
        {lv for d in per_target.values() for kd in d["aggregate"]["levels"].values() for lv in kd}
    )
    if not levels_all:
        return
    color = {lv: _LEVEL_CMAP(i / max(len(levels_all) - 1, 1)) for i, lv in enumerate(levels_all)}
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets, strict=True):
        levels = per_target[t]["aggregate"]["levels"]
        kinds = [k for k in FAIL_KINDS if levels.get(k)]
        bottoms = np.zeros(len(kinds))
        for lv in levels_all:
            fracs = []
            for k in kinds:
                d = levels[k]
                total = sum(d.values())
                fracs.append(d.get(lv, 0) / total if total else 0.0)
            ax.bar(kinds, fracs, bottom=bottoms, color=color[lv], label=lv)
            bottoms += np.asarray(fracs)
        ax.set_title(t, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("responsible level fraction")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[lv]) for lv in levels_all]
    fig.legend(
        handles,
        levels_all,
        loc="lower center",
        fontsize=8,
        ncol=len(levels_all),
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle("Responsible FPN level distribution by failure kind", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "level_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(target: str, ablation: list, out: Path) -> None:
    """层级消融分组柱状：x=被置零层级，每失效类型一组，y=计数相对基线的变化。"""
    if not ablation:
        return
    base = dict(ablation[0][1])
    rows = ablation[1:]
    if not rows:
        return
    names = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 4))
    width = 0.8 / len(FAIL_KINDS)
    xs = np.arange(len(names))
    for i, k in enumerate(FAIL_KINDS):
        deltas = [r[1].get(k, 0) - base.get(k, 0) for r in rows]
        bars = ax.bar(xs + i * width, deltas, width, label=k, color=_KIND_COLOR[k])
        ax.bar_label(bars, fontsize=6, fmt="%+d")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xticks(xs + width * (len(FAIL_KINDS) - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_ylabel("count change vs baseline")
    ax.set_title(f"{target}: failure count change after zeroing FPN level")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=130)
    plt.close(fig)


def plot_gt_verdicts(per_target: dict, out: Path) -> None:
    """真值裁决构成堆叠柱状：每目标一个子图。首列裁决（真值站在 oracle 一边）
    统一画绿色在底部，占比越高 oracle 越可信。"""
    targets = [t for t in per_target if per_target[t].get("gt")]
    if not targets:
        return
    fig, axes = plt.subplots(1, len(targets), figsize=(4.5 * len(targets), 4), sharey=True)
    axes = np.atleast_1d(axes)
    other_colors = ["#ff7f0e", "#9467bd", "#8c564b"]
    for ax, t in zip(axes, targets, strict=True):
        counts = per_target[t]["gt"]["counts"]
        kinds = [k for k in VERDICTS if sum(counts.get(k, {}).values())]
        bottoms = np.zeros(len(kinds))
        max_verdicts = max(len(VERDICTS[k]) for k in kinds)
        for vi in range(max_verdicts):
            fracs = []
            for k in kinds:
                cnt = counts[k]
                total = sum(cnt.values())
                order = VERDICTS[k]
                v = cnt.get(order[vi], 0) if vi < len(order) else 0
                fracs.append(v / total if total else 0.0)
            color = "#2ca02c" if vi == 0 else other_colors[(vi - 1) % len(other_colors)]
            ax.bar(kinds, fracs, bottom=bottoms, color=color)
            bottoms += np.asarray(fracs)
        ax.set_title(t, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("verdict fraction")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#2ca02c", "#ff7f0e", "#9467bd", "#8c564b")
    ]
    labels = ["GT confirms oracle", "2nd verdict", "3rd verdict", "4th verdict"]
    fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("GT validation: verdict composition per oracle kind", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "gt_verdicts.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _mean_shares(instances: list[dict]) -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, list[float]]] = {}
    for inst in instances:
        d = acc.setdefault(inst["kind"], {})
        for b, v in inst["shares"].items():
            d.setdefault(b, []).append(v)
    return {k: {b: sum(v) / len(v) for b, v in d.items() if v} for k, d in acc.items()}


def plot_nat_vs_gen(
    target: str, agg: dict, gen_instances: list[dict], buckets: list[str], out: Path
) -> None:
    """自然失效 vs 生成失效的份额比值对比，两边都以自然 agree 份额为分母。"""
    gen_shares = _mean_shares(gen_instances)
    agree = agg["mean_shares"].get("agree", {})
    kinds = [k for k in FAIL_KINDS if agg["ratio_vs_agree"].get(k) and gen_shares.get(k)]
    if not kinds or not agree:
        return
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 4), sharey=True)
    axes = np.atleast_1d(axes)
    xs = np.arange(len(buckets))
    for ax, k in zip(axes, kinds, strict=True):
        nat = [agg["ratio_vs_agree"][k].get(b, np.nan) for b in buckets]
        gen = [gen_shares[k].get(b, np.nan) / agree[b] if agree.get(b) else np.nan for b in buckets]
        ax.bar(xs - 0.2, nat, 0.4, color="#1f77b4")
        ax.bar(xs + 0.2, gen, 0.4, color="#d62728")
        ax.axhline(1.0, color="k", lw=1.0, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(buckets, rotation=40, ha="right", fontsize=7)
        ax.set_title(k, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("share ratio vs natural agree")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728"),
    ]
    fig.legend(
        handles,
        ["natural", "generated"],
        loc="lower center",
        fontsize=8,
        ncol=2,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(f"{target}: natural vs generated failure signatures", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "nat_vs_gen.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def write_det_metrics_md(per_target: dict, out: Path) -> None:
    """跨目标的检测分析汇总表：计数、真值核验、落框率。"""
    lines = ["# 检测分析汇总", ""]
    for t, data in per_target.items():
        agg = data["aggregate"]
        lines += [f"## {t}", ""]
        rows = [[k, str(agg["counts"].get(k, 0))] for k in (*FAIL_KINDS, "agree")]
        rows.append(["deep_miss", str(agg["deep_miss"])])
        lines += ["### 失效计数", "", *_md_table(["类型", "n"], rows), ""]
        if data.get("gt"):
            rows = []
            for k, order in VERDICTS.items():
                cnt = data["gt"]["counts"].get(k, {})
                total = sum(cnt.values())
                if not total:
                    continue
                rate = cnt.get(order[0], 0) / total
                cells = "，".join(f"{v}={cnt[v]}" for v in order if cnt.get(v))
                rows.append([k, str(total), f"{rate:.2f}", cells])
            lines += [
                "### 真值核验（首列裁决=真值站在 oracle 一边）",
                "",
                *_md_table(["类型", "n", "确认率", "裁决构成"], rows),
                "",
            ]
        rows = [[k, f"{v:.2f}"] for k, v in agg["inside_rate"].items()]
        lines += ["### 峰值落框率", "", *_md_table(["类型", "落框率"], rows), ""]
    (out / "det_metrics.md").write_text("\n".join(lines), encoding="utf-8")
