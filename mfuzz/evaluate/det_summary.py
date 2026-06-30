"""检测实验的跨实验汇总（output 重整后）。

发现并读取 output/det/<run>/<model>/data/result.json，产出一张跨 run/model 的
markdown 对照表，外加几张对比图：CNCov 增长叠加、累计失效叠加、RFT 与语义偏移叠加、
以及差分失效计数的分组柱状。每条记录一行（run + model），表里覆盖主循环标量指标和
差分失效计数。只用折线/柱状，文字一律英文，避免中文字体依赖；不画热力图。

新的 output 结构是 <run>/<model>/data/result.json，不再是 <目录>/result.json。本模块
按新结构发现文件，跳过 plot_combined 写出的 <run>/_run 这类没有 data/result.json 的目录。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mfuzz.core.records import FAILURE_KINDS  # noqa: E402

# 对照表展示的主循环标量指标：result.json 的 metrics 键 -> 表头。
SUMMARY_KEYS: dict[str, str] = {
    "n_seeds": "#Seeds",
    "n_rounds": "#Rounds",
    "n_fuzzed": "#Fuzzed",
    "n_new_failures": "#NewFail",
    "rft": "RFT",
    "cncov_0": "CNCov0",
    "cncov_final": "CNCovF",
    "cncov_gain": "CNCovGain",
    "input_valid_rate": "ValidRate",
    "n_gen_saved": "#GenSaved",
    "elapsed_time": "Elapsed(s)",
}

_FAIL_COLOR = {
    "miss": "#1f77b4",
    "spurious": "#d62728",
    "cls": "#9467bd",
    "loc": "#ff7f0e",
    "agree": "#2ca02c",
}


@dataclass
class DetResult:
    """一次 run/model 的汇总视图。name 用 "run/model" 标识，跨实验唯一。"""

    run: str
    model: str
    target: str
    path: Path
    metrics: dict[str, float]
    fail_counts: dict[str, int]  # 差分失效计数（含 agree），缺则空
    cncov_history: list[float] = field(default_factory=list)
    fail_history: dict[str, list[int]] = field(default_factory=dict)
    rft_history: list[float] = field(default_factory=list)
    sem_shift_history: list[float] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.run}/{self.model}"


def discover_results(root: str | Path) -> list[Path]:
    """在 root 下按 <run>/<model>/data/result.json 发现结果文件，路径排序后返回。

    root 既可是 output/det（扫所有 run），也可是 output/det/<run>（扫单个 run 的各
    model）。两种深度都用同一个 glob 兜住，再去重。
    """
    root = Path(root)
    found = set(root.glob("*/*/data/result.json"))  # root=output/det
    found |= set(root.glob("*/data/result.json"))  # root=output/det/<run>
    return sorted(found)


def _fail_counts(data: dict) -> dict[str, int]:
    """从 result.json 的 extra.det.aggregate.counts 取差分失效计数；没有就空字典。"""
    agg = data.get("extra", {}).get("det", {}).get("aggregate", {})
    counts = agg.get("counts", {})
    return {k: int(v) for k, v in counts.items()}


def _is_fuzz_run(data: dict) -> bool:
    """判定一个 result.json 是不是主循环写出的 fuzz-run 产物。

    fuzz-run 一定带一个非空的 metrics 标量字典。repair_finetune/repair_pilot 下的
    修复 sweep 产物（frontier/sweep schema，键是 sweep/responsible_layers/baseline_B
    一类）没有 metrics，喂进来只会解析出一整行空值，应当跳过而非误当 fuzz-run。
    """
    metrics = data.get("metrics")
    return isinstance(metrics, dict) and bool(metrics)


def _build_result(p: Path, data: dict) -> DetResult:
    """从已读出的 result.json 数据构造 DetResult。run/model 取自目录层级。"""
    model = p.parent.parent.name  # <model>/data/result.json
    run = p.parent.parent.parent.name  # <run>/<model>/data/result.json
    return DetResult(
        run=run,
        model=model,
        target=data.get("target_model", model),
        path=p,
        metrics=dict(data.get("metrics", {})),
        fail_counts=_fail_counts(data),
        cncov_history=data.get("cncov_history", []),
        fail_history=data.get("fail_history", {}),
        rft_history=data.get("rft_history", []),
        sem_shift_history=data.get("sem_shift_history", []),
    )


def load_result(path: str | Path, root: str | Path | None = None) -> DetResult:
    """读一个 result.json，构造 DetResult。run/model 从目录层级推断。"""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return _build_result(p, data)


def load_results(paths: list[str | Path]) -> list[DetResult]:
    """读取多个 result.json，跳过非 fuzz-run schema（修复 sweep）的文件。

    glob 发现的目录里混着修复 sweep 的 result.json，它们 schema 不同、没有 metrics。
    这里逐个读取并用 _is_fuzz_run 过滤，只保留 fuzz-run 产物，跳过的记一条日志。
    """
    results: list[DetResult] = []
    for path in paths:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not _is_fuzz_run(data):
            logger.info(f"summarize_det 跳过非 fuzz-run 产物（无 metrics）：{p}")
            continue
        results.append(_build_result(p, data))
    return results


def _fmt(key: str, v: float) -> str:
    if key.startswith("n_") or key == "elapsed_time":
        return f"{v:.0f}"
    return f"{v:.4g}"


def summary_table(results: list[DetResult]) -> str:
    """跨实验对照表：行是 run/model，列是主循环标量 + 四类差分失效计数。缺失记 —。"""
    fail_headers = [f"#{k}" for k in FAILURE_KINDS]
    headers = ["run", "model", "target", *SUMMARY_KEYS.values(), *fail_headers]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in results:
        cells = [r.run, r.model, r.target]
        for key in SUMMARY_KEYS:
            cells.append(_fmt(key, r.metrics[key]) if key in r.metrics else "—")
        for k in FAILURE_KINDS:
            cells.append(str(r.fail_counts[k]) if k in r.fail_counts else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_cncov_overlay(results: list[DetResult], out_path: Path) -> Path | None:
    """各实验 CNCov 增长曲线叠加。没有覆盖历史的实验跳过。"""
    has = [r for r in results if r.cncov_history]
    if not has:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in has:
        ax.plot(range(len(r.cncov_history)), r.cncov_history, "-o", ms=3, lw=1.6, label=r.name)
    ax.set_xlabel("round")
    ax.set_ylabel("CNCov")
    ax.set_title("Critical-unit coverage growth across experiments")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_cumulative_failures(results: list[DetResult], out_path: Path) -> Path | None:
    """各实验累计新失效曲线叠加（跨失效类型求和后逐轮累加）。无失效历史的实验跳过。"""
    has = [r for r in results if any(r.fail_history.values())]
    if not has:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in has:
        n_rounds = max((len(h) for h in r.fail_history.values()), default=0)
        cum, acc = [], 0
        for i in range(n_rounds):
            acc += sum(h[i] for h in r.fail_history.values() if i < len(h))
            cum.append(acc)
        ax.plot(range(1, len(cum) + 1), cum, "-o", ms=3, lw=1.6, label=r.name)
    ax.set_xlabel("round")
    ax.set_ylabel("#cumulative new failures")
    ax.set_title("Cumulative new failures across experiments")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_rft_overlay(results: list[DetResult], out_path: Path) -> Path | None:
    """RFT 与语义偏移随轮叠加，两格分开（量纲不同）。两段历史都空的实验跳过。"""
    has = [r for r in results if r.rft_history or r.sem_shift_history]
    if not has:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True)
    cmap = plt.get_cmap("tab10")
    panels = [
        (axes[0], "rft_history", "RFT", "rate of fresh triggers"),
        (axes[1], "sem_shift_history", "semantic shift", "semantic shift"),
    ]
    for ax, attr, title, ylab in panels:
        drew = False
        for idx, r in enumerate(has):
            hist = getattr(r, attr)
            if not hist:
                continue
            ax.plot(range(len(hist)), hist, "-o", ms=3, lw=1.6, color=cmap(idx % 10), label=r.name)
            drew = True
        ax.set_xlabel("round")
        ax.set_ylabel(ylab)
        ax.set_title(f"{title} across experiments")
        ax.grid(alpha=0.3)
        if drew:
            ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_failure_counts(results: list[DetResult], out_path: Path) -> Path | None:
    """差分失效计数分组柱状：x=失效类型，每个 run/model 一组。无计数的实验跳过。"""
    has = [r for r in results if r.fail_counts]
    if not has:
        return None
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(has)), 4.4))
    width = 0.8 / max(len(has), 1)
    xs = range(len(FAILURE_KINDS))
    for i, r in enumerate(has):
        vals = [r.fail_counts.get(k, 0) for k in FAILURE_KINDS]
        bars = ax.bar([x + i * width for x in xs], vals, width, label=r.name)
        ax.bar_label(bars, fontsize=6)
    ax.set_xticks([x + width * (len(has) - 1) / 2 for x in xs])
    ax.set_xticklabels(FAILURE_KINDS)
    ax.set_ylabel("#instances")
    ax.set_title("Differential failure counts by kind across experiments")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def summarize(root: str | Path, out_dir: str | Path) -> Path:
    """发现 root 下的全部 result.json，写对照表与对比图，返回对照表路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = discover_results(root)
    if not paths:
        raise FileNotFoundError(
            f"在 {root} 下没找到 <run>/<model>/data/result.json，请确认 output 已同步"
        )
    results = load_results(paths)
    if not results:
        raise FileNotFoundError(
            f"在 {root} 下发现的 result.json 都不是 fuzz-run 产物（都没有 metrics），无可汇总"
        )
    table_path = out / "summary.md"
    table_path.write_text(summary_table(results), encoding="utf-8")
    plot_cncov_overlay(results, out / "summary_cncov.png")
    plot_cumulative_failures(results, out / "summary_failures.png")
    plot_rft_overlay(results, out / "summary_rft.png")
    plot_failure_counts(results, out / "summary_failure_counts.png")
    return table_path
