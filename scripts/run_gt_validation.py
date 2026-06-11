"""真值核验的独立入口：对 run_struct_analysis 的落盘结果做离线核验。

核验逻辑在 mfuzz.evaluate.det_gt（配置驱动管线 scripts/run_det.py 复用同一模块，
其 gt 阶段在线调用）。本脚本保留给离线场景：结构分析结果已经落盘、只想补一次
真值核验时用。裁决标准见模块 docstring。

用法（服务器）：
    uv run python scripts/run_gt_validation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mfuzz.evaluate.det_gt import VERDICTS, load_gt, validate_instances

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ANN = _PROJECT_ROOT / "datasets" / "coco" / "annotations" / "instances_val2017.json"
_DEFAULT_RESULTS = _PROJECT_ROOT / "output_det" / "struct_analysis"


def print_table(target: str, counts: dict[str, dict[str, int]]) -> None:
    print(f"\n=== {target} ===")
    for kind, order in VERDICTS.items():
        cnt = counts[kind]
        total = sum(cnt.values())
        if total == 0:
            continue
        cells = "  ".join(f"{v}={cnt.get(v, 0)}" for v in order if cnt.get(v, 0))
        lead = order[0]
        rate = cnt.get(lead, 0) / total
        print(f"  {kind:9s} n={total:4d}  {lead}率={rate:.2f}  | {cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=_DEFAULT_RESULTS)
    ap.add_argument("--ann", type=Path, default=_DEFAULT_ANN)
    ap.add_argument("--out", type=Path, default=_DEFAULT_RESULTS.parent / "gt_validation")
    args = ap.parse_args()

    print(f"加载真值标注：{args.ann}")
    gt = load_gt(args.ann)

    args.out.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.results.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        report = validate_instances(
            data["instances"], gt, data["config"]["iou_thr"], data["config"]["loc_thr"]
        )
        report = {"target": path.stem, "config": data["config"], **report}
        print_table(path.stem, report["counts"])
        out_file = args.out / path.name
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"  -> {out_file}")


if __name__ == "__main__":
    main()
