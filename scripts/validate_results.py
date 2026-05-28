"""实验结果自动校验脚本（实现方案第七章）。

读一个或多个 result.json，逐条断言并打印检查表。任一 error 级别检查不通过，进程以
非零码退出，方便接入流水线。

用法：
    uv run python scripts/validate_results.py output/diff_cov
    uv run python scripts/validate_results.py output/diff output/diff_cov
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mfuzz.evaluate.validate import has_failures, summarize, validate_result

_STATUS_STYLE = {"pass": "green", "fail": "red", "skip": "dim"}


def _load(path: str) -> tuple[str, dict]:
    p = Path(path)
    rp = p / "result.json" if p.is_dir() else p
    return rp.parent.name, json.loads(rp.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 result.json 是否符合第七章预期")
    parser.add_argument("results", nargs="+", help="result.json 路径或含它的目录，可多个")
    args = parser.parse_args()

    console = Console()
    any_fail = False
    for path in args.results:
        name, data = _load(path)
        checks = validate_result(data)
        m = data.get("metrics", {})
        l2 = m.get("lambda2_init", "?")
        l3 = m.get("lambda3_init", "?")
        fb = int(m.get("feedback_enabled", 0))
        tgt = data.get("target_model", "?")
        table = Table(title=f"{name}  (λ2={l2} λ3={l3} fb={fb}, target={tgt})", show_lines=False)
        table.add_column("检查项")
        table.add_column("结果")
        table.add_column("详情", overflow="fold")
        for c in checks:
            table.add_row(c.name, f"[{_STATUS_STYLE[c.status]}]{c.status}[/]", c.detail)
        console.print(table)
        passed, fail_err, fail_warn, skipped = summarize(checks)
        console.print(
            f"  通过 {passed}，失败(error) {fail_err}，失败(warn) {fail_warn}，跳过 {skipped}\n"
        )
        any_fail = any_fail or has_failures(checks)

    if any_fail:
        console.print("[red]存在 error 级别不通过项[/]")
        sys.exit(1)
    console.print("[green]全部 error 级别检查通过[/]")


if __name__ == "__main__":
    main()
