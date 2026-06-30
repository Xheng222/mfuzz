#!/usr/bin/env python
"""稳健的实验启动器：自动选最空闲 GPU、设 expandable_segments、失败回退重试。

面向共享 GPU 上的长时间实验（如 regen_full 全量四模型数据生成）。三件事：

- 选卡：按"已使用显存最低、其次利用率最低"挑物理卡，避开他人占用重的卡。
- 显存：设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，缓解尺寸不一的输入
  在 CUDA 缓存分配器里的碎片化（变长 COCO 图过基线时尤甚）。
- 回退：run_fuzz 失败（含他人占卡导致的 OOM）后等待、重选卡再跑。run_fuzz 对已
  落 result.json 的目标会跳过（断点续跑），所以重试从失败的目标接着跑、不重头来。

用法（服务器上，建议 nohup 后台；PATH 要带 uv）：
    export PATH=$HOME/work/bin:$HOME/.local/bin:$PATH
    nohup uv run python scripts/run_lab_experiment.py configs/det/regen_full.toml \
      > output/det/regen_full/run_regen_full.log 2>&1 &

参数：<config> [max_retries=6] [wait_s=120]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time


def pick_gpu() -> tuple[int, int, int, int]:
    """返回 (index, util%, used MiB, free MiB)：已用显存最低、其次利用率最低的物理卡。"""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: list[tuple[int, int, int, int]] = []
    for line in out.strip().splitlines():
        idx, util, used, free = (int(x.strip()) for x in line.split(","))
        rows.append((idx, util, used, free))
    if not rows:
        raise RuntimeError("nvidia-smi 没返回任何 GPU")
    rows.sort(key=lambda r: (r[2], r[1]))  # 已用显存升序，再按利用率升序
    return rows[0]


def run(config: str, max_retries: int, wait_s: int) -> int:
    for attempt in range(1, max_retries + 1):
        try:
            idx, util, used, free = pick_gpu()
        except Exception as e:  # noqa: BLE001
            print(f"[尝试 {attempt}] 选卡失败：{e}；{wait_s}s 后重试", flush=True)
            time.sleep(wait_s)
            continue
        print(
            f"[尝试 {attempt}/{max_retries}] 选物理卡 GPU {idx}"
            f"（util {util}%、已用 {used} MiB、空闲 {free} MiB）",
            flush=True,
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(idx)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONPATH"] = "."
        rc = subprocess.call(
            [sys.executable, "-u", "scripts/run_fuzz.py", "--config", config], env=env
        )
        if rc == 0:
            print(f"[尝试 {attempt}] run_fuzz 成功完成", flush=True)
            return 0
        print(
            f"[尝试 {attempt}] run_fuzz 退出码 {rc}（可能 OOM 或他人占卡）；"
            f"{wait_s}s 后重选卡续跑（已完成目标会跳过）",
            flush=True,
        )
        time.sleep(wait_s)
    print(f"重试 {max_retries} 次仍未完成，放弃", flush=True)
    return 1


def main() -> int:
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/det/regen_full.toml"
    max_retries = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    wait_s = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    print(
        f"启动器：config={config} max_retries={max_retries} wait={wait_s}s",
        flush=True,
    )
    return run(config, max_retries, wait_s)


if __name__ == "__main__":
    sys.exit(main())
