#!/usr/bin/env bash
# 服务器侧启动器：自动选显存占用最低的 GPU 跑 run_fuzz.py，OOM 或抢不到卡时
# 等 20 分钟重试，最多 12 次。共享 GPU 上全量实验的防御性入口。
#
# 用法： nohup bash scripts/lab_run_det.sh configs/det/base.toml [gpu] > output_det/base_run.log 2>&1 &
# 第二个参数可显式指定 GPU 序号；省略则自动选显存占用最低的卡。
set -u
cfg="${1:-configs/det/base.toml}"
pin="${2:-}"
cd "$(dirname "$0")/.."
# nohup 的非交互 shell 不读 .bashrc，uv 所在目录要手动进 PATH
export PATH="$HOME/work/bin:$HOME/.local/bin:$PATH"

for i in $(seq 1 12); do
    if [ -n "$pin" ]; then
        g="$pin"
    else
        g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | sort -t, -k2 -n | head -1 | cut -d, -f1)
    fi
    free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$g" \
        | awk -F, '{print $1-$2}')
    echo "[lab_run_det] attempt $i: GPU $g (free ${free} MiB), config $cfg"
    if [ "$free" -lt 6000 ]; then
        echo "[lab_run_det] free memory < 6000 MiB, sleep 20min"
        sleep 1200
        continue
    fi
    if CUDA_VISIBLE_DEVICES="$g" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python scripts/run_fuzz.py --config "$cfg"; then
        echo "[lab_run_det] done"
        exit 0
    fi
    echo "[lab_run_det] run failed, sleep 20min before retry"
    sleep 1200
done
echo "[lab_run_det] gave up after 12 attempts"
exit 1
