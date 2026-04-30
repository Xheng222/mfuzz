# mfuzz — Multi-dimensional Feedback-driven DNN Security Testing Framework


## Directory Structure

```
src/mfuzz/
├── core/           # models.py, hooks.py, datasets.py, types.py
├── differential/   # ensemble.py, objective.py           (研究内容 1)
├── neurons/        # profiler.py, coverage.py, objective.py (研究内容 2)
├── semantic/       # protocol.py, feature.py, clip.py    (研究内容 3)
├── optimize/       # joint.py, operator.py, feedback.py  (研究内容 4)
├── engine/         # runner.py, seed_pool.py
└── evaluate/       # report.py
```

- `configs/` — TOML 配置文件，控制数据集、模型、neuron profiling、fuzzing 参数
- `scripts/` — 入口脚本 (run_fuzz.py) 和测试脚本
- `datasets/` — 数据集 (git ignored)
- `output*/` — 运行结果 (git ignored)
- `docs/` — 研究文档

## 环境

- Python 3.13
- 包管理工具：[uv](https://github.com/astral-sh/uv)
