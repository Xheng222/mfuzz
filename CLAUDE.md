# mfuzz — Multi-dimensional Feedback-driven DNN Security Testing Framework

## Purpose

研究项目"多维反馈驱动的中间层安全分析与测试框架"的代码与文档。

## Environment

- **Python**: 3.13, managed via `uv`
- **PyTorch**

## Commands

```bash
# Run a script
uv run python <script.py>

# Add a dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Sync environment after editing pyproject.toml
uv sync
```

## Directory Structure

```
src/mfuzz/
├── core/           # models.py, hooks.py, datasets.py, types.py
├── differential/   # ensemble.py, objective.py           (研究内容 1)
├── neurons/        # profiler.py, coverage.py, objective.py (研究内容 2)
├── semantic/       # protocol.py, feature.py, clip.py    (研究内容 3, 待实现)
├── optimize/       # joint.py, operator.py, feedback.py  (研究内容 4, 部分实现)
├── engine/         # runner.py, seed_pool.py
└── evaluate/       # report.py
```

- `configs/` — TOML 配置文件，控制数据集、模型、neuron profiling、fuzzing 参数
- `scripts/` — 入口脚本（run_fuzz.py）和测试脚本
- `datasets/` — 数据集（symlink to NeuraL-Coverage, git ignored）
- `output*/` — 运行结果：result.json, curves.png, defects/ (git ignored)
- `references/` — 相关开源项目（CriticalFuzz、NeuraL-Coverage、NSGen）的算法参考，应该深入研究，但不要原样复制代码，因为可能遇到依赖不同的情况（git ignored）
- `docs/` — 研究文档


## Rearch Docs

- 研究文档位于 `docs/` 目录，包含实验指南、报告、设计文档等。如果有新的实验或设计需要记录，需要记录在 `docs/` 下的对应文件夹中
- `docs/实现方案.md` 是完整的实现方案文档，包含模块设计、当前状态和任务清单。每完成一个任务应更新其中的 checkbox。
- `docs/materials/` 文件夹下的文档是一些研究材料，包含研究目标、内容、方案等，是项目必须遵守的要求。如果与现有的实现方案有冲突，需要提醒用户选择是否调整实现方案
- `docs/` 下的其它对应文文件夹中包含每个阶段的实验报告和指南，如 `docs/phase_1/` 等。 `docs/phase_1_2/` 则代表是第一阶段和第二阶段的联合实验指南

### Writing Style

当需要写作 `docs/` 下的文档时，写作基于事实，不能编造，写作风格介于书面学术写作和口语描述之间。保证所有的句子有主语，不要用复杂的长难句，尽量用短句输出。替换掉所有的非日常词汇。减少列表的使用，除非的确需要。保持段落之间的逻辑连贯，避免跳跃式的叙述


