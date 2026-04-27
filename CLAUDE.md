# mfuzz — Multi-dimensional Feedback-driven DNN Security Testing Framework

## Purpose

研究项目"多维反馈驱动的中间层安全分析与测试框架"的实现代码。通过梯度驱动的联合优化，自动生成能触发 DNN 预测缺陷的测试样本。

## Environment

- **Python**: 3.13, managed via `uv`
- **PyTorch**: 2.12 nightly + CUDA 13.2 (RTX 5070 Ti)
- **OS**: Windows 11
- **VCS**: jj (git backend)

## Commands

```bash
# Run the fuzzing framework
uv run python scripts/run_fuzz.py                    # use configs/default.toml
uv run python scripts/run_fuzz.py configs/my.toml    # custom config

# Add a dependency
uv add <package>

# Run tests
uv run pytest tests/
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
- `scripts/` — 入口脚本（run_fuzz.py）和冒烟测试
- `datasets/` — 数据集（symlink to NeuraL-Coverage, git ignored）
- `output*/` — 运行结果：result.json, curves.png, defects/ (git ignored)
- `references/` — 相关开源项目（CriticalFuzz、NeuraL-Coverage、NSGen）的算法参考，应该深入研究，但不要原样复制代码，因为可能遇到依赖不同的情况（git ignored）
- `docs/` — 研究文档

## Key Concepts

- **差分测试**: 多模型（ResNet50/VGG16/MobileNetV2）互为先知，预测分歧 = 缺陷信号
- **梯度驱动变异**: 与 CriticalFuzz/NeuraL-Coverage/NSGen 不同，本框架用 obj 梯度直接驱动输入变异，不是图像变换
- **关键神经元**: 通过 cl(n,T) 识别对决策有关键影响的神经元子集，CNCov 跟踪覆盖率。critical_threshold 影响 D_en 规模——过低会导致 CNCov 从第 1 轮即为 1.0，失去引导作用
- **缺陷分类**: 用关键神经元激活模式作为缺陷"指纹"，对缺陷样本聚类分组，衡量缺陷多样性（待实现）
- **联合优化**: obj_total = obj_1 + λ_2·obj_cov - λ_3·obj_sem（语义项待实现）

## Configuration

所有参数通过 `configs/*.toml` 管理。关键配置项：

- `random_seed`: 全局随机种子，保证实验可复现
- `[neurons] enabled`: 是否开启关键神经元覆盖引导
- `[neurons] critical_threshold`: 关键神经元阈值，影响 D_en 规模和 CNCov 区分度
- `[fuzz] lambda2`: 覆盖目标权重，设为 0 则退化为纯差分 fuzzing
- `[fuzz] pgd_steps`: PGD 迭代步数，影响变异强度和速度

## Implementation Status

| Phase | Module | Status |
|-------|--------|--------|
| 1 | core, differential, engine | Done |
| 2 | neurons (profiler, coverage, objective) | Done |
| 3 | semantic (语义约束) | Not started |
| 4 | optimize (动态反馈) | Partial (operator done, feedback not started) |
| 5 | evaluate | Partial (report done, comparison not started) |

### Commit Rules

- Commit with jj, follow `Conventional Commits`. **Only commit when the user explicitly asks.**
- **Scope**: In default, use scope `(claude)` for Claude-initiated commits, but if a specific scope is better, use it instead (e.g., `(neurons)`, `(engine)`, `(report.py)`, etc.), and mention in the commit message body that it's a Claude-initiated commit.
- **Pre-commit check**: Always run `git status` first to review changed files and ensure no unwanted binary files are staged. **Do NOT** use `jj status` for this purpose — jj's auto-snapshot may cause binary files to be accidentally committed.
- **Message format**: Multi-line. First line is the conventional commit title, followed by a blank line and a brief body describing what changed.
- **Language**: Chinese or English, decided freely per commit.

```bash
jj commit -m "feat(claude): 初始化项目环境

- 使用 uv 初始化 Python 3.13 环境
- 添加 .gitignore
"
```

## Rearch Docs

- 研究文档位于 `docs/` 目录，包含实验指南、报告、设计文档等。如果有新的实验或设计需要记录，需要记录在 `docs/` 下的对应文件夹中
- `docs/实现方案.md` 是完整的实现方案文档，包含模块设计、当前状态和任务清单。每完成一个任务应更新其中的 checkbox。
- `docs/materials/` 文件夹下的文档是一些研究材料，包含研究目标、内容、方案等，是项目必须遵守的要求。如果与现有的实现方案有冲突，需要提醒用户选择是否调整实现方案
- `docs/` 下的其它对应文文件夹中包含每个阶段的实验报告和指南，如 `docs/phase_1/` 等。 `docs/phase_1_2/` 则代表是第一阶段和第二阶段的联合实验指南

### Writing Style

当需要写作 `docs/` 下的文档时，写作基于事实，不能编造，写作风格介于书面学术写作和口语描述之间。保证所有的句子有主语，不要用复杂的长难句，尽量用短句输出。替换掉所有的非日常词汇。减少列表的使用，除非的确需要。保持段落之间的逻辑连贯，避免跳跃式的叙述


## Conventions

- Type hints everywhere, dataclass for data structures, Protocol for pluggable interfaces
- No comments unless the WHY is non-obvious
