# mfuzz — Multi-dimensional Feedback-driven DNN Security Testing Framework

## Purpose

研究项目"多维反馈驱动的中间层安全分析与测试框架"的实现代码。通过梯度驱动的联合优化，自动生成能触发 DNN 预测缺陷的测试样本。

## Environment

- **Python**: 3.13, managed via [uv](https://docs.astral.sh/uv/)
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
- `output/` — 运行结果：result.json, curves.png, defects/ (git ignored)
- `实现方案.md` — 完整的模块设计与实验规划文档

## Key Concepts

- **差分测试**: 多模型（ResNet50/VGG16/MobileNetV2）互为先知，预测分歧 = 缺陷信号
- **梯度驱动变异**: 与 CriticalFuzz/NeuraL-Coverage/NSGen 不同，本框架用 obj 梯度直接驱动输入变异，不是图像变换
- **关键神经元**: 通过 cl(n,T) 识别对决策有关键影响的神经元子集，CNCov 跟踪覆盖率
- **联合优化**: obj_total = obj_1 + λ_2·obj_cov - λ_3·obj_sem（语义项待实现）

## Configuration

所有参数通过 `configs/*.toml` 管理。关键配置项：

- `[neurons] enabled`: 是否开启关键神经元覆盖引导
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

## Conventions

- Type hints everywhere, dataclass for data structures, Protocol for pluggable interfaces
- No comments unless the WHY is non-obvious
- Commit with jj, follow Conventional Commits, scope `(claude)` for Claude-initiated commits
- Reference projects (CriticalFuzz, NeuraL-Coverage, NSGen) are in `F:\claude\reference\` — algorithm reference only, do not copy code
