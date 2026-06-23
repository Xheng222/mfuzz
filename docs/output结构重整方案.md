# output 目录结构重整方案

## 背景与目标

我们在重跑全部实验之前，先把 output 的目录结构重新设计一遍。现在有两个产物根：

- `output/`：profiling 缓存（`output/profiles/`）加上分类阶段的 run（`output/cls_imagenet100/<model>/` 等）。
- `output_det/`：检测阶段的 run，每个 run 形如 `output_det/<run>/<model>/`。

单个检测 run/model 目录下混着多种东西：核心数据（`result.json`、`detail.json`、`metrics.md`）、随轮曲线图、任务图表、生成的变异图（`gen/`，单模型约 70M，最占地方）、抽查标注图（`viz/`）、探针产物（`probes/`）。run 级别还有 `det_metrics.md` 和几张汇总图。

这次重整要达到三个目标：

1. 每个实验的目录层级清晰，一眼能区分"核心数据"、"图表"、"样本图"、"探针"。
2. 重图样本（`gen/`、`viz/`）不再撑大目录，并且默认不拉回本地，需要时再单独拉。
3. 缓存与 run 产物分开，缓存不跟着某一次 run 走。

## 当前产物来源的代码地图

把"谁写了什么、写到哪里"先理清楚，方案才有落点。一次完整运行的入口是 `scripts/run_fuzz.py`，它从配置的 `[run] out` 取根目录 `out`，再对每个目标模型构造 `out_t = out / target` 作为该 run/model 的目录，把 `out_t` 一路传给适配器、主循环和报告。

run/model 目录（`out_t`）下的产物：

| 产物 | 写出位置 | 代码 |
| --- | --- | --- |
| `result.json` | 直接在 `out_t` 下 | `mfuzz/evaluate/run_report.py` `generate_run_report` |
| `metrics.md` | 直接在 `out_t` 下 | `run_report.py` `write_metrics_md` |
| `loop_curves.png`、`probe_curves.png` | 直接在 `out_t` 下 | `run_report.py` `plot_loop_curves` / `plot_probe_curves` |
| `detail.json` | 直接在 `out_t` 下 | `mfuzz/tasks/detection.py` `analyze` |
| `share_ratio.png`、`ablation.png`、`nat_vs_gen.png` | 直接在 `out_t` 下 | `mfuzz/evaluate/det_report.py`，经 `detection.py` `plot_extras` |
| `gen/r###_<stem>.png` 变异图 | `out_t / "gen/..."` | `detection.py` `judge`，第 430 行拼路径，`_save_png` 落盘 |
| `viz/<kind>/<stem>_####.jpg` 标注图 | `out_t / "viz/..."` | `mfuzz/tasks/det_analysis.py` `run_attribution`，第 219 行 |
| `probes/ood_units/ood_units.json` | `out_t / "probes" / "ood_units"` | `mfuzz/probes/ood_units.py` 第 62 行 |

run 根目录（`out`）下的汇总产物，由 `DetectionAdapter.plot_combined` 写出：`failure_counts.png`、`level_distribution.png`、`gt_verdicts.png`、`det_metrics.md`（代码在 `det_report.py`）。

缓存独立于 run：`cache_dir`（配置默认 `output/profiles`）存覆盖标定缓存 `det_<model>_<dataset>_n<N>.pt`，缓存键是"目标模型 + 标定集目录名 + 图数"，跨 run 复用，由 `detection.py` `build_det_profile` 读写。

一个容易踩的点：`detail.json` 里每条失效记录的 `png` 字段存的是变异图的相对路径（`image_ref`），生成失效归因 `attribute_generated`（`det_analysis.py` 第 438 行）又会用 `out_dir / fr.image_ref` 把这张图重新读回来做带图前向。所以变异图的写路径和这个相对路径必须一致，改写路径时要让 `image_ref` 跟着改，读侧才不会断。`viz_rel` 同理写进 `detail.json` 的 `viz` 字段，但目前没有代码再读它，只供人工抽查。

另一处独立写 `viz/` 的是离线工具 `scripts/run_struct_analysis.py`（第 197 行），它把抽查标注图写到 `<out>/<target>/viz/`，逻辑与主管线的 `run_attribution` 平行。

同步脚本 `scripts/sync_lab.ps1` 用 rsync 白名单：push 只推 `mfuzz/`、`scripts/`、`configs/`、`tests/`、`.python-version`；pull 只取 `output*`（`--include=/output*/***`），所以两个根都会被拉回。`output_det/<run>/<model>/gen/` 那约 70M 一份的变异图现在会原样拉回本地。

## 推荐方案：轻量档

我推荐轻量档，并且这次就把它实现掉。它改动小、不动配置、不打断现有的读写链路，同时解决了"样本撑大目录"和"默认不拉回本地"这两个最痛的点。完整档作为后续可选项列在后面，等你拍板再做。

### 目标目录结构（轻量档）

保持两个根不变，只把 run/model 目录内部的两类样本图收进一个统一的 `samples/` 子目录：

```
output/
├── profiles/                       # 覆盖标定缓存，跨 run 复用（不动）
│   └── det_<model>_<dataset>_n<N>.pt
└── cls_imagenet100/<model>/        # 分类 run（不动）

output_det/
└── <run>/                          # 一个实验一个 run，如 base、scale_base
    ├── det_metrics.md              # run 级汇总表（不动）
    ├── failure_counts.png          # run 级汇总图（不动）
    ├── level_distribution.png
    ├── gt_verdicts.png
    └── <model>/                    # 每个目标模型一份
        ├── result.json             # 核心数据
        ├── detail.json
        ├── metrics.md
        ├── loop_curves.png         # 图表
        ├── probe_curves.png
        ├── share_ratio.png
        ├── ablation.png
        ├── nat_vs_gen.png
        ├── samples/                # 新增：所有样本图收进这里
        │   ├── gen/r###_<stem>.png     # 生成的变异图（原 gen/）
        │   └── viz/<kind>/...jpg        # 抽查标注图（原 viz/）
        └── probes/                 # 探针产物（不动）
            └── ood_units/ood_units.json
```

变化只有一处：原来直接挂在 model 目录下的 `gen/` 和 `viz/` 合并到 `samples/gen/` 和 `samples/viz/`。核心数据、图表、探针的位置都不动，重跑命令、读 `result.json` 的聚合脚本完全不受影响。

### 需要改动的文件与具体位置（轻量档）

1. `mfuzz/tasks/detection.py` 第 430 行：变异图相对路径前缀从 `gen/` 改成 `samples/gen/`。因为这个字符串同时是 `image_ref`，`attribute_generated` 用 `out_dir / image_ref` 重读，写路径一改，读路径自动跟上，不用再动读侧。
2. `mfuzz/tasks/det_analysis.py` 第 219 行：抽查标注图相对路径前缀从 `viz/` 改成 `samples/viz/`。
3. `scripts/run_struct_analysis.py` 第 197 行：同样把 `viz/` 改成 `samples/viz/`，让离线工具与主管线一致。
4. `scripts/sync_lab.ps1` 的 `$PullFilter`：在 `--include=/output*/***` 之前加一条 `--exclude=*/samples/`（或更精确地 `--exclude=/output_det/*/*/samples/`），让 pull 默认跳过样本图。需要某个模型的样本时，临时改 filter 或用 `-RemoteHost` 之外的单独 rsync 命令拉。

`save_viz` / `_save_png` 里都有 `out_file.parent.mkdir(parents=True, exist_ok=True)`，多一级 `samples/` 目录会被自动创建，不用额外建目录。

### 重跑与同步如何使用新结构

重跑实验的命令完全不变，仍然是 `uv run python scripts/run_fuzz.py --config configs/det/<exp>.toml`。配置里的 `[run] out` 不用改，样本会自动落到 `output_det/<run>/<model>/samples/` 下。

同步时，`pwsh scripts/sync_lab.ps1 pull` 默认拉回核心数据、图表、探针，但跳过 `samples/`，本地目录因此保持精简。需要在本地看某次 run 的变异图或标注图时，再用一条带具体路径的 rsync（或临时把 `--exclude=*/samples/` 去掉）单独拉那一份。push 方向不涉及 output，不用动。

## 备选方案：完整档（先不实现）

完整档把两个根并到 `output/` 之下，缓存独立成 `cache/`，每个 run/model 再按用途分四个子目录。它结构最清晰，但要改全部配置的 `[run] out`、改 `cache_dir`、改报告代码里多处直接挂在 `out_t` 下的写路径，还要同步更新文档和同步脚本，改动面大。建议等轻量档跑通、确认重跑顺畅后再决定是否升级。

### 目标目录结构（完整档）

```
output/
├── cache/
│   └── profiles/det_<model>_<dataset>_n<N>.pt   # 原 output/profiles
├── det/
│   └── <run>/
│       ├── _run/                                # run 级汇总（图 + det_metrics.md）
│       └── <model>/
│           ├── data/      result.json、detail.json、metrics.md
│           ├── figures/   loop_curves、share_ratio、ablation、nat_vs_gen 等
│           ├── samples/   gen/、viz/
│           └── probes/    ood_units/...
└── cls/
    └── <run>/<model>/{data,figures,probes}
```

### 完整档需要改的地方

- 全部 `configs/**/*.toml` 的 `[run] out`：`output_det/<run>` → `output/det/<run>`，`output/cls_*` → `output/cls/<run>`。
- 全部配置的 `cache_dir`：`output/profiles` → `output/cache/profiles`。
- `mfuzz/evaluate/run_report.py`：`result.json`、`metrics.md` 落 `data/` 子目录，曲线图落 `figures/`。
- `mfuzz/evaluate/det_report.py` 与 `detection.py` `plot_extras` / `plot_combined`：任务图表落 `figures/`，run 级汇总落 `_run/`。
- `detection.py` `analyze`：`detail.json` 落 `data/`。
- `gen/`、`viz/` 同轻量档收进 `samples/`。
- `sync_lab.ps1`：pull filter 改成 `output/`，并排除 `samples/`。
- 聚合脚本（`mfuzz/evaluate/`、`scripts/` 里读 `result.json` 的部分）：读路径加 `data/` 一级。
- 文档：`CLAUDE.md` 的目录结构说明、`docs/实验登记.md` 的路径。

因为牵动配置和聚合脚本的读路径，完整档要和重跑、聚合一起整体切换，不适合零敲碎打。
