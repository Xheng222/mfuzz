---
orchestrator: A
domain: 实验、写作、可视化、审计
---

## 当前计划 (完成后清理内容)

修复线转向论文级全量验证（两步走）。起因：retina loc 用归因指向的 FPN 责任结构重跑仍无定论（loc 1/12、不与对照分开），反思定位到实验设置/数据问题——训练数据对全部触发图盲分、被 miss 主导、loc 监督稀薄；save_failures=400 还截断了 loc/cls 触发图落盘。

第一步全量生成期间出过两处 GPU OOM，都已在代码层根治（不再绑运维环境变量）：初始覆盖遍历的 make_batches 一次性把全部种子图载入显存，改惰性生成器；run_baseline 等变长图过 CUDA 分配器碎片化，把 expandable_segments 下沉到 run_fuzz 代码并给 ablate_levels 补逐图释放。独立复核 worker 确认这两处是仅有的非流式/碎片化通道、修复到位。另外发现初始覆盖率过高（CNCov_0=0.90）：覆盖是种子并集、随种子数饱和，t_cov=0.85 是 200 种子时标定的，5000 种子下失去增长空间。在 5000 种子上用 calibrate_det_tcov 重扫，t_cov 改 0.95（四模型初始覆盖统一回 0.30–0.31，t_freq/critical_tau 不变）。

- 第一步（今天，重跑中）：regen_full 全量四模型数据生成，t_cov 重标 0.95。GPU0、构建阶段。明天 pull。
- 第二步（明天）：按失效类别筛可复用微调数据集；改 run_repair_finetune.py 的 _build_ab_paths 让 A/B 按 kind 选取（集中 loc/cls 监督）、带容量配平对照、报 loc 框平均 IoU；重做 loc/cls 修复验证。

基础设施变更：root 盘 100% 满（他人占用），output 迁到 NAS 符号链接（/home/nas511），sync pull 加 --copy-dirlinks 跟随。YOLO 确认可作 fuzz 目标（forward_graph），已纳入四模型全目标；AGENTS.md 旧"YOLO 黑盒投票者"说明已改正。

覆盖增长图与数据集解耦（本会话定）：regen_full 的 5000 种子让初始覆盖饱和、覆盖曲线天生平，它只当结构层缺陷定位的数据集、覆盖不进正文；覆盖有效性图另起一条 cov_growth_1000（1000 种子 @ t_cov=0.85、150 轮、faster_rcnn/retinanet/fcos 三模型）。选 1000@0.85 的依据是一轮种子数×t_cov 扫描：t_cov 定天花板、种子数定起点，只有 t_cov≲0.86 能让曲线爬过 0.8；1000@0.85 起点 0.691、终点 0.897、过线最干净。全篇 t_cov 统一 0.85，regen_full 跑在 0.95 不为一致性重跑（其覆盖不上报）。清理了被取代的旧全量实验 output/det/regen（139M，2026-06-24、50 轮旧配置）。扫描细节与曲线见实验流。

## 正在运行的 worker / 作业 (完成后清理内容)

| 作业 | 流 | 位置 | 状态 | 预计 |
|------|----|------|------|------|
| regen_full 全量四模型数据生成（结构层缺陷定位数据集，t_cov=0.95） | 实验 | 服务器 GPU0，启动器 PID 403065、worker PID 403071 | 运行中；faster_rcnn+retinanet 已出 result.json，进 fcos | 数小时~十几小时，明天 pull |
| cov_growth_1000 三模型覆盖增长图（1000 种子 @ t_cov=0.85、150 轮） | 实验 | 服务器 GPU2，PID 3326717 | 运行中；faster_rcnn 构建中 | 约 2 小时 |

注：经 calibrate_det_tcov 在 5000 种子上重扫，t_cov 由 0.85 改 0.95（清旧产物后经 run_lab_experiment.py 启动器重跑）。启动器自动选空闲卡 + 设 expandable_segments + 失败重试 + 整目标续跑（run_fuzz 对已落 result.json 的目标跳过）。两处 OOM 已在代码层根治，不再依赖运维环境变量。loguru 日志块缓冲，用进程状态 + result.json 判进度。

## 检查与合并记录 (完成后清理内容)

本会话各流成果已检查、今日合并进 linux（经用户同意）：

| 流 | 成果 | 检查 |
|----|------|------|
| 审计 | 模块 5（loop/adapter 分层，独立 grep 证实零耦合）+ 两条建议落实（profiler `cl>=thr`、spec 补 out=ĉ，ruff 绿） | 通过 |
| 可视化 | summarize_det 真数据复验（两 regen 行对得上）+ 修 schema-drift + 四类表+前沿图 | 通过 |
| 写作 | 研究现状 W1 文献地图 + W2 第一批正文（约 30 篇、待核实文献剔除）；W3 阻塞 | 通过 |
| 实验 | --responsible 覆盖 + regen_full 全量配置 + sync --copy-dirlinks 修复 + NAS 迁移 + retina-FPN 无定论与反思记录 | 通过 |

本会话后续又合并两批进 linux（OOM 根因修复 + t_cov 重标），linux 现位于 a46a644d：

- make_batches 改惰性生成器修初始覆盖 all-at-once OOM + 稳健启动器 run_lab_experiment.py（yxqpwyzu 29d2f19d）
- 碎片化防护下沉到 run_fuzz 代码（import torch 前 setdefault expandable_segments）+ ablate_levels 逐图释放（xvkoloyr 09ad2bfc）；独立复核 worker 确认两处根因、补这两处残留缺口
- regen_full t_cov 0.85→0.95 重标（a46a644d）

## 下次开工起点 (完成后填写)

- 实验：pwsh sync_lab.ps1 pull -Apply（已带 --copy-dirlinks）拉回 output/det/regen_full/<model>/data。先看四模型各失效类别产量（loc/cls 触发图够不够厚）。再做第二步：改 _build_ab_paths 按 kind 选 A/B、容量配平对照、报平均 IoU，重做 loc/cls 修复验证。注意 regen_full 是 t_cov=0.95 重标版、在 GPU0 跑（启动器 PID 403065、worker PID 403071），先确认跑完（进程退 + 四个 result.json 落地）再 pull 读。读 result.json 时核一眼 CNCov_0 应在 0.30–0.31（不是旧的 0.90）、循环有覆盖增长。另：cov_growth_1000（GPU2、PID 3326717）是覆盖图专用跑，跑完 pull output/det/cov_growth_1000/<model>/data/result.json 读 cncov_history，出 faster_rcnn/retinanet/fcos 三模型覆盖增长图（论文覆盖有效性图，1000 种子 @ t_cov=0.85）。faster_rcnn 单模型曲线已验证 0.691→0.897、第 ~25 轮过 0.8。
- 可视化：regen_full 数据落地后，summarize_det 直接产四模型 fuzzing campaign 对照表 + 四曲线（论文图）。四类修复表等第二步修复前沿。
- 写作：W1+W2 完成。W3 阻塞——需用户用 docs/paper_plan/Day6/研究现状-深度研究提示词.md 第二批提示词重跑深度研究（结构归因+模型编辑+DeepFault），放 references/deeepresearch/。W4、W5 在其后。
- 审计：模块 1–5 完成 + 两条建议落实。下一个模块 6（optimize joint.py/feedback.py，spec 4.4）。
- 服务器/基础设施：output 是符号链接 → /home/nas511/dongyajie/mfuzz_output；root 盘满、写盘走 NAS。pull 已修复跟随符号链接。

## 等待用户决定

- 用户重跑第二批深度研究以解阻 W3。
