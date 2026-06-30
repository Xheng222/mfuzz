---
orchestrator: A
domain: 实验、写作、可视化、审计
---

## 当前计划 (完成后清理内容)

修复线转向论文级全量验证（两步走）。起因：retina loc 用归因指向的 FPN 责任结构重跑仍无定论（loc 1/12、不与对照分开），反思定位到实验设置/数据问题——训练数据对全部触发图盲分、被 miss 主导、loc 监督稀薄；save_failures=400 还截断了 loc/cls 触发图落盘。

- 第一步（今天，运行中）：regen_full 全量四模型数据生成。GPU2、健康（标定缓存命中、5000 种子基线）。明天 pull。
- 第二步（明天）：按失效类别筛可复用微调数据集；改 run_repair_finetune.py 的 _build_ab_paths 让 A/B 按 kind 选取（集中 loc/cls 监督）、带容量配平对照、报 loc 框平均 IoU；重做 loc/cls 修复验证。

基础设施变更：root 盘 100% 满（他人占用），output 迁到 NAS 符号链接（/home/nas511），sync pull 加 --copy-dirlinks 跟随。YOLO 确认可作 fuzz 目标（forward_graph），已纳入四模型全目标；AGENTS.md 旧"YOLO 黑盒投票者"说明已改正。

## 正在运行的 worker / 作业 (完成后清理内容)

| 作业 | 流 | 位置 | 状态 | 预计 |
|------|----|------|------|------|
| regen_full 全量四模型数据生成 | 实验 | 服务器 GPU2，worker PID 376477 | 运行中、健康（首跑 OOM 已修） | 数小时~十几小时，明天 pull |

注：首跑（PID 371538）在 5000 种子基线 CUDA OOM——变异图尺寸不一致致 CUDA 分配器碎片化（非泄漏），加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 重挂后显存稳定约 10GiB、越过原 OOM 点；num_images=5000 未降。loguru 日志仍像有缓冲，用进程状态 + result.json 判进度。

## 检查与合并记录 (完成后清理内容)

本会话各流成果已检查、今日合并进 linux（经用户同意）：

| 流 | 成果 | 检查 |
|----|------|------|
| 审计 | 模块 5（loop/adapter 分层，独立 grep 证实零耦合）+ 两条建议落实（profiler `cl>=thr`、spec 补 out=ĉ，ruff 绿） | 通过 |
| 可视化 | summarize_det 真数据复验（两 regen 行对得上）+ 修 schema-drift + 四类表+前沿图 | 通过 |
| 写作 | 研究现状 W1 文献地图 + W2 第一批正文（约 30 篇、待核实文献剔除）；W3 阻塞 | 通过 |
| 实验 | --responsible 覆盖 + regen_full 全量配置 + sync --copy-dirlinks 修复 + NAS 迁移 + retina-FPN 无定论与反思记录 | 通过 |

## 下次开工起点 (完成后填写)

- 实验：pwsh sync_lab.ps1 pull -Apply（已带 --copy-dirlinks）拉回 output/det/regen_full/<model>/data。先看四模型各失效类别产量（loc/cls 触发图够不够厚）。再做第二步：改 _build_ab_paths 按 kind 选 A/B、容量配平对照、报平均 IoU，重做 loc/cls 修复验证。注意 regen_full 在 GPU2 跑（worker PID 376477，expandable_segments 版），先确认跑完（进程退 + 四个 result.json 落地）再 pull 读；若又 OOM 看 run_regen_full.oom.log 对比。
- 可视化：regen_full 数据落地后，summarize_det 直接产四模型 fuzzing campaign 对照表 + 四曲线（论文图）。四类修复表等第二步修复前沿。
- 写作：W1+W2 完成。W3 阻塞——需用户用 docs/paper_plan/Day6/研究现状-深度研究提示词.md 第二批提示词重跑深度研究（结构归因+模型编辑+DeepFault），放 references/deeepresearch/。W4、W5 在其后。
- 审计：模块 1–5 完成 + 两条建议落实。下一个模块 6（optimize joint.py/feedback.py，spec 4.4）。
- 服务器/基础设施：output 是符号链接 → /home/nas511/dongyajie/mfuzz_output；root 盘满、写盘走 NAS。pull 已修复跟随符号链接。

## 等待用户决定

- 用户重跑第二批深度研究以解阻 W3。
