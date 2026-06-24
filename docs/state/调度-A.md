---
orchestrator: A
domain: 实验、写作、可视化、审计
---

## 当前计划 (完成后清理内容)

Day10。修复线做了一次根本转向：原来在干净 val2017 上取跨模型自然分歧做微调，数据基础不成立——缺陷来自 fuzz，数据却是分布内、不触发缺陷的，且模型训练已在该分布上饱和，轻量微调修不动。改成闭环：fuzz 产出缺陷、结构归因定位、轻量定向微调修复，全部在生成失效上做；自然分歧降为负基线。框架文档已按此重写（docs/paper_plan/定位与修复验证框架.md）。

- 旧的自然分歧版 fcos 全量定向微调已停（服务器进程已 kill，BOARD 队列移除）。
- 产物前提：生成失效与归因在早先 output 清理时被删，必须先跑 regen 恢复。已派后台 worker 把省着版 regen（configs/det/regen.toml：标定缩 3000、fcos+retinanet）挂上服务器，worker 已确认健康越过标定、进 fuzzing 循环在产失效后返回。
- 写作挂起、可视化暂停、审计本轮不动。

## 正在运行的 worker (完成后清理内容)

| worker | 流 | 工作区 | 任务 | 状态 | 预计 |
|--------|----|--------|------|------|------|
| regen 产物再生 | 实验 | 实验 | 服务器跑省着版 fuzz 恢复生成失效+归因 | 已返回（确认健康起跑）；服务器 nohup 作业仍在跑、跑完不通知 | 整体约 2h |

## 检查与合并记录 (完成后清理内容)

| worker | 流 | 检查结论 | 是否已合并 | 下一步 |
|--------|----|----------|-----------|--------|
| regen 产物再生 | 实验 | worker 贴日志确认：3000 张缩标定生效（新缓存 n3000、不命中 n118287 旧全量）、两遍流式标定无 OOM、fcos 进循环产失效（轮 0→40 覆盖 0.839→0.968、新失效持续）、显存稳 ~4.8GiB。GPU0、PID 2975624、日志 output/det/regen/run_regen.log。 | 框架文档+regen.toml+实验.md 已封存进 linux | 等 result.json 落地→拉回 |


## 下次开工起点 (完成后填写)

- 实验：regen 在服务器后台跑（省着版，fcos+retinanet，GPU0，日志 output/det/regen/run_regen.log）。约两小时后产物落 output/det/regen/{fcos,retinanet}/data/result.json（生成失效 + aggregate.layer_drilldown）。开工三步：① `pwsh -File scripts/sync_lab.ps1 pull -Apply` 拉产物，确认 result.json 与 layer_drilldown 落地；② 把 scripts/run_repair_finetune.py 改成从生成失效加载触发输入——监督用变异前的正确检测（不是 GT、不是自然分歧共识），评测看触发输入上的缺陷率（现版本从干净 val2017 取自然分歧，已废弃）；③ 在生成失效上重做四类（虚检/漏检/误分类/定位偏移）的定位与修复，复用责任子网/对照层/损失/协议骨架（设计稿 docs/paper_plan/定向微调试点实验设计.md），自然分歧旧结论降为负基线。判据与对照纪律见 docs/paper_plan/定位与修复验证框架.md。
- 审计：模块 1-4 完成。下一个模块 5（engine/loop.py 与各 adapter 的机制/评测分层）。两条待用户定的建议（均非正确性 bug）：统一分位边界 `>=`/`>`（检测侧离散 freq 在打结处对 `>=` 更脆）；spec 第七节补一句"频率/obj_cov 的 out 实指归一化激活 ĉ"。
- 可视化：等生成失效的真实 result.json 后在真数据上复核 summarize_det；四类统一表仍等实验。
- 写作：仍挂起，等用户带回外部深度研究核心论文，用 writing-worker。
- 工作区：本轮实验工作区改动（框架文档+regen.toml+实验.md）与主检出调度/BOARD 改动已合并到 linux，5 个工作区停到新 linux 头。坑：推进 linux 后要及时把空闲工作区 rebase/jj new 到新头，别在落后工作区上改文件。

## 等待用户决定

- 修复线闭环范围用户已确认（改框架文档、自然分歧降负基线），不再等待。
- 审计两条建议是否改代码（非阻塞、低优先）：统一分位边界 `>=`/`>`、spec 补"out 实指归一化激活"一句。
