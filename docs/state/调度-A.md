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
| （无活跃 worker；loc generated 全量是后台 GPU 作业，见 BOARD 队列） | | | | | |

## 检查与合并记录 (完成后清理内容)

| worker | 流 | 检查结论 | 是否已合并 | 下一步 |
|--------|----|----------|-----------|--------|
| regen 产物再生 | 实验 | 跑完。worker 确认健康起跑；调度者第二确认独立验证产物：fcos 738 / retina 575 失效、覆盖 0.971/0.979，result.json 有 aggregate.layer_drilldown，detail.json failures[] 带 anchor（共识正确框=监督目标）/seed_image/png，触发图 samples/gen（fcos 99 / retina 107 PNG，n_gen_saved=400）。修复线数据齐备。 | 已拉回；实验.md 记录与设定待合并 | 改 run_repair_finetune.py 从生成失效加载 |


## 下次开工起点 (完成后填写)

- 实验：✓ 代码已改读生成失效并 smoke 通过；✓ 读了生成失效归因下钻（loc 干净指向 head.regression_head/fcos、retina loc 偏 FPN、cls 不可分且稀）。**loc generated 全量 sweep 在服务器 GPU1 后台跑**（fcos→retina 串行，三对照，日志 output/det/repair_finetune/run_loc_gen.log，PID 3726035，产物 output/det/repair_finetune/generated_loc_{fcos,retinanet}/data）。下一步：跑完 `pwsh -File scripts/sync_lab.ps1 pull -Apply` 拉回，读 frontier.png 与 result.json，按判据看同等 agree 损失下 responsible=head.regression_head 是否明显超 random/bottom——这是 loc 定位的第二确认（worker 产出 + 我复核两个 agent）。cls 暂定负向案例（不可分+稀疏，densify 低价值），按 2×2 报告；虚检/漏检的免训练抑制因果结论保留。对照纪律与判据见 docs/paper_plan/定位与修复验证框架.md。
- 审计：模块 1-4 完成。下一个模块 5（engine/loop.py 与各 adapter 的机制/评测分层）。两条待用户定的建议（均非正确性 bug）：统一分位边界 `>=`/`>`（检测侧离散 freq 在打结处对 `>=` 更脆）；spec 第七节补一句"频率/obj_cov 的 out 实指归一化激活 ĉ"。
- 可视化：等生成失效的真实 result.json 后在真数据上复核 summarize_det；四类统一表仍等实验。
- 写作：仍挂起，等用户带回外部深度研究核心论文，用 writing-worker。
- 工作区：本轮实验工作区改动（框架文档+regen.toml+实验.md）与主检出调度/BOARD 改动已合并到 linux，5 个工作区停到新 linux 头。坑：推进 linux 后要及时把空闲工作区 rebase/jj new 到新头，别在落后工作区上改文件。

## 等待用户决定

- 修复线闭环范围用户已确认（改框架文档、自然分歧降负基线），不再等待。
- 审计两条建议是否改代码（非阻塞、低优先）：统一分位边界 `>=`/`>`、spec 补"out 实指归一化激活"一句。
