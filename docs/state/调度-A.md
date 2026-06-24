---
orchestrator: A
domain: 实验、写作、可视化、审计
---

## 当前计划 (完成后清理内容)

Day10。5 待定项按推荐锁定：解冻整子网（单层留消融）、loc 用 `1-IoU`（GIoU 留变体）、cls 同时推正类压错类（仅两通道，只推正类留变体）、学习率按权重范数归一化、步数 1/3/10/30 × 学习率 1e-4/1e-3/1e-2。

- 实验链路（本轮全做完）：入口实现 → 我审查 → 服务器 smoke 通过 → 全量 OOM → 梯度累加修复（本地逐元素等价 + 服务器显存恒定 3.7/3.8 GiB 双验）→ **fcos 全量（A=500、loc+cls、responsible+random）已上 GPU2 后台跑**，健康进 sweep、显存稳定，等数小时出结果。GPU 作业见 BOARD 队列。
- bottom 对照缺口：服务器无 base/fcos 的 layer_drilldown，本轮跳过。补法二选一：跑 max_iterations=0 的分析配置出 aggregate.layer_drilldown，或脚本内联算 _layer_drilldown。首轮诊断用 responsible+random 可接受。
- 教训：新训练代码的 smoke 要带满步数档看峰值显存，别只验正确性——这次 OOM 就是小 smoke 漏掉的。
- 审计、可视化、写作：本轮不动。

## 正在运行的 worker (完成后清理内容)

| worker | 流 | 工作区 | 任务 | 状态 | 预计 |
|--------|----|--------|------|------|------|
| （无活跃 worker；fcos 全量是后台 GPU 作业，见 BOARD 队列） | | | | | |

## 检查与合并记录 (完成后清理内容)

| worker | 流 | 检查结论 | 是否已合并 | 下一步 |
|--------|----|----------|-----------|--------|
| 实验-修OOM | 实验 | 梯度累加修复正确：本地逐元素等价（atol 1e-6）、服务器 A=100 显存恒定 3.7/3.8 GiB。我读 diff 确认。已提交 3e9031e。 | 已合并 | 全量 |
| 实验-全量launch（A=500） | 实验 | fcos 全量健康进 sweep、显存稳定不再 OOM、B 基线与首跑一致（确定性）。GPU2 后台跑、PID 2458268/2458372。记录在 实验.md 待合并。 | 记录待合并 | 等结果→拉回分析前沿 |


## 下次开工起点 (完成后填写)

- 实验：5 待定项已锁定、入口已实现并修好 OOM。**fcos 定向微调全量（A=500、loc+cls、responsible+random）在服务器 GPU2 后台跑**（日志 output/det/repair_finetune/run_fcos_full.log，PID 2458268/2458372，BOARD 有队列）。下一步：跑完用 `pwsh -File scripts/sync_lab.ps1 pull -Apply` 拉 output，看 output/det/repair_finetune/{loc_fcos,cls_fcos}/data/result.json 与 frontier.png，按框架判据读"责任 head.regression_head/classification_head vs 随机层"的代价收益前沿（同等 agree 损失下责任修复幅度是否明显超随机）。之后补三件：bottom 对照（drilldown 缺口见当前计划）、retinanet 同协议、按首轮落点调步数/学习率档（首点 s1/lr1e-4 几乎不动，落点偏低代价端）。方差（队列 3）留到点结果定稿后。对照纪律与判据见 docs/paper_plan/定位与修复验证框架.md。
- 审计：模块 1-4 完成。下一个模块 5（engine/loop.py 与各 adapter 的机制/评测分层）。两条待用户定的建议（均非正确性 bug）：统一分位边界 `>=`/`>`（检测侧离散 freq 在打结处对 `>=` 更脆）；spec 第七节补一句"频率/obj_cov 的 out 实指归一化激活 ĉ"。
- 可视化：等实验产出真实 result.json 后在真数据上复核 summarize_det；四类统一表仍等实验。
- 写作：仍挂起，等用户带回外部深度研究核心论文，用 writing-worker。
- 工作区：本会话实验、审计、default 三处有改动并合并到 linux；合并后把全部 5 个工作区停到新 linux 头。坑：推进 linux 后要及时把空闲工作区 rebase/jj new 到新头，别在落后工作区上改文件。

## 等待用户决定

- 实验 5 待定项已锁定（见"当前计划"），不再等待。
- 审计两条建议是否改代码（非阻塞、低优先）：统一分位边界 `>=`/`>`、spec 补"out 实指归一化激活"一句。
