---
orchestrator: A
domain: 实验、写作、可视化、审计
---

## 当前计划 (完成后清理内容)

Day10。5 待定项按推荐锁定：解冻整子网（单层留消融）、loc 用 `1-IoU`（GIoU 留变体）、cls 同时推正类压错类（仅两通道，只推正类留变体）、学习率按权重范数归一化、步数 1/3/10/30 × 学习率 1e-4/1e-3/1e-2。

- 实验（主线）进展：①入口实现 + 我审查通过。②服务器小 smoke（fcos 20/20）通过：真 torchvision 半前向无 API 不匹配、cls 回找在真 COCO 命中（n_inst>0）、权重恢复正确。③用户选"fcos 先跑一轮（loc+cls 全格点+bottom）"，全量首跑 **OOM**：train_one_config 把损失图跨 A 集 500 图累加、最后才一次 backward，500 张前向图同压显存 ~21.6 GiB 撑爆 24G 卡。smoke 太小没暴露。④已派 worker 改梯度累加（每图 backward 释放图、grad 累加、按 n_terms 归一后 step），本地 lint/smoke + 服务器小规模显存检查，**不跑全量**，等我审查修复 + 显存结果后再跑全量。
- bottom 对照缺口：服务器上**没有 base/fcos 跑、grep output 无 layer_drilldown**，bottom 自动跳过、首轮只能 responsible+random。需要后续补：要么跑一个 max_iterations=0 的分析配置出 aggregate.layer_drilldown，要么在 finetune 脚本内联算 _layer_drilldown。首轮诊断用 responsible+random 可接受。
- 教训：smoke 只验了正确性、没验显存随规模累加；以后新训练代码的 smoke 要带满步数档看峰值显存。
- 审计、可视化、写作：本轮不动。

## 正在运行的 worker (完成后清理内容)

| worker | 流 | 工作区 | 任务 | 状态 | 预计 |
|--------|----|--------|------|------|------|
| 实验-修OOM | 实验 | F:/doc/安全缺陷/工作区/实验 | 改梯度累加修 OOM + 本地验证 + 服务器小规模显存检查（不跑全量） | 后台运行 | - |

## 检查与合并记录 (完成后清理内容)

| worker | 流 | 检查结论 | 是否已合并 | 下一步 |
|--------|----|----------|-----------|--------|
| 实验-实现 | 实验 | 代码合格：finetune.py + run_repair_finetune.py，loc 端到端、cls 半前向回找机制完整；依赖都对得上。已提交 8fc2829。 | 已合并 | — |
| 实验-跑smoke | 实验 | fcos 20/20 smoke 通过：真半前向无 API 问题、cls 回找命中、权重恢复正确。未改代码。 | 记录待合并 | 全量 |
| 实验-全量launch | 实验 | 全量首跑 OOM（损失图跨 A 集累加）；服务器无 base/drilldown，bottom 跳过。未出结果。 | 记录待合并 | 修 OOM 后重跑 |


## 下次开工起点 (完成后填写)

- 实验：定向微调试点设计稿已出并合并，见 docs/paper_plan/定向微调试点实验设计.md。**派 GPU 作业前先拍板 5 个待定项**（设计稿第十节）：解冻粒度（整子网 vs `conv.0` 单层，倾向整子网、单层留作消融）、loc 损失形式（`1-IoU` vs GIoU，倾向 `1-IoU`）、cls 监督通道（只推正类 vs 同时压错类）、学习率是否按责任子网权重范数归一化、步数/学习率档取值。核心新代码：run_repair_finetune.py（只解冻指定子网多步微调入口）+ cls 的 NMS 前 logit 损失路径（从失效实例回找 anchor 位置）+ 两条对照层（随机层、最低归因层）选层。对照纪律与判据见 docs/paper_plan/定位与修复验证框架.md。GPU 空闲。
- 审计：模块 1-4 完成。下一个模块 5（engine/loop.py 与各 adapter 的机制/评测分层）。两条待用户定的建议（均非正确性 bug）：统一分位边界 `>=`/`>`（检测侧离散 freq 在打结处对 `>=` 更脆）；spec 第七节补一句"频率/obj_cov 的 out 实指归一化激活 ĉ"。
- 可视化：等实验产出真实 result.json 后在真数据上复核 summarize_det；四类统一表仍等实验。
- 写作：仍挂起，等用户带回外部深度研究核心论文，用 writing-worker。
- 工作区：本会话实验、审计、default 三处有改动并合并到 linux；合并后把全部 5 个工作区停到新 linux 头。坑：推进 linux 后要及时把空闲工作区 rebase/jj new 到新头，别在落后工作区上改文件。

## 等待用户决定

- 实验 5 待定项已锁定（见"当前计划"），不再等待。
- 审计两条建议是否改代码（非阻塞、低优先）：统一分位边界 `>=`/`>`、spec 补"out 实指归一化激活"一句。
