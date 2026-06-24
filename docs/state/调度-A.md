---
orchestrator: A
domain: 实验、写作、可视化、审计
---

## 当前计划 (完成后清理内容)

Day10。5 待定项按推荐锁定：解冻整子网（单层留消融）、loc 用 `1-IoU`（GIoU 留变体）、cls 同时推正类压错类（仅两通道，只推正类留变体）、学习率按权重范数归一化、步数 1/3/10/30 × 学习率 1e-4/1e-3/1e-2。

- 实验（主线）：定向微调入口已实现（finetune.py + run_repair_finetune.py），本地 lint + 逻辑 smoke 通过，我已审查代码并核实关键依赖（categories/judge_image/graph_index/layer_drilldown 都对得上）。下一步**上服务器先跑小 smoke**（少图少格点，验证真权重的 HeadLogits 半前向、cls 回找命中、judge 链路），smoke 过再跑全量扫描。bottom 对照层的 drilldown 从已有 base 跑的 extra.det.aggregate.layer_drilldown 取（格式已对齐）。
- 服务器运行决定：新代码本地无法对真模型/真 COCO 验证，所以分两段——先 20 图级 smoke 抓 API/链路 bug，再 500 图全量扫描，避免在未验证代码上烧串行 GPU。
- 审计、可视化、写作：本轮不动。

## 正在运行的 worker (完成后清理内容)

| worker | 流 | 工作区 | 任务 | 状态 | 预计 |
|--------|----|--------|------|------|------|
| 实验-跑smoke | 实验 | F:/doc/安全缺陷/工作区/实验 | 上服务器跑小 smoke（少图少格点）验证真模型 API 与 cls 回找链路 | 派发中 | - |

## 检查与合并记录 (完成后清理内容)

| worker | 流 | 检查结论 | 是否已合并 | 下一步 |
|--------|----|----------|-----------|--------|
| 实验-实现 | 实验 | 代码合格：finetune.py + run_repair_finetune.py，loc 端到端、cls 半前向回找机制完整。我审查并核实 categories/judge_image/graph_index/layer_drilldown 依赖都对得上。本地 lint+逻辑 smoke 过；真模型 API 与 cls 命中率只能服务器验。 | 待合并 | 上服务器小 smoke → 全量扫描 |


## 下次开工起点 (完成后填写)

- 实验：定向微调试点设计稿已出并合并，见 docs/paper_plan/定向微调试点实验设计.md。**派 GPU 作业前先拍板 5 个待定项**（设计稿第十节）：解冻粒度（整子网 vs `conv.0` 单层，倾向整子网、单层留作消融）、loc 损失形式（`1-IoU` vs GIoU，倾向 `1-IoU`）、cls 监督通道（只推正类 vs 同时压错类）、学习率是否按责任子网权重范数归一化、步数/学习率档取值。核心新代码：run_repair_finetune.py（只解冻指定子网多步微调入口）+ cls 的 NMS 前 logit 损失路径（从失效实例回找 anchor 位置）+ 两条对照层（随机层、最低归因层）选层。对照纪律与判据见 docs/paper_plan/定位与修复验证框架.md。GPU 空闲。
- 审计：模块 1-4 完成。下一个模块 5（engine/loop.py 与各 adapter 的机制/评测分层）。两条待用户定的建议（均非正确性 bug）：统一分位边界 `>=`/`>`（检测侧离散 freq 在打结处对 `>=` 更脆）；spec 第七节补一句"频率/obj_cov 的 out 实指归一化激活 ĉ"。
- 可视化：等实验产出真实 result.json 后在真数据上复核 summarize_det；四类统一表仍等实验。
- 写作：仍挂起，等用户带回外部深度研究核心论文，用 writing-worker。
- 工作区：本会话实验、审计、default 三处有改动并合并到 linux；合并后把全部 5 个工作区停到新 linux 头。坑：推进 linux 后要及时把空闲工作区 rebase/jj new 到新头，别在落后工作区上改文件。

## 等待用户决定

- 实验 5 待定项已锁定（见"当前计划"），不再等待。
- 审计两条建议是否改代码（非阻塞、低优先）：统一分位边界 `>=`/`>`、spec 补"out 实指归一化激活"一句。
