---
orchestrator: A
domain: 实验、写作、可视化、审计
---

## 下次开工起点

- 实验：重跑误分类归因。除了把归因层换到分类子网，还要把归因目标从 `scores_g` 改成 NMS 前的 logit（依据审计流的 scores_g 结论），借此一并验证"用概率导致归因弱"这条因果假设。被停的实验 worker 无法恢复，需重新派发。
- 写作：等用户带回外部深度研究的核心论文后，写研究现状正文。届时用 writing-worker，需先重载会话才能识别这个自定义 subagent。
- 可视化：编写汇总脚本，读取新结构 `output/det/<run>/<model>/data/result.json`，产出对照表与曲线。
- 收尾小项：`compare.py` 等离线读取侧的 `目录/result.json` 要改成 `data/result.json`（output 重整遗留，不挡重跑）。

## 等待用户决定

- 暂无。
