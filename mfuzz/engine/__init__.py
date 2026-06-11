"""engine：任务无关的反馈驱动主循环。

- loop：run_loop 通用循环（调度 → 变异 → 判定 → 反馈），任务语义在
  mfuzz/tasks 的适配器后面，观测走 core/probe 的事件总线；含统一种子池
  GenericSeedPool（多维优先级与退役）。
"""
