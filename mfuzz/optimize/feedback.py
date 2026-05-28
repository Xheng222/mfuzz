"""动态反馈（实现方案 4.4）。

滑动窗口统计四个指标：覆盖增长率 ΔCNCov、RFT、输入语义偏移均值 d̄_sem、路径新颖样本
比例。按规则逐轮调权重：

- 覆盖停滞（窗口内 ΔCNCov 均值低于阈）→ 提 λ2，把更多力气推向覆盖。
- 语义偏移超阈（窗口内 d̄_sem 均值高于阈）→ 提 λ3，收紧语义、语义质量优先于覆盖扩展。
- 两者都未触发 → 对应 λ 以较小因子回落，避免权重单调爬到上界后失去调节余地。

权重用乘法因子小步调，夹在 lambda*_bounds 内。λ 历史由调用方记录。enabled=False 时
原样返回初值，给出静态权重基线。

反馈规则的阈值与乘法因子是 4.4 定性规则的具体化，见实现方案第十章 Phase 4 偏离记录。
"""

from __future__ import annotations

from collections import deque

from mfuzz.core.types import FeedbackConfig, FeedbackState


class FeedbackController:
    """持有当前 λ2 / λ3 与四指标滑动窗口，每轮按反馈规则更新权重。"""

    def __init__(
        self,
        config: FeedbackConfig,
        lambda2_init: float,
        lambda3_init: float,
        sem_shift_threshold: float,
        lambda2_bounds: list[float],
        lambda3_bounds: list[float],
    ) -> None:
        self.config = config
        self.lambda2 = lambda2_init
        self.lambda3 = lambda3_init
        # λ 的夹界随权重同处各自模块（neurons/semantic），由调用方传入。
        self.lambda2_bounds = lambda2_bounds
        self.lambda3_bounds = lambda3_bounds
        # 初值为 0 的模块视作关闭（消融），始终保持 0、不进调整也不被下界夹起来。
        self._lam2_active = lambda2_init > 0.0
        self._lam3_active = lambda3_init > 0.0
        # sem_shift_threshold 由调用方解析：config 给 <=0 时取 1-γ_input，否则用 config 值。
        self.sem_shift_threshold = sem_shift_threshold
        self._cncov_win: deque[float] = deque(maxlen=config.window)
        self._sem_win: deque[float] = deque(maxlen=config.window)
        self.coverage_stalled = False  # 供调度器降重复种子优先级

    def _clamp(self, value: float, bounds: list[float]) -> float:
        return min(max(value, bounds[0]), bounds[1])

    def step(self, state: FeedbackState) -> tuple[float, float]:
        """记入本轮四指标，按规则更新并返回 (λ2, λ3)。"""
        self._cncov_win.append(state.delta_cncov)
        self._sem_win.append(state.mean_sem_shift)
        if not self.config.enabled:
            return self.lambda2, self.lambda3

        mean_dcncov = sum(self._cncov_win) / len(self._cncov_win)
        mean_sem = sum(self._sem_win) / len(self._sem_win)
        self.coverage_stalled = mean_dcncov < self.config.cov_stall_eps

        if self._lam2_active:
            factor = self.config.step_up if self.coverage_stalled else self.config.step_down
            self.lambda2 = self._clamp(self.lambda2 * factor, self.lambda2_bounds)
        if self._lam3_active:
            factor = (
                self.config.step_up
                if mean_sem > self.sem_shift_threshold
                else self.config.step_down
            )
            self.lambda3 = self._clamp(self.lambda3 * factor, self.lambda3_bounds)
        return self.lambda2, self.lambda3
