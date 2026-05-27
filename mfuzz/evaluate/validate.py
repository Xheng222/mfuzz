"""实验结果自动校验（实现方案第七章）。

读 result.json 的 dict，按第七章的检查项逐条断言，返回检查结果列表。检查是模式感知的：
覆盖、聚类相关项只在 diff+cov 及以后的模式查，纯差分模式查置信度下降。跨实验才能判定
的项（diff+cov 覆盖曲线高于纯差分、融合 vs 纯频率、动态 vs 静态）这里标成 skip，交给
compare 处理。

前三项直接守旧实现的两个 bug：关键占比不接近 100%、CNCov 不在第一轮饱和、覆盖梯度
非零。
"""

from __future__ import annotations

from dataclasses import dataclass

# 检查项严重级别。error 不通过则整体判失败，warn 仅提示，skip 表示本次不适用。
ERROR = "error"
WARN = "warn"
SKIP = "skip"


@dataclass
class Check:
    name: str
    status: str  # "pass" | "fail" | "skip"
    severity: str  # ERROR | WARN | SKIP
    detail: str


def _is_monotonic_nondec(xs: list[float], tol: float = 1e-6) -> bool:
    return all(xs[i + 1] >= xs[i] - tol for i in range(len(xs) - 1))


def validate_result(result: dict) -> list[Check]:
    m: dict[str, float] = result.get("metrics", {})
    mode: str = result.get("mode", "?")
    cncov_hist: list[float] = result.get("cncov_history", [])
    curves: dict[str, list[float]] = result.get("curves", {})
    checks: list[Check] = []

    def add(name: str, ok: bool, severity: str, detail: str) -> None:
        checks.append(Check(name, "pass" if ok else "fail", severity, detail))

    has_cov = mode != "diff" and "critical_ratio" in m

    # 1. 关键神经元占比合理，不接近 100%。
    if has_cov:
        ratio = m.get("critical_ratio", 0.0)
        add(
            "关键神经元占比 ∈ [0.3, 0.82]",
            0.30 <= ratio <= 0.82,
            ERROR,
            f"critical_ratio={ratio:.3f}",
        )
    else:
        checks.append(Check("关键神经元占比", "skip", SKIP, "纯差分模式无 profiling"))

    # 2. CNCov_0 明显小于 1.0，且 CNCov 单调增长、不在第一轮饱和。
    if has_cov and cncov_hist:
        c0 = cncov_hist[0]
        add("CNCov_0 明显小于 1.0", c0 < 0.9, ERROR, f"cncov_0={c0:.3f}")
        add(
            "CNCov 随轮次单调不降",
            _is_monotonic_nondec(cncov_hist),
            ERROR,
            "检查 cncov_history 序列",
        )
        no_sat = len(cncov_hist) < 2 or cncov_hist[1] < 0.99
        add(
            "CNCov 不在第一轮饱和",
            no_sat,
            ERROR,
            f"round1={cncov_hist[1]:.3f}" if len(cncov_hist) > 1 else "单点",
        )
        gain = m.get("cncov_gain", cncov_hist[-1] - c0)
        add("CNCov 有正增益", gain > 0, ERROR, f"cncov_gain={gain:.3f}")
    elif has_cov:
        checks.append(Check("CNCov 序列", "skip", SKIP, "无 cncov_history"))

    # 3. 存在未覆盖神经元时覆盖梯度非零。
    if has_cov:
        nun = m.get("n_uncovered_final", 0.0)
        gnorm = m.get("mean_cov_grad_norm", 0.0)
        ok = nun <= 0 or gnorm > 0
        add("有未覆盖神经元时覆盖梯度>0", ok, ERROR, f"uncovered={nun:.0f}, cov_grad={gnorm:.4g}")

    # 4. RFT 大于零；参考共识保持率合理。
    rft = m.get("rft", 0.0)
    add("RFT > 0", rft > 0, ERROR, f"rft={rft:.3f}")
    hold = m.get("ref_consensus_hold_rate")
    if hold is not None:
        add("参考共识保持率 ∈ (0,1]", 0 < hold <= 1, ERROR, f"hold={hold:.3f}")
        if hold < 0.8:
            checks.append(Check("参考共识保持率偏低", "fail", WARN, f"hold={hold:.3f} < 0.8"))

    # 5. 目标置信度随迭代下降（纯差分看 target_conf 曲线）。
    conf = curves.get("target_conf", [])
    if conf and len(conf) >= 2:
        add("目标置信度随步数下降", conf[-1] < conf[0], ERROR, f"{conf[0]:.3f}->{conf[-1]:.3f}")

    # 6. 语义：输入有效率不过低、平均 S_input 在阈值之上（Phase 3 前是占位）。
    if "mean_s_input" in m:
        ivr = m.get("input_valid_rate", 0.0)
        checks.append(
            Check(
                "输入有效率不过低",
                "pass" if ivr >= 0.5 else "fail",
                WARN,
                f"ivr={ivr:.3f}（Phase3 前占位）",
            )
        )

    # 7. 缺陷聚类簇数大于 1。纯差分模式不记激活向量、不聚类，跳过。
    if mode == "diff":
        checks.append(Check("缺陷聚类簇数 > 1", "skip", SKIP, "纯差分模式不记激活向量"))
    elif "n_clusters" in m:
        nc = m.get("n_clusters", 0.0)
        add("缺陷聚类簇数 > 1", nc > 1, ERROR, f"n_clusters={nc:.0f}")

    # 8. 激活越界：看有没有大量关键神经元被推出 profiling 范围（ĉ 越出 [0,1]）。
    #    越界本身是正常现象，只在占比偏大时告警，无论如何都把数据打印出来。
    if "n_ood_neurons" in m:
        n_ood = m.get("n_ood_neurons", 0.0)
        n_crit = m.get("n_critical", 0.0)
        lo = m.get("min_activation", 0.0)
        hi = m.get("max_activation", 0.0)
        frac = n_ood / n_crit if n_crit > 0 else 0.0
        # 用 c_hat 而非 ĉ 字面量：Windows 非 UTF-8 控制台/重定向下打印该字符会崩
        # UnicodeEncodeError，与图表里把 ĉ 写成 c_hat 同理。
        detail = f"越界神经元 {n_ood:.0f}/{n_crit:.0f}（{frac:.1%}），c_hat∈[{lo:.3g}, {hi:.3g}]"
        if frac > 0.1:  # 启发式阈值：超一成关键神经元越界值得留意
            checks.append(Check("激活越界占比偏大", "fail", WARN, detail))
        else:
            checks.append(Check("激活越界占比正常", "pass", WARN, detail))

    # 跨实验项：留给 compare。
    checks.append(Check("diff+cov 覆盖高于纯差分", "skip", SKIP, "需跨实验，见 compare"))
    checks.append(Check("融合 vs 纯频率差异", "skip", SKIP, "需跨实验，见 compare"))
    checks.append(Check("动态 vs 静态权重差异", "skip", SKIP, "需跨实验，见 compare"))
    return checks


def summarize(checks: list[Check]) -> tuple[int, int, int, int]:
    """返回 (通过, 失败-error, 失败-warn, 跳过) 计数。"""
    passed = sum(1 for c in checks if c.status == "pass")
    fail_err = sum(1 for c in checks if c.status == "fail" and c.severity == ERROR)
    fail_warn = sum(1 for c in checks if c.status == "fail" and c.severity == WARN)
    skipped = sum(1 for c in checks if c.status == "skip")
    return passed, fail_err, fail_warn, skipped


def has_failures(checks: list[Check]) -> bool:
    """是否存在 error 级别的不通过项。"""
    return any(c.status == "fail" and c.severity == ERROR for c in checks)
