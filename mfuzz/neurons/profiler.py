"""关键神经元离线识别。

两个维度融合成关键度。频率维度：给定激活阈值 t，神经元输出超过 t 即算激活，
频率项是它在训练子集里被激活的比例。归因维度：

    S(n, c) = E_{x in D_c}[ | dF(x)_c / d out(n, x) · out(n, x) | ]

综合关键度：

    cl(n, T) = α · Freq(out(n, T) > t) + (1 - α) · Norm(S(n, c))

α=1 退化为纯频率，α=0 退化为纯归因，Norm 把敏感度按百分位归一化到与频率可比。
用分位而非绝对阈值，是因为各层激活量纲差异大、实测 cl 分布极偏，绝对阈值不可比
也难校准（详见 _critical_mask）。

全局关键集合 D_en 和类关键集合 D_en^c 各用一个分位阈值，且取值不同。全局 τ_global
宽（保留约 1-τ_global 的神经元），让 CNCov 有足够覆盖广度；类关键 τ_class 严，让
每个 D_en^c 小而类专属。两者要求是相反的：D_en 大才覆盖得广，D_en^c 小才在类别间
拉得开（集合太大则各类高度重合，CCCov 投到任何一类都≈全局 CNCov，区分不出类别）。
D_en^c 直接按各类自己的 cl_c 取前 1-τ_class，不强制嵌套进 D_en——按 cl_c 排在前面
但全局不够格的"类专属"神经元，恰是区分类别的主力，保留它们比追求严格子集更重要。

工程约束（见实现方案第一、四章）：各层激活量纲差异很大，固定阈值 t 直接作用
在原始激活上会让某些层几乎全激活、某些层几乎不激活。这里先按每个神经元在完整
profiling 数据上的最大值把激活归一化，t 于是表示"超过自身峰值的若干比例"，对
所有层含义一致。覆盖统计与覆盖目标都复用同一套 scale。

全局频率在完整 profiling 数据上流式统计，不抽样（覆盖度要求面向完整集合）。
全局 D_en 用全局频率加全局归因；类关键集合 D_en^c 用类别 c 自己的数据算频率
与归因（研究方案 T -> T_c），各类集合因此真正拉开。profiling 只跑一次并缓存，
缓存键含模型、数据集、val_fraction、t、τ_global、τ_class、α 和共识类别集合。

融合权重 α 一个旋钮就能取到三种关键度配置：α=1 纯频率，α=0 纯归因，中间为融合
（研究内容 2 的消融）。α=1 时归因那一遍带梯度的前向直接跳过，省开销。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from mfuzz.core.hooks import ActivationExtractor

_EPS = 1e-8


def flatten_acts(acts: dict[str, Tensor], layers: list[str]) -> Tensor:
    """按固定层序把各层 (B, C) 激活拼成 (B, N)，保留计算图。"""
    return torch.cat([acts[name] for name in layers], dim=1)


def _pct_rank(v: Tensor) -> Tensor:
    """百分位归一化到 [0, 1]。归因 S 的分布长尾严重，min-max 会把绝大多数神经元
    压到接近 0，使归因几乎不影响排序；百分位排名让归因均匀铺开，和频率（也是
    [0, 1] 的比例量）口径可比，融合权重 α 才真正起作用。"""
    n = v.numel()
    if n <= 1:
        return torch.zeros_like(v)
    order = v.argsort()
    ranks = torch.empty_like(v)
    ranks[order] = torch.arange(n, dtype=v.dtype, device=v.device)
    return ranks / (n - 1)


def _critical_mask(cl: Tensor, tau: float) -> Tensor:
    """按分位选关键神经元：保留 cl 高于 τ 分位的神经元，占比约 1-τ。

    各层激活量纲差异大，绝对阈值 cl > τ 跨层跨模型都不可比，且实测 cl 分布极度
    偏向 0，绝对阈值要么选空要么对参数极敏感。改用分位阈值，关键占比由 τ 直接
    控制（τ 越大越严、保留越少），方向与绝对阈值一致，且天然落在合理区间。"""
    thr = torch.quantile(cl, tau)
    return cl > thr


@dataclass
class NeuronProfile:
    """一次 profiling 的结果。所有向量沿全局神经元轴 (N,)，轴序由 layers 定。"""

    model_name: str
    layers: list[str]
    counts: dict[str, int]  # 层名 -> 神经元数
    t: float  # 归一化激活阈值
    tau: float  # τ_global，全局关键度分位阈值，保留 cl 高于该分位的神经元（占比约 1-τ_global）
    tau_class: float  # τ_class，类关键分位阈值，比全局严，D_en^c 更小更类专属
    alpha: float  # cl 融合权重；α=1 纯频率，α=0 纯归因，中间为融合
    scale: Tensor  # (N,) 每神经元归一化尺度
    freq: Tensor  # (N,) 激活频率
    cl: Tensor  # (N,) 全局关键度
    cl_per_class: dict[int, Tensor]  # c -> (N,) 类关键度 cl_c，便于不重算地改 τ_class
    critical: Tensor  # (N,) bool，全局关键集合 D_en
    critical_per_class: dict[int, Tensor]  # c -> (N,) bool，类关键集合 D_en^c

    @property
    def num_neurons(self) -> int:
        return int(self.scale.numel())

    @property
    def num_critical(self) -> int:
        return int(self.critical.sum())

    @property
    def critical_ratio(self) -> float:
        n = self.num_neurons
        return self.num_critical / n if n else 0.0

    def class_critical_ratios(self) -> dict[int, float]:
        """各类 D_en^c 占全部神经元的比例，约 1-τ_class，供观察类集合规模。"""
        n = self.num_neurons
        return {c: int(m.sum()) / n if n else 0.0 for c, m in self.critical_per_class.items()}

    def class_critical_mask_at(self, c: int, tau: float) -> Tensor:
        """用任意分位阈值重算类别 c 的关键集合，不必重跑 profiling，供校准 τ_class。"""
        return _critical_mask(self.cl_per_class[c], tau)

    def to(self, device: torch.device | str) -> NeuronProfile:
        self.scale = self.scale.to(device)
        self.freq = self.freq.to(device)
        self.cl = self.cl.to(device)
        self.cl_per_class = {c: v.to(device) for c, v in self.cl_per_class.items()}
        self.critical = self.critical.to(device)
        self.critical_per_class = {c: m.to(device) for c, m in self.critical_per_class.items()}
        return self

    def cl_percentiles(self, qs: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95)) -> dict[float, float]:
        """cl 分布分位数，供校准 τ。"""
        ps = torch.tensor(qs, device=self.cl.device)
        vals = torch.quantile(self.cl, ps)
        return {q: float(v) for q, v in zip(qs, vals, strict=True)}

    def ratio_at(self, tau: float) -> float:
        """若把分位阈值设为 tau，关键神经元占比会是多少（约 1-τ），供校准。"""
        return float(_critical_mask(self.cl, tau).float().mean())


def _streaming_scale(
    extractor: ActivationExtractor,
    layers: list[str],
    loader: Iterable[tuple[Tensor, Any]],
    device: torch.device,
) -> Tensor:
    """流式过一遍全量 profiling 数据，求每神经元峰值作归一化尺度。不存全部激活。"""
    scale: Tensor | None = None
    seen = 0
    with torch.no_grad():
        for x, _ in loader:
            batch_max = flatten_acts(extractor.extract(x.to(device)), layers).amax(dim=0)  # (N,)
            scale = batch_max if scale is None else torch.maximum(scale, batch_max)
            seen += x.shape[0]
    assert scale is not None, "profiling 数据为空"
    logger.info(f"峰值统计：{seen} 样本，{scale.numel()} 神经元")
    return scale.clamp_min(_EPS)


def _streaming_freq(
    extractor: ActivationExtractor,
    layers: list[str],
    loader: Iterable[tuple[Tensor, Any]],
    t: float,
    scale: Tensor,
    device: torch.device,
) -> Tensor:
    """流式数每神经元归一化激活超过 t 的样本比例。"""
    count: Tensor | None = None
    total = 0
    with torch.no_grad():
        for x, _ in loader:
            fired = (flatten_acts(extractor.extract(x.to(device)), layers) > t * scale).sum(dim=0)
            count = fired if count is None else count + fired
            total += x.shape[0]
    assert count is not None and total > 0, "profiling 数据为空"
    return count.float() / total


def _streaming_class_stats(
    extractor: ActivationExtractor,
    layers: list[str],
    loader: Iterable[tuple[Tensor, Any]],
    c: int,
    t: float,
    scale: Tensor,
    device: torch.device,
    need_attr: bool,
) -> tuple[Tensor | None, Tensor]:
    """对类别 c 的全部数据流式算 (归因 S(n,c), 类频率 freq_c)。

    类频率 freq_c 用全局峰值 scale 归一化，口径和全局频率一致。归因要梯度流到
    激活，输入需 requires_grad；只在 need_attr 时算，并和频率统计共用同一次前向。
    """
    s_acc: Tensor | None = None
    fired: Tensor | None = None
    total = 0
    for x, _ in loader:
        xb = x.to(device)
        if need_attr:
            xb = xb.clone().requires_grad_(True)
            out, acts = extractor.forward_with_acts(xb)
            flat = flatten_acts(acts, layers)
            prob_c = torch.softmax(out, dim=1)[:, c]
            act_list = [acts[name] for name in layers]
            grads = torch.autograd.grad(prob_c.sum(), act_list, allow_unused=True)
            contribs = [
                (a.detach() * g).abs() if g is not None else torch.zeros_like(a)
                for a, g in zip(act_list, grads, strict=True)
            ]
            s_flat = torch.cat(contribs, dim=1).sum(dim=0)
            s_acc = s_flat if s_acc is None else s_acc + s_flat
            batch_fired = (flat.detach() > t * scale).sum(dim=0)
        else:
            with torch.no_grad():
                batch_fired = (flatten_acts(extractor.extract(xb), layers) > t * scale).sum(dim=0)
        fired = batch_fired if fired is None else fired + batch_fired
        total += x.shape[0]
    assert fired is not None and total > 0, f"类别 {c} 数据为空"
    freq_c = fired.float() / total
    s_c = (s_acc / total) if (need_attr and s_acc is not None) else None
    return s_c, freq_c


def _cache_key(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _save(path: Path, profile: NeuronProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": profile.model_name,
        "layers": profile.layers,
        "counts": profile.counts,
        "t": profile.t,
        "tau": profile.tau,
        "tau_class": profile.tau_class,
        "alpha": profile.alpha,
        "scale": profile.scale.cpu(),
        "freq": profile.freq.cpu(),
        "cl": profile.cl.cpu(),
        "cl_per_class": {c: v.cpu() for c, v in profile.cl_per_class.items()},
        "critical": profile.critical.cpu(),
        "critical_per_class": {c: m.cpu() for c, m in profile.critical_per_class.items()},
    }
    torch.save(payload, path)


def _load(path: Path) -> NeuronProfile:
    p = torch.load(path, map_location="cpu", weights_only=False)
    return NeuronProfile(
        model_name=p["model_name"],
        layers=p["layers"],
        counts=p["counts"],
        t=p["t"],
        tau=p["tau"],
        tau_class=p["tau_class"],
        alpha=p["alpha"],
        scale=p["scale"],
        freq=p["freq"],
        cl=p["cl"],
        cl_per_class=p["cl_per_class"],
        critical=p["critical"],
        critical_per_class=p["critical_per_class"],
    )


def build_profile(
    model: nn.Module,
    model_name: str,
    profile_loader: Iterable[tuple[Tensor, Any]],
    class_loaders: Mapping[int, Iterable[tuple[Tensor, Any]]],
    *,
    t: float,
    tau: float,
    tau_class: float,
    alpha: float,
    dataset_name: str,
    val_fraction: float,
    cache_dir: str | Path,
    device: torch.device,
) -> NeuronProfile:
    """识别关键神经元并缓存。已有匹配缓存则直接读取。

    全局关键集合 D_en 用完整 profiling 数据的频率加归因，按 τ_global 取前 1-τ_global；
    类关键集合 D_en^c 用类别 c 自己的数据算频率与归因（研究方案的 T -> T_c），按更严的
    τ_class 取前 1-τ_class，集合更小更类专属，CCCov 才能在类别间拉开。两个阈值分开是
    因为 CNCov 要覆盖广度（D_en 大）、CCCov 要类别区分度（D_en^c 小而专），要求相反。
    只对传入 class_loaders 的共识类别算，不枚举全部类别。
    """
    classes = sorted(class_loaders)
    key = _cache_key(
        {
            "model": model_name,
            "dataset": dataset_name,
            "val_fraction": val_fraction,
            "t": t,
            "tau": tau,
            "tau_class": tau_class,
            "alpha": alpha,
            "classes": classes,
        }
    )
    cache_path = Path(cache_dir) / f"{model_name}_{key}.pt"
    if cache_path.exists():
        logger.info(f"命中 profiling 缓存 {cache_path}")
        return _load(cache_path).to(device)

    extractor = ActivationExtractor(model)
    layers = extractor.layer_names
    counts = extractor.neuron_counts(next(iter(profile_loader))[0][:1].to(device))

    # 全局：完整 profiling 数据上的峰值与频率（流式，不存全部激活）。
    scale = _streaming_scale(extractor, layers, profile_loader, device)
    freq_global = _streaming_freq(extractor, layers, profile_loader, t, scale, device)

    need_attr = alpha < 1.0
    s_by_class: dict[int, Tensor] = {}
    freq_by_class: dict[int, Tensor] = {}
    for c in classes:
        s_c, freq_c = _streaming_class_stats(
            extractor, layers, class_loaders[c], c, t, scale, device, need_attr
        )
        freq_by_class[c] = freq_c
        if s_c is not None:
            s_by_class[c] = s_c

    if need_attr:
        s_bar = torch.stack([s_by_class[c] for c in classes], dim=0).mean(dim=0)  # (N,)
        cl = alpha * freq_global + (1.0 - alpha) * _pct_rank(s_bar)
    else:
        cl = freq_global.clone()
    critical = _critical_mask(cl, tau)

    cl_per_class: dict[int, Tensor] = {}
    critical_per_class: dict[int, Tensor] = {}
    for c in classes:
        if need_attr:
            cl_c = alpha * freq_by_class[c] + (1.0 - alpha) * _pct_rank(s_by_class[c])
        else:
            cl_c = freq_by_class[c]  # 纯频率（α=1），但每类频率不同，D_en^c 仍各异
        cl_per_class[c] = cl_c
        critical_per_class[c] = _critical_mask(cl_c, tau_class)  # 用更严的 τ_class

    profile = NeuronProfile(
        model_name=model_name,
        layers=layers,
        counts=counts,
        t=t,
        tau=tau,
        tau_class=tau_class,
        alpha=alpha,
        scale=scale,
        freq=freq_global,
        cl=cl,
        cl_per_class=cl_per_class,
        critical=critical,
        critical_per_class=critical_per_class,
    )
    _save(cache_path, profile)
    cls_ratios = profile.class_critical_ratios()
    cls_mean = sum(cls_ratios.values()) / len(cls_ratios) if cls_ratios else 0.0
    logger.info(
        f"profiling 完成：{profile.num_neurons} 神经元，全局关键 {profile.num_critical} "
        f"（占比 {profile.critical_ratio:.3f}，τ_global={tau}）；"
        f"类关键 {len(cls_ratios)} 类，平均占比 {cls_mean:.3f}（τ_class={tau_class}）；"
        f"缓存写入 {cache_path}"
    )
    return profile
