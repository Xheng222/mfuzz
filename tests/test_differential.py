"""Phase 1 单元测试：四类分流、差分目标、共识过滤。

不加载预训练模型，用常量小模型构造确定的预测，跑得快。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mfuzz.core.types import Seed
from mfuzz.differential.consensus import filter_consensus
from mfuzz.differential.ensemble import Ensemble
from mfuzz.differential.objective import differential_objective
from mfuzz.differential.triage import Verdict, triage


class ConstModel(nn.Module):
    """无论输入，都给指定类别压倒性 logit。argmax 恒为该类别。"""

    def __init__(self, cls: int, num_classes: int = 5) -> None:
        super().__init__()
        self.cls = cls
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], self.num_classes, device=x.device)
        out[:, self.cls] = 10.0
        return out


# ---- triage ----


def test_triage_defect() -> None:
    # 目标偏离 c，所有参考仍预测 c -> A
    assert triage(target_label=2, reference_labels=[1, 1], consensus_label=1) is Verdict.DEFECT


def test_triage_consensus() -> None:
    # 所有模型仍保持 c -> B
    assert triage(target_label=1, reference_labels=[1, 1], consensus_label=1) is Verdict.CONSENSUS


def test_triage_gray_refs_disagree() -> None:
    # 参考彼此分歧 -> C
    assert triage(target_label=2, reference_labels=[1, 3], consensus_label=1) is Verdict.GRAY


def test_triage_gray_refs_drift() -> None:
    # 参考一致但整体漂移离开 c -> C（不可归因于目标）
    assert triage(target_label=2, reference_labels=[2, 2], consensus_label=1) is Verdict.GRAY


def test_triage_semantic_fail() -> None:
    assert (
        triage(target_label=2, reference_labels=[1, 1], consensus_label=1, semantic_ok=False)
        is Verdict.SEMANTIC_FAIL
    )


# ---- differential objective ----


def test_differential_objective_formula() -> None:
    probs = {
        "t": torch.tensor([[0.1, 0.9]]),
        "r1": torch.tensor([[0.8, 0.2]]),
        "r2": torch.tensor([[0.7, 0.3]]),
    }
    c = torch.tensor([0])
    obj = differential_objective(probs, target="t", c=c, lambda1=1.0)
    # (0.8 + 0.7) - 1.0 * 0.1 = 1.4
    assert torch.allclose(obj, torch.tensor([1.4]), atol=1e-6)


def test_differential_objective_lambda() -> None:
    probs = {
        "t": torch.tensor([[0.5, 0.5]]),
        "r1": torch.tensor([[0.6, 0.4]]),
    }
    c = torch.tensor([0])
    obj = differential_objective(probs, target="t", c=c, lambda1=2.0)
    # 0.6 - 2.0 * 0.5 = -0.4
    assert torch.allclose(obj, torch.tensor([-0.4]), atol=1e-6)


# ---- consensus ----


def _seeds(n: int) -> list[Seed]:
    return [Seed(image=torch.randn(3, 8, 8), true_label=0) for _ in range(n)]


def test_consensus_accept_all() -> None:
    # 三个模型预测同一类 -> 全部通过，共识标签为该类
    models = {"a": ConstModel(3), "b": ConstModel(3), "c": ConstModel(3)}
    ensemble = Ensemble(models, target="a")
    result = filter_consensus(ensemble, _seeds(6), batch_size=4)
    assert result.accepted == 6
    assert result.acceptance_rate == 1.0
    for s in result.seeds:
        assert s.consensus_label == 3
        assert set(s.model_confidences) == {"a", "b", "c"}
        assert all(v > 0.99 for v in s.model_confidences.values())


def test_consensus_reject_disagree() -> None:
    # 一个模型预测不同类 -> 无一致，全部拒绝
    models = {"a": ConstModel(1), "b": ConstModel(1), "c": ConstModel(2)}
    ensemble = Ensemble(models, target="a")
    result = filter_consensus(ensemble, _seeds(5), batch_size=2)
    assert result.accepted == 0
    assert result.acceptance_rate == 0.0
