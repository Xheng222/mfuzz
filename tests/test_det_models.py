"""检测模型适配层单元测试。

纯 CPU 部分测 yolo11 的结构桶映射与 load_detectors 的名称路由；带权重的
detect 冒烟需要 ultralytics 与 weights/yolo11n.pt，缺一即跳过。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mfuzz.core.det_models import load_detectors, yolo_bucket


def test_yolo_bucket_backbone_levels() -> None:
    assert yolo_bucket("model.0.conv") == "backbone.stem"
    assert yolo_bucket("model.1.conv") == "backbone.C2"
    assert yolo_bucket("model.3.conv") == "backbone.C3"
    assert yolo_bucket("model.6.m.0.cv1.conv") == "backbone.C4"
    assert yolo_bucket("model.9.cv2.conv") == "backbone.C5"
    assert yolo_bucket("model.10.attn.qkv.conv") == "backbone.C5"


def test_yolo_bucket_neck_and_head() -> None:
    assert yolo_bucket("model.13.cv1.conv") == "neck.td"
    assert yolo_bucket("model.16.cv2.conv") == "neck.td"
    assert yolo_bucket("model.17.conv") == "neck.bu"
    assert yolo_bucket("model.22.m.0.cv2.conv") == "neck.bu"
    assert yolo_bucket("model.23.cv2.0.0.conv") == "head.reg"
    assert yolo_bucket("model.23.cv3.1.0.0.conv") == "head.cls"
    assert yolo_bucket("model.23.dfl.conv") == "head.reg"
    # 序号歧义保护：model.1 与 model.10/11 不能混桶
    assert yolo_bucket("model.11.weird") == "neck.td"
    assert yolo_bucket("nope") == "other"


def test_load_detectors_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="未知检测模型"):
        load_detectors(["ssd300"])


@pytest.mark.skipif(
    not Path("weights/yolo11n.pt").exists(), reason="需要 weights/yolo11n.pt 与 ultralytics"
)
def test_yolo_detect_smoke() -> None:
    import torch

    det = load_detectors(["yolo11n"])["yolo11n"]
    # 类别名空间应与 torchvision 的 80 个真实 COCO 类一致（差分按名对齐的前提）
    assert len(det.label_names()) == 80
    img = torch.rand(3, 320, 320)
    dets = det.detect(img, score_thr=0.5)
    for d in dets:
        assert d.box.shape == (4,)
        assert d.label in det.label_names()
        assert d.score >= 0.5
