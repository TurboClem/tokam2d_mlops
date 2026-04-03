"""Faster R-CNN builder with competition-tuned thresholds."""

import logging
from pathlib import Path

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

logger = logging.getLogger(__name__)


def build_model(
    score_thresh: float = 0.05,
    nms_thresh: float = 0.30,
    detections_per_img: int = 150,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Build a Faster R-CNN ResNet-50 FPN with competition-tuned settings.

    Note: detections_per_img=150 (not the default 100) — blobs can be numerous
    in turb_dwi frames.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.nms_thresh = nms_thresh
    model.roi_heads.detections_per_img = detections_per_img
    logger.info(
        "Built Faster R-CNN | score_thresh=%.2f | nms_thresh=%.2f | det/img=%d",
        score_thresh, nms_thresh, detections_per_img,
    )
    return model


def save_model(model: torch.nn.Module, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Saved model → %s", path)


def load_model(path: Path, device: str = "cpu", **kwargs) -> torch.nn.Module:
    model = build_model(pretrained=False, **kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    logger.info("Loaded model ← %s  (device=%s)", path, device)
    return model
