"""
Pseudo-label generation for turb_i frames (Phase 2 of training).

Strategy (from best submission):
  - Only accept frames where >= 1 box has score >= PSEUDO_SCORE_THRESH
  - Apply quality gate: discard tiny boxes and very elongated (non-blob-like) boxes
  - If fewer than MIN_PL_FRAMES frames pass, skip pseudo-labeling entirely
    rather than poison the model with garbage labels
  - Labeled data is heavily oversampled vs pseudo-labeled data
  - PL learning rate is much lower (noisy labels)
"""

import logging

import h5py
import numpy as np
import torch
from torchvision.ops import nms

logger = logging.getLogger(__name__)

# Quality gate thresholds (matching best submission)
PSEUDO_SCORE_THRESH = 0.55
PSEUDO_NMS_THRESH = 0.30
PL_MIN_BOX_AREA = 100       # pixels² — discard tiny boxes
PL_MAX_ASPECT_RATIO = 8.0   # discard very elongated, non-blob-like boxes


def _box_quality_ok(box_np) -> bool:
    """Return True if a predicted box looks like a real blob."""
    x1, y1, x2, y2 = box_np
    w, h = x2 - x1, y2 - y1
    if w * h < PL_MIN_BOX_AREA:
        return False
    ar = max(w, h) / max(min(w, h), 1e-3)
    return ar <= PL_MAX_ASPECT_RATIO


def generate_pseudo_labels(model, turb_h5_path, device, val_transform):
    """
    Run the current model on all turb_i frames and collect confident predictions.

    Args:
        model:         Trained model (will be set to eval mode).
        turb_h5_path:  Path to turb_i HDF5 file.
        device:        "cuda" or "cpu".
        val_transform: The validation transform (ToTensor + PerImageMinMaxNormalize).

    Returns:
        accepted_frames: list of image tensors that passed the quality gate.
        pseudo_targets:  list of target dicts with 'boxes', 'labels', 'frame_index'.
    """
    logger.info(
        "Generating pseudo-labels (score>=%.2f) on turb_i…", PSEUDO_SCORE_THRESH
    )

    with h5py.File(str(turb_h5_path), "r") as f:
        key = list(f.keys())[0]
        raw = f[key][:]

    accepted_frames, pseudo_targets = [], []
    model.eval()

    with torch.no_grad():
        for i, frame in enumerate(raw):
            frame = frame.astype(np.float32)
            if frame.ndim == 2:
                frame = frame[np.newaxis]
            t = torch.from_numpy(frame)
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)

            img, _ = val_transform(t, {"boxes": None, "labels": None, "frame_index": -1})
            out = model([img.to(device)])[0]
            boxes, scores = out["boxes"], out["scores"]

            mask = scores >= PSEUDO_SCORE_THRESH
            boxes, scores = boxes[mask], scores[mask]
            if len(boxes) == 0:
                continue

            keep = nms(boxes, scores, PSEUDO_NMS_THRESH)
            boxes = boxes[keep]

            good = [j for j, b in enumerate(boxes.cpu().numpy()) if _box_quality_ok(b)]
            if not good:
                continue

            boxes = boxes[good]
            accepted_frames.append(img)
            pseudo_targets.append(
                {
                    "boxes": boxes.cpu(),
                    "labels": torch.ones(len(boxes), dtype=torch.int64),
                    "frame_index": f"turb-pl-{i}",
                }
            )

    logger.info(
        "  %d / %d turb_i frames accepted as pseudo-labels.",
        len(accepted_frames), len(raw),
    )
    return accepted_frames, pseudo_targets
