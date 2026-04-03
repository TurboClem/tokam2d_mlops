"""
Evaluation metrics for Tokamak blob detection.

compute_ap50: AP@IoU=0.5 matching the competition scorer logic.

Note: the competition used IoMean (intersection / mean area) instead of IoU,
but this module implements standard IoU-based AP50 for fast local validation.
The competition-exact metric is in scoring_program/scoring.py.
"""

import logging

import numpy as np
import torch
from torchvision.ops import nms

logger = logging.getLogger(__name__)


def compute_ap50(model: torch.nn.Module, dataloader, device: str) -> float:
    """
    Compute AP@IoU=0.5 on a validation dataloader.

    Returns:
        float in [0, 1].
    """
    model.eval()
    all_preds: dict = {}
    all_targets: dict = {}

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                fid = target["frame_index"]
                keep = (
                    nms(output["boxes"], output["scores"], 0.5)
                    if len(output["boxes"]) > 0
                    else torch.tensor([], dtype=torch.long)
                )
                all_preds[fid] = {
                    "boxes": output["boxes"][keep].cpu().numpy(),
                    "scores": output["scores"][keep].cpu().numpy(),
                }
                if target["boxes"] is not None:
                    all_targets[fid] = {"boxes": target["boxes"].cpu().numpy()}

    if not all_targets:
        return 0.0

    num_gt = 0
    tp_records = np.empty((0, 2))  # (score, is_tp)

    for fid, gt in all_targets.items():
        gt_boxes = gt["boxes"]
        num_gt += len(gt_boxes)

        if fid not in all_preds or len(all_preds[fid]["boxes"]) == 0:
            continue

        pred_boxes = all_preds[fid]["boxes"]
        pred_scores = all_preds[fid]["scores"]
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        tps = np.zeros(len(pred_boxes))

        for i, pred in enumerate(pred_boxes):
            ious = []
            for gt_box in gt_boxes:
                x1 = max(pred[0], gt_box[0])
                y1 = max(pred[1], gt_box[1])
                x2 = min(pred[2], gt_box[2])
                y2 = min(pred[3], gt_box[3])
                if x2 <= x1 or y2 <= y1:
                    ious.append(0.0)
                    continue
                inter = (x2 - x1) * (y2 - y1)
                union = (
                    (pred[2] - pred[0]) * (pred[3] - pred[1])
                    + (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    - inter
                )
                ious.append(inter / union if union > 0 else 0.0)

            best = int(np.argmax(ious))
            if ious[best] >= 0.5 and not gt_matched[best]:
                tps[i] = 1
                gt_matched[best] = True

        tp_records = np.concatenate(
            [tp_records, np.column_stack([pred_scores, tps])]
        )

    if num_gt == 0 or len(tp_records) == 0:
        return 0.0

    tp_records = tp_records[np.argsort(-tp_records[:, 0])]
    tp_cum = np.concatenate([[0], np.cumsum(tp_records[:, 1])])
    fp_cum = np.concatenate([[0], np.cumsum(1 - tp_records[:, 1])])
    recall = tp_cum / num_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-10)

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    ap = float(np.trapezoid(precision, recall))
    logger.debug("AP50=%.4f (GT boxes: %d)", ap, num_gt)
    return ap
