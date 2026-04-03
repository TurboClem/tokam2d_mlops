"""
Inference utilities: Test-Time Augmentation (TTA) + Weighted Box Fusion (WBF).

At inference time, the best submission runs hflip + vflip augmentations
and fuses the predictions with WBF instead of plain NMS.
This typically improves AP50 by ~2-4 points over greedy NMS.
"""

import logging

import numpy as np
import torch
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


def weighted_box_fusion(
    boxes_list: list,
    scores_list: list,
    iou_thresh: float = 0.4,
    img_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted Box Fusion (WBF) — averages overlapping boxes weighted by score.

    Boxes are normalised to [0, 1] internally, then scaled back.

    Args:
        boxes_list:  List of (N_i, 4) arrays, one per augmentation.
        scores_list: List of (N_i,) arrays, one per augmentation.
        iou_thresh:  IoU threshold for box clustering.
        img_size:    Image size used for normalisation.

    Returns:
        (boxes, scores) as numpy arrays after fusion.
    """
    all_b = [b / img_size for b in boxes_list]
    clusters: list[dict] = []

    for b_arr, s_arr in zip(all_b, scores_list):
        for box, score in zip(b_arr, s_arr):
            matched, best_iou = -1, iou_thresh
            for ci, cl in enumerate(clusters):
                rep = cl["wb"]
                ix1, iy1 = max(box[0], rep[0]), max(box[1], rep[1])
                ix2, iy2 = min(box[2], rep[2]), min(box[3], rep[3])
                if ix2 <= ix1 or iy2 <= iy1:
                    iou = 0.0
                else:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = (
                        (box[2] - box[0]) * (box[3] - box[1])
                        + (rep[2] - rep[0]) * (rep[3] - rep[1])
                        - inter
                    )
                    iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    matched = ci

            if matched == -1:
                clusters.append({"boxes": [box], "scores": [score], "wb": box.copy(), "s": score})
            else:
                cl = clusters[matched]
                cl["boxes"].append(box)
                cl["scores"].append(score)
                sa = np.array(cl["scores"])
                ba = np.array(cl["boxes"])
                cl["wb"] = (ba * sa[:, None]).sum(0) / sa.sum()
                cl["s"] = sa.mean()

    if not clusters:
        return np.zeros((0, 4)), np.zeros(0)

    return (
        np.array([c["wb"] for c in clusters]) * img_size,
        np.array([c["s"] for c in clusters]),
    )


def predict_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: str,
    score_thresh: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference with hflip + vflip TTA and fuse via WBF.

    Args:
        model:        Trained model in eval mode.
        image:        Single image tensor (C, H, W).
        device:       "cuda" or "cpu".
        score_thresh: Pre-filter threshold before WBF.

    Returns:
        (boxes, scores) numpy arrays.
    """
    _, H, W = image.shape
    augs = [image, TF.hflip(image), TF.vflip(image)]
    model.eval()
    boxes_list, scores_list = [], []

    with torch.no_grad():
        for k, aug in enumerate(augs):
            out = model([aug.to(device)])[0]
            b = out["boxes"].cpu().numpy()
            s = out["scores"].cpu().numpy()
            # Un-flip bounding boxes
            if k == 1:
                b[:, [0, 2]] = W - b[:, [2, 0]]
            if k == 2:
                b[:, [1, 3]] = H - b[:, [3, 1]]
            mask = s >= score_thresh
            boxes_list.append(b[mask])
            scores_list.append(s[mask])

    return weighted_box_fusion(boxes_list, scores_list, iou_thresh=0.4, img_size=max(H, W))
