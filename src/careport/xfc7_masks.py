"""XF-C7 S1 mask instrument (protocol v1.0 section 7; pilot-record specification of 2026-10-22).

PVPS panoptic labels are uint16 PNGs holding semantic_class * divisor + instance_id (camera_segmentation.proto).
A vehicle instance is a (semantic class, instance id > 0) pair in TYPE_CAR 2, TRUCK 3, BUS 4 or OTHER_LARGE_VEHICLE 5.
Its tight box uses pixel edges (column a..b gives u0 = a, u1 = b + 1) and its centre is the tight-box centre.
Masks are linked one-to-one to human 2D boxes in the same image by Hungarian matching on 1 - IoU, keeping IoU >= 0.7.
"""
from __future__ import annotations

import io
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

from careport.xfc7_observations import hull_iou

VEHICLE_CLASSES = (2, 3, 4, 5)


def decode_panoptic(png: bytes, divisor: int) -> Tuple[np.ndarray, np.ndarray]:
    """Semantic class and instance id arrays from a panoptic PNG."""
    pan = np.asarray(Image.open(io.BytesIO(png)), dtype=np.int64)
    return pan // divisor, pan % divisor


def instance_boxes(semantic: np.ndarray, instance: np.ndarray, classes: Sequence[int] = VEHICLE_CLASSES) -> pd.DataFrame:
    """One row per instance of the given classes: semantic, instance, tight box (u0, v0, u1, v1), centre (mu, mv), n_px."""
    sel = np.isin(semantic, classes) & (instance > 0)
    vv, uu = np.nonzero(sel)
    cols = ["semantic", "instance", "u0", "v0", "u1", "v1", "mu", "mv", "n_px"]
    if len(uu) == 0:
        return pd.DataFrame(columns=cols)
    px = pd.DataFrame({"semantic": semantic[vv, uu], "instance": instance[vv, uu], "u": uu, "v": vv})
    g = px.groupby(["semantic", "instance"])
    b = pd.DataFrame({"u0": g["u"].min(), "v0": g["v"].min(), "u1": g["u"].max() + 1, "v1": g["v"].max() + 1,
                      "n_px": g.size()}).reset_index()
    b["mu"] = 0.5 * (b["u0"] + b["u1"])
    b["mv"] = 0.5 * (b["v0"] + b["v1"])
    return b[cols]


def link_masks(mask_boxes: np.ndarray, human_boxes: np.ndarray, min_iou: float = 0.7) -> List[Tuple[int, int]]:
    """(mask index, human box index) pairs, one-to-one by Hungarian matching on 1 - IoU, IoU >= min_iou."""
    if len(mask_boxes) == 0 or len(human_boxes) == 0:
        return []
    iou = hull_iou(mask_boxes, human_boxes)
    rows, cols = linear_sum_assignment(1.0 - iou)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
