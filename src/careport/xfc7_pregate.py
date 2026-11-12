"""XF-C7 pre-freeze measurements that read no human image label and no residual.

G3 timing conventions: per-camera delays of the trigger, the image-centre (pose) time and the readout,
relative to the frame start (first top-LiDAR scan), from `camera_image` metadata columns only.

Population: how many vehicles meet the protocol's §4 label criteria (class, LiDAR points, a camera-synced box)
per most-visible camera and speed band, and the predicted along-track shift v * delay in metres. These use only
`lidar_box` and `lidar_camera_synced_box`; they size the study without touching `camera_box`.
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd

CAMERA_NAMES: Dict[int, str] = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
EXPECTED_SWEEP_ORDER = ("SIDE_LEFT", "FRONT_LEFT", "FRONT", "FRONT_RIGHT", "SIDE_RIGHT")

# Protocol §4 thresholds (draft v0.1).
TYPE_VEHICLE = 1
MIN_LIDAR_POINTS = 20
MOVING_MIN_SPEED = 3.0
FAST_MIN_SPEED = 8.0
STATIC_MAX_SPEED = 0.5


def _q(x: pd.Series, q: float) -> float:
    return float(np.quantile(x, q)) if len(x) else float("nan")


def camera_timing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-camera timing summary in milliseconds relative to the frame start.

    Expects columns camera_name, frame_timestamp_micros, camera_trigger_time, camera_readout_done_time,
    pose_timestamp, shutter (times in seconds except the frame timestamp in microseconds).
    """
    frame_s = df["frame_timestamp_micros"].to_numpy(dtype=np.int64) / 1e6
    d = pd.DataFrame({
        "camera_name": df["camera_name"].to_numpy(),
        "trigger": (df["camera_trigger_time"].to_numpy() - frame_s) * 1000.0,
        "pose": (df["pose_timestamp"].to_numpy() - frame_s) * 1000.0,
        "readout": (df["camera_readout_done_time"].to_numpy() - df["camera_trigger_time"].to_numpy()) * 1000.0,
        "shutter": df["shutter"].to_numpy() * 1000.0,
    })
    rows = []
    for code, name in CAMERA_NAMES.items():
        g = d[d["camera_name"] == code]
        rows.append({
            "camera": name,
            "n_images": int(len(g)),
            "trigger_delay_ms_median": _q(g["trigger"], 0.5),
            "trigger_delay_ms_p05": _q(g["trigger"], 0.05),
            "trigger_delay_ms_p95": _q(g["trigger"], 0.95),
            "pose_delay_ms_median": _q(g["pose"], 0.5),
            "pose_delay_ms_p05": _q(g["pose"], 0.05),
            "pose_delay_ms_p95": _q(g["pose"], 0.95),
            "readout_ms_median": _q(g["readout"], 0.5),
            "shutter_ms_median": _q(g["shutter"], 0.5),
        })
    return pd.DataFrame(rows)


def expected_order_holds(median_delay_ms_by_camera: Mapping[str, float]) -> bool:
    """True if the delays rise strictly in the order the top LiDAR is expected to sweep the cameras."""
    vals = [median_delay_ms_by_camera[c] for c in EXPECTED_SWEEP_ORDER]
    return all(a < b for a, b in zip(vals, vals[1:]))


def planar_speed(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    return np.hypot(vx, vy)


def population_table(boxes: pd.DataFrame, synced: pd.DataFrame,
                     delay_ms_by_camera: Mapping[str, float]) -> pd.DataFrame:
    """Counts of §4-eligible vehicles per most-visible camera and speed band, with predicted shifts.

    boxes: segment, frame, object_id, type, points, vx, vy (from lidar_box).
    synced: segment, frame, object_id, most_visible_camera_name (from lidar_camera_synced_box).
    delay_ms_by_camera: camera name -> delay of that camera's capture after the frame start, in ms.
    """
    key = ["segment", "frame", "object_id"]
    m = boxes.merge(synced[key + ["most_visible_camera_name"]], on=key, how="inner")
    m = m[(m["type"] == TYPE_VEHICLE) & (m["points"] >= MIN_LIDAR_POINTS)]
    speed = planar_speed(m["vx"].to_numpy(dtype=float), m["vy"].to_numpy(dtype=float))
    m = m.assign(speed=speed)
    rows = []
    for code, name in CAMERA_NAMES.items():
        g = m[m["most_visible_camera_name"] == code]
        moving = g[g["speed"] >= MOVING_MIN_SPEED]
        delay = delay_ms_by_camera.get(name, float("nan"))
        shift = moving["speed"] * (delay / 1000.0)
        rows.append({
            "camera": name,
            "eligible_vehicles": int(len(g)),
            "moving_ge_3": int(len(moving)),
            "moving_ge_8": int((g["speed"] >= FAST_MIN_SPEED).sum()),
            "static_le_0_5": int((g["speed"] <= STATIC_MAX_SPEED).sum()),
            "moving_tracks": int(moving[["segment", "object_id"]].drop_duplicates().shape[0]),
            "moving_segments": int(moving["segment"].nunique()),
            "moving_speed_ms_median": _q(moving["speed"], 0.5),
            "moving_shift_m_median": _q(shift, 0.5) if len(shift) and np.isfinite(delay) else float("nan"),
            "moving_shift_m_p90": _q(shift, 0.9) if len(shift) and np.isfinite(delay) else float("nan"),
        })
    return pd.DataFrame(rows)
