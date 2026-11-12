"""XF-C7 timing model: when the top LiDAR swept an azimuth, and when a camera column was exposed.

- lidar_sweep_time_ms: Sun et al. (CVPR 2020) synchronisation convention: the frame starts at the backward direction
  and the sweep reaches a camera's +x axis at camera_center_offset/360 x 100 ms (SIDE_LEFT 90 deg -> 25 ms,
  FRONT 180 deg -> 50 ms, SIDE_RIGHT 270 deg -> 75 ms). The offsets are defined against the vehicle's backward
  direction, so theta is the azimuth in VEHICLE axes about the top-LiDAR position (counter-clockwise from +x);
  the top LiDAR's own axes are yawed by about 148 deg and must not be used. t = T * ((pi - theta) mod 2 pi) / (2 pi).
- column_time: Waymo's GetPixelTimestamp (camera_model.cc) for horizontal readouts: trigger + shutter/2 +
  readout * x / W (LEFT_TO_RIGHT) or readout * (W - x) / W (RIGHT_TO_LEFT), readout = done - trigger - shutter.
All times in the units passed in (ms in this project), measured from the frame start.
"""
from __future__ import annotations

import numpy as np

LEFT_TO_RIGHT = 2
RIGHT_TO_LEFT = 4

# Amendment A1 (pilot record, 2026-10-20): G3b measured the sweep at the vehicle's backward direction 0.50 ms after
# the frame start and a period of 98.96 ms for the 2,650 columns; t_lidar uses these instead of (0, frame interval).
A1_PHASE_MS = 0.50
A1_PERIOD_MS = 98.96


def azimuth_about(points_vehicle: np.ndarray, sensor_xy: np.ndarray) -> np.ndarray:
    """Azimuth (rad, counter-clockwise from vehicle +x) of points (N, 3) seen from a sensor at sensor_xy (vehicle frame)."""
    p = np.asarray(points_vehicle, dtype=float)
    s = np.asarray(sensor_xy, dtype=float)
    return np.arctan2(p[:, 1] - s[1], p[:, 0] - s[0])


def lidar_sweep_time_ms(theta: np.ndarray, period_ms: float = 100.0, phase_ms: float = 0.0) -> np.ndarray:
    th = np.asarray(theta, dtype=float)
    return phase_ms + period_ms * np.mod(np.pi - th, 2 * np.pi) / (2 * np.pi)


def column_time(x: np.ndarray, width: float, trigger: float, shutter: float, readout_done: float,
                direction: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    readout = readout_done - trigger - shutter
    base = trigger + 0.5 * shutter
    if direction == LEFT_TO_RIGHT:
        return base + readout * x / width
    if direction == RIGHT_TO_LEFT:
        return base + readout * (width - x) / width
    raise ValueError(f"unsupported readout direction {direction}")
