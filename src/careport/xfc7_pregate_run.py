"""Run the XF-C7 pre-freeze measurements (G3 timing conventions and population sizing).

Usage: PYTHONPATH=src python -m careport.xfc7_pregate_run <output_dir>
Reads WAYMO_V2_ROOT from the environment. Touches only metadata columns of `camera_image` (no image bytes),
`lidar_box` and `lidar_camera_synced_box`. Never reads `camera_box`, `projected_lidar_box` or masks.
Writes aggregate tables only (JSON and Markdown).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq

from careport.xfc7_pregate import (
    CAMERA_NAMES,
    EXPECTED_SWEEP_ORDER,
    camera_timing_table,
    expected_order_holds,
    population_table,
)

CI = "[CameraImageComponent]"
LB = "[LiDARBoxComponent]"
SB = "[LiDARCameraSyncedBoxComponent]"


def _read(root: str, component: str, columns: list[str]) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(root, "validation", component, "*.parquet")))
    if len(files) != 202:
        raise SystemExit(f"{component}: expected 202 files, found {len(files)}")
    return pd.concat([pq.read_table(f, columns=columns).to_pandas() for f in files], ignore_index=True)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 64
    out_dir = argv[1]
    root = os.environ["WAYMO_V2_ROOT"]

    cam = _read(root, "camera_image", [
        "key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name",
        f"{CI}.pose_timestamp", f"{CI}.rolling_shutter_params.shutter",
        f"{CI}.rolling_shutter_params.camera_trigger_time", f"{CI}.rolling_shutter_params.camera_readout_done_time",
    ])
    timing_in = pd.DataFrame({
        "camera_name": cam["key.camera_name"],
        "frame_timestamp_micros": cam["key.frame_timestamp_micros"],
        "camera_trigger_time": cam[f"{CI}.rolling_shutter_params.camera_trigger_time"],
        "camera_readout_done_time": cam[f"{CI}.rolling_shutter_params.camera_readout_done_time"],
        "pose_timestamp": cam[f"{CI}.pose_timestamp"],
        "shutter": cam[f"{CI}.rolling_shutter_params.shutter"],
    })
    timing = camera_timing_table(timing_in)
    pose_median = dict(zip(timing["camera"], timing["pose_delay_ms_median"]))
    trigger_median = dict(zip(timing["camera"], timing["trigger_delay_ms_median"]))

    lb = _read(root, "lidar_box", [
        "key.segment_context_name", "key.frame_timestamp_micros", "key.laser_object_id",
        f"{LB}.type", f"{LB}.num_lidar_points_in_box", f"{LB}.speed.x", f"{LB}.speed.y",
    ])
    boxes = pd.DataFrame({
        "segment": lb["key.segment_context_name"], "frame": lb["key.frame_timestamp_micros"],
        "object_id": lb["key.laser_object_id"], "type": lb[f"{LB}.type"],
        "points": lb[f"{LB}.num_lidar_points_in_box"], "vx": lb[f"{LB}.speed.x"], "vy": lb[f"{LB}.speed.y"],
    })
    sb = _read(root, "lidar_camera_synced_box", [
        "key.segment_context_name", "key.frame_timestamp_micros", "key.laser_object_id",
        f"{SB}.most_visible_camera_name",
    ])
    synced = pd.DataFrame({
        "segment": sb["key.segment_context_name"], "frame": sb["key.frame_timestamp_micros"],
        "object_id": sb["key.laser_object_id"], "most_visible_camera_name": sb[f"{SB}.most_visible_camera_name"],
    })
    pop = population_table(boxes, synced, pose_median)

    result = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {"camera_image_rows": int(len(cam)), "lidar_box_rows": int(len(boxes)),
                   "lidar_camera_synced_box_rows": int(len(synced)),
                   "camera_name_codes_seen": sorted(int(x) for x in cam["key.camera_name"].unique())},
        "timing": timing.to_dict(orient="records"),
        "pose_delay_order_holds": expected_order_holds(pose_median),
        "trigger_delay_order_holds": expected_order_holds(trigger_median),
        "expected_order": list(EXPECTED_SWEEP_ORDER),
        "population": pop.to_dict(orient="records"),
        "population_totals": {k: int(pop[k].sum()) for k in
                              ["eligible_vehicles", "moving_ge_3", "moving_ge_8", "static_le_0_5"]},
        "moving_segments_any_camera": int(boxes.merge(synced, on=["segment", "frame", "object_id"])
                                          .query("type == 1 and points >= 20")
                                          .assign(s=lambda d: (d.vx ** 2 + d.vy ** 2) ** 0.5)
                                          .query("s >= 3")["segment"].nunique()),
        "camera_names": CAMERA_NAMES,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pregate_g3_population.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    pd.set_option("display.width", 200)
    print(timing.round(2).to_string(index=False))
    print("pose order holds:", result["pose_delay_order_holds"], "| trigger order holds:", result["trigger_delay_order_holds"])
    print(pop.round(3).to_string(index=False))
    print("totals:", result["population_totals"], "| segments with a moving eligible vehicle:",
          result["moving_segments_any_camera"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
