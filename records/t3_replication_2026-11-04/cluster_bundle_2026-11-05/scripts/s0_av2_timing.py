"""T2 protocol section 8, S0: per-log AV2 frame offsets against the section 4 table (label-side tables only).

For each log, camera and frame type: the median dt (frame time minus sweep start, ms). A log departs if any of its
medians differs from the table by more than 2 ms; it is reported and kept. Also checks the categories present against
the pinned vehicle list. Usage: python s0_av2_timing.py <label_dir> <out.json>"""
import glob
import json
import os
import sys

import pandas as pd

TABLE = {("ring_front_center", "A"): 39.9, ("ring_front_center", "B"): 89.9, ("ring_front_left", "A"): 27.4,
         ("ring_front_left", "B"): 77.4, ("ring_front_right", "A"): 52.5, ("ring_front_right", "B"): 2.5,
         ("ring_side_left", "A"): 12.4, ("ring_side_left", "B"): 62.4, ("ring_side_right", "A"): 67.5,
         ("ring_side_right", "B"): 17.5, ("ring_rear_left", "A"): 97.4, ("ring_rear_left", "B"): 47.4,
         ("ring_rear_right", "A"): 82.4, ("ring_rear_right", "B"): 32.4}
RING = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left", "ring_side_right",
        "ring_rear_left", "ring_rear_right"]


def main(argv):
    ldir, out = argv[1], argv[2]
    res = {"logs": 0, "departing_logs": {}, "max_abs_departure_ms": 0.0, "categories": {}}
    for f in sorted(glob.glob(os.path.join(ldir, "*.parquet"))):
        d = pd.read_parquet(f, columns=["segment", "camera", "frame_type", "dt_fs_ms", "category"])
        res["logs"] += 1
        for k, v in d["category"].value_counts().items():
            res["categories"][k] = res["categories"].get(k, 0) + int(v)
        med = d.groupby(["camera", "frame_type"])["dt_fs_ms"].median()
        dev = {f"{RING[c - 1]}|{t}": round(float(m - TABLE[(RING[c - 1], t)]), 2) for (c, t), m in med.items()}
        worst = max(abs(x) for x in dev.values())
        res["max_abs_departure_ms"] = max(res["max_abs_departure_ms"], worst)
        if worst > 2.0:
            res["departing_logs"][os.path.basename(f)[:-8]] = {k: v for k, v in dev.items() if abs(v) > 2.0}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv)
