"""EXPLORATORY E2 for XF-C7 (pilot record, 2026-10-22): G8(c) synced-box residuals per segment x camera, with counts.

Re-runs g8_checks.run_segment unchanged (model A1) and writes n and median per cell; the medians must equal those in
g8_checks_all_segment_medians.csv (checked here). Label-only: never reads camera_box or masks.

Run inside WSL:  ~/envs/wod/bin/python e2_synced_cells.py <validation_dir> <out_csv>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g8_checks import run_segment  # noqa: E402


def main(vdir, out):
    segs = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(vdir, "lidar_box", "*.parquet")))
    assert len(segs) == 202
    rows = []
    for s in segs:
        _, c, _ = run_segment(vdir, s)
        rows += [r for r in c if r["model"] == "A1"]
    c = pd.DataFrame(rows)
    cells = c.groupby(["segment", "camera", "readout"])["res_ms"].agg(n="size", res_ms="median").reset_index()
    ref = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "g8_checks_all_segment_medians.csv"))
    m = ref.merge(cells, on=["segment", "camera", "readout"], suffixes=("_ref", ""))
    assert len(m) == len(ref) == len(cells) and np.allclose(m["res_ms_ref"], m["res_ms"]), "does not reproduce G8(c)"
    cells.to_csv(out, index=False, lineterminator="\n")
    print(len(cells), "cells;", int((cells["n"] >= 30).sum()), "with n >= 30; reproduction of G8(c) medians: ok")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
