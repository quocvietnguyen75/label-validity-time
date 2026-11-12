"""Pre-freeze record (1 Nov 2026): the section 6 copying-bound separations, blind-safe.

Separation per camera = E2-weighted mean of tau_lidar - tau_frame (LiDAR-time seeds) and of tau_cap - tau_frame
(capture-time seeds, added after the audit) over the common moving rows, where the weights are
E0's common-set weights (xfc7_ext_e0.common_moving) and tau_* are the human set's label-only timing columns.
Reads only keys, sdot, sigma_b, tau_frame, tau_lidar and tau_cap; never u_ref, v_ref, p0, z_b, d_u, d_v or s, so no tau-hat,
residual or displacement can be formed. Run from D:/paper with PYTHONPATH=src and WAYMO_V2_ROOT set.

The values were first obtained on 1 Nov by an equivalent one-off command in the agent session (same columns, same
weights written out by hand); e2_pair was not run on the real sets."""
import os

import numpy as np
import pandas as pd

from careport.xfc7_ext_e0 import K4, common_moving

root = os.environ["WAYMO_V2_ROOT"]
cams = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT"}
h = pd.read_parquet(os.path.join(root, "derived", "g5", "moving_analysis_set.parquet"),
                    columns=K4 + ["sdot", "sigma_b", "tau_frame", "tau_lidar", "tau_cap"])
for s in ("s050", "s030"):
    d = pd.read_parquet(os.path.join(root, "derived", "ext_e0", s, "moving_analysis_set_det.parquet"),
                        columns=K4 + ["sigma_b"])
    cm = common_moving(h, d).merge(h[K4 + ["tau_frame", "tau_lidar", "tau_cap"]], on=K4)
    for col, label in (("tau_lidar", "frame-LiDAR"), ("tau_cap", "frame-capture")):
        vals = {name: float(np.average((g[col] - g["tau_frame"]), weights=g["w_pair"]))
                for c, name in cams.items() for g in [cm[cm["camera"] == c]]}
        print(s, "common rows", len(cm), f"| {label} separation (ms):", ", ".join(f"{k} {v:.2f}" for k, v in vals.items()))
