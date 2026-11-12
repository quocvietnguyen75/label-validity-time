"""Pre-freeze check (1 Nov 2026): E1 thresholds 5 + B_c^det + hw for the margin-flag-excluded run (margin < 0.2 dropped).
Reads only margin, sdot, sigma_b and keys from the detector moving sets; recomputes n_track and weights; DE and B_c from E0.
No tau-hat, residual or displacement is computed. Run from D:/paper with WAYMO_V2_ROOT set."""
import json
import os

import numpy as np
import pandas as pd

root = os.environ["WAYMO_V2_ROOT"]
E = "newpaper/autonomous_driving_crossmodal_fit_selection_2026-10-13/extension_2026-10-31"
cams = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT"}
for s in ("s050", "s030"):
    mo = pd.read_parquet(os.path.join(root, "derived", "ext_e0", s, "moving_analysis_set_det.parquet"),
                         columns=["segment", "frame", "camera", "object_id", "margin", "sdot", "sigma_b"])
    summ = json.load(open(f"{E}/E0_summary_{s}.json", encoding="utf-8"))
    keep = mo[~(mo["margin"] < 0.2)].copy()
    print(s, "margin-flag-excluded drops", len(mo) - len(keep), "of", len(mo))
    for c, name in cams.items():
        g = keep[keep["camera"] == c].copy()
        g["n_track"] = g.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
        w = g["sdot"] ** 2 / (g["sigma_b"] ** 2 * g["n_track"])
        var = g["sigma_b"] ** 2 / g["sdot"] ** 2
        v = float((w ** 2 * var).sum() / w.sum() ** 2)
        hw = 1.96 * np.sqrt(summ["detector"]["cameras"][name]["DE"] * v)
        B = summ["detector"]["budget"][name]["B_c_ms"]
        print(f"  {name:11s} hw_excl {hw:.2f} threshold_excl {5 + B + hw:.1f}")
