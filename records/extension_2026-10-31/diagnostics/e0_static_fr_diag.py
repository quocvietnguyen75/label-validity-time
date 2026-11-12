"""Static-only diagnostic of E0's design effect per camera (detector vs human). Prints aggregates only.
No moving row, tau-hat or moving residual is touched."""
import os
import sys

import numpy as np
import pandas as pd

from careport.xfc7_estimators import design_effect_mean, weighted_mean_cr1
from careport.xfc7_ext_e0 import run_reference
from careport.xfc7_g5_run import CAM, load_boxes

root = os.environ["WAYMO_V2_ROOT"]
lab = pd.read_parquet(os.path.join(root, "derived", "g4_label_only.parquet"))
ref = pd.read_parquet(os.path.join(root, "derived", "ext_e0", "s050", "reference_table.parquet"))
out = {}
for name, boxes in (("human", load_boxes(os.path.join(root, "validation"))), ("det", ref)):
    st = run_reference(lab, boxes)["st"]
    out[name] = st
    print(f"== {name}: static used {len(st)}")
    for c, sc in st.groupby("camera"):
        de = design_effect_mean(sc["r"], sc["w"], sc["sigma_b"] ** 2, sc["segment"])
        m, se = weighted_mean_cr1(sc["r"], sc["w"], sc["segment"])
        g = sc.groupby("segment").apply(lambda x: pd.Series({"wshare": x["w"].sum(), "rbar": np.average(x["r"], weights=x["w"]), "n": len(x)}), include_groups=False)
        g["wshare"] /= g["wshare"].sum()
        g["infl"] = (g["wshare"] * (g["rbar"] - m)) ** 2
        g["infl"] /= g["infl"].sum()
        top = g.sort_values("infl", ascending=False).head(3)
        print(f"  {CAM[c]:11s} n={len(sc):6d} segs={sc['segment'].nunique():3d} DE={de:9.1f} mean={m:6.2f}px se={se:5.2f} "
              f"sigma_b med={sc['sigma_b'].median():5.2f} min={sc['sigma_b'].min():5.3f} | w top-seg share {g['wshare'].max():.2f} | "
              f"top-3 segs' share of CR1 variance {top['infl'].sum():.2f} (wshare {', '.join(f'{v:.2f}' for v in top['wshare'])}; "
              f"rbar {', '.join(f'{v:.1f}' for v in top['rbar'])} px; n {', '.join(str(int(v)) for v in top['n'])})")
        if name == "det" and c == 3:
            fr = sc
# where FRONT_RIGHT detector weight concentrates: by zero-line bin
fr = out["det"][out["det"]["camera"] == 3]
binc = [c for c in ("viewpoint", "range_bin", "column_third", "ego_class", "bin") if c in fr.columns]
print("FR det columns available:", [c for c in fr.columns if "bin" in c or c in ("viewpoint", "zl_level")][:12])
wsum = fr["w"].sum()
small = fr.nsmallest(20, "sigma_b")[["segment", "sigma_b", "w", "r", "n_track"]]
print("FR det: weight share of 1% smallest-sigma rows:", round(fr.nsmallest(max(1, len(fr) // 100), "sigma_b")["w"].sum() / wsum, 3))
print("FR det: 20 smallest sigma_b (seg hashed):")
small = small.assign(segment=small["segment"].map(lambda s: s[:6]))
print(small.to_string(index=False))
