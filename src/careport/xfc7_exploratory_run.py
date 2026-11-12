"""XF-C7 EXPLORATORY E1 and E2 (pilot record, 2026-10-22; specified there before running). Neither changes a verdict.

Usage: PYTHONPATH=src python -m careport.xfc7_exploratory_run <vehicle_label_only.parquet> <g5_summary.json> <e2_cells.csv|-> <out_dir>
E1: static residual after its zero line, per readout direction, on xi and xi x ego speed with camera intercepts, CR1 by
    segment; implied moving-set slope Sum w s (b_xi + b_xiv v_ego) / sdot / Sum w (approximate, descriptive).
E2: G8(c) synced-box cell medians (cells with n >= 30), share with |median| > 5 ms, and the Spearman correlation across
    cells with |static offset| (median static residual of the segment x camera, in ms at the cell's median moving sdot).
Writes <out_dir>/exploratory_v2.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from careport.xfc7_estimators import wls_cr1
from careport.xfc7_g5_core import build_sets
from careport.xfc7_g5_run import sha
from careport.xfc7_primary import CAM, RO


def e1(st: pd.DataFrame, mo: pd.DataFrame) -> dict:
    out = {}
    for r, g in st.groupby("readout"):
        X = np.stack([g["xi"].to_numpy(), (g["xi"] * g["ego_speed"]).to_numpy()], axis=1)
        b, se = wls_cr1(g["r"], X, g["w"], g["camera"], g["segment"])
        m = mo[mo["readout"] == r]
        implied = float((m["w"] * m["s"] * (b[0] + b[1] * m["ego_speed"]) / m["sdot"]).sum() / m["w"].sum())
        out[RO[int(r)]] = {"obs": int(len(g)), "segments": int(g["segment"].nunique()),
                           "b_xi_px": {"est": float(b[0]), "lo": float(b[0] - 1.96 * se[0]), "hi": float(b[0] + 1.96 * se[0])},
                           "b_xi_x_ego_speed_px_per_mps": {"est": float(b[1]), "lo": float(b[1] - 1.96 * se[1]), "hi": float(b[1] + 1.96 * se[1])},
                           "implied_moving_slope_ms_per_xi": implied}
    i = {k: v["b_xi_x_ego_speed_px_per_mps"] for k, v in out.items()}
    excl = any(v["lo"] > 0 or v["hi"] < 0 for v in i.values())
    differ = np.sign(i["L2R"]["est"]) != np.sign(i["R2L"]["est"]) or i["L2R"]["lo"] > i["R2L"]["hi"] or i["R2L"]["lo"] > i["L2R"]["hi"]
    out["appears"] = bool(excl and differ)
    out["implied_conflict_ms_per_xi"] = out["L2R"]["implied_moving_slope_ms_per_xi"] - out["R2L"]["implied_moving_slope_ms_per_xi"]
    return out


def e2(cells: pd.DataFrame, st: pd.DataFrame, lab: pd.DataFrame) -> dict:
    c = cells[cells["n"] >= 30].copy()
    off = st.groupby(["segment", "camera"])["r"].median().rename("static_px").reset_index()
    sd = lab[lab["incl_moving"]].groupby(["segment", "camera"])["sdot"].median().rename("sdot_med").reset_index()
    off = off.merge(sd, on=["segment", "camera"])
    off["camera"] = off["camera"].map(lambda k: CAM[int(k)])
    m = c.merge(off, on=["segment", "camera"], how="inner")
    m["static_ms"] = m["static_px"] / m["sdot_med"]
    rho, p = spearmanr(m["res_ms"].abs(), m["static_ms"].abs())
    return {"cells_n_ge_30": int(len(c)), "share_abs_median_gt_5ms": float((c["res_ms"].abs() > 5).mean()),
            "cells_with_static_offset": int(len(m)), "spearman_abs": float(rho), "spearman_p": float(p),
            "by_camera_share_gt_5ms": {k: float((g["res_ms"].abs() > 5).mean()) for k, g in c.groupby("camera")}}


def main(argv):
    lab_path, g5_path, cells_path, out_dir = argv[1:5]
    g5 = json.load(open(g5_path, encoding="utf-8"))
    assert sha(lab_path) == g5["inputs"]["label_only_sha256"]
    lab = pd.read_parquet(lab_path)
    prim = pd.read_parquet(g5["association"]["table"])
    st, mo, _ = build_sets(lab, prim)
    out = {"E1": e1(st, mo)}
    if cells_path != "-":
        out["E2"] = e2(pd.read_csv(cells_path), st, lab)
        out["E2"]["cells_sha256"] = sha(cells_path)
    path = os.path.join(out_dir, "exploratory_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
