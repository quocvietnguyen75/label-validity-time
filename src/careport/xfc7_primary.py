"""XF-C7 primary analysis (protocol v1.0 section 5): tau-hat, per-camera means and differences, pooled slopes, decision.

`analyse` is pure: it takes the moving analysis set (one row per included observation with p0, u_ref, d, s, z_b, sdot,
weights w, xi, the four predictions and the readout direction) and the G5 eligibility lists, and returns estimates
with 95 % segment-clustered (CR1) intervals and the decision. Intervals use 1.96 standard errors.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from careport.xfc7_decision import decide
from careport.xfc7_estimators import weighted_mean_cr1, wls_slope_cr1
from careport.xfc7_zero_line import tau_hat

CAM = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
RO = {2: "L2R", 4: "R2L"}
HYP = {"frame": "tau_frame", "lidar": "tau_lidar", "pose": "tau_pose", "cam": "tau_cap"}


def _ci(est: float, se: float) -> Dict[str, float]:
    return {"est": float(est), "se": float(se), "lo": float(est - 1.96 * se), "hi": float(est + 1.96 * se)}


def add_tau_hat(mo: pd.DataFrame) -> pd.DataFrame:
    out = mo.copy()
    out["tau_hat"] = tau_hat(out[["p0_u", "p0_v"]].to_numpy(), out[["u_ref", "v_ref"]].to_numpy(), out[["d_u", "d_v"]].to_numpy(),
                             out["s"].to_numpy(), out["z_b"].to_numpy(), out["sdot"].to_numpy())
    return out


def analyse(mo: pd.DataFrame, decision_cameras: Sequence[str], slope_cells: Sequence[str]) -> Dict[str, object]:
    d = add_tau_hat(mo)
    cams: Dict[str, Dict] = {}
    for c, g in d.groupby("camera"):
        name = CAM[int(c)]
        tb, tbse = weighted_mean_cr1(g["tau_hat"], g["w"], g["segment"])
        delta = {h: _ci(*weighted_mean_cr1(g["tau_hat"] - g[col], g["w"], g["segment"])) for h, col in HYP.items()}
        cams[name] = {"obs": int(len(g)), "tracks": int(g[["segment", "object_id"]].drop_duplicates().shape[0]),
                      "segments": int(g["segment"].nunique()), "tau_bar": _ci(tb, tbse), "delta": delta}
    slopes: Dict[str, Dict] = {}
    for r, g in d.groupby("readout"):
        slopes[RO[int(r)]] = {h: _ci(*wls_slope_cr1(g["tau_hat"] - g[col], g["xi"], g["w"], g["camera"], g["segment"]))
                              for h, col in HYP.items()}
        slopes[RO[int(r)]]["obs"] = int(len(g))
    cell_slopes: Dict[str, Dict] = {}
    for (c, r), g in d.groupby(["camera", "readout"]):
        if g["segment"].nunique() > 2 and g["xi"].std() > 0:
            cell_slopes[f"{CAM[int(c)]}|{RO[int(r)]}"] = {
                h: _ci(*wls_slope_cr1(g["tau_hat"] - g[col], g["xi"], g["w"], g["camera"], g["segment"])) for h, col in HYP.items()}
    delta_ci = {h: {c: (cams[c]["delta"][h]["lo"], cams[c]["delta"][h]["hi"]) for c in decision_cameras if c in cams} for h in HYP}
    slope_ci = {h: {r: (slopes[r][h]["lo"], slopes[r][h]["hi"]) for r in slope_cells if r in slopes} for h in HYP}
    dec = decide(delta_ci, slope_ci, [c for c in decision_cameras if c in cams], [r for r in slope_cells if r in slopes])
    return {"cameras": cams, "slopes": slopes, "cell_slopes_display": cell_slopes, "decision": dec,
            "decision_cameras": list(decision_cameras), "slope_cells": list(slope_cells)}
