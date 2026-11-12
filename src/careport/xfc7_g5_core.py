"""XF-C7: build the static zero line and the moving analysis set from a given association (logic of xfc7_g5_run).

Used by the robustness runs (gate x0.5, x2) so that every run rebuilds its zero line and moving set from its own
association, exactly as G5 did for the primary association. `build_sets` with the G5 primary association reproduces
G5's moving analysis set (checked by xfc7_robustness_run before use).
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from careport.xfc7_zero_line import LEVELS, lookup_zero_line, zero_line_tables

K4 = ["segment", "frame", "camera", "object_id"]


def build_sets(lab: pd.DataFrame, assoc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    d = lab.merge(assoc[K4 + ["camera_object_id", "u_ref", "v_ref", "w_ref", "cost", "margin"]], on=K4, how="left")
    d["assigned"] = d["camera_object_id"].notna()
    st = d[d["incl_static"] & d["assigned"]].copy()
    st["e"] = (st["p0_u"] - st["u_ref"]) * st["e_u"] + (st["p0_v"] - st["v_ref"]) * st["e_v"]
    tables = zero_line_tables(st)
    st = st.join(lookup_zero_line(st[LEVELS[0]], tables))
    st = st[~st["no_zero_line"]].copy()
    st["r"] = st["e"] - st["z_b"]
    st["n_track"] = st.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    st["w"] = 1.0 / (st["sigma_b"] ** 2 * st["n_track"])
    mo = d[d["incl_moving"] & d["assigned"]].copy()
    mo = mo.join(lookup_zero_line(mo[LEVELS[0]], tables))
    mo = mo[~mo["no_zero_line"] & (mo["sigma_b"] > 0)].copy()
    mo["n_track"] = mo.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    mo["w"] = mo["sdot"] ** 2 / (mo["sigma_b"] ** 2 * mo["n_track"])
    mo["var_tau"] = mo["sigma_b"] ** 2 / mo["sdot"] ** 2
    return st, mo, tables
