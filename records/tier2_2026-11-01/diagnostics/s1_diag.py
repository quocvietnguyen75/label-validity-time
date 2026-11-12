"""T2 S1 diagnostic (parked vehicles only; blind-safe: no moving row is read).

Breaks the S1 statistic (median |p0 - u_ref| per camera over all assigned parked observations) down by sample part
(P01's ten logs vs the forty drawn), by log, by range, by box size and by image position, and gives the signed
components. Purpose: tell a projection (calibration/distortion) fault, which grows toward the image edge, from a
box-geometry effect, which scales with the box's size in the image.
Usage: python s1_diag.py <static_all.parquet> <out.json>"""
import json
import sys

import numpy as np
import pandas as pd

RING = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left", "ring_side_right",
        "ring_rear_left", "ring_rear_right"]
P01 = {"3ca11a5e-50b2-3cc3-af7a-ce7ab02b9954", "58d01358-5927-36fa-9e11-d18d1dc1f4f0",
       "6b14d7c0-20f9-390b-af38-507a5de5998c", "81d2b40a-c579-3e9c-b520-bee26cda947d",
       "c556f8e0-a001-3586-b2cf-d3256685c39f", "c780d53a-2d37-3cd8-9e89-530966aef53e",
       "f648b945-6c70-3105-bd23-9502894e37d4", "fd5c6932-2ee2-3cfb-9bdc-0b30bfb33a91",
       "9a448a80-0e9a-3bf0-90f3-21750dfef55a", "d5d6f11c-3026-3e0e-9d67-c111233e22de"}


def med(x):
    return round(float(np.median(x)), 1) if len(x) else None


def main(argv):
    st = pd.read_parquet(argv[1])
    st = st.assign(du=st["p0_u"] - st["u_ref"], dv=st["p0_v"] - st["v_ref"])
    st["off"] = np.hypot(st["du"], st["dv"])
    st["rel"] = st["off"] / st["w_ref"]
    st["hull_w"] = st["h0_u1"] - st["h0_u0"]
    st["p01"] = st["segment"].isin(P01)
    out = {}
    for c, g in st.groupby("camera"):
        name = RING[int(c) - 1]
        W, H = (1550.0, 2048.0) if name == "ring_front_center" else (2048.0, 1550.0)
        r = np.hypot((g["p0_u"] - W / 2) / (W / 2), (g["p0_v"] - H / 2) / (H / 2))
        per_log = g.groupby("segment")["off"].median()
        rng = pd.cut(g["range_m"], [0, 10, 20, 40, 1e9], labels=["<10", "10-20", "20-40", ">40"])
        wq = pd.qcut(g["w_ref"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        rad = pd.cut(r, [0, 0.4, 0.7, 1.0, 9], labels=["0-0.4", "0.4-0.7", "0.7-1.0", ">1.0"])
        out[name] = {
            "n": int(len(g)), "median_px": med(g["off"]),
            "p01_logs": {"n": int(g["p01"].sum()), "median_px": med(g.loc[g["p01"], "off"])},
            "drawn_logs": {"n": int((~g["p01"]).sum()), "median_px": med(g.loc[~g["p01"], "off"])},
            "per_log_median_px": {"logs": int(len(per_log)), "min": round(float(per_log.min()), 1),
                                  "q50": med(per_log), "max": round(float(per_log.max()), 1),
                                  "logs_over_15": int((per_log > 15).sum())},
            "signed_median_px": {"du": med(g["du"]), "dv": med(g["dv"])},
            "median_rel_offset_of_box_width": round(float(np.median(g["rel"])), 3),
            "by_range_m": {str(k): {"n": int(len(v)), "median_px": med(v["off"]), "rel": round(float(np.median(v["rel"])), 3) if len(v) else None}
                           for k, v in g.groupby(rng, observed=False)},
            "by_box_width_quartile": {str(k): {"n": int(len(v)), "w_ref_median": med(v["w_ref"]), "median_px": med(v["off"]),
                                               "rel": round(float(np.median(v["rel"])), 3)}
                                      for k, v in g.groupby(wq, observed=False)},
            "by_normalised_radius": {str(k): {"n": int(len(v)), "median_px": med(v["off"]),
                                              "rel": round(float(np.median(v["rel"])), 3) if len(v) else None}
                                     for k, v in g.groupby(rad, observed=False)},
        }
    json.dump(out, open(argv[2], "w", encoding="utf-8"), indent=1)
    for k, v in out.items():
        print(k, v["median_px"], "| P01", v["p01_logs"]["median_px"], "drawn", v["drawn_logs"]["median_px"],
              "| per-log", v["per_log_median_px"], "| signed", v["signed_median_px"], "| rel", v["median_rel_offset_of_box_width"])
        print("   range", {a: (b["median_px"], b["rel"], b["n"]) for a, b in v["by_range_m"].items()})
        print("   width", {a: (b["w_ref_median"], b["median_px"], b["rel"]) for a, b in v["by_box_width_quartile"].items()})
        print("   radius", {a: (b["median_px"], b["rel"], b["n"]) for a, b in v["by_normalised_radius"].items()})


if __name__ == "__main__":
    main(sys.argv)
