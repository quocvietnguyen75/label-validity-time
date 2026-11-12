"""Pre-freeze tables asked for by the T2 audit (findings 1 and 14). Blind-safe: label-side columns and parked rows only.

(1) S1 signed-offset table: median du = p0_u - u_ref and dv = p0_v - v_ref of assigned parked vehicles, per camera, by
    range band, by normalised radius band and by horizontal third; also by sample part beyond 40 m. No pass/fail limit.
(2) Label-side shares per camera: moving rows removed by the seam rule (among rows that pass every other moving
    condition), and the share of included moving rows whose theta_A or theta_B uses the azimuth fallback (< 5 points).
Usage (repository root, env loaded): python <this> <out.json>"""
import json
import os
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


def signed(g):
    return {"n": int(len(g)), "du": med(g["du"]), "dv": med(g["dv"])}


def s1_signed(st):
    st = st.assign(du=st["p0_u"] - st["u_ref"], dv=st["p0_v"] - st["v_ref"])
    out = {}
    for c, g in st.groupby("camera"):
        name = RING[int(c) - 1]
        W, H = (1550.0, 2048.0) if name == "ring_front_center" else (2048.0, 1550.0)
        r = np.hypot((g["p0_u"] - W / 2) / (W / 2), (g["p0_v"] - H / 2) / (H / 2))
        rng = pd.cut(g["range_m"], [0, 10, 20, 40, 1e9], labels=["<10", "10-20", "20-40", ">40"])
        rad = pd.cut(r, [0, 0.4, 0.7, 1.0, 9], labels=["0-0.4", "0.4-0.7", "0.7-1.0", ">1.0"])
        third = np.clip(np.floor(3 * g["p0_u"] / W), 0, 2).map({0: "left", 1: "centre", 2: "right"})
        far = g[g["range_m"] > 40]
        far_r = r[g["range_m"] > 40]
        out[name] = {
            "all": signed(g), "beyond_40m": signed(far),
            "by_range_m": {str(k): signed(v) for k, v in g.groupby(rng, observed=False)},
            "beyond_40m_by_radius": {str(k): signed(v) for k, v in far.groupby(pd.cut(far_r, [0, 0.4, 0.7, 1.0, 9],
                                     labels=["0-0.4", "0.4-0.7", "0.7-1.0", ">1.0"]), observed=False)},
            "beyond_40m_by_third": {str(k): signed(v) for k, v in far.groupby(third[g["range_m"] > 40])},
            "by_radius_all_ranges": {str(k): signed(v) for k, v in g.groupby(rad, observed=False)},
            "beyond_40m_p01_logs": signed(far[far["segment"].isin(P01)]),
            "beyond_40m_drawn_logs": signed(far[~far["segment"].isin(P01)]),
        }
    return out


def label_shares(lab):
    out = {}
    pre = (lab["most_visible"] & (lab["num_pts"] >= 20) & lab["inside_all"] & ~lab["overlap"] & lab["static_proj_ok"]
           & lab["speed"].notna() & (lab["speed"] >= 3.0) & (lab["sdot"] >= 0.25) & (lab["along_axis_deg"] >= 20.0))
    for c, g in lab[pre].groupby("camera"):
        inc = g[~g["seam"]]
        out[RING[int(c) - 1]] = {"moving_before_seam_rule": int(len(g)), "removed_by_seam_rule": int(g["seam"].sum()),
                                 "share_removed": round(float(g["seam"].mean()), 3) if len(g) else None,
                                 "included": int(len(inc)),
                                 "share_theta_A_fallback": round(float(inc["theta_fallback_A"].mean()), 3) if len(inc) else None,
                                 "share_theta_B_fallback": round(float(inc["theta_fallback_B"].mean()), 3) if len(inc) else None}
    tot = lab[pre]
    out["_all"] = {"moving_before_seam_rule": int(len(tot)), "removed_by_seam_rule": int(tot["seam"].sum()),
                   "share_removed": round(float(tot["seam"].mean()), 3)}
    return out


def main(argv):
    root = os.environ["AV2_SENSOR_ROOT"]
    st = pd.read_parquet(os.path.join(root, "derived", "t2_static", "all50_s050", "static_all.parquet"),
                         columns=["segment", "camera", "p0_u", "p0_v", "u_ref", "v_ref", "range_m"])
    cols = ["camera", "most_visible", "num_pts", "inside_all", "overlap", "static_proj_ok", "speed", "sdot",
            "along_axis_deg", "seam", "theta_fallback_A", "theta_fallback_B"]
    lab = pd.read_parquet(os.path.join(root, "derived", "t2_labels_all50_label_side.parquet"), columns=cols)
    out = {"s1_signed_offsets_px_score_0_5": s1_signed(st), "label_side_shares_all50": label_shares(lab)}
    with open(argv[1], "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    for k, v in out["s1_signed_offsets_px_score_0_5"].items():
        print(k, "all", v["all"], "| >40m", v["beyond_40m"], "| >40m thirds", v["beyond_40m_by_third"])
    for k, v in out["label_side_shares_all50"].items():
        print(k, v)


if __name__ == "__main__":
    main(sys.argv)
