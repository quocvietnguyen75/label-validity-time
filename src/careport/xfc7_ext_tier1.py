"""XF-C7 extension, tier 1: E1-E5 of EXT protocol v1.0, executed ONCE after the freeze.

E1: the frozen v2.0 rule (xfc7_primary_v2_run.verdicts, xfc7_decision_v2) on the detector analysis set, with the
    detector's own B_c^det, in the primary and the margin-flag-excluded runs.
E2: the paired difference d_i = tau-hat_human - tau-hat_det on observations both references assign; weighted mean per
    camera, 95 % CR1 by segment; agree / disagree / inconclusive against +-5 ms; the copying bound on rho.
E3: d_i regressed on 1/sdot (intercept in ms = copying-like, slope in px = detector pixel bias); descriptive.
E4: injected shifts of the matched detector boxes, as B4 did for human boxes (xfc7_revision_shift_run).
E5: E1 and E2 at the sensitivity score threshold.

Usage: PYTHONPATH=src python -m careport.xfc7_ext_tier1 <label_only.parquet> <e0_primary.json> <e0_sensitivity.json>
       <ext_protocol.md> <expected_protocol_sha256> <out.json>"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from careport.xfc7_association import associate_images
from careport.xfc7_decision_v2 import combine_runs, decide_rejection, rejected
from careport.xfc7_estimators import weighted_mean_cr1, wls_intercept_slope_cr1
from careport.xfc7_g5_core import K4, build_sets
from careport.xfc7_posthoc import shift_along
from careport.xfc7_primary import CAM, HYP, add_tau_hat, analyse
from careport.xfc7_primary_v2_run import verdicts

FROZEN_MOVING_SHA = "54016253c11a29c367dc0c2590fcbdf793d380e6a6529cafcb98e2ef6bbff32f"
# E0 records, pinned before the freeze (EXT v1.0 section 14); B_c^det, the camera list and the moving-set hashes come from them
E0_PRIMARY_SHA = "12238bd66cdb9f231c781142b1ef5b9a912b7d144fb3d40825668393e8d89ca1"       # E0_summary_s050.json
E0_SENSITIVITY_SHA = "f8cfd7d8aed42b7ec6bf82d6b0b2fae5086184473633168d572f9028a8a29e91"   # E0_summary_s030.json
ASSOC_PRIMARY_SHA = "0951961a345debe2c30aeab62b475f54c289751b89bd20064944ccf7094cb96f"    # ext_e0/s050/association.parquet
V2_DECISION_CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT"]
MARGIN_MS = 5.0
IMG = ["segment", "frame", "camera", "camera_object_id"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def check_sha(path, expected, what):
    """The file's SHA-256 if it equals the pinned value; otherwise stop before anything is computed."""
    got = sha(path)
    if got != expected:
        raise SystemExit(f"{what}: hash {got} does not match the pinned {expected}; stop")
    return got


def classify_pair(lo: float, hi: float, margin: float = MARGIN_MS) -> str:
    if lo >= -margin and hi <= margin:
        return "agree"
    if lo > margin or hi < -margin:
        return "disagree"
    return "inconclusive"


def overall_pair(decisions) -> str:
    vals = list(decisions.values())
    if "disagree" in vals:
        return "disagree"
    return "agree" if vals.count("agree") >= 2 else "inconclusive"


def copy_bound(lo: float, hi: float, separation: float) -> float:
    """Copying under frame start moves the human estimate toward LiDAR time by rho x separation (> 0)."""
    return max(hi, 0.0) / separation


def _pair_rows(mo_h: pd.DataFrame, mo_d: pd.DataFrame) -> pd.DataFrame:
    timing = [c for c in ("tau_frame", "tau_lidar", "tau_cap") if c in mo_h.columns]
    h = add_tau_hat(mo_h)[K4 + ["tau_hat", "sdot", "sigma_b"] + timing]
    d = add_tau_hat(mo_d)[K4 + ["tau_hat", "sigma_b"]]
    m = h.merge(d, on=K4, suffixes=("_h", "_d"))
    m["d"] = m["tau_hat_h"] - m["tau_hat_d"]
    n_track = m.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    m["w_pair"] = m["sdot"] ** 2 / ((m["sigma_b_h"] ** 2 + m["sigma_b_d"] ** 2) * n_track)
    return m


def e2_pair(mo_h: pd.DataFrame, mo_d: pd.DataFrame, cameras=None, min_tracks: int = 30):
    m = _pair_rows(mo_h, mo_d)
    out, dec = {"cameras": {}}, {}
    for c, g in m.groupby("camera"):
        name = CAM[int(c)]
        if cameras is not None and name not in cameras:
            continue
        est, se = weighted_mean_cr1(g["d"], g["w_pair"], g["segment"])
        lo, hi = est - 1.96 * se, est + 1.96 * se
        tracks = int(g[["segment", "object_id"]].drop_duplicates().shape[0])
        (b0, s0), (b1, s1) = wls_intercept_slope_cr1(g["d"], 1.0 / g["sdot"], g["w_pair"], g["segment"])
        row = {"common_obs": int(len(g)), "common_tracks": tracks, "segments": int(g["segment"].nunique()),
               "delta_pair": {"est": float(est), "lo": float(lo), "hi": float(hi), "se": float(se)},
               "e3_intercept_ms": {"est": b0, "se": s0}, "e3_slope_px": {"est": b1, "se": s1}}
        if tracks >= min_tracks:
            row["decision"] = classify_pair(lo, hi)
            dec[name] = row["decision"]
            if name in V2_DECISION_CAMERAS and {"tau_frame", "tau_lidar"} <= set(g.columns):
                # EXT v1.0 section 6: the separation on the common rows, with E2's weights
                sep = float(np.average(g["tau_lidar"] - g["tau_frame"], weights=g["w_pair"]))
                row["frame_lidar_separation_ms"] = sep
                row["copy_bound_rho"] = copy_bound(lo, hi, sep)
                if "tau_cap" in g.columns:      # seeds at the labelled position match capture time (audit finding 10)
                    sep_c = float(np.average(g["tau_cap"] - g["tau_frame"], weights=g["w_pair"]))
                    row["frame_capture_separation_ms"] = sep_c
                    row["copy_bound_rho_capture_seed"] = copy_bound(lo, hi, sep_c)
        else:
            row["decision"] = "descriptive (< 30 common tracks)"
        out["cameras"][name] = row
    out["overall"] = overall_pair(dec) if dec else "no camera with >= 30 common tracks"
    return out


def frame_failure_labels(res, cams, budgets, margin: float = MARGIN_MS):
    """Per camera, for H_frame (EXT v1.0 section 10): rejected, capacity failure (|est| > 5 + B_c but the CI reaches
    inside) or estimate failure (|est| <= 5 + B_c). Descriptive; the verdict rule is unchanged."""
    out = {}
    for c in cams:
        if c not in res["cameras"]:
            continue
        d = res["cameras"][c]["delta"]["frame"]
        if rejected((d["est"], d["lo"], d["hi"]), budgets[c], margin):
            out[c] = "rejected"
        else:
            out[c] = "capacity failure" if abs(d["est"]) > margin + budgets[c] else "estimate failure"
    return out


def frame_ruled_out_at_point_estimates(res, cams, budgets) -> bool:
    """Would the v2.0 rule rule out H_frame if every interval shrank to its point estimate?"""
    pts = {"frame": {c: (res["cameras"][c]["delta"]["frame"]["est"],) * 3 for c in cams if c in res["cameras"]}}
    return bool(decide_rejection(pts, budgets, [c for c in cams if c in res["cameras"]])["frame"]["ruled_out"])


def tier2_gate(e1_replicated: bool, capacity_exception: bool, e2: dict) -> dict:
    """EXT v1.0 section 11: E1 replicates and no decision camera disagrees in E2; or the capacity exception (H_frame
    missed only through capacity failures) together with an overall E2 'agree'."""
    none_disagree = all(v.get("decision") != "disagree" for v in e2["cameras"].values())
    if e1_replicated and none_disagree:
        route = "replicated"
    elif capacity_exception and e2.get("overall") == "agree":
        route = "capacity exception"
    else:
        route = None
    return {"E1_replicated": bool(e1_replicated), "capacity_exception": bool(capacity_exception),
            "no_camera_disagrees": bool(none_disagree), "E2_overall": e2.get("overall"), "route": route,
            "pass": route is not None}


def e1_verdicts(mo_d: pd.DataFrame, e0: dict):
    cams = [c for c, v in e0["detector"]["cameras"].items() if v["tracks"] >= 30]
    budgets = {c: v["B_c_ms"] for c, v in e0["detector"]["budget"].items()}
    res = analyse(mo_d, [], [])
    prim = verdicts(res, cams, budgets)
    sub = mo_d[(mo_d["margin"] >= 0.2) | np.isinf(mo_d["margin"])].copy()
    sub["n_track"] = sub.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    sub["w"] = sub["sdot"] ** 2 / (sub["sigma_b"] ** 2 * sub["n_track"])
    rm = analyse(sub, [], [])
    mexc = verdicts(rm, cams, budgets)
    final = combine_runs(prim, mexc)
    at_points = frame_ruled_out_at_point_estimates(res, cams, budgets) and frame_ruled_out_at_point_estimates(rm, cams, budgets)
    return {"cameras_in_analysis": cams, "budgets_ms": budgets, "final_verdicts": final,
            "frame_labels_primary": frame_failure_labels(res, cams, budgets),
            "frame_labels_margin_flag_excluded": frame_failure_labels(rm, cams, budgets),
            "capacity_exception": bool(final["frame"] != "ruled out" and at_points),
            "verdicts_primary_run": prim, "verdicts_margin_flag_excluded": mexc,
            "estimates": {c: {"tracks": v["tracks"], "obs": v["obs"], "delta": v["delta"]} for c, v in res["cameras"].items()},
            "replicated": None}


def e4_shifts(lab, ref, mo_d, assoc, deltas=(-20.0, -10.0, 10.0, 20.0)):
    base = analyse(mo_d, [], [])
    matched = mo_d[K4 + ["sdot", "d_u", "d_v"]].merge(assoc[K4 + ["camera_object_id"]], on=K4, how="left")
    matched = matched[matched["camera_object_id"].notna()]
    runs = {}
    for dlt in deltas:
        uv = shift_along(np.zeros((len(matched), 2)), matched[["d_u", "d_v"]].to_numpy(float), matched["sdot"].to_numpy(float), dlt)
        sh = matched[IMG].assign(du=uv[:, 0], dv=uv[:, 1]).drop_duplicates(IMG)
        b = ref.merge(sh, on=IMG, how="left")
        b["u"] = b["u"] + b["du"].fillna(0.0)
        b["v"] = b["v"] + b["dv"].fillna(0.0)
        a2 = associate_images(lab, b.drop(columns=["du", "dv"]), 1.0)
        _, mo2, _ = build_sets(lab, a2)
        r = analyse(mo2, [], [])
        chk = mo_d[K4].merge(assoc[K4 + ["camera_object_id"]], on=K4, how="left").merge(
            a2[K4 + ["camera_object_id"]].rename(columns={"camera_object_id": "cid_new"}), on=K4, how="left")
        run = {"assignment_changed_share": float((chk["camera_object_id"].fillna("") != chk["cid_new"].fillna("")).mean()),
               "cameras": {}}
        for c, x in r["cameras"].items():
            if c in base["cameras"]:
                d0, d1 = base["cameras"][c]["delta"]["lidar"]["est"], x["delta"]["lidar"]["est"]
                run["cameras"][c] = {"recovered_shift_ms": d1 - d0, "recovery_ratio": (d1 - d0) / dlt}
        runs[f"{dlt:+.0f}ms"] = run
    return runs


def e4_pass(runs, cams, band=(0.9, 1.1)) -> bool:
    """EXT v1.0 section 8: recovery within 90-110 % in every decision camera, in every shift run."""
    return all(band[0] <= r["cameras"][c]["recovery_ratio"] <= band[1]
               for r in runs.values() for c in cams if c in r["cameras"])


def main(argv):
    if len(argv) != 7:
        print(__doc__)
        return 64
    lab_path, e0_path, e0s_path, proto_path, proto_sha, out_path = argv[1:7]
    if sha(proto_path) != proto_sha:
        raise SystemExit("frozen EXT protocol hash mismatch")
    root = os.environ["WAYMO_V2_ROOT"]
    check_sha(e0_path, E0_PRIMARY_SHA, "E0 primary summary")
    check_sha(e0s_path, E0_SENSITIVITY_SHA, "E0 sensitivity summary")
    e0, e0s = (json.load(open(p, encoding="utf-8")) for p in (e0_path, e0s_path))
    if sha(lab_path) != e0["inputs"]["label_only_sha256"]:
        raise SystemExit("label-only file hash mismatch")
    h_path = os.path.join(root, "derived", "g5", "moving_analysis_set.parquet")
    if sha(h_path) != FROZEN_MOVING_SHA:
        raise SystemExit("frozen human moving analysis set hash mismatch")
    mo_h = pd.read_parquet(h_path)
    out = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "inputs": {"protocol_sha256": proto_sha, "e0_primary_sha256": sha(e0_path), "e0_sensitivity_sha256": sha(e0s_path)}}
    for tag, summ in (("primary", e0), ("sensitivity_score_0_3", e0s)):
        odir = os.path.join(root, "derived", "ext_e0", f"s{int(round(summ['score_min'] * 100)):03d}")
        m_path = os.path.join(odir, "moving_analysis_set_det.parquet")
        if sha(m_path) != summ["detector"]["moving_analysis_set_sha256"]:
            raise SystemExit(f"{tag}: detector moving set does not match its E0 record")
        mo_d = pd.read_parquet(m_path)
        e1 = e1_verdicts(mo_d, summ)
        e1["replicated"] = e1["final_verdicts"]["frame"] == "ruled out"
        e2 = e2_pair(mo_h, mo_d, cameras=V2_DECISION_CAMERAS)
        out[tag] = {"E1": e1, "E2_E3": e2}
        if tag == "primary":
            lab = pd.read_parquet(lab_path)
            check_sha(os.path.join(odir, "reference_table.parquet"), summ["inputs"]["reference_table_sha256"], "reference table")
            check_sha(os.path.join(odir, "association.parquet"), ASSOC_PRIMARY_SHA, "E0 association table")
            ref = pd.read_parquet(os.path.join(odir, "reference_table.parquet"))
            assoc = pd.read_parquet(os.path.join(odir, "association.parquet"))
            out[tag]["E4"] = e4_shifts(lab, ref, mo_d, assoc)
            out[tag]["E4_pass_band_90_110"] = e4_pass(out[tag]["E4"], e1["cameras_in_analysis"])
    p = out["primary"]
    out["tier2_gate"] = tier2_gate(p["E1"]["replicated"], p["E1"]["capacity_exception"], p["E2_E3"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps({"E1": p["E1"]["final_verdicts"], "E2": {c: v["decision"] for c, v in p["E2_E3"]["cameras"].items()},
                      "E2_overall": p["E2_E3"]["overall"], "gate": out["tier2_gate"]}, indent=1))
    print("results:", out_path, "sha256", sha(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
