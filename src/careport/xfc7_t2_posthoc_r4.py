"""XF-C7 post hoc set R4 (pilot record entry '2026-11-03, post hoc set R4', pinned before computing).

All post hoc; no frozen verdict or published interval changes. Inputs are the frozen moving sets and result files,
hash-checked against the pin before anything is computed.
- R4-1 (review item R18): small-sample intervals for every tested camera and convention, in both runs, for AV2 and for
  WOD human boxes and the WOD detector. (a) primary: CR2 with Bell-McCaffrey degrees of freedom under a working variance
  proportional to 1/w; (b) secondary: wild cluster bootstrap-t (unrestricted residuals, Webb six-point weights, 99,999
  draws, seed 20261103, CR1 studentization); (c) descriptive: BM df and the Kish effective number of clusters. The
  frozen rule (outside +-(5 + B_c) in >= 2 cameras, both runs) is applied to (a) and, separately, to (b). The published
  CR1 intervals are reproduced first; a mismatch stops the run.
- R4-2 (R17): K3's copying threshold rho >= (lo_frame - (5 + B_c)) / tau_seed under the quadrature and summed
  compositions (human boxes, primary run, LiDAR-time and static-point seeds); binding value = second largest camera.
- R4-3 (R29): (lo_start - B_c) x 10 m/s for AV2 side left and rear left, from the planned intervals.

Usage (repository root, env loaded): PYTHONPATH=src python -m careport.xfc7_t2_posthoc_r4 <cycle_dir> <out.json>
"""
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from careport.xfc7_decision_v2 import combine_runs, decide_rejection
from careport.xfc7_estimators import _codes, weighted_mean_cr1
from careport.xfc7_primary import CAM, HYP, add_tau_hat
from careport.xfc7_t2_posthoc import composition_budgets
from careport.xfc7_t2_run import _reweight, add_theta_hat
from careport.xfc7_t2_static import CAM_T2, HYP_T2, t_crit

MARGIN_MS = 5.0
SEED = 20261103
DRAWS = 99999
REPRO_TOL_MS = 0.01
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])


# ---------- R4-1 (a): CR2 with Bell-McCaffrey df ----------
def _shares(w, cluster):
    w = np.asarray(w, dtype=float)
    codes, G = _codes(cluster)
    Sg = np.bincount(codes, weights=w, minlength=G)
    return codes, G, Sg, Sg / Sg.sum()


def kish_effective_clusters(w, cluster) -> float:
    _, _, _, a = _shares(w, cluster)
    return float(1.0 / (a * a).sum())


def cr2_bm_weighted_mean(y, w, cluster) -> dict:
    """Weighted mean with a CR2 standard error and Bell-McCaffrey df, working variance proportional to 1/w.
    For the intercept-only WLS fit, V = sum_g s_g^2 / (S^2 (1 - a_g)); the df come from the G x G matrix
    M_st = (delta_st a_s - a_s a_t) / sqrt((1 - a_s)(1 - a_t)) as (tr M)^2 / tr(M^2)."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    codes, G, Sg, a = _shares(w, cluster)
    S = float(w.sum())
    mu = float((w * y).sum() / S)
    if G < 2 or np.any(a >= 1.0):
        nan = float("nan")
        return {"est": mu, "se": nan, "df": nan, "t": nan, "lo": nan, "hi": nan, "G": int(G), "kish_G": float(1.0 / (a * a).sum())}
    s = np.bincount(codes, weights=w * (y - mu), minlength=G)
    se = float(math.sqrt(float((s * s / (1.0 - a)).sum())) / S)
    r = 1.0 / np.sqrt(1.0 - a)
    M = (np.diag(a) - np.outer(a, a)) * np.outer(r, r)
    df = float(np.trace(M) ** 2 / np.trace(M @ M))
    t = float(stats.t.ppf(0.975, df))
    return {"est": mu, "se": se, "df": df, "t": t, "lo": mu - t * se, "hi": mu + t * se, "G": int(G),
            "kish_G": float(1.0 / (a * a).sum())}


# ---------- R4-1 (b): wild cluster bootstrap-t ----------
def webb_weights(rng, shape) -> np.ndarray:
    return WEBB[rng.integers(0, 6, size=shape)]


def wild_cluster_bootstrap_t(y, w, cluster, draws=DRAWS, seed=SEED, chunk=10000) -> dict:
    """Percentile-t interval for the weighted mean: unrestricted residuals, Webb six-point cluster weights,
    CR1 studentization in every draw (y* = mu + v_g e_i)."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    codes, G, Sg, a = _shares(w, cluster)
    S = float(w.sum())
    mu = float((w * y).sum() / S)
    s = np.bincount(codes, weights=w * (y - mu), minlength=G)
    se = float(math.sqrt((G / (G - 1)) * float((s * s).sum())) / S)
    rng = np.random.default_rng(seed)
    ts = []
    left = int(draws)
    while left > 0:
        k = min(chunk, left)
        V = webb_weights(rng, (k, G))
        num = V @ s
        s_star = V * s[None, :] - a[None, :] * num[:, None]
        se_star = np.sqrt((G / (G - 1)) * (s_star ** 2).sum(axis=1)) / S
        ts.append((num / S) / se_star)
        left -= k
    t = np.concatenate(ts)
    q_lo, q_hi = (float(x) for x in np.quantile(t, [0.025, 0.975]))
    return {"est": mu, "se_cr1": se, "lo": mu - q_hi * se, "hi": mu - q_lo * se, "t_quantiles": [q_lo, q_hi],
            "draws": int(draws), "seed": int(seed), "weights": "webb6", "G": int(G)}


# ---------- decisions ----------
def decide(intervals, budgets, cams, margin=MARGIN_MS) -> dict:
    """intervals: hypothesis -> camera -> (est, lo, hi); the frozen additive rule."""
    return decide_rejection(intervals, budgets, [c for c in cams if any(c in v for v in intervals.values())], margin=margin)


# ---------- R4-2 and R4-3 ----------
def copying_threshold(lo: float, tol: float, tau_seed: float) -> float:
    return float((lo - tol) / tau_seed)


def binding_value(values, k=2) -> float:
    return float(sorted(values, reverse=True)[k - 1])


def tolerance_adjusted_misplacement_m(lo_ms: float, b_c_ms: float, speed_mps: float) -> float:
    return float((lo_ms - b_c_ms) * speed_mps / 1000.0)


# ---------- runner ----------
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


PINNED = {
    "av2_moving_set": ("$AV2/derived/t2_static/all50_s050/moving_analysis_set_t2.parquet", "cdccdbe561e83209eafb9cf1b23468b80aba5c73a6aa913c6f76fca6f4e97d65"),
    "wod_human_moving_set": ("$WOD/derived/g5/moving_analysis_set.parquet", "54016253c11a29c367dc0c2590fcbdf793d380e6a6529cafcb98e2ef6bbff32f"),
    "wod_detector_moving_set": ("$WOD/derived/ext_e0/s050/moving_analysis_set_det.parquet", "2ca109db8a63d3ad7a43b9bb05dbae3babc44cfe259cbf6ff75425b5e3c1ff54"),
    "s3_summary": ("tier2_2026-11-01/S3_summary_all50_s050.json", "23c05c24949e1ae6858f5b557cfcb4eab071e293208926236edfb483a43317d2"),
    "e0_summary": ("extension_2026-10-31/E0_summary_s050.json", "12238bd66cdb9f231c781142b1ef5b9a912b7d144fb3d40825668393e8d89ca1"),
    "g5_summary": ("pilot/g5_summary.json", "5b5deac23258455aa10de9a3695d3892d6c57716745916a52f39a634727f9ea8"),
    "primary_results_v2": ("pilot/primary_results_v2.json", "84614460cb374e26e765ecf75a4711755cb38c6786f448ef33905834ff5f5f1e"),
    "tier1_results": ("extension_2026-10-31/tier1_results_2026-11-01.json", "634709cb57cef942230037a24ecc1c2381b27925cb66467d6d05041c88e98f25"),
    "t2_results": ("tier2_2026-11-01/T2_RESULTS_2026-11-01.json", "6faa981f71330b4d719140e93eac348bc0eecb28e494abdf5b1921adb504b6e3"),
    "revision2_v2": ("pilot/revision2_v2.json", "f327e66239e28f1094b039ac85cfb8046b3be206081abd722202373e71a98b26"),
}
PIN_RECORD_SHA = "3e67c9fe9d7d7620e01c26d42cdb067b6a9d7b3547aa2f60f652ae0295fe81a9"


def _cells(frame, names, hyp, ycol, cams):
    """camera name -> hypothesis -> (y, w, cluster) for the cameras in `cams`."""
    out = {}
    for c, g in frame.groupby("camera"):
        name = names.get(int(c))
        if name not in cams:
            continue
        out[name] = {h: (g[ycol].to_numpy(float) - g[col].to_numpy(float), g["w"].to_numpy(float), g["segment"].to_numpy())
                     for h, col in hyp.items()}
    return out


def _arm(runs, cams, budgets, hyp, crit, stored, label):
    """runs: run tag -> cells. crit: 'z' (1.96, WOD) or 't' (t with G - 1 df, AV2). stored: run tag -> camera ->
    hypothesis -> {est, lo, hi} or None (verdicts checked instead). Returns per-cell results and the three decisions."""
    res, iv = {}, {k: {} for k in ("cr1", "cr2_bm", "bootstrap")}
    for tag, cells in runs.items():
        res[tag] = {}
        for k in iv:
            iv[k][tag] = {h: {} for h in hyp}
        for cam, by_h in cells.items():
            res[tag][cam] = {}
            for h, (y, w, cl) in by_h.items():
                est, se = weighted_mean_cr1(y, w, cl)
                G = int(pd.Series(cl).nunique())
                q = 1.96 if crit == "z" else t_crit(G)
                cr1 = {"est": float(est), "se": float(se), "lo": float(est - q * se), "hi": float(est + q * se), "crit": float(q), "G": G}
                ref = (stored.get(tag) or {}).get(cam, {}).get(h) if stored.get(tag) is not None else None
                if ref is not None:
                    for key in ("est", "lo", "hi"):
                        if abs(cr1[key] - float(ref[key])) > REPRO_TOL_MS:
                            raise SystemExit(f"{label} {tag} {cam} {h} {key}: {cr1[key]:.4f} != stored {float(ref[key]):.4f}; stop")
                a = cr2_bm_weighted_mean(y, w, cl)
                b = wild_cluster_bootstrap_t(y, w, cl)
                res[tag][cam][h] = {"cr1": cr1, "cr2_bm": a, "bootstrap": b}
                iv["cr1"][tag][h][cam] = (cr1["est"], cr1["lo"], cr1["hi"])
                iv["cr2_bm"][tag][h][cam] = (a["est"], a["lo"], a["hi"])
                iv["bootstrap"][tag][h][cam] = (b["est"], b["lo"], b["hi"])
    decisions = {}
    for k, by_run in iv.items():
        per_run = {tag: decide(by_run[tag], budgets, cams) for tag in by_run}
        decisions[k] = {"runs": per_run, "final": combine_runs(per_run["primary"], per_run["margin_flag_excluded"])}
    return res, decisions


def _margin_subset(mo):
    sub = mo[(mo["margin"] >= 0.2) | np.isinf(mo["margin"])].copy()
    sub["n_track"] = sub.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    sub["w"] = sub["sdot"] ** 2 / (sub["sigma_b"] ** 2 * sub["n_track"])
    return sub


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 64
    cycle, out_path = argv[1], argv[2]
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; written once; stop")
    roots = {"$AV2": os.environ["AV2_SENSOR_ROOT"], "$WOD": os.environ["WAYMO_V2_ROOT"]}
    paths = {}
    for k, (p, _) in PINNED.items():
        root = next((r for t, r in roots.items() if p.startswith(t)), None)
        paths[k] = os.path.join(root, p.split("/", 1)[1]) if root else os.path.join(cycle, p)
    for k, (_, h) in PINNED.items():
        got = sha(paths[k])
        if got != h:
            raise SystemExit(f"{k}: hash {got} != pinned {h}; stop")
    load = lambda k: json.load(open(paths[k], encoding="utf-8"))
    s3, e0, g5, pv2, t1, t2, rv2 = (load(k) for k in ("s3_summary", "e0_summary", "g5_summary", "primary_results_v2",
                                                      "tier1_results", "t2_results", "revision2_v2"))
    out = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "pin": "pilot/PILOT_RECORD_XF-C7.md, entry '2026-11-03, post hoc set R4' (record SHA-256 after the pin: " + PIN_RECORD_SHA + ")",
           "inputs_sha256": {k: h for k, (_, h) in PINNED.items()}, "status": "post hoc; no frozen verdict or published interval changes",
           "settings": {"bootstrap_draws": DRAWS, "seed": SEED, "bootstrap_weights": "webb6", "repro_tolerance_ms": REPRO_TOL_MS,
                        "cr2_working_variance": "proportional to 1/w", "margin_ms": MARGIN_MS}}

    # ---- AV2 ----
    decision = s3["decision_cameras"]
    av2_budgets = {c: float(s3["budget"][c]["B_c_ms"]) for c in decision}
    mo_all = pd.read_parquet(paths["av2_moving_set"])
    mo = add_theta_hat(mo_all[mo_all["camera"].map(lambda c: CAM_T2.get(int(c)) in decision)])
    margin = add_theta_hat(_reweight(mo_all[mo_all["camera"].map(lambda c: CAM_T2.get(int(c)) in decision)][
        lambda d: (d["margin"] >= 0.2) | np.isinf(d["margin"])]))
    v = t2["primary"]["verdicts"]
    av2_runs = {"primary": _cells(mo, CAM_T2, HYP_T2, "theta_hat", decision),
                "margin_flag_excluded": _cells(margin, CAM_T2, HYP_T2, "theta_hat", decision)}
    av2_stored = {"primary": v["estimates"], "margin_flag_excluded": v["estimates_margin_flag_excluded"]}
    av2_res, av2_dec = _arm(av2_runs, decision, av2_budgets, HYP_T2, "t", av2_stored, "AV2")
    if av2_dec["cr1"]["final"] != v["final"]:
        raise SystemExit("AV2: reproduced CR1 verdicts differ from the frozen ones; stop")
    out["R4_1_AV2"] = {"cameras": decision, "budgets_ms": av2_budgets, "cells": av2_res, "decisions": av2_dec}

    # ---- WOD human boxes ----
    h_cams = pv2["cameras_in_analysis"]
    h_budgets = {c: float(pv2["budgets_ms"][c]) for c in h_cams}
    mo_h = pd.read_parquet(paths["wod_human_moving_set"])
    wod_h_runs = {"primary": _cells(add_tau_hat(mo_h), CAM, HYP, "tau_hat", h_cams),
                  "margin_flag_excluded": _cells(add_tau_hat(_margin_subset(mo_h)), CAM, HYP, "tau_hat", h_cams)}
    h_stored = {"primary": {c: pv2["estimates"][c]["delta"] for c in h_cams},
                "margin_flag_excluded": pv2["robustness"]["margin_flag_excluded"]["delta"]}
    h_res, h_dec = _arm(wod_h_runs, h_cams, h_budgets, HYP, "z", h_stored, "WOD human")
    if h_dec["cr1"]["final"] != pv2["final_verdicts"]:
        raise SystemExit("WOD human: reproduced CR1 verdicts differ from the frozen ones; stop")
    out["R4_1_WOD_human"] = {"cameras": h_cams, "budgets_ms": h_budgets, "cells": h_res, "decisions": h_dec}

    # ---- WOD detector ----
    e1 = t1["primary"]["E1"]
    d_cams = e1["cameras_in_analysis"]
    d_budgets = {c: float(e1["budgets_ms"][c]) for c in d_cams}
    mo_d = pd.read_parquet(paths["wod_detector_moving_set"])
    wod_d_runs = {"primary": _cells(add_tau_hat(mo_d), CAM, HYP, "tau_hat", d_cams),
                  "margin_flag_excluded": _cells(add_tau_hat(_margin_subset(mo_d)), CAM, HYP, "tau_hat", d_cams)}
    d_stored = {"primary": {c: e1["estimates"][c]["delta"] for c in d_cams}, "margin_flag_excluded": None}
    d_res, d_dec = _arm(wod_d_runs, d_cams, d_budgets, HYP, "z", d_stored, "WOD detector")
    for h in HYP:
        if d_dec["cr1"]["runs"]["margin_flag_excluded"][h]["rejected_in"] != e1["verdicts_margin_flag_excluded"][h]["rejected_in"]:
            raise SystemExit(f"WOD detector margin run, {h}: reproduced rejections differ from the stored ones; stop")
    if d_dec["cr1"]["final"] != e1["final_verdicts"]:
        raise SystemExit("WOD detector: reproduced CR1 verdicts differ from the frozen ones; stop")
    out["R4_1_WOD_detector"] = {"cameras": d_cams, "budgets_ms": d_budgets, "cells": d_res, "decisions": d_dec}

    # ---- R4-2 ----
    k3 = rv2["K3_retention_bounds"]["human"]
    comps = {c: {k: float(g5["budget"][c][k]) for k in ("neighbour_bins_ms", "ego_class_spread_ms", "ego_speed_slope_ms")} for c in k3}
    b = composition_budgets(comps)
    r42 = {"cameras": {}, "binding": {}}
    for c, row in k3.items():
        if abs(MARGIN_MS + b[c]["max"] - float(row["tolerance_ms"])) > 1e-6:
            raise SystemExit(f"R4-2 {c}: 5 + B_max does not reproduce K3's tolerance; stop")
        for seed, tau in (("lidar_seed", row["tau_bar_lidar_ms"]), ("static_point_seed", row["tau_bar_cap_ms"])):
            if abs(copying_threshold(row["lo_frame_ms"], MARGIN_MS + b[c]["max"], tau) - float(row[seed]["tolerance"])) > 1e-9:
                raise SystemExit(f"R4-2 {c} {seed}: the max composition does not reproduce K3; stop")
        r42["cameras"][c] = {comp: {"tolerance_ms": MARGIN_MS + b[c][comp],
                                    "lidar_seed": copying_threshold(row["lo_frame_ms"], MARGIN_MS + b[c][comp], row["tau_bar_lidar_ms"]),
                                    "static_point_seed": copying_threshold(row["lo_frame_ms"], MARGIN_MS + b[c][comp], row["tau_bar_cap_ms"])}
                             for comp in ("max", "rss", "sum")}
    for comp in ("max", "rss", "sum"):
        r42["binding"][comp] = {seed: binding_value([r42["cameras"][c][comp][seed] for c in k3]) for seed in ("lidar_seed", "static_point_seed")}
    out["R4_2_copying_threshold"] = r42

    # ---- R4-3 ----
    r43 = {}
    for cam in ("ring_side_left", "ring_rear_left"):
        lo = float(v["estimates"][cam]["start"]["lo"])
        r43[cam] = {"lo_start_ms": lo, "B_c_ms": av2_budgets[cam], "misplacement_m_at_10_mps": tolerance_adjusted_misplacement_m(lo, av2_budgets[cam], 10.0)}
    out["R4_3_tolerance_adjusted_misplacement"] = r43

    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print("written:", out_path, "sha256", sha(out_path))
    for label, dec in (("AV2", av2_dec), ("WOD human", h_dec), ("WOD detector", d_dec)):
        for k in ("cr1", "cr2_bm", "bootstrap"):
            print(f"{label:13s} {k:9s} final={dec[k]['final']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
