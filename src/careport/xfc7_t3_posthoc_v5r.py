"""XF-C7 post hoc set V5R: v5 review items R10, R11, R13, R14 and R15 on the T3 primary sample.

The plan is the pilot record entry '2026-11-05, post hoc set V5R', pinned before anything was computed; the record's
SHA-256 just after the pin is PIN_RECORD_SHA. Everything here is post hoc. No frozen verdict and no published interval
changes. The T3 runner's functions are reused unchanged (xfc7_t3_run.py 0bf9c798...).

- V5R-1 (R11; R10 described): the pooled slope of theta-hat on 1/sdot (reproduced), the within-camera slope (one
  intercept per decision camera, CR1 by log, t(G - 1)), which governs, per-camera slopes and intercepts, camera weight
  shares and the implied mean bias per camera.
- V5R-2 (R10): theta-hat' = theta-hat - beta / sdot with beta fixed (W-pt governs; W-up, P-pt, P-up), then the frozen
  three-method rule in both runs, planned budgets, and re-judged under the quadrature and summed compositions.
  Bias model: theta-hat = theta + b / sdot for a constant along-motion pixel offset b that z_b does not remove; the same
  slope also absorbs any real dependence of theta on image speed, so this is a sensitivity analysis, not a correction.
- V5R-3 (R13): the midpass convention (theta_A + theta_B) / 2, three-method rule, with its decidability.
- V5R-4 (R15): the shift toward the sweep start that leaves fewer than two rejecting cameras, from stored intervals.
- V5R-5 (R14): the residual of each frame-time mode from its LiDAR crossing (G3), S0's departure and AV2's documented
  synchronization accuracy, giving a bound on a constant image-timestamp error.

Usage (repository root): PYTHONPATH=src python -m careport.xfc7_t3_posthoc_v5r <cycle_dir> <moving_set.parquet> <out.json>
"""
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from careport.xfc7_estimators import wls_slope_cr1
from careport.xfc7_t2_posthoc import composition_budgets
from careport.xfc7_t2_posthoc_r4 import DRAWS, SEED
from careport.xfc7_t2_run import _reweight, add_theta_hat, slope_ci_t
from careport.xfc7_t2_static import CAM_T2, HYP_T2, capacity, t_crit
from careport.xfc7_t3_run import MARGIN_MS, METHODS, _rejudge, cells, theta_bar, verdicts_three

PIN_RECORD_SHA = "3c8cc61705f493edeed10da10918e5301003727a23e42bafdeb0f4a40799d0b4"
AV2_SYNC_MS = 1.39          # Wilson et al., arXiv:2301.00493: "Our synchronization accuracy is within [-1.39, 1.39] ms"
PERIOD_MS = 100.0
RUN_TAGS = ("primary", "margin_flag_excluded")
T3 = "t3_replication_2026-11-04"
INPUTS = {   # path relative to the cycle folder (the moving set is passed in): pinned SHA-256
    "moving_set": "b6df2c96d5142ac83f33c88ffacce4f97ba8353056d5bd730404920a6d48ec40",
    f"{T3}/cluster_results/derived/t3_static/S3_summary_s050.json":
        "e8f9229a76e28a726cfc387833912c7c32bd626bb9419c177ee9e6d2c994df3c",
    f"{T3}/cluster_results/results/T3_RESULTS_2026-11-05.json":
        "e28f472dca12f9438bc9d7fe7bc4cd95d3cbe0ff488a1dcfd388b8ae94d6f3dc",
    "tier2_2026-11-01/g3_av2_timing_p01.json": "7bc6bf53d3a2c45dcf3a2e27655cbe4ac02cd1c87f6481c96968bd94d60a838e",
    f"{T3}/cluster_results/derived/s0_t3.json": "c1f4811cf6285de9164f84ab57cec36601ddb587db7d73a4f4d11b7b19464ddd",
}
MIDPASS_CONTRASTS = [("start", "midpass"), ("A", "midpass"), ("B", "midpass"), ("mid", "midpass")]


# ---------- V5R-1 ----------
def within_slope(frame) -> dict:
    """WLS of theta-hat on 1/sdot with one intercept per camera; CR1 by log; t(G - 1)."""
    b, se = wls_slope_cr1(frame["theta_hat"].to_numpy(float), 1.0 / frame["sdot"].to_numpy(float),
                          frame["w"].to_numpy(float), frame["camera"].to_numpy(), frame["segment"].to_numpy())
    G = int(frame["segment"].nunique())
    t = t_crit(G)
    return {"slope": float(b), "se": float(se), "lo": float(b - t * se), "hi": float(b + t * se), "G": G}


def camera_shares(frame, names) -> dict:
    tot = float(frame["w"].sum())
    return {names[int(c)]: float(g["w"].sum()) / tot for c, g in frame.groupby("camera")}


def implied_bias(frame, names, beta) -> dict:
    """beta times the weighted mean of 1/sdot, per camera (ms)."""
    return {names[int(c)]: float(beta * (g["w"] / g["sdot"]).sum() / g["w"].sum()) for c, g in frame.groupby("camera")}


# ---------- V5R-2 and V5R-3 ----------
def adjust(frame, beta):
    out = frame.copy()
    out["theta_hat"] = out["theta_hat"] - beta / out["sdot"]
    return out


def add_midpass(frame):
    ok = frame["theta_A"].notna() & frame["theta_B"].notna()
    out = frame[ok].copy()
    out["theta_midpass"] = (out["theta_A"] + out["theta_B"]) / 2.0
    return out, int((~ok).sum())


def midpass_decidable(th, thr, contrasts=MIDPASS_CONTRASTS) -> dict:
    """th: run -> camera -> hypothesis -> theta-bar; thr: run -> camera -> 5 + B_c + hw_BM. A contrast is decidable if
    at least two cameras clear its threshold in every run (T3 section 9's rule)."""
    keys = [f"{a}-{b}" for a, b in contrasts]
    runs, cams = {}, {}
    for tag in th:
        cap = capacity({c: th[tag][c] for c in th[tag] if c in thr[tag]}, thr[tag], contrasts)
        cams[tag] = cap["cameras"]
        runs[tag] = {k: sum(bool(cap["cameras"][c][k]["decidable"]) for c in cap["cameras"]) >= 2 for k in keys}
    return {"runs": runs, "decidable": {k: all(r[k] for r in runs.values()) for k in keys}, "cameras": cams}


# ---------- V5R-4 ----------
def shift_threshold(iv, budgets, methods=METHODS, margin=MARGIN_MS) -> float:
    """iv: run -> camera -> {'start': {method: (est, lo, hi)}}. Per camera the clearance lo - (margin + B_c), smallest over
    `methods`; per run the second-largest clearance over cameras; the smaller over runs."""
    per_run = []
    for by_cam in iv.values():
        cl = sorted((min(by_cam[c]["start"][m][1] for m in methods) - (margin + budgets[c]) for c in by_cam), reverse=True)
        per_run.append(cl[1] if len(cl) >= 2 else float("nan"))
    return float(min(per_run))


# ---------- V5R-5 ----------
def wrap_ms(x, period=PERIOD_MS) -> float:
    return float((x + period / 2) % period - period / 2)


def crossing_offset(profile, az, period=PERIOD_MS) -> float:
    """The median point offset at azimuth `az`, interpolated linearly between 10-degree bin centres, circular in azimuth,
    unwrapping the time difference between the two bracketing centres to within half a period."""
    pts = sorted(((float(k.split("..")[0]) + float(k.split("..")[1])) / 2.0, float(val)) for k, val in profile.items())
    cs = [c for c, _ in pts]
    vs = [x for _, x in pts]
    cs_ext = [cs[-1] - 360.0] + cs + [cs[0] + 360.0]
    vs_ext = [vs[-1]] + vs + [vs[0]]
    a = ((az + 180.0) % 360.0) - 180.0
    for i in range(len(cs_ext) - 1):
        c0, c1 = cs_ext[i], cs_ext[i + 1]
        if c0 <= a <= c1:
            v0 = vs_ext[i]
            dv = wrap_ms(vs_ext[i + 1] - v0, period)
            return float((v0 + dv * (a - c0) / (c1 - c0)) % period)
    raise ValueError(f"azimuth {az} not bracketed")


def crossing_residuals(g3) -> dict:
    """Per ring camera and LiDAR (A: laser_number < 32, B: >= 32), the observed frame-offset mode nearest the
    predicted crossing (mod 100 ms) minus that crossing, wrapped to [-50, 50) ms; r_max over all pairs."""
    pairs = {}
    for cam, info in g3["cameras"].items():
        az = float(info["azimuth_deg_median"])
        modes = [float(m) for m in g3["frame_offsets_ms"][cam]["modes_ms"]]
        pairs[cam] = {}
        for lid, key in (("A", "lower_laser_numbers"), ("B", "upper_laser_numbers")):
            pred = crossing_offset(g3["lidar_profile"][key], az)
            res = min((wrap_ms(m - pred) for m in modes), key=abs)
            pairs[cam][lid] = {"azimuth_deg": az, "predicted_crossing_ms": pred, "residual_ms": float(res)}
    r = max(abs(p["residual_ms"]) for by in pairs.values() for p in by.values())
    return {"pairs": pairs, "r_max_ms": float(r)}


# ---------- the run ----------
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _triples(three, hyps=None):
    """verdicts_three intervals -> run -> camera -> hypothesis -> method -> (est, lo, hi)."""
    out = {}
    for tag, by_cam in three["intervals"].items():
        out[tag] = {c: {h: {m: (float(ints[m]["est"]), float(ints[m]["lo"]), float(ints[m]["hi"])) for m in METHODS}
                        for h, ints in by_h.items() if hyps is None or h in hyps} for c, by_h in by_cam.items()}
    return out


def _verdict_summary(three, comps_live):
    """Final verdicts and per-run rejecting cameras, planned budgets and each composition."""
    out = {"planned": {"final": three["three"]["final"],
                       "rejected_in": {tag: {h: v["rejected_in"] for h, v in r.items()} for tag, r in three["three"]["runs"].items()}}}
    for name, b in comps_live.items():
        rj = _rejudge(three, b, list(b))
        out[name] = {"final": rj["final"], "rejected_in": {tag: {h: v["rejected_in"] for h, v in r.items()} for tag, r in rj["runs"].items()}}
    return out


def run(cycle, moving_path) -> dict:
    paths = {k: (moving_path if k == "moving_set" else os.path.join(cycle, k)) for k in INPUTS}
    for k, p in paths.items():
        got = sha(p)
        if got != INPUTS[k]:
            raise SystemExit(f"{k}: SHA-256 {got} differs from the pin {INPUTS[k]}; stop")
    load = lambda k: json.load(open(paths[k], encoding="utf-8"))
    s3 = load(f"{T3}/cluster_results/derived/t3_static/S3_summary_s050.json")
    stored = load(f"{T3}/cluster_results/results/T3_RESULTS_2026-11-05.json")["primary"]
    g3 = load("tier2_2026-11-01/g3_av2_timing_p01.json")
    s0 = load(f"{T3}/cluster_results/derived/s0_t3.json")
    mo = pd.read_parquet(paths["moving_set"])

    cams = list(s3["decision_cameras"])
    budgets = {c: float(s3["budget"][c]["B_c_ms"]) for c in cams}
    fail = list(stored["t2_block"]["timing_model_failures"])
    live = [c for c in cams if c not in fail]
    in_live = lambda d: d[d["camera"].map(lambda c: CAM_T2.get(int(c)) in live)]
    dm = add_theta_hat(in_live(mo))
    excl = add_theta_hat(_reweight(in_live(mo)[lambda d: (d["margin"] >= 0.2) | np.isinf(d["margin"])]))
    frames = {"primary": dm, "margin_flag_excluded": excl}
    b_live = {c: budgets[c] for c in live}
    comps = composition_budgets({c: s3["budget"][c] for c in cams})
    comps_live = {name: {c: comps[c][name] for c in live} for name in ("rss", "sum")}
    out = {"pin": "pilot/PILOT_RECORD_XF-C7.md, entry '2026-11-05, post hoc set V5R' (record SHA-256 after the pin: "
                  + PIN_RECORD_SHA + ")", "inputs_sha256": dict(INPUTS), "cameras": live, "budgets_ms": b_live,
           "settings": {"margin_ms": MARGIN_MS, "draws": DRAWS, "seed": SEED, "methods": list(METHODS)}}

    # reproduction gate: every stored three-method interval within 0.01 ms, the final verdicts, the pooled 1/sdot slope
    three0 = verdicts_three({tag: cells(f, CAM_T2, HYP_T2, live) for tag, f in frames.items()}, b_live, live)
    st_iv = stored["three_method"]["intervals"]
    worst = 0.0
    for tag in RUN_TAGS:
        for c in live:
            for h in HYP_T2:
                for m in METHODS:
                    for key in ("lo", "hi", "est"):
                        d = abs(float(three0["intervals"][tag][c][h][m][key]) - float(st_iv[tag][c][h][m][key]))
                        worst = max(worst, d)
                        if not d <= 0.01:
                            raise SystemExit(f"interval {tag} {c} {h} {m} {key} not reproduced ({d}); stop")
    if three0["three"]["final"] != stored["three_method"]["three"]["final"]:
        raise SystemExit("three-method final verdicts not reproduced; stop")
    ok = _reweight(dm) if fail else dm
    pooled = slope_ci_t(ok["theta_hat"], 1.0 / ok["sdot"], ok["w"], ok["segment"])
    st_sl = stored["t2_block"]["C3_inv_sdot"]
    for k in ("slope", "slope_lo", "slope_hi"):
        if abs(pooled[k] - float(st_sl[k])) > 0.01:
            raise SystemExit(f"pooled 1/sdot {k} not reproduced; stop")
    out["reproduction"] = {"intervals_max_abs_diff_ms": worst, "final_verdicts_equal": True,
                           "pooled_slope": pooled, "stored_pooled_slope": st_sl}

    # V5R-1
    within = within_slope(ok)
    per_cam = {CAM_T2[int(c)]: slope_ci_t(g["theta_hat"], 1.0 / g["sdot"], g["w"], g["segment"])
               for c, g in ok.groupby("camera") if g["segment"].nunique() >= 2}
    th_rows = {tag: {CAM_T2[int(c)]: theta_bar(g, HYP_T2) for c, g in f.groupby("camera")} for tag, f in frames.items()}
    alpha_vs = {}
    for c, r in per_cam.items():
        band = MARGIN_MS + budgets[c]
        alpha_vs[c] = {h: {"lo": r["intercept_lo"] - th_rows["primary"][c][h], "hi": r["intercept_hi"] - th_rows["primary"][c][h],
                           "outside_band": bool(r["intercept_lo"] - th_rows["primary"][c][h] > band
                                                or r["intercept_hi"] - th_rows["primary"][c][h] < -band)} for h in HYP_T2}
    out["V5R_1"] = {"pooled": pooled, "within_camera": within, "per_camera": per_cam,
                    "per_camera_intercept_vs_conventions_primary": alpha_vs,
                    "weight_shares": {tag: camera_shares(f, CAM_T2) for tag, f in frames.items()},
                    "implied_bias_ms": {"within": {tag: implied_bias(f, CAM_T2, within["slope"]) for tag, f in frames.items()},
                                        "pooled": {tag: implied_bias(f, CAM_T2, pooled["slope"]) for tag, f in frames.items()}}}

    # V5R-2
    betas = {"W-pt": within["slope"], "W-up": within["hi"], "P-pt": pooled["slope"], "P-up": pooled["slope_hi"]}
    out["V5R_2"] = {"betas_px": betas, "governs": "W-pt", "analyses": {}}
    for name, beta in betas.items():
        three = verdicts_three({tag: cells(adjust(f, beta), CAM_T2, HYP_T2, live) for tag, f in frames.items()}, b_live, live)
        out["V5R_2"]["analyses"][name] = {"verdicts": _verdict_summary(three, comps_live), "intervals": _triples(three)}

    # V5R-3
    mp, dropped = {}, {}
    for tag, f in frames.items():
        mp[tag], dropped[tag] = add_midpass(f)
    hyp_mp = {"midpass": "theta_midpass"}
    three_mp = verdicts_three({tag: cells(f, CAM_T2, hyp_mp, live) for tag, f in mp.items()}, b_live, live)
    three_mp_adj = verdicts_three({tag: cells(adjust(f, within["slope"]), CAM_T2, hyp_mp, live) for tag, f in mp.items()},
                                  b_live, live)
    hyp_all = dict(HYP_T2, midpass="theta_midpass")
    th = {tag: {CAM_T2[int(c)]: theta_bar(g, hyp_all) for c, g in f.groupby("camera") if CAM_T2[int(c)] in live}
          for tag, f in mp.items()}
    thr = {tag: {c: float(stored["gate_bm"][tag]["thresholds_ms"][c]) for c in live} for tag in RUN_TAGS}
    out["V5R_3"] = {"rows_dropped": dropped, "rows": {tag: int(len(f)) for tag, f in mp.items()},
                    "theta_hat": {"verdicts": _verdict_summary(three_mp, comps_live), "intervals": _triples(three_mp)},
                    "theta_hat_W_pt": {"verdicts": _verdict_summary(three_mp_adj, comps_live), "intervals": _triples(three_mp_adj)},
                    "theta_bar": th, "decidability": midpass_decidable(th, thr)}

    # V5R-4, from the stored intervals
    iv = {tag: {c: {"start": {m: (float(st_iv[tag][c]["start"][m]["est"]), float(st_iv[tag][c]["start"][m]["lo"]),
                                  float(st_iv[tag][c]["start"][m]["hi"])) for m in METHODS}} for c in live} for tag in RUN_TAGS}
    out["V5R_4"] = {"three_method_ms": shift_threshold(iv, b_live, METHODS), "cr1_ms": shift_threshold(iv, b_live, ("cr1",))}

    # V5R-5
    cr = crossing_residuals(g3)
    s0_dep = float(s0["max_abs_departure_ms"])
    out["V5R_5"] = {"crossing": cr, "s0_max_abs_departure_ms": s0_dep, "s0_departing_logs": len(s0["departing_logs"]),
                    "av2_documented_sync_ms": AV2_SYNC_MS, "bound_ms": cr["r_max_ms"] + s0_dep + AV2_SYNC_MS,
                    "conditions": ["AV2's documented exposure centring and synchronization accuracy (arXiv:2301.00493)",
                                   "T3 logs share G3's LiDAR spin profile (10 T2 logs)", "LiDAR point times are correct"]}
    return out


def main(argv):
    if len(argv) != 4:
        print(__doc__)
        return 64
    cycle, moving, out_path = argv[1:4]
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; the run is done once and never overwritten; stop")
    res = run(cycle, moving)
    res["run_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print("results:", out_path, "sha256", sha(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
