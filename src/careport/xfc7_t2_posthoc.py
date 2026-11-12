"""XF-C7 tier 2, post hoc set V1-V5 for the v4 revision (pilot record entry of 2026-11-02, pinned before computing).

All post hoc; none changes a frozen verdict. Inputs are the frozen tier-2 (and tier-1) artefacts, hash-checked here.
- V1 tolerance composition: verdicts under B_max (frozen), B_rss and B_sum from the frozen budget components.
- V2 decidability with each camera's realized half-width of delta_start in place of the predicted one.
- V3 joint WLS of theta-hat on speed and 1/sdot with camera fixed effects (CR1 by log, t with G - 1 df).
- V4 two further conventions: the per-box midpoint (theta_A + theta_B)/2 and a point-weighted pooled time.
- V5 checks: seam-excluded rows restored; leave-one-log-out range; theta-hat quantiles; per-frame delta_start;
  velocity scale implied by the two-frame difference.

Usage (repository root, env loaded): PYTHONPATH=src python -m careport.xfc7_t2_posthoc <cycle_dir> <out.json>
"""
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from careport.xfc7_decision_v2 import combine_runs, decide_rejection
from careport.xfc7_estimators import weighted_mean_cr1, wls_cr1
from careport.xfc7_t2_run import _reweight, add_theta_hat, mean_ci_t, verdicts_t2
from careport.xfc7_t2_static import CAM_T2, CONTRASTS, t_crit
from careport.xfc7_zero_line import LEVELS, lookup_zero_line, zero_line_tables

COMPONENTS = ("neighbour_bins_ms", "ego_class_spread_ms", "ego_speed_slope_ms")
MARGIN_MS = 5.0
K4 = ["segment", "frame", "camera", "object_id"]


# ---------- V1 ----------
def composition_budgets(components):
    out = {}
    for c, comp in components.items():
        x = [float(comp[k]) for k in COMPONENTS]
        out[c] = {"max": max(x), "rss": math.sqrt(sum(v * v for v in x)), "sum": sum(x)}
    return out


def _delta(est, cams, hyps):
    return {h: {c: (est[c][h]["est"], est[c][h]["lo"], est[c][h]["hi"]) for c in cams if c in est} for h in hyps}


def verdicts_by_composition(est_primary, est_margin, components, cams, hyps):
    """The frozen two-camera, two-run rule applied to stored intervals under each tolerance composition."""
    b = composition_budgets(components)
    out = {}
    for comp in ("max", "rss", "sum"):
        budgets = {c: b[c][comp] for c in cams}
        runs = {"primary": decide_rejection(_delta(est_primary, cams, hyps), budgets, cams),
                "margin_flag_excluded": decide_rejection(_delta(est_margin, cams, hyps), budgets, cams)}
        out[comp] = {"tolerance_ms": {c: MARGIN_MS + budgets[c] for c in cams}, "runs": runs,
                     "final": combine_runs(runs["primary"], runs["margin_flag_excluded"])}
    return out


def tier1_intervals(e1, cams):
    """The tier-1 E1 block nests each camera's intervals under 'delta'; unwrap to camera -> hypothesis -> interval."""
    return {c: e1["estimates"][c]["delta"] for c in cams}


def rejections_by_composition(est_primary, components, cams, hyps):
    """Primary run only (WOD detector: the margin run's intervals were not stored)."""
    b = composition_budgets(components)
    return {comp: decide_rejection(_delta(est_primary, cams, hyps), {c: b[c][comp] for c in cams}, cams)
            for comp in ("max", "rss", "sum")}


# ---------- V2 ----------
def realized_decidability(capacity_cams, budgets, realized_hw, contrasts=CONTRASTS):
    cams = {}
    for c in budgets:
        cams[c] = {}
        for a, b in contrasts:
            key = f"{a}-{b}"
            gap = float(capacity_cams[c][key]["gap"])
            thr = MARGIN_MS + float(budgets[c]) + float(realized_hw[c])
            cams[c][key] = {"gap": gap, "threshold_realized": thr, "decidable": bool(gap > thr)}
    dataset = {f"{a}-{b}": sum(cams[c][f"{a}-{b}"]["decidable"] for c in cams) >= 2 for a, b in contrasts}
    return {"cameras": cams, "dataset": dataset}


# ---------- V3 ----------
def _wls_block(g, grp):
    X = np.column_stack([g["speed"].to_numpy(float), 1.0 / g["sdot"].to_numpy(float)])
    b, se = wls_cr1(g["theta_hat"].to_numpy(float), X, g["w"].to_numpy(float), grp, g["segment"].to_numpy())
    G = int(g["segment"].nunique())
    t = t_crit(G)
    names = ("speed", "inv_sdot")
    out = {n: {"coef": float(b[i]), "se": float(se[i]), "lo": float(b[i] - t * se[i]), "hi": float(b[i] + t * se[i])}
           for i, n in enumerate(names)}
    out.update({"G": G, "n": int(len(g)), "units": {"speed": "ms per m/s", "inv_sdot": "px"}})
    return out


def joint_speed_model(mo):
    """theta-hat on speed and 1/sdot jointly, camera fixed effects (within-camera demeaning), CR1 by log."""
    return _wls_block(mo, mo["camera"].to_numpy())


def per_camera_speed_models(mo, names):
    return {names[int(c)]: _wls_block(g, np.zeros(len(g))) for c, g in mo.groupby("camera") if g["segment"].nunique() >= 2}


# ---------- V4 ----------
def add_conventions(mo):
    mo = mo.copy()
    a, b = mo["theta_A"].to_numpy(float), mo["theta_B"].to_numpy(float)
    na, nb = mo["n_pts_A"].to_numpy(float), mo["n_pts_B"].to_numpy(float)
    mo["theta_mp"] = (a + b) / 2.0
    tot = na + nb
    with np.errstate(invalid="ignore", divide="ignore"):
        mo["theta_pool"] = np.where(tot > 0, (na * a + nb * b) / tot, (a + b) / 2.0)
    return mo


# ---------- V5 ----------
def seam_only_moving(lab):
    """Rows that meet every moving condition of the T2 label side except the seam rule, and are seam rows."""
    unit = (lab["most_visible"] & (lab["num_pts"] >= 20) & lab["inside_all"] & ~lab["overlap"] & lab["static_proj_ok"]
            & lab["speed"].notna())
    moving = unit & (lab["speed"] >= 3.0) & (lab["sdot"] >= 0.25) & (lab["along_axis_deg"] >= 20.0)
    return lab[moving & lab["seam"]].copy()


def weighted_quantiles(x, w, qs):
    o = np.argsort(x)
    x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    cw = np.cumsum(w) / w.sum()
    return [float(x[min(int(np.searchsorted(cw, q - 1e-12)), len(x) - 1)]) for q in qs]


def loo_range(g, col):
    full = weighted_mean_cr1(g["theta_hat"] - g[col], g["w"], g["segment"])[0]
    reps = []
    for s in g["segment"].unique():
        h = g[g["segment"] != s]
        reps.append(float(np.average(h["theta_hat"] - h[col], weights=h["w"])))
    return {"full": float(full), "min": float(min(reps)), "max": float(max(reps)), "n_logs": len(reps)}


def velocity_scale(tf):
    return {"est": tf["diff_ms"] / 50.0, "lo": tf["lo"] / 50.0, "hi": tf["hi"] / 50.0}


def per_frame_delta(g):
    return {ft: mean_ci_t(_reweight(gf)["theta_hat"], _reweight(gf)["w"], gf["segment"]) for ft, gf in g.groupby("frame_type")}


def seam_restored(mo, lab, assoc, static_all, names, decision):
    """Restore the seam-excluded moving rows that are assigned and get a frozen zero line; recompute start and mid."""
    codes = [c for c, n in names.items() if n in decision]
    extra = seam_only_moving(lab[lab["camera"].isin(codes)])
    extra = extra.merge(assoc[K4 + ["camera_object_id", "u_ref", "v_ref", "w_ref", "cost", "margin"]], on=K4, how="left")
    extra = extra[extra["camera_object_id"].notna()].copy()
    tables = zero_line_tables(static_all)
    extra = extra.join(lookup_zero_line(extra[LEVELS[0]], tables))
    extra = extra[~extra["no_zero_line"] & (extra["sigma_b"] > 0)].copy()
    extra["restored"] = True
    base = mo[mo["camera"].isin(codes)].copy()
    base["restored"] = False
    both = _reweight(pd.concat([base, extra], ignore_index=True, sort=False))
    both = add_theta_hat(both)
    comp = {}
    for c, g in both.groupby("camera"):
        comp[names[int(c)]] = {tag: {"obs": int(len(s)), "tracks": int(s[["segment", "object_id"]].drop_duplicates().shape[0]),
                                     "mean_speed": float(np.average(s["speed"], weights=s["w"])) if len(s) else float("nan"),
                                     "mean_theta_hat": float(np.average(s["theta_hat"], weights=s["w"])) if len(s) else float("nan")}
                               for tag, s in (("retained", g[~g["restored"]]), ("restored", g[g["restored"]]))}
    return both, comp


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


PINNED = {
    "t2_results": ("tier2_2026-11-01/T2_RESULTS_2026-11-01.json", "6faa981f71330b4d719140e93eac348bc0eecb28e494abdf5b1921adb504b6e3"),
    "s3_summary": ("tier2_2026-11-01/S3_summary_all50_s050.json", "23c05c24949e1ae6858f5b557cfcb4eab071e293208926236edfb483a43317d2"),
    "tier1_results": ("extension_2026-10-31/tier1_results_2026-11-01.json", "634709cb57cef942230037a24ecc1c2381b27925cb66467d6d05041c88e98f25"),
    "e0_summary": ("extension_2026-10-31/E0_summary_s050.json", "12238bd66cdb9f231c781142b1ef5b9a912b7d144fb3d40825668393e8d89ca1"),
    "moving_set": ("$AV2/derived/t2_static/all50_s050/moving_analysis_set_t2.parquet", "cdccdbe561e83209eafb9cf1b23468b80aba5c73a6aa913c6f76fca6f4e97d65"),
    "static_all": ("$AV2/derived/t2_static/all50_s050/static_all.parquet", "1752cc82ec34328d76604bc9d5548ecc183b9add10848b481f076b1d6a87cb33"),
    "association": ("$AV2/derived/t2_static/all50_s050/association.parquet", "f8569d4ca675b4ce38593c90cd3e2b36cfd7c14a35b9a08e2916757c14e45830"),
    "label_side": ("$AV2/derived/t2_labels_all50_label_side.parquet", "b4f595ef2818397c4f0f6603da7d12c22c328cdc3bbeebae425a8e4766ae96a7"),
}


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 64
    cycle, out_path = argv[1], argv[2]
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; written once; stop")
    av2 = os.environ["AV2_SENSOR_ROOT"]
    paths = {k: (p.replace("$AV2", av2) if p.startswith("$AV2") else os.path.join(cycle, p)) for k, (p, _) in PINNED.items()}
    for k, (_, h) in PINNED.items():
        got = sha(paths[k])
        if got != h:
            raise SystemExit(f"{k}: hash {got} != pinned {h}; stop")
    load = lambda k: json.load(open(paths[k], encoding="utf-8"))
    t2, s3, t1, e0 = load("t2_results"), load("s3_summary"), load("tier1_results"), load("e0_summary")
    decision = s3["decision_cameras"]
    v = t2["primary"]["verdicts"]
    hyps = ["start", "A", "B", "mid"]
    out = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "pin": "pilot/PILOT_RECORD_XF-C7.md, entry '2026-11-02, post hoc set V1-V5' (pinned 2026-11-02T03:38Z)",
           "inputs_sha256": {k: h for k, (_, h) in PINNED.items()}, "status": "post hoc; no frozen verdict changes"}
    # V1
    comp_av2 = {c: s3["budget"][c] for c in decision}
    v1_av2 = verdicts_by_composition(v["estimates"], v["estimates_margin_flag_excluded"], comp_av2, decision, hyps)
    if v1_av2["max"]["final"] != v["final"]:
        raise SystemExit("V1 under B_max does not reproduce the frozen AV2 verdicts; stop")
    e1 = t1["primary"]["E1"]
    det_cams = e1["cameras_in_analysis"]
    v1_wod = rejections_by_composition(tier1_intervals(e1, det_cams), {c: e0["detector"]["budget"][c] for c in det_cams}, det_cams,
                                       ["frame", "lidar", "cam", "pose"])
    out["V1_tolerance_composition"] = {
        "AV2": {"budgets": composition_budgets(comp_av2), "verdicts": v1_av2},
        "WOD_detector_primary_run_only": {"budgets": composition_budgets({c: e0["detector"]["budget"][c] for c in det_cams}),
                                          "rejections": v1_wod},
        "WOD_human": "quoted from K5 (pilot record, 25 Oct): quadrature ruled out in all six runs; linear sum in four of six "
                     "(not with gate x2 or the frustum-edge exclusion)"}
    # V2
    hw = {c: (v["estimates"][c]["start"]["hi"] - v["estimates"][c]["start"]["lo"]) / 2.0 for c in decision}
    out["V2_realized_decidability"] = dict(realized_decidability(s3["capacity"]["cameras"], {c: s3["budget"][c]["B_c_ms"] for c in decision}, hw),
                                           realized_half_width_ms=hw,
                                           predicted_dataset={f"{a}-{b}": sum(bool(s3["capacity"]["cameras"][c][f"{a}-{b}"]["decidable"])
                                                                              for c in decision) >= 2 for a, b in CONTRASTS})
    # data
    mo_all = pd.read_parquet(paths["moving_set"])
    mo = add_theta_hat(mo_all[mo_all["camera"].map(lambda c: CAM_T2.get(int(c)) in decision)])
    # V3
    out["V3_speed_models"] = {"joint_camera_fixed_effects": joint_speed_model(mo), "per_camera": per_camera_speed_models(mo, CAM_T2)}
    # V4
    mo4 = add_conventions(mo)
    budgets = {c: s3["budget"][c]["B_c_ms"] for c in decision}
    out["V4_conventions"] = verdicts_t2(mo4, CAM_T2, budgets, hyp={"midpoint": "theta_mp", "pooled": "theta_pool"})
    # V5
    lab = pd.read_parquet(paths["label_side"])
    assoc = pd.read_parquet(paths["association"])
    static_all = pd.read_parquet(paths["static_all"])
    both, comp = seam_restored(mo_all, lab, assoc, static_all, CAM_T2, decision)
    v5 = {"a_seam_restored": {"comparison": comp,
                              "verdicts": verdicts_t2(both, CAM_T2, budgets, hyp={"start": "theta_start", "mid": "theta_mid"})}}
    for name in ("ring_rear_left", "ring_side_left"):
        code = [k for k, n in CAM_T2.items() if n == name][0]
        g = mo[mo["camera"] == code]
        v5[name] = {"b_leave_one_log_out_start": loo_range(g, "theta_start"),
                    "c_theta_hat_quantiles_5_25_50_75_95": weighted_quantiles(g["theta_hat"].to_numpy(), g["w"].to_numpy(), [0.05, 0.25, 0.5, 0.75, 0.95]),
                    "d_per_frame_delta_start": per_frame_delta(g),
                    "e_velocity_scale_from_two_frame": velocity_scale(t2["primary"]["two_frame"][name])}
    out["V5_checks"] = v5
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print("written:", out_path, "sha256", sha(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
