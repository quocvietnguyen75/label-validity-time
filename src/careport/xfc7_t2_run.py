"""XF-C7 tier 2: the single post-freeze run (T2 protocol sections 9-11). Computes theta-hat for the first time.

AV2 (confirmatory): per decision camera, delta_H = weighted mean(theta-hat - theta_H) with a CR1 interval (Student t,
G - 1 df) for H in {start, A, B, mid}; rejected outside +-(5 + B_c); ruled out if rejected in >= 2 cameras in both the
primary and the margin-flag-excluded run (frozen xfc7_decision_v2 rule). Also: the two-frame timing check, pi-hat
(P01's mixture), C2 (pooled theta-bar and its sign), C3 (slope of theta-hat on speed), the P01 subgroup, and the
score >= 0.3 sensitivity. TruckScenes (descriptive): camera-level zero line, theta-bar against each hypothesis.

Usage (from the repository root, env loaded):
  PYTHONPATH=src python -m careport.xfc7_t2_run <config.json> <config_sha256> <protocol.md> <protocol_sha256> <out.json>
The config pins every input by path (relative to $AV2_SENSOR_ROOT / $TRUCKSCENES_DERIVED, or to the repository root)
and SHA-256, and names the protocol file. The run refuses to start if <out.json> exists, if a hash differs, if the
protocol is not the one the config names, or if a file it reads is not pinned."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from careport.xfc7_decision_v2 import combine_runs, decide_rejection
from careport.xfc7_estimators import weighted_mean_cr1, wls_intercept_slope_cr1
from careport.xfc7_primary_run import jackknife_delta
from careport.xfc7_t2_static import CAM_T2, CONTRASTS, HYP_T2, t_crit
from careport.xfc7_zero_line import apply_segment_correction, segment_intercepts, tau_hat

MARGIN_MS = 5.0
TWO_FRAME_LIMIT_MS = 10.0


def mean_ci_t(y, w, cluster):
    est, se = weighted_mean_cr1(y, w, cluster)
    G = int(pd.Series(cluster).nunique())
    t = t_crit(G)
    return {"est": float(est), "se": float(se), "lo": float(est - t * se), "hi": float(est + t * se), "G": G, "t": t}


def slope_ci_t(y, x, w, cluster):
    (b0, s0), (b1, s1) = wls_intercept_slope_cr1(y, x, w, cluster)
    G = int(pd.Series(cluster).nunique())
    t = t_crit(G)
    return {"intercept": b0, "intercept_lo": b0 - t * s0, "intercept_hi": b0 + t * s0,
            "slope": b1, "slope_lo": b1 - t * s1, "slope_hi": b1 + t * s1, "G": G}


def add_theta_hat(mo):
    mo = mo.copy()
    mo["theta_hat"] = tau_hat(mo[["p0_u", "p0_v"]].to_numpy(), mo[["u_ref", "v_ref"]].to_numpy(),
                              mo[["d_u", "d_v"]].to_numpy(), mo["s"].to_numpy(), mo["z_b"].to_numpy(), mo["sdot"].to_numpy())
    return mo


def _reweight(mo):
    mo = mo.copy()
    mo["n_track"] = mo.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    mo["w"] = mo["sdot"] ** 2 / (mo["sigma_b"] ** 2 * mo["n_track"])
    return mo


def _estimates(mo, names, hyp):
    out = {}
    for c, g in mo.groupby("camera"):
        name = names.get(int(c))
        if name is None:
            continue
        out[name] = {h: mean_ci_t(g["theta_hat"] - g[col], g["w"], g["segment"]) for h, col in hyp.items()}
        out[name]["_tracks"] = int(g[["segment", "object_id"]].drop_duplicates().shape[0])
    return out


def verdicts_t2(mo, names, budgets, hyp=None, exclude=()):
    """Rejection verdicts in the primary and margin-flag-excluded runs; `exclude` drops timing-model failures."""
    hyp = HYP_T2 if hyp is None else hyp
    cams = [n for n in budgets if n not in exclude]
    runs = {}
    for tag, sub in (("primary", mo), ("margin_flag_excluded", _reweight(mo[(mo["margin"] >= 0.2) | np.isinf(mo["margin"])]))):
        est = _estimates(sub, names, hyp)
        delta = {h: {c: (est[c][h]["est"], est[c][h]["lo"], est[c][h]["hi"]) for c in cams if c in est} for h in hyp}
        runs[tag] = {"estimates": est, "verdicts": decide_rejection(delta, budgets, [c for c in cams if c in est])}
    # an interval that cannot be formed (G < 2) counts as "not rejected"; list such cells so they are never silent
    undefined = sorted({(c, h) for r in runs.values() for c, e in r["estimates"].items() for h in hyp
                        if not np.isfinite(e[h]["lo"]) or not np.isfinite(e[h]["hi"])})
    return {"final": combine_runs(runs["primary"]["verdicts"], runs["margin_flag_excluded"]["verdicts"]),
            "estimates": runs["primary"]["estimates"], "runs": {k: v["verdicts"] for k, v in runs.items()},
            "estimates_margin_flag_excluded": runs["margin_flag_excluded"]["estimates"], "cameras": cams,
            "undefined_intervals": [list(x) for x in undefined]}


def two_frame_check(mo, names, limit=TWO_FRAME_LIMIT_MS):
    out = {}
    for c, g in mo.groupby("camera"):
        name = names.get(int(c))
        if name is None or g["frame_type"].nunique() < 2:
            continue
        r = slope_ci_t(g["theta_hat"], (g["frame_type"] == "A").astype(float), g["w"], g["segment"])
        out[name] = {"diff_ms": r["slope"], "lo": r["slope_lo"], "hi": r["slope_hi"], "G": r["G"],
                     "timing_model_failure": bool(r["slope_lo"] > limit or r["slope_hi"] < -limit)}
    return out


def mixture_pi(mo):
    """P01's mixture: mean theta = theta_B + pi (theta_A - theta_B). Primary: WLS through the origin of
    theta-hat - theta_B on theta_A - theta_B, CR1 by log (t, G - 1 df). The with-intercept fit is a descriptive check."""
    y = (mo["theta_hat"] - mo["theta_B"]).to_numpy(float)
    x = (mo["theta_A"] - mo["theta_B"]).to_numpy(float)
    w = mo["w"].to_numpy(float)
    sxx = float((w * x * x).sum())
    pi = float((w * x * y).sum() / sxx)
    sc = pd.Series(w * x * (y - pi * x)).groupby(mo["segment"].to_numpy()).sum().to_numpy()
    G = len(sc)
    se = float(np.sqrt((G / (G - 1)) * (sc ** 2).sum()) / sxx) if G > 1 else float("nan")
    t = t_crit(G)
    chk = slope_ci_t(y, x, w, mo["segment"])
    return {"pi_hat": pi, "pi_lo": pi - t * se, "pi_hi": pi + t * se, "G": G,
            "with_intercept_check": {"pi": chk["slope"], "pi_lo": chk["slope_lo"], "pi_hi": chk["slope_hi"],
                                     "intercept_ms": chk["intercept"], "intercept_lo": chk["intercept_lo"],
                                     "intercept_hi": chk["intercept_hi"]}}


def camera_zero_line(st, min_n=30):
    out = {}
    for c, g in st.groupby("camera"):
        if len(g) >= min_n:
            med = float(np.median(g["e"]))
            sig = float(1.4826 * np.median(np.abs(g["e"] - med)))
            if sig > 0:                                  # sigma_b = 0 would give infinite weights
                out[int(c)] = {"z_b": med, "sigma_b": sig, "n": int(len(g))}
    return out


def _sign(ci):
    return "positive" if ci["lo"] > 0 else ("negative" if ci["hi"] < 0 else "undetermined")


TS_MIN_OBS = 20


def ts_block(st, mo, hypotheses, names, report, min_obs=TS_MIN_OBS):
    """T2 section 11 (descriptive): camera-level zero line from assigned parked rows; theta-bar, C2's sign and
    delta_H only for the pinned `report` cameras, each hypothesis on the rows where its hit time exists (an interval
    only with >= min_obs such rows; otherwise the count alone)."""
    zl = camera_zero_line(st)
    mo = mo[mo["camera"].isin(list(zl))].copy()
    mo["z_b"] = mo["camera"].map(lambda c: zl[int(c)]["z_b"])
    mo["sigma_b"] = mo["camera"].map(lambda c: zl[int(c)]["sigma_b"])
    mo = add_theta_hat(_reweight(mo))
    counts = {names[int(c)]: {"obs": int(len(g)), "tracks": int(g[["segment", "object_id"]].drop_duplicates().shape[0]),
                              "scenes": int(g["segment"].nunique())} for c, g in mo.groupby("camera")}
    out = {"zero_line": {names[c]: v for c, v in zl.items()}, "counts": counts, "theta_bar": {}, "C2_sign": {}, "delta": {}}
    for c, g in mo.groupby("camera"):
        name = names[int(c)]
        if name not in report:
            continue
        out["theta_bar"][name] = mean_ci_t(g["theta_hat"], g["w"], g["segment"])
        out["C2_sign"][name] = _sign(out["theta_bar"][name])
        d, n = {}, {}
        for h in hypotheses:
            gh = _reweight(g[g[f"theta_{h}"].notna()])
            n[h] = int(len(gh))
            if len(gh) >= min_obs:
                d[h] = mean_ci_t(gh["theta_hat"] - gh[f"theta_{h}"], gh["w"], gh["segment"])
        out["delta"][name] = dict(d, _n=n)
    return out


def secondaries(mo, names, st, hyp=None):
    """Labelled secondaries (section 10), never changing a verdict: the segment jackknife of delta_H (frozen
    `jackknife_delta`) and delta_H after the per-log intercept correction (frozen `segment_intercepts` /
    `apply_segment_correction`; rows whose log x camera cell has no intercept are dropped and reweighted)."""
    hyp = HYP_T2 if hyp is None else hyp
    jk, seg = {}, {}
    ic = segment_intercepts(st) if st is not None else None
    for c, g in mo.groupby("camera"):
        name = names.get(int(c))
        if name is None:
            continue
        d = g.rename(columns={"theta_hat": "tau_hat"})
        G = int(d["segment"].nunique())
        jk[name] = {}
        for h, col in hyp.items():
            est, se = jackknife_delta(d, col) if G >= 2 else (float("nan"), float("nan"))
            jk[name][h] = {"est": est, "se": se, "lo": est - t_crit(G) * se, "hi": est + t_crit(G) * se, "G": G}
        if ic is not None:
            m = apply_segment_correction(d, ic)
            m = _reweight(m[m["c_seg"].notna()])
            seg[name] = {h: mean_ci_t(m["tau_hat_seg"] - m[col], m["w"], m["segment"]) for h, col in hyp.items()} if len(m) else {}
            seg[name]["_rows_with_intercept"] = int(len(m))
    return {"jackknife": jk, "segment_intercept": seg}


def av2_block(mo, names, budgets, decision, capacity_cams=None, st=None):
    """One AV2 analysis. Timing-model failures (two-frame check) leave the two-camera rule and the pooled C2, C3 and
    pi-hat estimates; two or more failures reject the timing model (section 10). `capacity_cams` (the S3 summary's
    per-camera capacity flags) gives each contrast's decidability among the cameras that pass the timing check."""
    mo = add_theta_hat(mo[mo["camera"].map(lambda c: names.get(int(c)) in decision)])
    tf = two_frame_check(mo, names)
    fail = [c for c, v in tf.items() if v["timing_model_failure"]]
    v = verdicts_t2(mo, names, {c: budgets[c] for c in decision}, exclude=fail)
    ok = _reweight(mo[~mo["camera"].map(lambda c: names[int(c)] in fail)]) if fail else mo
    out = {"verdicts": v, "two_frame": tf, "timing_model_failures": fail, "timing_model_rejected": len(fail) >= 2}
    if ok["segment"].nunique() >= 2:
        pooled = mean_ci_t(ok["theta_hat"], ok["w"], ok["segment"])
        out.update({"pi": mixture_pi(ok), "C2_pooled_theta_bar": pooled, "C2_sign": _sign(pooled),
                    "C3_speed": slope_ci_t(ok["theta_hat"], ok["speed"], ok["w"], ok["segment"]),
                    "C3_inv_sdot": slope_ci_t(ok["theta_hat"], 1.0 / ok["sdot"], ok["w"], ok["segment"])})
    else:                                                # every camera failed the timing check, or < 2 logs
        out.update({k: None for k in ("pi", "C2_pooled_theta_bar", "C3_speed", "C3_inv_sdot")})
        out["C2_sign"] = "undefined"
    out.update({
           "C3_speed_by_camera": {names[int(c)]: slope_ci_t(g["theta_hat"], g["speed"], g["w"], g["segment"])
                                  for c, g in mo.groupby("camera") if g["segment"].nunique() >= 2},
           "theta_bar_by_camera": {names[int(c)]: mean_ci_t(g["theta_hat"], g["w"], g["segment"]) for c, g in mo.groupby("camera")}})
    if capacity_cams is not None:
        live = [c for c in decision if c not in fail]
        out["decidable_after_timing"] = {f"{a}-{b}": sum(bool(capacity_cams.get(c, {}).get(f"{a}-{b}", {}).get("decidable"))
                                                         for c in live) >= 2 for a, b in CONTRASTS}
    if st is not None:
        out["secondary"] = secondaries(mo, names, st)
    return out


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _check(path, expected, what):
    got = sha(path)
    if got != expected:
        raise SystemExit(f"{what}: hash {got} != pinned {expected}; stop")


def main(argv):
    if len(argv) != 6:
        print(__doc__)
        return 64
    cfg_path, cfg_sha, proto, proto_sha, out_path = argv[1:6]
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; the run is done once and never overwritten; stop")
    _check(cfg_path, cfg_sha, "config")
    _check(proto, proto_sha, "protocol")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if os.path.basename(cfg["protocol"]) != os.path.basename(proto):
        raise SystemExit(f"protocol {os.path.basename(proto)} is not the one the config names ({cfg['protocol']}); stop")
    pinned = {item["path"] for item in cfg["inputs"]}
    reads = [cfg[t][k] for t in ("primary", "sensitivity_score_0_3") for k in ("s3_summary", "moving_set", "static_zero_lined")]
    if cfg.get("truckscenes"):
        reads += [cfg["truckscenes"]["static_all"], cfg["truckscenes"]["moving_assigned"]]
    unpinned = [r for r in reads if r not in pinned]
    if unpinned:
        raise SystemExit(f"files read but not pinned in the config: {unpinned}; stop")
    roots = {"$AV2_SENSOR_ROOT": os.environ["AV2_SENSOR_ROOT"], "$TRUCKSCENES_DERIVED": os.environ.get("TRUCKSCENES_DERIVED", "")}
    path = lambda p: next((p.replace(k, v) for k, v in roots.items() if p.startswith(k)), p)
    for item in cfg["inputs"]:
        _check(path(item["path"]), item["sha256"], item["what"])
    out = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "config_sha256": cfg_sha, "protocol_sha256": proto_sha}
    for tag in ("primary", "sensitivity_score_0_3"):
        with open(path(cfg[tag]["s3_summary"]), encoding="utf-8") as f:
            s3 = json.load(f)
        exp = cfg["expected"]
        if s3.get("S1_rule") != exp["S1_rule"] or sorted(s3["decision_cameras"]) != sorted(exp["decision_cameras"][tag]):
            raise SystemExit(f"{tag}: summary S1 rule or decision cameras differ from the config's expected block; stop")
        mo = pd.read_parquet(path(cfg[tag]["moving_set"]))
        st = pd.read_parquet(path(cfg[tag]["static_zero_lined"]))
        budgets = {c: s3["budget"][c]["B_c_ms"] for c in s3["decision_cameras"]}
        cap = s3["capacity"]["cameras"]
        out[tag] = av2_block(mo, CAM_T2, budgets, s3["decision_cameras"], capacity_cams=cap, st=st)
        if tag == "primary" and cfg.get("p01_logs"):
            sub = av2_block(mo[mo["segment"].isin(cfg["p01_logs"])], CAM_T2, budgets, s3["decision_cameras"],
                            capacity_cams=cap, st=st)
            sub["verdicts"]["final_descriptive"] = sub["verdicts"].pop("final")      # section 10: no verdict
            sub["role"] = "P01's ten logs; descriptive subgroup with the 50-log budgets and cameras; no verdict"
            out["p01_subgroup"] = sub
    if cfg.get("truckscenes"):
        ts = cfg["truckscenes"]
        out["truckscenes_descriptive"] = ts_block(pd.read_parquet(path(ts["static_all"])),
                                                  pd.read_parquet(path(ts["moving_assigned"])), ts["hypotheses"],
                                                  {int(k): v for k, v in ts["camera_names"].items()}, ts["report_cameras"])
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    p = out["primary"]
    print(json.dumps({"final": p["verdicts"]["final"], "two_frame_failures": p["timing_model_failures"],
                      "pi": round(p["pi"]["pi_hat"], 3), "C2": p["C2_sign"]}, indent=1))
    print("results:", out_path, "sha256", sha(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
