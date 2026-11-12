"""XF-C7 T3: the single post-freeze run on the 408 fresh AV2 logs (PILOT_PROTOCOL_XF-C7_T3_v1_0.md).

T2's AV2 analysis (xfc7_t2_run, unchanged) with three changes:
- section 10: per decision camera and hypothesis, three 95 % intervals for delta_H = weighted mean(theta-hat -
  theta_H): CR1 with t(G - 1) (frozen mean_ci_t), CR2 with Bell-McCaffrey df (R4's cr2_bm_weighted_mean) and the wild
  cluster bootstrap-t (R4's wild_cluster_bootstrap_t, Webb six-point, 99,999 draws, seed 20261103). A camera rejects H
  only if all three intervals lie outside +-(5 + B_c) (xfc7_decision_v2.rejected on each); H is ruled out if it is
  rejected in >= 2 cameras in both the primary and the margin-flag-excluded run (frozen combine_runs);
- section 9: the gate's half-width uses the Bell-McCaffrey df, hw_BM = t(nu_BM) sqrt(DE V); nu_BM depends on the
  weights and logs only;
- section 11: labelled secondaries S-a (tolerance compositions), S-b (T2's 50 logs on H100 tables, with T2's frozen
  budgets and with the H100 ones), S-c (pooled 458 logs), S-d (injected-shift recovery), S-e (score >= 0.3), S-f (T2's
  jackknife and per-log intercept secondaries, via av2_block), S-g (val / train subgroups, descriptive) and S-h
  (realised against predicted half-widths).

Usage (repository root, env loaded):
  PYTHONPATH=src python -m careport.xfc7_t3_run <config.json> <config_sha256> <protocol.md> <protocol_sha256> <out.json>
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

from careport.xfc7_decision_v2 import combine_runs, decide_rejection, rejected
from careport.xfc7_posthoc import shift_along
from careport.xfc7_t2_posthoc import composition_budgets
from careport.xfc7_t2_posthoc_r4 import DRAWS, SEED, cr2_bm_weighted_mean, kish_effective_clusters, wild_cluster_bootstrap_t
from careport.xfc7_t2_run import _reweight, add_theta_hat, av2_block, mean_ci_t
from careport.xfc7_t2_static import CAM_T2, CONTRASTS, HYP_T2, K4, capacity, run_reference_t2

MARGIN_MS = 5.0
METHODS = ("cr1", "cr2_bm", "bootstrap")
IMG = ["segment", "frame", "camera", "camera_object_id"]
E4_DELTAS = (-20.0, -10.0, 10.0, 20.0)
E4_BAND = (0.9, 1.1)
NAN = float("nan")


# ---------- section 10: three intervals and the per-camera rule ----------
def intervals_three(y, w, cluster, draws=DRAWS, seed=SEED) -> dict:
    """The three 95 % intervals of the weighted mean of y; every interval is NaN below two logs."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    cluster = np.asarray(cluster)
    G = int(pd.Series(cluster).nunique())
    if G < 2:
        est = float((w * y).sum() / w.sum()) if w.sum() > 0 else NAN
        und = {"est": est, "lo": NAN, "hi": NAN, "G": G}
        return {m: dict(und) for m in METHODS}
    return {"cr1": mean_ci_t(y, w, cluster), "cr2_bm": cr2_bm_weighted_mean(y, w, cluster),
            "bootstrap": wild_cluster_bootstrap_t(y, w, cluster, draws=draws, seed=seed)}


def _triple(d):
    return (float(d["est"]), float(d["lo"]), float(d["hi"]))


def camera_rejects(ints, budget, margin=MARGIN_MS) -> bool:
    """ints: method -> (est, lo, hi). True only if every method's interval lies outside +-(margin + budget)."""
    return all(rejected(ints[m], budget, margin) for m in METHODS)


def decide_three(by_method, budgets, cams, margin=MARGIN_MS) -> dict:
    """by_method: method -> hypothesis -> camera -> (est, lo, hi). The per-camera reading of section 10."""
    out = {}
    hyps = list(by_method[METHODS[0]])
    for h in hyps:
        live = [c for c in cams if all(c in by_method[m].get(h, {}) for m in METHODS)]
        rej = [c for c in live if camera_rejects({m: by_method[m][h][c] for m in METHODS}, budgets[c], margin)]
        per = {m: [c for c in cams if c in by_method[m].get(h, {}) and rejected(by_method[m][h][c], budgets[c], margin)]
               for m in METHODS}
        out[h] = {"rejected_in": rej, "ruled_out": len(rej) >= 2, "rejected_in_by_method": per}
    return out


def cells(frame, names, hyp, cams, ycol="theta_hat"):
    """camera name -> hypothesis -> (y, w, cluster) for the cameras in `cams` (R4's _cells, with theta-hat)."""
    out = {}
    for c, g in frame.groupby("camera"):
        name = names.get(int(c))
        if name not in cams:
            continue
        out[name] = {h: (g[ycol].to_numpy(float) - g[col].to_numpy(float), g["w"].to_numpy(float), g["segment"].to_numpy())
                     for h, col in hyp.items()}
    return out


def verdicts_three(runs, budgets, cams, draws=DRAWS, seed=SEED, margin=MARGIN_MS) -> dict:
    """runs: 'primary' and 'margin_flag_excluded' -> camera -> hypothesis -> (y, w, cluster). Returns every interval,
    the three-method verdicts (section 10) and each method's own verdicts under the frozen rule."""
    res, iv = {}, {m: {} for m in METHODS}
    hyps = None
    for tag, by_cam in runs.items():
        res[tag] = {}
        for m in METHODS:
            iv[m][tag] = {}
        for cam, by_h in by_cam.items():
            res[tag][cam] = {}
            for h, (y, w, cl) in by_h.items():
                ints = intervals_three(y, w, cl, draws=draws, seed=seed)
                res[tag][cam][h] = ints
                for m in METHODS:
                    iv[m][tag].setdefault(h, {})[cam] = _triple(ints[m])
            hyps = hyps or list(by_h)
        for m in METHODS:
            for h in hyps or []:
                iv[m][tag].setdefault(h, {})
    three_runs = {tag: decide_three({m: iv[m][tag] for m in METHODS}, budgets, cams, margin) for tag in runs}
    by_method = {}
    for m in METHODS:
        per = {tag: decide_rejection(iv[m][tag], budgets, [c for c in cams if any(c in v for v in iv[m][tag].values())],
                                     margin=margin) for tag in runs}
        by_method[m] = {"runs": per, "final": combine_runs(per["primary"], per["margin_flag_excluded"])}
    undefined = sorted({(tag, c, h, m) for tag, r in res.items() for c, by_h in r.items() for h, ints in by_h.items()
                        for m in METHODS if not (np.isfinite(ints[m]["lo"]) and np.isfinite(ints[m]["hi"]))})
    return {"intervals": res, "three": {"runs": three_runs,
                                        "final": combine_runs(three_runs["primary"], three_runs["margin_flag_excluded"])},
            "by_method": by_method, "undefined_intervals": [list(x) for x in undefined]}


def _label(est, rej, finite, band):
    if rej:
        return "rejected"
    if not np.isfinite(est):
        return "undefined interval"
    if abs(est) <= band:
        return "estimate failure"
    return "capacity failure" if finite else "undefined interval"


def failure_labels(three, budgets, hw_pred, margin=MARGIN_MS, capacity=None) -> dict:
    """Descriptive labels (T2 section 10, T3 sections 10 and 11 S-h); they change no verdict. Per run, camera and
    hypothesis, for the three-method rule and for each method: 'rejected'; 'estimate failure' if |est| <= 5 + B_c;
    'capacity failure' if |est| > 5 + B_c but an interval reaches inside the band; 'undefined interval' if an interval
    cannot be formed and the rest does not settle it. Each method also gets its realised half-width and whether it
    exceeds the predicted hw_BM of that run (`hw_pred`: run -> camera -> ms; None where there is no prediction).
    Section 10 defines the two failures for decidable contrasts, so with `capacity` (run -> camera -> contrast ->
    {'decidable': ...}, the hw_BM gate's flags) each label also says which contrasts involving its hypothesis this
    camera clears in that run (`camera_clears`)."""
    out = {}
    for tag, by_cam in three["intervals"].items():
        out[tag] = {}
        for c, by_h in by_cam.items():
            if c not in budgets:
                continue
            band = margin + budgets[c]
            pred = hw_pred.get(tag, {}).get(c)
            cap = capacity.get(tag, {}).get(c, {}) if capacity is not None else None
            out[tag][c] = {}
            for h, ints in by_h.items():
                per = {}
                for m in METHODS:
                    est, lo, hi = _triple(ints[m])
                    finite = bool(np.isfinite(lo) and np.isfinite(hi))
                    hw = (hi - lo) / 2 if finite else NAN
                    exceeds = bool(hw > pred) if finite and pred is not None and np.isfinite(pred) else None
                    per[m] = {"label": _label(est, rejected((est, lo, hi), budgets[c], margin), finite, band),
                              "realised_hw_ms": hw, "realised_exceeds_hw_bm": exceeds}
                est = float(ints[METHODS[0]]["est"])
                all_rej = all(per[m]["label"] == "rejected" for m in METHODS)
                all_fin = all(np.isfinite(per[m]["realised_hw_ms"]) for m in METHODS)
                out[tag][c][h] = {"three": _label(est, all_rej, all_fin, band), "by_method": per,
                                  "predicted_hw_bm_ms": pred}
                if cap is not None:
                    out[tag][c][h]["camera_clears"] = {f"{a}-{b}": bool(cap.get(f"{a}-{b}", {}).get("decidable"))
                                                       for a, b in CONTRASTS if h in (a, b)}
    return out


def reading(decidable, timing_model_rejected, composition=None) -> dict:
    """The protocol's fixed readings for one analysis, beside its verdicts (which stay as computed):
    - T3 sections 10 and 12 (T2 section 10): two or more two-frame failures reject the timing model, and every
      estimate is then descriptive;
    - T3 sections 9 and 12 (T2 section 9): the H_start verdict is informative against H' only if start-H' is
      decidable (`decidable`: two cameras in both runs, after the two-frame check);
    - T3 section 11, S-a: a composition whose own gate leaves no contrast decidable is 'undecidable under B_<name>',
      never 'not ruled out'."""
    dec = [k for k, v in decidable.items() if v]
    out = {"timing_model_rejected": bool(timing_model_rejected), "all_descriptive": bool(timing_model_rejected),
           "decidable_contrasts": dec, "h_start_informative_against": [h for h in ("A", "B", "mid") if f"start-{h}" in dec]}
    if composition is not None:
        out["undecidable"] = not dec
        out["label"] = f"undecidable under B_{composition}" if not dec else None
    return out


# ---------- section 9: design table and the hw_BM gate ----------
def nu_bm(w, cluster) -> float:
    """Bell-McCaffrey df of the weighted mean; a function of the weights and logs only (y is a placeholder)."""
    w = np.asarray(w, dtype=float)
    return float(cr2_bm_weighted_mean(np.zeros(w.size), w, cluster)["df"])


def hw_bm(de, v, nu) -> float:
    return float(stats.t.ppf(0.975, nu) * math.sqrt(de * v)) if np.isfinite(nu) and nu > 0 else NAN


def v_ind(g) -> float:
    var = g["sigma_b"].to_numpy(float) ** 2 / g["sdot"].to_numpy(float) ** 2
    w = g["w"].to_numpy(float)
    return float((w ** 2 * var).sum() / w.sum() ** 2)


def theta_bar(g, hyp=HYP_T2) -> dict:
    """Label-side predicted theta-bar per hypothesis under the rows' weights w, as the static step computes it: rows
    without a prediction for a hypothesis are left out of it, and it is NaN if no row has one. No theta-hat."""
    w = g["w"].to_numpy(float)
    out = {}
    for h, col in hyp.items():
        ok = g[col].notna().to_numpy()
        out[h] = float((w[ok] * g[col].to_numpy(float)[ok]).sum() / w[ok].sum()) if ok.any() else NAN
    return out


def design_row(g, hyp=None) -> dict:
    """One camera's moving rows (segment, object_id, w, sdot, sigma_b): weights-only design quantities; with `hyp` and
    its label-side prediction columns present, also the predicted theta-bar under these rows' weights."""
    w = g["w"].to_numpy(float)
    S = float(w.sum())
    by_log = g.groupby("segment")["w"].sum().to_numpy(float)
    by_track = g.groupby(["segment", "object_id"])["w"].sum().sort_values(ascending=False).to_numpy(float)
    row = {"tracks": int(len(by_track)), "logs": int(len(by_log)), "nu_bm": nu_bm(w, g["segment"].to_numpy()),
           "kish_logs": kish_effective_clusters(w, g["segment"].to_numpy()), "top_log_share": float(by_log.max() / S),
           "top5_track_share": float(by_track[:5].sum() / S), "V_ind": v_ind(g)}
    if hyp and all(col in g.columns for col in hyp.values()):
        row["theta_bar"] = theta_bar(g, hyp)
    return row


def design_table(mo, names, cams, hyp=HYP_T2) -> dict:
    """Per camera, for the primary and the margin-flag-excluded run (weights recomputed as T2)."""
    out = {}
    excl = _reweight(mo[(mo["margin"] >= 0.2) | np.isinf(mo["margin"])])
    for tag, frame in (("primary", mo), ("margin_flag_excluded", excl)):
        out[tag] = {names[int(c)]: design_row(g, hyp) for c, g in frame.groupby("camera") if names.get(int(c)) in cams}
    return out


def gate_bm(theta_bar, budgets, de, v, nu, contrasts=CONTRASTS, margin=MARGIN_MS) -> dict:
    """Thresholds 5 + B_c + hw_BM,c and the frozen capacity table over them."""
    hw = {c: hw_bm(de[c], v[c], nu[c]) for c in theta_bar}
    thr = {c: margin + budgets[c] + hw[c] for c in theta_bar}
    return {"hw_bm_ms": hw, "thresholds_ms": thr, "capacity": capacity(theta_bar, thr, contrasts)}


RUNS = ("primary", "margin_flag_excluded")


def run_theta(s3, des, tag, cams) -> dict:
    """Predicted theta-bar per camera for one run: the S3 summary's for the primary run; for the margin-flag-excluded
    run, the label-side predictions averaged under that run's weights (design_table), as T2's t2_margin_capacity.py."""
    if tag == "primary":
        return {c: s3["cameras"][c]["predicted_theta_bar"] for c in cams if c in des["primary"]}
    return {c: des[tag][c]["theta_bar"] for c in cams if c in des[tag]}


def gates_both(s3, des, cams, budgets) -> dict:
    """The hw_BM gate of both runs: each run's own theta-bar, V_ind and nu_BM, with DE from the S3 summary."""
    out = {}
    for tag in RUNS:
        th = run_theta(s3, des, tag, cams)
        out[tag] = gate_bm(th, {c: budgets[c] for c in th}, {c: float(s3["cameras"][c]["DE"]) for c in th},
                           {c: des[tag][c]["V_ind"] for c in th}, {c: des[tag][c]["nu_bm"] for c in th})
    return out


def decidable_both(gates, cams, contrasts=CONTRASTS) -> dict:
    """Section 9 (T3): a contrast is decidable only if at least two of `cams` clear its hw_BM threshold in every run
    of `gates` (primary and margin-flag-excluded), as the verdict needs both runs. A camera absent from a run's gate
    counts as not clearing; passing only the cameras that survive the two-frame check gives decidability after it."""
    keys = [f"{a}-{b}" for a, b in contrasts]
    runs = {tag: {k: sum(bool(g["capacity"]["cameras"].get(c, {}).get(k, {}).get("decidable")) for c in cams) >= 2
                  for k in keys} for tag, g in gates.items()}
    return {"runs": runs, "decidable": {k: all(r[k] for r in runs.values()) for k in keys}}


# ---------- S-d: injected-shift recovery on AV2 ----------
def shift_reference(ref, mo, delta) -> pd.DataFrame:
    """Move each detector box assigned to a moving row of `mo` by -delta * sdot along d-hat (EXT v1.0 section 8)."""
    m = mo[mo["camera_object_id"].notna()]
    uv = shift_along(np.zeros((len(m), 2)), m[["d_u", "d_v"]].to_numpy(float), m["sdot"].to_numpy(float), delta)
    sh = m[IMG].assign(du=uv[:, 0], dv=uv[:, 1]).drop_duplicates(IMG)
    b = ref.merge(sh, on=IMG, how="left")
    b["u"] = b["u"] + b["du"].fillna(0.0)
    b["v"] = b["v"] + b["dv"].fillna(0.0)
    return b[list(ref.columns)]


def _delta_start(mo, names, cams):
    out = {}
    for c, g in mo.groupby("camera"):
        n = names.get(int(c))
        if n in cams and len(g):
            out[n] = float((g["w"] * (g["theta_hat"] - g["theta_start"])).sum() / g["w"].sum())
    return out


def e4_av2(lab, ref, mo_base, cams, names, deltas=E4_DELTAS, static_fn=run_reference_t2) -> dict:
    """Re-associate with shifted boxes and report the recovered share of each shift per decision camera. A zero shift
    is run first: it must reproduce the baseline moving set's theta-hat."""
    mo_base = add_theta_hat(mo_base)
    d0 = _delta_start(mo_base, names, cams)

    def one(delta):
        res = static_fn(lab, shift_reference(ref, mo_base, delta))
        mo2 = add_theta_hat(res["mo"][res["mo"]["camera"].map(lambda c: names.get(int(c)) in cams)])
        chk = mo_base[K4 + ["camera_object_id"]].merge(
            res["assoc"][K4 + ["camera_object_id"]].rename(columns={"camera_object_id": "cid_new"}), on=K4, how="left")
        changed = float((chk["camera_object_id"].fillna("") != chk["cid_new"].fillna("")).mean()) if len(chk) else NAN
        return mo2, changed

    mo0, ch0 = one(0.0)
    k = K4 + ["theta_hat", "w"]
    a, b = mo_base[k].sort_values(K4).reset_index(drop=True), mo0[k].sort_values(K4).reset_index(drop=True)
    same = len(a) == len(b) and bool((a[K4].astype(str).values == b[K4].astype(str).values).all()) and \
        bool(np.allclose(a[["theta_hat", "w"]].to_numpy(float), b[["theta_hat", "w"]].to_numpy(float), rtol=0, atol=1e-9))
    out = {"baseline_delta_start_ms": d0, "baseline_reproduced": bool(same and ch0 == 0.0), "zero_shift_changed_share": ch0,
           "runs": {}}
    for dlt in deltas:
        mo2, changed = one(dlt)
        d1 = _delta_start(mo2, names, cams)
        out["runs"][f"{dlt:+.0f}ms"] = {"assignment_changed_share": changed, "cameras": {
            c: {"recovered_shift_ms": d1[c] - d0[c], "recovery_ratio": (d1[c] - d0[c]) / dlt} for c in d0 if c in d1},
            "missing_cameras": [c for c in d0 if c not in d1]}
    # every decision camera with baseline rows must recover within the band in every shifted run; a camera that
    # loses its rows fails, rather than dropping out of the check
    out["pass_band_90_110"] = all(c in r["cameras"] and E4_BAND[0] <= r["cameras"][c]["recovery_ratio"] <= E4_BAND[1]
                                  for r in out["runs"].values() for c in d0)
    return out


# ---------- the run ----------
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _check(path, expected, what):
    got = sha(path)
    if got != expected:
        raise SystemExit(f"{what}: {path} has SHA-256 {got}, pinned {expected}; stop")


def _decision_frame(mo, cams):
    return add_theta_hat(mo[mo["camera"].map(lambda c: CAM_T2.get(int(c)) in cams)])


def analysis(mo, s3, cams=None, budgets=None, components=None, draws=DRAWS, seed=SEED, st=None, with_block=True):
    """One AV2 analysis: T2's av2_block (CR1 verdicts, two-frame check, pi-hat, C2, C3 and, with st, the jackknife and
    per-log intercept secondaries), the three-method verdicts, the design table and the hw_BM gate, C2's CR2-BM
    interval, the descriptive failure labels and the fixed readings."""
    cams = list(s3["decision_cameras"]) if cams is None else list(cams)
    budgets = {c: float(s3["budget"][c]["B_c_ms"]) for c in cams} if budgets is None else budgets
    out = {"cameras": cams, "budgets_ms": budgets}
    if with_block:
        out["t2_block"] = av2_block(mo, CAM_T2, budgets, cams, capacity_cams=s3["capacity"]["cameras"], st=st)
    fail = out["t2_block"]["timing_model_failures"] if with_block else []
    trej = bool(out["t2_block"]["timing_model_rejected"]) if with_block else False
    live = [c for c in cams if c not in fail]
    dm = _decision_frame(mo, live)
    excl = add_theta_hat(_reweight(mo[mo["camera"].map(lambda c: CAM_T2.get(int(c)) in live)][
        lambda d: (d["margin"] >= 0.2) | np.isinf(d["margin"])]))
    runs = {"primary": cells(dm, CAM_T2, HYP_T2, live), "margin_flag_excluded": cells(excl, CAM_T2, HYP_T2, live)}
    out["three_method"] = verdicts_three(runs, {c: budgets[c] for c in live}, live, draws=draws, seed=seed)
    # section 10, C2: the pooled theta-bar over the decision cameras that pass the two-frame check (av2_block's rows)
    # gets a CR2-BM interval beside av2_block's CR1 one
    out["C2_pooled_cr2_bm"] = (cr2_bm_weighted_mean(dm["theta_hat"].to_numpy(float), dm["w"].to_numpy(float),
                                                    dm["segment"].to_numpy()) if dm["segment"].nunique() >= 2 else None)
    des = design_table(mo, CAM_T2, cams)
    out["design"] = des
    comps = composition_budgets(components) if components else {"max": None}
    gates = gates_both(s3, des, cams, budgets)
    out["gate_bm"] = gates
    out["decidable_bm"] = {"all_decision_cameras": decidable_both(gates, cams), "after_timing": decidable_both(gates, live)}
    out["reading"] = reading(out["decidable_bm"]["after_timing"]["decidable"], trej)
    out["failure_labels"] = failure_labels(out["three_method"], {c: budgets[c] for c in live},
                                           {tag: gates[tag]["hw_bm_ms"] for tag in RUNS},
                                           capacity={tag: gates[tag]["capacity"]["cameras"] for tag in RUNS})
    present = [c for c in live if c in out["three_method"]["intervals"]["primary"] and c in des["primary"]]
    out["cameras_without_rows"] = [c for c in cams if c not in des["primary"]]
    if components:
        out["compositions"] = {}
        for name in ("rss", "sum"):
            b = {c: comps[c][name] for c in present}
            g2 = gates_both(s3, des, present, b)
            d2 = decidable_both(g2, present)
            out["compositions"][name] = {"budgets_ms": b, "verdicts": {"three": _rejudge(out["three_method"], b, present)},
                                         "gates": g2, "gate_primary": g2["primary"], "decidable_bm": d2,
                                         "reading": reading(d2["decidable"], trej, composition=name)}
    iv_p = out["three_method"]["intervals"]["primary"]
    out["halfwidths"] = {c: {"predicted_t_G_minus_1_ms": float(s3["cameras"][c]["predicted_halfwidth_ms"]),
                             "predicted_bm_ms": gates["primary"]["hw_bm_ms"].get(c, NAN),
                             "realised_start_ms": {m: (iv_p[c]["start"][m]["hi"] - iv_p[c]["start"][m]["lo"]) / 2
                                                   for m in METHODS}} for c in present}
    return out


def _rejudge(three, budgets, cams, margin=MARGIN_MS):
    """The three-method verdicts of stored intervals under other budgets (S-a)."""
    runs = {}
    for tag, by_cam in three["intervals"].items():
        bm = {m: {} for m in METHODS}
        for c, by_h in by_cam.items():
            if c not in cams:
                continue
            for h, ints in by_h.items():
                for m in METHODS:
                    bm[m].setdefault(h, {})[c] = _triple(ints[m])
        runs[tag] = decide_three(bm, budgets, cams, margin)
    return {"runs": runs, "final": combine_runs(runs["primary"], runs["margin_flag_excluded"])}


def subgroup_analyses(mo, s3, groups, draws=DRAWS, seed=SEED) -> dict:
    """S-g (section 11): each subgroup of logs with the full sample's budgets and decision cameras and no T2 block.
    Descriptive, so every final verdict, of the three-method rule and of each method, is renamed final_descriptive."""
    out = {}
    for name, logs in groups.items():
        sub = analysis(mo[mo["segment"].isin(set(logs))], s3, draws=draws, seed=seed, with_block=False)
        tm = sub["three_method"]
        tm["three"]["final_descriptive"] = tm["three"].pop("final")
        for m in METHODS:
            tm["by_method"][m]["final_descriptive"] = tm["by_method"][m].pop("final")
        out[name] = sub
    return out


def main(argv):
    if len(argv) != 6:
        print(__doc__)
        return 64
    cfg_path, cfg_sha, proto, proto_sha, out_path = argv[1:6]
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; the run is done once and never overwritten; stop")
    _check(cfg_path, cfg_sha, "config")
    _check(proto, proto_sha, "protocol")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    if os.path.basename(cfg["protocol"]) != os.path.basename(proto):
        raise SystemExit(f"protocol {os.path.basename(proto)} is not the one the config names ({cfg['protocol']}); stop")
    roots = {"$AV2_SENSOR_ROOT": os.environ["AV2_SENSOR_ROOT"]}
    path = lambda p: next((p.replace(k, v) for k, v in roots.items() if p.startswith(k)), p)
    pinned = {item["path"]: item for item in cfg["inputs"]}
    reads = [p for blk in cfg["analyses"].values() for p in blk["reads"].values()] + list(cfg["comparison"].values())
    unpinned = [r for r in reads if r not in pinned]
    if unpinned:
        raise SystemExit(f"files read but not pinned in the config: {unpinned}; stop")
    for item in cfg["inputs"]:
        _check(path(item["path"]), item["sha256"], item["what"])
    boot = cfg["bootstrap"]
    if (boot["draws"], boot["seed"]) != (DRAWS, SEED):
        raise SystemExit("bootstrap settings differ from R4's; stop")
    out = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "config_sha256": cfg_sha,
           "protocol_sha256": proto_sha, "settings": {"margin_ms": MARGIN_MS, "bootstrap": boot, "methods": list(METHODS)}}
    load_json = lambda p: json.load(open(path(p), encoding="utf-8"))

    # reproduction check: T2's frozen CR1 verdicts from T2's own moving set (stops on a mismatch)
    comp = cfg["comparison"]
    t2_s3, t2_res = load_json(comp["t2_s3_summary"]), load_json(comp["t2_results"])
    t2_mo = pd.read_parquet(path(comp["t2_moving_set"]))
    rep = analysis(t2_mo, t2_s3, draws=DRAWS, seed=SEED, with_block=True)
    if rep["t2_block"]["verdicts"]["final"] != t2_res["primary"]["verdicts"]["final"]:
        raise SystemExit("T2's frozen CR1 verdicts are not reproduced; stop")
    r4 = load_json(comp["r4_results"])["R4_1_AV2"]
    for tag in ("primary", "margin_flag_excluded"):
        for c, by_h in r4["cells"][tag].items():
            for h, ints in by_h.items():
                for m in METHODS:
                    got = rep["three_method"]["intervals"][tag][c][h][m]
                    for key in ("lo", "hi"):
                        if abs(float(got[key]) - float(ints[m][key])) > 0.01:
                            raise SystemExit(f"R4 {tag} {c} {h} {m} {key} not reproduced; stop")
    out["reproduction"] = {"t2_cr1_final": rep["t2_block"]["verdicts"]["final"], "r4_intervals_within_ms": 0.01,
                           "t2_three_method_final": rep["three_method"]["three"]["final"]}

    for tag, blk in cfg["analyses"].items():
        r = blk["reads"]
        s3 = load_json(r["s3_summary"])
        exp = blk["expected"]
        if s3.get("S1_rule") != exp["S1_rule"] or sorted(s3["decision_cameras"]) != sorted(exp["decision_cameras"]):
            raise SystemExit(f"{tag}: summary S1 rule or decision cameras differ from the config; stop")
        mo = pd.read_parquet(path(r["moving_set"]))
        st = pd.read_parquet(path(r["static_zero_lined"])) if "static_zero_lined" in r else None
        comps = {c: s3["budget"][c] for c in s3["decision_cameras"]}
        # S-b (i) needs no branch: its block reads T2's frozen S3 summary (budgets, decision cameras, capacity) with
        # the H100 moving set and parked table of the same logs
        res = analysis(mo, s3, components=comps if blk.get("compositions") else None, st=st)
        res["role"] = blk["role"]
        if blk.get("subgroups"):                                        # S-g, descriptive
            res["subgroups"] = subgroup_analyses(mo, s3, blk["subgroups"])
        if blk.get("e4"):                                               # S-d, inside the run
            lab = pd.read_parquet(path(r["label_side"]))
            ref = pd.read_parquet(path(r["reference_table"]))
            res["e4"] = e4_av2(lab, ref, mo[mo["camera"].map(lambda c: CAM_T2.get(int(c)) in s3["decision_cameras"])],
                               s3["decision_cameras"], CAM_T2)
            del lab, ref
        out[tag] = res
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({t: out[t]["three_method"]["three"]["final"] for t in cfg["analyses"]}, indent=1))
    print("results:", out_path, "sha256", sha(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
