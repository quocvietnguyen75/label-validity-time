"""XF-C7 post hoc set V5S: v5 review suggested items S23, S51, S52, S54 and the display values for R9.

The plan is the pilot record entry '2026-11-05, post hoc set V5S', pinned before anything was computed; the record's
SHA-256 just after the pin is PIN_RECORD_SHA. Arithmetic on stored results only; everything here is post hoc. No frozen
verdict and no published interval changes.

- V5S-1 (S52): the misplacement floor after subtracting B_c, (lo_start - B_c) x 10 m/s, CR1 and over three methods.
- V5S-2 (S23): the six contrasts' decidability at realised half-widths (largest over the three methods and both
  conventions), two cameras in both runs.
- V5S-3 (S51): the all-range projection screen for the cameras that reject the sweep start.
- V5S-4 (S54): the WOD detector's margin run under the quadrature tolerance, from R4's stored intervals.
- V5S-5 (R9): display tables of all three intervals in both runs for the AV2 replication.

Usage (repository root): PYTHONPATH=src python -m careport.xfc7_t3_posthoc_v5s <cycle_dir> <out_dir>
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from careport.xfc7_decision_v2 import combine_runs, decide_rejection
from careport.xfc7_t2_posthoc import composition_budgets
from careport.xfc7_t2_static import CONTRASTS

PIN_RECORD_SHA = "07f5f7a80cc27b0f6e21bac9c2d2aba05bdda99ae663de04fe62ff97ccb0713b"
MARGIN_MS = 5.0
METHODS = ("cr1", "cr2_bm", "bootstrap")
RUNS = ("primary", "margin_flag_excluded")
SPEED_MPS = 10.0
SCREEN_LIMIT_PX = 15.0
T3 = "t3_replication_2026-11-04"
INPUTS = {
    f"{T3}/cluster_results/results/T3_RESULTS_2026-11-05.json": "e28f472dca12f9438bc9d7fe7bc4cd95d3cbe0ff488a1dcfd388b8ae94d6f3dc",
    f"{T3}/cluster_results/derived/t3_static/S3_summary_s050.json": "e8f9229a76e28a726cfc387833912c7c32bd626bb9419c177ee9e6d2c994df3c",
    "tier2_2026-11-01/posthoc_2026-11-03/r4_results.json": "8ad246fabce6d50cdc400483377427eb7ce742ce491e5f43b24c6dd00ce46189",
    "extension_2026-10-31/E0_summary_s050.json": "12238bd66cdb9f231c781142b1ef5b9a912b7d144fb3d40825668393e8d89ca1",
    "tier2_2026-11-01/posthoc_2026-11-02/t2_posthoc_v1_v5.json": "1aa4ecda455e2eb0b3ed379fba72676f8d51ab160cea65ccdd5d5d28b286c63e",
}
TABLE_ORDER = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left", "ring_side_right",
               "ring_rear_left", "ring_rear_right"]
HYP_LABEL = {"start": "Sweep start", "A": "LiDAR A", "B": "LiDAR B", "mid": "Mid-sweep"}
METHOD_LABEL = {"cr1": "CR1", "cr2_bm": "CR2-BM", "bootstrap": "Bootstrap"}


# ---------- V5S-1 ----------
def floor_metres(iv, budgets, cams, methods, speed=SPEED_MPS) -> dict:
    """iv: camera -> hypothesis -> method -> {lo, ...}. Per camera (smallest lower end over `methods` of delta_start,
    minus B_c) x speed, in m; the floor and its camera."""
    per = {c: (min(float(iv[c]["start"][m]["lo"]) for m in methods) - float(budgets[c])) * speed / 1000.0 for c in cams}
    cam = min(per, key=per.get)
    return {"per_camera_m": per, "floor_m": per[cam], "binding_camera": cam}


# ---------- V5S-2 ----------
def realised_hw(by_h, a, b, methods) -> float:
    return float(max((float(by_h[h][m]["hi"]) - float(by_h[h][m]["lo"])) / 2.0 for h in (a, b) for m in methods))


def realised_decidability(th, budgets, hw, contrasts=CONTRASTS, margin=MARGIN_MS) -> dict:
    """th: run -> camera -> hypothesis -> theta-bar; hw: run -> camera -> 'a-b' -> realised half-width. A camera clears a
    contrast if |theta_a - theta_b| > margin + B_c + hw; the contrast is decidable if two cameras clear in every run."""
    keys = [f"{a}-{b}" for a, b in contrasts]
    cams, runs = {}, {}
    for tag in th:
        cams[tag] = {}
        for c in th[tag]:
            if c not in hw[tag]:
                continue
            cams[tag][c] = {}
            for a, b in contrasts:
                k = f"{a}-{b}"
                gap = abs(float(th[tag][c][a]) - float(th[tag][c][b]))
                thr = margin + float(budgets[c]) + float(hw[tag][c][k])
                cams[tag][c][k] = {"gap": gap, "threshold_realised": thr, "decidable": bool(gap > thr)}
        runs[tag] = {k: sum(cams[tag][c][k]["decidable"] for c in cams[tag]) >= 2 for k in keys}
    return {"cameras": cams, "runs": runs, "decidable": {k: all(r[k] for r in runs.values()) for k in keys}}


# ---------- V5S-3 ----------
def screen_support(written, rejecting_by_run, limit=SCREEN_LIMIT_PX) -> dict:
    passing = [c for c, v in written.items() if float(v["median_px"]) < limit]
    rp = {tag: [c for c in rej if c in passing] for tag, rej in rejecting_by_run.items()}
    return {"passing": passing, "rejecting_and_passing": rp,
            "rests_on_two_passing_cameras_in_both_runs": all(len(v) >= 2 for v in rp.values())}


# ---------- V5S-4 ----------
def judge_stored(cells, budgets, cams, hyps, method, runs=RUNS, margin=MARGIN_MS) -> dict:
    """The frozen additive rule on stored intervals (cells: run -> camera -> hypothesis -> method -> {est, lo, hi})."""
    per = {}
    for tag in runs:
        iv = {h: {c: (float(cells[tag][c][h][method]["est"]), float(cells[tag][c][h][method]["lo"]),
                      float(cells[tag][c][h][method]["hi"])) for c in cams if c in cells[tag]} for h in hyps}
        per[tag] = decide_rejection(iv, budgets, cams, margin=margin)
    return {"runs": per, "final": combine_runs(per[runs[0]], per[runs[1]])}


# ---------- V5S-5 ----------
def min_clearance(iv_by_run, cam, h, band, methods, runs=RUNS) -> float:
    """Smallest clearance of the rejecting side over runs and methods: max(lo - band, -hi - band)."""
    vals = []
    for tag in runs:
        for m in methods:
            d = iv_by_run[tag][cam][h][m]
            vals.append(max(float(d["lo"]) - band, -float(d["hi"]) - band))
    return float(min(vals))


def _num(x) -> str:
    s = f"{float(x):.1f}"
    if s in ("-0.0", "0.0"):
        return "0.0"
    return s.replace("-", "−")


def ci_text(d) -> str:
    return f"{_num(d['est'])} [{_num(d['lo'])}, {_num(d['hi'])}]"


def full_table_md(iv_by_run, rej_by_run, budgets) -> str:
    lines = ["| Camera | 5 + B_c | Convention | Run | CR1 | CR2-BM | Bootstrap | Three-method rejection |",
             "|---|---|---|---|---|---|---|---|"]
    for c in TABLE_ORDER:
        if c not in iv_by_run["primary"]:
            continue
        for h in ("start", "A", "B", "mid"):
            for tag in RUNS:
                d = iv_by_run[tag][c][h]
                lines.append(f"| {c.replace('ring_', '').replace('_', ' ')} | {_num(MARGIN_MS + budgets[c])} | {HYP_LABEL[h]} | "
                             f"{'primary' if tag == 'primary' else 'margin excluded'} | {ci_text(d['cr1'])} | "
                             f"[{_num(d['cr2_bm']['lo'])}, {_num(d['cr2_bm']['hi'])}] | "
                             f"[{_num(d['bootstrap']['lo'])}, {_num(d['bootstrap']['hi'])}] | "
                             f"{'yes' if c in rej_by_run[tag][h] else 'no'} |")
    return "\n".join(lines) + "\n"


# ---------- the run ----------
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def run(cycle) -> dict:
    for k, want in INPUTS.items():
        got = sha(os.path.join(cycle, k))
        if got != want:
            raise SystemExit(f"{k}: SHA-256 {got} differs from the pin {want}; stop")
    load = lambda k: json.load(open(os.path.join(cycle, k), encoding="utf-8"))
    t3 = load(f"{T3}/cluster_results/results/T3_RESULTS_2026-11-05.json")["primary"]
    s3 = load(f"{T3}/cluster_results/derived/t3_static/S3_summary_s050.json")
    r4 = load("tier2_2026-11-01/posthoc_2026-11-03/r4_results.json")["R4_1_WOD_detector"]
    e0 = load("extension_2026-10-31/E0_summary_s050.json")
    v1 = load("tier2_2026-11-01/posthoc_2026-11-02/t2_posthoc_v1_v5.json")["V1_tolerance_composition"]["WOD_detector_primary_run_only"]

    iv = t3["three_method"]["intervals"]
    cams = [c for c in t3["cameras"] if c not in t3["t2_block"]["timing_model_failures"]]
    budgets = {c: float(t3["budgets_ms"][c]) for c in cams}
    rej = {tag: {h: list(v["rejected_in"]) for h, v in t3["three_method"]["three"]["runs"][tag].items()} for tag in RUNS}
    out = {"pin": "pilot/PILOT_RECORD_XF-C7.md, entry '2026-11-05, post hoc set V5S' (record SHA-256 after the pin: "
                  + PIN_RECORD_SHA + ")", "inputs_sha256": dict(INPUTS), "cameras": cams, "budgets_ms": budgets}

    # V5S-1
    four = rej["primary"]["start"]
    cr1 = floor_metres(iv["primary"], budgets, four, ("cr1",))
    if round(cr1["floor_m"], 2) != 0.21:
        raise SystemExit(f"the published 0.21 m (CR1) is not reproduced: {cr1['floor_m']}; stop")
    out["V5S_1"] = {"cameras": four, "cr1_primary": cr1, "three_methods_primary": floor_metres(iv["primary"], budgets, four, METHODS),
                    "three_methods_margin": floor_metres(iv["margin_flag_excluded"], budgets, four, METHODS)}

    # V5S-2
    th = {"primary": {c: s3["cameras"][c]["predicted_theta_bar"] for c in cams},
          "margin_flag_excluded": {c: t3["design"]["margin_flag_excluded"][c]["theta_bar"] for c in cams}}
    hw = {tag: {c: {f"{a}-{b}": realised_hw(iv[tag][c], a, b, METHODS) for a, b in CONTRASTS} for c in cams} for tag in RUNS}
    out["V5S_2"] = {"realised": realised_decidability(th, budgets, hw), "stored_hw_bm": t3["decidable_bm"]["after_timing"],
                    "realised_hw_ms": hw}

    # V5S-3
    by_rule = s3["S1_by_rule"]
    out["V5S_3"] = {"rule_in_force": s3["S1_rule"], "written_all_range": by_rule["written"], "far40": by_rule["far40"],
                    "support": screen_support(by_rule["written"], {tag: rej[tag]["start"] for tag in RUNS})}

    # V5S-4: reproduction first, then quadrature
    wcams, cells = list(r4["cameras"]), r4["cells"]
    hyps = list(cells["primary"][wcams[0]])
    planned = judge_stored(cells, {c: float(r4["budgets_ms"][c]) for c in wcams}, wcams, hyps, "cr1")
    for tag in RUNS:
        for h in hyps:
            if planned["runs"][tag][h]["rejected_in"] != r4["decisions"]["cr1"]["runs"][tag][h]["rejected_in"]:
                raise SystemExit(f"R4 WOD detector CR1 {tag} {h} not reproduced; stop")
    comp = composition_budgets({c: e0["detector"]["budget"][c] for c in wcams})
    rss = {c: comp[c]["rss"] for c in wcams}
    q = {m: judge_stored(cells, rss, wcams, hyps, m) for m in METHODS}
    for h in hyps:
        if q["cr1"]["runs"]["primary"][h]["rejected_in"] != v1["rejections"]["rss"][h]["rejected_in"]:
            raise SystemExit(f"stored V1 quadrature primary-run rejections not reproduced for {h}; stop")
    out["V5S_4"] = {"cameras": wcams, "rss_budgets_ms": rss, "planned_reproduced": True, "quadrature": q, "governs": "cr1"}

    # V5S-5
    bold = {h: rej["primary"][h] for h in ("start", "A", "B", "mid")}
    out["V5S_5"] = {"bold_cells_primary": bold,
                    "min_clearance_ms": {h: {c: min_clearance(iv, c, h, MARGIN_MS + budgets[c], METHODS) for c in cs}
                                         for h, cs in bold.items()}}
    out["_tables"] = {"full_md": full_table_md(iv, rej, budgets)}
    return out


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 64
    cycle, out_dir = argv[1:3]
    out_path = os.path.join(out_dir, "v5s_results.json")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; the run is done once and never overwritten; stop")
    res = run(cycle)
    tables = res.pop("_tables")
    res["run_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    tpath = os.path.join(out_dir, "V5S_TABLE_AV2_THREE_METHODS.md")
    with open(tpath, "x", encoding="utf-8", newline="\n") as f:
        f.write("# AV2 replication: all three intervals in both runs (post hoc set V5S-5, display values)\n\n"
                "Generated from `T3_RESULTS_2026-11-05.json` (`e28f472d…`); δ_H in ms, 95% intervals.\n\n" + tables["full_md"])
    print("results:", out_path, "sha256", sha(out_path))
    print("table:", tpath, "sha256", sha(tpath))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
