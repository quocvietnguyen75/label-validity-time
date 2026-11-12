"""XF-C7 T3, before the freeze: turn the static-step outputs into the protocol's [S3] values.

Reads design values only: the S3 summaries (budgets, design effects, label-side predicted theta-bar, S1, counts), the
weights-only design tables written by t3_design_table.py, the S0 and label-side summaries, and the MEASURED lines of
the static jobs' logs. It never opens a moving analysis set, so no theta-hat can be formed.

Usage (any of the optional inputs may be left out):
  python -I -X utf8 t3_fill_s3.py --s050 S3.json DESIGN.json --s030 S3.json DESIGN.json
      [--pooled S3.json DESIGN.json] [--s0 s0.json] [--labels labels_summary.json] [--log job.out ...]
      --out values.json --md values.md
"""
import argparse
import hashlib
import json
import re
import sys

CONTRASTS = ("start-A", "start-B", "start-mid", "A-B", "A-mid", "B-mid")
GRID = 100000  # pi step 1e-5 for the mixture band
MEASURED = re.compile(r"^MEASURED module=(\S+) peak_rss_GB=([0-9.]+) wall_s=([0-9.]+) exit=(-?\d+)\s*$", re.M)


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def decision_cameras(s3):
    """As xfc7_t2_static.decision_cameras: >= 30 moving tracks and an S1 pass under the rule in force."""
    s1 = s3["S1_projection_check"]
    return sorted(c for c in s3["cameras_with_30_tracks"] if s1.get(c, {}).get("pass", False))


def camera_rows(s3, design, run="primary", comp="max"):
    """One row per decision camera of the design table, in its order, under the hw_BM gate of `comp`."""
    gate = design["gates"][run][comp]
    excl = design["gates"]["margin_flag_excluded"][comp]["thresholds_ms"]
    t_old = design["gates"]["primary"].get("t_G_minus_1_thresholds_ms") or {}
    rows = []
    for c in design["decision_cameras"]:
        if c not in gate["thresholds_ms"]:
            continue
        d = design["design"][run][c]
        cap = gate["capacity"]["cameras"][c]
        rows.append({
            "camera": c, "tracks": d["tracks"], "logs": d["logs"],
            "B_c_ms": s3["budget"][c]["B_c_ms"],
            "hw_t_ms": s3["cameras"][c]["predicted_halfwidth_ms"],
            "threshold_t_ms": t_old.get(c),
            "nu_bm": d["nu_bm"], "hw_bm_ms": gate["hw_bm_ms"][c],
            "kish_logs": d["kish_logs"], "top_log_share": d["top_log_share"], "top5_track_share": d["top5_track_share"],
            "threshold_ms": gate["thresholds_ms"][c], "threshold_excluded_ms": excl.get(c),
            "gaps": {k: cap[k]["gap"] for k in CONTRASTS},
            "over": {k: bool(cap[k]["decidable"]) for k in CONTRASTS},
            "DE": s3["cameras"][c]["DE"],
            "two_frame_hw_ms": s3["cameras"][c]["predicted_halfwidth_two_frame_diff_ms"],
        })
    return rows


def decidable(design, run="primary", comp="max"):
    ds = design["gates"][run][comp]["capacity"]["dataset"]
    return [k for k in CONTRASTS if ds.get(k)]


def gate_failed(dataset):
    """Protocol header: the gate fails if no contrast is decidable in two cameras."""
    return not any(bool(v) for v in dataset.values())


def decidable_both_runs(design, comp="max"):
    """T3 section 9: a contrast is decidable for T3 only if two decision cameras clear its hw_BM threshold in both the
    primary and the margin-flag-excluded run (xfc7_t3_run.decidable_both)."""
    p = design["gates"]["primary"][comp]["capacity"]["dataset"]
    e = design["gates"]["margin_flag_excluded"][comp]["capacity"]["dataset"]
    return [k for k in CONTRASTS if p.get(k) and e.get(k)]


def _intervals(pis, mask):
    out, start = [], None
    for p, m in zip(pis, mask):
        if m and start is None:
            start = p
        if not m and start is not None:
            out.append([start, prev])
            start = None
        prev = p
    if start is not None:
        out.append([start, prev])
    return out


def mixture_band(theta, thresholds, grid=GRID):
    """T2 section 9, 'Against the mixture': a per-box mixture with A-share pi predicts a gap to H_start of
    |pi*theta_A + (1 - pi)*theta_B - theta_start|. A camera rejects H_start where that gap exceeds its threshold.
    Returns each camera's rejection set, the band of pi where fewer than two cameras reject, and the largest
    |delta_A| = (1 - pi)|theta_B - theta_A| inside that band per camera."""
    cams = sorted(c for c in thresholds if c in theta)
    pis = [i / grid for i in range(grid + 1)]
    rej = {}
    for c in cams:
        a, b, s = theta[c]["A"], theta[c]["B"], theta[c]["start"]
        rej[c] = [abs(p * a + (1 - p) * b - s) > thresholds[c] for p in pis]
    count = [sum(rej[c][i] for c in cams) for i in range(len(pis))]
    band_mask = [n < 2 for n in count]
    band = _intervals(pis, band_mask)
    in_band = [p for p, m in zip(pis, band_mask) if m]
    dmax = {}
    for c in cams:
        a, b = theta[c]["A"], theta[c]["B"]
        dmax[c] = max(((1 - p) * abs(b - a) for p in in_band), default=None)
    return {"per_camera": {c: _intervals(pis, rej[c]) for c in cams}, "band": band, "max_abs_delta_A_in_band": dmax}


def parse_measured(text):
    return [{"module": m.group(1), "peak_rss_GB": float(m.group(2)), "wall_s": float(m.group(3)),
             "exit": int(m.group(4))} for m in MEASURED.finditer(text)]


def _f(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def table_markdown(rows):
    head = ("| Camera | Tracks (logs) | B^det | hw (t_{G−1}) | ν_BM | hw^BM | Kish logs | Top log | Top 5 tracks "
            "| Threshold [excluded run] | " + " | ".join(k.replace("-", "–") for k in CONTRASTS) + " |")
    lines = [head, "|" + "---|" * (10 + len(CONTRASTS))]
    for r in rows:
        gaps = [f"**{r['gaps'][k]:.1f}**" if r["over"][k] else f"{r['gaps'][k]:.1f}" for k in CONTRASTS]
        lines.append(
            f"| {r['camera']} | {r['tracks']} ({r['logs']}) | {_f(r['B_c_ms'])} | {_f(r['hw_t_ms'])} | {_f(r['nu_bm'])} "
            f"| {_f(r['hw_bm_ms'])} | {_f(r['kish_logs'])} | {r['top_log_share']:.0%} | {r['top5_track_share']:.0%} "
            f"| {_f(r['threshold_ms'])} [{_f(r['threshold_excluded_ms'])}] | " + " | ".join(gaps) + " |")
    return "\n".join(lines)


def summarise(s3, design):
    rows = camera_rows(s3, design)
    th = {c: s3["cameras"][c]["predicted_theta_bar"] for c in design["decision_cameras"]}
    out = {"decision_cameras": design["decision_cameras"],
           "decision_cameras_recomputed": decision_cameras(s3),
           "S1": s3["S1_projection_check"], "S1_rule": s3["S1_rule"],
           "cameras": {c: {k: s3["cameras"][c][k] for k in ("obs", "tracks", "segments", "DE", "predicted_halfwidth_ms",
                                                            "predicted_halfwidth_two_frame_diff_ms")}
                       for c in s3["cameras"]},
           "counts": s3["counts"], "inputs": s3["inputs"], "rows": rows, "gates": {}}
    for run in ("primary", "margin_flag_excluded"):
        for comp in ("max", "rss", "sum"):
            g = design["gates"][run][comp]
            out["gates"][f"{run}/{comp}"] = {"decidable": decidable(design, run, comp),
                                             "gate_failed": gate_failed(g["capacity"]["dataset"]),
                                             "thresholds_ms": g["thresholds_ms"]}
    out["t3_gate"] = {comp: {"decidable": decidable_both_runs(design, comp),
                             "gate_failed": not decidable_both_runs(design, comp)} for comp in ("max", "rss", "sum")}
    excl = design["design"]["margin_flag_excluded"]
    no_tb = [c for c in design["decision_cameras"] if c in excl and "theta_bar" not in excl[c]]
    if no_tb:
        raise SystemExit(f"design table has no per-run theta-bar for {no_tb}: rebuild it with t3_design_table.py "
                         "57e03b4c or later; stop")
    th_run = {"primary": th, "margin_flag_excluded": {c: excl[c]["theta_bar"] for c in design["decision_cameras"] if c in excl}}
    out["mixture"] = {run: mixture_band(th_run[run], design["gates"][run]["max"]["thresholds_ms"])
                      for run in ("primary", "margin_flag_excluded")}
    out["tolerance_ms"] = {c: 5.0 + s3["budget"][c]["B_c_ms"] for c in design["decision_cameras"]}
    return out


def markdown(name, summ):
    lines = [f"### {name}", "", table_markdown(summ["rows"]), ""]
    for comp, g in summ["t3_gate"].items():
        lines.append(f"- **decidable for T3 (both runs), {comp}:** {g['decidable'] or 'none'}; gate failed {g['gate_failed']}")
    for key, g in summ["gates"].items():
        lines.append(f"- {key}: decidable {g['decidable'] or 'none'}; gate failed {g['gate_failed']}")
    for run, m in summ["mixture"].items():
        band = "; ".join(f"{lo:.3f}–{hi:.3f}" for lo, hi in m["band"]) or "none"
        per = "; ".join(f"{c}: " + (", ".join(f"{lo:.3f}–{hi:.3f}" for lo, hi in v) or "none")
                        for c, v in m["per_camera"].items())
        lines.append(f"- mixture ({run}): fewer than two cameras reject H_start for pi in {band}. Per camera: {per}")
        lines.append("  - max |delta_A| in band vs 5 + B^det: " + "; ".join(
            f"{c} {_f(v)} vs {_f(summ['tolerance_ms'][c])}" for c, v in m["max_abs_delta_A_in_band"].items()))
    lines.append("- two-frame predicted half-widths (ms): " + "; ".join(
        f"{r['camera']} {_f(r['two_frame_hw_ms'])}" for r in summ["rows"]))
    lines.append("- t_{G−1} thresholds (ms): " + "; ".join(f"{r['camera']} {_f(r['threshold_t_ms'])}" for r in summ["rows"]))
    lines.append("- S1 (" + summ["S1_rule"] + "): " + "; ".join(
        f"{c} n={v['n']} med={v['median_px']:.1f} p90={v['p90_px']:.1f} {'pass' if v['pass'] else 'FAIL'}"
        for c, v in summ["S1"].items()))
    lines.append("- tracks per camera: " + "; ".join(f"{c} {v['tracks']} ({v['segments']})" for c, v in summ["cameras"].items()))
    return "\n".join(lines)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--s050", nargs=2, required=True)
    ap.add_argument("--s030", nargs=2)
    ap.add_argument("--pooled", nargs=2)
    ap.add_argument("--s0")
    ap.add_argument("--labels")
    ap.add_argument("--log", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args(argv[1:])
    values, md, inputs = {}, [], {}
    for name, pair in (("score_0_5", a.s050), ("score_0_3", a.s030), ("pooled_458_score_0_5", a.pooled)):
        if not pair:
            continue
        s3, des = load(pair[0]), load(pair[1])
        if des["inputs"]["s3_summary_sha256"] != sha(pair[0]):
            raise SystemExit(f"{name}: design table was not built from this S3 summary")
        inputs[name] = {"s3_summary": sha(pair[0]), "design": sha(pair[1])}
        values[name] = summarise(s3, des)
        if values[name]["decision_cameras_recomputed"] != sorted(values[name]["decision_cameras"]):
            raise SystemExit(f"{name}: decision cameras disagree with the S3 summary")
        md.append(markdown(name, values[name]))
    if a.s0:
        inputs["s0"] = sha(a.s0)
        values["s0"] = load(a.s0)
        md.append(f"### S0\n\n- logs {values['s0']['logs']}, max |departure| {values['s0']['max_abs_departure_ms']} ms, "
                  f"departing logs {len(values['s0']['departing_logs'])}")
    if a.labels:
        inputs["labels"] = sha(a.labels)
        values["labels"] = load(a.labels)
        md.append("### Label side\n\n- " + json.dumps({k: v for k, v in values["labels"].items() if k != "file"}))
    runs = []
    for p in a.log:
        with open(p, encoding="utf-8", errors="replace") as fh:
            runs.append({"log": p.replace("\\", "/").rsplit("/", 1)[-1], "measured": parse_measured(fh.read())})
    values["measured"] = runs
    if runs:
        md.append("### Run time\n\n" + "\n".join(f"- {r['log']}: {r['measured']}" for r in runs))
    values["inputs_sha256"] = inputs
    with open(a.out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(values, fh, indent=1, default=float)
    with open(a.md, "x", encoding="utf-8", newline="\n") as fh:
        fh.write("\n\n".join(md) + "\n")
    print("\n\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
