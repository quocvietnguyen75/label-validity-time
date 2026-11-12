"""Transcribe the T3 result file into markdown tables (SAMPLE_PIN addendum 15). Written after the run, test-first.

It copies verdicts, rejection sets, failure labels and readings from the file and never recomputes them.
Computed here, and marked where they appear:
- the S-f secondaries' band marks (the file holds their intervals only);
- the half-width ratios;
- consistency checks that stop the tool: each three-method rejection set is the intersection of the
  per-method sets, the per-method sets agree between the file's two copies of them, the point estimate is the
  same under all three methods, and S-d's pass flag agrees with its cells.
It also compares the runner's design rows with the frozen protocol's three design tables (sections 9 and 11).

Run from this folder:
  python -I -X utf8 t3_results_tables.py RESULTS.json PROTOCOL.md OUT.md [LABEL_SHARES.json ...]
"""
import hashlib
import json
import os
import sys

HYP = ("start", "A", "B", "mid")
METHODS = ("cr1", "cr2_bm", "bootstrap")
RUNS = ("primary", "margin_flag_excluded")
CONTRASTS = ("start-A", "start-B", "start-mid", "A-B", "A-mid", "B-mid")
BLOCKS = ("primary", "sensitivity_score_0_3", "t2_h100_with_t2_frozen_budgets", "t2_h100_recomputed",
          "t2_h100_recomputed_score_0_3", "pooled_457")
PROTOCOL_TABLE_OF = (("primary", 0), ("sensitivity_score_0_3", 1), ("pooled_457", 2))
MINUS = "−"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def num(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}".replace("-", MINUS)


def interval(d, nd=1):
    return f"[{num(d['lo'], nd)}, {num(d['hi'], nd)}]"


def ci(d, nd=1):
    return f"{num(d['est'], nd)} {interval(d, nd)}"


def outside(d, band):
    return d["lo"] > band or d["hi"] < -band


def finals(final):
    return "; ".join(f"{h}: {final[h]}" for h in HYP)


def final_of(x):
    return x["final"] if "final" in x else x["final_descriptive"]


def names(cams):
    return ", ".join(cams) or "none"


def write_once(path, text):
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def check_consistency(name, b):
    tm = b["three_method"]
    for run in RUNS:
        for h in HYP:
            per = {m: tm["by_method"][m]["runs"][run][h]["rejected_in"] for m in METHODS}
            inner = tm["three"]["runs"][run][h].get("rejected_in_by_method")
            if inner is not None and {m: sorted(inner[m]) for m in METHODS} != {m: sorted(v) for m, v in per.items()}:
                raise SystemExit(f"{name} {run} {h}: the two copies of the per-method rejections differ; stop")
            both = set(per["cr1"]) & set(per["cr2_bm"]) & set(per["bootstrap"])
            if both != set(tm["three"]["runs"][run][h]["rejected_in"]):
                raise SystemExit(f"{name} {run} {h}: three-method set is not the per-method intersection; stop")
            for c in b["cameras"]:
                ests = [tm["intervals"][run][c][h][m]["est"] for m in METHODS]
                if max(ests) - min(ests) > 1e-9:
                    raise SystemExit(f"{name} {run} {c} {h}: point estimates differ between methods; stop")


def rejection_lines(b):
    tm = b["three_method"]
    sources = [("three-method", tm["three"]["runs"])] + [(m, tm["by_method"][m]["runs"]) for m in METHODS]
    out = []
    for label, runs in sources:
        for run in RUNS:
            out.append(f"- {label}, {run}: " + "; ".join(f"{h} {names(runs[run][h]['rejected_in'])}" for h in HYP))
    return "\n".join(out)


def t2_rule_lines(b):
    v = b["t2_block"]["verdicts"]
    out = [f"- CR1 only (T2's frozen rule, after the two-frame check), {run}: "
           + "; ".join(f"{h} {names(v['runs'][run][h]['rejected_in'])}" for h in HYP) for run in RUNS]
    out.append("- CR1 only (T2's frozen rule) final: " + finals(v["final"]))
    out.append(f"- CR1 only undefined intervals: {v['undefined_intervals'] or 'none'}")
    return "\n".join(out)


def estimates_table(b, run, margin=5.0):
    tm = b["three_method"]
    lines = ["| Camera | H | Band ±(5 + B^det) | δ_H | CR1 | CR2-BM | Bootstrap | All three reject |",
             "|---|---|---|---|---|---|---|---|"]
    for c in b["cameras"]:
        band = margin + b["budgets_ms"][c]
        for h in HYP:
            ivs = tm["intervals"][run][c][h]
            cells = []
            for m in METHODS:
                s = interval(ivs[m])
                cells.append(f"**{s}**" if c in tm["by_method"][m]["runs"][run][h]["rejected_in"] else s)
            three = "yes" if c in tm["three"]["runs"][run][h]["rejected_in"] else "no"
            lines.append(f"| {c} | {h} | ±{num(band)} | {num(ivs['cr1']['est'])} | " + " | ".join(cells) + f" | {three} |")
    return "\n".join(lines)


def _versus(exceeds):
    return "no prediction" if exceeds is None else ("above hw^BM" if exceeds else "not above hw^BM")


def compact_table(b, run, margin=5.0):
    tm = b["three_method"]
    lines = ["| Camera | Band ±(5 + B^det) | δ_start | δ_A | δ_B | δ_mid |", "|---|---|---|---|---|---|"]
    for c in b["cameras"]:
        cells = []
        for h in HYP:
            s = ci(tm["intervals"][run][c][h]["cr1"])
            cells.append(f"**{s}**" if c in tm["three"]["runs"][run][h]["rejected_in"] else s)
        lines.append(f"| {c} | {num(margin + b['budgets_ms'][c])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def shares_table(s):
    lines = ["| Camera | Moving rows before the seam rule | Removed by the seam rule | Included | θ_A fallback "
             "| θ_B fallback |", "|---|---|---|---|---|---|"]
    for c, x in s["label_side_shares"].items():
        if c != "_all":
            lines.append(f"| {c} | {x['moving_before_seam_rule']:,} | {x['removed_by_seam_rule']:,} "
                         f"({pct(x['share_removed'])}) | {x['included']:,} | {pct(x['share_theta_A_fallback'])} "
                         f"| {pct(x['share_theta_B_fallback'])} |")
    a = s["label_side_shares"]["_all"]
    lines.append(f"| all cameras | {a['moving_before_seam_rule']:,} | {a['removed_by_seam_rule']:,} "
                 f"({pct(a['share_removed'])}) | {a['moving_before_seam_rule'] - a['removed_by_seam_rule']:,} | – | – |")
    return "\n".join(lines)


def failure_table(b, run):
    fl = b["failure_labels"][run]
    lines = ["| Camera | H | Contrasts this camera clears | All three | CR1 (realised hw) | CR2-BM (realised hw) "
             "| Bootstrap (realised hw) | Predicted hw^BM |", "|---|---|---|---|---|---|---|---|"]
    for c in b["cameras"]:
        for h in HYP:
            x = fl[c][h]
            clears = [k for k, v in x["camera_clears"].items() if v]
            cells = [f"{y['label']} ({num(y['realised_hw_ms'])}, {_versus(y['realised_exceeds_hw_bm'])})"
                     for y in (x["by_method"][m] for m in METHODS)]
            lines.append(f"| {c} | {h} | {', '.join(clears) or 'none'} | {x['three']} | " + " | ".join(cells)
                         + f" | {num(x['predicted_hw_bm_ms'])} |")
    return "\n".join(lines)


def halfwidth_table(b):
    lines = ["| Camera | Predicted hw (t_{G−1}) | Predicted hw^BM | Realised CR1 | Realised CR2-BM | Realised bootstrap "
             "| CR1 ÷ t_{G−1} prediction | CR2-BM ÷ hw^BM |", "|---|---|---|---|---|---|---|---|"]
    for c in b["cameras"]:
        x = b["halfwidths"][c]
        r = x["realised_start_ms"]
        lines.append(f"| {c} | {num(x['predicted_t_G_minus_1_ms'])} | {num(x['predicted_bm_ms'])} | {num(r['cr1'])} "
                     f"| {num(r['cr2_bm'])} | {num(r['bootstrap'])} | {num(r['cr1'] / x['predicted_t_G_minus_1_ms'], 2)}× "
                     f"| {num(r['cr2_bm'] / x['predicted_bm_ms'], 2)}× |")
    return "\n".join(lines)


def two_frame_table(b):
    t2 = b["t2_block"]
    lines = ["| Camera | A-synced minus B-synced θ̄, CR1 | Logs | Failure |", "|---|---|---|---|"]
    for c in b["cameras"]:
        x = t2["two_frame"][c]
        lines.append(f"| {c} | {num(x['diff_ms'])} [{num(x['lo'])}, {num(x['hi'])}] | {x['G']} "
                     f"| {'yes' if x['timing_model_failure'] else 'no'} |")
    lines.append("")
    lines.append(f"- failing cameras: {names(t2['timing_model_failures'])}; timing model rejected: {t2['timing_model_rejected']}")
    return "\n".join(lines)


def design_rows(b):
    rows = []
    for c in b["cameras"]:
        de = b["design"]["primary"][c]
        cap = b["gate_bm"]["primary"]["capacity"]["cameras"][c]
        gaps = [f"**{num(cap[k]['gap'])}**" if cap[k]["decidable"] else num(cap[k]["gap"]) for k in CONTRASTS]
        rows.append(
            f"| {c} | {de['tracks']} ({de['logs']}) | {num(b['budgets_ms'][c])} "
            f"| {num(b['halfwidths'][c]['predicted_t_G_minus_1_ms'])} | {num(de['nu_bm'])} "
            f"| {num(b['gate_bm']['primary']['hw_bm_ms'][c])} | {num(de['kish_logs'])} | {de['top_log_share']:.0%} "
            f"| {de['top5_track_share']:.0%} | {num(b['gate_bm']['primary']['thresholds_ms'][c])} "
            f"[{num(b['gate_bm']['margin_flag_excluded']['thresholds_ms'][c])}] | " + " | ".join(gaps) + " |")
    return rows


DESIGN_HEAD = ("| Camera | Tracks (logs) | B^det | hw (t_{G−1}) | ν_BM | hw^BM | Kish logs | Top log | Top 5 tracks "
               "| Threshold [excluded run] | start–A | start–B | start–mid | A–B | A–mid | B–mid |\n"
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def protocol_tables(text):
    tables, cur, skip = [], None, False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("| Camera | Tracks (logs) |"):
            cur, skip = [], True
            tables.append(cur)
            continue
        if cur is not None:
            if skip:
                skip = False
                continue
            if s.startswith("|"):
                cur.append(s)
            else:
                cur = None
    return tables


def decidable_lines(b):
    dec = b["decidable_bm"]
    out = []
    for key in ("all_decision_cameras", "after_timing"):
        for run in RUNS:
            out.append(f"- decidable ({key}, {run}): " + (", ".join(k for k, v in dec[key]["runs"][run].items() if v) or "none"))
        out.append(f"- decidable ({key}, both runs): " + (", ".join(k for k, v in dec[key]["decidable"].items() if v) or "none"))
    out.append("- reading: " + json.dumps(b["reading"], ensure_ascii=False))
    return "\n".join(out)


def estimate_lines(b):
    t2 = b["t2_block"]
    pi, wi = t2["pi"], t2["pi"]["with_intercept_check"]
    c2, c2b = t2["C2_pooled_theta_bar"], b["C2_pooled_cr2_bm"]
    s, r = t2["C3_speed"], t2["C3_inv_sdot"]
    out = ["- θ̄ per camera (CR1): " + "; ".join(f"{c} {ci(t2['theta_bar_by_camera'][c])}" for c in b["cameras"]),
           f"- π̂ = {num(pi['pi_hat'], 3)} [{num(pi['pi_lo'], 3)}, {num(pi['pi_hi'], 3)}], G = {pi['G']}. "
           f"With-intercept check: π {num(wi['pi'], 3)} [{num(wi['pi_lo'], 3)}, {num(wi['pi_hi'], 3)}], "
           f"intercept {num(wi['intercept_ms'])} [{num(wi['intercept_lo'])}, {num(wi['intercept_hi'])}] ms",
           f"- C2: pooled θ̄ = {ci(c2)} (CR1, G = {c2['G']}); CR2-BM {interval(c2b)} (df {num(c2b['df'])}, "
           f"Kish G {num(c2b['kish_G'])}); sign {t2['C2_sign']}",
           f"- C3 on speed: slope {num(s['slope'], 2)} [{num(s['slope_lo'], 2)}, {num(s['slope_hi'], 2)}] ms per m/s, "
           f"intercept {num(s['intercept'])} [{num(s['intercept_lo'])}, {num(s['intercept_hi'])}], G = {s['G']}",
           f"- C3 on 1/ṡ: slope {num(r['slope'], 2)} [{num(r['slope_lo'], 2)}, {num(r['slope_hi'], 2)}], "
           f"intercept {num(r['intercept'])} [{num(r['intercept_lo'])}, {num(r['intercept_hi'])}], G = {r['G']}",
           "- C3 per camera (speed slope, ms per m/s): " + "; ".join(
               f"{c} {num(x['slope'], 2)} [{num(x['slope_lo'], 2)}, {num(x['slope_hi'], 2)}]"
               for c, x in ((c, t2["C3_speed_by_camera"][c]) for c in b["cameras"]))]
    return "\n".join(out)


def secondaries_table(b, margin=5.0):
    sec = b["t2_block"]["secondary"]
    lines = ["| Camera | Band | Jackknife δ_start | δ_A | δ_B | δ_mid | Per-log intercept δ_start (logs) | δ_A | δ_B | δ_mid |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in b["cameras"]:
        band = margin + b["budgets_ms"][c]

        def mark(d):
            return f"{ci(d)} †" if outside(d, band) else ci(d)
        jk = [mark(sec["jackknife"][c][h]) for h in HYP]
        si = [mark(sec["segment_intercept"][c][h]) for h in HYP]
        si[0] += f" ({sec['segment_intercept'][c]['start']['G']})"
        lines.append(f"| {c} | ±{num(band)} | " + " | ".join(jk + si) + " |")
    lines.append("")
    lines.append("† the interval lies entirely outside the band (marked by this tool; S-f changes no verdict).")
    return "\n".join(lines)


def composition_lines(b):
    out = []
    for comp, cb in b.get("compositions", {}).items():
        tm = cb["verdicts"]["three"]
        cams = list(cb["budgets_ms"])
        out.append(f"#### B_{comp}")
        out.append("- budgets B (ms): " + "; ".join(f"{c} {num(cb['budgets_ms'][c])}" for c in cams))
        out.append("- hw^BM thresholds, primary [excluded run] (ms): " + "; ".join(
            f"{c} {num(cb['gates']['primary']['thresholds_ms'][c])} [{num(cb['gates']['margin_flag_excluded']['thresholds_ms'][c])}]"
            for c in cams))
        for run in RUNS:
            out.append(f"- decidable ({run}): " + (", ".join(k for k, v in cb["decidable_bm"]["runs"][run].items() if v) or "none"))
        out.append("- decidable (both runs): " + (", ".join(k for k, v in cb["decidable_bm"]["decidable"].items() if v) or "none"))
        for run in RUNS:
            out.append(f"- three-method, {run}: " + "; ".join(f"{h} {names(tm['runs'][run][h]['rejected_in'])}" for h in HYP))
        out.append("- final: " + finals(tm["final"]))
        out.append("- reading: " + json.dumps(cb["reading"], ensure_ascii=False))
        out.append("")
    return "\n".join(out)


def e4_table(e4, cams):
    shifts = list(e4["runs"])
    lines = ["| Camera | Baseline δ_start | " + " | ".join(f"Δ = {s.replace('-', MINUS)}" for s in shifts) + " |",
             "|---|---|" + "---|" * len(shifts)]
    any_out = False
    for c in cams:
        cells = []
        for s in shifts:
            x = e4["runs"][s]["cameras"].get(c)
            if x is None:
                cells.append("missing")
                any_out = True
                continue
            cell = f"{num(100 * x['recovery_ratio'])}%"
            if not 0.9 <= x["recovery_ratio"] <= 1.1:
                cell, any_out = f"**{cell}**", True
            cells.append(f"{cell} ({num(x['recovered_shift_ms'])})")
        lines.append(f"| {c} | {num(e4['baseline_delta_start_ms'][c])} | " + " | ".join(cells) + " |")
    missing = {s: e4["runs"][s]["missing_cameras"] for s in shifts if e4["runs"][s]["missing_cameras"]}
    if (not any_out and not missing) != e4["pass_band_90_110"]:
        raise SystemExit("S-d: the file's pass flag disagrees with its cells; stop")
    lines.append("")
    lines.append("Cells: recovered share of Δ (recovered shift, ms). **Bold**: outside 90–110%.")
    lines.append("- changed assignments: " + "; ".join(
        f"Δ = {s.replace('-', MINUS)} {num(100 * e4['runs'][s]['assignment_changed_share'], 2)}%" for s in shifts))
    lines.append(f"- missing cameras: {missing or 'none'}")
    lines.append(f"- baseline reproduced: {e4['baseline_reproduced']}; changed share at zero shift: "
                 f"{e4['zero_shift_changed_share']}; pass band 90–110% in every cell (file): {e4['pass_band_90_110']}")
    return "\n".join(lines)


def verdict_section(b):
    tm = b["three_method"]
    out = ["- three-method final: " + finals(final_of(tm["three"]))]
    out += [f"- {m} alone, final: " + finals(final_of(tm["by_method"][m])) for m in METHODS]
    out.append(rejection_lines(b))
    out.append(f"- undefined intervals: {tm['undefined_intervals'] or 'none'}; cameras without rows: "
               f"{names(b['cameras_without_rows'])}")
    if "t2_block" in b:
        out.append(t2_rule_lines(b))
    return "\n".join(out)


def block_section(name, b, margin):
    check_consistency(name, b)
    out = [f"## {name}", "", f"Role: {b['role']}" if "role" in b else "", "",
           f"Decision cameras: {names(b['cameras'])}. Budgets B^det (ms): "
           + "; ".join(f"{c} {num(b['budgets_ms'][c])}" for c in b["cameras"]), "",
           "### Verdicts", "", verdict_section(b), "", decidable_lines(b), ""]
    for run in RUNS:
        out += [f"### Compact table, {run} run", "", "CR1 intervals. Bold: all three methods reject (the file's "
                "three-method list).", "", compact_table(b, run, margin), ""]
    for run in RUNS:
        out += [f"### Estimates, {run} run", "", "Bold: that method's interval lies outside the band (the file's "
                "per-method rejection list).", "", estimates_table(b, run, margin), ""]
    for run in RUNS:
        out += [f"### Failure labels, {run} run", "", failure_table(b, run), ""]
    out += ["### Half-widths of δ_start (S-h)", "", halfwidth_table(b), "", "### Design rows (from the run)", "",
            DESIGN_HEAD, "\n".join(design_rows(b)), ""]
    if "t2_block" in b:
        out += ["### Two-frame check", "", two_frame_table(b), "", "### θ̄, π̂, C2, C3", "", estimate_lines(b), "",
                "### S-f secondaries (CR1)", "", secondaries_table(b, margin), ""]
    if b.get("compositions"):
        out += ["### S-a compositions", "", composition_lines(b)]
    return "\n".join(out)


def main(argv):
    if len(argv) < 4:
        raise SystemExit(__doc__)
    res_path, proto_path, out_path = argv[1:4]
    share_paths = argv[4:]
    with open(res_path, encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("protocol_sha256") != sha(proto_path):
        raise SystemExit("the protocol given is not the one the run used; stop")
    with open(proto_path, encoding="utf-8") as fh:
        tables = protocol_tables(fh.read())
    margin = d["settings"]["margin_ms"]
    md = ["# T3 result tables (generated)", "",
          f"- Source: `{os.path.basename(res_path)}` SHA-256 `{sha(res_path)}`; run {d['run_utc']}.",
          f"- Config SHA-256 `{d['config_sha256']}`; protocol `{os.path.basename(proto_path)}` SHA-256 `{d['protocol_sha256']}`.",
          f"- Generated by `t3_results_tables.py` SHA-256 `{sha(os.path.abspath(__file__))}`.",
          f"- Settings: {json.dumps(d['settings'], ensure_ascii=False)}.", "",
          "## Reproduction (protocol §15)", "",
          f"- T2's CR1 finals: {finals(d['reproduction']['t2_cr1_final'])}",
          f"- T2's three-method finals (R4): {finals(d['reproduction']['t2_three_method_final'])}",
          f"- R4 intervals reproduced within {d['reproduction']['r4_intervals_within_ms']} ms", "",
          "## Design rows against the frozen protocol's tables (§9, §11)", ""]
    for name, idx in PROTOCOL_TABLE_OF:
        gen = design_rows(d[name])
        same = idx < len(tables) and gen == tables[idx]
        md.append(f"- {name} against protocol table {idx + 1}: {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            prot = tables[idx] if idx < len(tables) else []
            for i, row in enumerate(gen):
                if i >= len(prot) or row != prot[i]:
                    md.append(f"  - run:      {row}")
                    md.append(f"  - protocol: {prot[i] if i < len(prot) else '(missing)'}")
    md.append("")
    for name in BLOCKS:
        md.append(block_section(name, d[name], margin))
    e4 = d["primary"]["e4"]
    md += ["## S-d injected-shift recovery (primary)", "", e4_table(e4, d["primary"]["cameras"]), ""]
    for g, b in d["primary"]["subgroups"].items():
        md.append(block_section(f"S-g subgroup: {g} (descriptive)", b, margin))
    for p in share_paths:
        with open(p, encoding="utf-8") as fh:
            s = json.load(fh)
        md += [f"## Label-side seam-rule and fallback shares: `{os.path.basename(p)}`", "",
               f"- File SHA-256 `{sha(p)}`; label side `{s['input']}` SHA-256 `{s['input_sha256']}`, {s['rows']:,} rows.",
               "", shares_table(s), ""]
    text = "\n".join(md) + "\n"
    write_once(out_path, text)
    print("\n".join(ln for ln in md[:24] if ln))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
