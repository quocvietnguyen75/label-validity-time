"""Table I's AV2 rows and the derived AV2 numbers of the v5 letter, from the T3 result file. Written test-first.

Rejection marks are copied from the file's three-method lists (primary run). A dagger marks a camera that clears no
contrast in the predicted-capacity gate (the file's gate_bm, primary run). Computed here, as arithmetic on the
reported intervals:
- clearances: for each camera that rejects the sweep start in a run, the smallest distance by which its three
  intervals clear the band (lowest lower end minus 5 + B_c). A shift of every estimate toward the sweep start
  leaves fewer than two rejecting cameras in a run once it exceeds that run's second-largest clearance; the result
  needs both runs, so the shift that removes it is the smaller of the two runs' thresholds;
- metres at a given speed: the CR1 interval, and the lowest of the three lower ends minus B_c;
- per camera, the largest difference between CR2-BM's or the bootstrap's interval ends and CR1's.

Usage (from this folder): python -I -X utf8 av2_t3_values.py T3_RESULTS.json OUT.json
"""
import hashlib
import json
import sys

PINNED = "e28f472dca12f9438bc9d7fe7bc4cd95d3cbe0ff488a1dcfd388b8ae94d6f3dc"
HYP = ("start", "A", "B", "mid")
METHODS = ("cr1", "cr2_bm", "bootstrap")
RUNS = ("primary", "margin_flag_excluded")
RING = ("ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left", "ring_side_right",
        "ring_rear_left", "ring_rear_right")
NAMES = {"ring_front_center": "Front center", "ring_front_left": "Front left", "ring_front_right": "Front right",
         "ring_side_left": "Side left", "ring_side_right": "Side right", "ring_rear_left": "Rear left",
         "ring_rear_right": "Rear right"}
MARGIN_MS = 5.0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tex_num(x, nd=1):
    s = f"{x:.{nd}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def tex_ci(d, nd=1):
    return f"{tex_num(d['est'], nd)} [{tex_num(d['lo'], nd)}, {tex_num(d['hi'], nd)}]"


def table_rows(b, margin=MARGIN_MS):
    tm = b["three_method"]
    cap = b["gate_bm"]["primary"]["capacity"]["cameras"]
    rows = []
    for c in (c for c in RING if c in b["cameras"]):
        de = b["design"]["primary"][c]
        dagger = "" if any(x["decidable"] for x in cap[c].values()) else "$^\\dagger$"
        cells = []
        for h in HYP:
            s = tex_ci(tm["intervals"]["primary"][c][h]["cr1"])
            cells.append(f"\\textbf{{{s}}}" if c in tm["three"]["runs"]["primary"][h]["rejected_in"] else s)
        rows.append(f"{NAMES[c]}{dagger} & {de['tracks']:,} ({de['logs']}; {de['kish_logs']:.1f}) & "
                    + " & ".join(cells) + f" & {margin + b['budgets_ms'][c]:.1f} \\\\")
    return rows


def clearances(b, margin=MARGIN_MS):
    tm = b["three_method"]
    per_run, threshold = {}, {}
    for run in RUNS:
        cl = {}
        for c in tm["three"]["runs"][run]["start"]["rejected_in"]:
            ivs = tm["intervals"][run][c]["start"]
            band = margin + b["budgets_ms"][c]
            if ivs["cr1"]["est"] > 0:
                cl[c] = min(ivs[m]["lo"] for m in METHODS) - band
            else:
                cl[c] = -max(ivs[m]["hi"] for m in METHODS) - band
        per_run[run] = cl
        vals = sorted(cl.values(), reverse=True)
        threshold[run] = vals[1] if len(vals) >= 2 else 0.0
    return {"per_run": per_run, "threshold": threshold, "shift_to_remove_ms": min(threshold.values())}


def metres(b, speed=10.0):
    tm = b["three_method"]
    out = {}
    for c in tm["three"]["runs"]["primary"]["start"]["rejected_in"]:
        ivs = tm["intervals"]["primary"][c]["start"]
        lo_min = min(ivs[m]["lo"] for m in METHODS)
        out[c] = {"cr1_lo_m": ivs["cr1"]["lo"] * speed / 1000, "cr1_hi_m": ivs["cr1"]["hi"] * speed / 1000,
                  "after_budget_min_m": (lo_min - b["budgets_ms"][c]) * speed / 1000}
    return out


def method_spread(b, run="primary"):
    ints = b["three_method"]["intervals"][run]
    return {c: max(max(abs(ints[c][h][m]["lo"] - ints[c][h]["cr1"]["lo"]), abs(ints[c][h][m]["hi"] - ints[c][h]["cr1"]["hi"]))
                   for h in HYP for m in ("cr2_bm", "bootstrap")) for c in b["cameras"]}


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    src, out = argv[1:]
    got = sha(src)
    if got != PINNED:
        raise SystemExit(f"{src}: SHA-256 {got} is not the pinned {PINNED}; stop")
    with open(src, encoding="utf-8") as fh:
        b = json.load(fh)["primary"]
    res = {"source": src, "source_sha256": got, "table_rows": table_rows(b), "clearances_ms": clearances(b),
           "metres_at_10_m_s": metres(b), "method_spread_primary_ms": method_spread(b)}
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print("\n".join(res["table_rows"]))
    print(json.dumps({k: v for k, v in res.items() if k not in ("table_rows", "source")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
