"""XF-C7 V7C post hoc check (plan V7C_PLAN_2026-11-10.md, SHA-256 a275ee48...): how far the AV2 replication's camera
means move moving vehicles' boxes, against the 0.5, 1, 2 and 4 m centre-distance thresholds of AV2's detection metric.

Post hoc and descriptive. It changes no verdict and re-estimates no validity time: it is arithmetic on each stored
observation's label speed and its camera's stored estimate. Reports hold aggregates only.

Usage (PYTHONPATH=src, from the repository root, with relative paths):
  python -m careport.xfc7_v7c <moving_analysis_set_t2.parquet> <T3_RESULTS.json> <av2_t3_sample.txt> <plan.md> <out.json>
"""
import hashlib
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from careport.xfc7_t3_posthoc_v6r import RING, T3_SHA, speed_summary

PLAN_SHA = "a275ee489caa04095c04ddddd74f818d09a5e45093549e42a80170ae6724a57b"
MOVING_SHA = "b6df2c96d5142ac83f33c88ffacce4f97ba8353056d5bd730404920a6d48ec40"
SEED = 20261110
N_BOOT = 2000
THRESHOLDS_M = (0.5, 1.0, 2.0, 4.0)
METHODS = ("cr1", "cr2_bm", "bootstrap")
CODE = {name: i for i, name in enumerate(RING, 1)}   # camera codes of the stored table, in RING order
NAME = {i: name for name, i in CODE.items()}
REJECTING = ("ring_front_left", "ring_side_left", "ring_rear_left", "ring_rear_right")
EXPECTED_ROWS = 40529
# plan section 3: recorded values to reproduce, to 0.1, before any share is computed
SPEED_TARGETS = {"_all": ("8.8", "14.3"), "ring_front_left": ("9.1", "14.3"), "ring_side_left": ("9.5", "14.4"),
                 "ring_rear_left": ("9.6", "14.6"), "ring_rear_right": ("7.2", "13.9")}
MEAN_TARGETS = {"ring_front_left": "52.1", "ring_side_left": "41.1", "ring_rear_left": "56.8", "ring_rear_right": "73.2"}


def check_sha256(path, expected):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise ValueError(f"{path}: SHA-256 {got} differs from the pinned {expected}")
    return got


def split_map(lines):
    """{log id: 'val' or 'train'} from lines 'val/<log>' or 'train/<log>'."""
    out = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        split, _, log = line.partition("/")
        if split not in ("val", "train") or not log:
            raise ValueError(f"unexpected sample line {line!r}")
        if log in out:
            raise ValueError(f"log {log} listed twice")
        out[log] = split
    return out


def displacement(speed, theta_ms):
    """Metres a box moves along the motion if it is taken as valid theta_ms earlier than it is."""
    return speed * theta_ms / 1000.0


def shares(d):
    d = np.asarray(d, dtype=float)
    return {t: float(np.mean(d > t)) for t in THRESHOLDS_M}


def summarize(d):
    d = np.asarray(d, dtype=float)
    return {"n": int(len(d)), "share_gt": {str(t): s for t, s in shares(d).items()},
            "median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90))}


def offsets(cell, b_c):
    """M, L and U (ms) of one camera from its stored three-method cell (plan section 4)."""
    ests = [cell[m]["est"] for m in METHODS]
    if any(not math.isclose(e, ests[0], rel_tol=0.0, abs_tol=1e-9) for e in ests):
        raise ValueError(f"the three methods' estimates differ: {ests}")
    lo = min(cell[m]["lo"] for m in METHODS)
    hi = max(cell[m]["hi"] for m in METHODS)
    return {"M": float(ests[0]), "L": float(max(lo - b_c, 0.0)), "U": float(hi)}


def tracks(df):
    """One row per track (segment, camera, object_id), with the track's median speed."""
    return df.groupby(["segment", "camera", "object_id"], as_index=False)["speed"].median()


def _theta(cameras, offs, key):
    return np.array([offs[int(c)][key] for c in cameras], dtype=float)


def population(df, offs, cameras):
    sub = df[df["camera"].isin(cameras)]
    t = tracks(sub)
    out = {"counts": {"observations": int(len(sub)), "tracks": int(len(t)), "logs": int(sub["segment"].nunique())},
           "observation": {}, "track": {}}
    for key in ("M", "L", "U"):
        out["observation"][key] = summarize(displacement(sub["speed"].to_numpy(float), _theta(sub["camera"], offs, key)))
        out["track"][key] = summarize(displacement(t["speed"].to_numpy(float), _theta(t["camera"], offs, key)))
    return out


def bootstrap_shares(df, offs, seed=SEED, n_boot=N_BOOT, key="M"):
    """95% percentile intervals of the observation shares from a log-cluster bootstrap, offsets held fixed."""
    d = displacement(df["speed"].to_numpy(float), _theta(df["camera"], offs, key))
    logs = df["segment"].to_numpy()
    uniq, inv = np.unique(logs, return_inverse=True)
    n_log = np.bincount(inv, minlength=len(uniq)).astype(float)
    over = np.stack([np.bincount(inv, weights=(d > t).astype(float), minlength=len(uniq)) for t in THRESHOLDS_M], axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    tot = n_log[draws].sum(axis=1)
    out = {}
    for j, t in enumerate(THRESHOLDS_M):
        s = over[draws, j].sum(axis=1) / tot
        lo, hi = np.percentile(s, [2.5, 97.5])
        out[str(t)] = (float(lo), float(hi))
    return out


def validate(df, t3):
    """Plan section 3: reproduce V6R-4's speeds and Table I's camera means; stop if either fails."""
    rows, ok = [], True
    for cam, (med_t, p90_t) in SPEED_TARGETS.items():
        v = df["speed"] if cam == "_all" else df.loc[df["camera"] == CODE[cam], "speed"]
        s = speed_summary(v)
        got = (f"{s['median']:.1f}", f"{s['p90']:.1f}")
        rows.append({"what": f"speed {cam}", "recorded": [med_t, p90_t], "computed": got, "unrounded": [s["median"], s["p90"]]})
        ok &= got == (med_t, p90_t)
    cells = t3["primary"]["three_method"]["intervals"]["primary"]
    for cam, target in MEAN_TARGETS.items():
        est = cells[cam]["start"]["cr1"]["est"]
        rows.append({"what": f"camera mean {cam}", "recorded": target, "computed": f"{est:.1f}", "unrounded": est})
        ok &= f"{est:.1f}" == target
    return ok, rows


def run(parquet, t3_path, sample_path, plan_path, out_path):
    hashes = {"moving_set": check_sha256(parquet, MOVING_SHA), "t3_results": check_sha256(t3_path, T3_SHA),
              "plan": check_sha256(plan_path, PLAN_SHA)}
    with open(sample_path, "rb") as fh:
        hashes["sample_list"] = hashlib.sha256(fh.read()).hexdigest()
    df = pd.read_parquet(parquet, columns=["segment", "camera", "object_id", "speed"])
    if len(df) != EXPECTED_ROWS:
        raise SystemExit(f"{len(df)} rows, expected {EXPECTED_ROWS}")
    with open(t3_path, encoding="utf-8") as fh:
        t3 = json.load(fh)

    ok, val_rows = validate(df, t3)
    if not ok:
        print(json.dumps(val_rows, indent=1))
        raise SystemExit("validation failed (plan section 3): no share computed")

    with open(sample_path, encoding="utf-8") as fh:
        splits = split_map(fh.read().splitlines())
    missing = sorted(set(df["segment"]) - set(splits))
    if missing:
        raise SystemExit(f"{len(missing)} logs of the table are not in the sample list")
    df["split"] = df["segment"].map(splits)

    cells = t3["primary"]["three_method"]["intervals"]["primary"]
    budgets = t3["primary"]["budgets_ms"]
    offs, off_report = {}, {}
    for cam in RING:
        o = offsets(cells[cam]["start"], budgets[cam])
        offs[CODE[cam]] = o
        off_report[cam] = {**o, "B_c": budgets[cam], "rejecting": cam in REJECTING}

    rej = [CODE[c] for c in REJECTING]
    pops = {
        "P1_rejecting_all_logs": population(df, offs, rej),
        "P2_all_seven_descriptive": population(df, offs, list(CODE.values())),
        "P3_rejecting_validation_logs": population(df[df["split"] == "val"], offs, rej),
        "P4_rejecting_training_logs": population(df[df["split"] == "train"], offs, rej),
    }
    p1 = df[df["camera"].isin(rej)]
    boot = bootstrap_shares(p1, offs)

    res = {
        "check": "V7C (post hoc, descriptive)", "plan": "V7C_PLAN_2026-11-10.md", "plan_sha256": PLAN_SHA,
        "inputs": {"moving_set": parquet, "t3_results": t3_path, "sample_list": sample_path},
        "input_sha256": hashes,
        "validation": {"passed": ok, "rows": val_rows},
        "logs": {"in_sample_list": len(splits), "with_moving_rows": int(df["segment"].nunique()),
                 "sample_logs_without_moving_rows": len(splits) - int(df["segment"].nunique())},
        "thresholds_m": list(THRESHOLDS_M),
        "offsets_ms": off_report,
        "populations": pops,
        "bootstrap_P1_observation_M": {"seed": SEED, "n_boot": N_BOOT, "unit": "log", "interval": "95% percentile",
                                       "shares_ci": boot},
        "context_share_of_vehicle_cuboids_moving": ("not computed: t3_label_shares_407.json holds counts of moving "
                                                    "observations per camera, not a share of all vehicle cuboids"),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "computed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    p = pops["P1_rejecting_all_logs"]["observation"]
    print("validation passed; P1 by observation, share > 0.5 m: M {:.3f}, L {:.3f}, U {:.3f}; > 1 m under M {:.3f}".format(
        p["M"]["share_gt"]["0.5"], p["L"]["share_gt"]["0.5"], p["U"]["share_gt"]["0.5"], p["M"]["share_gt"]["1.0"]))
    return res


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    run(*sys.argv[1:6])
