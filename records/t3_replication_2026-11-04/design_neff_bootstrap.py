"""XF-C7 T3 design: expected Kish effective number of logs per camera against sample size (2026-11-04).

Reads only the columns segment, camera, object_id and w of the tier-2 moving analysis set (score >= 0.5, all 50 logs).
No theta-hat, residual, image position or estimate is read. Logs are resampled with replacement from the 50 (the 8
logs without a moving row count as empty logs), and the Kish effective number of logs, (sum W_g)^2 / sum W_g^2 over
the logs' total weights W_g, is computed per camera. The tier-2 results on these logs are already known; this only
sizes the fresh sample. The bootstrap cannot produce heavier weight tails than the 50 logs show, so it is optimistic
if the true tail is heavier.
Usage: python design_neff_bootstrap.py <moving_analysis_set_t2.parquet> <out.json>"""
import json
import sys

import numpy as np
import pyarrow.parquet as pq

NAMES = {1: "ring_front_center", 2: "ring_front_left", 3: "ring_front_right", 4: "ring_side_left",
         5: "ring_side_right", 6: "ring_rear_left", 7: "ring_rear_right"}
ALL_LOGS, REPS, SEED = 50, 4000, 20261104
SIZES = (50, 108, 208, 308, 408, 508, 808)


def kish(v):
    v = v[v > 0]
    return float(v.sum() ** 2 / (v ** 2).sum()) if len(v) else 0.0


def main(src, out):
    d = pq.read_table(src, columns=["segment", "camera", "object_id", "w"]).to_pandas()
    W = d.groupby(["camera", "segment"])["w"].sum().unstack(fill_value=0.0)
    M = np.hstack([W.to_numpy(), np.zeros((W.shape[0], ALL_LOGS - W.shape[1]))])
    cams = [NAMES.get(int(c), str(c)) for c in W.index]
    res = {"source_columns": ["segment", "camera", "object_id", "w"], "reps": REPS, "seed": SEED,
           "observed": {c: {"logs_with_rows": int((M[i] > 0).sum()), "kish": round(kish(M[i]), 2)}
                        for i, c in enumerate(cams)},
           "bootstrap": {}}
    rng = np.random.default_rng(SEED)
    for n in SIZES:
        draws = {c: [] for c in cams}
        for _ in range(REPS):
            S = M[:, rng.integers(0, ALL_LOGS, n)]
            for i, c in enumerate(cams):
                draws[c].append(kish(S[i]))
        res["bootstrap"][str(n)] = {c: {"median": round(float(np.median(v)), 1), "p10": round(float(np.percentile(v, 10)), 1)}
                                    for c, v in draws.items()}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res["observed"]))
    for n in SIZES:
        print(n, res["bootstrap"][str(n)])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
