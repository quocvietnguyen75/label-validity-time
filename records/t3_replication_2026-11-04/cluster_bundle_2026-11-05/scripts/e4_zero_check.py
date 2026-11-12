"""XF-C7 T3 code check (optional, on T2's known tables): S-d's zero shift must reproduce the saved moving set exactly.
Usage: python e4_zero_check.py <label_side.parquet> <reference_table.parquet> <moving_set.parquet> <S3_summary.json> <out.json>"""
import json
import sys

import pandas as pd

from careport import xfc7_t3_run as t3
from careport.xfc7_t2_static import CAM_T2

lab, ref, mo = (pd.read_parquet(p) for p in sys.argv[1:4])
s3 = json.load(open(sys.argv[4], encoding="utf-8"))
cams = s3["decision_cameras"]
res = t3.e4_av2(lab, ref, mo[mo["camera"].map(lambda c: CAM_T2.get(int(c)) in cams)], cams, CAM_T2, deltas=())
json.dump(res, open(sys.argv[5], "w", encoding="utf-8"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ("baseline_reproduced", "zero_shift_changed_share")}))
sys.exit(0 if res["baseline_reproduced"] else 1)
