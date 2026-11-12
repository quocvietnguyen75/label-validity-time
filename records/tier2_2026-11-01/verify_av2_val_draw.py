"""Verify the 40 drawn AV2 val logs against the bucket listings saved by download_av2_val_draw.sh (G1 analogue).

For every listed object outside sensors/cameras/stereo_* (not downloaded by design): the local file exists, has the
listed size, and, for single-part objects (ETag without '-'), its MD5 equals the ETag. Prints aggregates only.
Usage: python verify_av2_val_draw.py <av2_root> <draw.txt> <out.json>"""
import hashlib
import json
import os
import re
import sys


def main(argv):
    root, draw, out = argv[1], argv[2], argv[3]
    ids = re.findall(r"[0-9a-f-]{36}", open(draw, encoding="utf-8").read())
    man = os.path.join(root, "_manifest", "tier2_val_draw")
    res = {"logs": len(ids), "objects_expected": 0, "missing": 0, "size_mismatch": 0, "md5_checked": 0,
           "md5_mismatch": 0, "multipart_skipped": 0, "bytes": 0, "logs_with_problems": []}
    for i in ids:
        bad = 0
        prefix = f"datasets/av2/sensor/val/{i}/"
        for line in open(os.path.join(man, f"etags_val_{i}.tsv"), encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            key, size, etag = parts[0], int(parts[1]), parts[2].strip('"')
            rel = key[len(prefix):]
            if rel.startswith("sensors/cameras/stereo_"):
                continue
            res["objects_expected"] += 1
            p = os.path.join(root, "val", i, rel)
            if not os.path.exists(p):
                res["missing"] += 1
                bad += 1
                continue
            if os.path.getsize(p) != size:
                res["size_mismatch"] += 1
                bad += 1
                continue
            res["bytes"] += size
            if "-" in etag:
                res["multipart_skipped"] += 1
                continue
            res["md5_checked"] += 1
            if hashlib.md5(open(p, "rb").read()).hexdigest() != etag:
                res["md5_mismatch"] += 1
                bad += 1
        if bad:
            res["logs_with_problems"].append(i)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv)
