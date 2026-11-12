"""Verify the T3 AV2 sample against the bucket listings saved by download_av2_t3.sh (G1 analogue).

For every listed object outside sensors/cameras/stereo_* (not downloaded by design): the local file exists, has the
listed size, and, for single-part objects (ETag without '-'), its MD5 equals the ETag. A listing that is empty or
unreadable counts as a problem. Prints aggregates only.
Usage: python verify_av2_t3.py <av2_t3_root> <av2_t3_sample.txt> <out.json>"""
import hashlib
import json
import os
import re
import sys


def md5(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main(argv):
    root, sample, out = argv[1], argv[2], argv[3]
    entries = re.findall(r"(val|train)/([0-9a-f-]{36})", open(sample, encoding="utf-8").read())
    man = os.path.join(root, "_manifest", "t3")
    res = {"logs": len(entries), "objects_expected": 0, "missing": 0, "size_mismatch": 0, "md5_checked": 0,
           "md5_mismatch": 0, "multipart_skipped": 0, "bytes": 0, "logs_with_problems": [], "bad_listings": []}
    for split, i in entries:
        bad, rows = 0, 0
        prefix = f"datasets/av2/sensor/{split}/{i}/"
        lp = os.path.join(man, f"etags_{split}_{i}.tsv")
        lines = open(lp, encoding="utf-8").read().splitlines() if os.path.exists(lp) else []
        for line in lines:
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            rows += 1
            key, size, etag = parts[0], int(parts[1]), parts[2].strip('"')
            rel = key[len(prefix):]
            if rel.startswith("sensors/cameras/stereo_"):
                continue
            res["objects_expected"] += 1
            p = os.path.join(root, split, i, rel)
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
            if md5(p) != etag:
                res["md5_mismatch"] += 1
                bad += 1
        if rows == 0:
            res["bad_listings"].append(f"{split}/{i}")
            bad += 1
        if bad:
            res["logs_with_problems"].append(f"{split}/{i}")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv)
