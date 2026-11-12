"""XF-C7 T3 on the H100 cluster: verify each AV2 log as soon as its download is complete, and hand verified logs to the
frozen label-side and detector steps (SAMPLE_PIN_T3_2026-11-04.md, addendum 5). Standard library only.

A log counts as downloaded when its bucket listing (<root>/_manifest/etags_<split>_<id>.tsv, which jobs/download.sbatch
writes before its sync) lists at least MIN_OBJECTS objects and every listed object outside sensors/cameras/stereo_*
exists locally with the listed size. aws s3 sync writes each file under a temporary name and renames it when it is
complete, so a file present under its own name is whole. While a download job is running, the listing must also be at
least MIN_AGE_S old, so a listing still being written is never read. The log is then verified once, as
verify_av2_t3.py does (G1 analogue): the MD5 of every single-part object (ETag without '-') must equal its ETag. The
result goes to <marker_dir>/<split>_<id>.json. When the download jobs have ended and a full pass finds nothing new,
every log still unverified gets a failed marker, and <marker_dir>/_DONE.json holds the totals. Reads files and
listings only; computes nothing from the data.

Usage:
  python t3_follow.py verify <root> <list.txt> <marker_dir> <download_job_ids> <max_hours>
  python t3_follow.py ready <root> <list.txt> <marker_dir> <out_dir> <n>
      prints up to n verified log directories without <out_dir>/<id>.parquet, in list order
  python t3_follow.py status <list.txt> <marker_dir> <out_dir>
"""
import hashlib
import json
import os
import subprocess
import sys
import time

MIN_AGE_S, MIN_OBJECTS, POLL_S = 120, 1000, 60
ACTIVE = ("PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "CONFIGURING", "COMPLETING")


def entries(list_path):
    out = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                split, i = line.split("/")
                out.append((split, i))
    return out


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def marker_path(mdir, split, i):
    return os.path.join(mdir, f"{split}_{i}.json")


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def md5(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def listing(root, split, i, require_age=True, now=None):
    """[(relative path, size, etag)] outside the stereo cameras, or None if not (yet) usable."""
    p = os.path.join(root, "_manifest", f"etags_{split}_{i}.tsv")
    if not os.path.exists(p):
        return None
    if require_age and (now if now is not None else time.time()) - os.path.getmtime(p) < MIN_AGE_S:
        return None
    prefix = f"datasets/av2/sensor/{split}/{i}/"
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f.read().splitlines():
            parts = line.split("\t")
            if len(parts) != 3 or not parts[0].startswith(prefix):
                continue
            rel = parts[0][len(prefix):]
            if rel.startswith("sensors/cameras/stereo_"):
                continue
            rows.append((rel, int(parts[1]), parts[2].strip('"')))
    return rows if len(rows) >= MIN_OBJECTS else None


def missing_count(root, split, i, rows, stop_at=None):
    """Listed objects absent or of another size; counting stops at stop_at."""
    base, n = os.path.join(root, split, i), 0
    for rel, size, _ in rows:
        try:
            bad = os.path.getsize(os.path.join(base, rel)) != size
        except OSError:
            bad = True
        n += bad
        if stop_at is not None and n >= stop_at:
            break
    return n


def verify_log(root, split, i, rows):
    base = os.path.join(root, split, i)
    r = {"log": f"{split}/{i}", "objects": len(rows), "bytes": 0, "md5_checked": 0, "md5_mismatch": 0,
         "multipart_skipped": 0}
    for rel, size, etag in rows:
        r["bytes"] += size
        if "-" in etag:
            r["multipart_skipped"] += 1
            continue
        r["md5_checked"] += 1
        if md5(os.path.join(base, rel)) != etag:
            r["md5_mismatch"] += 1
    r["ok"] = r["md5_mismatch"] == 0
    return r


def downloads_running(ids):
    """True while any download job is active; True as well when Slurm cannot be asked (keep waiting)."""
    try:
        out = subprocess.run(["sacct", "-n", "-X", "-P", "-o", "State", "-j", ids], capture_output=True, text=True,
                             timeout=60)
    except Exception:
        return True
    if out.returncode != 0 or not out.stdout.strip():
        return True
    return any(s.split()[0] in ACTIVE for s in out.stdout.split("\n") if s.strip())


def one_pass(root, ents, mdir, require_age, log=print):
    progress = 0
    for split, i in ents:
        mp = marker_path(mdir, split, i)
        if os.path.exists(mp):
            continue
        rows = listing(root, split, i, require_age)
        if rows is None or missing_count(root, split, i, rows, stop_at=1):
            continue
        r = verify_log(root, split, i, rows)
        write_json(mp, r)
        progress += 1
        log(f"{time.strftime('%H:%M:%S', time.gmtime())} verified {split}/{i} ok={r['ok']} objects={r['objects']}")
    return progress


def close_out(root, ents, mdir):
    """Failed markers for every log still unverified; then the totals in _DONE.json."""
    for split, i in ents:
        mp = marker_path(mdir, split, i)
        if os.path.exists(mp):
            continue
        rows = listing(root, split, i, require_age=False)
        write_json(mp, {"log": f"{split}/{i}", "ok": False, "reason": "incomplete after the downloads ended",
                        "listing_usable": rows is not None,
                        "objects_missing_or_wrong_size": missing_count(root, split, i, rows) if rows else None})
    tot = {"logs": len(ents), "logs_ok": 0, "objects": 0, "bytes": 0, "md5_checked": 0, "md5_mismatch": 0,
           "multipart_skipped": 0, "failed_logs": []}
    for split, i in ents:
        r = read_json(marker_path(mdir, split, i))
        if r.get("ok"):
            tot["logs_ok"] += 1
        else:
            tot["failed_logs"].append(r["log"])
        for k in ("objects", "bytes", "md5_checked", "md5_mismatch", "multipart_skipped"):
            tot[k] += int(r.get(k) or 0)
    tot["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(os.path.join(mdir, "_DONE.json"), tot)
    return tot


def cmd_verify(root, list_path, mdir, ids, max_hours, sleep=time.sleep, running=None):
    running = running or downloads_running
    os.makedirs(mdir, exist_ok=True)
    ents = entries(list_path)
    t_end = time.time() + 3600 * float(max_hours)
    while True:
        dl = running(ids)
        progress = one_pass(root, ents, mdir, require_age=dl)
        if all(os.path.exists(marker_path(mdir, s, i)) for s, i in ents):
            break
        if progress == 0 and (not dl or time.time() > t_end):
            break
        if progress == 0:
            sleep(POLL_S)
    tot = close_out(root, ents, mdir)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in tot.items()}), flush=True)
    return 0 if tot["logs_ok"] == tot["logs"] else 1


def ready(root, list_path, mdir, odir, n):
    out = []
    for split, i in entries(list_path):
        if os.path.exists(os.path.join(odir, i + ".parquet")):
            continue
        mp = marker_path(mdir, split, i)
        if not os.path.exists(mp):
            continue
        if read_json(mp).get("ok"):
            out.append(os.path.join(root, split, i))
            if len(out) >= n:
                break
    return out


def status(list_path, mdir, odir):
    ents = entries(list_path)
    marks = [marker_path(mdir, s, i) for s, i in ents]
    ok = sum(1 for m in marks if os.path.exists(m) and read_json(m).get("ok"))
    failed = sum(1 for m in marks if os.path.exists(m) and not read_json(m).get("ok"))
    done = sum(1 for _, i in ents if os.path.exists(os.path.join(odir, i + ".parquet")))
    return {"logs": len(ents), "verified_ok": ok, "verified_failed": failed, "outputs": done,
            "verifier_done": os.path.exists(os.path.join(mdir, "_DONE.json"))}


def main(argv):
    if len(argv) >= 7 and argv[1] == "verify":
        return cmd_verify(argv[2], argv[3], argv[4], argv[5], argv[6])
    if len(argv) >= 7 and argv[1] == "ready":
        out = ready(argv[2], argv[3], argv[4], argv[5], int(argv[6]))
        if out:
            print("\n".join(out))
        return 0
    if len(argv) >= 5 and argv[1] == "status":
        print(json.dumps(status(argv[2], argv[3], argv[4])))
        return 0
    print(__doc__)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
