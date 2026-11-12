"""XF-C7 gate G1: check downloaded Waymo v2 files against the saved gcloud bucket listings.

Usage: python -I verify_listing.py [--only SEGMENT_LIST] <WAYMO_V2_ROOT> <component> [<component> ...]
(--only restricts the check to the segments named in a text file, one per line.)
Reads <root>/_manifest/listing_<component>.json and compares <root>/validation/<component>/
by file size and MD5 (the listing's base64 md5_hash). Prints one PASS/FAIL line per component
and exits 1 if any component fails. Reads file bytes only to hash them; never parses the data.
"""
import base64
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class Result:
    component: str
    expected: int
    ok: int
    missing: Tuple[str, ...]
    mismatched: Tuple[str, ...]


def md5_b64(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def expected_files(objs: Iterable[dict], component: str,
                   only: Optional[Iterable[str]] = None) -> Dict[str, Tuple[int, Optional[str]]]:
    """Map relative file name to (size, md5) for one component, skipping folder markers.

    `only`: optional segment names (whitespace and line endings stripped); restricts to <name>.parquet.
    """
    prefix = f"validation/{component}/"
    keep = None if only is None else {s.strip() + ".parquet" for s in only if s.strip()}
    out: Dict[str, Tuple[int, Optional[str]]] = {}
    for o in objs:
        name = str(o.get("name", ""))
        if not name.startswith(prefix):
            continue
        rel = name[len(prefix):]
        if not rel or rel.endswith("/"):
            continue
        if keep is not None and rel not in keep:
            continue
        out[rel] = (int(o["size"]), o.get("md5_hash"))
    return out


def verify_component(objs: Iterable[dict], component: str, local_dir: str,
                     only: Optional[Iterable[str]] = None) -> Result:
    exp = expected_files(objs, component, only)
    missing, mismatched = [], []
    for rel, (size, md5) in sorted(exp.items()):
        p = os.path.join(local_dir, rel)
        if not os.path.isfile(p):
            missing.append(rel)
        elif os.path.getsize(p) != size or (md5 and md5_b64(p) != md5):
            mismatched.append(rel)
    return Result(component, len(exp), len(exp) - len(missing) - len(mismatched), tuple(missing), tuple(mismatched))


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 64
    args = list(argv[1:])
    only = None
    if "--only" in args:
        i = args.index("--only")
        with open(args[i + 1], encoding="utf-8") as f:
            only = [ln for ln in f if ln.strip()]
        del args[i:i + 2]
    root, components = args[0], args[1:]
    failed = False
    for c in components:
        with open(os.path.join(root, "_manifest", f"listing_{c}.json"), encoding="utf-8") as f:
            objs = json.load(f)
        r = verify_component(objs, c, os.path.join(root, "validation", c), only)
        bad = bool(r.missing or r.mismatched)
        failed |= bad
        print(f"{'FAIL' if bad else 'PASS'} {c}: {r.ok}/{r.expected} files match size and MD5; "
              f"missing {len(r.missing)}; mismatched {len(r.mismatched)}")
        for x in (r.missing + r.mismatched)[:5]:
            print(f"    {x}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
