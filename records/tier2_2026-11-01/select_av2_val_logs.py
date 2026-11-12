"""Tier 2 (XF-C7 extension): fixed-seed draw of additional Argoverse 2 Sensor validation logs (2026-11-01).

Rule, written before any AV2 label or image is read beyond P01's ten logs' schema:
- frame: the 150 val log ids in the public bucket listing of 2026-11-01 (av2_val_listing_2026-11-01.txt);
- exclude P01's two val logs (9a448a80..., d5d6f11c...), which are already on disk;
- sort the remaining ids as strings, then random.Random(20261101).sample(ids, 40).
The draw does not look at speed, scene, length or any content. Whether these logs enter the tier-2 analysis is a
user decision recorded at the tier-2 freeze; downloading them commits to nothing.
Usage: python select_av2_val_logs.py <listing.txt>   (prints one id per line)"""
import random
import re
import sys

SEED, K = 20261101, 40
P01_VAL = {"9a448a80-0e9a-3bf0-90f3-21750dfef55a", "d5d6f11c-3026-3e0e-9d67-c111233e22de"}


def draw(ids, seed=SEED, k=K, exclude=frozenset(P01_VAL)):
    pool = sorted(set(ids) - set(exclude))
    return random.Random(seed).sample(pool, k)


if __name__ == "__main__":
    ids = re.findall(r"PRE ([0-9a-f-]{36})/", open(sys.argv[1], encoding="utf-8").read())
    assert len(ids) == 150, len(ids)
    for i in draw(ids):
        print(i)
