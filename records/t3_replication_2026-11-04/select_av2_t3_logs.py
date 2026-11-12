"""XF-C7 T3 (AV2 replication): the fresh Argoverse 2 Sensor sample, fixed on 2026-11-04 before any download.

Rule, written before any file of these logs is read:
- val: every id in the 2026-11-04 bucket listing (identical to the 1 Nov listing, 40a6a91c...) minus P01's two val logs
  and minus the 40 logs drawn for tier 2 (tier2_2026-11-01/av2_val_draw_seed20261101.txt). All 108 enter; no draw.
- train: the 700 ids in the 2026-11-04 bucket listing minus P01's eight train logs, sorted as strings, then
  random.Random(20261104).sample(pool, 300).
The rule reads ids only: speed, scene, city, length and content play no part. The 392 train logs left over stay
untouched. P01's ids are those of pilot/download_av2.sh and tier2_2026-11-01/build_t2_run_config.py.
Usage: python select_av2_t3_logs.py <val_listing> <train_listing> <t2_draw.txt> <out_sample.txt>
Writes one '<split>/<id>' per line (val sorted, then train in draw order) and prints counts."""
import random
import re
import sys

SEED, K_TRAIN = 20261104, 300
P01_TRAIN = {"3ca11a5e-50b2-3cc3-af7a-ce7ab02b9954", "58d01358-5927-36fa-9e11-d18d1dc1f4f0",
             "6b14d7c0-20f9-390b-af38-507a5de5998c", "81d2b40a-c579-3e9c-b520-bee26cda947d",
             "c556f8e0-a001-3586-b2cf-d3256685c39f", "c780d53a-2d37-3cd8-9e89-530966aef53e",
             "f648b945-6c70-3105-bd23-9502894e37d4", "fd5c6932-2ee2-3cfb-9bdc-0b30bfb33a91"}
P01_VAL = {"9a448a80-0e9a-3bf0-90f3-21750dfef55a", "d5d6f11c-3026-3e0e-9d67-c111233e22de"}
ID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def listing_ids(path):
    return re.findall(r"PRE (" + ID + r")/", open(path, encoding="utf-8").read())


def select(val_ids, train_ids, t2_drawn, seed=SEED, k=K_TRAIN):
    val_rest = sorted(set(val_ids) - P01_VAL - set(t2_drawn))
    pool = sorted(set(train_ids) - P01_TRAIN)
    return val_rest, random.Random(seed).sample(pool, k)


if __name__ == "__main__":
    val_ids, train_ids = listing_ids(sys.argv[1]), listing_ids(sys.argv[2])
    drawn = re.findall(ID, open(sys.argv[3], encoding="utf-8").read())
    assert len(val_ids) == 150 and len(train_ids) == 700 and len(drawn) == 40, (len(val_ids), len(train_ids), len(drawn))
    assert P01_VAL <= set(val_ids) and P01_TRAIN <= set(train_ids) and set(drawn) <= set(val_ids)
    val_rest, train_draw = select(val_ids, train_ids, drawn)
    assert len(val_rest) == 108 and len(set(train_draw)) == K_TRAIN
    with open(sys.argv[4], "w", encoding="utf-8", newline="\n") as f:
        for i in val_rest:
            f.write(f"val/{i}\n")
        for i in train_draw:
            f.write(f"train/{i}\n")
    print(f"val {len(val_rest)}, train {len(train_draw)}, total {len(val_rest) + len(train_draw)}")
