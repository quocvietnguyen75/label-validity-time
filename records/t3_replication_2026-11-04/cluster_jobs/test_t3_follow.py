"""Synthetic tests for t3_follow.py (run: python -m unittest test_t3_follow -v, from this folder)."""
import hashlib
import json
import os
import tempfile
import time
import unittest

import t3_follow as tf


def put(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


class Fixture:
    def __init__(self, root):
        self.root = root

    def log(self, split, i, files, stereo=True, etag_override=None, age_s=1000):
        """Write a bucket listing for files {rel: bytes} (plus one stereo object) and return its rows."""
        prefix = f"datasets/av2/sensor/{split}/{i}/"
        lines = []
        for rel, data in files.items():
            etag = (etag_override or {}).get(rel, hashlib.md5(data).hexdigest())
            lines.append(f'{prefix}{rel}\t{len(data)}\t"{etag}"')
        if stereo:
            lines.append(f'{prefix}sensors/cameras/stereo_front_left/1.jpg\t10\t"00"')
        lp = os.path.join(self.root, "_manifest", f"etags_{split}_{i}.tsv")
        put(lp, ("\n".join(lines) + "\n").encode())
        t = time.time() - age_s
        os.utime(lp, (t, t))

    def write(self, split, i, files):
        for rel, data in files.items():
            put(os.path.join(self.root, split, i, rel), data)


# Bounds the loop when a test keeps downloads "running", so a broken helper fails instead of hanging.
BOUND_H = 0.0003

FILES = {"annotations.feather": b"ann" * 10, "sensors/cameras/ring_front_center/1.jpg": b"img1",
         "sensors/lidar/1.feather": b"lidar"}


class FollowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "root")
        self.mdir = os.path.join(self.tmp.name, "markers")
        self.odir = os.path.join(self.tmp.name, "out")
        os.makedirs(self.odir)
        self.fx = Fixture(self.root)
        self.old_min = tf.MIN_OBJECTS
        tf.MIN_OBJECTS = 2
        self.list_path = os.path.join(self.tmp.name, "list.txt")

    def tearDown(self):
        tf.MIN_OBJECTS = self.old_min
        self.tmp.cleanup()

    def set_list(self, *logs):
        with open(self.list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs) + "\n")

    def marker(self, split, i):
        p = tf.marker_path(self.mdir, split, i)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    def test_complete_log_is_verified_and_handed_over_once(self):
        self.set_list("val/a")
        self.fx.log("val", "a", FILES)
        self.fx.write("val", "a", FILES)
        rc = tf.cmd_verify(self.root, self.list_path, self.mdir, "1", BOUND_H, sleep=lambda s: None, running=lambda ids: True)
        self.assertEqual(rc, 0)
        m = self.marker("val", "a")
        self.assertTrue(m["ok"])
        self.assertEqual((m["objects"], m["md5_checked"], m["md5_mismatch"]), (3, 3, 0))
        self.assertEqual(tf.ready(self.root, self.list_path, self.mdir, self.odir, 3), [os.path.join(self.root, "val", "a")])
        put(os.path.join(self.odir, "a.parquet"), b"x")
        self.assertEqual(tf.ready(self.root, self.list_path, self.mdir, self.odir, 3), [])

    def test_incomplete_log_waits_while_downloading_and_fails_after(self):
        self.set_list("val/a", "val/b")
        for i in ("a", "b"):
            self.fx.log("val", i, FILES)
        self.fx.write("val", "a", FILES)
        self.fx.write("val", "b", {k: v for k, v in FILES.items() if "lidar" not in k})
        tf.one_pass(self.root, tf.entries(self.list_path), self.mdir, require_age=True, log=lambda s: None)
        self.assertIsNotNone(self.marker("val", "a"))
        self.assertIsNone(self.marker("val", "b"))
        calls = iter([True, True, False, False])
        rc = tf.cmd_verify(self.root, self.list_path, self.mdir, "1", 1, sleep=lambda s: None,
                           running=lambda ids: next(calls))
        self.assertEqual(rc, 1)
        m = self.marker("val", "b")
        self.assertFalse(m["ok"])
        self.assertEqual(m["objects_missing_or_wrong_size"], 1)
        done = json.load(open(os.path.join(self.mdir, "_DONE.json"), encoding="utf-8"))
        self.assertEqual((done["logs"], done["logs_ok"], done["failed_logs"]), (2, 1, ["val/b"]))
        self.assertEqual(tf.ready(self.root, self.list_path, self.mdir, self.odir, 5), [os.path.join(self.root, "val", "a")])

    def test_wrong_content_same_size_fails_and_is_not_handed_over(self):
        self.set_list("val/a")
        self.fx.log("val", "a", FILES)
        bad = dict(FILES, **{"sensors/lidar/1.feather": b"LIDAR"})
        self.fx.write("val", "a", bad)
        tf.cmd_verify(self.root, self.list_path, self.mdir, "1", BOUND_H, sleep=lambda s: None, running=lambda ids: True)
        m = self.marker("val", "a")
        self.assertFalse(m["ok"])
        self.assertEqual(m["md5_mismatch"], 1)
        self.assertEqual(tf.ready(self.root, self.list_path, self.mdir, self.odir, 3), [])

    def test_young_listing_is_not_read_while_downloading(self):
        self.set_list("val/a")
        self.fx.log("val", "a", FILES, age_s=0)
        self.fx.write("val", "a", FILES)
        tf.one_pass(self.root, tf.entries(self.list_path), self.mdir, require_age=True, log=lambda s: None)
        self.assertIsNone(self.marker("val", "a"))
        tf.one_pass(self.root, tf.entries(self.list_path), self.mdir, require_age=False, log=lambda s: None)
        self.assertTrue(self.marker("val", "a")["ok"])

    def test_stereo_ignored_multipart_size_only_short_listing_unusable(self):
        self.set_list("val/a")
        self.fx.log("val", "a", FILES, etag_override={"annotations.feather": "abc-2"})
        rows = tf.listing(self.root, "val", "a")
        self.assertEqual(len(rows), 3)
        self.assertFalse(any("stereo_" in r[0] for r in rows))
        self.fx.write("val", "a", FILES)
        r = tf.verify_log(self.root, "val", "a", rows)
        self.assertEqual((r["multipart_skipped"], r["md5_checked"], r["ok"]), (1, 2, True))
        tf.MIN_OBJECTS = 4
        self.assertIsNone(tf.listing(self.root, "val", "a"))

    def test_partial_file_blocks_while_downloading_and_fails_after(self):
        self.set_list("val/a")
        self.fx.log("val", "a", FILES)
        self.fx.write("val", "a", dict(FILES, **{"sensors/lidar/1.feather": b"lid"}))
        tf.one_pass(self.root, tf.entries(self.list_path), self.mdir, require_age=True, log=lambda s: None)
        self.assertIsNone(self.marker("val", "a"))
        rc = tf.cmd_verify(self.root, self.list_path, self.mdir, "1", 1, sleep=lambda s: None, running=lambda ids: False)
        self.assertEqual(rc, 1)
        m = self.marker("val", "a")
        self.assertFalse(m["ok"])
        self.assertEqual(m["objects_missing_or_wrong_size"], 1)

    def test_status_counts(self):
        self.set_list("val/a", "val/b")
        self.fx.log("val", "a", FILES)
        self.fx.write("val", "a", FILES)
        tf.cmd_verify(self.root, self.list_path, self.mdir, "1", 1, sleep=lambda s: None, running=lambda ids: False)
        put(os.path.join(self.odir, "a.parquet"), b"x")
        s = tf.status(self.list_path, self.mdir, self.odir)
        self.assertEqual((s["verified_ok"], s["verified_failed"], s["outputs"], s["verifier_done"]), (1, 1, 1, True))


if __name__ == "__main__":
    unittest.main()
