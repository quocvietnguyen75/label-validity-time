"""Synthetic tests for verify_listing.py (XF-C7 gate G1). Run: python -I -m unittest discover -s <pilot dir> -p test_verify_listing.py"""
import base64
import hashlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_listing as vl  # noqa: E402


def _md5_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


class VerifyListingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.good = b"good parquet bytes"
        self.bad = b"corrupted bytes!!!"
        with open(os.path.join(self.dir, "seg_a.parquet"), "wb") as f:
            f.write(self.good)
        with open(os.path.join(self.dir, "seg_b.parquet"), "wb") as f:
            f.write(self.bad)  # same length as the listed object, different content
        self.objs = [
            {"name": "validation/camera_box/", "size": 6, "md5_hash": "x"},  # folder marker
            {"name": "validation/camera_box/seg_a.parquet", "size": len(self.good), "md5_hash": _md5_b64(self.good)},
            {"name": "validation/camera_box/seg_b.parquet", "size": len(self.good), "md5_hash": _md5_b64(self.good)},
            {"name": "validation/camera_box/seg_c.parquet", "size": 10, "md5_hash": _md5_b64(b"0123456789")},
            {"name": "validation/lidar_box/seg_a.parquet", "size": 3, "md5_hash": "y"},  # other component
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_expected_files_skip_folder_marker_and_other_components(self):
        exp = vl.expected_files(self.objs, "camera_box")
        self.assertEqual(sorted(exp), ["seg_a.parquet", "seg_b.parquet", "seg_c.parquet"])

    def test_verify_component_flags_missing_and_checksum_mismatch(self):
        r = vl.verify_component(self.objs, "camera_box", self.dir)
        self.assertEqual(r.expected, 3)
        self.assertEqual(r.ok, 1)
        self.assertEqual(r.missing, ("seg_c.parquet",))
        self.assertEqual(r.mismatched, ("seg_b.parquet",))

    def test_size_mismatch_is_flagged_without_hashing(self):
        objs = [{"name": "validation/camera_box/seg_a.parquet", "size": len(self.good) + 1, "md5_hash": _md5_b64(self.good)}]
        r = vl.verify_component(objs, "camera_box", self.dir)
        self.assertEqual(r.mismatched, ("seg_a.parquet",))

    def test_only_restricts_to_listed_segments_and_strips_line_endings(self):
        exp = vl.expected_files(self.objs, "camera_box", only=["seg_a\r", " seg_b "])
        self.assertEqual(sorted(exp), ["seg_a.parquet", "seg_b.parquet"])
        r = vl.verify_component(self.objs, "camera_box", self.dir, only=["seg_a"])
        self.assertEqual((r.expected, r.ok, r.missing, r.mismatched), (1, 1, (), ()))

    def test_all_match_passes(self):
        objs = [{"name": "validation/camera_box/seg_a.parquet", "size": len(self.good), "md5_hash": _md5_b64(self.good)}]
        r = vl.verify_component(objs, "camera_box", self.dir)
        self.assertEqual((r.expected, r.ok, r.missing, r.mismatched), (1, 1, (), ()))


if __name__ == "__main__":
    unittest.main()
