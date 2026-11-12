import os
import tempfile
import unittest

import numpy as np


def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").close()


class ListImagesTest(unittest.TestCase):
    def test_ring_cameras_only_sorted_with_camera_index_and_timestamp(self):
        from careport.xfc7_t2_detect import RING, list_images
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "val", "abc")
            _touch(os.path.join(log, "sensors", "cameras", "ring_front_left", "200.jpg"))
            _touch(os.path.join(log, "sensors", "cameras", "ring_front_left", "100.jpg"))
            _touch(os.path.join(log, "sensors", "cameras", "ring_front_center", "150.jpg"))
            _touch(os.path.join(log, "sensors", "cameras", "stereo_front_left", "100.jpg"))
            imgs = list_images(log)
            self.assertEqual([(c, t) for _, c, t, _ in imgs], [(1, 150), (2, 100), (2, 200)])
            self.assertTrue(all(s == "abc" for s, _, _, _ in imgs))
            self.assertEqual(RING[0], "ring_front_center")
            self.assertEqual(RING[1], "ring_front_left")

    def test_batches_never_mix_cameras(self):
        from careport.xfc7_t2_detect import camera_batches
        imgs = [("s", 1, t, f"p{t}") for t in range(5)] + [("s", 2, t, f"q{t}") for t in range(3)]
        b = list(camera_batches(imgs, 4))
        self.assertEqual([len(x) for x in b], [4, 1, 3])
        self.assertTrue(all(len({c for _, c, _, _ in x}) == 1 for x in b))


class NativeSizeTest(unittest.TestCase):
    def test_av2_native_sizes_keep_scale_one(self):
        from careport.xfc7_t2_detect import AV2_MAX_SIZE, AV2_MIN_SIZE
        for h, w in ((1550, 2048), (2048, 1550)):
            scale = min(AV2_MIN_SIZE / min(h, w), AV2_MAX_SIZE / max(h, w))
            self.assertEqual(scale, 1.0)


class RowsTest(unittest.TestCase):
    def test_rows_carry_log_camera_and_timestamp(self):
        from careport.xfc7_ext_detect import RAW_COLUMNS
        from careport.xfc7_t2_detect import rows_for
        r = rows_for("log1", 3, 315975802860228000, np.array([[0, 0, 10, 10], [5, 5, 9, 9]], float),
                     np.array([3, 1]), np.array([0.9, 0.9]))
        self.assertEqual(list(r.columns), RAW_COLUMNS)
        self.assertEqual(len(r), 1)                                   # the person box is dropped
        self.assertEqual(int(r["frame"].iloc[0]), 315975802860228000)
        self.assertEqual(int(r["camera"].iloc[0]), 3)


if __name__ == "__main__":
    unittest.main()
