import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from careport.xfc7_ext_detect import (COCO_VEHICLE, RAW_COLUMNS, class_agnostic_nms, detect_images,
                                      reference_table, vehicle_detections)


class VehicleDetectionsTest(unittest.TestCase):
    def test_keeps_coco_vehicle_classes_above_floor_and_drops_others(self):
        boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 40], [1, 1, 2, 2], [0, 0, 4, 4]], float)
        labels = np.array([3, 7, 8, 1, 6])          # car, train, truck, person, bus
        scores = np.array([0.9, 0.9, 0.04, 0.9, 0.5])
        keep = vehicle_detections(boxes, labels, scores, score_floor=0.05)
        np.testing.assert_array_equal(keep, [0, 4])  # train and person dropped; truck below the floor
        self.assertEqual(COCO_VEHICLE, {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"})


class NmsTest(unittest.TestCase):
    def test_car_and_truck_on_one_vehicle_collapse_to_the_higher_score(self):
        boxes = np.array([[100, 100, 200, 180], [102, 101, 201, 182], [400, 100, 450, 140]], float)
        scores = np.array([0.6, 0.8, 0.7])
        keep = class_agnostic_nms(boxes, scores, iou=0.7)
        np.testing.assert_array_equal(sorted(keep), [1, 2])

    def test_disjoint_boxes_all_kept(self):
        boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
        keep = class_agnostic_nms(boxes, np.array([0.5, 0.9]), iou=0.7)
        np.testing.assert_array_equal(sorted(keep), [0, 1])


class ReferenceTableTest(unittest.TestCase):
    def test_threshold_nms_and_drop_in_columns(self):
        raw = pd.DataFrame({
            "segment": ["s"] * 4, "frame": [7] * 4, "camera": [1] * 4,
            "x1": [100.0, 102.0, 400.0, 600.0], "y1": [100.0, 101.0, 100.0, 50.0],
            "x2": [200.0, 201.0, 450.0, 640.0], "y2": [180.0, 182.0, 140.0, 90.0],
            "score": [0.6, 0.8, 0.7, 0.3], "label": [3, 8, 3, 3]})
        ref = reference_table(raw, score_min=0.5, nms_iou=0.7)
        self.assertEqual(list(ref.columns), ["segment", "frame", "camera", "camera_object_id", "u", "v", "w"])
        self.assertEqual(len(ref), 2)                       # duplicate collapsed, low score dropped
        row = ref.sort_values("u").iloc[0]
        self.assertAlmostEqual(row["u"], 151.5)             # centre of the kept truck box (102..201)
        self.assertAlmostEqual(row["v"], 141.5)
        self.assertAlmostEqual(row["w"], 99.0)
        self.assertTrue(ref["camera_object_id"].is_unique)


class DetectImagesTest(unittest.TestCase):
    def test_runs_a_model_over_images_and_returns_raw_rows(self):
        def fake_model(img):                                  # img: H x W x 3 uint8
            h, w = img.shape[:2]
            return (np.array([[0, 0, w / 2, h / 2], [1, 1, 2, 2]], float), np.array([3, 1]), np.array([0.9, 0.9]))

        images = [("seg", 10, 1, np.zeros((4, 6, 3), np.uint8)), ("seg", 10, 2, np.zeros((2, 8, 3), np.uint8))]
        raw = detect_images(images, fake_model, score_floor=0.05)
        self.assertEqual(list(raw.columns), RAW_COLUMNS)
        self.assertEqual(len(raw), 2)                         # the person box is dropped in each image
        np.testing.assert_allclose(raw["x2"].to_numpy(), [3.0, 4.0])
        self.assertEqual(set(raw["camera"]), {1, 2})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "raw.parquet")
            raw.to_parquet(p)
            self.assertEqual(list(pd.read_parquet(p).columns), RAW_COLUMNS)


class ResumeManifestTest(unittest.TestCase):
    def _fresh(self, sha="abc"):
        return {"weights_sha256": sha, "min_size": 1280, "started": "t2", "files": {}}

    def test_resume_keeps_finished_files_and_logs_the_run(self):
        from careport.xfc7_ext_detect import resume_manifest
        import json
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "run_manifest.json")
            json.dump({"weights_sha256": "abc", "min_size": 1280, "started": "t1", "files": {"a.parquet": {"images": 3}}},
                      open(p, "w"))
            m = resume_manifest(p, self._fresh())
            self.assertEqual(m["files"], {"a.parquet": {"images": 3}})
            self.assertEqual(m["runs"], ["t1", "t2"])

    def test_resume_refuses_changed_settings(self):
        from careport.xfc7_ext_detect import resume_manifest
        import json
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "run_manifest.json")
            json.dump({"weights_sha256": "abc", "min_size": 1280, "started": "t1", "files": {}}, open(p, "w"))
            with self.assertRaises(SystemExit):
                resume_manifest(p, self._fresh(sha="xyz"))

    def test_no_previous_manifest(self):
        from careport.xfc7_ext_detect import resume_manifest
        m = resume_manifest(os.path.join(tempfile.gettempdir(), "does_not_exist_xfc7.json"), self._fresh())
        self.assertEqual(m["runs"], ["t2"])


class RecycleTest(unittest.TestCase):
    """The pass runs in child processes of at most N segments each (memory grows ~1 GB per segment read, 31 Oct)."""

    def test_parse_args(self):
        from careport.xfc7_ext_detect import parse_args
        self.assertEqual(parse_args(["p", "img", "out"]), ("img", "out", 4, None, None))
        self.assertEqual(parse_args(["p", "img", "out", "4", "--per-process", "10"]), ("img", "out", 4, None, 10))
        self.assertEqual(parse_args(["p", "img", "out", "4", "--max-new", "10"]), ("img", "out", 4, 10, None))
        self.assertIsNone(parse_args(["p", "img"]))

    def test_pending_files_skips_finished_segments(self):
        from careport.xfc7_ext_detect import pending_files
        with tempfile.TemporaryDirectory() as d:
            files = [os.path.join(d, "in", f) for f in ("a.parquet", "b.parquet", "c.parquet")]
            os.makedirs(os.path.join(d, "out"))
            open(os.path.join(d, "out", "b.parquet"), "w").close()
            self.assertEqual(pending_files(files, os.path.join(d, "out")), [files[0], files[2]])

    def test_drive_runs_children_until_nothing_remains(self):
        from careport.xfc7_ext_detect import EX_MORE, drive
        left, calls = [25], []

        def child():
            calls.append(1)
            left[0] = max(0, left[0] - 10)
            return EX_MORE if left[0] else 0

        self.assertEqual(drive(child, lambda: left[0]), 0)
        self.assertEqual(len(calls), 3)

    def test_drive_stops_on_a_failed_child(self):
        from careport.xfc7_ext_detect import drive
        calls = []
        self.assertEqual(drive(lambda: calls.append(1) or 1, lambda: 5), 1)
        self.assertEqual(len(calls), 1)

    def test_drive_stops_when_a_child_finishes_nothing(self):
        from careport.xfc7_ext_detect import EX_MORE, drive
        calls = []
        self.assertNotIn(drive(lambda: calls.append(1) or EX_MORE, lambda: 5), (0, EX_MORE))
        self.assertEqual(len(calls), 1)

    def test_drive_with_nothing_pending_runs_no_child(self):
        from careport.xfc7_ext_detect import drive
        calls = []
        self.assertEqual(drive(lambda: calls.append(1) or 0, lambda: 0), 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
