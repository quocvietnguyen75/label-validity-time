import io
import unittest

import numpy as np
from PIL import Image

from careport.xfc7_masks import decode_panoptic, instance_boxes, link_masks


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint16)).save(buf, format="PNG")
    return buf.getvalue()


class DecodeTest(unittest.TestCase):
    def test_round_trip_semantic_and_instance(self):
        sem = np.array([[2, 2, 0], [9, 3, 3]])
        inst = np.array([[1, 1, 0], [1, 2, 2]])
        s, i = decode_panoptic(_png(sem * 1000 + inst), 1000)
        np.testing.assert_array_equal(s, sem)
        np.testing.assert_array_equal(i, inst)


class InstanceBoxTest(unittest.TestCase):
    def test_vehicle_instances_tight_boxes_pixel_edges(self):
        sem = np.array([[2, 2, 0, 9], [0, 3, 3, 9], [4, 0, 0, 0]])
        inst = np.array([[1, 1, 0, 1], [0, 2, 2, 1], [0, 0, 0, 0]])   # class 4 pixel has instance 0: excluded
        b = instance_boxes(sem, inst, classes=(2, 3, 4, 5)).sort_values("semantic").reset_index(drop=True)
        self.assertEqual(b["semantic"].tolist(), [2, 3])
        np.testing.assert_allclose(b[["u0", "v0", "u1", "v1"]].to_numpy(), [[0, 0, 2, 1], [1, 1, 3, 2]])
        np.testing.assert_allclose(b[["mu", "mv"]].to_numpy(), [[1.0, 0.5], [2.0, 1.5]])
        self.assertEqual(b["n_px"].tolist(), [2, 2])

    def test_same_instance_id_in_two_classes_is_two_instances(self):
        sem = np.array([[2, 3]])
        inst = np.array([[1, 1]])
        self.assertEqual(len(instance_boxes(sem, inst, classes=(2, 3))), 2)


class LinkTest(unittest.TestCase):
    def test_one_to_one_with_iou_floor(self):
        masks = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 0.0, 110.0, 10.0], [200.0, 0.0, 210.0, 10.0]])
        boxes = np.array([[0.0, 0.0, 10.0, 11.0],      # IoU 10/11 with mask 0
                          [101.0, 0.0, 111.0, 10.0],   # IoU 90/110 = 0.818 with mask 1
                          [205.0, 0.0, 215.0, 10.0]])  # IoU 50/150 with mask 2: below 0.7
        pairs = link_masks(masks, boxes, min_iou=0.7)
        self.assertEqual(sorted(pairs), [(0, 0), (1, 1)])

    def test_conflict_resolved_by_total_iou(self):
        masks = np.array([[0.0, 0.0, 10.0, 10.0], [0.5, 0.0, 10.5, 10.0]])
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0]])
        self.assertEqual(sorted(link_masks(masks, boxes, min_iou=0.7)), [(0, 0), (1, 1)])

    def test_empty_inputs(self):
        self.assertEqual(link_masks(np.zeros((0, 4)), np.array([[0.0, 0.0, 1.0, 1.0]]), 0.7), [])
        self.assertEqual(link_masks(np.array([[0.0, 0.0, 1.0, 1.0]]), np.zeros((0, 4)), 0.7), [])


if __name__ == "__main__":
    unittest.main()
