import unittest


class KeyframeListTest(unittest.TestCase):
    def test_keyframe_camera_images_by_scene_with_camera_index(self):
        from careport.xfc7_t2_ts_detect import keyframe_images
        sensor = {"s1": "CAMERA_LEFT_FRONT", "s2": "CAMERA_RIGHT_BACK", "s3": "LIDAR_LEFT"}
        cal = {"c1": {"sensor_token": "s1"}, "c2": {"sensor_token": "s2"}, "c3": {"sensor_token": "s3"}}
        samples = {"a": {"scene_token": "S"}, "b": {"scene_token": "S"}}
        scenes = {"S": "scene-x"}
        sd = [{"calibrated_sensor_token": "c1", "is_key_frame": True, "sample_token": "a", "timestamp": 20, "filename": "f1.jpg"},
              {"calibrated_sensor_token": "c2", "is_key_frame": True, "sample_token": "a", "timestamp": 10, "filename": "f2.jpg"},
              {"calibrated_sensor_token": "c1", "is_key_frame": False, "sample_token": "a", "timestamp": 70, "filename": "f3.jpg"},
              {"calibrated_sensor_token": "c3", "is_key_frame": True, "sample_token": "a", "timestamp": 5, "filename": "f4.pcd"}]
        imgs = keyframe_images(sd, sensor, cal, samples, scenes, "/root")
        self.assertEqual(sorted((s, c, t) for s, c, t, _ in imgs), [("scene-x", 1, 20), ("scene-x", 4, 10)])

    def test_native_size(self):
        from careport.xfc7_t2_ts_detect import TS_MAX_SIZE, TS_MIN_SIZE
        self.assertEqual(min(TS_MIN_SIZE / 943, TS_MAX_SIZE / 1980), 1.0)


if __name__ == "__main__":
    unittest.main()
