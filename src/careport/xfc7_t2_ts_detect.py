"""XF-C7 tier 2, step S2 for TruckScenes: the tier-1 COCO detector over keyframe camera images (descriptive branch).

Same instrument as tier 1 and xfc7_t2_detect, at native resolution (min_size 943, max_size 1980; images 1980x943).
Keyframes only, because boxes exist only on 2 Hz samples. Reads images and the sample_data/sensor tables only.
Raw table per scene: <out_dir>/<scene>.parquet with xfc7_ext_detect.RAW_COLUMNS (segment = scene name,
frame = image timestamp in microseconds, camera = index into TS_CAMERAS).

Usage: PYTHONPATH=src python -m careport.xfc7_t2_ts_detect <truckscenes_root> <out_dir> [batch]"""
import glob
import hashlib
import json
import os
import sys
import time

import pandas as pd

from careport.xfc7_ext_detect import COCO_VEHICLE, RAW_COLUMNS, SCORE_FLOOR
from careport.xfc7_t2_detect import camera_batches, rows_for
from careport.xfc7_t2_ts_labels import TS_CAMERAS

TS_MIN_SIZE, TS_MAX_SIZE = 943, 1980


def keyframe_images(sample_data, sensor, cal, samples, scene_names, root):
    """[(scene_name, camera_index, timestamp, path)] for keyframe camera images."""
    out = []
    for sd in sample_data:
        if not sd["is_key_frame"]:
            continue
        ch = sensor[cal[sd["calibrated_sensor_token"]]["sensor_token"]]
        if ch not in TS_CAMERAS:
            continue
        scene = scene_names[samples[sd["sample_token"]]["scene_token"]]
        out.append((scene, TS_CAMERAS.index(ch) + 1, int(sd["timestamp"]), os.path.join(root, sd["filename"])))
    return sorted(out, key=lambda r: (r[0], r[1], r[2]))


def main(argv):
    if len(argv) not in (3, 4):
        print(__doc__)
        return 64
    root, out_dir = argv[1], argv[2]
    batch = int(argv[3]) if len(argv) == 4 else 4
    meta = glob.glob(os.path.join(root, "v1.*"))[0]
    L = lambda n: json.load(open(os.path.join(meta, n + ".json"), encoding="utf-8"))
    sensor = {s["token"]: s["channel"] for s in L("sensor")}
    cal = {c["token"]: c for c in L("calibrated_sensor")}
    samples = {s["token"]: s for s in L("sample")}
    scene_names = {s["token"]: s["name"] for s in L("scene")}
    imgs = keyframe_images(L("sample_data"), sensor, cal, samples, scene_names, root)
    import torch
    import torchvision
    from torchvision.io import decode_jpeg, read_file
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=SCORE_FLOOR, box_detections_per_img=300,
                                       min_size=TS_MIN_SIZE, max_size=TS_MAX_SIZE).eval().cuda()
    ckpt = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(weights.url))
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"dataset": "truckscenes_keyframes", "weights_sha256": hashlib.sha256(open(ckpt, "rb").read()).hexdigest(),
                "torch": torch.__version__, "torchvision": torchvision.__version__, "min_size": TS_MIN_SIZE,
                "max_size": TS_MAX_SIZE, "box_score_thresh": SCORE_FLOOR, "box_detections_per_img": 300,
                "per_class_nms_iou": 0.5, "batch": batch, "classes": {str(k): v for k, v in COCO_VEHICLE.items()},
                "cameras": TS_CAMERAS, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}}
    by_scene = {}
    for r in imgs:
        by_scene.setdefault(r[0], []).append(r)
    for scene, items in sorted(by_scene.items()):
        f = os.path.join(out_dir, scene + ".parquet")
        parts = []
        with torch.inference_mode():
            for b in camera_batches(items, batch):
                xs = [decode_jpeg(read_file(p), device="cuda").float().div(255) for _, _, _, p in b]
                for (sc, ci, ts, _), o in zip(b, model(xs)):
                    parts.append(rows_for(sc, ci, ts, o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy()))
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RAW_COLUMNS)
        raw.to_parquet(f, index=False)
        manifest["files"][scene + ".parquet"] = {"images": len(items), "detections": len(raw),
                                                 "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()}
        print(scene, len(items), len(raw), flush=True)
    json.dump(manifest, open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
