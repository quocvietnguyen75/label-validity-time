"""XF-C7 tier 2, step S2: the tier-1 COCO detector over Argoverse 2 ring-camera images (T2 protocol v0.1 section 6).

Same instrument as tier 1 (xfc7_ext_detect: model, weights, score floor 0.05, per-class NMS 0.5, at most 300
detections, class map), at native resolution: min_size 1550 and max_size 2048, so neither the 2048x1550 ring images
nor the 1550x2048 front-centre images are rescaled. Batches never mix cameras, so no image is padded to another
camera's shape. Reads images only: no annotation, pose or LiDAR file.

Raw table per log: <out_dir>/<log_id>.parquet with xfc7_ext_detect.RAW_COLUMNS; segment = log id, frame = image
timestamp_ns, camera = index into RING (1-based). xfc7_ext_detect.reference_table turns it into the reference.

Usage: PYTHONPATH=src python -m careport.xfc7_t2_detect <out_dir> <batch> [--per-process N] <log_dir> [<log_dir> ...]
       (child flag --max-new N; exit 75 while logs remain)"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from careport.xfc7_ext_detect import COCO_VEHICLE, EX_MORE, RAW_COLUMNS, SCORE_FLOOR, _rows, drive, resume_manifest

RING = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left", "ring_side_right",
        "ring_rear_left", "ring_rear_right"]
AV2_MIN_SIZE, AV2_MAX_SIZE = 1550, 2048


def list_images(log_dir):
    """[(log_id, camera_index, timestamp_ns, path)] for the ring cameras, sorted by camera then time."""
    log = os.path.basename(os.path.normpath(log_dir))
    out = []
    for ci, cam in enumerate(RING, 1):
        for p in glob.glob(os.path.join(log_dir, "sensors", "cameras", cam, "*.jpg")):
            out.append((log, ci, int(os.path.basename(p)[:-4]), p))
    return sorted(out, key=lambda r: (r[1], r[2]))


def camera_batches(images, batch):
    """Consecutive batches of at most `batch` images, never mixing cameras."""
    cur = []
    for r in images:
        if cur and (len(cur) == batch or cur[-1][1] != r[1]):
            yield cur
            cur = []
        cur.append(r)
    if cur:
        yield cur


def rows_for(log, camera, ts, boxes, labels, scores):
    return _rows(log, np.int64(ts), camera, boxes, labels, scores, SCORE_FLOOR)


def parse_args(argv):
    args, opts = list(argv[1:]), {}
    for flag in ("--max-new", "--per-process"):
        if flag in args:
            i = args.index(flag)
            opts[flag] = int(args[i + 1])
            del args[i:i + 2]
    if len(args) < 3:
        return None
    return args[0], int(args[1]), args[2:], opts.get("--max-new"), opts.get("--per-process")


def pending_logs(logs, out_dir):
    return [L for L in logs if not os.path.exists(os.path.join(out_dir, os.path.basename(os.path.normpath(L)) + ".parquet"))]


def main(argv):
    parsed = parse_args(argv)
    if parsed is None:
        print(__doc__)
        return 64
    out_dir, batch, logs, max_new, per_process = parsed
    os.makedirs(out_dir, exist_ok=True)
    if per_process:
        cmd = [sys.executable, "-X", "utf8", "-m", "careport.xfc7_t2_detect", out_dir, str(batch), "--max-new",
               str(per_process)] + logs
        return drive(lambda: subprocess.call(cmd), lambda: len(pending_logs(logs, out_dir)))
    import torch
    import torchvision
    from torchvision.io import decode_jpeg, read_file
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=SCORE_FLOOR, box_detections_per_img=300,
                                       min_size=AV2_MIN_SIZE, max_size=AV2_MAX_SIZE).eval().cuda()
    ckpt = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(weights.url))
    manifest = {"dataset": "av2_sensor_ring", "model": "fasterrcnn_resnet50_fpn_v2", "weights": weights.url,
                "weights_sha256": hashlib.sha256(open(ckpt, "rb").read()).hexdigest(),
                "torch": torch.__version__, "torchvision": torchvision.__version__,
                "device": torch.cuda.get_device_name(0), "min_size": AV2_MIN_SIZE, "max_size": AV2_MAX_SIZE,
                "box_score_thresh": SCORE_FLOOR, "box_detections_per_img": 300, "per_class_nms_iou": 0.5,
                "batch": batch, "classes": {str(k): v for k, v in COCO_VEHICLE.items()}, "cameras": RING,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}}
    mpath = os.path.join(out_dir, "run_manifest.json")
    manifest = resume_manifest(mpath, manifest)
    done_new = 0
    for n, L in enumerate(logs, 1):
        log = os.path.basename(os.path.normpath(L))
        seg_file = os.path.join(out_dir, log + ".parquet")
        if os.path.exists(seg_file):
            continue
        if max_new is not None and done_new >= max_new:
            break
        t0 = time.time()
        imgs = list_images(L)
        parts = []
        with torch.inference_mode():
            for b in camera_batches(imgs, batch):
                xs = [decode_jpeg(read_file(p), device="cuda").float().div(255) for _, _, _, p in b]
                for (lg, ci, ts, _), o in zip(b, model(xs)):
                    parts.append(rows_for(lg, ci, ts, o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(),
                                          o["scores"].cpu().numpy()))
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RAW_COLUMNS)
        raw.to_parquet(seg_file + ".tmp", index=False)
        os.replace(seg_file + ".tmp", seg_file)
        manifest["files"][log + ".parquet"] = {"images": len(imgs), "detections": len(raw),
                                               "sha256": hashlib.sha256(open(seg_file, "rb").read()).hexdigest()}
        print(f"[{n}/{len(logs)}] {log}: {len(imgs)} images, {len(raw)} dets, {time.time() - t0:.0f} s", flush=True)
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)
        done_new += 1
    return EX_MORE if pending_logs(logs, out_dir) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
