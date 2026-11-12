"""XF-C7 extension, tier 1: a COCO-only 2D detector as the image reference (decision of 2026-10-31).

The detector was trained only on human-drawn boxes in COCO, so it has never seen a Waymo projection. Its boxes replace
Waymo's human `camera_box` rows as a drop-in table (segment, frame, camera, camera_object_id, u, v, w) for the frozen
G5 association. Two steps, kept apart:
- `main` runs the detector over the camera images and stores every COCO vehicle-class detection at score >= 0.05
  (raw table; torchvision's per-class NMS at IoU 0.5 only). It reads images only: no label, box or residual.
- `reference_table` turns the raw table into the reference: score threshold and class-agnostic NMS (a vehicle found as
  both car and truck becomes one box). Its parameters are pinned in the extension protocol, not here.
Class map: COCO car, motorcycle, bus and truck. Waymo labels motorcycles as vehicles and does not label trains or trams
(docs/labeling_specifications.md at commit 99a4cb3).

Usage: PYTHONPATH=src python -m careport.xfc7_ext_detect <camera_image_dir> <out_dir> [batch] [--per-process N]
Writes <out_dir>/<segment>.parquet per input file (skipped if present) and <out_dir>/run_manifest.json.
--per-process N runs the pass in child processes of at most N new segments each (child flag: --max-new N, exit 75
while segments remain). Reason, 31 Oct: the process's commit charge grows by about 1 GB per segment file read, so one
long process ran out of memory at segment 118. Recycling changes no setting, batch or order."""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

EX_MORE = 75       # child stopped after --max-new segments; more remain
EX_STUCK = 70      # a child finished no segment

import numpy as np
import pandas as pd

COCO_VEHICLE = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
RAW_COLUMNS = ["segment", "frame", "camera", "x1", "y1", "x2", "y2", "score", "label"]
REF_COLUMNS = ["segment", "frame", "camera", "camera_object_id", "u", "v", "w"]
SCORE_FLOOR = 0.05
IMG = "[CameraImageComponent].image"


def vehicle_detections(boxes, labels, scores, score_floor=SCORE_FLOOR):
    """Indices of COCO vehicle-class detections at or above the score floor."""
    keep = np.isin(np.asarray(labels), list(COCO_VEHICLE)) & (np.asarray(scores) >= score_floor)
    return np.flatnonzero(keep)


def class_agnostic_nms(boxes, scores, iou):
    """Greedy NMS over all classes together; returns kept indices, highest score first."""
    boxes, scores = np.asarray(boxes, float), np.asarray(scores, float)
    order = np.argsort(-scores, kind="stable")
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ov = inter / (area[i] + area[rest] - inter)
        order = rest[ov <= iou]
    return np.array(keep, dtype=int)


def _rows(segment, frame, camera, boxes, labels, scores, score_floor):
    k = vehicle_detections(boxes, labels, scores, score_floor)
    b = np.asarray(boxes, float)[k].reshape(-1, 4)
    return pd.DataFrame({"segment": segment, "frame": np.int64(frame), "camera": int(camera),
                         "x1": b[:, 0], "y1": b[:, 1], "x2": b[:, 2], "y2": b[:, 3],
                         "score": np.asarray(scores, float)[k], "label": np.asarray(labels, int)[k]},
                        columns=RAW_COLUMNS)


def detect_images(images, model, score_floor=SCORE_FLOOR):
    """images: iterable of (segment, frame, camera, H x W x 3 uint8); model(img) -> (boxes xyxy, labels, scores)."""
    parts = [_rows(s, f, c, *model(img), score_floor) for s, f, c, img in images]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RAW_COLUMNS)


def reference_table(raw, score_min, nms_iou):
    """Drop-in reference rows from the raw detections: threshold, class-agnostic NMS per image, centre and width."""
    out = []
    for (seg, frame, cam), g in raw[raw["score"] >= score_min].groupby(["segment", "frame", "camera"], sort=True):
        b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        k = class_agnostic_nms(b, g["score"].to_numpy(float), nms_iou)
        b = b[k]
        out.append(pd.DataFrame({"segment": seg, "frame": np.int64(frame), "camera": int(cam),
                                 "camera_object_id": [f"{seg}/{frame}/{cam}/{j}" for j in range(len(k))],
                                 "u": (b[:, 0] + b[:, 2]) / 2, "v": (b[:, 1] + b[:, 3]) / 2, "w": b[:, 2] - b[:, 0]},
                                columns=REF_COLUMNS))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=REF_COLUMNS)


def resume_manifest(path, fresh):
    """Continue an earlier run's manifest: keep its finished files, log each start, refuse changed settings."""
    if not os.path.exists(path):
        return dict(fresh, runs=[fresh["started"]])
    old = json.load(open(path, encoding="utf-8"))
    for k, v in fresh.items():
        if k not in ("started", "files", "runs") and old.get(k) != v:
            raise SystemExit(f"run_manifest.json setting {k!r} differs from this run ({old.get(k)!r} != {v!r}); stop")
    runs = old.get("runs", [old.get("started")])
    return dict(fresh, files=dict(old.get("files", {})), runs=runs + [fresh["started"]], started=old.get("started"))


def parse_args(argv):
    """(img_dir, out_dir, batch, max_new, per_process), or None if the arguments are wrong."""
    args, opts = list(argv[1:]), {}
    for flag in ("--max-new", "--per-process"):
        if flag in args:
            i = args.index(flag)
            opts[flag] = int(args[i + 1])
            del args[i:i + 2]
    if len(args) not in (2, 3):
        return None
    return args[0], args[1], int(args[2]) if len(args) == 3 else 4, opts.get("--max-new"), opts.get("--per-process")


def pending_files(files, out_dir):
    """Input files whose segment output does not exist yet."""
    return [f for f in files if not os.path.exists(os.path.join(out_dir, os.path.basename(f)))]


def drive(run_child, remaining):
    """Run child passes until nothing remains; stop on a failed child or a child that finishes no segment."""
    left = remaining()
    while left:
        rc = run_child()
        now = remaining()
        if rc not in (0, EX_MORE):
            return rc
        if now >= left:
            return EX_STUCK
        left = now
    return 0


def main(argv):
    parsed = parse_args(argv)
    if parsed is None:
        print(__doc__)
        return 64
    img_dir, out_dir, batch, max_new, per_process = parsed
    files = sorted(glob.glob(os.path.join(img_dir, "*.parquet")))
    if per_process:
        cmd = [sys.executable, "-X", "utf8", "-m", "careport.xfc7_ext_detect", img_dir, out_dir, str(batch),
               "--max-new", str(per_process)]
        return drive(lambda: subprocess.call(cmd), lambda: len(pending_files(files, out_dir)))
    import pyarrow.parquet as pq
    import torch
    import torchvision
    from torchvision.io import decode_jpeg
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

    os.makedirs(out_dir, exist_ok=True)
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=SCORE_FLOOR, box_detections_per_img=300,
                                       min_size=1280, max_size=1920).eval().cuda()
    ckpt = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(weights.url))
    manifest = {"model": "fasterrcnn_resnet50_fpn_v2", "weights": weights.url,
                "weights_sha256": hashlib.sha256(open(ckpt, "rb").read()).hexdigest(),
                "torch": torch.__version__, "torchvision": torchvision.__version__,
                "device": torch.cuda.get_device_name(0), "min_size": 1280, "max_size": 1920,
                "box_score_thresh": SCORE_FLOOR, "box_detections_per_img": 300, "per_class_nms_iou": 0.5,
                "classes": {str(k): v for k, v in COCO_VEHICLE.items()}, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "files": {}}
    manifest = resume_manifest(os.path.join(out_dir, "run_manifest.json"), manifest)
    done_new = 0
    for n, f in enumerate(files, 1):
        seg_file = os.path.join(out_dir, os.path.basename(f))
        if os.path.exists(seg_file):
            continue
        if max_new is not None and done_new >= max_new:
            break
        t0 = time.time()
        t = pq.read_table(f, columns=["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name", IMG])
        segs, frames, cams = (t.column(c).to_pylist() for c in ("key.segment_context_name", "key.frame_timestamp_micros",
                                                               "key.camera_name"))
        jpegs = t.column(IMG)
        parts = []
        with torch.inference_mode():
            for i in range(0, len(segs), batch):
                idx = range(i, min(i + batch, len(segs)))
                xs = [decode_jpeg(torch.frombuffer(bytearray(jpegs[j].as_py()), dtype=torch.uint8), device="cuda")
                      .float().div(255) for j in idx]
                for j, o in zip(idx, model(xs)):
                    parts.append(_rows(segs[j], frames[j], cams[j], o["boxes"].cpu().numpy(),
                                       o["labels"].cpu().numpy(), o["scores"].cpu().numpy(), SCORE_FLOOR))
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RAW_COLUMNS)
        raw.to_parquet(seg_file + ".tmp", index=False)
        os.replace(seg_file + ".tmp", seg_file)
        manifest["files"][os.path.basename(f)] = {"images": len(segs), "detections": len(raw),
                                                  "sha256": hashlib.sha256(open(seg_file, "rb").read()).hexdigest()}
        print(f"[{n}/{len(files)}] {os.path.basename(f)}: {len(segs)} images, {len(raw)} dets, {time.time() - t0:.0f} s",
              flush=True)
        with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)
        done_new += 1
    return EX_MORE if pending_files(files, out_dir) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
