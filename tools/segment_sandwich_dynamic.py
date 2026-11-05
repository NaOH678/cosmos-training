"""Causal automatic DINO + SAM2 image baseline, shared by replay and online callers.

This measures detection/segmentation quality before adding video memory. It does
not train SAM2, use fixed boxes, read future frames, or implement robot arm cut.
Object names/queries are reusable annotations; generated boxes and masks are not.
Instance masks are preserved separately from the legacy 2/3/4 semantic map.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def box_iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = float(np.maximum(hi - lo, 0).prod())
    union = float(np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod())
    return intersection / max(union - intersection, 1e-9)


class DynamicSegmenter:
    """Persistent models; process_rgb uses only its current RGB argument.

    Use this same callable/config for training replay and online validation.
    Returned instance indices are frame-local, not persistent track identities.
    """

    def __init__(self, config, dino_path, sam_checkpoint):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("This trial requires an available CUDA GPU")
        self.torch = torch
        self.config = config
        self.processor = AutoProcessor.from_pretrained(dino_path, local_files_only=True)
        self.detector = (
            AutoModelForZeroShotObjectDetection.from_pretrained(dino_path, local_files_only=True).to("cuda").eval()
        )
        self.sam = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", sam_checkpoint, device="cuda"))

    def detect(self, rgb):
        from PIL import Image

        image = Image.fromarray(rgb)
        detections = []
        missing = []
        queries = [(obj, query) for obj in self.config["objects"] for query in obj["texts"]]
        inputs = self.processor(
            images=[image] * len(queries), text=[query for _, query in queries], padding=True, return_tensors="pt"
        ).to("cuda")
        output = self.detector(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            output,
            inputs.input_ids,
            threshold=self.config["box_threshold"],
            text_threshold=self.config["text_threshold"],
            target_sizes=[rgb.shape[:2]] * len(queries),
        )
        for obj in self.config["objects"]:
            candidates = []
            for (query_obj, query), result in zip(queries, results):
                if query_obj["id"] != obj["id"]:
                    continue
                for box, score in zip(result["boxes"].cpu().numpy(), result["scores"].cpu().numpy()):
                    box = np.clip(box, [0, 0, 0, 0], [image.width, image.height] * 2)
                    if (box[2:] > box[:2]).all():
                        candidates.append(dict(box=box, score=float(score), query=query, object=obj))
            kept = []
            for candidate in sorted(candidates, key=lambda x: -x["score"]):
                if all(box_iou(candidate["box"], other["box"]) < self.config["nms_iou"] for other in kept):
                    kept.append(candidate)
                if len(kept) >= obj["max_instances"]:
                    break
            detections.extend(kept)
            if not kept:
                missing.append(obj["name"])
        return detections, missing

    def process_rgb(self, rgb):
        """Return masks, detections and missing objects; no manual/fixed-box fallback."""
        torch = self.torch
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            detections, missing = self.detect(rgb)
            torch.cuda.synchronize()
            detected = time.perf_counter()
            masks = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if detections:
                    self.sam.set_image(rgb)
                for item in detections:
                    choices, scores, _ = self.sam.predict(box=item["box"], multimask_output=True)
                    best = int(np.argmax(scores))
                    masks.append(np.asarray(choices[best], dtype=bool))
                    item["sam_score"] = float(scores[best])
        torch.cuda.synchronize()
        finished = time.perf_counter()
        masks = np.asarray(masks, dtype=bool).reshape(-1, *rgb.shape[:2])
        labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
        # Preserve all instance masks; the semantic preview follows legacy hand priority.
        for label in (4, 3, 2):
            for mask, item in zip(masks, detections):
                if item["object"]["label"] == label:
                    labels[mask] = label
        for item in detections:
            item["box"] = item["box"].tolist()
        timing = dict(dino_ms=1000 * (detected - started), sam_ms=1000 * (finished - detected))
        return labels, masks, detections, missing, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/deployment/sandwich_segmentation.json"))
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", default="all", help="all (default), or comma-separated audit frames")
    parser.add_argument("--prepare-only", action="store_true", help="extract original RGB without loading models")
    checkpoints = "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/checkpoints/"
    parser.add_argument("--dino", default=checkpoints + "grounding-dino-base")
    parser.add_argument("--sam-checkpoint", default=checkpoints + "sam2.1_hiera_large.pt")
    args = parser.parse_args()
    import cv2

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "config.json").write_bytes(config_bytes)
    segmenter = None if args.prepare_only else DynamicSegmenter(config, args.dino, args.sam_checkpoint)
    cap = cv2.VideoCapture(str(args.episode_dir / "videos/head.mp4"))
    if not cap.isOpened():
        raise RuntimeError("Cannot open episode head video")
    wanted = None if args.frames == "all" else {int(f) for f in args.frames.split(",")}
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    semantic = None
    if segmenter is not None and wanted is None:
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(args.out_dir / "segmentation.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot create segmentation video")
        semantic = np.lib.format.open_memmap(
            args.out_dir / "labels.npy", mode="w+", dtype=np.uint8, shape=(total, height, width)
        )
    records = []
    frame = 0
    while wanted is None or (wanted and frame <= max(wanted)):
        ok, bgr = cap.read()
        if not ok:
            break
        if wanted is None or frame in wanted:
            cv2.imwrite(str(args.out_dir / f"rgb_{frame:04d}.png"), bgr)
            if segmenter is not None:
                labels, masks, items, missing, timing = segmenter.process_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                np.savez_compressed(args.out_dir / f"masks_{frame:04d}.npz", labels=labels, instances=masks)
                overlay = bgr.copy()
                for mask, item in zip(masks, items):
                    color = np.random.default_rng(item["object"]["id"]).integers(40, 240, 3)
                    overlay[mask] = (0.45 * overlay[mask] + 0.55 * color).astype(np.uint8)
                    x, y, x1, y1 = map(int, item["box"])
                    cv2.rectangle(overlay, (x, y), (x1, y1), color.tolist(), 1)
                    cv2.putText(overlay, item["object"]["name"], (x, max(y - 3, 12)), 0, 0.4, color.tolist(), 1)
                cv2.imwrite(str(args.out_dir / f"preview_{frame:04d}.png"), overlay)
                if writer is not None:
                    writer.write(overlay)
                    semantic[frame] = labels
                records.append(dict(frame=frame, detections=items, missing=missing, timing=timing))
                with (args.out_dir / "progress.jsonl").open("a") as progress:
                    progress.write(json.dumps(records[-1]) + "\n")
                print(json.dumps(dict(frame=frame, missing=missing, timing=timing)), flush=True)
            else:
                records.append(dict(frame=frame, status="RGB extracted; GPU segmentation not run"))
        frame += 1
    cap.release()
    if writer is not None:
        writer.release()
        semantic.flush()
        if frame != total:
            raise RuntimeError(f"Decoded {frame} frames, expected {total}; output is incomplete")
    if not records:
        raise RuntimeError("No requested frames were processed")
    found = {r["frame"] for r in records}
    report = dict(
        episode=args.episode_dir.name,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        prepare_only=args.prepare_only,
        mode=config["mode"],
        source_fps=fps,
        checkpoint=args.sam_checkpoint,
        dino=args.dino,
        missing_requested_frames=sorted(wanted - found) if wanted is not None else [],
        limitations=[
            "No video memory or persistent instance ID",
            "No URDF arm removal",
            "No online latency validation",
        ],
        frames=records,
    )
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
