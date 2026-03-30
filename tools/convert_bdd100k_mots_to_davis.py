#!/usr/bin/env python3
import argparse
import collections
import io
import json
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


VEHICLE_CATEGORIES = {
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "train",
    "trailer",
    "truck",
}


@dataclass
class VideoMeta:
    split: str
    video_name: str
    json_path: str
    frame_count: int
    vehicle_track_count: int


def build_palette():
    palette = []
    for i in range(256):
        palette.extend(((i * 37) % 256, (i * 67) % 256, (i * 97) % 256))
    palette[:3] = [0, 0, 0]
    return palette


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert BDD100K MOTS zips into a DAVIS-style video dataset."
    )
    parser.add_argument("--images-zip", required=True, help="Path to image zip")
    parser.add_argument("--labels-zip", required=True, help="Path to label zip")
    parser.add_argument("--output-root", required=True, help="Output dataset root")
    parser.add_argument(
        "--dataset-name",
        default="BDD100K_MOTS_vehicle",
        help="Directory name created under output-root",
    )
    parser.add_argument(
        "--vehicle-categories",
        nargs="+",
        default=sorted(VEHICLE_CATEGORIES),
        help="Categories kept in the converted dataset",
    )
    return parser.parse_args()


def frame_stem_from_name(frame_name: str) -> str:
    return Path(frame_name).stem.rsplit("-", 1)[-1]


def build_member_path(prefix: str, video_name: str, frame_name: str) -> str:
    frame_path = Path(frame_name)
    if frame_path.parent.as_posix() not in ("", "."):
        return f"{prefix}/{frame_path.as_posix()}"
    return f"{prefix}/{video_name}/{frame_name}"


def discover_videos(labels_zip: zipfile.ZipFile, vehicle_categories):
    metas = []
    total_frames = 0
    for split in ("train", "val"):
        prefix = f"bdd100k/labels/seg_track_20/polygons/{split}/"
        json_paths = sorted(
            name
            for name in labels_zip.namelist()
            if name.startswith(prefix) and name.endswith(".json")
        )
        split_total = len(json_paths)
        for idx, json_path in enumerate(json_paths, start=1):
            frames = json.loads(labels_zip.read(json_path))
            video_name = Path(json_path).stem
            vehicle_track_ids = {
                label["id"]
                for frame in frames
                for label in frame.get("labels", [])
                if label.get("category") in vehicle_categories
            }
            if not vehicle_track_ids:
                continue
            metas.append(
                VideoMeta(
                    split=split,
                    video_name=video_name,
                    json_path=json_path,
                    frame_count=len(frames),
                    vehicle_track_count=len(vehicle_track_ids),
                )
            )
            total_frames += len(frames)
            if idx % 20 == 0 or idx == split_total:
                print(
                    f"[discover:{split}] processed {idx}/{split_total} videos; "
                    f"kept so far: {len(metas)} videos, {total_frames} frames",
                    flush=True,
                )
    return metas, total_frames


def ensure_output_root(dataset_root: Path):
    dataset_root.mkdir(parents=True, exist_ok=True)


def convert_video(
    meta: VideoMeta,
    images_zip: zipfile.ZipFile,
    labels_zip: zipfile.ZipFile,
    dataset_root: Path,
    vehicle_categories,
    palette,
):
    frames = json.loads(labels_zip.read(meta.json_path))
    split_root = dataset_root / meta.split
    img_dir = split_root / "JPEGImages" / meta.video_name
    ann_dir = split_root / "Annotations" / meta.video_name
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    existing_imgs = len(list(img_dir.glob("*.jpg")))
    existing_masks = len(list(ann_dir.glob("*.png")))
    if existing_imgs == len(frames) and existing_masks == len(frames):
        return len(frames), None, True, 0
    if existing_imgs or existing_masks:
        for path in img_dir.glob("*.jpg"):
            path.unlink()
        for path in ann_dir.glob("*.png"):
            path.unlink()

    track_freq = collections.Counter()
    for frame in frames:
        for label in frame.get("labels", []):
            if label.get("category") in vehicle_categories:
                track_freq[label["id"]] += 1
    kept_track_ids = {
        track_id
        for track_id, _ in sorted(
            track_freq.items(), key=lambda item: (-item[1], item[0])
        )[:255]
    }
    dropped_track_count = max(0, len(track_freq) - len(kept_track_ids))

    track_id_map = {}
    next_track_id = 1

    for frame in frames:
        frame_name = frame["name"]
        frame_stem = frame_stem_from_name(frame_name)

        image_zip_path = build_member_path(
            f"bdd100k/images/seg_track_20/{meta.split}",
            meta.video_name,
            frame_name,
        )
        with images_zip.open(image_zip_path) as src, open(
            img_dir / f"{frame_stem}.jpg", "wb"
        ) as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

        bitmask_zip_path = build_member_path(
            f"bdd100k/labels/seg_track_20/bitmasks/{meta.split}",
            meta.video_name,
            frame_name.replace(".jpg", ".png"),
        )
        with labels_zip.open(bitmask_zip_path) as src:
            rgba = np.array(Image.open(io.BytesIO(src.read())).convert("RGBA"))

        alpha = rgba[:, :, 3]
        out_mask = np.zeros(alpha.shape, dtype=np.uint8)

        for local_idx, label in enumerate(frame.get("labels", []), start=1):
            if label.get("category") not in vehicle_categories:
                continue
            track_id = label["id"]
            if track_id not in kept_track_ids:
                continue
            if track_id not in track_id_map:
                track_id_map[track_id] = next_track_id
                next_track_id += 1
            out_mask[alpha == local_idx] = track_id_map[track_id]

        out_image = Image.fromarray(out_mask, mode="P")
        out_image.putpalette(palette)
        out_image.save(ann_dir / f"{frame_stem}.png")

    return len(frames), len(track_id_map), False, dropped_track_count


def write_split_lists(dataset_root: Path, metas):
    by_split = {"train": [], "val": []}
    for meta in metas:
        by_split[meta.split].append(meta.video_name)

    for split, videos in by_split.items():
        videos = sorted(videos)
        split_root = dataset_root / split
        (split_root / f"{split}_videos.txt").write_text(
            "\n".join(videos) + ("\n" if videos else ""),
            encoding="utf-8",
        )


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    args = parse_args()
    vehicle_categories = set(args.vehicle_categories)
    dataset_root = Path(args.output_root) / args.dataset_name
    ensure_output_root(dataset_root)

    palette = build_palette()
    start = time.time()

    with zipfile.ZipFile(args.labels_zip) as labels_zip:
        metas, total_frames = discover_videos(labels_zip, vehicle_categories)

    total_videos = len(metas)
    print(
        f"Found {total_videos} videos with kept categories "
        f"{sorted(vehicle_categories)}; total frames: {total_frames}",
        flush=True,
    )
    if total_videos == 0:
        print("No matching videos found. Nothing to convert.")
        return

    processed_frames = 0
    processed_videos = 0

    with zipfile.ZipFile(args.images_zip) as images_zip, zipfile.ZipFile(
        args.labels_zip
    ) as labels_zip:
        for meta in metas:
            video_start = time.time()
            frame_count, object_count, was_skipped, dropped_track_count = convert_video(
                meta,
                images_zip,
                labels_zip,
                dataset_root,
                vehicle_categories,
                palette,
            )
            processed_videos += 1
            processed_frames += frame_count

            elapsed = time.time() - start
            frames_per_sec = processed_frames / elapsed if elapsed > 0 else 0.0
            remaining_frames = total_frames - processed_frames
            eta = remaining_frames / frames_per_sec if frames_per_sec > 0 else 0.0
            pct = processed_frames / total_frames * 100
            if was_skipped:
                status = "skipped existing output"
            else:
                status = f"{object_count} kept objects"
                if dropped_track_count:
                    status += f", dropped {dropped_track_count} overflow tracks"
            print(
                f"[{processed_videos}/{total_videos} videos] "
                f"{meta.split}/{meta.video_name}: {frame_count} frames, "
                f"{status}, "
                f"progress {processed_frames}/{total_frames} frames ({pct:.2f}%), "
                f"video {time.time() - video_start:.1f}s, "
                f"avg {frames_per_sec:.1f} fps, ETA {format_eta(eta)}",
                flush=True,
            )

    write_split_lists(dataset_root, metas)
    total_elapsed = time.time() - start
    print(
        f"Done. Output written to {dataset_root}. "
        f"Elapsed: {format_eta(total_elapsed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
