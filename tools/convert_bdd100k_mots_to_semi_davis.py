#!/usr/bin/env python3
import argparse
import json
import shutil
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


BDD_WIDTH = 1280
BDD_HEIGHT = 720


@dataclass
class VideoMeta:
    split: str
    video_name: str
    json_path: str
    frame_count: int
    first_frame_object_count: int
    nonempty_frames: int
    nonempty_ratio: float
    kept_track_ids: list[int]
    kept_track_id_map: dict[int, int]
    category_histogram: dict[str, int]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert BDD100K MOTS into a semi-supervised DAVIS-style dataset. "
            "Only objects visible in the first frame are kept, and later-appearing "
            "objects are dropped from all masks."
        )
    )
    parser.add_argument("--images-zip", required=True, help="Path to image zip")
    parser.add_argument("--labels-zip", required=True, help="Path to label zip")
    parser.add_argument("--output-root", required=True, help="Output directory")
    parser.add_argument(
        "--dataset-name",
        default="BDD100K_MOTS_semi",
        help="Directory name created under output-root",
    )
    parser.add_argument(
        "--min-nonempty-frames",
        type=int,
        default=100,
        help="Keep videos whose first-frame objects remain visible in at least this many frames",
    )
    parser.add_argument(
        "--min-nonempty-ratio",
        type=float,
        default=0.5,
        help="Keep videos whose first-frame objects remain visible in at least this frame ratio",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Optional cap for debugging; 0 means no cap",
    )
    return parser.parse_args()


def build_palette():
    palette = []
    for i in range(256):
        palette.extend(((i * 37) % 256, (i * 67) % 256, (i * 97) % 256))
    palette[:3] = [0, 0, 0]
    return palette


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def frame_stem_from_name(frame_name: str) -> str:
    return Path(frame_name).stem.rsplit("-", 1)[-1]


def build_member_path(prefix: str, video_name: str, frame_name: str) -> str:
    frame_path = Path(frame_name)
    if frame_path.parent.as_posix() not in ("", "."):
        return f"{prefix}/{frame_path.as_posix()}"
    return f"{prefix}/{video_name}/{frame_name}"


def clamp_vertices(vertices):
    clipped = []
    for x, y in vertices:
        clipped_x = min(max(int(round(x)), 0), BDD_WIDTH - 1)
        clipped_y = min(max(int(round(y)), 0), BDD_HEIGHT - 1)
        clipped.append((clipped_x, clipped_y))
    return clipped


def ordered_first_frame_track_ids(first_frame_labels):
    ordered_ids = []
    seen = set()
    for label in first_frame_labels:
        track_id = label["id"]
        if track_id in seen:
            continue
        seen.add(track_id)
        ordered_ids.append(track_id)
    return ordered_ids


def category_histogram_for_first_frame(first_frame_labels):
    counts = Counter()
    for label in first_frame_labels:
        counts[label["category"]] += 1
    return dict(sorted(counts.items()))


def discover_videos(labels_zip: zipfile.ZipFile, min_nonempty_frames: int, min_nonempty_ratio: float):
    kept = []
    skipped = []

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
            first_frame_labels = frames[0].get("labels", [])
            kept_track_ids = ordered_first_frame_track_ids(first_frame_labels)
            if not kept_track_ids:
                skipped.append(
                    {
                        "split": split,
                        "video_name": video_name,
                        "reason": "no_first_frame_objects",
                        "frame_count": len(frames),
                    }
                )
                continue

            kept_track_id_set = set(kept_track_ids)
            nonempty_frames = sum(
                any(label["id"] in kept_track_id_set for label in frame.get("labels", []))
                for frame in frames
            )
            nonempty_ratio = nonempty_frames / len(frames)
            if nonempty_frames < min_nonempty_frames or nonempty_ratio < min_nonempty_ratio:
                skipped.append(
                    {
                        "split": split,
                        "video_name": video_name,
                        "reason": "below_threshold",
                        "frame_count": len(frames),
                        "first_frame_object_count": len(kept_track_ids),
                        "nonempty_frames": nonempty_frames,
                        "nonempty_ratio": round(nonempty_ratio, 6),
                    }
                )
                continue

            kept_track_id_map = {
                track_id: mapped_id for mapped_id, track_id in enumerate(kept_track_ids, start=1)
            }
            kept.append(
                VideoMeta(
                    split=split,
                    video_name=video_name,
                    json_path=json_path,
                    frame_count=len(frames),
                    first_frame_object_count=len(kept_track_ids),
                    nonempty_frames=nonempty_frames,
                    nonempty_ratio=nonempty_ratio,
                    kept_track_ids=kept_track_ids,
                    kept_track_id_map=kept_track_id_map,
                    category_histogram=category_histogram_for_first_frame(first_frame_labels),
                )
            )

            if idx % 20 == 0 or idx == split_total:
                kept_in_split = sum(meta.split == split for meta in kept)
                print(
                    f"[discover:{split}] {idx}/{split_total} checked, kept {kept_in_split}, "
                    f"skipped {idx - kept_in_split}",
                    flush=True,
                )

    return kept, skipped


def ensure_output_dirs(dataset_root: Path):
    dataset_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (dataset_root / split / "JPEGImages").mkdir(parents=True, exist_ok=True)
        (dataset_root / split / "Annotations").mkdir(parents=True, exist_ok=True)


def draw_label_polygon(draw: ImageDraw.ImageDraw, mapped_id: int, label: dict):
    for poly in label.get("poly2d", []):
        vertices = poly.get("vertices", [])
        if len(vertices) < 3:
            continue
        draw.polygon(clamp_vertices(vertices), fill=mapped_id)


def convert_video(meta: VideoMeta, images_zip: zipfile.ZipFile, labels_zip: zipfile.ZipFile, dataset_root: Path, palette):
    frames = json.loads(labels_zip.read(meta.json_path))
    split_root = dataset_root / meta.split
    img_dir = split_root / "JPEGImages" / meta.video_name
    ann_dir = split_root / "Annotations" / meta.video_name
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    existing_imgs = len(list(img_dir.glob("*.jpg")))
    existing_masks = len(list(ann_dir.glob("*.png")))
    if existing_imgs == len(frames) and existing_masks == len(frames):
        return True
    if existing_imgs or existing_masks:
        for path in img_dir.glob("*.jpg"):
            path.unlink()
        for path in ann_dir.glob("*.png"):
            path.unlink()

    for frame in frames:
        frame_name = frame["name"]
        frame_stem = frame_stem_from_name(frame_name)

        image_member = build_member_path(
            f"bdd100k/images/seg_track_20/{meta.split}",
            meta.video_name,
            frame_name,
        )
        with images_zip.open(image_member) as src, open(img_dir / f"{frame_stem}.jpg", "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

        mask_img = Image.new("L", (BDD_WIDTH, BDD_HEIGHT), color=0)
        draw = ImageDraw.Draw(mask_img)
        for label in frame.get("labels", []):
            mapped_id = meta.kept_track_id_map.get(label["id"])
            if mapped_id is None:
                continue
            draw_label_polygon(draw, mapped_id, label)

        output_mask = mask_img.convert("P")
        output_mask.putpalette(palette)
        output_mask.save(ann_dir / f"{frame_stem}.png")

    return False


def write_split_lists(dataset_root: Path, metas: list[VideoMeta]):
    by_split = {"train": [], "val": []}
    for meta in metas:
        by_split[meta.split].append(meta.video_name)
    for split, videos in by_split.items():
        videos = sorted(videos)
        (dataset_root / split / f"{split}_videos.txt").write_text(
            "\n".join(videos) + ("\n" if videos else ""),
            encoding="utf-8",
        )


def write_summary(dataset_root: Path, metas: list[VideoMeta], skipped: list[dict], args):
    summary = {
        "dataset_name": args.dataset_name,
        "filters": {
            "min_nonempty_frames": args.min_nonempty_frames,
            "min_nonempty_ratio": args.min_nonempty_ratio,
        },
        "kept_videos": len(metas),
        "skipped_videos": len(skipped),
        "splits": {},
    }
    for split in ("train", "val"):
        split_metas = [meta for meta in metas if meta.split == split]
        summary["splits"][split] = {
            "videos": len(split_metas),
            "frames": sum(meta.frame_count for meta in split_metas),
            "avg_first_frame_objects": round(
                sum(meta.first_frame_object_count for meta in split_metas) / max(len(split_metas), 1), 4
            ),
            "avg_nonempty_ratio": round(
                sum(meta.nonempty_ratio for meta in split_metas) / max(len(split_metas), 1), 6
            ),
        }
    (dataset_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "skipped_videos.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    dataset_root = Path(args.output_root) / args.dataset_name
    ensure_output_dirs(dataset_root)
    palette = build_palette()
    start = time.time()

    with zipfile.ZipFile(args.labels_zip) as labels_zip:
        metas, skipped = discover_videos(
            labels_zip,
            min_nonempty_frames=args.min_nonempty_frames,
            min_nonempty_ratio=args.min_nonempty_ratio,
        )

    if args.max_videos > 0:
        metas = metas[: args.max_videos]

    total_videos = len(metas)
    total_frames = sum(meta.frame_count for meta in metas)
    print(
        f"Kept {total_videos} videos and {total_frames} frames for {args.dataset_name}.",
        flush=True,
    )
    if total_videos == 0:
        write_summary(dataset_root, metas, skipped, args)
        print("No videos matched the filter. Wrote summary only.", flush=True)
        return

    processed_videos = 0
    processed_frames = 0

    with zipfile.ZipFile(args.images_zip) as images_zip, zipfile.ZipFile(args.labels_zip) as labels_zip:
        for meta in metas:
            video_start = time.time()
            was_skipped = convert_video(meta, images_zip, labels_zip, dataset_root, palette)
            processed_videos += 1
            processed_frames += meta.frame_count

            elapsed = time.time() - start
            frames_per_sec = processed_frames / elapsed if elapsed > 0 else 0.0
            remaining_frames = total_frames - processed_frames
            eta = remaining_frames / frames_per_sec if frames_per_sec > 0 else 0.0
            pct = processed_frames / total_frames * 100 if total_frames else 100.0
            status = "skipped existing output" if was_skipped else (
                f"first-frame objects {meta.first_frame_object_count}, "
                f"nonempty {meta.nonempty_frames}/{meta.frame_count} ({meta.nonempty_ratio:.1%})"
            )
            print(
                f"[{processed_videos}/{total_videos} videos] "
                f"{meta.split}/{meta.video_name}: {meta.frame_count} frames, {status}, "
                f"progress {processed_frames}/{total_frames} frames ({pct:.2f}%), "
                f"video {time.time() - video_start:.1f}s, avg {frames_per_sec:.1f} fps, "
                f"ETA {format_eta(eta)}",
                flush=True,
            )

    write_split_lists(dataset_root, metas)
    write_summary(dataset_root, metas, skipped, args)
    total_elapsed = time.time() - start
    print(f"Done. Output written to {dataset_root}. Elapsed: {format_eta(total_elapsed)}", flush=True)


if __name__ == "__main__":
    main()
