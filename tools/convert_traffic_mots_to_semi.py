#!/usr/bin/env python3
"""
Build a unified traffic-scene semi-supervised VOS dataset.

The output follows the same folder convention as BDD100K_MOTS_semi:

  <dataset_root>/
    train/
      JPEGImages/<video_name>/*.jpg
      Annotations/<video_name>/*.png
      train_videos.txt

BDD100K_MOTS_semi is copied into the output as-is.
DAVIS, MOSE, and KITTI-MOTS are converted to a DAVIS-style semi-supervised
format by keeping only the objects that already exist in the first frame.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


DAVIS_VIDEOS = [
    "bike-packing",
    "bus",
    "car-roundabout",
    "car-shadow",
    "car-turn",
    "crossing",
    "drift-chicane",
    "drift-straight",
    "drift-turn",
    "loading",
    "longboard",
    "mbike-trick",
    "motorbike",
    "night-race",
    "rallye",
    "rollerblade",
    "scooter-black",
    "scooter-board",
    "scooter-gray",
    "soapbox",
    "stroller",
    "stunt",
    "train",
    "tuk-tuk",
    "walking",
]


MOSE_VIDEOS = [
    # Road / vehicle heavy sequences observed in the sample lists.
    "07779415",
    "93723f88",
    "e71f9faa",
    "255f86ef",
    "dc54aab2",
    "198021ad",
    "c690220c",
    "3ba93641",
    "da7dee28",
    "5e80b3dd",
    "4fa28c89",
    "56b9ce41",
    "c060b1e6",
    "3f849de8",
    "ddf80bcd",
    "e35a8262",
    "e5e9eb29",
    "5488796e",
    "24df9789",
    "7ce2c865",
    "80701b6c",
    "b624a46e",
    "c6ad7aaf",
    "77ccb57d",
    "072e7b3f",
    "4d979d99",
    "75924351",
    "af5666b2",
    "a938c6b2",
    "06ebc138",
    "5949ee84",
    "5348d547",
    "c0734cb3",
    "5f150435",
    "495be321",
    "da7080cd",
    "80d978e9",
    "94d3403a",
]


KITTI_VIDEOS = [f"{idx:04d}" for idx in range(21)]


@dataclass
class ConvertedVideo:
    source: str
    video_name: str
    frame_count: int
    first_frame_object_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a unified traffic-scene semi-supervised dataset."
    )
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/data_set",
        help="Parent directory where the new dataset will be created.",
    )
    parser.add_argument(
        "--dataset-name",
        default="traffic_mots_semi",
        help="Name of the new dataset directory.",
    )
    parser.add_argument(
        "--bdd-root",
        default="/root/autodl-tmp/data_set/BDD100K_MOTS_semi",
        help="Existing BDD100K_MOTS_semi root.",
    )
    parser.add_argument(
        "--mose-root",
        default="/root/autodl-tmp/data_set/MOSE/train/train",
        help="MOSE train/train root containing JPEGImages and Annotations.",
    )
    parser.add_argument(
        "--davis-root",
        default="/root/autodl-tmp/data_set/DAVIS-2017-trainval/DAVIS",
        help="DAVIS root containing JPEGImages/Full-Resolution and Annotations/Full-Resolution.",
    )
    parser.add_argument(
        "--kitti-root",
        default="/root/autodl-tmp/data_set/KITTI_MOTS",
        help="KITTI-MOTS root containing training/image_02 and Annotations/instances.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output dataset root before writing.",
    )
    return parser.parse_args()


def build_palette() -> list[int]:
    palette = []
    for i in range(256):
        palette.extend(((i * 37) % 256, (i * 67) % 256, (i * 97) % 256))
    palette[:3] = [0, 0, 0]
    return palette


def ensure_clean_root(dataset_root: Path, overwrite: bool) -> None:
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dataset_root} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(dataset_root)
    (dataset_root / "train" / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (dataset_root / "train" / "Annotations").mkdir(parents=True, exist_ok=True)


def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def list_video_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def load_mask_array(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im)


def first_frame_ids(mask: np.ndarray) -> list[int]:
    ids = [int(v) for v in np.unique(mask) if int(v) != 0]
    return ids


def save_paletted_mask(mask: np.ndarray, dst: Path, palette: list[int]) -> None:
    img = Image.fromarray(mask.astype(np.uint8), mode="P")
    img.putpalette(palette)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def copy_bdd_dataset(bdd_root: Path, output_root: Path) -> list[ConvertedVideo]:
    converted: list[ConvertedVideo] = []
    train_root = output_root / "train"

    for split in ("train", "val"):
        img_root = bdd_root / split / "JPEGImages"
        ann_root = bdd_root / split / "Annotations"
        if not img_root.exists() or not ann_root.exists():
            continue

        for src_video_dir in list_video_dirs(img_root):
            video_name = src_video_dir.name
            src_ann_dir = ann_root / video_name
            if not src_ann_dir.exists():
                continue

            dst_img_dir = train_root / "JPEGImages" / video_name
            dst_ann_dir = train_root / "Annotations" / video_name
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_ann_dir.mkdir(parents=True, exist_ok=True)

            frame_files = sorted(src_video_dir.iterdir())
            for frame in frame_files:
                if frame.is_file():
                    safe_link_or_copy(frame, dst_img_dir / frame.name)
            for frame in sorted(src_ann_dir.iterdir()):
                if frame.is_file():
                    safe_link_or_copy(frame, dst_ann_dir / frame.name)

            converted.append(
                ConvertedVideo(
                    source=f"bdd:{split}",
                    video_name=video_name,
                    frame_count=len([p for p in frame_files if p.is_file()]),
                    first_frame_object_count=len(
                        first_frame_ids(load_mask_array(sorted(src_ann_dir.iterdir())[0]))
                    )
                    if sorted(src_ann_dir.iterdir())
                    else 0,
                )
            )

    return converted


def convert_davis_like_dataset(
    *,
    source_name: str,
    video_names: list[str],
    images_root: Path,
    masks_root: Path,
    output_root: Path,
    palette: list[int],
    source_image_ext: str = ".jpg",
    output_image_ext: str = ".jpg",
) -> list[ConvertedVideo]:
    converted: list[ConvertedVideo] = []
    train_img_root = output_root / "train" / "JPEGImages"
    train_ann_root = output_root / "train" / "Annotations"

    for video_name in video_names:
        src_img_dir = images_root / video_name
        src_ann_dir = masks_root / video_name
        if not src_img_dir.exists() or not src_ann_dir.exists():
            continue

        src_images = sorted(src_img_dir.glob(f"*{source_image_ext}"))
        src_masks = sorted(src_ann_dir.glob("*.png"))
        if not src_images or not src_masks:
            continue

        first_mask = load_mask_array(src_masks[0])
        keep_ids = first_frame_ids(first_mask)
        if not keep_ids:
            continue

        id_map = {old_id: new_id for new_id, old_id in enumerate(keep_ids, start=1)}
        dst_img_dir = train_img_root / video_name
        dst_ann_dir = train_ann_root / video_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_ann_dir.mkdir(parents=True, exist_ok=True)

        for src_img in src_images:
            dst_img = dst_img_dir / (src_img.stem + output_image_ext)
            safe_link_or_copy(src_img, dst_img)

        for src_mask in src_masks:
            mask = load_mask_array(src_mask)
            out = np.zeros_like(mask, dtype=np.uint8)
            for old_id, new_id in id_map.items():
                out[mask == old_id] = new_id
            save_paletted_mask(out, dst_ann_dir / src_mask.name, palette)

        converted.append(
            ConvertedVideo(
                source=source_name,
                video_name=video_name,
                frame_count=len(src_masks),
                first_frame_object_count=len(keep_ids),
            )
        )

    return converted


def convert_kitti_mots(
    kitti_root: Path, output_root: Path, palette: list[int], video_names: list[str]
) -> list[ConvertedVideo]:
    converted: list[ConvertedVideo] = []
    src_img_root = kitti_root / "training" / "image_02"
    src_ann_root = kitti_root / "Annotations" / "instances"
    train_img_root = output_root / "train" / "JPEGImages"
    train_ann_root = output_root / "train" / "Annotations"

    for video_name in video_names:
        src_img_dir = src_img_root / video_name
        src_ann_dir = src_ann_root / video_name
        if not src_img_dir.exists() or not src_ann_dir.exists():
            continue

        src_images = sorted(src_img_dir.glob("*.png"))
        src_masks = sorted(src_ann_dir.glob("*.png"))
        if not src_images or not src_masks:
            continue

        first_mask = load_mask_array(src_masks[0])
        keep_ids = first_frame_ids(first_mask)
        if not keep_ids:
            continue

        id_map = {old_id: new_id for new_id, old_id in enumerate(keep_ids, start=1)}
        dst_img_dir = train_img_root / video_name
        dst_ann_dir = train_ann_root / video_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_ann_dir.mkdir(parents=True, exist_ok=True)

        for src_img in src_images:
            with Image.open(src_img) as im:
                im.convert("RGB").save(dst_img_dir / f"{src_img.stem}.jpg", quality=95)

        for src_mask in src_masks:
            mask = load_mask_array(src_mask)
            out = np.zeros_like(mask, dtype=np.uint8)
            for old_id, new_id in id_map.items():
                out[mask == old_id] = new_id
            save_paletted_mask(out, dst_ann_dir / src_mask.name, palette)

        converted.append(
            ConvertedVideo(
                source="kitti_mots",
                video_name=video_name,
                frame_count=len(src_masks),
                first_frame_object_count=len(keep_ids),
            )
        )

    return converted


def build_summary(videos: list[ConvertedVideo]) -> dict:
    by_source: dict[str, list[ConvertedVideo]] = {}
    for video in videos:
        by_source.setdefault(video.source, []).append(video)

    summary = {
        "total_videos": len(videos),
        "sources": {},
    }
    for source, items in sorted(by_source.items()):
        summary["sources"][source] = {
            "videos": len(items),
            "frames": sum(v.frame_count for v in items),
            "avg_first_frame_objects": round(
                sum(v.first_frame_object_count for v in items) / max(len(items), 1), 4
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root) / args.dataset_name
    ensure_clean_root(output_root, overwrite=args.overwrite)
    palette = build_palette()
    start = time.time()

    all_videos: list[ConvertedVideo] = []

    print("Copying BDD100K_MOTS_semi into the unified train pool...", flush=True)
    all_videos.extend(copy_bdd_dataset(Path(args.bdd_root), output_root))

    print("Converting DAVIS traffic scenes...", flush=True)
    all_videos.extend(
        convert_davis_like_dataset(
            source_name="davis",
            video_names=DAVIS_VIDEOS,
            images_root=Path(args.davis_root) / "JPEGImages" / "Full-Resolution",
            masks_root=Path(args.davis_root) / "Annotations" / "Full-Resolution",
            output_root=output_root,
            palette=palette,
        )
    )

    print("Converting MOSE traffic scenes...", flush=True)
    all_videos.extend(
        convert_davis_like_dataset(
            source_name="mose",
            video_names=MOSE_VIDEOS,
            images_root=Path(args.mose_root) / "JPEGImages",
            masks_root=Path(args.mose_root) / "Annotations",
            output_root=output_root,
            palette=palette,
        )
    )

    print("Converting KITTI-MOTS training split...", flush=True)
    all_videos.extend(
        convert_kitti_mots(Path(args.kitti_root), output_root, palette, KITTI_VIDEOS)
    )

    combined_names = sorted({video.video_name for video in all_videos})
    write_text(output_root / "train" / "train_videos.txt", combined_names)
    write_text(output_root / "train_videos.txt", combined_names)

    source_manifest = {
        "bdd_videos": sorted(
            [v.video_name for v in all_videos if v.source.startswith("bdd:")]
        ),
        "davis_videos": sorted([v.video_name for v in all_videos if v.source == "davis"]),
        "mose_videos": sorted([v.video_name for v in all_videos if v.source == "mose"]),
        "kitti_videos": sorted([v.video_name for v in all_videos if v.source == "kitti_mots"]),
    }
    (output_root / "selected_videos.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "conversion_summary.json").write_text(
        json.dumps(build_summary(all_videos), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Short README for future reuse.
    (output_root / "README.md").write_text(
        "# traffic_mots_semi\n\n"
        "Unified traffic-scene semi-supervised VOS dataset.\n\n"
        "- BDD100K_MOTS_semi: copied as-is.\n"
        "- DAVIS / MOSE / KITTI-MOTS: converted to keep only the objects present in frame 0.\n"
        "- Everything is flattened into `train/` so it can be split later as needed.\n",
        encoding="utf-8",
    )

    elapsed = time.time() - start
    print(f"Done. Wrote {len(all_videos)} videos to {output_root} in {elapsed:.1f}s.", flush=True)
    print((output_root / "conversion_summary.json").as_posix(), flush=True)


if __name__ == "__main__":
    main()
