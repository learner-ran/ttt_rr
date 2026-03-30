#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path


ROOT = Path("/root/autodl-tmp/data_set/traffic_mots_semi")
TRAIN_ROOT = ROOT / "train"
VAL_ROOT = ROOT / "val"
IMG_ROOT = TRAIN_ROOT / "JPEGImages"
ANN_ROOT = TRAIN_ROOT / "Annotations"
TRAIN_LIST = TRAIN_ROOT / "train_videos_split_4to1.txt"
VAL_LIST = VAL_ROOT / "val_videos.txt"
SUMMARY_CSV = ROOT / "split_4to1_summary.csv"
SUMMARY_JSON = ROOT / "split_4to1_summary.json"

BIN_SPECS = [
    ("1-50", 1, 50),
    ("51-100", 51, 100),
    ("101-150", 101, 150),
    ("151-200", 151, 200),
    ("201-300", 201, 300),
    ("301+", 301, 10**9),
]


def read_counts() -> list[tuple[str, int]]:
    videos = []
    img_dirs = sorted([p for p in IMG_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)
    ann_names = {p.name for p in ANN_ROOT.iterdir() if p.is_dir()}
    for p in img_dirs:
        if p.name not in ann_names:
            raise RuntimeError(f"Missing annotation dir for {p.name}")
        frame_count = sum(1 for f in p.iterdir() if f.is_file())
        videos.append((p.name, frame_count))
    return videos


def bin_name(frame_count: int) -> str:
    for name, lo, hi in BIN_SPECS:
        if lo <= frame_count <= hi:
            return name
    raise RuntimeError(f"No frame-count bin for {frame_count}")


def largest_remainder_quotas(videos_by_bin: dict[str, list[tuple[str, int]]], val_ratio: float) -> dict[str, int]:
    exact = {}
    floor = {}
    frac = []
    total_floor = 0
    total_target = int(round(sum(len(v) for v in videos_by_bin.values()) * val_ratio))
    for name, vids in videos_by_bin.items():
        q = len(vids) * val_ratio
        exact[name] = q
        floor[name] = math.floor(q)
        total_floor += floor[name]
        frac.append((q - floor[name], name))
    need = total_target - total_floor
    frac.sort(key=lambda x: (-x[0], x[1]))
    quotas = dict(floor)
    for _, name in frac[:need]:
        quotas[name] += 1
    return quotas


def greedy_subset(videos: list[tuple[str, int]], quota: int, target_frames: float) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    if quota <= 0:
        return [], list(videos)
    if quota >= len(videos):
        return list(videos), []

    ordered = sorted(videos, key=lambda x: (-x[1], x[0]))
    val, train = [], []
    val_frames = 0

    for idx, item in enumerate(ordered):
        name, frames = item
        remaining = len(ordered) - idx
        slots_left = quota - len(val)
        if slots_left == 0:
            train.append(item)
            continue
        if slots_left == remaining:
            val.append(item)
            val_frames += frames
            continue

        current_gap = abs(target_frames - val_frames)
        new_gap = abs(target_frames - (val_frames + frames))
        take = new_gap < current_gap

        if not take:
            train.append(item)
            continue

        val.append(item)
        val_frames += frames

    while len(val) < quota:
        train.sort(key=lambda x: (-x[1], x[0]))
        item = train.pop(0)
        val.append(item)
        val_frames += item[1]

    improved = True
    while improved:
        improved = False
        current_gap = abs(target_frames - sum(fr for _, fr in val))
        best = None
        for vi, v in enumerate(val):
            for ti, t in enumerate(train):
                new_frames = sum(fr for _, fr in val) - v[1] + t[1]
                new_gap = abs(target_frames - new_frames)
                if new_gap + 1e-9 < current_gap:
                    best = (vi, ti, new_gap)
                    current_gap = new_gap
        if best is not None:
            vi, ti, _ = best
            val[vi], train[ti] = train[ti], val[vi]
            improved = True

    return sorted(val), sorted(train)


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return
        dst.unlink()
    dst.symlink_to(src)


def write_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def main() -> None:
    videos = read_counts()
    videos_by_bin: dict[str, list[tuple[str, int]]] = {name: [] for name, _, _ in BIN_SPECS}
    for item in videos:
        videos_by_bin[bin_name(item[1])].append(item)

    quotas = largest_remainder_quotas(videos_by_bin, val_ratio=0.2)
    val_videos: list[tuple[str, int]] = []
    train_videos: list[tuple[str, int]] = []
    summary_rows = []

    for name, _, _ in BIN_SPECS:
        vids = videos_by_bin[name]
        quota = quotas[name]
        target_frames = sum(fr for _, fr in vids) * 0.2
        val_bin, train_bin = greedy_subset(vids, quota, target_frames)
        val_videos.extend(val_bin)
        train_videos.extend(train_bin)
        summary_rows.append(
            {
                "bin": name,
                "videos_total": len(vids),
                "videos_train": len(train_bin),
                "videos_val": len(val_bin),
                "frames_total": sum(fr for _, fr in vids),
                "frames_train": sum(fr for _, fr in train_bin),
                "frames_val": sum(fr for _, fr in val_bin),
            }
        )

    val_names = sorted(name for name, _ in val_videos)
    train_names = sorted(name for name, _ in train_videos)

    if set(val_names) & set(train_names):
        raise RuntimeError("Train/val overlap detected")
    if len(val_names) + len(train_names) != len(videos):
        raise RuntimeError("Split lost videos")

    (VAL_ROOT / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (VAL_ROOT / "Annotations").mkdir(parents=True, exist_ok=True)

    for name in val_names:
        ensure_symlink(IMG_ROOT / name, VAL_ROOT / "JPEGImages" / name)
        ensure_symlink(ANN_ROOT / name, VAL_ROOT / "Annotations" / name)

    existing_val_img = {p.name for p in (VAL_ROOT / "JPEGImages").iterdir()}
    existing_val_ann = {p.name for p in (VAL_ROOT / "Annotations").iterdir()}
    for stale in sorted(existing_val_img - set(val_names)):
        (VAL_ROOT / "JPEGImages" / stale).unlink()
    for stale in sorted(existing_val_ann - set(val_names)):
        (VAL_ROOT / "Annotations" / stale).unlink()

    write_list(TRAIN_LIST, train_names)
    write_list(VAL_LIST, val_names)

    total_frames = sum(fr for _, fr in videos)
    split_summary = {
        "videos_total": len(videos),
        "videos_train": len(train_names),
        "videos_val": len(val_names),
        "frames_total": total_frames,
        "frames_train": sum(fr for _, fr in train_videos),
        "frames_val": sum(fr for _, fr in val_videos),
        "video_ratio_train": len(train_names) / len(videos),
        "video_ratio_val": len(val_names) / len(videos),
        "frame_ratio_train": sum(fr for _, fr in train_videos) / total_frames,
        "frame_ratio_val": sum(fr for _, fr in val_videos) / total_frames,
        "bins": summary_rows,
    }

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin",
                "videos_total",
                "videos_train",
                "videos_val",
                "frames_total",
                "frames_train",
                "frames_val",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    SUMMARY_JSON.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    print(json.dumps(split_summary, indent=2))


if __name__ == "__main__":
    main()
