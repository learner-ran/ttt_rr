#!/usr/bin/env python3
import argparse
import math
from pathlib import Path


def parse_results(results_path: Path):
    seq_sum = {}
    seq_cnt = {}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue
            head = parts[0].lower()
            if head.startswith("sequence") or head.startswith("global"):
                continue
            if len(parts) < 5:
                continue
            seq = parts[0]
            try:
                jf_val = float(parts[2])
            except ValueError:
                continue
            seq_sum[seq] = seq_sum.get(seq, 0.0) + jf_val
            seq_cnt[seq] = seq_cnt.get(seq, 0) + 1
    return {s: seq_sum[s] / seq_cnt[s] for s in seq_sum}


def count_frames(img_root: Path, sequences):
    frames = {}
    missing = []
    for seq in sequences:
        seq_dir = img_root / seq
        if not seq_dir.is_dir():
            missing.append(seq)
            continue
        count = 0
        for p in seq_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                count += 1
        frames[seq] = count
    return frames, missing


def build_bins(bounds):
    bounds = sorted(bounds)
    edges = [0] + bounds + [math.inf]
    labels = []
    for i in range(1, len(edges)):
        lo = edges[i - 1] + 1 if edges[i - 1] != 0 else 0
        hi = edges[i]
        label = f"{lo}-{int(hi)}" if math.isfinite(hi) else f"{lo}+"
        labels.append(label)
    return edges, labels


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean J&F by frame-count bins from results.csv"
    )
    parser.add_argument("--results_csv", required=True, help="Path to results.csv")
    parser.add_argument("--img_root", required=True, help="Path to JPEGImages root")
    parser.add_argument(
        "--bins",
        default="30,60,90,150,250",
        help="Comma-separated upper bounds for frame-count bins",
    )
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    img_root = Path(args.img_root)
    bounds = [int(x) for x in args.bins.split(",") if x.strip()]

    seq_avg = parse_results(results_path)
    frames, missing = count_frames(img_root, seq_avg.keys())
    if missing:
        raise SystemExit(f"Missing sequences in img_root: {missing[:5]} ...")

    edges, labels = build_bins(bounds)
    bucket = {label: [] for label in labels}
    for seq, jf in seq_avg.items():
        frame_cnt = frames.get(seq, 0)
        for i in range(1, len(edges)):
            if edges[i - 1] < frame_cnt <= edges[i]:
                bucket[labels[i - 1]].append((seq, frame_cnt, jf))
                break

    print("bin,count,avg_J&F,min_frames,max_frames")
    for label in labels:
        items = bucket[label]
        if not items:
            print(f"{label},0,NA,NA,NA")
            continue
        jf_avg = sum(x[2] for x in items) / len(items)
        min_f = min(x[1] for x in items)
        max_f = max(x[1] for x in items)
        print(f"{label},{len(items)},{jf_avg:.2f},{min_f},{max_f}")


if __name__ == "__main__":
    main()
