#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DATASET_ROOT = Path("/root/autodl-tmp/data_set/traffic_mots_semi")
OUTPUT_DIR = Path("/root/autodl-tmp/output_traffic_mots_val/visualizations")
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

ENTRIES = [
    {
        "dataset": "BDD100K-MOTS",
        "sequence": "00268999-0b20ef00",
        "frame": 56,
        "split": "train",
        "tag": "(a)",
    },
    {
        "dataset": "KITTI-MOTS",
        "sequence": "0000",
        "frame": 50,
        "split": "train",
        "tag": "(b)",
    },
    {
        "dataset": "DAVIS",
        "sequence": "bus",
        "frame": 40,
        "split": "train",
        "tag": "(c)",
    },
    {
        "dataset": "MOSE",
        "sequence": "4fa28c89",
        "frame": 20,
        "split": "train",
        "tag": "(d)",
    },
]

ROW_SPECS = [("image", "Image"), ("gt", "GT")]


def sorted_files(root: Path, suffix: str) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == suffix)


def load_entry(entry: dict) -> tuple[Image.Image, Image.Image]:
    split = entry["split"]
    seq = entry["sequence"]
    frame = entry["frame"]
    image_root = DATASET_ROOT / split / "JPEGImages" / seq
    ann_root = DATASET_ROOT / split / "Annotations" / seq
    image_files = sorted_files(image_root, ".jpg")
    ann_files = sorted_files(ann_root, ".png")
    if frame >= len(image_files) or frame >= len(ann_files):
        raise IndexError(f"Frame {frame} out of range for {split}/{seq}")
    image = Image.open(image_files[frame]).convert("RGB")
    ann = Image.open(ann_files[frame])
    return image, ann


def overlay_all_masks(image: Image.Image, ann: Image.Image) -> Image.Image:
    rgb = np.array(image, dtype=np.uint8)
    mask = np.array(ann, dtype=np.uint8)
    fg = (mask > 0) & (mask < 255)

    out = rgb.copy()
    alpha = 0.42
    out[fg] = (rgb[fg] * (1.0 - alpha) + np.array([255, 70, 70]) * alpha).astype(np.uint8)

    boundary = fg.copy()
    if fg.shape[0] > 2 and fg.shape[1] > 2:
        inner = fg[1:-1, 1:-1]
        eroded = inner.copy()
        eroded &= fg[:-2, 1:-1]
        eroded &= fg[2:, 1:-1]
        eroded &= fg[1:-1, :-2]
        eroded &= fg[1:-1, 2:]
        boundary[1:-1, 1:-1] = inner & ~eroded
    out[boundary] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(out, mode="RGB")


def render_tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (18, 18, 18))
    fitted = ImageOps.contain(image, size, method=Image.Resampling.BICUBIC)
    left = (size[0] - fitted.width) // 2
    top = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if FONT_PATH.exists():
        header_font = ImageFont.truetype(str(FONT_PATH), 26)
        row_font = ImageFont.truetype(str(FONT_PATH), 22)
        note_font = ImageFont.truetype(str(FONT_PATH), 16)
    else:
        header_font = row_font = note_font = ImageFont.load_default()

    tile_w, tile_h = 330, 190
    label_w = 86
    gap_x, gap_y = 14, 12
    border = 24
    header_h = 48

    row_tiles: dict[tuple[str, int], Image.Image] = {}
    manifest_lines = [
        "Traffic dataset diversity figure",
        "",
        "Selected source datasets and samples:",
    ]

    for col, entry in enumerate(ENTRIES):
        image, ann = load_entry(entry)
        gt_overlay = overlay_all_masks(image, ann)
        row_tiles[("image", col)] = render_tile(image, (tile_w, tile_h))
        row_tiles[("gt", col)] = render_tile(gt_overlay, (tile_w, tile_h))
        manifest_lines.append(
            f"{entry['dataset']}: split={entry['split']}, sequence={entry['sequence']}, frame={entry['frame']}"
        )

    canvas_w = border * 2 + label_w + len(ENTRIES) * tile_w + (len(ENTRIES) - 1) * gap_x
    canvas_h = border * 2 + header_h + len(ROW_SPECS) * tile_h + (len(ROW_SPECS) - 1) * gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    for col, entry in enumerate(ENTRIES):
        tile_x = border + label_w + col * (tile_w + gap_x)
        header_text = f"{entry['tag']} {entry['dataset']}"
        draw.text((tile_x + 4, border + 10), header_text, font=header_font, fill=(35, 35, 35))
        for row, (kind, label) in enumerate(ROW_SPECS):
            tile_y = border + header_h + row * (tile_h + gap_y)
            if col == 0:
                bbox = draw.textbbox((0, 0), label, font=row_font)
                label_y = tile_y + (tile_h - (bbox[3] - bbox[1])) // 2
                draw.text((border, label_y), label, font=row_font, fill=(35, 35, 35))

            canvas.paste(row_tiles[(kind, col)], (tile_x, tile_y))
            draw.rectangle((tile_x, tile_y, tile_x + tile_w, tile_y + tile_h), outline=(210, 210, 210), width=1)

    out_path = OUTPUT_DIR / "traffic_dataset_source_paper_grid.png"
    canvas.save(out_path)
    (OUTPUT_DIR / "traffic_dataset_source_paper_grid.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    print(out_path)


if __name__ == "__main__":
    main()
