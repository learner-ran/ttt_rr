#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DATASET_ROOT = Path("/root/autodl-tmp/data_set/BDD100K_MOTS_semi/val")
OUTPUT_ROOT = Path("/root/autodl-tmp/output_bdd_semi_val")

METHOD_DIRS = {
    "nomem": OUTPUT_ROOT / "bdd_baseline_current_only",
    "single": OUTPUT_ROOT / "bdd_restricted_maskmem_m1_l1",
    "sam2": OUTPUT_ROOT / "output_bdd_semi_large",
    "dual": OUTPUT_ROOT / "bdd_restricted_maskmem_m3_l1",
}

ROW_SPECS = [
    ("original", "Original"),
    ("gt_zoom", "GT zoom"),
    ("nomem", "No-memory"),
    ("single", "Single explicit"),
    ("sam2", "SAM2"),
    ("dual", "Dual stream"),
]

PAPER_ROW_SPECS = [
    ("original", "Image"),
    ("gt_zoom", "GT zoom"),
    ("nomem", "No mem"),
    ("single", "Single"),
    ("sam2", "SAM2"),
    ("dual", "Dual"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 7x3 comparison grid for BDD semi-val predictions."
    )
    parser.add_argument(
        "--sequence",
        default="b1d22ed6-f1cac061",
        help="BDD video sequence id.",
    )
    parser.add_argument(
        "--obj",
        type=int,
        default=10,
        help="Object id inside the mask PNG.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0, 9, 88, 144],
        help="0-based frame indices to visualize.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="BDD semi-val root containing JPEGImages and Annotations.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root containing the four prediction directories.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=OUTPUT_ROOT / "visualizations" / "bdd_compare_b1d22ed6_obj010_frames_0_9_88_144",
        help="Directory for the grid PNG and all exported sub-images.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=330,
        help="Tile width in the final canvas.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=190,
        help="Tile height in the final canvas.",
    )
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="Render a cleaner paper-ready grid with compact labels and no long title.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        help="TTF/OTF font path for rendering labels.",
    )
    parser.add_argument(
        "--grid-name",
        default="grid.png",
        help="Output filename for the composed grid image inside save-dir.",
    )
    return parser.parse_args()


def frame_to_png(frame_idx: int) -> str:
    return f"{frame_idx + 1:07d}.png"


def frame_to_jpg(frame_idx: int) -> str:
    return f"{frame_idx + 1:07d}.jpg"


def load_mask(mask_path: Path, obj_id: int) -> np.ndarray:
    return np.array(Image.open(mask_path), dtype=np.uint8) == obj_id


def load_rgb(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def compute_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return float(inter / union) if union else 1.0


def compute_crop_box(mask_stack: list[np.ndarray], width: int, height: int) -> tuple[int, int, int, int]:
    union = np.zeros((height, width), dtype=bool)
    for mask in mask_stack:
        union |= mask

    ys, xs = np.where(union)
    if len(xs) == 0:
        return 0, 0, width, height

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1

    bw = x1 - x0
    bh = y1 - y0
    pad_x = max(40, int(bw * 0.6))
    pad_y = max(30, int(bh * 0.6))

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)

    return x0, y0, x1, y1


def render_tile(image: Image.Image, size: tuple[int, int], is_mask: bool) -> Image.Image:
    fill = (0, 0, 0) if is_mask else (18, 18, 18)
    canvas = Image.new("RGB", size, fill)
    method = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    fitted = ImageOps.contain(image, size, method=method)
    left = (size[0] - fitted.width) // 2
    top = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def mask_to_image(mask: np.ndarray) -> Image.Image:
    arr = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    arr[mask] = 255
    return Image.fromarray(arr, mode="RGB")


def overlay_mask_on_rgb(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = (255, 64, 64)) -> Image.Image:
    rgb = np.array(image, dtype=np.uint8)
    out = rgb.copy()
    alpha = 0.45
    out[mask] = (rgb[mask] * (1.0 - alpha) + np.array(color) * alpha).astype(np.uint8)

    # Thin contour keeps the GT readable even on bright regions.
    boundary = mask.copy()
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        inner = mask[1:-1, 1:-1]
        eroded = inner.copy()
        eroded &= mask[:-2, 1:-1]
        eroded &= mask[2:, 1:-1]
        eroded &= mask[1:-1, :-2]
        eroded &= mask[1:-1, 2:]
        boundary = mask.copy()
        boundary[1:-1, 1:-1] = inner & ~eroded
    out[boundary] = np.array((255, 255, 255), dtype=np.uint8)
    return Image.fromarray(out, mode="RGB")


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    draw.text(xy, text, font=font, fill=fill)


def main() -> None:
    args = parse_args()

    image_root = args.dataset_root / "JPEGImages" / args.sequence
    gt_root = args.dataset_root / "Annotations" / args.sequence
    method_roots = {
        "nomem": args.output_root / "bdd_baseline_current_only" / args.sequence,
        "single": args.output_root / "bdd_restricted_maskmem_m1_l1" / args.sequence,
        "sam2": args.output_root / "output_bdd_semi_large" / args.sequence,
        "dual": args.output_root / "bdd_restricted_maskmem_m3_l1" / args.sequence,
    }

    if args.font_path.exists():
        label_font = ImageFont.truetype(str(args.font_path), 22 if args.paper_style else 16)
        header_font = ImageFont.truetype(str(args.font_path), 24 if args.paper_style else 16)
        title_font = ImageFont.truetype(str(args.font_path), 24)
        metric_font = ImageFont.truetype(str(args.font_path), 16 if args.paper_style else 14)
    else:
        label_font = header_font = title_font = metric_font = ImageFont.load_default()

    row_specs = PAPER_ROW_SPECS if args.paper_style else ROW_SPECS
    title_h = 8 if args.paper_style else 42
    header_h = 44 if args.paper_style else 38
    row_h = args.tile_height
    label_w = 92 if args.paper_style else 120
    gap_x = 12 if args.paper_style else 10
    gap_y = 12 if args.paper_style else 10
    border = 24 if args.paper_style else 18

    ncols = len(args.frames)
    nrows = len(row_specs)

    canvas_w = border * 2 + label_w + ncols * args.tile_width + (ncols - 1) * gap_x
    canvas_h = (
        border * 2
        + title_h
        + header_h
        + nrows * row_h
        + (nrows - 1) * gap_y
    )
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    metrics_lines = [
        f"sequence: {args.sequence}",
        f"object_id: {args.obj:03d}",
        f"frames: {', '.join(str(frame_idx) for frame_idx in args.frames)}",
        "note: J == IoU for these binary object masks.",
        "",
    ]

    if not args.paper_style:
        title = (
            f"BDD semi-val comparison  seq={args.sequence}  obj={args.obj:03d}  "
            f"(dual: bdd_restricted_maskmem_m3_l1)"
        )
        draw_text(draw, (border, border + 4), title, title_font, (20, 20, 20))

    top_y = border + title_h

    for col, frame_idx in enumerate(args.frames):
        header_x = border + label_w + col * (args.tile_width + gap_x)
        png_name = frame_to_png(frame_idx)
        jpg_name = frame_to_jpg(frame_idx)

        gt_mask = load_mask(gt_root / png_name, args.obj)
        pred_masks = {
            name: load_mask(root / png_name, args.obj) for name, root in method_roots.items()
        }

        rgb = load_rgb(image_root / jpg_name)
        crop_box = compute_crop_box(
            [gt_mask] + [pred_masks[name] for name in ("nomem", "single", "sam2", "dual")],
            rgb.width,
            rgb.height,
        )
        full_original = render_tile(rgb, (args.tile_width, args.tile_height), is_mask=False)
        gt_overlay_zoom = render_tile(
            overlay_mask_on_rgb(rgb, gt_mask).crop(crop_box),
            (args.tile_width, args.tile_height),
            is_mask=False,
        )

        export_dir = args.save_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        frame_tag = f"frame_{frame_idx:03d}"
        full_original.save(export_dir / f"{frame_tag}_original.png")
        gt_overlay_zoom.save(export_dir / f"{frame_tag}_gt_zoom.png")
        metrics_lines.append(f"[{frame_tag}]")
        metrics_lines.append(f"{frame_tag}_original.png")
        metrics_lines.append(f"{frame_tag}_gt_zoom.png")
        for method_name in ("nomem", "single", "sam2", "dual"):
            iou = compute_iou(gt_mask, pred_masks[method_name])
            pred_tile = render_tile(
                mask_to_image(pred_masks[method_name]).crop(crop_box),
                (args.tile_width, args.tile_height),
                is_mask=True,
            )
            pred_tile.save(export_dir / f"{frame_tag}_{method_name}_mask.png")
            metrics_lines.append(
                f"{frame_tag}_{method_name}_mask.png  IoU={iou:.3f}  J={iou:.3f}"
            )
        metrics_lines.append("")

        col_title = f"({chr(ord('a') + col)}) Frame {frame_idx}" if args.paper_style else f"Frame {frame_idx}"
        draw_text(draw, (header_x + 4, top_y + 8), col_title, header_font, (35, 35, 35))

        for row, (kind, label) in enumerate(row_specs):
            tile_x = border + label_w + col * (args.tile_width + gap_x)
            tile_y = top_y + header_h + row * (row_h + gap_y)

            if col == 0:
                bbox = draw.textbbox((0, 0), label, font=label_font)
                label_y = tile_y + (row_h - (bbox[3] - bbox[1])) // 2
                draw_text(draw, (border, label_y), label, label_font, (35, 35, 35))

            if kind == "original":
                canvas.paste(full_original, (tile_x, tile_y))
                draw.rectangle((tile_x, tile_y, tile_x + args.tile_width, tile_y + args.tile_height), outline=(210, 210, 210), width=1)
                continue

            if kind == "gt_zoom":
                canvas.paste(gt_overlay_zoom, (tile_x, tile_y))
                draw.rectangle((tile_x, tile_y, tile_x + args.tile_width, tile_y + args.tile_height), outline=(210, 210, 210), width=1)
                continue

            pred_mask = pred_masks[kind]
            tile_img = mask_to_image(pred_mask).crop(crop_box)
            tile = render_tile(tile_img, (args.tile_width, args.tile_height), is_mask=True)
            canvas.paste(tile, (tile_x, tile_y))

            iou = compute_iou(gt_mask, pred_mask)
            tile_draw = ImageDraw.Draw(canvas)
            text = f"IoU {iou:.3f}"
            tx = tile_x + 8
            ty = tile_y + args.tile_height - 18
            metric_w = 84 if args.paper_style else 64
            tile_draw.rectangle((tx - 4, ty - 2, tx + metric_w, ty + 12), fill=(0, 0, 0))
            draw_text(tile_draw, (tx, ty - (1 if args.paper_style else 0)), text, metric_font, (235, 235, 235))

            draw.rectangle((tile_x, tile_y, tile_x + args.tile_width, tile_y + args.tile_height), outline=(210, 210, 210), width=1)

    grid_path = args.save_dir / args.grid_name
    canvas.save(grid_path)
    (args.save_dir / "metrics.txt").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    print(f"saved to {grid_path}")


if __name__ == "__main__":
    main()
