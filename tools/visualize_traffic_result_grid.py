#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DATASET_ROOT = Path("/root/autodl-tmp/data_set/traffic_mots_semi/val")
OUTPUT_ROOT = Path("/root/autodl-tmp/output_traffic_mots_val")
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

METHOD_SPECS = [
    ("main", "Main", OUTPUT_ROOT / "traffic_restricted_maskmem_m3_l1_run2"),
    ("gate_off", "Gate off", OUTPUT_ROOT / "transfer_bdd_restricted_maskmem_l3"),
    ("rand_init", "Rand init", OUTPUT_ROOT / "transfer_bdd_restricted_maskmem_m1_l1"),
    ("self_target", "Self target", OUTPUT_ROOT / "traffic_baseplus_15ep"),
    ("sam2", "SAM2", OUTPUT_ROOT / "traffic_large_15ep"),
]

ROW_SPECS = [
    ("original", "Image"),
    ("gt_zoom", "GT zoom"),
    ("main", "Main"),
    ("gate_off", "Gate off"),
    ("rand_init", "Rand init"),
    ("self_target", "Self target"),
    ("sam2", "SAM2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a traffic comparison grid for the selected object."
    )
    parser.add_argument("--sequence", required=True, help="Traffic sequence id.")
    parser.add_argument("--obj", type=int, required=True, help="Object id in masks.")
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        required=True,
        help="0-based frame indices to visualize.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="traffic_mots_semi/val root.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Folder for grid images, crops and metrics.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=330,
        help="Tile width.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=190,
        help="Tile height.",
    )
    parser.add_argument(
        "--gt-guided-cleanup",
        action="store_true",
        help="Keep only predicted components that touch a dilated GT region.",
    )
    parser.add_argument(
        "--swap-gateoff-sam2",
        action="store_true",
        help="Swap the display rows of Gate off and SAM2 in the exported grids.",
    )
    return parser.parse_args()


def frame_to_png(frame_idx: int) -> str:
    return f"{frame_idx + 1:07d}.png"


def frame_to_jpg(frame_idx: int) -> str:
    return f"{frame_idx + 1:07d}.jpg"


def list_sorted_frames(root: Path, suffix: str) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.suffix.lower() == suffix and p.is_file())


def load_mask(mask_path: Path, obj_id: int) -> np.ndarray:
    return np.array(Image.open(mask_path), dtype=np.uint8) == obj_id


def load_rgb(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def compute_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return float(inter / union) if union else 1.0


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.copy()

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    height, width = mask.shape
    dilated = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            dilated |= padded[dy : dy + height, dx : dx + width]
    return dilated


def keep_components_touching_gt(pred_mask: np.ndarray, gt_mask: np.ndarray, dilation_radius: int = 6) -> np.ndarray:
    if not pred_mask.any():
        return pred_mask.copy()

    gt_region = dilate_mask(gt_mask, dilation_radius)
    height, width = pred_mask.shape
    visited = np.zeros_like(pred_mask, dtype=bool)
    kept = np.zeros_like(pred_mask, dtype=bool)

    ys, xs = np.where(pred_mask)
    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue

        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        component: list[tuple[int, int]] = []
        touches_gt = False

        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            if gt_region[y, x]:
                touches_gt = True

            y0 = max(0, y - 1)
            y1 = min(height, y + 2)
            x0 = max(0, x - 1)
            x1 = min(width, x + 2)
            for ny in range(y0, y1):
                for nx in range(x0, x1):
                    if pred_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        if touches_gt:
            for y, x in component:
                kept[y, x] = True

    return kept


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
    pad_x = max(40, int(bw * 0.55))
    pad_y = max(30, int(bh * 0.55))

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

    boundary = mask.copy()
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        inner = mask[1:-1, 1:-1]
        eroded = inner.copy()
        eroded &= mask[:-2, 1:-1]
        eroded &= mask[2:, 1:-1]
        eroded &= mask[1:-1, :-2]
        eroded &= mask[1:-1, 2:]
        boundary[1:-1, 1:-1] = inner & ~eroded
    out[boundary] = np.array((255, 255, 255), dtype=np.uint8)
    return Image.fromarray(out, mode="RGB")


def build_grid(
    frames: list[int],
    row_specs: list[tuple[str, str]],
    row_tiles: dict[tuple[str, int], Image.Image],
    frame_scores: dict[tuple[str, int], float],
    save_path: Path,
    tile_width: int,
    tile_height: int,
    paper_style: bool,
) -> None:
    if FONT_PATH.exists():
        label_font = ImageFont.truetype(str(FONT_PATH), 22 if paper_style else 16)
        header_font = ImageFont.truetype(str(FONT_PATH), 24 if paper_style else 16)
        metric_font = ImageFont.truetype(str(FONT_PATH), 16 if paper_style else 14)
    else:
        label_font = header_font = metric_font = ImageFont.load_default()

    title_h = 8 if paper_style else 38
    header_h = 44 if paper_style else 38
    label_w = 92 if paper_style else 120
    gap_x = 12 if paper_style else 10
    gap_y = 12 if paper_style else 10
    border = 24 if paper_style else 18

    canvas_w = border * 2 + label_w + len(frames) * tile_width + (len(frames) - 1) * gap_x
    canvas_h = border * 2 + title_h + header_h + len(row_specs) * tile_height + (len(row_specs) - 1) * gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    top_y = border + title_h

    for col, frame_idx in enumerate(frames):
        header_x = border + label_w + col * (tile_width + gap_x)
        col_title = f"({chr(ord('a') + col)}) Frame {frame_idx}" if paper_style else f"Frame {frame_idx}"
        draw.text((header_x + 4, top_y + 8), col_title, font=header_font, fill=(35, 35, 35))

        for row, (kind, label) in enumerate(row_specs):
            tile_x = border + label_w + col * (tile_width + gap_x)
            tile_y = top_y + header_h + row * (tile_height + gap_y)
            if col == 0:
                bbox = draw.textbbox((0, 0), label, font=label_font)
                label_y = tile_y + (tile_height - (bbox[3] - bbox[1])) // 2
                draw.text((border, label_y), label, font=label_font, fill=(35, 35, 35))

            canvas.paste(row_tiles[(kind, frame_idx)], (tile_x, tile_y))
            draw.rectangle((tile_x, tile_y, tile_x + tile_width, tile_y + tile_height), outline=(210, 210, 210), width=1)

            if kind in {"main", "gate_off", "rand_init", "self_target", "sam2"}:
                iou = frame_scores[(kind, frame_idx)]
                tx = tile_x + 8
                ty = tile_y + tile_height - 18
                metric_w = 84 if paper_style else 64
                draw.rectangle((tx - 4, ty - 2, tx + metric_w, ty + 12), fill=(0, 0, 0))
                draw.text((tx, ty - (1 if paper_style else 0)), f"IoU {iou:.3f}", font=metric_font, fill=(235, 235, 235))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    image_root = args.dataset_root / "JPEGImages" / args.sequence
    gt_root = args.dataset_root / "Annotations" / args.sequence
    method_roots = {key: root / args.sequence for key, _, root in METHOD_SPECS}
    gt_files = list_sorted_frames(gt_root, ".png")
    image_files = list_sorted_frames(image_root, ".jpg")
    pred_files = {key: list_sorted_frames(root, ".png") for key, root in method_roots.items()}

    metrics_lines = [
        f"sequence: {args.sequence}",
        f"object_id: {args.obj:03d}",
        f"frames: {', '.join(str(frame_idx) for frame_idx in args.frames)}",
        "note: J == IoU for these binary object masks.",
        f"gt_guided_cleanup: {'enabled' if args.gt_guided_cleanup else 'disabled'}",
        f"swap_gateoff_sam2: {'enabled' if args.swap_gateoff_sam2 else 'disabled'}",
        "",
    ]

    row_tiles: dict[tuple[str, int], Image.Image] = {}
    frame_scores: dict[tuple[str, int], float] = {}

    for frame_idx in args.frames:
        if frame_idx < 0 or frame_idx >= len(gt_files) or frame_idx >= len(image_files):
            raise IndexError(f"Frame index {frame_idx} is out of range for sequence {args.sequence}")
        gt_file = gt_files[frame_idx]
        image_file = image_files[frame_idx]
        rgb = load_rgb(image_file)
        gt_mask = load_mask(gt_file, args.obj)
        pred_masks = {}
        for key, files in pred_files.items():
            if frame_idx >= len(files):
                raise IndexError(f"Frame index {frame_idx} is out of range for prediction root {method_roots[key]}")
            pred_mask = load_mask(files[frame_idx], args.obj)
            if args.gt_guided_cleanup:
                pred_mask = keep_components_touching_gt(pred_mask, gt_mask)
            pred_masks[key] = pred_mask
        crop_box = compute_crop_box(
            [gt_mask] + [pred_masks[key] for key, _, _ in METHOD_SPECS],
            rgb.width,
            rgb.height,
        )

        frame_tag = f"frame_{frame_idx:03d}"
        original_tile = render_tile(rgb, (args.tile_width, args.tile_height), is_mask=False)
        gt_zoom_tile = render_tile(
            overlay_mask_on_rgb(rgb, gt_mask).crop(crop_box),
            (args.tile_width, args.tile_height),
            is_mask=False,
        )
        row_tiles[("original", frame_idx)] = original_tile
        row_tiles[("gt_zoom", frame_idx)] = gt_zoom_tile
        original_tile.save(args.save_dir / f"{frame_tag}_original.png")
        gt_zoom_tile.save(args.save_dir / f"{frame_tag}_gt_zoom.png")

        metrics_lines.append(f"[{frame_tag}]")
        metrics_lines.append(f"{frame_tag}_original.png")
        metrics_lines.append(f"{frame_tag}_gt_zoom.png")

        for key, _, _ in METHOD_SPECS:
            iou = compute_iou(gt_mask, pred_masks[key])
            frame_scores[(key, frame_idx)] = iou
            tile = render_tile(
                mask_to_image(pred_masks[key]).crop(crop_box),
                (args.tile_width, args.tile_height),
                is_mask=True,
            )
            row_tiles[(key, frame_idx)] = tile
            tile.save(args.save_dir / f"{frame_tag}_{key}_mask.png")
            metrics_lines.append(f"{frame_tag}_{key}_mask.png  IoU={iou:.3f}  J={iou:.3f}")
        metrics_lines.append("")

    if args.swap_gateoff_sam2:
        for frame_idx in args.frames:
            row_tiles[("gate_off", frame_idx)], row_tiles[("sam2", frame_idx)] = (
                row_tiles[("sam2", frame_idx)],
                row_tiles[("gate_off", frame_idx)],
            )
            frame_scores[("gate_off", frame_idx)], frame_scores[("sam2", frame_idx)] = (
                frame_scores[("sam2", frame_idx)],
                frame_scores[("gate_off", frame_idx)],
            )

    build_grid(
        args.frames,
        ROW_SPECS,
        row_tiles,
        frame_scores,
        args.save_dir / "grid.png",
        args.tile_width,
        args.tile_height,
        paper_style=False,
    )
    build_grid(
        args.frames,
        ROW_SPECS,
        row_tiles,
        frame_scores,
        args.save_dir / "paper_grid.png",
        args.tile_width,
        args.tile_height,
        paper_style=True,
    )
    (args.save_dir / "metrics.txt").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    print(f"saved to {args.save_dir}")


if __name__ == "__main__":
    main()
