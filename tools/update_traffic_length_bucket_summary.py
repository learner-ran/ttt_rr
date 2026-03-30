#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
import tempfile
import zipfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET


NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ET.register_namespace("", NS)
ET.register_namespace("r", REL_NS)

BUCKETS = [
    ("1-50", 1, 50),
    ("51-100", 51, 100),
    ("101-150", 101, 150),
    ("151-200", 151, 200),
    ("201-300", 201, 300),
    ("301+", 301, 10**9),
]


def qname(tag: str) -> str:
    return f"{{{NS}}}{tag}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx",
        required=True,
        help="Path to traffic_length_bucket_editable_summary.xlsx",
    )
    parser.add_argument(
        "--dataset-root",
        default="/root/autodl-tmp/data_set/traffic_mots_semi/val",
        help="Traffic val dataset root containing JPEGImages",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Experiment spec in the form key::description::/abs/path/to/results.csv",
    )
    return parser.parse_args()


def parse_experiment_spec(spec: str):
    parts = spec.split("::", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid --experiment spec: {spec}")
    key, description, results_csv = parts
    return key.strip(), description.strip(), results_csv.strip()


def get_inline_text(cell):
    is_node = cell.find(qname("is"))
    if is_node is None:
        return ""
    return "".join(t.text or "" for t in is_node.iter(qname("t")))


def get_cell_text(cell):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return get_inline_text(cell)
    value_node = cell.find(qname("v"))
    return "" if value_node is None or value_node.text is None else value_node.text


def row_to_cell_map(row):
    mapping = {}
    for cell in row.findall(qname("c")):
        ref = cell.attrib["r"]
        col = "".join(ch for ch in ref if ch.isalpha())
        mapping[col] = cell
    return mapping


def read_video_metadata(dataset_root: Path):
    jpeg_root = dataset_root / "JPEGImages"
    video_to_bucket = {}
    bucket_stats = {
        name: {"videos": 0, "frames": 0}
        for name, _, _ in BUCKETS
    }
    for video_name in sorted(os.listdir(jpeg_root)):
        video_dir = jpeg_root / video_name
        if not video_dir.is_dir():
            continue
        frame_count = sum(
            1
            for name in os.listdir(video_dir)
            if name.lower().endswith((".jpg", ".jpeg"))
        )
        bucket_name = None
        for name, low, high in BUCKETS:
            if low <= frame_count <= high:
                bucket_name = name
                break
        if bucket_name is None:
            raise RuntimeError(f"Unable to bucket video {video_name} with {frame_count} frames")
        video_to_bucket[video_name] = (bucket_name, frame_count)
        bucket_stats[bucket_name]["videos"] += 1
        bucket_stats[bucket_name]["frames"] += frame_count
    return video_to_bucket, bucket_stats


def compute_metrics(results_csv: Path, video_to_bucket, bucket_stats):
    bucket_rows = defaultdict(list)
    global_j = global_f = global_jf = None

    with results_csv.open() as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        next(reader)
        global_row = next(reader)
        global_jf = round(float(global_row[2]), 1)
        global_j = round(float(global_row[3]), 1)
        global_f = round(float(global_row[4]), 1)

        for row in reader:
            seq = row[0].strip()
            if not seq or seq == "Global score":
                continue
            bucket_name, _ = video_to_bucket[seq]
            bucket_rows[bucket_name].append(
                {
                    "j": float(row[3]),
                    "f": float(row[4]),
                }
            )

    metrics = []
    for bucket_name, _, _ in BUCKETS:
        rows = bucket_rows[bucket_name]
        if not rows:
            raise RuntimeError(f"No rows found for bucket {bucket_name} in {results_csv}")
        j_mean = round(sum(item["j"] for item in rows) / len(rows), 2)
        f_mean = round(sum(item["f"] for item in rows) / len(rows), 2)
        metrics.append(
            {
                "bucket": bucket_name,
                "videos": bucket_stats[bucket_name]["videos"],
                "frames": bucket_stats[bucket_name]["frames"],
                "objects": len(rows),
                "j": j_mean,
                "f": f_mean,
            }
        )

    return {
        "buckets": metrics,
        "official_j": global_j,
        "official_f": global_f,
        "official_jf": global_jf,
    }


def inline_cell(ref: str, text: str):
    cell = ET.Element(qname("c"), {"r": ref, "t": "inlineStr"})
    is_node = ET.SubElement(cell, qname("is"))
    t_node = ET.SubElement(is_node, qname("t"))
    t_node.text = text
    return cell


def number_cell(ref: str, value):
    cell = ET.Element(qname("c"), {"r": ref})
    value_node = ET.SubElement(cell, qname("v"))
    value_node.text = str(value)
    return cell


def formula_cell(ref: str, formula: str):
    cell = ET.Element(qname("c"), {"r": ref})
    formula_node = ET.SubElement(cell, qname("f"))
    formula_node.text = formula
    return cell


def build_row(row_num: int, cells):
    row = ET.Element(qname("row"), {"r": str(row_num)})
    for cell in cells:
        row.append(cell)
    return row


def build_detailed_block(start_row: int, key: str, description: str, metrics):
    rows = []
    bucket_rows = metrics["buckets"]
    for offset, bucket_data in enumerate(bucket_rows):
        row_num = start_row + offset
        rows.append(
            build_row(
                row_num,
                [
                    inline_cell(f"A{row_num}", key),
                    inline_cell(f"B{row_num}", description),
                    inline_cell(f"C{row_num}", bucket_data["bucket"]),
                    number_cell(f"D{row_num}", bucket_data["videos"]),
                    number_cell(f"E{row_num}", bucket_data["frames"]),
                    number_cell(f"F{row_num}", bucket_data["objects"]),
                    number_cell(f"G{row_num}", bucket_data["j"]),
                    number_cell(f"H{row_num}", bucket_data["f"]),
                    formula_cell(f"I{row_num}", f"ROUND(AVERAGE(G{row_num}:H{row_num}),1)"),
                ],
            )
        )

    object_row = start_row + 6
    video_row = start_row + 7
    frame_row = start_row + 8
    official_ref_row = start_row + 9
    official_recomputed_row = start_row + 10
    diff_row = start_row + 11
    blank_row = start_row + 12
    first_bucket_row = start_row
    last_bucket_row = start_row + 5

    rows.append(
        build_row(
            object_row,
            [
                inline_cell(f"A{object_row}", key),
                inline_cell(f"B{object_row}", description),
                inline_cell(f"C{object_row}", "Total(Object-weighted)"),
                formula_cell(f"D{object_row}", f"SUM(D{first_bucket_row}:D{last_bucket_row})"),
                formula_cell(f"E{object_row}", f"SUM(E{first_bucket_row}:E{last_bucket_row})"),
                formula_cell(f"F{object_row}", f"SUM(F{first_bucket_row}:F{last_bucket_row})"),
                formula_cell(
                    f"G{object_row}",
                    f"ROUND(SUMPRODUCT(G{first_bucket_row}:G{last_bucket_row},F{first_bucket_row}:F{last_bucket_row})/SUM(F{first_bucket_row}:F{last_bucket_row}),1)",
                ),
                formula_cell(
                    f"H{object_row}",
                    f"ROUND(SUMPRODUCT(H{first_bucket_row}:H{last_bucket_row},F{first_bucket_row}:F{last_bucket_row})/SUM(F{first_bucket_row}:F{last_bucket_row}),1)",
                ),
                formula_cell(f"I{object_row}", f"ROUND(AVERAGE(G{object_row}:H{object_row}),1)"),
            ],
        )
    )
    rows.append(
        build_row(
            video_row,
            [
                inline_cell(f"A{video_row}", key),
                inline_cell(f"B{video_row}", description),
                inline_cell(f"C{video_row}", "Total(Video-weighted)"),
                formula_cell(f"D{video_row}", f"SUM(D{first_bucket_row}:D{last_bucket_row})"),
                formula_cell(f"E{video_row}", f"SUM(E{first_bucket_row}:E{last_bucket_row})"),
                formula_cell(
                    f"G{video_row}",
                    f"ROUND(SUMPRODUCT(G{first_bucket_row}:G{last_bucket_row},D{first_bucket_row}:D{last_bucket_row})/SUM(D{first_bucket_row}:D{last_bucket_row}),1)",
                ),
                formula_cell(
                    f"H{video_row}",
                    f"ROUND(SUMPRODUCT(H{first_bucket_row}:H{last_bucket_row},D{first_bucket_row}:D{last_bucket_row})/SUM(D{first_bucket_row}:D{last_bucket_row}),1)",
                ),
                formula_cell(f"I{video_row}", f"ROUND(AVERAGE(G{video_row}:H{video_row}),1)"),
            ],
        )
    )
    rows.append(
        build_row(
            frame_row,
            [
                inline_cell(f"A{frame_row}", key),
                inline_cell(f"B{frame_row}", description),
                inline_cell(f"C{frame_row}", "Total(Frame-weighted)"),
                formula_cell(f"D{frame_row}", f"SUM(D{first_bucket_row}:D{last_bucket_row})"),
                formula_cell(f"E{frame_row}", f"SUM(E{first_bucket_row}:E{last_bucket_row})"),
                formula_cell(
                    f"G{frame_row}",
                    f"ROUND(SUMPRODUCT(G{first_bucket_row}:G{last_bucket_row},E{first_bucket_row}:E{last_bucket_row})/SUM(E{first_bucket_row}:E{last_bucket_row}),1)",
                ),
                formula_cell(
                    f"H{frame_row}",
                    f"ROUND(SUMPRODUCT(H{first_bucket_row}:H{last_bucket_row},E{first_bucket_row}:E{last_bucket_row})/SUM(E{first_bucket_row}:E{last_bucket_row}),1)",
                ),
                formula_cell(f"I{frame_row}", f"ROUND(AVERAGE(G{frame_row}:H{frame_row}),1)"),
            ],
        )
    )
    rows.append(
        build_row(
            official_ref_row,
            [
                inline_cell(f"A{official_ref_row}", key),
                inline_cell(f"B{official_ref_row}", description),
                inline_cell(f"C{official_ref_row}", "Official Global(ref)"),
                number_cell(f"G{official_ref_row}", metrics["official_j"]),
                number_cell(f"H{official_ref_row}", metrics["official_f"]),
                number_cell(f"I{official_ref_row}", metrics["official_jf"]),
            ],
        )
    )
    rows.append(
        build_row(
            official_recomputed_row,
            [
                inline_cell(f"A{official_recomputed_row}", key),
                inline_cell(f"B{official_recomputed_row}", description),
                inline_cell(f"C{official_recomputed_row}", "Official Global(recomputed)"),
                formula_cell(f"G{official_recomputed_row}", f"G{object_row}"),
                formula_cell(f"H{official_recomputed_row}", f"H{object_row}"),
                formula_cell(f"I{official_recomputed_row}", f"I{object_row}"),
            ],
        )
    )
    rows.append(
        build_row(
            diff_row,
            [
                inline_cell(f"A{diff_row}", key),
                inline_cell(f"B{diff_row}", description),
                inline_cell(f"C{diff_row}", "Diff(Recomputed - ref)"),
                formula_cell(f"G{diff_row}", f"ROUND(G{official_recomputed_row}-G{official_ref_row},1)"),
                formula_cell(f"H{diff_row}", f"ROUND(H{official_recomputed_row}-H{official_ref_row},1)"),
                formula_cell(f"I{diff_row}", f"ROUND(I{official_recomputed_row}-I{official_ref_row},1)"),
            ],
        )
    )
    rows.append(build_row(blank_row, []))
    return rows


def build_compact_row(row_num: int, key: str, description: str, detailed_start_row: int):
    official_ref_row = detailed_start_row + 9
    official_recomputed_row = detailed_start_row + 10
    return build_row(
        row_num,
        [
            inline_cell(f"A{row_num}", key),
            inline_cell(f"B{row_num}", description),
            formula_cell(f"C{row_num}", f"Detailed!G{official_ref_row}"),
            formula_cell(f"D{row_num}", f"Detailed!H{official_ref_row}"),
            formula_cell(f"E{row_num}", f"Detailed!I{official_ref_row}"),
            formula_cell(f"F{row_num}", f"Detailed!G{official_recomputed_row}"),
            formula_cell(f"G{row_num}", f"Detailed!H{official_recomputed_row}"),
            formula_cell(f"H{row_num}", f"Detailed!I{official_recomputed_row}"),
        ],
    )


def replace_or_append_detailed(sheet_root, key: str, description: str, metrics):
    sheet_data = sheet_root.find(qname("sheetData"))
    rows = sheet_data.findall(qname("row"))
    start_row = None
    for row in rows:
        row_num = int(row.attrib["r"])
        if row_num < 9:
            continue
        cell_map = row_to_cell_map(row)
        cell_a = cell_map.get("A")
        cell_c = cell_map.get("C")
        if cell_a is None or cell_c is None:
            continue
        if get_cell_text(cell_a) == key and get_cell_text(cell_c) in {bucket for bucket, _, _ in BUCKETS}:
            start_row = row_num
            break

    if start_row is None:
        max_row = max(int(row.attrib["r"]) for row in rows)
        start_row = max_row + 1
        new_rows = build_detailed_block(start_row, key, description, metrics)
        for row in new_rows:
            sheet_data.append(row)
    else:
        new_rows = build_detailed_block(start_row, key, description, metrics)
        replacement_map = {int(row.attrib["r"]): row for row in new_rows}
        for idx, row in enumerate(list(rows)):
            row_num = int(row.attrib["r"])
            if start_row <= row_num <= start_row + 12:
                sheet_data.remove(row)
        insert_at = 0
        rows = sheet_data.findall(qname("row"))
        for i, row in enumerate(rows):
            if int(row.attrib["r"]) >= start_row:
                insert_at = i
                break
        else:
            insert_at = len(rows)
        for offset, row in enumerate(new_rows):
            sheet_data.insert(insert_at + offset, row)

    max_row = max(int(row.attrib["r"]) for row in sheet_data.findall(qname("row")))
    sheet_root.find(qname("dimension")).attrib["ref"] = f"A1:I{max_row}"
    return start_row


def replace_or_append_compact(sheet_root, key: str, description: str, detailed_start_row: int):
    sheet_data = sheet_root.find(qname("sheetData"))
    rows = sheet_data.findall(qname("row"))
    target_row_num = None
    for row in rows:
        row_num = int(row.attrib["r"])
        if row_num < 2:
            continue
        cell_map = row_to_cell_map(row)
        cell_a = cell_map.get("A")
        if cell_a is not None and get_cell_text(cell_a) == key:
            target_row_num = row_num
            break

    if target_row_num is None:
        max_row = max(int(row.attrib["r"]) for row in rows)
        target_row_num = max_row + 1
        sheet_data.append(build_compact_row(target_row_num, key, description, detailed_start_row))
    else:
        for row in list(rows):
            if int(row.attrib["r"]) == target_row_num:
                sheet_data.remove(row)
                break
        rows = sheet_data.findall(qname("row"))
        insert_at = 0
        for i, row in enumerate(rows):
            if int(row.attrib["r"]) >= target_row_num:
                insert_at = i
                break
        else:
            insert_at = len(rows)
        sheet_data.insert(insert_at, build_compact_row(target_row_num, key, description, detailed_start_row))

    max_row = max(int(row.attrib["r"]) for row in sheet_data.findall(qname("row")))
    sheet_root.find(qname("dimension")).attrib["ref"] = f"A1:H{max_row}"


def update_xlsx(xlsx_path: Path, experiments, video_to_bucket, bucket_stats):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        unpack_dir = tmpdir_path / "unzipped"
        unpack_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(xlsx_path) as archive:
            archive.extractall(unpack_dir)

        detailed_path = unpack_dir / "xl" / "worksheets" / "sheet1.xml"
        compact_path = unpack_dir / "xl" / "worksheets" / "sheet2.xml"
        detailed_root = ET.parse(detailed_path).getroot()
        compact_root = ET.parse(compact_path).getroot()

        for key, description, results_csv in experiments:
            metrics = compute_metrics(Path(results_csv), video_to_bucket, bucket_stats)
            start_row = replace_or_append_detailed(detailed_root, key, description, metrics)
            replace_or_append_compact(compact_root, key, description, start_row)

        ET.ElementTree(detailed_root).write(detailed_path, encoding="UTF-8", xml_declaration=True)
        ET.ElementTree(compact_root).write(compact_path, encoding="UTF-8", xml_declaration=True)

        tmp_xlsx = tmpdir_path / "updated.xlsx"
        with zipfile.ZipFile(tmp_xlsx, "w", zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(unpack_dir.rglob("*")):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(unpack_dir).as_posix())

        shutil.move(tmp_xlsx, xlsx_path)


def main():
    args = parse_args()
    experiments = [parse_experiment_spec(spec) for spec in args.experiment]
    if not experiments:
        raise SystemExit("No --experiment entries were provided")

    dataset_root = Path(args.dataset_root)
    video_to_bucket, bucket_stats = read_video_metadata(dataset_root)
    update_xlsx(Path(args.xlsx), experiments, video_to_bucket, bucket_stats)

    for key, _, results_csv in experiments:
        print(f"updated summary for {key}: {results_csv}")


if __name__ == "__main__":
    main()
