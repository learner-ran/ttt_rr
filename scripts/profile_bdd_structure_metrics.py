#!/usr/bin/env python
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


def read_video_list(video_list_file):
    with open(video_list_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_frame_names(video_dir):
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names


def load_first_frame_masks(video_dir, mask_dir, video_name):
    frame_names = get_frame_names(os.path.join(video_dir, video_name))
    first_frame_name = frame_names[0]
    mask_path = Path(mask_dir) / video_name / f"{first_frame_name}.png"
    mask = np.array(Image.open(mask_path)).astype(np.uint8)
    obj_ids = [int(x) for x in np.unique(mask) if x > 0]
    return {obj_id: (mask == obj_id) for obj_id in obj_ids}


def build_predictor(cfg, ckpt, device="cuda"):
    predictor = build_sam2_video_predictor(
        config_file=cfg,
        ckpt_path=ckpt,
        device=device,
        mode="eval",
        hydra_overrides_extra=[
            "++model.fill_hole_area=0",
        ],
    )
    return predictor


def disable_ttt_inner_update(predictor):
    model = getattr(predictor, "model", predictor)
    if not hasattr(model, "ttt_module") or model.ttt_module is None:
        return False
    model.ttt_module.should_update = lambda pred_iou, training: False
    return True


def prepare_state(predictor, base_video_dir, input_mask_dir, video_name):
    inference_state = predictor.init_state(
        video_path=os.path.join(base_video_dir, video_name),
        async_loading_frames=False,
    )
    first_masks = load_first_frame_masks(base_video_dir, input_mask_dir, video_name)
    for obj_id, obj_mask in first_masks.items():
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            mask=obj_mask,
        )
    return inference_state


def profile_single_step(predictor, inference_state):
    predictor.propagate_in_video_preflight(inference_state)
    obj_output_dict = inference_state["output_dict_per_obj"][0]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_flops=True,
            profile_memory=True,
            record_shapes=False,
        ) as prof:
            start = time.perf_counter()
            predictor._run_single_frame_inference(
                inference_state=inference_state,
                output_dict=obj_output_dict,
                frame_idx=1,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - start

    events = prof.key_averages()
    total_flops = 0
    for evt in events:
        total_flops += getattr(evt, "flops", 0) or 0

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {
        "step_time_ms": elapsed_s * 1000.0,
        "step_peak_mem_mb": peak_mem_mb,
        "step_flops": float(total_flops),
    }


def profile_full_video(predictor, inference_state):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    count = 0
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _frame_idx, _obj_ids, _video_res_masks in predictor.propagate_in_video(
            inference_state
        ):
            count += 1
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    fps = count / elapsed_s if elapsed_s > 0 else 0.0
    return {
        "num_frames_profiled": count,
        "video_elapsed_s": elapsed_s,
        "video_fps": fps,
        "video_peak_mem_mb": peak_mem_mb,
    }


def run_case(name, cfg, ckpt, base_video_dir, input_mask_dir, video_name, disable_update):
    predictor = build_predictor(cfg, ckpt)
    ttt_update_disabled = False
    if disable_update:
        ttt_update_disabled = disable_ttt_inner_update(predictor)

    state_for_step = prepare_state(predictor, base_video_dir, input_mask_dir, video_name)
    step_stats = profile_single_step(predictor, state_for_step)

    state_for_video = prepare_state(predictor, base_video_dir, input_mask_dir, video_name)
    video_stats = profile_full_video(predictor, state_for_video)

    model = getattr(predictor, "model", predictor)
    result = {
        "case": name,
        "config": cfg,
        "checkpoint": ckpt,
        "video_name": video_name,
        "num_maskmem": int(getattr(model, "num_maskmem", -1)),
        "use_full_memory": bool(getattr(model, "use_full_memory", False)),
        "has_ttt": bool(hasattr(model, "ttt_module") and model.ttt_module is not None),
        "ttt_update_disabled": bool(ttt_update_disabled),
        **step_stats,
        **video_stats,
    }
    if result["has_ttt"]:
        result["ttt_num_layers"] = int(model.ttt_module.config.num_layers)
        result["ttt_pool_size"] = int(model.ttt_module.config.pool_size)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_video_dir", required=True)
    parser.add_argument("--input_mask_dir", required=True)
    parser.add_argument("--video_list_file", required=True)
    parser.add_argument("--video_name", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    videos = read_video_list(args.video_list_file)
    video_name = args.video_name or videos[0]

    large_case = run_case(
        name="sam2_large_no_ttt",
        cfg="/root/autodl-tmp/ttt_rr/sam2/configs/sam2.1/sam2.1_hiera_l_no_ttt.yaml",
        ckpt="/root/autodl-tmp/ttt_rr/checkpoints/sam2.1_hiera_large.pt",
        base_video_dir=args.base_video_dir,
        input_mask_dir=args.input_mask_dir,
        video_name=video_name,
        disable_update=False,
    )
    ttt_case = run_case(
        name="restricted_m3_l1_ttt_no_inner_update",
        cfg="/root/autodl-tmp/ttt_rr/sam2/configs/sam2.1/sam2_ttt_inference_l_restricted_maskmem_m3_l1.yaml",
        ckpt="/root/autodl-tmp/ttt_rr/logs/sam_ttt_bdd_restricted_maskmem_m3_l1_run1/checkpoints/checkpoint.pt",
        base_video_dir=args.base_video_dir,
        input_mask_dir=args.input_mask_dir,
        video_name=video_name,
        disable_update=True,
    )

    q_large = 4096
    k_large = 7 * 4096
    k_m3 = 3 * 4096
    explicit_token_reduction_pct = (k_large - k_m3) / k_large * 100.0
    proxy_token_reduction_pct = (k_large - (k_m3 + 32 * 32)) / k_large * 100.0

    summary = {
        "video_name": video_name,
        "cases": [large_case, ttt_case],
        "derived": {
            "explicit_memory_tokens_large": k_large,
            "explicit_memory_tokens_m3": k_m3,
            "explicit_attention_qk_large": q_large * k_large,
            "explicit_attention_qk_m3": q_large * k_m3,
            "explicit_memory_reduction_pct": explicit_token_reduction_pct,
            "proxy_token_reduction_with_1layer_ttt_pct": proxy_token_reduction_pct,
            "step_peak_mem_delta_pct": (
                (ttt_case["step_peak_mem_mb"] - large_case["step_peak_mem_mb"])
                / large_case["step_peak_mem_mb"]
                * 100.0
                if large_case["step_peak_mem_mb"] > 0
                else None
            ),
            "video_peak_mem_delta_pct": (
                (ttt_case["video_peak_mem_mb"] - large_case["video_peak_mem_mb"])
                / large_case["video_peak_mem_mb"]
                * 100.0
                if large_case["video_peak_mem_mb"] > 0
                else None
            ),
            "step_flops_delta_pct": (
                (ttt_case["step_flops"] - large_case["step_flops"])
                / large_case["step_flops"]
                * 100.0
                if large_case["step_flops"] > 0
                else None
            ),
            "fps_delta_pct": (
                (ttt_case["video_fps"] - large_case["video_fps"])
                / large_case["video_fps"]
                * 100.0
                if large_case["video_fps"] > 0
                else None
            ),
        },
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md = []
    md.append(f"# BDD Structure Profiling\n")
    md.append(f"- Video: `{video_name}`\n")
    md.append(f"- Large baseline: `{large_case['config']}`\n")
    md.append(f"- TTT case: `{ttt_case['config']}` (TTT inner update disabled)\n")
    md.append("")
    md.append("| Case | num_maskmem | TTT | Step time (ms) | Step peak mem (MB) | Step FLOPs | Full-video FPS | Full-video peak mem (MB) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in [large_case, ttt_case]:
        md.append(
            f"| {item['case']} | {item['num_maskmem']} | {int(item['has_ttt'])} | "
            f"{item['step_time_ms']:.2f} | {item['step_peak_mem_mb']:.2f} | "
            f"{item['step_flops']:.0f} | {item['video_fps']:.2f} | {item['video_peak_mem_mb']:.2f} |"
        )
    md.append("")
    md.append("## Derived")
    md.append(
        f"- Explicit-memory token reduction from 7 to 3 anchors: "
        f"{summary['derived']['explicit_memory_reduction_pct']:.2f}%"
    )
    md.append(
        f"- Proxy reduction after adding 1-layer TTT on 32x32 pooled tokens: "
        f"{summary['derived']['proxy_token_reduction_with_1layer_ttt_pct']:.2f}%"
    )
    md.append(
        f"- Step FLOPs delta (TTT-disabled-structure vs large): "
        f"{summary['derived']['step_flops_delta_pct']:.2f}%"
    )
    md.append(
        f"- Step peak memory delta: "
        f"{summary['derived']['step_peak_mem_delta_pct']:.2f}%"
    )
    md.append(
        f"- Full-video peak memory delta: "
        f"{summary['derived']['video_peak_mem_delta_pct']:.2f}%"
    )
    md.append(
        f"- Full-video FPS delta: "
        f"{summary['derived']['fps_delta_pct']:.2f}%"
    )

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
