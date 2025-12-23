#!/usr/bin/env python3
"""
SAM-TTT 推理/评估脚本 (DAVIS 数据集)

使用方法:
    python scripts/eval_ttt_davis.py \
        --checkpoint ./checkpoints/sam_ttt_davis.pt \
        --davis_path ./data_set/DAVIS-2017-trainval/DAVIS \
        --output_dir ./output_davis_val_ttt
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.modeling.sam_ttt.ttt_module import TTTModule, TTTConfig, TTTCache
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def load_davis_sequences(davis_path: str, split: str = "val"):
    """加载 DAVIS 数据集序列"""
    imageset_path = os.path.join(davis_path, "ImageSets", "2017", f"{split}.txt")
    
    if not os.path.exists(imageset_path):
        raise FileNotFoundError(f"DAVIS imageset not found: {imageset_path}")
    
    with open(imageset_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    
    return sequences


def load_sequence_frames(davis_path: str, seq_name: str):
    """加载序列的所有帧和标注"""
    frames_dir = os.path.join(davis_path, "JPEGImages", "480p", seq_name)
    annot_dir = os.path.join(davis_path, "Annotations", "480p", seq_name)
    
    frame_names = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    frames = []
    first_mask = None
    
    for i, fname in enumerate(frame_names):
        # 加载帧
        frame_path = os.path.join(frames_dir, fname)
        frame = np.array(Image.open(frame_path))
        frames.append(frame)
        
        # 加载第一帧的标注
        if i == 0:
            annot_name = fname.replace('.jpg', '.png')
            annot_path = os.path.join(annot_dir, annot_name)
            if os.path.exists(annot_path):
                first_mask = np.array(Image.open(annot_path))
    
    return frames, first_mask, frame_names


def compute_iou(pred: np.ndarray, gt: np.ndarray, obj_id: int = 1) -> float:
    """计算 IoU"""
    pred_binary = (pred == obj_id).astype(np.float32)
    gt_binary = (gt == obj_id).astype(np.float32)
    
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def save_mask(mask: np.ndarray, save_path: str):
    """保存 mask"""
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img.save(save_path)


def eval_sequence(
    predictor,
    frames: list,
    first_mask: np.ndarray,
    output_dir: str,
    frame_names: list,
    use_ttt: bool = True
):
    """评估单个序列"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化
    inference_state = predictor.init_state(video_path=None, images=frames)
    
    # 添加第一帧的 prompt
    obj_ids = np.unique(first_mask)
    obj_ids = obj_ids[obj_ids > 0]  # 排除背景
    
    for obj_id in obj_ids:
        mask = (first_mask == obj_id).astype(np.uint8)
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=int(obj_id),
            mask=mask
        )
    
    # 传播
    masks = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
    ):
        # out_mask_logits: [num_obj, 1, H, W]
        if out_mask_logits.shape[0] > 0:
            # 合并所有对象的 mask
            combined_mask = torch.zeros_like(out_mask_logits[0, 0])
            for i, obj_id in enumerate(out_obj_ids):
                obj_mask = (out_mask_logits[i, 0] > 0).float()
                combined_mask = torch.maximum(combined_mask, obj_mask * obj_id)
            
            masks[out_frame_idx] = combined_mask.cpu().numpy().astype(np.uint8)
    
    # 保存 masks
    for frame_idx, mask in masks.items():
        fname = frame_names[frame_idx].replace('.jpg', '.png')
        save_path = os.path.join(output_dir, fname)
        save_mask(mask, save_path)
    
    return masks


def main():
    parser = argparse.ArgumentParser(description="SAM-TTT Evaluation on DAVIS")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--davis_path", type=str, required=True, help="DAVIS dataset path")
    parser.add_argument("--output_dir", type=str, default="./output_davis_eval", help="Output directory")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--use_ttt", action="store_true", help="Enable TTT during inference")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAM-TTT Evaluation on DAVIS")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  DAVIS path: {args.davis_path}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Split: {args.split}")
    print(f"  Use TTT: {args.use_ttt}")
    print("=" * 60)
    
    # 检查路径
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.davis_path):
        print(f"Error: DAVIS path not found: {args.davis_path}")
        sys.exit(1)
    
    # 加载模型
    print("\nLoading model...")
    predictor = build_sam2_video_predictor(
        config_file="sam2/configs/sam2.1/sam2_ttt_davis.yaml",
        ckpt_path=args.checkpoint,
        device=args.device,
    )
    
    # 验证 W_init 加载
    if hasattr(predictor.model, 'ttt_module'):
        ttt = predictor.model.ttt_module
        print(f"\nTTT Module loaded:")
        print(f"  num_layers: {ttt.num_layers}")
        print(f"  W_init norms: {[f'{w.norm().item():.4f}' for w in ttt.W_init]}")
    
    # 加载序列
    sequences = load_davis_sequences(args.davis_path, args.split)
    print(f"\nFound {len(sequences)} sequences")
    
    # 评估
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    for seq_name in tqdm(sequences, desc="Evaluating"):
        seq_output_dir = os.path.join(args.output_dir, seq_name)
        
        try:
            frames, first_mask, frame_names = load_sequence_frames(args.davis_path, seq_name)
            
            if first_mask is None:
                print(f"\nWarning: No annotation for {seq_name}, skipping")
                continue
            
            masks = eval_sequence(
                predictor,
                frames,
                first_mask,
                seq_output_dir,
                frame_names,
                use_ttt=args.use_ttt
            )
            
            results[seq_name] = len(masks)
            
        except Exception as e:
            print(f"\nError processing {seq_name}: {e}")
            results[seq_name] = -1
    
    # 保存结果
    results_path = os.path.join(args.output_dir, "results.csv")
    with open(results_path, 'w') as f:
        f.write("sequence,num_frames\n")
        for seq, n in results.items():
            f.write(f"{seq},{n}\n")
    
    print(f"\nResults saved to: {results_path}")
    print("=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
