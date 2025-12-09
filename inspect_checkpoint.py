import torch
import os
import argparse
import sys

def inspect_checkpoint(ckpt_path):
    print(f"Inspecting checkpoint: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    try:
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model" in ckpt:
            state_dict = ckpt["model"]
            print("Loaded state_dict from 'model' key.")
        else:
            state_dict = ckpt
            print("Loaded state_dict directly.")
        
        print(f"Total keys in checkpoint: {len(state_dict)}")
        
        # Filter keys
        me_keys = [k for k in state_dict.keys() if "ME" in k]
        routefuse_keys = [k for k in state_dict.keys() if "routefuse" in k]
        decoder_keys = [k for k in state_dict.keys() if "sam_mask_decoder" in k]
        image_encoder_keys = [k for k in state_dict.keys() if "image_encoder" in k]
        
        print("-" * 30)
        print(f"ME (Mix Embedding) keys found: {len(me_keys)}")
        print(f"RouteFuse keys found: {len(routefuse_keys)}")
        print(f"Mask Decoder keys found: {len(decoder_keys)}")
        print(f"Image Encoder keys found: {len(image_encoder_keys)}")
        print("-" * 30)
        
        # Check values
        if len(me_keys) > 0:
            key = me_keys[0]
            val = state_dict[key].float()
            print(f"Sample ME Key: {key}")
            print(f"  Shape: {val.shape}")
            print(f"  Mean value: {val.mean().item():.6f}")
            print(f"  Std dev: {val.std().item():.6f}")
            if val.std().item() == 0:
                print("  WARNING: Weights appear to be all same (possibly uninitialized or frozen zero).")
            
            # Check for BatchNorm stats
            bn_running_mean = [k for k in me_keys if "running_mean" in k]
            bn_running_var = [k for k in me_keys if "running_var" in k]
            
            print(f"ME BatchNorm running_mean keys: {len(bn_running_mean)}")
            if len(bn_running_mean) > 0:
                rm_key = bn_running_mean[0]
                rm_val = state_dict[rm_key].float()
                print(f"Sample BN running_mean ({rm_key}):")
                print(f"  Mean: {rm_val.mean().item():.6f}, Std: {rm_val.std().item():.6f}")
                print(f"  Values: {rm_val[:5].tolist()}...")
                
            if len(bn_running_var) > 0:
                rv_key = bn_running_var[0]
                rv_val = state_dict[rv_key].float()
                print(f"Sample BN running_var ({rv_key}):")
                print(f"  Mean: {rv_val.mean().item():.6f}, Std: {rv_val.std().item():.6f}")
                print(f"  Values: {rv_val[:5].tolist()}...")
        else:
            print("WARNING: No ME keys found! This checkpoint might be the base model, not the trained one.")

        if len(routefuse_keys) > 0:
            key = routefuse_keys[0]
            val = state_dict[key].float()
            print(f"Sample RouteFuse Key: {key}")
            print(f"  Mean value: {val.mean().item():.6f}")
        
        print("-" * 30)

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect SAM2 Checkpoint for TTT modules")
    parser.add_argument("ckpt_path", type=str, help="Path to the .pt checkpoint file")
    args = parser.parse_args()
    
    inspect_checkpoint(args.ckpt_path)
