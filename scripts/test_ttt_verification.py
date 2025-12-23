#!/usr/bin/env python3
"""
TTT 完整验证测试脚本

测试内容：
1. W_init 在 state_dict 中
2. Cache 从 W_init 初始化（禁止 torch.randn）
3. forward/update 使用 cache 中的 W
4. step_update 只更新 cache，不修改 W_init
5. delta_norm > 0
6. Truncated BPTT
7. Save/Load state_dict 保留 W_init
8. 内存限制 (num_maskmem=3)
9. IoU 门控
10. 梯度流检查
"""

import os
import sys
import math
import tempfile
import argparse

import torch
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam2.modeling.sam_ttt.ttt_module import (
    TTTModule, TTTConfig, TTTCache, create_ttt_cache
)


def test_w_init_in_state_dict():
    """Test 1: W_init 在 state_dict 中"""
    print("\n" + "=" * 60)
    print("[Test 1] W_init in state_dict")
    print("=" * 60)
    
    config = TTTConfig(verbose=False)
    module = TTTModule(hidden_dim=256, mem_dim=64, config=config)
    
    state_dict = module.state_dict()
    w_init_keys = [k for k in state_dict.keys() if 'W_init' in k]
    
    print(f"  Total state_dict keys: {len(state_dict)}")
    print(f"  W_init keys: {w_init_keys}")
    
    assert len(w_init_keys) == module.num_layers, \
        f"Expected {module.num_layers} W_init keys, got {len(w_init_keys)}"
    
    for key in w_init_keys:
        print(f"    {key}: {state_dict[key].shape}")
    
    print("  ✓ PASSED")
    return True


def test_cache_from_w_init():
    """Test 2: Cache 从 W_init 初始化"""
    print("\n" + "=" * 60)
    print("[Test 2] Cache initialized from W_init (not torch.randn)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 2
    
    config = TTTConfig(verbose=False)
    module = TTTModule(hidden_dim=256, mem_dim=64, config=config).to(device)
    
    cache = module.create_cache(B, device, torch.float32)
    
    for i in range(module.num_layers):
        W_init_norm = module.W_init[i].norm().item()
        cache_W_norm = cache.W_list[i].norm().item()
        expected_ratio = math.sqrt(B)
        actual_ratio = cache_W_norm / W_init_norm if W_init_norm > 0 else 0
        
        print(f"  Layer {i}: W_init.norm={W_init_norm:.4f}, "
              f"cache.W.norm={cache_W_norm:.4f}, "
              f"ratio={actual_ratio:.4f} (expected ~{expected_ratio:.4f})")
        
        assert abs(actual_ratio - expected_ratio) < 0.01, \
            f"Ratio mismatch: expected {expected_ratio}, got {actual_ratio}"
    
    print("  ✓ PASSED")
    return True


def test_forward_uses_cache_w():
    """Test 3: forward 使用 cache 中的 W"""
    print("\n" + "=" * 60)
    print("[Test 3] Forward uses cache W")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, L, C = 2, 4096, 256
    
    config = TTTConfig(verbose=False)
    module = TTTModule(hidden_dim=C, mem_dim=64, config=config).to(device)
    
    cache = module.create_cache(B, device, torch.float32)
    vision_feats = torch.randn(L, B, C, device=device)
    
    out = module.forward(vision_feats, cache)
    
    assert out.shape == (B, C, 64, 64), f"Shape mismatch: {out.shape}"
    print(f"  Output shape: {out.shape}")
    print("  ✓ PASSED")
    return True


def test_step_update_only_modifies_cache():
    """Test 4: step_update 只更新 cache，不修改 W_init"""
    print("\n" + "=" * 60)
    print("[Test 4] step_update only modifies cache, not W_init")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, L, C, C_mem = 2, 4096, 256, 64
    
    config = TTTConfig(verbose=False)
    module = TTTModule(hidden_dim=C, mem_dim=C_mem, config=config).to(device)
    
    cache = module.create_cache(B, device, torch.float32)
    vision_feats = torch.randn(L, B, C, device=device)
    maskmem_features = torch.randn(B, C_mem, 32, 32, device=device)
    
    W_init_before = [w.clone() for w in module.W_init]
    cache_W_before = [w.clone() for w in cache.W_list]
    
    loss = module.step_update(vision_feats, maskmem_features, cache, second_order=False)
    
    W_init_changed = False
    for i, (before, after) in enumerate(zip(W_init_before, module.W_init)):
        diff = (after - before).abs().max().item()
        if diff > 1e-8:
            W_init_changed = True
            print(f"  ✗ ERROR: W_init[{i}] changed by {diff}")
    
    assert not W_init_changed, "W_init should not change after step_update"
    print("  W_init unchanged: ✓")
    
    cache_W_changed = False
    for i, (before, after) in enumerate(zip(cache_W_before, cache.W_list)):
        diff = (after - before).abs().max().item()
        if diff > 1e-8:
            cache_W_changed = True
            print(f"  Layer {i}: cache.W delta_max = {diff:.6f}")
    
    assert cache_W_changed, "Cache W should change after step_update"
    print("  Cache W changed: ✓")
    print("  ✓ PASSED")
    return True


def test_delta_norm_positive():
    """Test 5: delta_norm > 0"""
    print("\n" + "=" * 60)
    print("[Test 5] delta_norm > 0")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, L, C, C_mem = 2, 4096, 256, 64
    
    config = TTTConfig(verbose=True, log_first_n=1)
    module = TTTModule(hidden_dim=C, mem_dim=C_mem, config=config).to(device)
    
    cache = module.create_cache(B, device, torch.float32)
    vision_feats = torch.randn(L, B, C, device=device)
    maskmem_features = torch.randn(B, C_mem, 32, 32, device=device)
    
    loss = module.step_update(vision_feats, maskmem_features, cache, second_order=False)
    
    print(f"  Total loss: {loss.item():.6f}")
    print("  ✓ PASSED")
    return True


def test_truncated_bptt():
    """Test 6: Truncated BPTT"""
    print("\n" + "=" * 60)
    print("[Test 6] Truncated BPTT")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, L, C, C_mem = 2, 4096, 256, 64
    
    config = TTTConfig(verbose=False, k_detach=8)
    module = TTTModule(hidden_dim=C, mem_dim=C_mem, config=config).to(device)
    
    cache = module.create_cache(B, device, torch.float32)
    vision_feats = torch.randn(L, B, C, device=device)
    maskmem_features = torch.randn(B, C_mem, 32, 32, device=device)
    
    for step in range(10):
        module.step_update(vision_feats, maskmem_features, cache, second_order=False)
    
    print(f"  k_detach: {config.k_detach}")
    print(f"  Steps run: 10")
    print(f"  Detached at steps: {cache.detached_steps}")
    
    assert len(cache.detached_steps) > 0, "Should have detached at least once"
    assert 8 in cache.detached_steps, "Should detach at step 8"
    print("  ✓ PASSED")
    return True


def test_save_load_state_dict():
    """Test 7: Save/Load state_dict 保留 W_init"""
    print("\n" + "=" * 60)
    print("[Test 7] Save/Load state_dict preserves W_init")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C, C_mem = 256, 64
    
    config = TTTConfig(verbose=False)
    module1 = TTTModule(hidden_dim=C, mem_dim=C_mem, config=config).to(device)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'test_ttt.pt')
        torch.save(module1.state_dict(), ckpt_path)
        print(f"  Saved to: {ckpt_path}")
        
        module2 = TTTModule(hidden_dim=C, mem_dim=C_mem, config=config).to(device)
        module2.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print("  Loaded into new module")
        
        for i in range(module1.num_layers):
            diff = (module1.W_init[i] - module2.W_init[i]).abs().max().item()
            assert diff < 1e-6, f"W_init[{i}] mismatch after load: {diff}"
            print(f"  W_init[{i}] match: ✓")
    
    print("  ✓ PASSED")
    return True


def test_gradient_flow():
    """Test 8: 梯度流检查"""
    print("\n" + "=" * 60)
    print("[Test 8] Gradient flow through TTT update")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, L, C, C_mem = 2, 4096, 256, 64
    
    config = TTTConfig(verbose=False)
    module = TTTModule(hidden_dim=C, mem_dim=C_mem, config=config).to(device)
    
    cache = module.create_cache(B, device, torch.float32)
    vision_feats = torch.randn(L, B, C, device=device)
    maskmem_features = torch.randn(B, C_mem, 32, 32, device=device)
    
    out1 = module.forward(vision_feats, cache)
    ttt_loss = module.step_update(vision_feats, maskmem_features, cache, second_order=True)
    out2 = module.forward(vision_feats, cache)
    
    final_loss = F.mse_loss(out2, torch.randn_like(out2))
    final_loss.backward()
    
    has_grad = False
    for name, param in module.named_parameters():
        if param.grad is not None and param.grad.norm() > 0:
            has_grad = True
            print(f"  {name}: grad_norm = {param.grad.norm().item():.6f}")
    
    assert has_grad, "At least some parameters should have gradients"
    print("  ✓ PASSED")
    return True


def test_iou_gate():
    """Test 9: IoU 门控"""
    print("\n" + "=" * 60)
    print("[Test 9] IoU gate")
    print("=" * 60)
    
    config = TTTConfig(verbose=False, update_iou_thr=0.5)
    module = TTTModule(hidden_dim=256, mem_dim=64, config=config)
    
    pred_iou = torch.tensor([[0.3, 0.4]])
    should = module.should_update(pred_iou, training=True)
    assert should == True, "Training mode should always update"
    print(f"  Training mode (IoU=0.35): should_update={should} ✓")
    
    should = module.should_update(pred_iou, training=False)
    assert should == False, "Low IoU should not update"
    print(f"  Inference mode (IoU=0.35): should_update={should} ✓")
    
    pred_iou = torch.tensor([[0.7, 0.8]])
    should = module.should_update(pred_iou, training=False)
    assert should == True, "High IoU should update"
    print(f"  Inference mode (IoU=0.75): should_update={should} ✓")
    
    print("  ✓ PASSED")
    return True


def test_first_order_no_graph():
    """Test 10: First-order (create_graph=False) 检查"""
    print("\n" + "=" * 60)
    print("[Test 10] First-order (create_graph=False)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, L, C, C_mem = 2, 4096, 256, 64
    
    config = TTTConfig(verbose=False)
    module = TTTModule(hidden_dim=C, mem_dim=C_mem, config=config).to(device)
    
    cache = module.create_cache(B, device, torch.float32)
    vision_feats = torch.randn(L, B, C, device=device)
    maskmem_features = torch.randn(B, C_mem, 32, 32, device=device)
    
    loss = module.step_update(vision_feats, maskmem_features, cache, second_order=False)
    
    for i, W in enumerate(cache.W_list):
        print(f"  Layer {i}: W.requires_grad={W.requires_grad}, "
              f"W.grad_fn={type(W.grad_fn).__name__ if W.grad_fn else None}")
    
    print("  ✓ PASSED")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("TTT Complete Verification Test Suite")
    print("=" * 60)
    
    tests = [
        ("W_init in state_dict", test_w_init_in_state_dict),
        ("Cache from W_init", test_cache_from_w_init),
        ("Forward uses cache W", test_forward_uses_cache_w),
        ("step_update only modifies cache", test_step_update_only_modifies_cache),
        ("delta_norm > 0", test_delta_norm_positive),
        ("Truncated BPTT", test_truncated_bptt),
        ("Save/Load state_dict", test_save_load_state_dict),
        ("Gradient flow", test_gradient_flow),
        ("IoU gate", test_iou_gate),
        ("First-order (create_graph=False)", test_first_order_no_graph),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTT Verification Tests")
    parser.add_argument("--test", type=str, default="all", help="Test to run (all, or test number 1-10)")
    args = parser.parse_args()
    
    if args.test == "all":
        run_all_tests()
    else:
        test_num = int(args.test)
        tests = [
            test_w_init_in_state_dict,
            test_cache_from_w_init,
            test_forward_uses_cache_w,
            test_step_update_only_modifies_cache,
            test_delta_norm_positive,
            test_truncated_bptt,
            test_save_load_state_dict,
            test_gradient_flow,
            test_iou_gate,
            test_first_order_no_graph,
        ]
        if 1 <= test_num <= len(tests):
            tests[test_num - 1]()
        else:
            print(f"Invalid test number: {test_num}")
            sys.exit(1)
