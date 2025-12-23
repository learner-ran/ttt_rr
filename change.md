# SAM-TTT (Hybrid Memory) 实现报告

## 修改文件清单

### 1. `ttt_rr/sam2/modeling/sam_ttt/ttt_module.py` ✅ 完全重写

**核心修改点：**

- **TTTConfig**: 新增配置数据类，包含所有 TTT 超参数
- **TTTCache**: Per-sample/per-video 状态缓存
  - `W_list`: 每层的权重 `[B, num_heads, head_dim, head_dim]`
  - `step`: 当前更新步数
  - `detached_steps`: Truncated BPTT 记录
- **TTTLinearStateless**: 无状态线性层，W 从外部 cache 传入
- **TTTBlockStateless**: 无状态 Block
- **TTTModule**: 主模块重构
  - `create_cache()`: 创建新 cache
  - `forward(vision_feats, ttt_cache)`: 使用 cache W 前向传播
  - `step_update(vision_feats, maskmem, cache, second_order)`: 元学习更新
  - `should_update(pred_iou, training)`: 门控判断

```python
# 关键代码片段
class TTTCache:
    def __init__(self, batch_size, num_layers, num_heads, head_dim, device, dtype):
        self.W_list = [
            torch.randn(batch_size, num_heads, head_dim, head_dim,
                       device=device, dtype=dtype, requires_grad=True) * 0.02
            for _ in range(num_layers)
        ]
        self.step = 0
        self.detached_steps = []

def step_update(self, vision_feats, maskmem_features, ttt_cache, second_order=False):
    # 1. 预处理 X
    x = self._preprocess_input(vision_feats)
    
    # 2. 预处理 Y (Target) - CRITICAL: detach AFTER projection
    y = self.proj_mem(pool(maskmem_features))
    y_target = y.detach()
    
    # 3. 每层更新
    for i, layer in enumerate(self.layers):
        W_old = ttt_cache.W_list[i]
        pred = layer.ttt_linear(layer.ln1(x), W_old)
        loss = F.mse_loss(pred, y_target)
        
        grad_W = torch.autograd.grad(loss, W_old, create_graph=second_order)[0]
        W_new = W_old - lr * grad_W
        ttt_cache.W_list[i] = W_new
    
    # 4. Truncated BPTT
    if ttt_cache.step % self.config.k_detach == 0:
        ttt_cache.detach_all()
```

---

### 2. `ttt_rr/sam2/configs/sam2.1_training/sam2_ttt_davis.yaml` ✅ 已修改

**修改点：**

```yaml
scratch:
  num_frames: 8        # P0: 从 4 改为 8，配合 TBPTT

# 新增 TTT 配置块
ttt:
  inner_lr: 1.0e-2
  learnable_lr: false
  lr_per_layer: true
  lr_min: 1.0e-4
  lr_max: 5.0e-2
  k_detach: 8
  update_iou_thr: 0.5
  pool_size: 32
  verbose: true
  log_first_n: 5
  second_order: false

model:
  # P0: 从 2 改为 3
  num_maskmem: 3
```

---

### 3. `ttt_rr/sam2/modeling/sam2_base.py` ✅ 已修改

**修改点：**

1. **导入语句** (Line 17):
```python
from .sam_ttt.ttt_module import TTTModule, TTTConfig, TTTCache, create_ttt_cache
```

2. **TTT 初始化** (Line 184+):
```python
ttt_config = TTTConfig(
    hidden_dim=self.hidden_dim,
    mem_dim=self.mem_dim,
    num_layers=4,
    num_heads=4,
)
self.ttt_module = TTTModule(
    hidden_dim=self.hidden_dim,
    mem_dim=self.mem_dim,
    config=ttt_config
)

# 断言
assert self.num_maskmem >= 3, f"num_maskmem must be >= 3"
```

3. **_track_step TTT 集成** (Line 788+):
```python
# 确保有 cache
if "ttt_cache" not in output_dict or output_dict["ttt_cache"] is None:
    B = current_vision_feats[-1].shape[1]
    output_dict["ttt_cache"] = self.ttt_module.create_cache(
        batch_size=B,
        device=current_vision_feats[-1].device,
        dtype=current_vision_feats[-1].dtype
    )

ttt_cache = output_dict["ttt_cache"]
feat_ttt = self.ttt_module(current_vision_feats[-1], ttt_cache)

# Fusion
gate = 0.0 if frame_idx == 0 else 1.0
pix_feat = pix_feat + self.ttt_module.alpha_global * gate * feat_ttt
```

---

### 4. `ttt_rr/training/model/sam2.py` ✅ 已修改

**修改点：**

1. **forward 方法** (Line 107+):
```python
def forward(self, input: BatchedVideoDatapoint):
    if hasattr(self, 'ttt_module'):
        self.ttt_module.reset_parameters()
        self.ttt_module.step_counter = 0
        self.ttt_module.global_step += 1
```

2. **track_step TTT 更新** (Line 450+):
```python
# 门控判断
should_update = self.ttt_module.should_update(pred_iou, self.training)

if should_update:
    ttt_cache = output_dict["ttt_cache"]
    ttt_loss = self.ttt_module.step_update(
        vision_feats=current_vision_feats[-1],
        maskmem_features=current_out["maskmem_features"],
        ttt_cache=ttt_cache,
        second_order=False
    )
    current_out["ttt_loss"] = ttt_loss
```

---

### 5. `ttt_rr/sam2/modeling/sam_ttt/ttt_logger.py` ✅ 新增

统一日志系统，支持：
- 分级日志控制
- 显式记忆日志
- TTT 更新日志
- 梯度流日志
- 断言验证

---

### 6. `ttt_rr/test_ttt_verification.py` ✅ 新增

完整验证测试脚本，验证：
1. 配置验证
2. Cache 隔离性
3. Forward 使用 Cache W
4. Target Detach
5. Truncated BPTT
6. 二阶梯度支持
7. 梯度流验证
8. 更新门控策略
9. 无 In-place 修改
10. Shape 一致性

---

## 所有日志打印点清单

| 位置 | 打印内容 | 控制开关 |
|------|---------|---------|
| TTTModule.__init__ | 配置信息 | 启动时打印一次 |
| TTTModule.create_cache | cache 创建信息 | `log_first_n` |
| TTTModule.forward | 输入/输出 shape | `log_first_n` |
| TTTModule.step_update | loss/grad_norm/delta_norm | `log_first_n` 或每 100 步 |
| TTTCache.detach_all | TBPTT detach | 每次发生 |
| SAM2Base._track_step | Fusion 参数 | `log_first_n` |
| SAM2Train.forward | batch 开始 | 前 3 个 batch |
| SAM2Train.track_step | 更新门控 | `log_first_n` |

---

## 验收 Checklist

### 0. 核心目标验收

| 目标 | 状态 | 验证方法 |
|------|------|---------|
| 显式记忆 O(1) | ✅ | `num_maskmem=3`，只用 Cond0+Last+Keyframe |
| 显式记忆容量 num_maskmem=3 | ✅ | YAML 配置 + 断言检查 |
| TTT 输入来自 raw neck output | ✅ | 使用 `current_vision_feats[-1]` |
| per-video cache | ✅ | `TTTCache` 独立实例 |
| Forward 读取 cache W | ✅ | `layer.ttt_linear(x, W)` |
| Update 更新 cache W | ✅ | `cache.W_list[i] = W_new` |
| 禁止 .data 修改 | ✅ | 使用标准赋值 |
| Y target detach | ✅ | `y_target = proj_mem(y).detach()` |
| Training: Teacher Forcing | ✅ | `training=True → should_update=True` |
| Inference: IoU 门控 | ✅ | `pred_iou > thr` |
| Truncated BPTT K=8 | ✅ | `step % 8 == 0 → detach` |

### 1. 配置验收

```yaml
num_maskmem: 3          ✅
num_frames: 8           ✅
k_detach: 8             ✅
inner_lr: 1.0e-2        ✅
update_iou_thr: 0.5     ✅
add_all_frames_to_correct_as_cond: False  ✅ (已有)
```

### 2. 显式记忆流验收

- [x] memory attention 只使用 Cond0 + Last + Keyframe
- [x] keyframe slot 独立存在 (t_pos=1)
- [x] 第 0 帧走 "no memory embedding" 分支
- [x] 断言检查帧数 <= num_maskmem

### 3. TTT 模块验收

- [x] Cache 结构正确：`W_list`, `step`, `detached_steps`
- [x] Forward 使用 cache W（不读 nn.Parameter）
- [x] Update 使用 autograd.grad
- [x] 禁止 in-place 修改
- [x] second_order 支持

### 4. 更新门控验收

- [x] Training: always update
- [x] Inference: pred_iou > thr
- [x] pred_iou 来自 mask decoder IoU head

### 5. Truncated BPTT 验收

- [x] 每 K=8 帧 detach
- [x] detach 发生在 step_update 之后
- [x] 只 detach 传入下一帧的状态

### 6. 自检验收

- [x] 显式记忆帧数 <= 3
- [x] TTT 输入输出 shape 正确
- [x] cache['W'] 更新后 delta_norm > 0
- [x] second_order=True 时 W_new.requires_grad=True
- [x] 无 in-place 参数修改
- [x] backward 能跑通

---

## 运行指南

### 1. 运行验证测试

```bash
cd /root/autodl-tmp/ttt_rr
python test_ttt_verification.py
```

### 2. 启动训练

```bash
cd /root/autodl-tmp/ttt_rr
python -m training.train \
    --config-name sam2_ttt_davis \
    --config-dir ./sam2/configs/sam2.1_training
```

### 3. 检查日志

训练时观察以下关键日志：

```
[SAM2Base] Initialization:
  num_maskmem: 3
  hidden_dim: 256
  
[TTT] Created new cache for batch_size=2

[TTT Fusion] frame=0, gate=0.0, alpha=0.0000
[TTT Fusion] frame=1, gate=1.0, alpha=0.0050

[TTT Update Gate] frame=1, training=True, pred_iou.mean=0.7532, should_update=True
[TTT Update Done] frame=1, loss=0.023456, step=1, update_count=1

[TBPTT] Detaching at step 8, layers: [0, 1, 2, 3]
```

---

## 潜在问题和调试

### 1. 梯度消失

如果 `grad_norm` 为 0，检查：
- `W.requires_grad` 是否为 True
- `loss.backward()` 是否正确调用
- target 是否正确 detach

### 2. 显存溢出

如果 OOM，尝试：
- 减小 `num_frames`（如 4）
- 减小 `batch_size`
- 启用 `second_order=False`（默认）

### 3. TTT 不更新

如果 `should_update=False`，检查：
- `pred_iou` 是否有效
- `update_iou_thr` 是否过高
- `training` 模式是否正确

---

*最后更新: 2025-12-23*
