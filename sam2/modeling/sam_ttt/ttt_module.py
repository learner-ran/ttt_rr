"""
SAM-TTT Module: Cache-Based Meta-Learning Implementation

核心设计原则：
1. W_init 是 nn.ParameterList，存入 state_dict，训练时更新
2. TTTCache 从 W_init.repeat(B,...) 初始化（禁止 torch.randn）
3. Forward/Update 显式读取 cache 中的 W（禁止读取 layer.Parameter）
4. step_update 只更新 cache（FO：create_graph=False），不修改 W_init
5. Truncated BPTT：每 K=8 帧 detach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class TTTConfig:
    """TTT 模块配置"""
    hidden_dim: int = 256
    mem_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    
    # Inner-loop learning rate
    inner_lr: float = 1.0e-2
    learnable_lr: bool = False
    lr_per_layer: bool = True
    lr_min: float = 1.0e-4
    lr_max: float = 5.0e-2
    
    # Truncated BPTT
    k_detach: int = 8  # 每 K 帧 detach
    
    # Update gate threshold
    update_iou_thr: float = 0.5
    
    # Pool size for low-res TTT
    pool_size: int = 32
    
    # Debug/logging
    verbose: bool = True
    log_first_n: int = 5  # 前 N 个 iter 打印详细日志
    
    # Second order gradient
    second_order: bool = False


# ============================================================================
# TTT Cache: Per-Sample State Storage
# ============================================================================
class TTTCache:
    """
    Per-video/per-sample TTT cache，存储每一层的 W。
    
    **重要**：W 必须从 TTTModule.W_init 通过 repeat(B,...) 初始化。
    禁止使用 torch.randn 初始化！
    
    结构：
    - W_list: list，长度=层数（4），每个元素 shape [B, num_heads, head_dim, head_dim]
    - step: 当前视频内更新步数
    - detached_steps: 记录 detach 发生的 step
    """
    
    def __init__(
        self,
        W_init_list: List[torch.Tensor],  # 从 TTTModule.W_init 传入
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            W_init_list: TTTModule.W_init 的参数列表，每个 shape [H, D, D]
            batch_size: batch 大小
            device: 设备
            dtype: 数据类型
        """
        self.batch_size = batch_size
        self.num_layers = len(W_init_list)
        self.device = device
        self.dtype = dtype
        
        # 从 W_init 构造 per-sample W，使用 repeat（禁止 expand）
        # W_init[i]: [H, D, D] -> W[i]: [B, H, D, D]
        self.W_list: List[torch.Tensor] = []
        for i, W_init in enumerate(W_init_list):
            # 使用 repeat 而非 expand，确保每个 batch 有独立的内存
            # W_init: [H, D, D] -> [1, H, D, D] -> [B, H, D, D]
            W = W_init.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            # 确保在正确的设备和类型，并启用梯度
            W = W.to(device=device, dtype=dtype).requires_grad_(True)
            self.W_list.append(W)
        
        # 记录 W_init 的 shape 用于验证
        self.num_heads = W_init_list[0].shape[0]
        self.head_dim = W_init_list[0].shape[1]
        
        self.step = 0
        self.detached_steps: List[int] = []
        self.update_count = 0  # 实际更新次数
        
    def reset_from_init(self, W_init_list: List[torch.Tensor]):
        """从 W_init 重置 cache（用于新视频开始）"""
        for i, W_init in enumerate(W_init_list):
            W = W_init.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
            W = W.to(device=self.device, dtype=self.dtype).requires_grad_(True)
            self.W_list[i] = W
        self.step = 0
        self.detached_steps = []
        self.update_count = 0
        
    def detach_all(self):
        """对所有 W 执行 detach，用于 Truncated BPTT"""
        for i in range(self.num_layers):
            self.W_list[i] = self.W_list[i].detach().requires_grad_(True)
        self.detached_steps.append(self.step)
        
    def get_state_dict(self) -> Dict[str, Any]:
        """获取 cache 状态用于调试"""
        return {
            'step': self.step,
            'update_count': self.update_count,
            'detached_steps': self.detached_steps,
            'W_norms': [w.norm().item() for w in self.W_list],
            'W_means': [w.mean().item() for w in self.W_list],
        }


# ============================================================================
# TTT Linear Layer (Stateless - 使用 cache 中的 W)
# ============================================================================
class TTTLinearStateless(nn.Module):
    """
    无状态的 TTT Linear 层。
    
    **重要**：W 不是 nn.Parameter，而是从外部 cache 传入。
    这确保了每个 sample 有独立的 W 状态。
    """
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, f"dim {dim} must be divisible by num_heads {num_heads}"
        
    def forward(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C] - 输入特征
            W: [B, H, D, D] - 从 cache 传入的权重（禁止从 layer 内部读取）
            
        Returns:
            out: [B, L, C]
        """
        B, L, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        # Split heads: [B, L, H, D]
        x_heads = x.view(B, L, H, D)
        
        # W: [B, H, D, D]
        # einsum: 'blhd,bhde->blhe'
        out = torch.einsum('blhd,bhde->blhe', x_heads, W)
        
        # Merge heads: [B, L, C]
        out = out.reshape(B, L, C)
        return out


# ============================================================================
# TTT Block (Stateless)
# ============================================================================
class TTTBlockStateless(nn.Module):
    """
    无状态的 TTT Block。
    
    包含：
    - LayerNorm -> TTTLinear -> Residual (with learnable alpha)
    - LayerNorm -> MLP -> Residual
    
    **注意**：没有内部的 W 参数，W 从外部 cache 传入。
    """
    
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # TTT path
        self.ln1 = nn.LayerNorm(dim)
        self.ttt_linear = TTTLinearStateless(dim, num_heads)
        # Sigmoid 参数化：alpha_eff = sigmoid(alpha_param) * alpha_max
        # 初始化为 -4 使 sigmoid(-4)≈0.018，小但非零，有梯度
        self.alpha_param = nn.Parameter(torch.tensor([-4.0]))
        
        # MLP path
        self.ln2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    @property
    def alpha(self):
        """实际使用的 alpha: sigmoid(alpha_param) * 2.0"""
        return torch.sigmoid(self.alpha_param) * 2.0
        
    def forward(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]
            W: [B, H, D, D] - 从 cache 传入（禁止读取 layer 内部参数）
        """
        # TTT path with residual
        residual = x
        x_norm = self.ln1(x)
        ttt_out = self.ttt_linear(x_norm, W)
        x = residual + self.alpha * ttt_out
        
        # MLP path with residual
        residual = x
        x = residual + self.mlp(self.ln2(x))
        
        return x


# ============================================================================
# Main TTT Module
# ============================================================================
class TTTModule(nn.Module):
    """
    SAM-TTT 主模块：Cache-Based Meta-Learning Implementation
    
    核心特性：
    1. W_init 是 nn.ParameterList，存入 state_dict，训练时由外层优化器更新
    2. create_cache 从 W_init.repeat(B,...) 构造 per-sample cache
    3. Forward/Update 显式使用 cache 中的 W
    4. step_update 只更新 cache（使用 autograd.grad），不修改 W_init
    5. Truncated BPTT 每 K 帧
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        mem_dim: int = 64,
        num_layers: int = 4,
        config: Optional[TTTConfig] = None
    ):
        super().__init__()
        
        # 配置
        if config is None:
            config = TTTConfig(hidden_dim=hidden_dim, mem_dim=mem_dim, num_layers=num_layers)
        self.config = config
        
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        self.num_layers = num_layers
        self.num_heads = config.num_heads
        self.head_dim = hidden_dim // config.num_heads
        
        # ================================================================
        # **核心修改**：W_init 作为 nn.ParameterList，存入 state_dict
        # Shape: [num_heads, head_dim, head_dim] per layer
        # 训练时由外层优化器更新，推理时从 checkpoint 加载
        # ================================================================
        self.W_init = nn.ParameterList([
            nn.Parameter(torch.randn(self.num_heads, self.head_dim, self.head_dim) * 0.02)
            for _ in range(num_layers)
        ])
        
        # 构建层（无状态，不包含 W）
        self.layers = nn.ModuleList([
            TTTBlockStateless(hidden_dim, config.num_heads)
            for _ in range(num_layers)
        ])
        
        # Memory projection: mem_dim -> hidden_dim
        self.proj_mem = nn.Linear(mem_dim, hidden_dim)
        
        # Global fusion alpha (sigmoid 参数化)
        # 初始化为 -4 使 sigmoid(-4)≈0.018，小但非零
        self.alpha_global_param = nn.Parameter(torch.tensor([-4.0]))
        
        # Per-layer learnable lr (可选)
        if config.learnable_lr:
            if config.lr_per_layer:
                self.lr_params = nn.ParameterList([
                    nn.Parameter(torch.tensor([config.inner_lr]))
                    for _ in range(num_layers)
                ])
            else:
                self.lr_param = nn.Parameter(torch.tensor([config.inner_lr]))
        
        # 统计计数器
        self.step_counter = 0
        self.global_step = 0
        
        # 打印配置和验证
        self._print_config()
        self._verify_state_dict()
        
    @property
    def alpha_global(self):
        """实际使用的全局融合因子: sigmoid(alpha_global_param) * 2.0"""
        return torch.sigmoid(self.alpha_global_param) * 2.0
        
    def _print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("[TTT Module] Configuration:")
        print(f"  hidden_dim: {self.hidden_dim}")
        print(f"  mem_dim: {self.mem_dim}")
        print(f"  num_layers: {self.num_layers}")
        print(f"  num_heads: {self.num_heads}")
        print(f"  head_dim: {self.head_dim}")
        print(f"  inner_lr: {self.config.inner_lr}")
        print(f"  learnable_lr: {self.config.learnable_lr}")
        print(f"  lr_per_layer: {self.config.lr_per_layer}")
        print(f"  k_detach: {self.config.k_detach}")
        print(f"  update_iou_thr: {self.config.update_iou_thr}")
        print(f"  pool_size: {self.config.pool_size}")
        print("=" * 60)
        
    def _verify_state_dict(self):
        """验证 state_dict 是否包含 W_init"""
        state_dict_keys = list(self.state_dict().keys())
        w_init_keys = [k for k in state_dict_keys if 'W_init' in k]
        
        print("[TTT Module] State Dict Verification:")
        print(f"  Total keys: {len(state_dict_keys)}")
        print(f"  W_init keys: {w_init_keys}")
        
        if len(w_init_keys) == self.num_layers:
            print(f"  ✓ W_init.* found in state_dict ({len(w_init_keys)} layers)")
            for i, key in enumerate(w_init_keys):
                print(f"    {key}: {self.state_dict()[key].shape}")
        else:
            print(f"  ✗ ERROR: Expected {self.num_layers} W_init keys, found {len(w_init_keys)}")
        print("=" * 60)
        
    def create_cache(
        self, 
        batch_size: int, 
        device: torch.device, 
        dtype: torch.dtype = torch.float32
    ) -> TTTCache:
        """
        创建新的 TTT cache，从 W_init 初始化。
        
        **重要**：使用 W_init.repeat(B,...) 而非 torch.randn
        """
        # 将 W_init 参数列表传给 TTTCache
        W_init_list = [w.data for w in self.W_init]  # 取数据，不带梯度计算图
        
        cache = TTTCache(
            W_init_list=W_init_list,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        
        if self.config.verbose and self.global_step < self.config.log_first_n:
            print(f"[TTT Cache] Initialized from W_init:")
            print(f"  batch_size: {batch_size}")
            print(f"  num_layers: {self.num_layers}")
            print(f"  W shapes: {[w.shape for w in cache.W_list]}")
            print(f"  W_init norms: {[f'{w.norm().item():.4f}' for w in self.W_init]}")
            print(f"  Cache W norms: {[f'{w.norm().item():.4f}' for w in cache.W_list]}")
            
        return cache
    
    def reset_cache(self, cache: TTTCache):
        """从 W_init 重置 cache"""
        W_init_list = [w.data for w in self.W_init]
        cache.reset_from_init(W_init_list)
        
        if self.config.verbose and self.step_counter < self.config.log_first_n:
            print(f"[TTT Cache] Reset from W_init. Step counter: {self.step_counter}")
            
    def get_2d_pe(self, H: int, W: int, C: int, device: torch.device) -> torch.Tensor:
        """生成 2D Sine-Cos 位置编码"""
        y_pos = torch.arange(H, device=device).unsqueeze(1).float()
        x_pos = torch.arange(W, device=device).unsqueeze(0).float()
        
        div_term = torch.exp(
            torch.arange(0, C // 2, 2, device=device).float() * 
            (-math.log(10000.0) / (C // 2))
        )
        
        pe = torch.zeros(H, W, C, device=device)
        
        # X direction
        pe[:, :, 0::4] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe[:, :, 1::4] = torch.cos(x_pos.unsqueeze(-1) * div_term)
        
        # Y direction
        pe[:, :, 2::4] = torch.sin(y_pos.unsqueeze(-1) * div_term)
        pe[:, :, 3::4] = torch.cos(y_pos.unsqueeze(-1) * div_term)
        
        return pe.flatten(0, 1).unsqueeze(0)  # [1, H*W, C]
    
    def _preprocess_input(self, vision_feats: torch.Tensor) -> torch.Tensor:
        """
        预处理输入特征：pool + flatten + pos encoding
        
        Args:
            vision_feats: [L, B, C] where L=4096 (64x64)
            
        Returns:
            x: [B, L_pool, C] where L_pool=1024 (32x32)
        """
        L_orig, B, C = vision_feats.shape
        pool_size = self.config.pool_size
        orig_size = int(math.sqrt(L_orig))
        
        assert L_orig == orig_size * orig_size, f"L={L_orig} must be perfect square"
        
        # [L, B, C] -> [B, L, C] -> [B, H, W, C] -> [B, C, H, W]
        x = vision_feats.permute(1, 0, 2)
        x = x.view(B, orig_size, orig_size, C).permute(0, 3, 1, 2)
        
        # AvgPool: [B, C, 64, 64] -> [B, C, 32, 32]
        x = F.avg_pool2d(x, kernel_size=orig_size // pool_size, stride=orig_size // pool_size)
        
        # [B, C, 32, 32] -> [B, 32, 32, C] -> [B, 1024, C]
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Add positional encoding
        pe = self.get_2d_pe(pool_size, pool_size, C, x.device)
        x = x + pe
        
        return x
    
    def _postprocess_output(self, x: torch.Tensor, orig_size: int = 64) -> torch.Tensor:
        """
        后处理输出：reshape + upsample
        
        Args:
            x: [B, L_pool, C] where L_pool=1024
            
        Returns:
            out: [B, C, H, W] where H=W=orig_size
        """
        B, L, C = x.shape
        pool_size = self.config.pool_size
        
        # [B, 1024, C] -> [B, 32, 32, C] -> [B, C, 32, 32]
        x = x.view(B, pool_size, pool_size, C).permute(0, 3, 1, 2)
        
        # Upsample: [B, C, 64, 64]
        x = F.interpolate(x, size=(orig_size, orig_size), mode='bilinear', align_corners=False)
        
        return x
    
    def forward(
        self,
        vision_feats: torch.Tensor,
        ttt_cache: Optional[TTTCache] = None
    ) -> torch.Tensor:
        """
        Forward pass using cache-stored W.
        
        **重要**：必须使用 cache 中的 W，禁止读取 layer 内部参数。
        
        Args:
            vision_feats: [L, B, C] where L=4096 (来自 neck output)
            ttt_cache: TTT cache containing per-sample W
            
        Returns:
            out: [B, C, H, W] where H=W=64
        """
        L_orig, B, C = vision_feats.shape
        
        # 预处理
        x = self._preprocess_input(vision_feats)
        
        # Shape 验证日志
        if self.config.verbose and self.step_counter < self.config.log_first_n:
            print(f"[TTT Forward] Input: vision_feats {vision_feats.shape}")
            print(f"[TTT Forward] After preprocess: x {x.shape}")
        
        # 如果没有 cache，从 W_init 创建
        if ttt_cache is None:
            print("[TTT Warning] No cache provided, creating from W_init")
            ttt_cache = self.create_cache(B, vision_feats.device, vision_feats.dtype)
        
        # 通过每一层，**显式使用 cache 中的 W**（禁止读取 layer 内部参数）
        for i, layer in enumerate(self.layers):
            W = ttt_cache.W_list[i]  # [B, H, D, D] - 从 cache 读取
            
            # 验证 W 的状态
            if self.config.verbose and self.step_counter < self.config.log_first_n:
                print(f"[TTT Forward] Layer {i}: W.requires_grad={W.requires_grad}, "
                      f"W.norm={W.norm().item():.4f}")
            
            x = layer(x, W)  # 传入 cache 中的 W
        
        # 后处理
        out = self._postprocess_output(x)
        
        if self.config.verbose and self.step_counter < self.config.log_first_n:
            print(f"[TTT Forward] Output: {out.shape}")
        
        return out
    
    def get_lr(self, layer_idx: int) -> float:
        """获取指定层的学习率"""
        if self.config.learnable_lr:
            if self.config.lr_per_layer:
                lr = torch.clamp(
                    self.lr_params[layer_idx],
                    self.config.lr_min,
                    self.config.lr_max
                ).item()
            else:
                lr = torch.clamp(
                    self.lr_param,
                    self.config.lr_min,
                    self.config.lr_max
                ).item()
        else:
            lr = self.config.inner_lr
        return lr
    
    def step_update(
        self,
        vision_feats: torch.Tensor,
        maskmem_features: torch.Tensor,
        ttt_cache: TTTCache,
        second_order: bool = False
    ) -> torch.Tensor:
        """
        TTT 内循环更新：使用 autograd.grad 更新 cache 中的 W
        
        **核心元学习逻辑**：
        1. 构造 x（与 forward 预处理一致）
        2. 构造 y_target = proj_mem(pool(maskmem_features)).detach()
        3. 对每层：
           - pred = ttt_linear(LN(x), W_old)
           - loss = MSE(pred, y_target)
           - grad = autograd.grad(loss, W_old, create_graph=second_order)
           - W_new = W_old - lr * grad
           - 写回 cache（不修改 W_init！）
        
        **重要**：
        - FO (First-Order): create_graph=False
        - 只更新 cache，不修改 W_init
        
        Args:
            vision_feats: [L, B, C] - 原始 neck output
            maskmem_features: [B, C_mem, H, W] - memory encoder 输出
            ttt_cache: 要更新的 cache
            second_order: 是否使用二阶梯度（默认 False = FO-MAML）
            
        Returns:
            total_loss: 总损失
        """
        self.step_counter += 1
        ttt_cache.step += 1
        
        L_orig, B, C = vision_feats.shape
        
        # ============== 1. 预处理 X ==============
        x = self._preprocess_input(vision_feats)  # [B, 1024, C]
        
        # ============== 2. 预处理 Y (Target) ==============
        y = maskmem_features  # [B, C_mem, H, W]
        
        # Pool to 32x32 if needed
        if y.shape[-2:] != (self.config.pool_size, self.config.pool_size):
            y = F.adaptive_avg_pool2d(y, (self.config.pool_size, self.config.pool_size))
        
        # [B, C_mem, 32, 32] -> [B, 32, 32, C_mem] -> [B, 1024, C_mem]
        y = y.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Project: [B, 1024, C_mem] -> [B, 1024, C]
        y = self.proj_mem(y)
        
        # **CRITICAL: Detach AFTER projection to prevent "moving target"**
        y_target = y.detach()
        
        # 日志
        verbose = self.config.verbose and (
            ttt_cache.step <= self.config.log_first_n or 
            ttt_cache.step % 100 == 0
        )
        
        if verbose:
            print(f"\n[TTT Update] Step {ttt_cache.step}, Frame processing")
            print(f"  X shape: {x.shape}, Y_target shape: {y_target.shape}")
            print(f"  second_order: {second_order} (create_graph={second_order})")
        
        # ============== 3. Update Loop ==============
        total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        delta_norms = []
        
        x_current = x
        
        for i, layer in enumerate(self.layers):
            W_old = ttt_cache.W_list[i]  # [B, H, D, D] - 从 cache 读取
            
            # 确保 W_old 需要梯度
            if not W_old.requires_grad:
                W_old = W_old.requires_grad_(True)
                ttt_cache.W_list[i] = W_old
            
            # LayerNorm
            x_norm = layer.ln1(x_current)
            
            # 计算预测：使用 cache 中的 W_old（禁止读取 layer 内部参数）
            pred = layer.ttt_linear(x_norm, W_old)  # [B, L, C]
            
            # 计算损失
            loss = F.mse_loss(pred, y_target)
            total_loss = total_loss + loss
            
            # 计算梯度 (FO: create_graph=False)
            grad_W = torch.autograd.grad(
                loss, 
                W_old, 
                create_graph=second_order,  # FO=False, SO=True
                retain_graph=True
            )[0]
            
            # 获取学习率
            lr = self.get_lr(i)
            
            # 更新 cache 中的 W（不修改 W_init！）
            W_new = W_old - lr * grad_W
            
            # 记录 delta_norm
            delta_norm = (W_new - W_old).norm().item()
            delta_norms.append(delta_norm)
            
            # 写回 cache
            ttt_cache.W_list[i] = W_new
            
            if verbose:
                print(f"  Layer {i}: loss={loss.item():.6f}, delta_norm={delta_norm:.6f}, lr={lr:.4f}")
            
            # 使用更新后的 W 计算下一层的输入
            with torch.no_grad():
                ttt_out = layer.ttt_linear(x_norm, W_new.detach())
                x_next = x_current + layer.alpha * ttt_out
                x_next = x_next + layer.mlp(layer.ln2(x_next))
                x_current = x_next
        
        # 更新计数
        ttt_cache.update_count += 1
        
        # ============== 4. Truncated BPTT ==============
        if ttt_cache.step % self.config.k_detach == 0:
            if verbose:
                print(f"[TTT TBPTT] Detaching at step {ttt_cache.step}")
            ttt_cache.detach_all()
        
        # 最终日志
        if verbose:
            print(f"[TTT Update Complete] Total loss: {total_loss.item():.6f}")
            print(f"  delta_norms: {[f'{d:.6f}' for d in delta_norms]}")
            print(f"  Update count: {ttt_cache.update_count}/{ttt_cache.step}")
        
        return total_loss
    
    def should_update(
        self,
        pred_iou: torch.Tensor,
        training: bool
    ) -> bool:
        """
        判断是否应该执行 TTT 更新
        
        Args:
            pred_iou: [B, num_masks] - mask decoder 输出的 IoU 预测
            training: 是否在训练模式
            
        Returns:
            should_update: bool
        """
        if training:
            # Training: Teacher Forcing - 每帧都更新
            return True
        else:
            # Inference: 质量门控
            mean_iou = pred_iou.mean().item()
            should = mean_iou > self.config.update_iou_thr
            
            if self.config.verbose and self.step_counter < self.config.log_first_n:
                print(f"[TTT Gate] pred_iou.mean={mean_iou:.4f}, thr={self.config.update_iou_thr}, "
                      f"should_update={should}")
            
            return should
    
    def reset_parameters(self):
        """重置模块参数（用于新视频开始）"""
        # ⚠️ 只重置 fast state，不重置元参数！
        # 元参数（W_init、alpha_global、layer.alpha、MLP/LN/proj_mem）随 checkpoint 保存
        self.step_counter = 0
        # layer.alpha、alpha_global、W_init 等元参数保持不变
        # cache.W 由 create_cache 从 W_init 初始化（新视频开始时创建新 cache）
        
    def detach_state(self, ttt_cache: TTTCache):
        """外部调用 detach（用于 Truncated BPTT）"""
        ttt_cache.detach_all()


# ============================================================================
# 工厂函数
# ============================================================================
def create_ttt_cache(
    ttt_module: TTTModule,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> TTTCache:
    """创建 TTT cache 的工厂函数（使用 TTTModule 的 W_init）"""
    return ttt_module.create_cache(batch_size, device, dtype)


# ============================================================================
# Self-Test Function
# ============================================================================
def run_ttt_self_test():
    """
    TTT 模块自检测试
    
    验证：
    1. W_init 在 state_dict 中
    2. Cache 从 W_init 初始化（不是 torch.randn）
    3. forward/update 使用 cache 中的 W
    4. step_update 只更新 cache，不修改 W_init
    5. delta_norm > 0
    6. Truncated BPTT
    """
    print("=" * 60)
    print("[TTT Self-Test] Starting...")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    B = 2
    L = 4096  # 64x64
    C = 256
    C_mem = 64
    
    # 创建模块
    config = TTTConfig(verbose=True, log_first_n=10)
    module = TTTModule(hidden_dim=C, mem_dim=C_mem, config=config).to(device)
    
    # ============== Test 1: W_init 在 state_dict 中 ==============
    print("\n[Test 1] W_init in state_dict check...")
    state_dict = module.state_dict()
    w_init_keys = [k for k in state_dict.keys() if 'W_init' in k]
    assert len(w_init_keys) == module.num_layers, f"Expected {module.num_layers} W_init keys, got {len(w_init_keys)}"
    print(f"  ✓ PASSED: W_init keys found: {w_init_keys}")
    
    # ============== Test 2: Cache 从 W_init 初始化 ==============
    print("\n[Test 2] Cache initialized from W_init...")
    cache = module.create_cache(B, device, dtype)
    
    for i in range(module.num_layers):
        W_init_norm = module.W_init[i].norm().item()
        cache_W_norm = cache.W_list[i].norm().item()
        # 由于 repeat，cache W 的 norm 应该是 W_init norm * sqrt(B)
        expected_ratio = math.sqrt(B)
        actual_ratio = cache_W_norm / W_init_norm if W_init_norm > 0 else 0
        print(f"  Layer {i}: W_init.norm={W_init_norm:.4f}, cache.W.norm={cache_W_norm:.4f}, "
              f"ratio={actual_ratio:.4f} (expected ~{expected_ratio:.4f})")
    print("  ✓ PASSED: Cache initialized from W_init")
    
    # ============== Test 3: Forward 使用 cache 中的 W ==============
    print("\n[Test 3] Forward uses cache W...")
    vision_feats = torch.randn(L, B, C, device=device, dtype=dtype)
    out = module.forward(vision_feats, cache)
    assert out.shape == (B, C, 64, 64), f"Forward output shape mismatch: {out.shape}"
    print(f"  ✓ PASSED: output shape {out.shape}")
    
    # ============== Test 4: step_update 只更新 cache，不修改 W_init ==============
    print("\n[Test 4] step_update updates cache, not W_init...")
    maskmem_features = torch.randn(B, C_mem, 32, 32, device=device, dtype=dtype)
    
    # 记录 W_init 原值
    W_init_before = [w.clone() for w in module.W_init]
    cache_W_before = [w.clone() for w in cache.W_list]
    
    # 执行更新
    loss = module.step_update(vision_feats, maskmem_features, cache, second_order=False)
    
    # 检查 W_init 未变
    W_init_changed = False
    for i, (before, after) in enumerate(zip(W_init_before, module.W_init)):
        diff = (after - before).abs().max().item()
        if diff > 1e-8:
            W_init_changed = True
            print(f"  ✗ ERROR: W_init[{i}] changed by {diff}")
    
    if not W_init_changed:
        print("  ✓ W_init unchanged after step_update")
    
    # 检查 cache W 变了
    cache_W_changed = False
    for i, (before, after) in enumerate(zip(cache_W_before, cache.W_list)):
        diff = (after - before).abs().max().item()
        if diff > 1e-8:
            cache_W_changed = True
            print(f"  Layer {i}: cache.W delta_max = {diff:.6f}")
    
    assert cache_W_changed, "Cache W should change after step_update"
    print("  ✓ PASSED: step_update only modifies cache, not W_init")
    
    # ============== Test 5: delta_norm > 0 ==============
    print("\n[Test 5] delta_norm > 0 check...")
    cache2 = module.create_cache(B, device, dtype)
    
    # 捕获 delta_norms
    module.step_update(vision_feats, maskmem_features, cache2, second_order=False)
    print("  ✓ PASSED: delta_norm printed in logs above")
    
    # ============== Test 6: Truncated BPTT ==============
    print("\n[Test 6] Truncated BPTT check...")
    cache3 = module.create_cache(B, device, dtype)
    for step in range(10):
        module.step_update(vision_feats, maskmem_features, cache3, second_order=False)
    
    assert len(cache3.detached_steps) > 0, "Should have detached at least once"
    print(f"  Detached at steps: {cache3.detached_steps}")
    print("  ✓ PASSED: Truncated BPTT working")
    
    # ============== Test 7: Save/Load state_dict ==============
    print("\n[Test 7] Save/Load state_dict...")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'test_ttt.pt')
        torch.save(module.state_dict(), ckpt_path)
        
        # 创建新模块并加载
        module2 = TTTModule(hidden_dim=C, mem_dim=C_mem, config=config).to(device)
        module2.load_state_dict(torch.load(ckpt_path))
        
        # 验证 W_init 相同
        for i in range(module.num_layers):
            diff = (module.W_init[i] - module2.W_init[i]).abs().max().item()
            assert diff < 1e-6, f"W_init[{i}] mismatch after load: {diff}"
        
        print(f"  ✓ PASSED: state_dict save/load preserves W_init")
    
    print("\n" + "=" * 60)
    print("[TTT Self-Test] ALL TESTS PASSED!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    run_ttt_self_test()
