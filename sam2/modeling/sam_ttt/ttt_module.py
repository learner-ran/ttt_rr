import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TTTLinear(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        
        # W: [num_heads, head_dim, head_dim]
        # Initialize W with small random values
        self.W = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)

    def reset_parameters(self):
        nn.init.normal_(self.W, std=0.02)

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        # Split heads: [B, L, H, D]
        x_heads = x.view(B, L, H, D)
        
        # Pred = einsum('blhd,hde->blhe', X_heads, W)
        out = torch.einsum('blhd,hde->blhe', x_heads, self.W)
        
        # Merge heads: [B, L, C]
        out = out.reshape(B, L, C)
        return out

class TTTBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ttt_linear = TTTLinear(dim, num_heads)
        self.alpha = nn.Parameter(torch.zeros(1))
        
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.1)
        )

    def reset_parameters(self):
        self.ttt_linear.reset_parameters()
        nn.init.zeros_(self.alpha)

    def forward(self, x):
        # x: [B, L, C]
        
        # Path 1: TTT
        residual = x
        x_norm = self.ln1(x)
        ttt_out = self.ttt_linear(x_norm)
        x = residual + self.alpha * ttt_out
        
        # Path 2: MLP
        residual = x
        # MLP block in requirements: LN -> Linear -> GELU -> Linear -> Dropout
        # My definition above includes LN.
        x = residual + self.mlp(x)
        
        return x

class TTTModule(nn.Module):
    def __init__(self, hidden_dim=256, mem_dim=64, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            TTTBlock(hidden_dim, num_heads=4) for _ in range(num_layers)
        ])
        
        # Projection for memory features (Y)
        self.proj_mem = nn.Linear(mem_dim, hidden_dim)
        
        # Global fusion alpha
        self.alpha_global = nn.Parameter(torch.zeros(1))
        
        self.step_counter = 0

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.step_counter = 0
        
    def get_2d_pe(self, H, W, C, device):
        # Fixed 2D Sine-Cos PE
        y_pos = torch.arange(H, device=device).unsqueeze(1) # [H, 1]
        x_pos = torch.arange(W, device=device).unsqueeze(0) # [1, W]
        
        div_term = torch.exp(torch.arange(0, C // 2, 2, device=device).float() * (-math.log(10000.0) / (C // 2)))
        
        pe = torch.zeros(H, W, C, device=device)
        
        # Sine/Cos for X
        pe[:, :, 0::4] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe[:, :, 1::4] = torch.cos(x_pos.unsqueeze(-1) * div_term)
        
        # Sine/Cos for Y
        pe[:, :, 2::4] = torch.sin(y_pos.unsqueeze(-1) * div_term)
        pe[:, :, 3::4] = torch.cos(y_pos.unsqueeze(-1) * div_term)
        
        return pe.flatten(0, 1).unsqueeze(0) # [1, L, C]

    def forward(self, vision_feats):
        # vision_feats: [L_orig, B, C] (L_orig=4096)
        L_orig, B, C = vision_feats.shape
        
        # 1. Permute: [B, L_orig, C]
        x = vision_feats.permute(1, 0, 2)
        
        # 2. Reshape: [B, 64, 64, C] -> [B, C, 64, 64]
        x = x.view(B, 64, 64, C).permute(0, 3, 1, 2)
        
        # 3. AvgPool: [B, C, 32, 32]
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        # 4. Flatten: [B, C, 32, 32] -> [B, 32*32, C]
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Add PE
        pe = self.get_2d_pe(32, 32, C, x.device)
        x = x + pe
        
        # Verify Shape
        if self.step_counter == 0: # Print only once or occasionally
             print(f"[Shape Check] TTT Input X_tokens: {x.shape}")
             assert x.shape == (B, 1024, 256), f"TTT Input Shape Mismatch: {x.shape}"
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x)
            
        # Output processing
        # [B, 1024, 256] -> [B, 32, 32, C]
        x = x.view(B, 32, 32, C).permute(0, 3, 1, 2)
        
        # Upsample: [B, C, 64, 64]
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        
        return x

    def step_update(self, vision_feats, maskmem_features, update_cache=True):
        # vision_feats: [L_orig, B, C]
        # maskmem_features: [B, C_mem, H_mem, W_mem]
        
        self.step_counter += 1
        
        # 1. Prepare X (Same as forward)
        L_orig, B, C = vision_feats.shape
        x = vision_feats.permute(1, 0, 2).view(B, 64, 64, C).permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        pe = self.get_2d_pe(32, 32, C, x.device)
        x = x + pe
        
        # 2. Prepare Y
        y = maskmem_features.detach()
        if y.shape[-2:] != (32, 32):
             y = F.adaptive_avg_pool2d(y, (32, 32))
        
        # [B, 64, 32, 32] -> [B, 32, 32, 64] -> [B, 1024, 64]
        y = y.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Project: [B, 1024, 256]
        y = self.proj_mem(y)
        
        if self.step_counter == 1:
            print(f"[Update Target Check] Y_tokens: {y.shape}")
            assert y.shape == (B, 1024, 256)
        
        # 3. Update Loop
        total_loss = 0
        lr = 0.1 # Heuristic learning rate for TTT
        
        for layer in self.layers:
            # Get X_norm (Input to TTTLinear)
            x_norm = layer.ln1(x)
            
            # Calculate Pred
            # [B, L, H, D]
            x_heads = x_norm.view(B, 1024, 4, 64)
            pred = torch.einsum('blhd,hde->blhe', x_heads, layer.ttt_linear.W)
            pred = pred.reshape(B, 1024, 256)
            
            # Loss
            loss = F.mse_loss(pred, y)
            total_loss += loss
            
            # Update W
            if update_cache:
                grad_W = torch.autograd.grad(loss, layer.ttt_linear.W, create_graph=False)[0]
                layer.ttt_linear.W.data.add_(-lr * grad_W)
                
                # Truncated BPTT
                # if self.step_counter % 8 == 0:
                #    layer.ttt_linear.W.detach_()
            
            # Pass through for next layer (using updated W)
            with torch.no_grad():
                 ttt_out_new = layer.ttt_linear(x_norm)
                 x = x + layer.alpha * ttt_out_new
                 # MLP
                 x = x + layer.mlp(x)

        return total_loss
