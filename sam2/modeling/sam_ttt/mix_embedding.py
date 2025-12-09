import torch
import torch.nn as nn
from .ttt import TTTLinear, TTTConfig

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=None): # bn argument kept for compatibility but ignored or used if passed explicitly, but we prefer GroupNorm
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        # Replace BatchNorm with GroupNorm for Batch Size 1 stability
        # Using 32 groups is a standard default for GroupNorm
        self.bn = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x

class ME(torch.nn.Module):
    def __init__(self, in_channels=512, out_channels=256):
        super(ME, self).__init__()
        
        # 动态创建配置，确保 hidden_size 匹配 out_channels (256)
        self.config = TTTConfig(use_cache=True, hidden_size=out_channels)

        self.depthwise_conv_reduce_channels_in = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                               stride=1, padding=0, groups=in_channels // 16)
        self.depthwise_conv_reduce_channels_out1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                               stride=1, padding=0, groups=in_channels // 16)
        self.depthwise_conv_reduce_channels_out2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                               stride=1, padding=0, groups=in_channels // 16)

        self.relu = nn.ReLU(True)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.seq_modeling_block = TTTLinear(config=self.config, layer_idx=0)
        self.norm = nn.LayerNorm(out_channels)
        self.embed_tokens = nn.Embedding(out_channels, out_channels, None)

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) == 1:
                    param_unsqueeze = param.unsqueeze(0)
                    nn.init.xavier_uniform_(param_unsqueeze)
                    param.data.copy_(param_unsqueeze.squeeze(0))
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, dense_embeddings_boundary, dense_embeddings_box, high_frequency, sparse_embeddings_box, route):
        dense_cat = torch.cat([dense_embeddings_boundary, dense_embeddings_box], dim=1)
        dense_cat_tmp = self.depthwise_conv_reduce_channels_in(dense_cat)
        dense_em = self.branch1(dense_cat)
        dense_em = dense_em + dense_cat_tmp

        if route == 2:
            B, C, H, W = high_frequency.shape
            # Convert input from (B, C, H, W) to (B, H*W, C)
            high_frequency = high_frequency.float().view(B, C, -1).permute(0, 2, 1)
            high_frequency = self.norm(high_frequency)
            position_ids = torch.arange(
                0,
                high_frequency.shape[1],
                dtype=torch.long,
                device=high_frequency.device,
            ).unsqueeze(0)

            high_frequency = self.seq_modeling_block(
                hidden_states=high_frequency,
                attention_mask=None,
                position_ids=position_ids,
                cache_params=None,
            )
            high_frequency = high_frequency.permute(0, 2, 1).view(B, C, H, W)
        
        dense_embeddings = torch.cat([dense_em, high_frequency], dim=1)

        if route == 1:
            dense_embeddings = self.depthwise_conv_reduce_channels_out1(dense_embeddings)
        if route == 2:
            dense_embeddings = self.depthwise_conv_reduce_channels_out2(dense_embeddings)
        
        sparse_embeddings = sparse_embeddings_box
        return dense_embeddings, sparse_embeddings
