import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
    def forward(self, x):
        # x: [B, C, H, W]
        return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias).permute(0, 3, 1, 2)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SS2D(nn.Module):
    def __init__(self, d_model, d_state=1, d_conv=3, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, d_conv, padding=(d_conv-1)//2, groups=self.d_inner, bias=False)
        
        self.x_proj_weight = nn.Parameter(torch.empty(4, self.dt_rank + self.d_state * 2, self.d_inner))
        self.dt_projs_weight = nn.Parameter(torch.empty(4, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(4, self.d_inner))
        self.A_logs = nn.Parameter(torch.empty(self.d_inner * 4, 1))
        self.Ds = nn.Parameter(torch.empty(self.d_inner * 4))
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        B, H, W, D = x.shape
        x = self.in_proj(x)
        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.conv2d(x)
        x = F.silu(x)
        x = rearrange(x, 'b d h w -> b h w d')
        x = self.out_norm(x)
        x = self.out_proj(x)
        return x

class VSSBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.op = SS2D(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(in_features=d_model, hidden_features=d_model * 4)

    def forward(self, x):
        x = x + self.op(self.norm(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VSSM(nn.Module):
    def __init__(self, num_classes=1000, depths=[2, 2, 15, 2], dims=[128, 256, 512, 1024]):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ChannelLayerNorm(64),
            nn.ReLU(),
            nn.Identity(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ChannelLayerNorm(128)
        )
        
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = nn.Module()
            layer.blocks = nn.ModuleList([VSSBlock(dims[i]) for _ in range(depths[i])])
            if i < len(depths) - 1:
                layer.downsample = nn.Sequential(
                    nn.Identity(),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.Identity(),
                    ChannelLayerNorm(dims[i+1])
                )
            self.layers.append(layer)
            
        self.classifier = nn.Module()
        self.classifier.norm = nn.LayerNorm(dims[-1])
        self.classifier.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b h w c')
        for layer in self.layers:
            for block in layer.blocks:
                x = block(x)
            if hasattr(layer, 'downsample'):
                x = rearrange(x, 'b h w c -> b c h w').contiguous()
                x = layer.downsample(x)
                x = rearrange(x, 'b c h w -> b h w c')
        
        x = x.mean(dim=(1, 2))
        x = self.classifier.norm(x)
        x = self.classifier.head(x)
        return x

def vssm_base(num_classes=1000):
    return VSSM(num_classes=num_classes, depths=[2, 2, 15, 2], dims=[128, 256, 512, 1024])
