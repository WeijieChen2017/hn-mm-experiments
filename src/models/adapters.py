import torch
import torch.nn as nn
import torch.nn.functional as F

class NumericAdapter(nn.Module):
    """Projects numeric features or small token sets into model width."""
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):  # x: [B,F]
        return self.proj(x)

class DoseAdapter3D(nn.Module):
    """Tiny 3D CNN over dose patch -> vector."""
    def __init__(self, out_dim:int, in_ch:int=1):
        super().__init__()
        ch = 16
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, ch, 3, 2, 1), nn.GroupNorm(4, ch), nn.GELU(),
            nn.Conv3d(ch, ch*2, 3, 2, 1), nn.GroupNorm(8, ch*2), nn.GELU(),
            nn.Conv3d(ch*2, ch*4, 3, 2, 1), nn.GroupNorm(16, ch*4), nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Linear(ch*4, out_dim)

    def forward(self, x):  # [B,1,Dz,Hy,Wx]
        z = self.net(x).flatten(1)
        return self.proj(z)

class ContourAdapter3D(nn.Module):
    """Tiny 3D CNN over signed distance map -> vector."""
    def __init__(self, out_dim:int, in_ch:int=1):
        super().__init__()
        ch = 8
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, ch, 3, 2, 1), nn.GroupNorm(4, ch), nn.GELU(),
            nn.Conv3d(ch, ch*2, 3, 2, 1), nn.GroupNorm(8, ch*2), nn.GELU(),
            nn.Conv3d(ch*2, ch*2, 3, 2, 1), nn.GroupNorm(8, ch*2), nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Linear(ch*2, out_dim)

    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.proj(z)
