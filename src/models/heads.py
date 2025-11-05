import torch.nn as nn

class BinaryHead(nn.Module):
    def __init__(self, in_dim:int, hidden:int=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)
