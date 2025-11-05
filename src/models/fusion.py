import torch
import torch.nn as nn

class AttnPool1D(nn.Module):
    """Simple attention pooling across a token sequence [B, T, D] -> [B, D]."""
    def __init__(self, d:int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.w, std=0.02)

    def forward(self, tokens):
        # scores = tanh(t)*w -> softmax
        scores = torch.tanh(tokens) @ self.w
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (tokens * attn).sum(dim=1)
