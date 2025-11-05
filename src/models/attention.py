import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """
    Lightweight cross-attention: CT tokens (queries) attend to all tokens (including themselves).
    """
    def __init__(self, d_model:int, n_heads:int=6, dropout:float=0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, q:torch.Tensor, kv:torch.Tensor):
        """
        q: [B, K, D]   (CT tokens)
        kv:[B, T, D]   (concat tokens)
        returns [B, T, D] with updated q in the first K positions
        """
        qn = self.ln_q(q); kvn = self.ln_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        q = q + out
        q = q + self.mlp(q)
        return torch.cat([q, kv[:, q.shape[1]:, :]], dim=1)
