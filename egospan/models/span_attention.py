"""Inter-span attention (our addition, not in the paper): cells of one encoder level, from every input grid, attend to
each other before the paper's averaging throws the layout away.

    feats (B, G, C, D, H, W)   encoder features of one level for the G input grids (past steps, then the union grid)
      -> tokens: Linear(C -> dim) + position (learned per axis) + time (learned per grid)   G * D * H * W tokens
      -> pre-norm transformer layers; a token sees every cell of its own grid and of earlier grids (block causal)
         optional span bias: score(q, k) += b(prior_k) per head, prior_k = shares of past orientation / peripheral /
         central / foveal / scene / "seen, not looked at" (peripheral and not central) / in camera view in cell k
      -> LayerNorm, Linear(dim -> C) initialised to zero: the residual update starts at 0, so a fresh model computes
         exactly what the model without this block computes
The caller adds the update to the level's features, so the decoder's skip at that level carries the comparison of
every cell with every other cell (and with where the gaze was in each past step).
"""
import torch
from torch import nn
from torch.nn import functional as F


class _Layer(nn.Module):
    def __init__(self, dim, heads, ff_mult, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff_mult * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype),
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.ff(self.n2(x)))


class SpanAttention(nn.Module):
    def __init__(self, channels, n_grids, dim=256, layers=2, heads=4, ff_mult=4, dropout=0.1, n_prior=0,
                 max_size=64):
        super().__init__()
        assert dim % heads == 0, (dim, heads)
        self.n_grids, self.heads = n_grids, heads
        self.inp = nn.Linear(channels, dim)
        self.pos = nn.ModuleList([nn.Embedding(max_size, dim) for _ in range(3)])
        self.time = nn.Parameter(torch.zeros(n_grids, dim))
        for e in self.pos:
            nn.init.trunc_normal_(e.weight, std=0.02)
        nn.init.trunc_normal_(self.time, std=0.02)
        self.layers = nn.ModuleList([_Layer(dim, heads, ff_mult, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.prior_bias = nn.Linear(n_prior, heads) if n_prior else None
        if self.prior_bias is not None:
            nn.init.zeros_(self.prior_bias.weight)
            nn.init.zeros_(self.prior_bias.bias)

    def positions(self, shape, device):
        idx = [torch.arange(s, device=device) for s in shape]
        ex, ey, ez = (e(i) for e, i in zip(self.pos, idx))
        return (ex[:, None, None] + ey[None, :, None] + ez[None, None, :]).reshape(-1, ex.shape[-1])

    def mask(self, b, cells, prior, device):
        """(B, heads, N, N) additive scores: -inf for later grids, + the span bias of the key cell."""
        g = torch.arange(self.n_grids, device=device).repeat_interleave(cells)
        m = torch.zeros(len(g), len(g), device=device).masked_fill(g[None, :] > g[:, None], float("-inf"))
        m = m.expand(b, self.heads, -1, -1)
        if self.prior_bias is not None and prior is not None:
            bias = self.prior_bias(prior.float())                         # (B, N, heads)
            m = m + bias.permute(0, 2, 1)[:, :, None, :]
        return m

    def tokens(self, feats, prior=None):
        """feats (B, G, C, D, H, W), prior (B, G, P, D, H, W) -> (B, G * D * H * W, dim) token outputs."""
        b, g, c = feats.shape[:3]
        shape = feats.shape[3:]
        assert g == self.n_grids, (g, self.n_grids)
        cells = shape.numel()
        x = self.inp(feats.flatten(3).transpose(2, 3))                    # (B, G, cells, dim)
        x = x + self.positions(shape, feats.device)[None, None] + self.time[None, :, None]
        x = x.reshape(b, g * cells, -1)
        p = None if prior is None else prior.flatten(3).transpose(2, 3).reshape(b, g * cells, -1)
        m = self.mask(b, cells, p, feats.device)
        for layer in self.layers:
            x = layer(x, m)
        return self.norm(x)

    def forward(self, feats, prior=None):
        """-> residual update (B, G, C, D, H, W) for every grid's features."""
        b, g, c = feats.shape[:3]
        shape = feats.shape[3:]
        y = self.out(self.tokens(feats, prior))                            # (B, G * cells, C)
        return y.view(b, g, shape.numel(), c).transpose(2, 3).reshape(b, g, c, *shape).to(feats.dtype)
