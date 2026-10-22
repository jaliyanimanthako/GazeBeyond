"""3D visual span forecasting network (Sec. 4, Fig. 3a).

    x: (B, Tp, 5, R, R, R)  previous spans from EgoSpanLift, per observed time step
       channels = [orientation, peripheral, central, foveal, scene]

    1. 3D U-Net encoder (shared over steps) -> deepest features (C, 2, 2, 2) -> pooled by R overall -> token v_t (C)
    2. global embedding v_head = encoder(union of all Tp grids), appended after v_1..v_Tp
    3. unidirectional (causal) transformer over [v_1, ..., v_Tp, v_head]; take the output at v_head
    4. U-Net decoder upsamples that (C, 1, 1, 1) embedding to R^3 with skip connections from the encoder:
       the embedding is broadcast to the deepest level (2^3) and added to the deepest skip (bottom="add"), or
       concatenated with it and convolved (bottom="conv"), then the usual decoders go 2^3 -> R^3
    5. 1x1 conv -> logits (B, n_out, R, R, R); sigmoid gives the soft occupancy of the future span union

Paper settings (Appendix A.2): C = 1024, encoder 5-128-256-512-1024 on 16^3 grids, each U-Net layer two
Conv-BatchNorm-ReLU-Dropout(0.1) blocks. Not specified in the paper, so exposed as options: transformer depth / heads /
FFN width, token pooling, which encoder features feed the skips, how the embedding enters the decoder, upsampling.

Ablations from Table 2 map onto options:
    head="none"       w/o global embedding: decode the transformer output of the last step instead of v_head
    head="learned"    a learned query token instead of the encoded union grid (as in the SensorySpotlight reimpl.)
    n_out=1           single-task (train one model per span level; select the level in the data config)
    (w/o previous span is an input ablation: data.drop_span_input=true)

Camera features (the SensorySpotlight `vid` input, video_dim > 0): lifted V-JEPA token features (video_dim channels + an
in-view mask, egospan/video.py) are reduced by a 1x1x1 conv to video_channels and concatenated, with the mask, to every
input grid (each time step and the union grid), so they reach both the tokens and the decoder's skip features.

Gaze-path branch (our addition, cue_dim > 0): the past gaze / head path, cues (B, steps, cue_dim) at 10 Hz, is encoded
by a GRU into one vector, projected to C and placed as the first transformer token, [v_gaze, v_1, ..., v_Tp, v_head],
so every later token (and so the decoded head output) can attend to it.

Inter-span attention (our addition, span_attention = "attn" | "bias"): the encoder features of level sa_level (default
2: 8 x 8 x 4 cells of 40 cm in the front half) of every input grid become tokens that attend to each other (block
causal over the grids), optionally with a learned bias from each cell's past span shares ("bias"); the zero-initialised
update is added to that level's features, so the decoder's skip there carries it (egospan/models/span_attention.py).

Copy path (our addition, copy_path = "add" | "gate"): the spans of the last observed second (copy_bins input bins) go
straight to the output, so the network can keep the recent attention region and only has to learn the changes.
    add    logits += w_l * (2 * last - 1), one weight per level, starting at 0 (= the model without it)
    gate   p = g * last + (1 - g) * sigmoid(logits), g = sigmoid(1x1 conv of the decoder features) per cell and
           level, starting at g ~ 0.02: the model can keep the last region where gaze is steady and override it
           where a shift is coming
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..lift import CHANNELS
from .blocks import Decoder, create_decoders, create_encoders
from .span_attention import SpanAttention

SPAN_PRIORS = CHANNELS + ("seen, not looked at",)


class SpanForecaster(nn.Module):
    def __init__(self, in_channels=5, n_out=4, num_steps=2, f_maps=(128, 256, 512, 1024), layer_order="cbrd",
                 num_groups=8, dropout=0.1, upsample="nearest", token_pool="avg", tf_layers=4, tf_heads=8,
                 tf_ff_mult=4, tf_dropout=0.1, head="union", skip="head", bottom="add", cue_dim=0, cue_hidden=256,
                 cue_layers=2, video_dim=0, video_channels=16, span_attention="none", sa_level=2, sa_dim=256,
                 sa_layers=2, sa_heads=4, sa_ff_mult=4, sa_dropout=0.1, copy_path="none", copy_bins=1,
                 copy_channels=None):
        super().__init__()
        assert span_attention in ("none", "attn", "bias"), span_attention
        assert copy_path in ("none", "add", "gate"), copy_path
        assert head in ("union", "learned", "none"), head
        assert skip in ("head", "last", "concat"), skip
        assert token_pool in ("avg", "max"), token_pool
        assert bottom in ("add", "conv"), bottom
        if skip == "head" and head != "union":
            skip = "last"                       # no encoded head grid to take skips from
        self.in_channels, self.n_out, self.num_steps = in_channels, n_out, num_steps
        self.video_dim = video_dim
        if video_dim:
            self.video_red = nn.Conv3d(video_dim, video_channels, 1)
        enc_in = in_channels + (video_channels + 1 if video_dim else 0)
        self.head, self.skip, self.token_pool, self.bottom_mode = head, skip, token_pool, bottom
        f_maps = list(f_maps)
        c = f_maps[-1]
        self.encoders = create_encoders(enc_in, f_maps, dropout_prob=dropout, layer_order=layer_order,
                                        num_groups=num_groups)
        self.cue_dim = cue_dim
        n_tok = num_steps + (head != "none") + (cue_dim > 0)
        if cue_dim:
            self.cue_encoder = nn.GRU(cue_dim, cue_hidden, cue_layers, batch_first=True)
            self.cue_proj = nn.Linear(cue_hidden, c)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, c))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if head == "learned":
            self.head_token = nn.Parameter(torch.zeros(1, 1, c))
            nn.init.trunc_normal_(self.head_token, std=0.02)
        layer = nn.TransformerEncoderLayer(c, tf_heads, tf_ff_mult * c, dropout=tf_dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, tf_layers, enable_nested_tensor=False)
        self.tf_norm = nn.LayerNorm(c)
        self.register_buffer("causal_mask", torch.triu(torch.full((n_tok, n_tok), float("-inf")), 1), persistent=False)
        if skip == "concat":                    # fuse the skips of all encoded grids with a 1x1 conv per level
            n_grids = num_steps + (head == "union")
            self.skip_fuse = nn.ModuleList([nn.Conv3d(w * n_grids, w, 1) for w in f_maps])
        if bottom == "conv":                    # 1^3 -> 2^3: concatenate with the deepest skip + DoubleConv
            self.bottom = Decoder(2 * c, c, conv_layer_order=layer_order, num_groups=num_groups,
                                  upsample="nearest", dropout_prob=dropout)
        self.decoders = create_decoders(f_maps, layer_order=layer_order, num_groups=num_groups, upsample=upsample,
                                        dropout_prob=dropout)
        self.final_conv = nn.Conv3d(f_maps[0], n_out, 1)
        # copy path: the last observed second's spans go straight to the output (zero / near-zero at the start)
        self.copy_path, self.copy_bins = copy_path, copy_bins
        self.copy_channels = list(copy_channels) if copy_channels is not None else list(range(n_out))
        if copy_path == "add":
            self.copy_w = nn.Parameter(torch.zeros(n_out))
        elif copy_path == "gate":
            self.copy_gate = nn.Conv3d(f_maps[0], n_out, 1)
            nn.init.zeros_(self.copy_gate.weight)
            nn.init.constant_(self.copy_gate.bias, -4.0)                    # g = 0.018: almost the plain forecast
        self.span_attention, self.sa_level, self.video_channels = span_attention, sa_level, video_channels
        if span_attention != "none":
            if not 1 <= sa_level <= len(f_maps):
                raise ValueError(f"sa_level {sa_level}: the encoder has levels 1..{len(f_maps)}")
            n_grids = num_steps + (head == "union")
            n_prior = (len(SPAN_PRIORS) + bool(video_dim)) if span_attention == "bias" else 0
            self.span_attn = SpanAttention(f_maps[sa_level - 1], n_grids, sa_dim, sa_layers, sa_heads, sa_ff_mult,
                                           sa_dropout, n_prior)

    def span_prior(self, grids, size):
        """grids (B, G, C, R, R, R') -> (B, G, P, *size): per encoder cell, the share of its voxels in each past span,
        scene, "seen, not looked at" (peripheral and not central) and, with camera features, the camera view."""
        o, p, c, f, s = (grids[:, :, CHANNELS.index(k)].float() for k in CHANNELS)
        maps = [o, p, c, f, s, p * (1 - c)]
        if self.video_dim:
            maps.append(grids[:, :, self.in_channels + self.video_channels].float())     # in-view mask
        m = torch.stack(maps, 2)
        b, g = m.shape[:2]
        k = 2 ** (self.sa_level - 1)
        pooled = F.avg_pool3d(m.flatten(0, 1), k) if k > 1 else m.flatten(0, 1)
        assert tuple(pooled.shape[-3:]) == tuple(size), (pooled.shape, size)
        return pooled.view(b, g, *pooled.shape[1:])

    def encode(self, g):
        """g: (N, 5, R, R, R) -> list of features per level, shallow to deep."""
        feats = []
        for enc in self.encoders:
            g = enc(g)
            feats.append(g)
        return feats

    def forward(self, x, cues=None, video=None):
        """x: (B, Tp, 5, R, R, R) float in {0, 1}; cues: (B, steps, cue_dim) when the gaze branch is on.
        Returns logits (B, n_out, R, R, R)."""
        b, tp = x.shape[:2]
        assert tp == self.num_steps, f"model built for Tp={self.num_steps}, got {tp}"
        last = x[:, -self.copy_bins:, self.copy_channels].amax(1) if self.copy_path != "none" else None
        if self.video_dim:
            if video is None:
                raise ValueError("this model uses camera features: pass video (set data.video_features=true)")
            video = video.to(x.dtype)
            feat = torch.cat([self.video_red(video[:, :self.video_dim]), video[:, self.video_dim:]], 1)
            x = torch.cat([x, feat[:, None].expand(-1, tp, -1, -1, -1, -1)], 2)
        grids = x
        if self.head == "union":
            grids = torch.cat([x, x.amax(1, keepdim=True)], 1)            # (B, Tp + 1, 5, R, R, R)
        n_g = grids.shape[1]
        feats = self.encode(grids.flatten(0, 1))                           # each (B * n_g, w, s, s, s)
        deep = feats[-1]
        tok = deep.mean((2, 3, 4)) if self.token_pool == "avg" else deep.amax((2, 3, 4))
        tok = tok.view(b, n_g, -1)
        if self.head == "learned":
            tok = torch.cat([tok, self.head_token.expand(b, -1, -1)], 1)
        if self.cue_dim:
            if cues is None:
                raise ValueError("this model has a gaze-path branch: pass cues (set data.gaze_paths=true)")
            _, hn = self.cue_encoder(cues.float())
            tok = torch.cat([self.cue_proj(hn[-1]).to(tok.dtype)[:, None], tok], 1)
        h = self.transformer(tok + self.pos, mask=self.causal_mask)
        h = self.tf_norm(h[:, -1])                                         # v~_head (or v~_Tp when head="none")
        feats = [f.view(b, n_g, *f.shape[1:]) for f in feats]
        if self.span_attention != "none":           # every cell of this level, of every grid, attends to every cell
            lv = self.sa_level - 1
            prior = self.span_prior(grids, feats[lv].shape[-3:]) if self.span_attention == "bias" else None
            feats[lv] = feats[lv] + self.span_attn(feats[lv], prior)
        if self.skip == "head":
            skips = [f[:, -1] for f in feats]
        elif self.skip == "last":
            skips = [f[:, self.num_steps - 1] for f in feats]
        else:
            skips = [fuse(f.flatten(1, 2)) for fuse, f in zip(self.skip_fuse, feats)]
        h = h[:, :, None, None, None]
        d = skips[-1] + h if self.bottom_mode == "add" else self.bottom(skips[-1], h)
        for dec, s in zip(self.decoders, reversed(skips[:-1])):
            d = dec(s, d)
        logits = self.final_conv(d)
        if self.copy_path == "add":                 # logits + w * (+1 where the span was last second, -1 elsewhere)
            logits = logits + self.copy_w.view(1, -1, 1, 1, 1).to(logits.dtype) * (2 * last.to(logits.dtype) - 1)
        elif self.copy_path == "gate":              # p = g * last second + (1 - g) * forecast, g learned per cell
            g = torch.sigmoid(self.copy_gate(d).float())
            p = g * last.float() + (1 - g) * torch.sigmoid(logits.float())
            logits = torch.logit(p.clamp(1e-6, 1 - 1e-6))
        return logits


class FlatUNet3D(nn.Module):
    """Baseline: all Tp grids stacked as Tp * 5 channels into a plain pytorch-3dunet style 3D U-Net (no temporal
    model). This is the SensorySpotlight "flat" U-Net; it is not part of the paper."""

    def __init__(self, in_channels=5, n_out=4, num_steps=2, f_maps=(32, 64, 128, 256), layer_order="cgr",
                 num_groups=8, dropout=0.1, upsample="nearest", **_):
        super().__init__()
        self.in_channels, self.num_steps = in_channels, num_steps
        self.encoders = create_encoders(in_channels * num_steps, list(f_maps), dropout_prob=dropout,
                                        layer_order=layer_order, num_groups=num_groups)
        self.decoders = create_decoders(list(f_maps), layer_order=layer_order, num_groups=num_groups,
                                        upsample=upsample, dropout_prob=dropout)
        self.final_conv = nn.Conv3d(f_maps[0], n_out, 1)

    def forward(self, x, cues=None, video=None):
        g = x.flatten(1, 2)
        feats = []
        for enc in self.encoders:
            g = enc(g)
            feats.insert(0, g)
        for dec, s in zip(self.decoders, feats[1:]):
            g = dec(s, g)
        return self.final_conv(g)


def soft_occupancy(logits):
    return torch.sigmoid(logits)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

