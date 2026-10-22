"""Learning objectives (Sec. 4). Spans occupy a tiny part of the grid (foveal: < 1 % of voxels), hence Dice.

Distance-aware term (our addition, `dice+dist`). Dice only rewards overlap: a prediction one voxel next to the target
costs as much as one a metre away. For the chosen levels (default central + foveal) we add

    forward   L_fwd = sum_i p_i D_i / (sum_i p_i + eps)        D_i = distance from voxel i to the nearest target voxel
              = the expected distance of the predicted probability mass from the target ("are my guesses close?")
    reverse   L_rev = mean_j  softmin-distance from target voxel j to the predicted mass
              ("is every part of the target covered?"; the forward term alone is happy with one confident voxel
              inside a target that has several spots, which the 2 s union often has)

with distances in voxels (20 cm = 1), so L ~ 1-3 for typical misses and dist_weight 0.05-0.2 puts it on Dice's scale.
mode "oneway" uses L_fwd, "bidirectional" (L_fwd + L_rev) / 2. Samples with an empty target at a level get no
distance term there (Dice still applies). Distances are computed on the fly from the (augmented, cropped) target.
"""
import torch
import torch.nn.functional as F

from .lift import LEVELS


def dice_loss(logits, target, smooth=1.0):
    """Eq. 4: 1 - 2 sum(Y~ * Y) / (sum Y~ + sum Y + 1), with Y~ = sigmoid(logits), per (sample, level), averaged."""
    p = torch.sigmoid(logits.float()).flatten(2)
    y = target.float().flatten(2)
    return (1 - 2 * (p * y).sum(-1) / (p.sum(-1) + y.sum(-1) + smooth)).mean()


def bce_loss(logits, target):
    """The L_BCE ablation of Table 2."""
    return F.binary_cross_entropy_with_logits(logits.float(), target.float())


def _voxel_coords(shape, device):
    axes = [torch.arange(n, device=device, dtype=torch.float32) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)        # (N, 3) in voxels


def distance_terms(prob, target, tau=1.0, max_target_voxels=2048, eps=1e-6):
    """prob, target: (X, Y, Z) for one sample and level -> (L_fwd, L_rev) in voxels, or None if the target is empty."""
    C = _voxel_coords(prob.shape, prob.device)
    y = target.reshape(-1) > 0.5
    if not y.any():
        return None
    G = C[y]
    if len(G) > max_target_voxels:                                   # bound memory on very large targets
        G = G[torch.randperm(len(G), device=G.device)[:max_target_voxels]]
    Dm = torch.cdist(C, G)                                           # (N, K) voxel-to-target-voxel distances
    p = prob.reshape(-1)
    fwd = (p * Dm.min(1).values).sum() / (p.sum() + eps)
    W = p[:, None] * torch.exp(-Dm / tau)                            # soft nearest predicted mass per target voxel
    rev = ((W * Dm).sum(0) / (W.sum(0) + eps)).mean()
    return fwd, rev


def distance_loss(logits, target, channels, mode="oneway", tau=1.0):
    """mean over (sample, chosen level) with a non-empty target of the forward (or bidirectional) distance term."""
    prob = torch.sigmoid(logits.float())
    terms = []
    for b in range(prob.shape[0]):
        for c in channels:
            t = distance_terms(prob[b, c], target[b, c], tau)
            if t is not None:
                terms.append(t[0] if mode == "oneway" else 0.5 * (t[0] + t[1]))
    return torch.stack(terms).mean() if terms else prob.sum() * 0.0


LOSSES = {"dice": dice_loss, "bce": bce_loss, "dice+bce": lambda l, t: dice_loss(l, t) + bce_loss(l, t)}


def build_loss(name, levels=LEVELS, dist_weight=0.1, dist_levels=("central", "foveal"), dist_mode="oneway",
               dist_tau=1.0):
    """name: dice | bce | dice+bce | dice+dist. levels: the output channels' level names (cfg.data.levels)."""
    if name == "dice+dist":
        if dist_mode not in ("oneway", "bidirectional"):
            raise ValueError(f"dist_mode {dist_mode!r}; choose oneway or bidirectional")
        channels = [list(levels).index(lv) for lv in dist_levels if lv in levels]
        if not channels:
            raise ValueError(f"none of dist_levels {list(dist_levels)} is among the trained levels {list(levels)}")
        return lambda l, t: dice_loss(l, t) + dist_weight * distance_loss(l, t, channels, dist_mode, dist_tau)
    if name not in LOSSES:
        raise ValueError(f"loss {name!r}; choose from {sorted(LOSSES) + ['dice+dist']}")
    return LOSSES[name]


def loss_from_config(cfg):
    t = cfg.train
    return build_loss(t.loss, cfg.data.levels, t.dist_weight, t.dist_levels, t.dist_mode, t.dist_tau)
