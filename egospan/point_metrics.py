"""Point-level scores: one yardstick for any grid (our addition).

IoU on voxels depends on the voxel size, and spherical cells are not voxels, so the two cannot be compared directly.
The thing both grids try to predict is the same: which scene points the person will look at. So score each model on
the future SLAM points themselves (egospan.data.build.future_points): a point counts as predicted for a span if the
voxel / cell it falls in is predicted on, and as true if the point really was in that span during the forecast window.

    IoU / F1 / precision / recall over the points of a sample, averaged over samples (NaN where undefined), as the
    voxel metrics. A grid with coarse cells pays here for every point it lumps together with true ones (precision),
    so even a perfect grid prediction scores below 1: report it as the representation's ceiling ("perfect grid").

Caveats: empty space costs nothing (there are no points in it), and dense surfaces weigh more than sparse ones.
"""
import numpy as np


def lookup(pred, xyz, sample, grid=None, cube_size=3.2):
    """pred (n, L, *shape) bool, xyz (M, 3) points in each sample's grid frame, sample (M,) their sample row ->
    (M, L) bool: is the voxel (grid None: an R^3 grid of side cube_size) or cell (grid: a SphericalGrid) of each point
    predicted on."""
    if grid is None:
        R = pred.shape[-1]
        ijk = np.clip(np.floor((xyz + cube_size / 2) * (R / cube_size)).astype(np.int64), 0, R - 1)
    else:
        ijk, _ = grid.index(xyz)
    return pred[sample, :, ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def point_scores(hit, truth, sample, n):
    """hit, truth (M, L) bool, sample (M,) -> dict of (n, L) per-sample iou / f1 / precision / recall."""
    L = hit.shape[1]
    out = {k: np.full((n, L), np.nan) for k in ("iou", "f1", "precision", "recall")}
    for c in range(L):
        tp = np.bincount(sample, weights=hit[:, c] & truth[:, c], minlength=n)
        p = np.bincount(sample, weights=hit[:, c], minlength=n)
        t = np.bincount(sample, weights=truth[:, c], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["iou"][:, c] = np.where(t > 0, tp / np.maximum(p + t - tp, 1), np.nan)
            out["f1"][:, c] = np.where(t > 0, 2 * tp / np.maximum(p + t, 1), np.nan)
            out["precision"][:, c] = np.where(p > 0, tp / np.maximum(p, 1), np.nan)
            out["recall"][:, c] = np.where(t > 0, tp / np.maximum(t, 1), np.nan)
    return out


def truth_bits(bits, levels):
    """(M,) uint8 span bits -> (M, L) bool for the requested levels (bit c = LEVELS[c])."""
    from .lift import LEVELS
    return np.stack([(bits >> LEVELS.index(l)) & 1 for l in levels], 1).astype(bool)
