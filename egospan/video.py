"""Camera features lifted into the 3D grid (the SensorySpotlight `vid` input, tools/fovs_train_v.py).

Per sample the lab stores a frozen V-JEPA 2 token map of the last ~0.4 s of RGB video (PCA-128, 16 x 16 image cells;
vjepa_{part}.npz / vjepa_all.npz: map (n, 128, 16, 16) fp16, glob (n, 1024), has (n,)). Lifting: every voxel of the
anchor-CPF grid takes the token of the image cell its centre projects into, using the lab's fitted mapping from CPF
azimuth / elevation to normalised preview-image coordinates
    u = 0.515 - 0.00754 az,   v = 0.451 - 0.00784 el    (degrees)
Voxels behind the wearer or outside the image get zeros; an in-view mask channel says which voxels can be seen.
This needs grids in the anchor CPF frame (grid_frame = local, as in the lab's data and our lab rebuild).
"""
import numpy as np

KX, KY, X0, Y0 = -0.00754, -0.00784, 0.515, 0.451
TOKENS = 16                                   # image cells per side


def lift_index(resolution=16, cube_size=3.2, tokens=TOKENS):
    """(R^3,) token index (0 .. tokens^2 - 1) of the image cell each voxel centre projects into, -1 if not visible.
    Voxel order = C order over (x, y, z) of the CPF grid, as everywhere else."""
    c = (np.arange(resolution) + 0.5) * (cube_size / resolution) - cube_size / 2
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    az, el = np.degrees(np.arctan2(x, z)), np.degrees(np.arcsin(y / r))
    u, v = X0 + KX * az, Y0 + KY * el
    ok = (z > 0) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    idx = np.floor(v * tokens).astype(int) * tokens + np.floor(u * tokens).astype(int)
    return np.where(ok, idx, -1).reshape(-1)


def image_cell(points, tokens=TOKENS):
    """(N, 3) points in the anchor CPF frame -> (N,) token index of the image cell they project into, -1 if unseen."""
    x, y, z = np.moveaxis(np.asarray(points, float), -1, 0)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    az, el = np.degrees(np.arctan2(x, z)), np.degrees(np.arcsin(np.clip(y / np.maximum(r, 1e-12), -1, 1)))
    u, v = X0 + KX * az, Y0 + KY * el
    ok = (z > 0) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    idx = np.floor(v * tokens).astype(int) * tokens + np.floor(u * tokens).astype(int)
    return np.where(ok, idx, -1)


def lift_index_spherical(grid, tokens=TOKENS):
    """(cells,) token index of every spherical cell's centre direction, -1 if not in the camera image."""
    return image_cell(grid.centers().reshape(-1, 3), tokens)


def lift(token_map, index, resolution=16):
    """token_map (C, tokens, tokens) -> (C + 1, *shape) float32: lifted features, then the in-view mask channel.
    resolution: R for an R^3 grid, or the grid shape (e.g. a spherical grid's (A, E, D))."""
    shape = (resolution,) * 3 if np.isscalar(resolution) else tuple(resolution)
    C = token_map.shape[0]
    m = np.asarray(token_map, np.float32).reshape(C, -1)
    seen = index >= 0
    out = np.zeros((C + 1, int(np.prod(shape))), np.float32)
    out[:C, seen] = m[:, index[seen]]
    out[C, seen] = 1.0
    return out.reshape(C + 1, *shape)
