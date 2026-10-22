"""Direction-adaptive spherical grid (our addition, not in the paper): azimuth x elevation x distance cells around the
wearer, dense in front, medium at the sides, coarse behind, still covering all 360 deg.

The visual spans are cones (2 / 8 / 30 deg around the gaze, 55 deg around head forward), so their natural geometry is
angular: a 2 deg cone covers about the same number of angular cells at any distance, while it covers a 20 cm Cartesian
voxel only partly at 1 m and several voxels' worth of angle at 0.5 m. Resolution is spent where attention goes: in our
data 90 % of future gaze is within 56 deg of the anchor head-forward axis, so the dense block covers the orientation
span, and the back (where the front-half crop has nothing) stays available at low cost.

Cells (defaults, in the anchor CPF frame: forward = z, up = y, azimuth = atan2(x, z), elevation = asin(y / r))
    azimuth    |az| <= 55 deg: 5 deg       55-115: 10 deg       115-180: 21.7 deg          -> 40 bins
    elevation  |el| <= 50 deg: 5 deg       50-90: 20 deg                                     -> 24 bins
    distance   0-0.4 m, then 7 geometric bins up to the corner of the paper's 3.2 m cube    -> 8 bins
40 x 24 x 8 = 7,680 cells (the full 16^3 cube has 4,096), every axis divisible by 8 for the U-Net's three poolings.
The bins are uneven in degrees but even in index, so convolutions run on an ordinary tensor: a warped grid, like the
log-polar model of cortical magnification. Only points inside the paper's cube are used, so a sample holds exactly
the points of the Cartesian version.

Scoring stays comparable with the paper: predictions are mapped back to the 20 cm Cartesian grid and scored against
the ordinary Cartesian targets. A voxel takes the max over the cells it overlaps (found with 8^3 sub-points per voxel),
the same rule as the target ("on if any point falls in it"): a true target maps back with no missed voxel, and a
small foveal prediction of one or two fine cells is not averaged away. mode="mean" (overlap-weighted) is available.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def _mirror(pos):
    """edges from 0 up -> symmetric edges from -max to +max."""
    pos = np.asarray(pos, float)
    return np.concatenate([-pos[:0:-1], pos])


def default_edges(cube_size=3.2):
    corner = float(np.sqrt(3) * cube_size / 2)
    yaw = _mirror(np.concatenate([np.arange(0, 56, 5), np.arange(65, 116, 10), [115 + 65 / 3, 115 + 130 / 3, 180]]))
    pitch = _mirror(np.concatenate([np.arange(0, 51, 5), [70, 90]]))
    depth = np.concatenate([[0.0], 0.4 * (corner / 0.4) ** (np.arange(8) / 7)])
    return yaw, pitch, depth


@dataclass(frozen=True)
class SphericalGrid:
    yaw_edges: tuple
    pitch_edges: tuple
    depth_edges: tuple
    cube_size: float = 3.2

    def __post_init__(self):
        y, p, d = (np.asarray(e, float) for e in (self.yaw_edges, self.pitch_edges, self.depth_edges))
        for name, e in (("yaw", y), ("pitch", p), ("depth", d)):
            if np.any(np.diff(e) <= 0):
                raise ValueError(f"{name} edges must increase")
        if y[0] != -180 or y[-1] != 180 or p[0] != -90 or p[-1] != 90 or d[0] != 0:
            raise ValueError("edges must cover azimuth -180..180, elevation -90..90 and distance from 0")
        if not np.allclose(y, -y[::-1]):
            raise ValueError("azimuth edges must be symmetric (left-right mirroring is the augmentation)")
        if d[-1] < np.sqrt(3) * self.cube_size / 2 - 1e-6:
            raise ValueError("the last distance edge must reach the cube corner, so every cube point has a cell")

    @property
    def shape(self):
        return (len(self.yaw_edges) - 1, len(self.pitch_edges) - 1, len(self.depth_edges) - 1)

    @property
    def n_cells(self):
        return int(np.prod(self.shape))

    def angles(self, points):
        """points (N, 3) in the grid frame -> azimuth, elevation (deg), distance (m)."""
        p = np.asarray(points, float)
        r = np.linalg.norm(p, axis=-1)
        az = np.degrees(np.arctan2(p[..., 0], p[..., 2]))
        el = np.degrees(np.arcsin(np.clip(p[..., 1] / np.maximum(r, 1e-12), -1, 1)))
        return az, el, r

    def index(self, points):
        """(N, 3) cell indices and an (N,) mask: inside the paper's cube (so the same points as the Cartesian grid)."""
        p = np.asarray(points, float)
        az, el, r = self.angles(p)
        a, b, c = self.shape
        i = np.clip(np.searchsorted(self.yaw_edges, az, side="right") - 1, 0, a - 1)
        j = np.clip(np.searchsorted(self.pitch_edges, el, side="right") - 1, 0, b - 1)
        k = np.clip(np.searchsorted(self.depth_edges, r, side="right") - 1, 0, c - 1)
        inside = (np.abs(p) < self.cube_size / 2).all(-1)
        return np.stack([i, j, k], -1), inside

    def voxelize(self, points, labels, out=None):
        """OR labelled points (grid frame) into a (C, A, E, D) bool grid."""
        if out is None:
            out = np.zeros((len(labels), *self.shape), bool)
        ijk, inside = self.index(points)
        for ch in range(len(labels)):
            m = labels[ch] & inside
            out[ch, ijk[m, 0], ijk[m, 1], ijk[m, 2]] = True
        return out

    def centers(self):
        """(A, E, D, 3) cell centres (mid azimuth, mid elevation, mid distance) in the grid frame (m)."""
        ya = np.radians((np.asarray(self.yaw_edges[:-1]) + np.asarray(self.yaw_edges[1:])) / 2)
        el = np.radians((np.asarray(self.pitch_edges[:-1]) + np.asarray(self.pitch_edges[1:])) / 2)
        r = (np.asarray(self.depth_edges[:-1]) + np.asarray(self.depth_edges[1:])) / 2
        Y, E, Rr = np.meshgrid(ya, el, r, indexing="ij")
        return np.stack([Rr * np.cos(E) * np.sin(Y), Rr * np.sin(E), Rr * np.cos(E) * np.cos(Y)], -1)

    def rasterize_ray(self, origin, direction, step=0.02):
        """(A, E, D) bool: cells crossed by origin + s * direction (s >= 0) inside the cube (grid coordinates)."""
        d = np.asarray(direction, float)
        d = d / (np.linalg.norm(d) + 1e-12)
        s = np.arange(step, np.sqrt(3) * self.cube_size + step, step)          # from just past the glasses
        ijk, inside = self.index(np.asarray(origin, float)[None] + s[:, None] * d[None])
        out = np.zeros(self.shape, bool)
        out[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]] = True
        return out

    def to_cartesian(self, a, resolution=16, mode="max"):
        """(..., A, E, D) -> (..., R, R, R) float. mode "max": each voxel takes the largest value of the cells it
        overlaps (as the target: on if anything in it is on); "mean": the overlap-weighted mean."""
        voxel, cell, weight, starts = _overlaps(self, resolution)
        a = np.asarray(a)
        lead = a.shape[:-3]
        flat = a.reshape(-1, self.n_cells).astype(np.float32)[:, cell]               # (M, pairs)
        if mode == "max":
            out = np.maximum.reduceat(flat, starts, axis=1)
        elif mode == "mean":
            out = np.add.reduceat(flat * weight, starts, axis=1)
        else:
            raise ValueError(mode)
        return out.reshape(*lead, resolution, resolution, resolution)


@lru_cache(maxsize=8)
def _overlaps(grid, resolution, sub=8):
    """(voxel, cell) overlap pairs sorted by voxel, the overlap share of each pair (sums to 1 per voxel) and the
    first pair of every voxel; from sub^3 sub-points per voxel (5 deg cells are >= 3.5 cm wide beyond 0.4 m, the
    sub-points 2.5 cm apart, so no overlapped cell is missed)."""
    from .lift import voxel_centers
    vs = grid.cube_size / resolution
    off = ((np.arange(sub) + 0.5) / sub - 0.5) * vs
    O = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    pts = voxel_centers(resolution, grid.cube_size).reshape(-1, 1, 3) + O[None]          # (R^3, sub^3, 3)
    ijk, _ = grid.index(pts.reshape(-1, 3))
    key = np.repeat(np.arange(resolution ** 3), sub ** 3) * grid.n_cells + np.ravel_multi_index(ijk.T, grid.shape)
    pairs, counts = np.unique(key, return_counts=True)                                   # sorted by voxel
    voxel, cell = pairs // grid.n_cells, pairs % grid.n_cells
    starts = np.flatnonzero(np.r_[True, voxel[1:] != voxel[:-1]])
    assert len(starts) == resolution ** 3
    return voxel, cell, (counts / sub ** 3).astype(np.float32), starts


def spherical_grid(lift):
    """SphericalGrid for a LiftConfig (or its dict from a benchmark's meta) with grid = "spherical", else None."""
    get = (lambda k, d=None: lift.get(k, d)) if isinstance(lift, dict) else (lambda k, d=None: getattr(lift, k, d))
    if get("grid", "cartesian") != "spherical":
        return None
    cube = float(get("cube_size", 3.2))
    y, p, d = default_edges(cube)
    y = get("sph_yaw_edges") or y
    p = get("sph_pitch_edges") or p
    d = get("sph_depth_edges") or d
    return SphericalGrid(tuple(float(v) for v in y), tuple(float(v) for v in p), tuple(float(v) for v in d), cube)
