"""EgoSpanLift (Sec. 3): lift egocentric gaze into multi-level volumetric visual spans over SLAM keypoints.

For one time step t with visible semidense keypoints, user pose E_t (CPF -> world, z forward) and gaze g_t (in CPF):

  Eq. 1  P_t   = visible points within a cube of side D around the user that pass a neighbour-based statistical
                 outlier filter (no confidence-based filtering: dynamic content such as hands and people is kept)
  Eq. 2  Q_t^e = points of P_t whose angle to the gaze ray (in the local frame) is below eccentricity e;
                 the orientation span uses the head-forward axis z instead of the gaze
  Eq. 3  V     = binary occupancy of each Q in an R^3 grid of side D: a voxel is on if at least one point falls in it

Levels (vision-science taxonomy): orientation 55 deg around head forward, near-peripheral 30, central 8 and foveal
2 deg around gaze, plus a "scene" channel with every point of P_t. Channel order everywhere: CHANNELS.

Grid frame. Eq. 3 voxelises world coordinates relative to the user position, i.e. the grid is axis-aligned with the
(gravity-aligned) world frame: `grid_frame="world"` (default, vertical axis = z). `grid_frame="local"` instead uses the
axes of the CPF at the anchor step (the SensorySpotlight rebuild; vertical axis ~ y). For forecasting, every grid of a
sample (all past steps and the future target) uses one anchor: the user's pose at the last observed step.
"""
from dataclasses import dataclass, field

import numpy as np

LEVELS = ("orientation", "peripheral", "central", "foveal")
CHANNELS = LEVELS + ("scene",)
RAY_CHANNEL = "gaze_ray"          # optional 6th input channel (build.window.ray_channel), see rasterize_ray
N_LEVELS = len(LEVELS)
N_CHANNELS = len(CHANNELS)


@dataclass
class LiftConfig:
    cube_size: float = 3.2                      # D (m)
    resolution: int = 16                        # R  -> D / R = 20 cm voxels
    orientation_deg: float = 55.0               # e_o, around the head-forward axis
    cone_deg: tuple = (30.0, 8.0, 2.0)          # e_p, e_c, e_f, around the gaze direction
    sor_neighbors: int = 20                     # statistical outlier removal (Open3D semantics) on P_t
    sor_std_ratio: float = 2.0                  # keep if mean kNN distance <= mean + ratio * std; 0 disables SOR
    sor_scope: str = "step"                     # "step": on each step's visible points (Eq. 1) | "global": once on
                                                # the whole cloud after the dist_std filter (the SensorySpotlight build)
    max_dist_std: float | None = None           # optional confidence filter on MPS dist_std (paper: off)
    grid_frame: str = "world"                   # "world" | "local"
    crop_around_user: bool = True               # Eq. 1 spatial term, around the user at each step
    grid: str = "cartesian"                     # "cartesian" (the paper: R^3 voxels) | "spherical" (our addition:
                                                # direction-adaptive azimuth x elevation x distance cells, egospan/
                                                # spherical.py; needs grid_frame = local)
    sph_yaw_edges: list | None = None           # spherical bin edges (deg, deg, m); None = the defaults in spherical.py
    sph_pitch_edges: list | None = None
    sph_depth_edges: list | None = None

    @property
    def voxel_size(self):
        return self.cube_size / self.resolution


def statistical_outlier_mask(xyz, nb_neighbors=20, std_ratio=2.0):
    """Neighbour-based statistical outlier removal: True for inliers. Same rule as Open3D's
    remove_statistical_outlier: mean distance to the k nearest neighbours <= global mean + std_ratio * std."""
    n = len(xyz)
    if std_ratio <= 0 or n <= nb_neighbors + 1:
        return np.ones(n, bool)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xyz).query(xyz, k=nb_neighbors + 1)
    md = d[:, 1:].mean(1)
    return md <= md.mean() + std_ratio * md.std()


def gravity_alignment(gravity_world):
    """Rotation R (3x3) such that R @ gravity points to -z. Identity if gravity is unknown or already -z."""
    if gravity_world is None:
        return np.eye(3)
    g = np.asarray(gravity_world, float)
    g = g / (np.linalg.norm(g) + 1e-12)
    t = np.array([0.0, 0.0, -1.0])
    v, c = np.cross(g, t), float(g @ t)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + c))


@dataclass
class GridFrame:
    """The frame a sample's grids live in: grid coords = R_grid_world @ (p_world - origin)."""
    origin: np.ndarray
    R_grid_world: np.ndarray
    vertical_axis: int

    def points(self, p_world):
        return (np.asarray(p_world) - self.origin) @ self.R_grid_world.T

    def dirs(self, v_world):
        return np.asarray(v_world) @ self.R_grid_world.T

    def pose(self, T_world_cpf):
        """CPF pose(s) (..., 4, 4) in world -> in grid coordinates."""
        T = np.array(T_world_cpf, float, copy=True)
        T[..., :3, :3] = self.R_grid_world @ T[..., :3, :3]
        T[..., :3, 3] = self.points(T[..., :3, 3])
        return T

    def matrix(self):
        """T_world_grid (4x4): maps grid coordinates back to world."""
        T = np.eye(4)
        T[:3, :3] = self.R_grid_world.T
        T[:3, 3] = self.origin
        return T


@dataclass
class EgoSpanLift:
    cfg: LiftConfig = field(default_factory=LiftConfig)
    R_align: np.ndarray = field(default_factory=lambda: np.eye(3))   # world -> gravity-aligned world

    def __post_init__(self):
        self.cos_orient = np.cos(np.radians(self.cfg.orientation_deg))
        self.cos_cones = np.cos(np.radians(np.asarray(self.cfg.cone_deg, float)))

    # ------------------------------------------------------------------------------------------------ frames
    def anchor(self, T_world_cpf):
        """Grid frame centred on the user (CPF origin) at the anchor step."""
        if self.cfg.grid_frame == "world":
            return GridFrame(T_world_cpf[:3, 3].copy(), self.R_align.copy(), vertical_axis=2)
        if self.cfg.grid_frame == "local":
            return GridFrame(T_world_cpf[:3, 3].copy(), T_world_cpf[:3, :3].T.copy(), vertical_axis=1)
        raise ValueError(self.cfg.grid_frame)

    # ------------------------------------------------------------------------------------------------ Eq. 1
    def select(self, pts_world, T_world_cpf, dist_std=None, sor=True):
        """Mask over the visible points of one step: spatial cube around the user (axes of the grid frame), optional
        confidence filter, then statistical outlier removal on what remains."""
        keep = np.ones(len(pts_world), bool)
        if self.cfg.max_dist_std is not None and dist_std is not None:
            keep &= dist_std < self.cfg.max_dist_std
        if self.cfg.crop_around_user:
            local = self.anchor(T_world_cpf).points(pts_world)
            keep &= (np.abs(local) < self.cfg.cube_size / 2).all(1)
        idx = np.flatnonzero(keep)
        if sor and len(idx):
            inl = statistical_outlier_mask(pts_world[idx], self.cfg.sor_neighbors, self.cfg.sor_std_ratio)
            keep[idx[~inl]] = False
        return keep

    # ------------------------------------------------------------------------------------------------ Eq. 2
    def classify(self, pts_world, T_world_cpf, gaze_cpf):
        """(5, N) bool: [orientation, peripheral, central, foveal, scene] membership of each point."""
        R, t = T_world_cpf[:3, :3], T_world_cpf[:3, 3]
        local = (pts_world - t) @ R                                  # E_t^-1 p, row-vector form
        u = local / (np.linalg.norm(local, axis=1, keepdims=True) + 1e-9)
        g = np.asarray(gaze_cpf, float)
        g = g / (np.linalg.norm(g) + 1e-12)
        out = np.empty((N_CHANNELS, len(pts_world)), bool)
        out[0] = u[:, 2] > self.cos_orient
        cg = u @ g
        for c, cos_e in enumerate(self.cos_cones, start=1):
            out[c] = cg > cos_e
        out[4] = True
        return out

    # ------------------------------------------------------------------------------------------------ Eq. 3
    def voxel_index(self, pts_world, frame):
        """(N, 3) int voxel indices in `frame` and an (N,) in-grid mask."""
        D, R = self.cfg.cube_size, self.cfg.resolution
        ijk = np.floor((frame.points(pts_world) + D / 2) * (R / D)).astype(np.int64)
        return ijk, ((ijk >= 0) & (ijk < R)).all(1)

    def voxelize(self, pts_world, labels, frame, out=None):
        """OR the labelled points into a (C, R, R, R) bool grid (allocated if `out` is None)."""
        R = self.cfg.resolution
        if out is None:
            out = np.zeros((len(labels), R, R, R), bool)
        ijk, inside = self.voxel_index(pts_world, frame)
        for c in range(len(labels)):
            m = labels[c] & inside
            out[c, ijk[m, 0], ijk[m, 1], ijk[m, 2]] = True
        return out

    # ------------------------------------------------------------------------------------------------ all
    def lift(self, pts_world, T_world_cpf, gaze_cpf, frame=None, dist_std=None):
        """One step end to end -> (5, R, R, R) bool. `frame` defaults to an anchor at this step."""
        frame = frame or self.anchor(T_world_cpf)
        keep = self.select(pts_world, T_world_cpf, dist_std)
        p = pts_world[keep]
        return self.voxelize(p, self.classify(p, T_world_cpf, gaze_cpf), frame)


def voxel_centers(resolution, cube_size):
    """(R, R, R, 3) voxel centre coordinates in the grid frame (m)."""
    c = (np.arange(resolution) + 0.5) * (cube_size / resolution) - cube_size / 2
    return np.stack(np.meshgrid(c, c, c, indexing="ij"), -1)


def rasterize_ray(origin, direction, cube_size=3.2, resolution=16, step_frac=0.25):
    """(R, R, R) bool: voxels crossed by the ray origin + s * direction (s >= 0) inside the grid, both given in grid
    coordinates (m). Sampled every step_frac voxels, so a ray never skips a voxel it passes through by more than a
    sliver. Independent of SLAM points: it records where the eyes pointed even where there are no keypoints."""
    vs = cube_size / resolution
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    s = np.arange(0.0, cube_size * np.sqrt(3) + vs, step_frac * vs)
    pts = np.asarray(origin, float)[None] + s[:, None] * d[None]
    ijk = np.floor((pts + cube_size / 2) / vs).astype(np.int64)
    inside = ((ijk >= 0) & (ijk < resolution)).all(1)
    out = np.zeros((resolution,) * 3, bool)
    out[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]] = True
    return out
