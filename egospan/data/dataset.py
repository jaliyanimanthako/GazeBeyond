"""PyTorch dataset over a packed benchmark (all.npz + splits.npz written by egospan.data.build.pack).

Grids stay bit-packed in memory and are unpacked per item, so FoVS-EgoExo scale data (341k samples) fits in RAM.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..lift import LEVELS, N_CHANNELS, N_LEVELS
from ..spherical import spherical_grid
from .augment import MirrorAugment, SpanAugment, transform_vectors
from .build import decode_inputs, input_channels, unpack_grids


class PackedBenchmark:
    """Lazily opened all.npz + splits.npz; shared by the datasets of one benchmark."""

    def __init__(self, root):
        self.root = root
        z = np.load(os.path.join(root, "all.npz"))
        self.arrays = {k: z[k] for k in z.files if k != "meta"}
        self.meta = json.loads(str(z["meta"]))
        self.splits = dict(np.load(os.path.join(root, "splits.npz")))
        self.R = self.meta["lift"]["resolution"]
        self.grid = spherical_grid(self.meta["lift"])             # None for the paper's R^3 voxels
        self.shape = self.grid.shape if self.grid is not None else (self.R,) * 3
        w = self.meta["window"]
        self.past_units, self.steps_per_unit = w["past_units"], w["steps_per_unit"]
        self.Tp = w.get("time_bins") or w["past_units"]          # input grids per sample (= model time steps)
        self.soft = bool(w.get("soft", False))                    # soft temporal bins: float inputs in [0, 1]
        self.in_channels = input_channels(w)                      # 5, or 6 with the gaze-ray channel
        self.Tf = self.meta["window"]["future_units"]
        self.vertical_axis = self.meta["vertical_axis"]
        self._gaze = None
        self._video = None

    def video(self):
        """Camera token maps per sample (scripts/attach_video_features.py) and the voxel -> image-cell lift index."""
        if self._video is None:
            from ..video import camera_of, lift_index, lift_index_spherical
            p = os.path.join(self.root, "video_all.npz")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"{p} missing: run scripts/attach_video_features.py --bench {self.root}")
            if self.vertical_axis != 1:
                raise ValueError("camera features are lifted with the CPF -> image mapping: they need grids in the "
                                 "anchor CPF frame (grid_frame = local)")
            z = np.load(p)
            if len(z["map"]) != len(self.arrays["X"]) or not np.array_equal(z["rec"], self.arrays["rec"]):
                raise ValueError(f"{p} is not aligned with all.npz: re-run attach_video_features.py")
            cam = camera_of(self.meta)
            index = (lift_index_spherical(self.grid, camera=cam) if self.grid is not None
                     else lift_index(self.R, self.meta["lift"]["cube_size"], camera=cam))
            self._video = (z["map"], index)
        return self._video

    def gaze(self):
        """Past gaze / head path per sample: (path (N, steps, F), layout). From <root>/gaze_all.npz (written by
        scripts/build_gaze_paths.py for the converted lab data) or, for our own builds, which store every sample's
        poses and gaze, computed from them (path_from_arrays: the same 11 features per step)."""
        if self._gaze is None:
            p = os.path.join(self.root, "gaze_all.npz")
            if not os.path.isfile(p) and {"gaze_grid", "poses_grid", "gaze_cpf"} <= set(self.arrays):
                self._gaze = path_from_arrays(self.arrays, self.past_units * self.steps_per_unit)
                return self._gaze
            if not os.path.isfile(p):
                raise FileNotFoundError(f"{p} missing: run scripts/build_gaze_paths.py --out {self.root}")
            g = np.load(p)
            if len(g["path"]) != len(self.arrays["X"]) or not np.array_equal(g["rec"], self.arrays["rec"]):
                raise ValueError(f"{p} is not aligned with all.npz: rebuild it")
            self._gaze = (g["path"], json.loads(str(g["meta"])))
        return self._gaze

    def indices(self, part, fold=None):
        key = part if fold is None else f"fold{fold}_{part}"
        if key not in self.splits:
            raise KeyError(f"{key} not in {self.root}/splits.npz (have {sorted(self.splits)})")
        return self.splits[key]

    def inputs(self, idx):
        """(..., Tp, 5, *shape): bool for binary inputs, float32 in [0, 1] for soft temporal bins."""
        return decode_inputs(self.arrays["X"][idx], (self.Tp, self.in_channels, *self.shape), self.soft)

    def future(self, idx, key="YF"):
        shape = self.shape if key == "YF" else (self.R,) * 3
        return unpack_grids(self.arrays[key][idx], (self.Tf, N_LEVELS, *shape))

    def target(self, idx, horizon=None, levels=LEVELS, key="YF"):
        """Union of the first `horizon` future units (all if None) for the requested levels."""
        yf = self.future(idx, key)
        y = yf[..., :horizon or self.Tf, :, :, :, :].any(-5)
        return y[..., [LEVELS.index(l) for l in levels], :, :, :]

    def eval_target(self, idx, horizon=None, levels=LEVELS):
        """The target that is scored: always the R^3 Cartesian grid (for spherical benchmarks the YF_cart copy)."""
        return self.target(idx, horizon, levels, "YF_cart" if self.grid is not None else "YF")

    def to_cartesian(self, a):
        """(..., *shape) -> (..., R, R, R) float (identity for Cartesian benchmarks)."""
        return np.asarray(a, np.float32) if self.grid is None else self.grid.to_cartesian(a, self.R)


GAZE_LAYOUT = {"vectors": [[0, 3], [3, 6]], "pseudovectors": [[6, 9]], "n_features": 11}


def path_from_arrays(arrays, steps):
    """The gaze branch's input from a benchmark that stores per-sample poses and gaze (our builds): per observed step
    (the first `steps` stored, i.e. the past window, oldest first) the features of scripts/build_gaze_paths.py:
        0:3 gaze direction in the grid frame (anchor)    3:6 gaze direction in the CPF at that step (eye-in-head)
        6:9 rotation from that step's CPF to the grid frame, as a rotation vector (rad)    9, 10 gaze / head valid
    (every step of a sample is valid: windows with a gap are not built)."""
    from scipy.spatial.transform import Rotation
    g = arrays["gaze_grid"][:, :steps].astype(np.float64)
    e = arrays["gaze_cpf"][:, :steps].astype(np.float64)
    R = arrays["poses_grid"][:, :steps, :3, :3].astype(np.float64)
    n = len(g)
    out = np.zeros((n, steps, GAZE_LAYOUT["n_features"]), np.float32)
    out[..., 0:3] = g / np.linalg.norm(g, axis=-1, keepdims=True)
    out[..., 3:6] = e / np.linalg.norm(e, axis=-1, keepdims=True)
    out[..., 6:9] = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(n, steps, 3)
    out[..., 9:11] = 1.0
    return out, dict(GAZE_LAYOUT)


class SpanDataset(Dataset):
    """Items: x (Tp, 5, R, R, R) float (Tp = input time bins), y (L, R, R, R) float, idx (index into all.npz).

    crop_front keeps only the half of the grid in front of the wearer (last axis, z >= 0) for the input, giving
    x (Tp, 5, R, R, R/2); crop_target (default: same as crop_front) does the same for y. Only valid for grids in the
    anchor-CPF frame (vertical axis 1, forward = last axis), e.g. the converted SensorySpotlight data.

    Spherical benchmarks: x and the training y are spherical cells; eval_targets=true (val / test) gives the R^3
    Cartesian target instead, and to_eval maps predictions to that grid. Augmentation = left-right mirror only."""

    def __init__(self, bench, part, fold=None, horizon=None, levels=LEVELS, augment=False, max_shift=2,
                 drop_span_input=False, seed=None, categories=None, crop_front=False, crop_target=None,
                 gaze_paths=False, video=False, eval_targets=False):
        self.bench = bench if isinstance(bench, PackedBenchmark) else PackedBenchmark(bench)
        self.idx = self.bench.indices(part, fold)
        if categories:
            self.idx = self.idx[np.isin(self.bench.arrays["category"][self.idx], list(categories))]
        self.horizon, self.levels, self.drop_span_input = horizon, list(levels), drop_span_input
        self.crop_front = crop_front
        self.crop_target = crop_front if crop_target is None else crop_target
        spherical = self.bench.grid is not None
        if spherical and (self.crop_front or self.crop_target):
            raise ValueError("crop_front is for Cartesian grids; a spherical grid has its own coarse back")
        self.eval_targets = eval_targets and spherical
        self.to_eval = self.bench.to_cartesian if self.eval_targets else None
        if (self.crop_front or self.crop_target) and self.bench.vertical_axis != 1:
            raise ValueError("crop_front needs grids in the anchor CPF frame (forward = last axis); this benchmark "
                             f"has vertical_axis={self.bench.vertical_axis} (world-aligned grids have no front)")
        self.gaze_paths = gaze_paths
        self.video = video
        self.video_dim = self.bench.video()[0].shape[1] if video else 0
        self.cue_dim = self.bench.gaze()[1]["n_features"] if gaze_paths else 0
        fixed = (2,) if self.crop_front else ()
        if not augment:
            self.aug = None
        elif spherical:
            self.aug = MirrorAugment(seed=seed)
        else:
            self.aug = SpanAugment(self.bench.vertical_axis, max_shift, seed=seed, fixed_axes=fixed)

    def _target(self, j):
        if self.eval_targets:
            return self.bench.eval_target(j, self.horizon, self.levels)
        return self.bench.target(j, self.horizon, self.levels)

    def _crop(self, a, on):
        return a[..., self.bench.R // 2:] if on else a

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.bench.inputs(j)
        y = self._target(j)
        if self.drop_span_input:                       # "w/o previous span": keep only scene occupancy
            x[:, :N_LEVELS] = False                    # (and drop the gaze ray: it is gaze information too)
            x[:, N_CHANNELS:] = False
        x, y = self._crop(x, self.crop_front), self._crop(y, self.crop_target)
        g = None
        if self.gaze_paths:
            path, layout = self.bench.gaze()
            g = path[j].astype(np.float32)
        v = None
        if self.video:                                 # lifted camera features + in-view mask, cropped like x
            from ..video import lift
            maps, index = self.bench.video()
            v = self._crop(lift(maps[j], index, self.bench.shape), self.crop_front)
        if self.aug is not None:                       # (val / test, where y may be Cartesian, are never augmented)
            perm, flips, shift = self.aug.sample()
            x, y = self.aug.apply(x, perm, flips, shift), self.aug.apply(y, perm, flips, shift)
            if v is not None:
                v = self.aug.apply(v, perm, flips, shift)
            if g is not None:                          # directions turn with the grid; translations don't affect them
                g = transform_vectors(g, perm, flips, layout["vectors"], layout["pseudovectors"])
        item = {"x": torch.from_numpy(x.astype(np.float32)), "y": torch.from_numpy(y.astype(np.float32)),
                "idx": int(j)}
        if g is not None:
            item["g"] = torch.from_numpy(g)
        if v is not None:
            item["v"] = torch.from_numpy(v.astype(np.float16))
        return item

    def all_targets(self):
        return self._crop(self._target(self.idx), self.crop_target)

    def all_inputs(self):
        x = self.bench.inputs(self.idx)
        if self.drop_span_input:
            x[:, :, :N_LEVELS] = False
            x[:, :, N_CHANNELS:] = False
        return self._crop(x, self.crop_front)


def worker_init_fn(worker_id):
    """Give each DataLoader worker its own augmentation stream."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if getattr(ds, "aug", None) is not None:
        ds.aug.rng = np.random.default_rng(info.seed % 2 ** 32)
