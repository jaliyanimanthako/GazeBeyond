"""Benchmark curation (Sec. 5.1, 5.2, Appendix A.1): RecordingStreams -> forecasting samples -> packed splits.

Sample (FoVS-Aria defaults): 2 s observed + 2 s forecast, sliding by 1 s, on the 10 Hz gaze timeline, grouped in 1 s
units. Every grid of a sample is in the anchor frame of the last observed step.

    X   (Tp, 5, R, R, R)  per observed unit, the union of its steps' [orientation, peripheral, central, foveal, scene]
                          (or per temporal bin, see below)
    YF  (Tf, 4, R, R, R)  per future unit, the union of its steps' four span levels
                          the paper's target is the union over all Tf units (or the first h units for Fig. 4-b)

Grids are stored bit-packed (np.packbits) per recording, then concatenated by `pack` into all.npz + splits.npz.

Temporal bins (our extension; the paper uses one binary union per 1 s unit). window.time_bins = B splits the observed
P = past_units * steps_per_unit steps into B bins instead of past_units:
    hard (soft=false)   B contiguous groups of P / B steps, binary union per group (B = past_units: the paper's input)
    soft (soft=true)    every step is first lifted to a binary per-step mask M_k (so SLAM point density does not leak
                        into the values), then shared between its two nearest bin centres by linear interpolation
                        (weights w_kb >= 0, sum_b w_kb = 1; steps beyond the first / last centre go fully to that bin):
                            X_b = sum_k w_kb M_k / sum_k w_kb   in [0, 1]
                        = the fraction of the bin's weighted time during which the voxel was active (dwell), with the
                        order of the bins carrying recency / direction. Stored as uint8 (value * 255).
The target is unchanged (binary union over the forecast window), so scores stay comparable with the paper.

Gaze-ray channel (window.ray_channel, our extension): a 6th input channel with, per step, the voxels along the gaze ray,
from the glasses' position at that step (in the anchor frame, so walking is accounted for) to the edge of the grid.
It does not need SLAM points, so it shows where the eyes pointed even at bare walls; it is gaze-direction evidence,
not foveal occupancy (it carries no depth). It is binned with the same hard / soft weights as the spans.

Spherical grids (lift.grid = "spherical", our extension, egospan/spherical.py): X and YF use the direction-adaptive
azimuth x elevation x distance cells instead of R^3 voxels; YF_cart additionally keeps the ordinary R^3 targets from
the same points, so predictions mapped back to the Cartesian grid are scored exactly like the paper's.

Choices the paper leaves open are parameters here (defaults in brackets):
    which points count as "keypoints of interest" in the outlier-sample filter      [orientation-span points in the
                                                                                     forecast window]
    validation split                                                                [10 % of recordings per location]
"""
import glob
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from ..lift import (N_CHANNELS, N_LEVELS, EgoSpanLift, LiftConfig, gravity_alignment, rasterize_ray,
                    statistical_outlier_mask)
from ..spherical import spherical_grid
from .streams import dir_to_yaw_pitch


@dataclass
class WindowConfig:
    steps_per_unit: int = 10          # 10 Hz gaze, 1 s units
    past_units: int = 2               # Tp
    future_units: int = 2             # Tf (FoVS-EgoExo: 4)
    stride_units: int = 1             # 1 s sliding window
    max_gap_factor: float = 1.5       # reject windows with a timestamp gap above this many nominal steps
    time_bins: int | None = None      # input bins over the observed window (None = past_units, the paper's 1 s units)
    soft: bool = False                # soft temporal bins (linear interpolation, dwell fractions, uint8 storage)
    ray_channel: bool = False         # add the gaze-ray channel (6 input channels instead of 5)
    anchor: str = "window"            # "window": every stride_units, target = union of the forecast window (the paper)
                                      # "shift": one sample per attention shift (egospan/shifts.py), anchored
                                      # shift_lag_steps before the gaze leaves the fixation, target = the cones of the
                                      # landing fixation (the lab's WHERE question in our grids)
    shift_lag_steps: int = 3          # 0.3 s at 10 Hz, as the lab's `lag` queries


def input_channels(win_cfg):
    return N_CHANNELS + bool(win_cfg.get("ray_channel") if isinstance(win_cfg, dict) else win_cfg.ray_channel)


def input_bins(win_cfg):
    return win_cfg.time_bins or win_cfg.past_units


def time_bin_weights(n_steps, n_bins, soft):
    """(n_steps, n_bins) weights of every observed step in every input bin; each row sums to 1.
    hard: step k belongs to bin k // (n_steps / n_bins). soft: linear interpolation between bin centres."""
    W = np.zeros((n_steps, n_bins))
    if not soft:
        if n_steps % n_bins:
            raise ValueError(f"hard time bins need steps ({n_steps}) divisible by bins ({n_bins})")
        W[np.arange(n_steps), np.arange(n_steps) // (n_steps // n_bins)] = 1.0
        return W
    width = n_steps / n_bins
    centres = (np.arange(n_bins) + 0.5) * width - 0.5                      # in step-index units
    u = np.clip((np.arange(n_steps) - centres[0]) / width, 0, n_bins - 1)  # position between centres
    lo = np.floor(u).astype(int)
    frac = u - lo
    W[np.arange(n_steps), lo] = 1 - frac
    hi = np.minimum(lo + 1, n_bins - 1)
    W[np.arange(n_steps), hi] += np.where(lo + 1 < n_bins, frac, 0.0)
    return W


def encode_inputs(X, soft):
    """per-sample storage of the input grids: bit-packed if binary, uint8 (value * 255) if soft."""
    return np.round(X.reshape(-1) * 255).astype(np.uint8) if soft else np.packbits(X.reshape(-1))


def decode_inputs(stored, shape, soft):
    """inverse of encode_inputs for one (F,) or many (n, F) samples -> bool or float32 grids of `shape`."""
    if soft:
        return stored.reshape(*stored.shape[:-1], *shape).astype(np.float32) / 255
    return unpack_grids(stored, shape)


def unpack_grids(packed, shape):
    n = int(np.prod(shape))
    return np.unpackbits(packed, axis=-1, count=n).reshape(*packed.shape[:-1], *shape).astype(bool)


# ------------------------------------------------------------------------------------------------ extraction
class StepPoints:
    """Eq. 1 + Eq. 2 per step of one recording: indices (into streams.points_world) of the selected visible points
    and their (5, n) span labels. They depend only on the step, not on the anchor, so they are cached."""

    def __init__(self, streams, lift_cfg, lifter):
        self.s, self.lifter, self.cache = streams, lifter, {}
        self.per_step_sor = lift_cfg.sor_scope == "step"
        ds = streams.points_dist_std
        if not self.per_step_sor:           # one outlier pass over the whole (confidence-filtered) cloud
            good = np.ones(len(streams.points_world), bool)
            if lift_cfg.max_dist_std is not None and ds is not None:
                good &= ds < lift_cfg.max_dist_std
            gi = np.flatnonzero(good)
            good[gi[~statistical_outlier_mask(streams.points_world[gi], lift_cfg.sor_neighbors,
                                              lift_cfg.sor_std_ratio)]] = False
            self.good = good

    def __call__(self, k):
        if k not in self.cache:
            s = self.s
            vis = s.visible(k)
            if not self.per_step_sor:
                vis = vis[self.good[vis]]
            ds = s.points_dist_std
            keep = self.lifter.select(s.points_world[vis], s.T_world_cpf[k], None if ds is None else ds[vis],
                                      sor=self.per_step_sor)
            vis = vis[keep]
            self.cache[k] = (vis, self.lifter.classify(s.points_world[vis], s.T_world_cpf[k], s.gaze_cpf[k]))
        return self.cache[k]

    def drop_before(self, k):
        for j in [j for j in self.cache if j < k]:
            del self.cache[j]


def extract_samples(streams, lift_cfg=None, win_cfg=None):
    """All forecasting samples of one recording -> dict of arrays (grids bit-packed), or None if there are none."""
    lift_cfg, win_cfg = lift_cfg or LiftConfig(), win_cfg or WindowConfig()
    R_align = gravity_alignment(streams.gravity_world) if lift_cfg.grid_frame == "world" else np.eye(3)
    lifter = EgoSpanLift(lift_cfg, R_align)
    sph = spherical_grid(lift_cfg)
    if sph is not None and lift_cfg.grid_frame != "local":
        raise ValueError("spherical grids are defined around the wearer's head: use grid_frame = local")
    spu = win_cfg.steps_per_unit
    P, F = win_cfg.past_units * spu, win_cfg.future_units * spu
    t, T_wc, gaze = streams.t_us, streams.T_world_cpf, streams.gaze_cpf
    dt = np.median(np.diff(t)) if len(t) > 1 else 1e5

    if win_cfg.anchor == "shift":
        return extract_shift_samples(streams, lift_cfg, win_cfg)
    if win_cfg.anchor != "window":
        raise ValueError(f"window.anchor {win_cfg.anchor!r}: window | shift")
    G = _Grids(streams, lift_cfg, win_cfg, lifter, sph)
    keys = ["X", "YF", "t0_us", "T_world_grid", "poses_grid", "gaze_grid", "gaze_cpf", "point_counts"]
    out = {k: [] for k in keys + (["YF_cart"] if sph is not None else [])}
    for t0 in range(P - 1, len(t) - F, win_cfg.stride_units * spu):
        win = np.arange(t0 - P + 1, t0 + F + 1)
        if not streams.valid[win].all() or np.any(np.diff(t[win]) > win_cfg.max_gap_factor * dt):
            continue
        frame = lifter.anchor(T_wc[t0])
        counts = np.zeros((2, N_CHANNELS), np.int64)
        X, counts[0] = G.past(frame, win[:P])
        YF, YFc, counts[1] = G.target(frame, [win[P + u * spu:P + (u + 1) * spu] for u in range(win_cfg.future_units)])
        poses = frame.pose(T_wc[win])
        out["X"].append(encode_inputs(X, win_cfg.soft))
        out["YF"].append(np.packbits(YF.reshape(-1)))
        if sph is not None:
            out["YF_cart"].append(np.packbits(YFc.reshape(-1)))
        out["t0_us"].append(t[t0])
        out["T_world_grid"].append(frame.matrix())
        out["poses_grid"].append(poses.astype(np.float32))
        out["gaze_grid"].append(np.einsum("sij,sj->si", poses[:, :3, :3], gaze[win]).astype(np.float32))
        out["gaze_cpf"].append(gaze[win].astype(np.float32))
        out["point_counts"].append(counts)
        G.steps.drop_before(t0 - P + 1 + win_cfg.stride_units * spu)
    if not out["X"]:
        return None
    arr = {k: np.stack(v) for k, v in out.items()}
    arr["meta"] = np.array(json.dumps(dict(
        name=streams.name, lift=asdict(lift_cfg), window=asdict(win_cfg),
        vertical_axis=lifter.anchor(T_wc[0]).vertical_axis, recording=_jsonable(streams.meta))))
    return arr


class _Grids:
    """lifts a sample's observed steps into the input grids and a set of steps into target grids (one anchor frame)."""

    def __init__(self, streams, lift_cfg, win_cfg, lifter, sph):
        self.s, self.lc, self.wc, self.lifter, self.sph = streams, lift_cfg, win_cfg, lifter, sph
        self.steps = StepPoints(streams, lift_cfg, lifter)
        self.R = lift_cfg.resolution
        self.shape = sph.shape if sph is not None else (self.R,) * 3
        P = win_cfg.past_units * win_cfg.steps_per_unit
        self.nb, self.soft, self.n_ch = input_bins(win_cfg), win_cfg.soft, input_channels(win_cfg)
        self.W = time_bin_weights(P, self.nb, self.soft)                    # (P, nb)

    def _mask(self, frame, p, lab):
        return self.sph.voxelize(frame.points(p), lab) if self.sph is not None else self.lifter.voxelize(p, lab, frame)

    def past(self, frame, steps):
        """X (nb, n_ch, *shape) over the observed steps (oldest first) and the point counts per channel."""
        s, W = self.s, self.W
        X = np.zeros((self.nb, self.n_ch, *self.shape), np.float32 if self.soft else bool)
        counts = np.zeros(N_CHANNELS, np.int64)
        for rel, k in enumerate(steps):
            vis, lab = self.steps(k)
            M = self._mask(frame, s.points_world[vis], lab)                 # binary per-step mask (5, *shape)
            if self.wc.ray_channel:                                         # ray from this step's glasses position
                T = s.T_world_cpf[k]
                o, d = frame.points(T[:3, 3]), frame.dirs(T[:3, :3] @ s.gaze_cpf[k])
                ray = (self.sph.rasterize_ray(o, d) if self.sph is not None
                       else rasterize_ray(o, d, self.lc.cube_size, self.R))
                M = np.concatenate([M, ray[None]])
            if self.soft:
                for b in np.flatnonzero(W[rel]):
                    X[b] += W[rel, b] * M
            else:
                X[int(np.argmax(W[rel]))] |= M
            counts += lab.sum(1)
        if self.soft:
            X /= W.sum(0)[:, None, None, None, None]
        return X, counts

    def target(self, frame, units):
        """YF (len(units), 4, *shape): per unit (a list of steps) the union of their span levels; YF_cart the same on
        R^3 voxels for spherical grids (else None); the point counts per channel."""
        s = self.s
        YF = np.zeros((len(units), N_LEVELS, *self.shape), bool)
        YFc = np.zeros((len(units), N_LEVELS, self.R, self.R, self.R), bool) if self.sph is not None else None
        counts = np.zeros(N_CHANNELS, np.int64)
        for u, steps in enumerate(units):
            for k in steps:
                vis, lab = self.steps(k)
                p = s.points_world[vis]
                if self.sph is not None:
                    self.sph.voxelize(frame.points(p), lab[:N_LEVELS], out=YF[u])
                    self.lifter.voxelize(p, lab[:N_LEVELS], frame, out=YFc[u])
                else:
                    self.lifter.voxelize(p, lab[:N_LEVELS], frame, out=YF[u])
                counts += lab.sum(1)
        return YF, YFc, counts


def extract_shift_samples(streams, lift_cfg, win_cfg, shift_cfg=None):
    """One sample per attention shift (egospan/shifts.py: 75 deg/s fixations held >= 150 ms, >= 30 deg between
    consecutive held fixations, <= 1 s apart), as the lab's WHERE questions (attention_forecast.py: the shift's step
    is the last sample at or before t_leave = A's last sample + 1 / fs; the query is shift_lag_steps before it):
        X        the usual input grids over the P observed steps ending at the query, in the query's anchor frame
        YF       (1, 4, *shape) the span levels of the LANDING fixation B (the union over its samples)
        land_dir the mean world gaze direction of B, in the anchor frame (the WHERE answer)
        gaze_now the mean world gaze direction of A, in the anchor frame (the "stay" prediction)
    plus shift_deg (angle A -> B), land_t0_us / land_t1_us (B's first / last sample) and the usual pose / gaze
    arrays over the observed steps. Shifts without P valid, gap-free steps before the query are skipped."""
    from ..shifts import ShiftConfig, attention_shifts, detect_fixations, unit
    shift_cfg = shift_cfg or ShiftConfig()
    R_align = gravity_alignment(streams.gravity_world) if lift_cfg.grid_frame == "world" else np.eye(3)
    lifter = EgoSpanLift(lift_cfg, R_align)
    sph = spherical_grid(lift_cfg)
    if sph is not None and lift_cfg.grid_frame != "local":
        raise ValueError("spherical grids are defined around the wearer's head: use grid_frame = local")
    P = win_cfg.past_units * win_cfg.steps_per_unit
    t, T_wc, gaze = streams.t_us, streams.T_world_cpf, streams.gaze_cpf
    dt = np.median(np.diff(t)) if len(t) > 1 else 1e5
    w = unit(np.einsum("nij,nj->ni", T_wc[:, :3, :3], gaze))
    fix = detect_fixations(w, t, streams.valid, shift_cfg)
    sh = attention_shifts(fix, t, shift_cfg)
    G = _Grids(streams, lift_cfg, win_cfg, lifter, sph)
    keys = ["X", "YF", "t0_us", "T_world_grid", "poses_grid", "gaze_grid", "gaze_cpf", "point_counts", "land_dir",
            "gaze_now", "shift_deg", "land_t0_us", "land_t1_us"]
    out = {k: [] for k in keys + (["YF_cart"] if sph is not None else [])}
    unit_us = int(round(1e6 / win_cfg.steps_per_unit))                     # 0.1 s at 10 Hz
    for a, b, ang in zip(sh["src"], sh["dst"], sh["angle_deg"]):
        # the lab's rule (attention_shift_labels / attention_forecast): the gaze leaves A at t_leave = (A's last
        # sample) + 1 / fs, the shift's step is the last sample at or before t_leave (the next sample, or A's last one
        # when the next comes a few us later than 1 / fs), and the question is shift_lag_steps before that step
        i = int(np.searchsorted(t, t[fix["end"][a]] + unit_us, side="right")) - 1
        q = i - win_cfg.shift_lag_steps
        past = np.arange(q - P + 1, q + 1)
        if past[0] < 0 or not streams.valid[past].all() or np.any(np.diff(t[past]) > win_cfg.max_gap_factor * dt):
            continue
        frame = lifter.anchor(T_wc[q])
        counts = np.zeros((2, N_CHANNELS), np.int64)
        X, counts[0] = G.past(frame, past)
        land = np.arange(fix["start"][b], fix["end"][b] + 1)
        YF, YFc, counts[1] = G.target(frame, [land])
        poses = frame.pose(T_wc[past])
        out["X"].append(encode_inputs(X, win_cfg.soft))
        out["YF"].append(np.packbits(YF.reshape(-1)))
        if sph is not None:
            out["YF_cart"].append(np.packbits(YFc.reshape(-1)))
        out["t0_us"].append(t[q])
        out["T_world_grid"].append(frame.matrix())
        out["poses_grid"].append(poses.astype(np.float32))
        out["gaze_grid"].append(np.einsum("sij,sj->si", poses[:, :3, :3], gaze[past]).astype(np.float32))
        out["gaze_cpf"].append(gaze[past].astype(np.float32))
        out["point_counts"].append(counts)
        out["land_dir"].append(unit(frame.dirs(fix["direction"][b])).astype(np.float32))
        out["gaze_now"].append(unit(frame.dirs(fix["direction"][a])).astype(np.float32))
        out["shift_deg"].append(np.float32(ang))
        out["land_t0_us"].append(t[fix["start"][b]])
        out["land_t1_us"].append(t[fix["end"][b]])
    if not out["X"]:
        return None
    arr = {k: np.stack(v) for k, v in out.items()}
    stored_win = dict(asdict(win_cfg), future_units=1)                      # the target is one unit: the landing
    arr["meta"] = np.array(json.dumps(dict(
        name=streams.name, lift=asdict(lift_cfg), window=stored_win, shift=asdict(shift_cfg),
        vertical_axis=lifter.anchor(T_wc[0]).vertical_axis, recording=_jsonable(streams.meta))))
    return arr


def future_points(streams, lift_cfg, win_cfg, t0_us):
    """For each anchor time in t0_us: the points behind the forecast target, for scoring any grid on the same points.

    -> list of (xyz (m, 3) float32 in the sample's grid frame, labels (m,) uint8 with bit c = span level c, past_1s,
    past_2s), one per anchor: every point selected (Eq. 1) at any step of the forecast window and inside the paper's
    cube, its bits the OR over those steps of its span labels (Eq. 2). Voxelising them gives exactly the R^3 target,
    binning them exactly the spherical one. past_1s / past_2s: the same bits over the last observed second / the whole
    observed window, plus bit 4 = the point was selected at all then (already in the map at prediction time);
    seen_ever (m,) bool: the point was observed at any earlier step of the recording (the whole SLAM history up to the
    anchor, not just the input window). Also returns the largest future gaze shift (deg) per anchor."""
    R_align = gravity_alignment(streams.gravity_world) if lift_cfg.grid_frame == "world" else np.eye(3)
    lifter = EgoSpanLift(lift_cfg, R_align)
    step_points = StepPoints(streams, lift_cfg, lifter)
    P, F = win_cfg.past_units * win_cfg.steps_per_unit, win_cfg.future_units * win_cfg.steps_per_unit
    t, T_wc, gaze = streams.t_us, streams.T_world_cpf, streams.gaze_cpf
    D = lift_cfg.cube_size
    first = np.full(len(streams.points_world), np.iinfo(np.int64).max)      # first step each point was observed at
    for k in range(len(t)):
        v = streams.visible(k)
        first[v] = np.minimum(first[v], k)
    out, shifts = [], []
    for t0 in sorted(np.asarray(t0_us), key=int):
        k0 = int(np.searchsorted(t, t0))
        if k0 >= len(t) or t[k0] != t0:
            raise ValueError(f"{streams.name}: no gaze step at anchor time {t0}")
        frame = lifter.anchor(T_wc[k0])

        def union(steps, seen_bit=0):
            ids, labs = [], []
            for k in steps:
                vis, lab = step_points(k)
                ids.append(vis)
                labs.append(np.packbits(lab[:N_LEVELS].T, axis=1, bitorder="little")[:, 0] | seen_bit)
            ids, labs = np.concatenate(ids), np.concatenate(labs)
            uid, inv = np.unique(ids, return_inverse=True)
            bits = np.zeros(len(uid), np.uint8)
            np.bitwise_or.at(bits, inv, labs)
            return uid, bits

        def at(uid, past):                                         # past bits of the points uid (0 if not seen)
            pid, pbits = past
            j = np.clip(np.searchsorted(pid, uid), 0, max(len(pid) - 1, 0))
            hit = (pid[j] == uid) if len(pid) else np.zeros(len(uid), bool)
            return np.where(hit, pbits[j] if len(pid) else 0, 0).astype(np.uint8)

        uid, bits = union(range(k0 + 1, k0 + F + 1))
        spu = win_cfg.steps_per_unit
        past1 = at(uid, union(range(k0 - spu + 1, k0 + 1), 16))
        past2 = at(uid, union(range(k0 - P + 1, k0 + 1), 16))
        xyz = frame.points(streams.points_world[uid]).astype(np.float32)
        inside = ((xyz >= -D / 2) & (xyz < D / 2)).all(1)          # the voxels' rule: floor((p + D/2) R / D) in [0, R)
        ever = first[uid] <= k0
        out.append((xyz[inside], bits[inside], past1[inside], past2[inside], ever[inside]))
        g = frame.dirs(np.einsum("sij,sj->si", T_wc[k0:k0 + F + 1, :3, :3], gaze[k0:k0 + F + 1]))
        g = g / np.linalg.norm(g, axis=1, keepdims=True)
        shifts.append(float(np.degrees(np.arccos(np.clip(g[1:] @ g[0], -1, 1))).max()))
        step_points.drop_before(k0 + 1 - P)
    order = np.argsort(np.argsort(np.asarray(t0_us).astype(np.int64)))   # back to the order asked for
    return [out[i] for i in order], np.asarray(shifts)[order]


def sample_keys(rec):
    """"<recording>#<position of the sample within its recording>" for every row: the key that matches samples between
    benchmarks built from the same recordings (the lab's files, our rebuilds, Cartesian or spherical)."""
    seen, keys = {}, []
    for r in np.asarray(rec).astype(str):
        p = seen.get(r, 0)
        seen[r] = p + 1
        keys.append(f"{r}#{p}")
    return np.array(keys)


def _jsonable(d):
    return json.loads(json.dumps(d, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def save_recording(arr, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **arr)


# ------------------------------------------------------------------------------------------------ packing
@dataclass
class PackConfig:
    protocol: str = "aea"                    # aea: test = test_locations | split_file: json {recording: split} | random
    test_locations: tuple = (4,)
    val_frac: float = 0.1                    # of the non-test recordings (per location for aea)
    seed: int = 0
    split_file: str | None = None
    exclude_file: str | None = None          # text file, one recording name per line (manual curation)
    min_points_sigma: float | None = 2.0     # drop samples whose keypoints of interest < mean - sigma * std
    points_channel: str = "orientation"      # which span's point count in the forecast window is "of interest"
    min_gaze_std_deg: float = 0.0            # drop recordings whose gaze yaw and pitch std are both below this
    cv_folds_by_location: bool = True        # also write location-wise CV folds (aea)


def pack(recs_dir, out_dir, cfg=None):
    """Concatenate per-recording files into out_dir/all.npz and write out_dir/splits.npz (index arrays)."""
    from ..lift import CHANNELS
    cfg = cfg or PackConfig()
    files = sorted(glob.glob(os.path.join(recs_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no recordings in {recs_dir}")
    excluded = set()
    if cfg.exclude_file:
        excluded = {ln.strip() for ln in open(cfg.exclude_file) if ln.strip() and not ln.startswith("#")}
    Z, metas = [], []
    for f in files:
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        if m["name"] in excluded:
            continue
        if cfg.min_gaze_std_deg > 0:
            yaw, pitch = dir_to_yaw_pitch(z["gaze_cpf"].reshape(-1, 3))
            if max(np.degrees(yaw).std(), np.degrees(pitch).std()) < cfg.min_gaze_std_deg:
                print(f"drop {m['name']}: gaze std below {cfg.min_gaze_std_deg} deg (miscalibration?)")
                continue
        Z.append({k: z[k] for k in z.files if k != "meta"})
        metas.append(m)
    keys = [k for k in Z[0]]
    allz = {k: np.concatenate([z[k] for z in Z]) for k in keys}
    rec = np.concatenate([[m["name"]] * len(z["X"]) for m, z in zip(metas, Z)])
    rmeta = {m["name"]: m["recording"] for m in metas}
    location = np.array([int(rmeta[r].get("location", -1)) for r in rec])
    category = np.array([str(rmeta[r].get("category", "")) for r in rec])

    keep = np.ones(len(rec), bool)
    if cfg.min_points_sigma is not None:
        c = allz["point_counts"][:, 1, CHANNELS.index(cfg.points_channel)].astype(float)
        thr = c.mean() - cfg.min_points_sigma * c.std()
        keep = c >= thr
        print(f"sample filter: {cfg.points_channel} points in forecast window >= {thr:.0f}: "
              f"drop {(~keep).sum()} of {len(keep)}")
    allz = {k: v[keep] for k, v in allz.items()}
    rec, location, category = rec[keep], location[keep], category[keep]

    split = assign_splits(rec, location, cfg)
    common = dict(metas[0])
    common.pop("name"), common.pop("recording")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "all.npz"), **allz, rec=rec, location=location, category=category, split=split,
             meta=np.array(json.dumps(common)))
    idx = {p: np.flatnonzero(split == p) for p in ("train", "val", "test")}
    if cfg.protocol == "aea" and cfg.cv_folds_by_location:
        idx.update(location_folds(rec, location, cfg))
    np.savez(os.path.join(out_dir, "splits.npz"), **idx)
    for p in ("train", "val", "test"):
        print(f"{p}: {len(idx[p])} samples from {len(set(rec[idx[p]]))} recordings")
    return idx


def assign_splits(rec, location, cfg):
    rng = np.random.default_rng(cfg.seed)
    names = sorted(set(rec))
    split = {}
    if cfg.protocol == "split_file":
        given = json.load(open(cfg.split_file))
        split.update({r: given[r] for r in names if r in given})
    elif cfg.protocol == "aea":
        loc_of = dict(zip(rec, location.tolist()))
        split.update({r: "test" for r in names if loc_of[r] in cfg.test_locations})
    loc_of = dict(zip(rec, location.tolist()))
    groups = {}
    for r in names:                                   # stratify the validation split by location (aea)
        if r not in split:
            groups.setdefault(loc_of[r] if cfg.protocol == "aea" else 0, []).append(r)
    for _, rs in sorted(groups.items()):
        rs = list(rng.permutation(rs))
        nval = max(1, round(cfg.val_frac * len(rs))) if len(rs) > 1 else 0
        split.update({r: "val" for r in rs[:nval]})
        split.update({r: "train" for r in rs[nval:]})
    return np.array([split[r] for r in rec])


def location_folds(rec, location, cfg):
    """Location-wise CV (SensorySpotlight): fold L tests on location L, validates on val_frac of each other location's
    recordings, trains on the rest."""
    rng = np.random.default_rng(cfg.seed)
    out = {}
    locs = sorted(set(location.tolist()) - {-1})
    for L in locs:
        val_recs = set()
        for l2 in locs:
            if l2 != L:
                rs = rng.permutation(sorted(set(rec[location == l2])))
                val_recs |= set(rs[:max(1, round(cfg.val_frac * len(rs)))])
        is_test = location == L
        is_val = np.isin(rec, list(val_recs)) & ~is_test
        out[f"fold{L}_test"] = np.flatnonzero(is_test)
        out[f"fold{L}_val"] = np.flatnonzero(is_val)
        out[f"fold{L}_train"] = np.flatnonzero(~is_test & ~is_val)
    return out
