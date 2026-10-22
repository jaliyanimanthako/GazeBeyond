"""3D scene features for the lab's WHERE model, from a spherical attention-shift benchmark: per WHERE question and per
candidate direction (468 = 36 azimuth x 13 elevation, 10 deg, in the head frame at the question), what our spherical
cells (azimuth x elevation x distance, anchor CPF frame) hold in that direction over the 2 s before the question.

    S=~/SensorySpotlight/data/traj_forecast/attention_shift/senses_suite
    python scripts/export_where_geometry.py --bench data/fovs_aria_shift_sph --suite $S          # -> $S/geo3d_aea_lag.npy
    python scripts/export_where_geometry.py ... --grid sph   # WHERE on our spherical cells (WS_GRID=sph) -> geo3d_aea_lag_sph.npy
    (then in SensorySpotlight: WS_VARIANTS=mem,3d,mem+3d ... tools/attention_senses_suite.py train aea lag location ...)

Our shift samples are anchored where the lab asks its questions (0.3 s before the shift), so each question is matched
to the sample of the same recording within --tol s. Every channel is computed only from what was observed up to the
question (EgoSpanLift's past window: no future SLAM points). Features per candidate (all in [0, 1]):
    0:10   for each input second (oldest first) and channel (orientation, peripheral, central, foveal, scene): the
           fraction of the candidate's sky patch (the spherical columns within --patch deg, or the column containing it)
           occupied at any distance
    10     nearness of the closest surface seen in the patch over the 2 s: 1 - d / d_max (0 = nothing seen)
    11     has: 1 if the question has a matched sample (else every feature is 0)
--grid sph: the candidates are our spherical columns themselves (40 x 24 = 960, the order of SensorySpotlight's
WS_GRID=sph: elevation-major), each its own patch (--patch 0 by default).
The frames are checked first: the lab's landing direction (truth, az / el) must agree with ours (land_dir) for the
matched questions; the best of the 48 axis flips / swaps is used and reported (identity is expected).
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attach_shift_video import match                # noqa: E402
from egospan.data.dataset import PackedBenchmark    # noqa: E402
from egospan.lift import N_CHANNELS                 # noqa: E402

CAZ, CEL = (a.ravel() for a in np.meshgrid(np.arange(-180, 180, 10.0), np.arange(-60, 61, 10.0)))   # the lab's 468
N_FEAT = 2 * N_CHANNELS + 2


def from_azel(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)], -1)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def angle(a, b):
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


def signed_permutations():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            M[range(3), perm] = signs
            yield M


def frame_map(lab_dirs, our_dirs):
    """the signed permutation M (ours = M @ lab) with the smallest median angle, and that angle."""
    best = min(((np.median(angle(lab_dirs @ M.T, our_dirs)), i, M) for i, M in enumerate(signed_permutations())),
               key=lambda x: x[0])
    return best[2], best[0]


def patches(grid, cand, patch_deg):
    """(n_cand, n_cols) bool: spherical columns (azimuth x elevation) in each candidate's patch."""
    centers = unit(grid.centers()[:, :, 0].reshape(-1, 3))                       # column centre directions
    A, B, _ = grid.shape
    M = (cand @ centers.T) > np.cos(np.radians(patch_deg))
    own = grid.index(cand)[0]                                                    # (n_cand, 3) cell of the direction
    M[np.arange(len(cand)), own[:, 0] * B + own[:, 1]] = True
    return M


def features(X, M, depth_mid):
    """X (n, Tp, C, A, B, D) bool -> (n, n_cand, N_FEAT - 1) float32 (without the `has` flag)."""
    n, Tp, C, A, B, D = X.shape
    col = X.any(-1).reshape(n, Tp * C, A * B).astype(np.float32)                # occupied at any distance
    frac = np.einsum("nkc,mc->nmk", col, M.astype(np.float32)) / M.sum(1)[None, :, None]
    scene = X[:, :, N_CHANNELS - 1].any(1).reshape(n, A * B, D)                  # surfaces seen over the 2 s
    near = np.where(scene, depth_mid[None, None], np.inf).min(-1)               # (n, cols) closest surface
    d = np.where(M[None], near[:, None], np.inf).min(-1)                         # (n, cand)
    nearness = np.where(np.isfinite(d), 1 - d / depth_mid[-1], 0.0)
    return np.concatenate([frac, nearness[..., None]], -1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="a spherical attention-shift benchmark (fovs_aria_shift_sph)")
    ap.add_argument("--suite", required=True, help="folder with the lab's data_<tag>.npz; the output goes there")
    ap.add_argument("--tag", default="aea_lag")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--grid", choices=("lab", "sph"), default="lab", help="WHERE's candidates: the lab's 468 or our cells")
    ap.add_argument("--patch", type=float, default=None, help="half-width of a candidate's sky patch (deg; lab 7.5, sph 0)")
    ap.add_argument("--max-frame-err", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    b = PackedBenchmark(a.bench)
    if b.grid is None or "land_dir" not in b.arrays:
        raise SystemExit(f"{a.bench} must be a spherical attention-shift benchmark (grid spherical, anchor shift)")
    if b.Tp != 2:
        raise SystemExit(f"expected 2 input seconds, the benchmark has {b.Tp} time bins")
    Q = np.load(os.path.join(a.suite, f"data_{a.tag}.npz"))
    q_rec, q_t = Q["rec"].astype(str), Q["t"].astype(np.float64)
    # question -> sample (the reverse of attach_shift_video's matching)
    j = match(q_rec, q_t, b.arrays["rec"].astype(str), b.arrays["t0_us"] / 1e6, a.tol)
    ok = j >= 0
    if ok.sum() < 20:
        raise SystemExit(f"only {ok.sum()} of {len(q_rec)} questions match a sample: wrong benchmark or anchor?")
    lab_truth = from_azel(*Q["truth"][ok].T)
    M, err = frame_map(lab_truth, b.arrays["land_dir"][j[ok]].astype(np.float64))
    print(f"matched {ok.sum()} of {len(q_rec)} questions ({ok.mean():.1%}); landing direction lab vs ours: "
          f"median {err:.2f} deg with axes {'unchanged' if np.allclose(M, np.eye(3)) else M.tolist()}")
    if "g0" in Q.files:
        print(f"  current gaze lab vs ours: median {np.median(angle(from_azel(*Q['g0'][ok].T) @ M.T, b.arrays['gaze_now'][j[ok]])):.2f} deg")
    if err > a.max_frame_err:
        raise SystemExit(f"the frames disagree ({err:.1f} deg > {a.max_frame_err}): not writing features")
    caz, cel = CAZ, CEL
    if a.grid == "sph":
        ye, pe = np.asarray(b.grid.yaw_edges, float), np.asarray(b.grid.pitch_edges, float)
        caz, cel = (x.ravel() for x in np.meshgrid((ye[:-1] + ye[1:]) / 2, (pe[:-1] + pe[1:]) / 2))
    patch = a.patch if a.patch is not None else (7.5 if a.grid == "lab" else 0.0)
    cand = from_azel(caz, cel) @ M.T                                             # candidates in our anchor frame
    P = patches(b.grid, cand, patch)
    edges = np.asarray(b.grid.depth_edges, float)
    depth_mid = (edges[:-1] + edges[1:]) / 2
    out = np.zeros((len(q_rec), len(cand), N_FEAT), np.float16)
    rows = np.flatnonzero(ok)
    for s in range(0, len(rows), 256):
        r = rows[s:s + 256]
        X = b.inputs(j[r])[:, :, :N_CHANNELS]                                    # (n, Tp, 5, A, B, D)
        out[r, :, :-1] = features(np.asarray(X) > 0.5, P, depth_mid)
        out[r, :, -1] = 1
    path = a.out or os.path.join(a.suite, f"geo3d_{a.tag}{'' if a.grid == 'lab' else '_sph'}.npy")
    np.save(path, out)
    f = out[ok].astype(np.float32)
    names = [f"{c}{k}" for k in range(2) for c in ("ori", "per", "cen", "fov", "scn")] + ["near"]
    print(f"wrote {path}: {out.shape}; mean over matched questions and candidates: "
          + ", ".join(f"{n} {f[..., i].mean():.3f}" for i, n in enumerate(names)))
    land = np.argmax(from_azel(caz, cel) @ from_azel(*Q["truth"][ok].T).T, 0)    # candidate nearest the landing
    hit = f[np.arange(len(land)), land]
    print("  at the landing candidate vs all: " + ", ".join(f"{n} {hit[:, i].mean():.3f}/{f[..., i].mean():.3f}"
                                                         for i, n in enumerate(names) if n in ("scn1", "near", "per1", "fov1")))


if __name__ == "__main__":
    main()
