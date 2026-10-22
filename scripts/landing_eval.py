"""Angle error of the predicted landing direction on attention-shift samples (build.window.anchor = shift): the lab's
WHERE metric, for our 3D forecasters.

    python scripts/landing_eval.py --runs runs/fovs_aria_shift_sph/cv/ft_f4 runs/fovs_aria_shift_sph/cv/scratch_f4
    python scripts/landing_eval.py --runs ... \
        --lab-queries /data/SS/SS/Dynamic_Fusion_ALFA/data/traj_forecast/attention_shift/senses_suite/data_aea_lag.npz

Per test sample, from the run's predicted span map (probs_test.npy, the --level span, foveal by default): the best
cell (highest probability), refined to the probability-weighted mean direction of the cells within 15 deg of it (as
WHERE refines its best direction), seen from the anchor (the eye at the query). Error = the angle to the landing
fixation's mean gaze direction (land_dir). Reported: median [95% CI over recordings], in / out of the camera view,
best of 3 (three peaks >= 30 deg apart), next to "stay" (the current fixation's direction, gaze_now), all over the
same samples. --lab-queries restricts every row to the samples that match one of the lab's WHERE questions (same
recording, anchor within 50 ms), so the numbers sit next to the lab's table on identical shifts.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.dataset import PackedBenchmark   # noqa: E402
from egospan.engine import load_run                # noqa: E402
from egospan.lift import voxel_centers             # noqa: E402
from egospan.video import image_cell               # noqa: E402


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def angle(a, b):
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


def cell_dirs(bench, shape):
    """(cells, 3) unit direction of every predicted cell's centre from the anchor."""
    if bench.grid is not None and tuple(shape) == bench.grid.shape:
        c = bench.grid.centers()
    elif tuple(shape) == (bench.R,) * 3:
        c = voxel_centers(bench.R, bench.meta["lift"]["cube_size"])
    else:
        raise SystemExit(f"predictions of shape {shape} match neither grid of {bench.root}")
    return unit(c.reshape(-1, 3))


def predict_dirs(prob, dirs, k=3, refine_deg=15.0, apart_deg=30.0):
    """prob (cells,) -> (k, 3): best direction (refined within refine_deg) and the next peaks >= apart_deg away."""
    p = prob.astype(np.float64).copy()
    out = []
    for _ in range(k):
        j = int(np.argmax(p))
        if p[j] <= 0:
            out.append(out[-1] if out else dirs[j])
            continue
        near = dirs @ dirs[j] > np.cos(np.radians(refine_deg))
        out.append(unit((prob[near, None] * dirs[near]).sum(0)))
        p[dirs @ dirs[j] > np.cos(np.radians(apart_deg))] = 0
    return np.stack(out)


def lab_match(bench, idx, path, tol=0.05):
    """mask over idx: samples matching a lab WHERE question (same recording, anchor within tol s)."""
    Z = np.load(path)
    lab = {}
    for r, t in zip(Z["rec"].astype(str), Z["t"].astype(np.float64)):
        lab.setdefault(r, []).append(t)
    lab = {r: np.sort(v) for r, v in lab.items()}
    ok = np.zeros(len(idx), bool)
    for i, j in enumerate(idx):
        r, t = str(bench.arrays["rec"][j]), bench.arrays["t0_us"][j] / 1e6
        v = lab.get(r)
        if v is not None and len(v):
            k = np.clip(np.searchsorted(v, t), 1, len(v) - 1) if len(v) > 1 else 0
            ok[i] = np.min(np.abs(v[max(k - 1, 0):k + 1] - t)) <= tol
    return ok


def boot_ci(err, groups, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = [np.flatnonzero(groups == g) for g in ug]
    meds = [np.median(err[np.concatenate([by[i] for i in rng.integers(0, len(ug), len(ug))])]) for _ in range(n)]
    return np.percentile(meds, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--level", default="foveal")
    ap.add_argument("--lab-queries", default=None, help="the lab's data_aea_lag.npz: evaluate on matching samples only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, ref = {}, None
    for r in a.runs:
        _, cfg = load_run(r, "cpu")
        b = PackedBenchmark(cfg.data.root)
        if "land_dir" not in b.arrays:
            raise SystemExit(f"{cfg.data.root} is not an attention-shift benchmark (build with build.window.anchor=shift)")
        idx = b.indices("test", cfg.data.fold)
        p = np.load(os.path.join(r, "probs_test.npy"))
        if len(p) != len(idx):
            raise SystemExit(f"{r}: probs_test.npy has {len(p)} rows, the test split {len(idx)}")
        c = list(cfg.data.levels).index(a.level)
        dirs = cell_dirs(b, p.shape[2:])
        land, now = b.arrays["land_dir"][idx].astype(np.float64), b.arrays["gaze_now"][idx].astype(np.float64)
        keep = lab_match(b, idx, a.lab_queries) if a.lab_queries else np.ones(len(idx), bool)
        key = (cfg.data.root, cfg.data.fold, int(keep.sum()))
        if ref is None:
            ref = key
        elif key != ref:
            raise SystemExit("the runs must share the benchmark, the fold (and the lab match)")
        preds = np.stack([predict_dirs(p[i, c].reshape(-1), dirs) for i in range(len(idx))])     # (n, 3, 3)
        err = angle(preds[:, 0], land)
        best3 = np.min(angle(preds, land[:, None]), 1)
        rows[os.path.basename(r.rstrip("/"))] = (err, best3)
        if "stay" not in rows:
            stay = angle(now, land)
            rows = {"stay (current fixation)": (stay, stay), **rows}
            groups = b.arrays["rec"][idx]
            inview = image_cell(land) >= 0
    n = int(keep.sum())
    print(f"{n} test shifts" + (f" (of {len(keep)}; matching the lab's questions)" if a.lab_queries else "")
          + f", {len(np.unique(groups[keep]))} recordings; in view {inview[keep].mean():.0%}")
    print(f"\n{'':>28s} {'median':>7s} {'[95% CI]':>14s} {'in view':>8s} {'out of view':>12s} {'best of 3':>10s}")
    out = {"n": n, "level": a.level}
    for name, (err, best3) in rows.items():
        e, g = err[keep], groups[keep]
        lo, hi = boot_ci(e, g)
        iv, ov = np.median(e[inview[keep]]), np.median(e[~inview[keep]]) if (~inview[keep]).any() else np.nan
        print(f"{name[:28]:>28s} {np.median(e):7.1f} {f'[{lo:.1f}, {hi:.1f}]':>14s} {iv:8.1f} {ov:12.1f} "
              f"{np.median(best3[keep]):10.1f}")
        out[name] = dict(median=float(np.median(e)), ci=[float(lo), float(hi)], in_view=float(iv),
                         out_of_view=float(ov), best_of_3=float(np.median(best3[keep])))
    path = a.out or os.path.join(a.runs[0], "landing_eval.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}  (degrees; the lab's WHERE table: stay 42.4, motion 33.9, + memory + V-JEPA 29.0)")


if __name__ == "__main__":
    main()
