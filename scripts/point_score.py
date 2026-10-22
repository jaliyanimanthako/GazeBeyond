"""Score runs with different grids on one yardstick: the future scene points (egospan/point_metrics.py).

    python scripts/point_score.py --points data/fovs_aria_sph/points_test_f4.npz \
        --runs runs/fovs_aria_lab/cv/front_vid_f4 runs/fovs_aria_sph/cv/sph_vid_f4

Each run's test predictions (probs_test.npy, threshold 0.5) are looked up at every future point of every sample in the
points file (scripts/build_future_points.py): R^3 voxels for Cartesian runs (front-half runs predict the back empty),
spherical cells for spherical runs scored natively. Samples are matched by "<recording>#<position>", so runs on the
converted lab data and on our rebuilds can be compared on the same samples. Also printed per grid:
    perfect grid    the true target in that grid: the best that grid can score on points (its ceiling)
    copy last 1 s   the last observed second's spans in that grid (persistence)
and, from the points themselves (points files with past bits):
    [points] copy last 1 s / past 2 s   exactly the points that were in the span then (persistence without a grid)
    coverage        share of the future true points already observed in the past 1 s / 2 s / at any earlier time
                    of the recording (the input window vs the whole SLAM map so far: is a longer scene memory needed?)
Tables: point IoU on all points, on the front half only (z >= 0, where every model predicts), point F1, and foveal
point IoU by largest future gaze shift, all at the first --thresholds value (default 0.5, as trained). With more
thresholds, a sweep table follows: a model scored with Dice on cells may be too generous for points at 0.5. Choosing a
threshold on the test split is a diagnostic only; for reported numbers pick it on validation.
Writes point_score.json next to the points file (or --out).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.build import sample_keys                                 # noqa: E402
from egospan.data.dataset import PackedBenchmark                           # noqa: E402
from egospan.engine import load_run                                        # noqa: E402
from egospan.lift import LEVELS                                            # noqa: E402
from egospan.point_metrics import lookup, point_scores, truth_bits         # noqa: E402

EDGES = (0, 8, 20, 45, 181)


def rows_for(bench, idx, keys_wanted):
    """positions in idx of the samples with keys_wanted (all must be there)."""
    keys = sample_keys(bench.arrays["rec"])[idx]
    where = {k: i for i, k in enumerate(keys)}
    missing = [k for k in keys_wanted if k not in where]
    if missing:
        raise SystemExit(f"{len(missing)} samples of the points file are not in this run's test split, "
                         f"e.g. {missing[:3]}: build the points for the same fold")
    return np.array([where[k] for k in keys_wanted])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5],
                    help="the first one is used for the main tables; more give a sweep, e.g. 0.5 0.6 0.7 0.8 0.9")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = np.load(a.points)
    keys, off, xyz, bits, shift = P["key"], P["off"], P["xyz"], P["bits"], P["shift"]
    n = len(keys)
    sample = np.repeat(np.arange(n), np.diff(off))
    truth = truth_bits(bits, LEVELS)
    front = xyz[:, 2] >= 0
    print(f"{n} samples, {len(xyz)} future points (median {int(np.median(np.diff(off)))} per sample)")

    preds, refs, sweep = {}, {}, {}
    benches = {}
    big = (shift >= 45)                                       # the large-shift group, for the sweep
    c_fov = LEVELS.index("foveal")

    def summary(hit):
        s = point_scores(hit, truth, sample, n)["iou"]
        return np.nanmean(s, 0).tolist() + [float(np.nanmean(s[big, c_fov]))]
    for r in a.runs:
        _, cfg = load_run(r, "cpu")
        if list(cfg.data.levels) != list(LEVELS):
            raise SystemExit(f"{r}: point scoring needs all four span levels")
        b = benches.setdefault(cfg.data.root, PackedBenchmark(cfg.data.root))
        idx = b.indices("test", cfg.data.fold)
        if cfg.data.test_categories:
            idx = idx[np.isin(b.arrays["category"][idx], list(cfg.data.test_categories))]
        j = rows_for(b, idx, keys)
        p = np.load(os.path.join(r, "probs_test.npy"))
        if len(p) != len(idx):
            raise SystemExit(f"{r}: probs_test.npy has {len(p)} rows, the test split {len(idx)}")
        p = p[j]
        if p.shape[2:] == (b.R,) * 3:
            grid, gname = None, f"{b.R}^3 voxels"
        elif b.grid is not None and p.shape[2:] == b.grid.shape:
            grid, gname = b.grid, "spherical cells"
        else:
            raise SystemExit(f"{r}: predictions of shape {p.shape[2:]} match neither grid of {cfg.data.root}")
        if grid is None and b.grid is not None:
            raise SystemExit(f"{r}: a spherical run scored on voxels: use eval.space=native for point scores")
        cube = b.meta["lift"]["cube_size"]
        name = os.path.basename(r.rstrip("/"))
        prob = lookup(p, xyz, sample, grid, cube)             # (M, L) predicted probability at every point
        preds[name] = prob > a.thresholds[0]
        if len(a.thresholds) > 1:
            sweep[name] = {t: summary(prob > t) for t in a.thresholds}
        del prob
        ref = f"[{gname}]"
        if ref not in refs:                                   # ceiling and persistence of this grid, once
            sel = idx[j]
            target = b.target(sel) if grid is not None else b.eval_target(sel)
            refs[f"{ref} perfect grid"] = lookup(target, xyz, sample, grid, cube)
            X = b.inputs(sel)
            last = (X[:, -(b.Tp // b.past_units):, :len(LEVELS)] > 0).any(1)
            refs[f"{ref} copy last 1 s"] = lookup(last, xyz, sample, grid, cube)
            refs[ref] = True
    table = {k: v for k, v in refs.items() if not isinstance(v, bool)}
    has_past = "past1" in P.files
    if has_past:                                              # persistence on the exact points, no grid
        table["[points] copy last 1 s"] = truth_bits(P["past1"], LEVELS)
        table["[points] copy past 2 s"] = truth_bits(P["past2"], LEVELS)
    table.update(preds)

    def block(title, mask=None, metric="iou"):
        print(f"\n== {title}")
        print(f"{'':>34s} " + " ".join(f"{l[:11]:>11s}" for l in LEVELS) + f" {'MEAN':>8s}")
        res = {}
        for name, hit in table.items():
            if mask is None:
                s = point_scores(hit, truth, sample, n)[metric]
            else:
                s = point_scores(hit[mask], truth[mask], sample[mask], n)[metric]
            m = np.nanmean(s, 0)
            res[name] = m.tolist()
            print(f"{name[:34]:>34s} " + " ".join(f"{v:11.4f}" for v in m) + f" {np.nanmean(m):8.4f}")
        return res

    out = {"samples": n, "points": int(len(xyz)),
           "iou": block("point IoU, all points in the cube"),
           "iou_front": block("point IoU, front half points only (z >= 0)", front),
           "f1": block("point F1, all points", metric="f1")}
    print("\n== foveal point IoU by largest future gaze shift")
    groups = [(lo, hi, (shift >= lo) & (shift < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    print(f"{'':>34s} " + " ".join(f"{f'{lo}-{hi} deg':>11s}" for lo, hi, _ in groups))
    out["foveal_by_shift"] = {}
    c = LEVELS.index("foveal")
    for name, hit in table.items():
        s = point_scores(hit, truth, sample, n)["iou"][:, c]
        vals = [float(np.nanmean(s[m])) for _, _, m in groups]
        out["foveal_by_shift"][name] = vals
        print(f"{name[:34]:>34s} " + " ".join(f"{v:11.4f}" for v in vals))
    print(f"{'samples':>34s} " + " ".join(f"{m.mean():11.1%}" for _, _, m in groups))
    if has_past:
        print("\n== coverage: share of the future true points already observed (in the map at prediction time)")
        print(f"{'':>34s} " + " ".join(f"{l[:11]:>11s}" for l in LEVELS) + f" {'foveal 45+':>11s}")
        out["coverage"] = {}
        rows = [("seen in the last 1 s", "past1"), ("seen in the past 2 s", "past2")]
        if "seen_ever" in P.files:
            rows.append(("seen any time before now", "seen_ever"))
        for label, key in rows:
            seen = P[key] if key == "seen_ever" else (P[key] & 16) > 0
            vals = []
            for c in range(len(LEVELS)):
                t = truth[:, c]
                vals.append(float((t & seen).sum() / max(t.sum(), 1)))
            t = truth[:, c_fov] & big[sample]
            vals.append(float((t & seen).sum() / max(t.sum(), 1)))
            out["coverage"][label] = vals
            print(f"{label:>34s} " + " ".join(f"{v:11.1%}" for v in vals))
    if sweep:
        print("\n== point IoU by threshold (test split: a diagnostic; pick thresholds on validation to report)")
        print(f"{'':>34s} " + " ".join(f"{l[:11]:>11s}" for l in LEVELS) + f" {'MEAN':>8s} {'foveal 45+':>11s}")
        for name, hit in table.items():
            if name in sweep:
                for t, v in sweep[name].items():
                    print(f"{f'{name[:26]} @ {t:g}':>34s} " + " ".join(f"{x:11.4f}" for x in v[:4])
                          + f" {np.mean(v[:4]):8.4f} {v[4]:11.4f}")
            elif "copy" in name:
                v = summary(hit)
                print(f"{name[:34]:>34s} " + " ".join(f"{x:11.4f}" for x in v[:4]) + f" {np.mean(v[:4]):8.4f} "
                      f"{v[4]:11.4f}")
        out["sweep"] = {k: {str(t): v for t, v in d.items()} for k, d in sweep.items()}
    out["threshold"] = a.thresholds[0]
    path = a.out or os.path.join(os.path.dirname(a.points), "point_score.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
