"""Compare two variants over location-wise cross-validation folds, fold by fold (paired).

    python scripts/cv_compare.py --a runs/fovs_aria_lab/cv/front --b runs/fovs_aria_lab/cv/front_gaze

Reads <prefix>_f1 ... <prefix>_f5 /metrics_test.json of both variants (threshold as trained, default 0.5) and prints
per-fold mean / foveal IoU, then per level the mean ± std of (b - a) over folds, how many folds b wins, and a paired t.
Both variants must be scored in the same space: IoU on spherical cells and on 20 cm voxels are different scales.
"""
import argparse
import json
import os

import numpy as np

LEVELS = ("orientation", "peripheral", "central", "foveal")


def load(prefix, f):
    path = f"{prefix}_f{f}/metrics_test.json"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} missing (did that fold finish?)")
    m = json.load(open(path))
    return np.array([m[lev]["iou"] for lev in LEVELS]), m.get("space", "16^3 voxels")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="run prefix of the reference variant (folders <a>_f1 ... <a>_fK)")
    ap.add_argument("--b", required=True, help="run prefix of the variant being tested")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    la, lb = [load(a.a, f) for f in a.folds], [load(a.b, f) for f in a.folds]
    spaces = {s for _, s in la + lb}
    if len(spaces) > 1:
        raise SystemExit(f"the runs are scored in different spaces {sorted(spaces)}: their IoUs are not comparable")
    A = np.stack([x for x, _ in la])                                    # (folds, levels)
    B = np.stack([x for x, _ in lb])
    na, nb = os.path.basename(a.a), os.path.basename(a.b)
    print(f"{'test location':>13s} {na + ' mean':>14s} {nb + ' mean':>18s} {'diff':>8s}   "
          f"{'a foveal':>9s} {'b foveal':>9s} {'diff':>8s}")
    for i, f in enumerate(a.folds):
        print(f"{f:>13d} {A[i].mean():14.4f} {B[i].mean():18.4f} {B[i].mean() - A[i].mean():+8.4f}   "
              f"{A[i, 3]:9.4f} {B[i, 3]:9.4f} {B[i, 3] - A[i, 3]:+8.4f}")
    k = len(a.folds)
    d = B - A
    print(f"\n{nb} - {na} over {k} folds: mean ± std, folds where {nb} wins, paired t")
    for i, lev in enumerate(LEVELS):
        sd = d[:, i].std(ddof=1)
        print(f"  {lev:12s} {d[:, i].mean():+.4f} ± {sd:.4f}   wins {int((d[:, i] > 0).sum())}/{k}   "
              f"t = {d[:, i].mean() / (sd / np.sqrt(k)) if sd > 0 else float('nan'):6.2f}")
    dm = d.mean(1)
    sd = dm.std(ddof=1)
    print(f"  {'MEAN':12s} {dm.mean():+.4f} ± {sd:.4f}   wins {int((dm > 0).sum())}/{k}   "
          f"t = {dm.mean() / (sd / np.sqrt(k)) if sd > 0 else float('nan'):6.2f}")
    print(f"\n{na} mean IoU per location: " + ", ".join(f"{v:.4f}" for v in A.mean(1))
          + f"   (spread {A.mean(1).max() - A.mean(1).min():.4f})")


if __name__ == "__main__":
    main()
