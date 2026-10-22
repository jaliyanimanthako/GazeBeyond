"""Convert the SensorySpotlight (aria-3d-analysis) built benchmark into this codebase's format, so SpanForecaster can be
trained on it without the raw Aria recordings.

Reads  <src>/{train,val,test}.npz   written by tools/fovs_build.py pack
       <src>/cv_splits.npz          optional, written by tools/fovs_cv_prep.py (location-wise folds 1..5)
Writes <out>/all.npz + <out>/splits.npz   (read by egospan.data.dataset.PackedBenchmark)

    python scripts/convert_sensoryspotlight.py \
        --src /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria --out data/fovs_aria_lab
    python scripts/train.py --config configs/fovs_aria_lab.yaml

What carries over: X (2 s observed x [orientation, peripheral, central, foveal, scene] x 16^3) and the future target.
The lab files store the target as four 0.5 s bins (YT); the paper's target is their union (= the lab's Y). Here the
bins become 4 future units, so data.horizon=4 (default: all) is the paper's setting and horizon=1..4 gives the 0.5..2 s
windows of Fig. 4-b. Grids are in the anchor CPF frame (vertical axis 1), so augmentation never flips axis 1.
What does not: per-step poses, so baselines.py's 2D-lift rows and eval_2d.py need the raw recordings.
"""
import argparse
import json
import os
import re
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.build import WindowConfig, unpack_grids   # noqa: E402
from egospan.lift import LiftConfig                          # noqa: E402

PARTS = ("train", "val", "test")
R = 16
X_BYTES, YT_BYTES = 2 * 5 * R ** 3 // 8, 16 * R ** 3 // 8


def location_of(rec):
    m = re.search(r"_loc(\d+)_", str(rec)) or re.match(r"loc(\d+)_", str(rec))
    return int(m.group(1)) if m else -1


def convert(src, out, check=200):
    Z = {p: np.load(os.path.join(src, f"{p}.npz")) for p in PARTS}
    for p in PARTS:
        assert Z[p]["X"].shape[1] == X_BYTES, f"{p}: X has {Z[p]['X'].shape[1]} bytes/sample, expected {X_BYTES}"
        assert Z[p]["YT"].shape[1] == YT_BYTES, f"{p}: YT has {Z[p]['YT'].shape[1]} bytes/sample, expected {YT_BYTES}"
    X = np.concatenate([Z[p]["X"] for p in PARTS])
    YF = np.concatenate([Z[p]["YT"] for p in PARTS])                      # (n, 4 bins x 4 levels x 16^3), bin-major
    rec = np.concatenate([Z[p]["rec"] for p in PARTS]).astype(str)
    split = np.concatenate([[p] * len(Z[p]["X"]) for p in PARTS])
    n = len(X)

    k = min(check, len(Z["test"]["X"]))               # the lab's union Y must equal the union of its bins
    yt = unpack_grids(Z["test"]["YT"][:k], (4, 4, R, R, R)).any(1)
    y = unpack_grids(Z["test"]["Y"][:k], (4, R, R, R))
    if not np.array_equal(yt, y):
        print(f"WARNING: union of YT bins differs from Y on {(yt != y).any((1, 2, 3, 4)).sum()} of {k} test samples")

    lift = asdict(LiftConfig(sor_neighbors=16, max_dist_std=0.1, grid_frame="local"))
    meta = dict(lift=lift, window=asdict(WindowConfig(future_units=4)), vertical_axis=1,
                dataset="aea", source="SensorySpotlight fovs_build.py", future_unit_seconds=0.5)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "all.npz"), X=X, YF=YF, rec=rec, split=split,
             location=np.array([location_of(r) for r in rec]), category=np.array([""] * n),
             meta=np.array(json.dumps(meta)))

    idx = {p: np.flatnonzero(split == p) for p in PARTS}
    cv = os.path.join(src, "cv_splits.npz")
    if os.path.isfile(cv):
        c = np.load(cv)
        folds = {key: c[key] for key in c.files if key.startswith("fold")}
        ok = all(len(v) == 0 or v.max() < n for v in folds.values())
        for key, v in folds.items():
            if key.endswith("_test") and len(v):
                locs = {location_of(r) for r in rec[v]}
                ok &= len(locs) == 1
        if ok:
            idx.update(folds)
        else:
            print("WARNING: cv_splits.npz does not line up with train+val+test; folds skipped")
    np.savez(os.path.join(out, "splits.npz"), **idx)
    for p in PARTS:
        print(f"{p}: {len(idx[p])} samples from {len(set(rec[idx[p]]))} recordings")
    print(f"folds: {sorted({k.split('_')[0] for k in idx if k.startswith('fold')}) or 'none'}  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    convert(a.src, a.out)
