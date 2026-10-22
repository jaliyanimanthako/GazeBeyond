"""Are two packed benchmarks the same samples, splits, folds and targets? (e.g. converted lab data vs a rebuild)

    python scripts/compare_benchmarks.py --a data/fovs_aria_lab --b data/rebuild_b4

Samples are matched by (recording, position within the recording), which is the anchor order in both pipelines.
For every split / fold present in both: sample counts, whether the same samples are in it, whether their targets are
identical, and the persistence baseline (copy the last observed second) computed from each benchmark's own inputs,
which must agree if inputs and targets agree. No GPU needed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.dataset import PackedBenchmark   # noqa: E402
from egospan.metrics import overlap_scores         # noqa: E402


def sample_keys(b):
    rec = b.arrays["rec"].astype(str)
    pos = np.zeros(len(rec), np.int64)
    seen = {}
    for i, r in enumerate(rec):                    # position of each sample within its recording, in storage order
        pos[i] = seen.get(r, 0)
        seen[r] = pos[i] + 1
    return np.array([f"{r}#{p}" for r, p in zip(rec, pos)])


def persistence_iou(b, idx):
    X, Y = b.inputs(idx), b.target(idx)
    last = (X[:, -(b.Tp // b.past_units):, :4] > 0).any(1)          # the last observed second
    return np.nanmean(overlap_scores(last, Y)["iou"], 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    a = ap.parse_args()
    A, B = PackedBenchmark(a.a), PackedBenchmark(a.b)
    ka, kb = sample_keys(A), sample_keys(B)
    print(f"A {a.a}: {len(ka)} samples, inputs {A.Tp} x {A.in_channels}   |   B {a.b}: {len(kb)} samples, "
          f"inputs {B.Tp} x {B.in_channels}")
    print(f"same set of samples overall: {set(ka) == set(kb)}")
    pos_b = {k: i for i, k in enumerate(kb)}
    keys = [k for k in A.splits if k in B.splits]
    print(f"\n{'split':14s} {'A n':>6s} {'B n':>6s} {'same samples':>13s} {'targets equal':>14s}   "
          "persistence mean IoU A / B")
    for key in sorted(keys, key=lambda s: (s.startswith("fold"), s)):
        ia, ib = A.splits[key], B.splits[key]
        same = set(ka[ia]) == set(kb[ib])
        teq, pa, pb = "", np.nan, np.nan
        if same and key.endswith("test"):
            matched = np.array([pos_b[k] for k in ka[ia] if k in pos_b])
            if len(matched) == len(ia):
                teq = str(bool(np.array_equal(A.target(ia), B.target(matched))))
            pa, pb = persistence_iou(A, ia).mean(), persistence_iou(B, ib).mean()
        print(f"{key:14s} {len(ia):6d} {len(ib):6d} {str(same):>13s} {teq:>14s}   {pa:.4f} / {pb:.4f}")


if __name__ == "__main__":
    main()
