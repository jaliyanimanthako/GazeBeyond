"""Look inside a packed benchmark: stored sizes, one sample's tensors, a batch, sanity checks and a printed slice.

    python scripts/inspect_data.py --root data/fovs_aria_lab
    python scripts/inspect_data.py --root data/fovs_aria_lab --split test --index 100
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.dataset import PackedBenchmark, SpanDataset   # noqa: E402
from egospan.lift import CHANNELS, LEVELS, RAY_CHANNEL          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--n-stats", type=int, default=2000, help="samples used for dataset-wide statistics")
    a = ap.parse_args()
    b = PackedBenchmark(a.root)
    R, Tp, Tf = b.R, b.Tp, b.Tf
    names = list(CHANNELS) + [RAY_CHANNEL] * (b.in_channels - len(CHANNELS))

    print("== 1. stored on disk (" + ("soft temporal bins: one uint8 per voxel" if b.soft
                                         else "bit-packed: 8 voxels per byte") + ")")
    per = b.arrays["X"].shape[1] * (1 if b.soft else 8)
    kind = "uint8 values (soft temporal bins)" if b.soft else "bits"
    C = b.in_channels
    print(f"   X  {b.arrays['X'].shape}  -> {per} {kind} = {Tp} x {C} x {R}^3 = {Tp * C * R ** 3}")
    print(f"   YF {b.arrays['YF'].shape} -> {b.arrays['YF'].shape[1] * 8} bits = {Tf} x 4 x {R}^3 = {Tf * 4 * R ** 3}")
    print("   splits: " + ", ".join(f"{p} {len(b.indices(p))}" for p in ("train", "val", "test")))

    ds = SpanDataset(b, a.split)
    item = ds[a.index]
    x, y = item["x"].numpy().astype(bool), item["y"].numpy().astype(bool)
    print(f"\n== 2. one sample ({a.split}[{a.index}], recording {b.arrays['rec'][item['idx']]})")
    print(f"   x {tuple(x.shape)}  = (input time bins Tp, channels {names}, X, Y, Z)")
    print(f"   y {tuple(y.shape)}     = (levels {list(LEVELS)}, X, Y, Z), union over the whole forecast window")
    for t in range(Tp):
        print(f"   x bin {t + 1}: " + "  ".join(f"{c} {x[t, i].sum():3d}" for i, c in enumerate(names)))
    print("   y future  : " + "  ".join(f"{c} {y[i].sum():3d}" for i, c in enumerate(LEVELS))
          + f"   (of {R ** 3} voxels)")

    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(ds, batch_size=16)))
    print(f"\n== 3. one batch as the model gets it\n   x {tuple(batch['x'].shape)} {batch['x'].dtype}   "
          f"y {tuple(batch['y'].shape)} {batch['y'].dtype}")

    idx = b.indices(a.split)[:a.n_stats]
    X, Y = b.inputs(idx), b.target(idx)
    print(f"\n== 4. statistics over {len(idx)} {a.split} samples")
    occ_x = X.mean((0, 1, 3, 4, 5))
    label = "mean value" if b.soft else "occupancy"
    print(f"   input {label}  " + "  ".join(f"{c} {100 * occ_x[i]:.2f}%" for i, c in enumerate(names)))
    print("   target occupancy " + "  ".join(f"{c} {100 * Y[:, i].mean():.2f}%" for i, c in enumerate(LEVELS)))
    print("   non-empty target " + "  ".join(f"{c} {100 * Y[:, i].any((1, 2, 3)).mean():.0f}%"
                                             for i, c in enumerate(LEVELS)) + "   (paper: foveal ~80%)")

    print("\n== 5. consistency checks (must all be True)")
    span_in_scene = (X[:, :, :4] <= X[:, :, 4:5]).all()
    nested_x = (X[:, :, 3] <= X[:, :, 2]).all() and (X[:, :, 2] <= X[:, :, 1]).all()
    nested_y = (Y[:, 3] <= Y[:, 2]).all() and (Y[:, 2] <= Y[:, 1]).all()
    print(f"   every span voxel is also a scene voxel:          {span_in_scene}")
    print(f"   foveal ⊆ central ⊆ peripheral (input):          {nested_x}")
    print(f"   foveal ⊆ central ⊆ peripheral (target):         {nested_y}")

    print(f"\n== 6. slice of {a.split}[{a.index}] through its fullest layer (rows = X, columns = Y)")
    g = x.any(0)                                         # both observed seconds
    layer = int(np.argmax(g[4].sum((0, 1)) + 10 * g[3].sum((0, 1))))
    sym = np.full((R, R), ".")
    order = [(4, "s")] + [(5, "r")] * (b.in_channels > 5) + [(0, "o"), (1, "p"), (2, "c"), (3, "F")]
    for i, s in order:
        sym[g[i, :, :, layer]] = s
    yl = np.full((R, R), ".")
    for i, s in ((0, "o"), (1, "p"), (2, "c"), (3, "F")):
        yl[y[i, :, :, layer]] = s
    print(f"   Z = {layer};  . empty  s scene  r gaze ray  o orientation  p peripheral  c central  F foveal")
    print(f"   {'input (past 2 s)':{2 * R}s}   target (future)")
    for r in range(R):
        print("   " + " ".join(sym[r]) + "    " + " ".join(yl[r]))


if __name__ == "__main__":
    main()
