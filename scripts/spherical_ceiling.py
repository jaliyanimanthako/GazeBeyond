"""How much does the spherical grid lose when its cells are mapped back to the paper's 20 cm voxels? (no training)

    python scripts/spherical_ceiling.py --bench data/fovs_aria_sph                 # all samples
    python scripts/spherical_ceiling.py --bench data/fovs_aria_sph --part test --fold 4
    python scripts/spherical_ceiling.py --bench data/fovs_aria_sph --cart-bench data/fovs_aria_rebuild   # + surfaces

A spherical model is scored after mapping its cells to the R^3 Cartesian grid. Even a perfect spherical prediction
(the true spherical target) does not map back exactly: a coarse cell spreads over several voxels, a fine cell covers
part of one. This prints, per span, the Cartesian IoU of the mapped true target with the max rule at 0.5 (how the model
is scored) and with the overlap-weighted mean for comparison, i.e. the ceiling the representation puts on the score,
plus the target sizes in cells and in voxels. Run it on a few recordings first (build with --limit) before building
everything.

--cart-bench (the Cartesian build of the same recordings, e.g. the lab-matching rebuild): also maps back only onto
voxels that held a surface in the past 2 s (known at prediction time), which removes most of the boundary voxels a
coarse cell lights up; and shows what that mask costs on its own (targets on surfaces not seen before).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.dataset import PackedBenchmark   # noqa: E402
from egospan.lift import LEVELS                    # noqa: E402
from egospan.metrics import overlap_scores         # noqa: E402

RULES = (("max", 0.5), ("mean", 0.5), ("mean", 0.25))      # (mapping rule, threshold); the model is scored with max


def match(bench, cart):
    """rows of `cart` holding the same samples as `bench` (by recording and anchor time); -1 where missing."""
    key = {(r, int(t)): i for i, (r, t) in enumerate(zip(cart.arrays["rec"], cart.arrays["t0_us"]))}
    return np.array([key.get((r, int(t)), -1) for r, t in zip(bench.arrays["rec"], bench.arrays["t0_us"])])


def ceiling(bench, idx, chunk=256, cart=None, rows=None):
    """-> {level: {"rule@thr": {iou, precision, recall}}, median target cells, median target voxels}. With `cart`
    (and `rows`, from match) also "max@0.5 on past surfaces" and "mask only" (the Cartesian target on past surfaces)."""
    rules = list(RULES) + ([("surf", 0.5), ("maskonly", 0.5)] if cart is not None else [])
    scores = {r: {k: [] for k in ("iou", "precision", "recall")} for r in rules}
    cells, voxels = [], []
    for s in range(0, len(idx), chunk):
        j = idx[s:s + chunk]
        ys = bench.target(j)                                   # (n, 4, A, E, D) spherical
        yc = bench.eval_target(j)                              # (n, 4, R, R, R) Cartesian
        preds = {}
        for mode in {m for m, _ in RULES}:
            mapped = bench.grid.to_cartesian(ys, bench.R, mode)
            for r in [r for r in RULES if r[0] == mode]:
                preds[r] = mapped > r[1]
        if cart is not None:
            if np.any(rows[j] < 0):
                raise SystemExit("some spherical samples are not in --cart-bench: build both from the same recordings")
            if not np.array_equal(cart.target(rows[j]), yc):
                raise SystemExit("--cart-bench targets differ from the spherical build's YF_cart: different settings?")
            surf = (cart.inputs(rows[j])[:, :, 4] > 0).any(1)[:, None]        # past scene voxels (n, 1, R, R, R)
            preds[("surf", 0.5)] = preds[("max", 0.5)] & surf
            preds[("maskonly", 0.5)] = yc & surf
        for r, pr in preds.items():
            o = overlap_scores(pr, yc)
            for k in scores[r]:
                scores[r][k].append(o[k])
        cells.append(ys.reshape(*ys.shape[:2], -1).sum(-1))
        voxels.append(yc.reshape(*yc.shape[:2], -1).sum(-1))
    cells, voxels = np.concatenate(cells), np.concatenate(voxels)
    out = {}
    for c, lev in enumerate(LEVELS):
        has = voxels[:, c] > 0
        out[lev] = {f"{m}@{t}": {k: float(np.nanmean(np.concatenate(v)[:, c])) for k, v in scores[(m, t)].items()}
                    for m, t in rules}
        out[lev]["median_cells"] = float(np.median(cells[has, c])) if has.any() else float("nan")
        out[lev]["median_voxels"] = float(np.median(voxels[has, c])) if has.any() else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--part", default="all", help="all, or train / val / test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--cart-bench", default=None, help="Cartesian build of the same recordings (past surfaces)")
    a = ap.parse_args()
    b = PackedBenchmark(a.bench)
    if b.grid is None:
        raise SystemExit(f"{a.bench} is a Cartesian benchmark (build with build.lift.grid=spherical)")
    idx = np.arange(len(b.arrays["X"])) if a.part == "all" else b.indices(a.part, a.fold)
    cart = PackedBenchmark(a.cart_bench) if a.cart_bench else None
    rows = match(b, cart) if cart is not None else None
    res = ceiling(b, idx, cart=cart, rows=rows)
    print(f"{len(idx)} samples, grid {b.grid.shape} = {b.grid.n_cells} cells, scored on {b.R}^3 voxels")
    rules = [f"{m}@{t}" for m, t in RULES] + (["surf@0.5", "maskonly@0.5"] if cart is not None else [])
    print(f"\n{'span':>12s} " + " ".join(f"{r:>10s}" for r in rules)
          + f"   {'max: prec / recall':>18s} {'cells':>6s} {'voxels':>6s}")
    for lev, r in res.items():
        print(f"{lev:>12s} " + " ".join(f"{r[k]['iou']:10.3f}" for k in rules)
              + f"   {r['max@0.5']['precision']:8.3f} / {r['max@0.5']['recall']:6.3f} {r['median_cells']:6.0f} "
              + f"{r['median_voxels']:6.0f}")
    print("\nIoU of the true spherical target mapped to the 20 cm voxels and scored like the paper. max@0.5 is how the "
          "model is scored: the ceiling a perfect spherical prediction reaches (recall 1; precision < 1 where cells are "
          "bigger than voxels).")
    if cart is not None:
        print("surf@0.5: max@0.5 kept only on voxels with a surface in the past 2 s. maskonly@0.5: the true Cartesian "
              "target on those voxels (what the mask alone costs).")


if __name__ == "__main__":
    main()
