"""Test IoU split by how far the gaze moves in the forecast window (largest angle between the future gaze and the gaze
at the last observed step), for one or more runs plus the persistence baseline (copy the last observed second).

    python scripts/shift_breakdown.py --runs runs/fovs_aria_lab/front_s0 runs/fovs_aria_lab/front_gaze_s0 \
        --lab-src /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria          # converted lab data: gaze from test.npz
    python scripts/shift_breakdown.py --runs runs/dummy_b4soft                    # our own builds store the gaze

All runs must use the same benchmark and test split. Threshold 0.5, full 16^3 grid (as compare_variants.py).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.dataset import PackedBenchmark   # noqa: E402
from egospan.engine import load_run                # noqa: E402
from egospan.lift import LEVELS                    # noqa: E402
from egospan.metrics import overlap_scores         # noqa: E402

EDGES = (0, 8, 20, 45, 181)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def gaze_shift(bench, idx, lab_src=None):
    """(n,) largest future gaze shift in degrees for the test samples idx."""
    if lab_src:
        z = np.load(os.path.join(lab_src, "test.npz"))
        if len(z["g_now"]) != len(idx):
            raise ValueError("lab test.npz and the benchmark's test split differ in size")
        g_now, g_fut = unit(z["g_now"]), unit(z["g_fut"])
    else:
        if "gaze_grid" not in bench.arrays:
            raise ValueError("this benchmark stores no per-step gaze (converted lab data?): pass --lab-src")
        P = bench.past_units * bench.steps_per_unit
        G = unit(bench.arrays["gaze_grid"][idx].astype(np.float64))
        g_now, g_fut = G[:, P - 1], G[:, P:]
    return np.degrees(np.arccos(np.clip((g_fut * g_now[:, None]).sum(-1), -1, 1))).max(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--lab-src", default=None, help="lab benchmark folder (has test.npz with g_now, g_fut)")
    ap.add_argument("--levels", nargs="+", default=["central", "foveal"])
    a = ap.parse_args()
    _, cfg = load_run(a.runs[0], "cpu")
    bench = PackedBenchmark(cfg.data.root)
    idx = bench.indices("test", cfg.data.fold)
    shift = gaze_shift(bench, idx, a.lab_src)
    native = bench.grid is not None and cfg.eval.space == "native"      # spherical run scored on its own cells
    Y = (bench.target if native else bench.eval_target)(idx, cfg.data.horizon, cfg.data.levels)
    X = bench.inputs(idx)
    last = X[:, -(bench.Tp // bench.past_units):] > 0                   # the last observed second, any bin
    names = list(cfg.data.levels)
    persist = np.stack([last[:, :, LEVELS.index(l)].any(1) for l in names], 1)
    if bench.grid is not None and not native:                           # spherical cells -> Cartesian voxels
        persist = bench.to_cartesian(persist) > 0.5
    preds = {"persistence (last 1 s)": persist}
    for r in a.runs:
        p = np.load(os.path.join(r, "probs_test.npy")) > 0.5
        if p.shape != Y.shape:
            raise SystemExit(f"{r}: predictions {p.shape[2:]} vs targets {Y.shape[2:]}: runs must share the benchmark "
                             "and the scoring space (eval.space)")
        preds[os.path.basename(r.rstrip("/"))] = p
    bins = [(lo, hi, (shift >= lo) & (shift < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    for lev in a.levels:
        c = names.index(lev)
        print(f"\n== {lev} IoU by largest future gaze shift")
        print(f"{'shift':>10s} {'samples':>8s}  " + "  ".join(f"{n[:22]:>22s}" for n in preds))
        ious = {n: overlap_scores(p[:, c:c + 1], Y[:, c:c + 1])["iou"][:, 0] for n, p in preds.items()}
        for lo, hi, m in bins:
            cells = "  ".join(f"{np.nanmean(v[m]):22.4f}" for v in ious.values())
            print(f"{lo:3d}-{hi:3d} deg {m.mean():8.1%}  " + cells)
        print(f"{'all':>10s} {1:8.0%}  " + "  ".join(f"{np.nanmean(v):22.4f}" for v in ious.values()))


if __name__ == "__main__":
    main()
