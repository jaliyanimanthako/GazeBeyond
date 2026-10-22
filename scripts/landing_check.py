"""Where do the eyes land after a gaze shift, relative to what the person saw before? (data only; --run adds the model)

    LAB=/data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria
    python scripts/landing_check.py --bench data/fovs_aria_lab --part all --lab-src $LAB      # every sample
    python scripts/landing_check.py --run runs/fovs_aria_lab/cv/front_vid_f4 --lab-src $LAB   # + what the model predicts

Landing voxels of a sample: future foveal voxels (union over the forecast window) that were NOT foveal in the past
window: the places the eyes newly reach. For each group of largest future gaze shift it reports
    covered by     share of the landing voxels (pooled over samples) that lay inside a past region: the past
                   orientation / peripheral span, "seen, not looked at" (past peripheral minus past central), the past
                   scene (surfaces), the camera view, the front half. High = the information was in the input.
    candidates     median size of that region on surfaces (region and past scene), and density = landing voxels in
                   it / its size: the hit rate of guessing one of its voxels at random (the chance level a model has
                   to beat to pick the destination from it)
    --run          recall of the model's foveal prediction (threshold 0.5) on: start voxels (future foveal that were
                   past foveal), landing voxels inside past peripheral, landing voxels outside it
Past regions are the union over all past input grids, in the grid of the last observed step. Shift groups need the
gaze (our builds store it; for the converted lab data pass --lab-src, which has g_now / g_fut per sample).
Writes landing_check.json next to the run (or in --out).
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.dataset import PackedBenchmark   # noqa: E402
from egospan.engine import load_run                # noqa: E402
from egospan.lift import CHANNELS                  # noqa: E402
from egospan.video import lift_index               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("shift_breakdown", os.path.join(HERE, "shift_breakdown.py"))
shift_breakdown = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shift_breakdown)

O, P, C, F, S = (CHANNELS.index(c) for c in ("orientation", "peripheral", "central", "foveal", "scene"))
REGIONS = ("past orientation", "past peripheral", "seen, not looked at", "past central", "past scene",
           "camera view", "front half")
CANDIDATES = ("past orientation", "past peripheral", "seen, not looked at")


def lab_shift(bench, idx, lab_src):
    """largest future gaze shift (deg) from the lab files: all of train + val + test (the converted data keeps that
    order) when every file has g_now / g_fut, else the lab test file for the test split only."""
    parts = [np.load(os.path.join(lab_src, f"{p}.npz")) for p in ("train", "val", "test")]
    if all("g_now" in z.files for z in parts) and sum(len(z["g_now"]) for z in parts) == len(bench.arrays["X"]):
        g_now = np.concatenate([z["g_now"] for z in parts])[idx]
        g_fut = np.concatenate([z["g_fut"] for z in parts])[idx]
        u = shift_breakdown.unit
        return np.degrees(np.arccos(np.clip((u(g_fut) * u(g_now)[:, None]).sum(-1), -1, 1))).max(1)
    return shift_breakdown.gaze_shift(bench, idx, lab_src)


def regions(past, R, vertical_axis, cube_size):
    """past (n, C, R, R, R) bool, union over the past grids -> dict name -> (n, R^3) bool."""
    n = len(past)
    flat = past.reshape(n, past.shape[1], -1)
    out = {"past orientation": flat[:, O], "past peripheral": flat[:, P],
           "seen, not looked at": flat[:, P] & ~flat[:, C], "past central": flat[:, C], "past scene": flat[:, S]}
    if vertical_axis == 1:                       # anchor-CPF grid: forward = last axis
        front = np.zeros((R, R, R), bool)
        front[..., R // 2:] = True
        out["front half"] = np.broadcast_to(front.reshape(-1), (n, R ** 3))
        out["camera view"] = np.broadcast_to(lift_index(R, cube_size) >= 0, (n, R ** 3))
    return out


def accumulate(acc, key, landing, start, regs, pred=None):
    a = acc.setdefault(key, {"samples": 0, "with_landing": 0, "landing": 0, "n_landing": [],
                             "covered": {}, "cand_size": {}, "cand_hits": {}, "cand_total": {},
                             "rec": {"start": [0, 0], "landing in past peripheral": [0, 0],
                                     "landing outside past peripheral": [0, 0]}})
    a["samples"] += len(landing)
    nl = landing.sum(1)
    a["with_landing"] += int((nl > 0).sum())
    a["landing"] += int(nl.sum())
    a["n_landing"] += nl[nl > 0].tolist()
    for name, r in regs.items():
        a["covered"][name] = a["covered"].get(name, 0) + int((landing & r).sum())
    for name in CANDIDATES:
        cand = regs[name] & regs["past scene"]
        size = cand.sum(1)
        a["cand_size"].setdefault(name, []).extend(size[nl > 0].tolist())
        a["cand_hits"][name] = a["cand_hits"].get(name, 0) + int((landing & cand).sum())
        a["cand_total"][name] = a["cand_total"].get(name, 0) + int(size[nl > 0].sum())
    if pred is not None:
        inside = landing & regs["past peripheral"]
        for name, m in (("start", start), ("landing in past peripheral", inside),
                        ("landing outside past peripheral", landing & ~regs["past peripheral"])):
            a["rec"][name][0] += int((pred & m).sum())
            a["rec"][name][1] += int(m.sum())


def summarise(acc):
    out = {}
    for key, a in acc.items():
        s = {"samples": a["samples"], "with_landing": a["with_landing"] / max(a["samples"], 1),
             "median_landing_voxels": float(np.median(a["n_landing"])) if a["n_landing"] else float("nan"),
             "covered": {k: v / max(a["landing"], 1) for k, v in a["covered"].items()},
             "candidates_median": {k: float(np.median(v)) if v else float("nan") for k, v in a["cand_size"].items()},
             "candidate_density": {k: a["cand_hits"][k] / max(a["cand_total"][k], 1) for k in a["cand_hits"]},
             "model_recall": {k: (h / t if t else float("nan")) for k, (h, t) in a["rec"].items()}}
        out[key] = s
    return out


def check(bench, idx, shift=None, probs=None, horizon=None, chunk=512):
    """-> summary dict keyed by shift group ('all' always)."""
    R, cube = bench.R, bench.meta["lift"]["cube_size"]
    acc = {}
    groups = [("all", None)]
    if shift is not None:
        e = shift_breakdown.EDGES
        groups += [(f"{lo}-{hi} deg", (lo, hi)) for lo, hi in zip(e[:-1], e[1:])]
    for s in range(0, len(idx), chunk):
        j = idx[s:s + chunk]
        past = (bench.inputs(j) > 0).any(1)                                    # (n, C, R, R, R)
        fut = bench.target(j, horizon, ["foveal"])[:, 0].reshape(len(j), -1)
        was = past[:, F].reshape(len(j), -1)
        landing, start = fut & ~was, fut & was
        regs = regions(past, R, bench.vertical_axis, cube)
        pred = None
        if probs is not None:
            pred = (probs[s:s + chunk].astype(np.float32) > 0.5).reshape(len(j), -1)
        for key, rng in groups:
            m = np.ones(len(j), bool) if rng is None else (shift[s:s + chunk] >= rng[0]) & (shift[s:s + chunk] < rng[1])
            if m.any():
                accumulate(acc, key, landing[m], start[m], {k: v[m] for k, v in regs.items()},
                           None if pred is None else pred[m])
    return summarise(acc)


def report(summary, with_model=False):
    keys = list(summary)
    w = 14
    line = lambda label, vals, fmt: print(f"{label:>34s} " + " ".join(fmt(v) for v in vals))   # noqa: E731
    print("\n== landing voxels: future foveal voxels that were not foveal in the past window")
    line("shift group", keys, lambda v: f"{v:>{w}s}")
    line("samples", [summary[k]["samples"] for k in keys], lambda v: f"{v:>{w}d}")
    line("with landing voxels", [summary[k]["with_landing"] for k in keys], lambda v: f"{v:>{w}.1%}")
    line("median landing voxels", [summary[k]["median_landing_voxels"] for k in keys], lambda v: f"{v:>{w}.0f}")
    print("\n-- share of landing voxels inside a past region (was the destination in the input?)")
    for r in REGIONS:
        if r in summary[keys[0]]["covered"]:
            line(r, [summary[k]["covered"][r] for k in keys], lambda v: f"{v:>{w}.1%}")
    print("\n-- candidate surfaces (region and past scene): median voxels | chance hit rate per guessed voxel")
    for r in CANDIDATES:
        line(f"{r}: size", [summary[k]["candidates_median"][r] for k in keys], lambda v: f"{v:>{w}.0f}")
        line(f"{r}: chance", [summary[k]["candidate_density"][r] for k in keys], lambda v: f"{v:>{w}.1%}")
    if with_model:
        print("\n-- model: recall of the predicted foveal span")
        for r in ("start", "landing in past peripheral", "landing outside past peripheral"):
            line(r, [summary[k]["model_recall"][r] for k in keys], lambda v: f"{v:>{w}.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="trained run: its benchmark and test split, plus its predictions")
    ap.add_argument("--bench", default=None, help="packed benchmark (when no --run)")
    ap.add_argument("--part", default="test", help="test / val / train, or 'all' (no --run)")
    ap.add_argument("--fold", type=int, default=None, help="cross-validation fold (no --run)")
    ap.add_argument("--lab-src", default=None, help="lab benchmark folder with g_now / g_fut (converted lab data)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    probs, horizon = None, None
    if a.run:
        _, cfg = load_run(a.run, "cpu")
        bench = PackedBenchmark(cfg.data.root)
        idx, horizon = bench.indices("test", cfg.data.fold), cfg.data.horizon
        if cfg.data.test_categories:
            idx = idx[np.isin(bench.arrays["category"][idx], list(cfg.data.test_categories))]
        all_probs = np.load(os.path.join(a.run, "probs_test.npy"))
        if len(all_probs) != len(idx):
            raise SystemExit(f"probs_test.npy has {len(all_probs)} rows, the test split {len(idx)}")
        probs = all_probs[:, list(cfg.data.levels).index("foveal")]
    else:
        if not a.bench:
            raise SystemExit("pass --run or --bench")
        bench = PackedBenchmark(a.bench)
        idx = np.arange(len(bench.arrays["X"])) if a.part == "all" else bench.indices(a.part, a.fold)
    if bench.grid is not None:
        raise SystemExit("landing_check works on Cartesian benchmarks (the past regions are R^3 grids)")
    shift = None
    if a.lab_src:
        shift = lab_shift(bench, idx, a.lab_src)
    elif "gaze_grid" in bench.arrays:
        shift = shift_breakdown.gaze_shift(bench, idx)
    else:
        print("no gaze available (pass --lab-src for the converted lab data): reporting all samples only")
    summary = check(bench, idx, shift, probs, horizon)
    report(summary, probs is not None)
    out = a.out or os.path.join(a.run or bench.root, "landing_check.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
