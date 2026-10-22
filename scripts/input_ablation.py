"""Which inputs does a trained model rely on? Re-score the test split with one input removed at a time.

    python scripts/input_ablation.py --run runs/fovs_aria_lab/cv/front_vid_f4 \
        --lab-src /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria          # optional: also split by future gaze shift

No retraining: the trained model sees the test inputs with one part replaced, in two ways
    zero      the input set to 0 (simple, but an all-empty grid may be unlike anything seen in training)
    shuffle   the input taken from a different test sample (realistic values, wrong sample): the fairer test
Inputs: each past span channel (orientation, peripheral, central, foveal), scene, the gaze ray (if the run has it),
all four spans together, the camera features (if the run uses them; the in-view mask is the same for every sample and
is kept) and the gaze path (if the run has the gaze branch). A span channel is removed in every past grid (and so in
the union grid the model builds from them). Scores: threshold 0.5 on the full 16^3 grid, as compare_variants.py.
A large drop means the model relies on that input; ~0 means it ignores it or gets the same from other inputs.
Writes <run>/input_ablation.json.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.engine import load_run, make_datasets, predict, resolve_device   # noqa: E402
from egospan.lift import CHANNELS, LEVELS                                      # noqa: E402
from egospan.metrics import overlap_scores                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("shift_breakdown", os.path.join(HERE, "shift_breakdown.py"))
shift_breakdown = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shift_breakdown)


def derangement(n, seed=0):
    """random permutation with no fixed point: sample i gets the input of a different sample."""
    if n < 2:
        raise ValueError("shuffle needs at least 2 test samples")
    p = np.random.default_rng(seed).permutation(n)
    out = np.empty(n, int)
    out[p] = np.roll(p, 1)
    return out


class Ablated(Dataset):
    """base dataset with some inputs zeroed or swapped in from another sample.
    channels: input channel indices of x (all past grids); video: drop the camera features; gaze: drop the gaze path."""

    def __init__(self, base, channels=(), video=False, gaze=False, mode="zero", seed=0):
        self.base, self.channels, self.video, self.gaze, self.mode = base, list(channels), video, gaze, mode
        self.perm = derangement(len(base), seed) if mode == "shuffle" else None
        self.to_eval = getattr(base, "to_eval", None)                 # spherical runs: cells -> R^3 voxels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        other = self.base[int(self.perm[i])] if self.mode == "shuffle" else None
        if self.channels:
            item["x"][:, self.channels] = 0 if other is None else other["x"][:, self.channels]
        if self.video:
            d = self.base.video_dim                     # feature channels; the last channel (in-view mask) is kept
            item["v"][:d] = 0 if other is None else other["v"][:d]
        if self.gaze:
            item["g"] = torch.zeros_like(item["g"]) if other is None else other["g"]
        return item


def ablations(ds, in_channels):
    """name -> dict(channels, video, gaze) for every input this run has."""
    out = {name: dict(channels=[CHANNELS.index(name)]) for name in CHANNELS}
    spans = [CHANNELS.index(l) for l in LEVELS]
    if in_channels > len(CHANNELS):
        ray = len(CHANNELS)                             # the gaze-ray channel comes after the 5 span channels
        out["gaze ray"] = dict(channels=[ray])
        spans = spans + [ray]
    out["all 4 spans"] = dict(channels=spans)
    if ds.video_dim:
        out["camera"] = dict(video=True)
    if ds.cue_dim:
        out["gaze path"] = dict(gaze=True)
    return out


def scores(prob, target, levels, threshold=0.5):
    """(n, L) per-sample IoU."""
    return overlap_scores(prob.astype(np.float32) > threshold, target)["iou"]


def run(run_dir, modes=("shuffle", "zero"), only=None, device="auto", batch_size=32, num_workers=0, seed=0,
        lab_src=None, progress=True):
    device = resolve_device(device)
    model, cfg = load_run(run_dir, device)
    bench, ds = make_datasets(cfg)
    test, levels = ds["test"], list(cfg.data.levels)
    todo = ablations(test, bench.in_channels)
    if only:
        unknown = set(only) - set(todo)
        if unknown:
            raise SystemExit(f"unknown inputs {sorted(unknown)}; this run has {list(todo)}")
        todo = {k: v for k, v in todo.items() if k in only}
    shift = shift_breakdown.gaze_shift(bench, test.idx, lab_src) if lab_src else None
    if shift is None and "gaze_grid" in bench.arrays:
        shift = shift_breakdown.gaze_shift(bench, test.idx)

    def score(dataset, desc):
        p, y = predict(model, dataset, device, batch_size, cfg.train.amp, num_workers, desc=desc, progress=progress)
        return scores(p, y, levels)

    res = {"intact": score(test, "intact")}
    for mode in modes:
        for name, kw in todo.items():
            res[f"{name} [{mode}]"] = score(Ablated(test, mode=mode, seed=seed, **kw), f"{name} [{mode}]")
    return res, levels, shift


def report(res, levels, shift=None):
    base = np.nanmean(res["intact"], 0)
    head = f"{'input removed':>26s} " + " ".join(f"{l[:11]:>11s}" for l in levels) + f" {'MEAN':>8s}"
    print("\n== test IoU with one input removed (change vs intact in brackets)")
    print(head)
    for name, iou in res.items():
        m = np.nanmean(iou, 0)
        if name == "intact":
            cells = " ".join(f"{v:11.4f}" for v in m) + f" {np.nanmean(m):8.4f}"
        else:
            cells = " ".join(f"{v - b:+11.4f}" for v, b in zip(m, base)) + f" {np.nanmean(m) - np.nanmean(base):+8.4f}"
        print(f"{name:>26s} {cells}")
    out = {"levels": levels, "mean_iou": {k: np.nanmean(v, 0).tolist() for k, v in res.items()}}
    if shift is not None:
        edges = shift_breakdown.EDGES
        groups = [(lo, hi, (shift >= lo) & (shift < hi)) for lo, hi in zip(edges[:-1], edges[1:])]
        out["by_shift"] = {}
        for lev in [l for l in ("central", "foveal") if l in levels]:
            c = levels.index(lev)
            print(f"\n== {lev} IoU by largest future gaze shift: intact, then change with the input removed")
            print(f"{'input removed':>26s} " + " ".join(f"{f'{lo}-{hi} deg':>11s}" for lo, hi, _ in groups))
            ref = [np.nanmean(res["intact"][m, c]) for _, _, m in groups]
            out["by_shift"][lev] = {}
            for name, iou in res.items():
                vals = [np.nanmean(iou[m, c]) for _, _, m in groups]
                out["by_shift"][lev][name] = vals
                if name == "intact":
                    cells = " ".join(f"{v:11.4f}" for v in vals)
                else:
                    cells = " ".join(f"{v - r:+11.4f}" for v, r in zip(vals, ref))
                print(f"{name:>26s} {cells}")
            print(f"{'samples':>26s} " + " ".join(f"{m.mean():11.1%}" for _, _, m in groups))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--modes", nargs="+", default=["shuffle", "zero"], choices=["shuffle", "zero"])
    ap.add_argument("--inputs", nargs="+", default=None, help="only these inputs (names as printed)")
    ap.add_argument("--lab-src", default=None, help="lab benchmark folder (test.npz with g_now, g_fut) for the "
                                                    "split by future gaze shift; only for the lab's test split")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res, levels, shift = run(a.run, a.modes, a.inputs, a.device, a.batch_size, a.num_workers, a.seed, a.lab_src)
    out = report(res, levels, shift)
    path = os.path.join(a.run, "input_ablation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
