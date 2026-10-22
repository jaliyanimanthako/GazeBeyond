"""Save the future scene points of a benchmark's samples, for point-level scoring (scripts/point_score.py).

    python scripts/build_future_points.py --bench data/fovs_aria_sph --part test --fold 4 \
        --slam-root /data/SS/Aria_data/mps_slam --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf

Re-reads the raw recordings and, for every selected sample, collects the points selected at any step of its forecast
window (inside the paper's cube, in the sample's grid frame) with their span bits, using the benchmark's own lift and
window settings (from its meta), so they reproduce its targets exactly. Writes <bench>/points_<part>[_f<fold>].npz:
    key (n,)       "<recording>#<position>" (matches the samples of any benchmark built from the same recordings)
    off (n + 1,)   points of sample i: xyz[off[i]:off[i + 1]]
    xyz (M, 3)     float32, m        bits (M,) uint8, bit c = span level c (orientation, peripheral, central, foveal)
    past1, past2   (M,) uint8: the point's span bits in the last observed second / the whole observed window, bit 4 =
                   it was observed then at all (in the map at prediction time; 0 = first seen in the future)
    seen_ever (M,) the point was observed at any step before the anchor (the whole SLAM history, not just the window)
    shift (n,)     largest future gaze shift (deg)
--synthetic rebuilds the recordings of `build_dataset.py synthetic` instead of reading raw files (tests).
"""
import argparse
import json
import os
import re
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.config import _from_dict                                        # noqa: E402
from egospan.data.build import WindowConfig, future_points, sample_keys     # noqa: E402
from egospan.data.dataset import PackedBenchmark                            # noqa: E402
from egospan.lift import LiftConfig                                          # noqa: E402


def read(name, a):
    if a["synthetic"]:
        from egospan.data.synthetic import make_recording
        m = re.match(r"synth_loc(\d+)_(\d+)$", name)
        return make_recording(name=name, seed=int(m.group(2)), location=int(m.group(1)), seconds=40)
    from egospan.data.aria import read_aea_server
    T = np.array(json.load(open(os.path.join(a["cpf_root"], f"{name}.json")))["T_device_cpf"])
    return read_aea_server(a["slam_root"], a["gaze_root"], name, T)


def one(job, a, lift, win):
    name, t0 = job
    pts, shift = future_points(read(name, a), lift, win, t0)
    return name, pts, shift


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--part", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--slam-root")
    ap.add_argument("--gaze-root")
    ap.add_argument("--cpf-root")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    b = PackedBenchmark(a.bench)
    if "t0_us" not in b.arrays:
        raise SystemExit(f"{a.bench} has no anchor times (converted lab data?): use one of our rebuilds")
    lift = _from_dict(LiftConfig, b.meta["lift"])
    win = _from_dict(WindowConfig, b.meta["window"])
    idx = b.indices(a.part, a.fold)
    rec, t0 = b.arrays["rec"][idx].astype(str), b.arrays["t0_us"][idx]
    keys = sample_keys(b.arrays["rec"])[idx]
    jobs = [(r, t0[rec == r]) for r in sorted(set(rec))]
    fn = partial(one, a=vars(a), lift=lift, win=win)
    got = {}
    if a.workers > 1:
        with Pool(a.workers) as pool:
            for name, pts, shift in pool.imap_unordered(fn, jobs):
                got[name] = (pts, shift)
                print(f"{name}: {len(pts)} samples, {sum(len(x[0]) for x in pts)} points", flush=True)
    else:
        for job in jobs:
            name, pts, shift = fn(job)
            got[name] = (pts, shift)
            print(f"{name}: {len(pts)} samples, {sum(len(x[0]) for x in pts)} points", flush=True)
    xyz, bits, past1, past2, ever, shifts, counts = [], [], [], [], [], [], []
    pos = {r: 0 for r in got}
    for r in rec:                                             # back to the benchmark's sample order
        (x, bt, p1, p2, ev), sh = got[r][0][pos[r]], got[r][1][pos[r]]
        pos[r] += 1
        xyz.append(x)
        bits.append(bt)
        past1.append(p1)
        past2.append(p2)
        ever.append(ev)
        shifts.append(sh)
        counts.append(len(x))
    tag = a.part + (f"_f{a.fold}" if a.fold is not None else "")
    out = a.out or os.path.join(a.bench, f"points_{tag}.npz")
    np.savez(out, key=keys, off=np.concatenate([[0], np.cumsum(counts)]), xyz=np.concatenate(xyz),
             bits=np.concatenate(bits), past1=np.concatenate(past1), past2=np.concatenate(past2),
             seen_ever=np.concatenate(ever), shift=np.asarray(shifts))
    print(f"wrote {out}: {len(keys)} samples, {sum(counts)} points (median {int(np.median(counts))} per sample)")


if __name__ == "__main__":
    main()
