"""Check a rebuild against the lab's built FoVS-Aria, recording by recording and sample by sample.

    python scripts/validate_against_lab.py --ours data/fovs_aria_rebuild \
        --lab /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria

Compares <ours>/recs/<rec>.npz (built with configs/fovs_aria_lab_rebuild.yaml, default 2 x 1 s binary inputs) with
<lab>/recs/<rec>.npz: the anchor timestamps, and for every common anchor the input grids X and the 2 s target union.
With identical settings every sample should match exactly; a small share of differing voxels points at boundary
effects (observation timestamps exactly on a 100 ms bin edge), a large one at a setup problem.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.build import unpack_grids   # noqa: E402

R = 16


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help="our build folder (has recs/)")
    ap.add_argument("--lab", required=True, help="the lab's benchmark folder (has recs/)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N recordings")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.ours, "recs", "*.npz")))[:a.limit]
    if not files:
        raise SystemExit(f"no recordings in {a.ours}/recs")
    tot = dict(ours=0, lab=0, common=0, x_exact=0, y_exact=0, x_vox_diff=0, x_vox=0, y_vox_diff=0, y_vox=0)
    print(f"{'recording':55s} {'ours':>5s} {'lab':>5s} {'common':>6s} {'X exact':>8s} {'Y exact':>8s}")
    for f in files:
        rec = os.path.basename(f)[:-4]
        lf = os.path.join(a.lab, "recs", f"{rec}.npz")
        if not os.path.isfile(lf):
            print(f"{rec:55s} missing in the lab's recs/")
            continue
        o, lab = np.load(f), np.load(lf)
        w = json.loads(str(o["meta"]))["window"]
        if (w.get("time_bins") or w["past_units"]) != 2 or w.get("soft") or w.get("ray_channel"):
            raise SystemExit("compare a build with the default inputs (2 x 1 s binary, no ray channel)")
        to, tl = o["t0_us"], lab["t0_us"]
        common, io, il = np.intersect1d(to, tl, return_indices=True)
        Xo = unpack_grids(o["X"][io], (2, 5, R, R, R))
        Xl = unpack_grids(lab["X"][il], (2, 5, R, R, R))
        Yo = unpack_grids(o["YF"][io], (2, 4, R, R, R)).any(1)
        Yl = unpack_grids(lab["Y"][il], (4, R, R, R))
        xe = (Xo == Xl).reshape(len(common), -1).all(1)
        ye = (Yo == Yl).reshape(len(common), -1).all(1)
        print(f"{rec[-50:]:55s} {len(to):5d} {len(tl):5d} {len(common):6d} {xe.mean():8.1%} {ye.mean():8.1%}",
              flush=True)
        for key, v in (("ours", len(to)), ("lab", len(tl)), ("common", len(common)), ("x_exact", xe.sum()),
                       ("y_exact", ye.sum()), ("x_vox_diff", (Xo != Xl).sum()), ("x_vox", (Xo | Xl).sum()),
                       ("y_vox_diff", (Yo != Yl).sum()), ("y_vox", (Yo | Yl).sum())):
            tot[key] += int(v)
    c = max(tot["common"], 1)
    print(f"\nsamples: ours {tot['ours']}, lab {tot['lab']}, common anchors {tot['common']}")
    print(f"identical samples: X {tot['x_exact'] / c:.2%}, target {tot['y_exact'] / c:.2%}")
    print(f"differing voxels among occupied ones: X {tot['x_vox_diff'] / max(tot['x_vox'], 1):.3%}, "
          f"target {tot['y_vox_diff'] / max(tot['y_vox'], 1):.3%}")


if __name__ == "__main__":
    main()
