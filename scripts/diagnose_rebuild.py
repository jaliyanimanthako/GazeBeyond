"""Find where a rebuild differs from the lab's build, for one recording.

    python scripts/diagnose_rebuild.py --rec AriaEverydayActivities_1.0.0_loc1_script1_seq5_rec1 \
        --ours data/fovs_aria_rebuild --lab /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria \
        --slam-root /data/SS/Aria_data/mps_slam --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf

Prints, in order: the derived calibration's consistency; per-channel voxel disagreement between the two builds (scene
identical but spans different -> gaze / cone; scene different too -> points / visibility / pose); the number of points
left after the lab's filters vs the lab's extract logs; and anchor poses and gaze against the lab's stored values.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.aria import read_aea_server            # noqa: E402
from egospan.data.build import unpack_grids              # noqa: E402
from egospan.lift import CHANNELS, LEVELS, statistical_outlier_mask   # noqa: E402

R = 16


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("rec", "ours", "lab", "slam-root", "gaze-root", "cpf-root"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    rec = a.rec

    print("== 1. derived calibration")
    cj = json.load(open(os.path.join(a.cpf_root, f"{rec}.json")))
    print(f"   samples {cj['samples']}, max rotation deviation {cj['rot_dev_deg_max']:.6f} deg, "
          f"max translation deviation {cj['trans_dev_mm_max']:.6f} mm   (both ~0 if the SLAM files are the lab's)")

    print("\n== 2. voxel disagreement per channel (fraction of occupied voxels that differ)")
    o, lab = np.load(os.path.join(a.ours, "recs", f"{rec}.npz")), np.load(os.path.join(a.lab, "recs", f"{rec}.npz"))
    _, io, il = np.intersect1d(o["t0_us"], lab["t0_us"], return_indices=True)
    Xo = unpack_grids(o["X"][io], (2, 5, R, R, R))
    Xl = unpack_grids(lab["X"][il], (2, 5, R, R, R))
    for c, name in enumerate(CHANNELS):
        a_, b_ = Xo[:, :, c], Xl[:, :, c]
        print(f"   {name:12s} ours {a_.sum():8d}  lab {b_.sum():8d}  only ours {(a_ & ~b_).sum():7d}  "
              f"only lab {(~a_ & b_).sum():7d}  differ {(a_ != b_).sum() / max((a_ | b_).sum(), 1):7.3%}")
    Yo = unpack_grids(o["YF"][io], (2, 4, R, R, R)).any(1)
    Yl = unpack_grids(lab["Y"][il], (4, R, R, R))
    for c, name in enumerate(LEVELS):
        print(f"   target {name:12s} differ {(Yo[:, c] != Yl[:, c]).sum() / max((Yo[:, c] | Yl[:, c]).sum(), 1):7.3%}")

    print("\n== 3. points after the lab's filters (dist_std < 0.1, one outlier pass, 16 neighbours)")
    T = np.array(cj["T_device_cpf"])
    s = read_aea_server(a.slam_root, a.gaze_root, rec, T)
    good = s.points_dist_std < 0.1
    gi = np.flatnonzero(good)
    good[gi[~statistical_outlier_mask(s.points_world[gi], 16, 2.0)]] = False
    print(f"   total {len(s.points_world)}, dist_std < 0.1: {len(gi)}, after outlier removal: {good.sum()}")
    for logf in sorted(glob.glob(os.path.join(a.lab, "extract*.log"))):
        for ln in open(logf):
            if rec[-28:] in ln:
                print(f"   lab {os.path.basename(logf)}: {ln.strip()}")
    vis = np.array([good[s.visible(k)].sum() for k in range(len(s))])
    if "npts" in lab.files:
        t_idx = np.searchsorted(s.t_us, lab["t0_us"][il])
        ours_npts = np.array([vis[t - 19:t + 21].mean() for t in t_idx])
        print(f"   mean visible points per step in each sample window: ours {ours_npts.mean():.1f}, "
              f"lab {lab['npts'][il].mean():.1f}  (max abs diff {np.abs(ours_npts - lab['npts'][il]).max():.2f})")

    print("\n== 4. anchor pose and gaze vs the lab's stored values")
    t_idx = np.searchsorted(s.t_us, lab["t0_us"][il])
    if "R_now" in lab.files:
        dR = np.einsum("nji,njk->nik", lab["R_now"][il], s.T_world_cpf[t_idx, :3, :3])
        ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        dt = 1000 * np.linalg.norm(lab["c_now"][il] - s.T_world_cpf[t_idx, :3, 3], axis=1)
        print(f"   CPF rotation diff max {ang.max():.6f} deg, position diff max {dt.max():.6f} mm")
    if "g_now" in lab.files:
        gw = np.einsum("nij,nj->ni", s.T_world_cpf[t_idx, :3, :3], s.gaze_cpf[t_idx])
        gl = lab["g_now"][il] / np.linalg.norm(lab["g_now"][il], axis=1, keepdims=True)
        print(f"   gaze direction diff max {np.degrees(np.arccos(np.clip((gw * gl).sum(1), -1, 1))).max():.6f} deg")
    print(f"   anchor valid in ours: {s.valid[t_idx].mean():.1%}")


if __name__ == "__main__":
    main()
