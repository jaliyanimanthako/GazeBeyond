"""Which point crop did the lab's built FoVS-Aria use? Tests several crop definitions on one recording in one run.

    python scripts/crop_hypotheses.py --rec AriaEverydayActivities_1.0.0_loc1_script1_seq5_rec1 \
        --lab /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria \
        --slam-root /data/SS/Aria_data/mps_slam --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf

The scene channel of the input grids depends only on which visible points are kept, so it isolates the crop. Every
hypothesis keeps the lab's anchor cube (|anchor-CPF coordinate| < D/2 on every axis) and the lab's point filters
(dist_std < 0.1 + one outlier pass), then adds one extra per-step or anchor test. The one with ~0 % difference is it.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.aria import read_aea_server                     # noqa: E402
from egospan.data.build import unpack_grids                       # noqa: E402
from egospan.lift import statistical_outlier_mask                 # noqa: E402

R, D = 16, 3.2
H = D / 2


def extra_tests(P, t_k, R_k, t_a, R_a):
    """name -> bool mask over points P (world) seen at step k, for anchor pose (t_a, R_a)."""
    lk, la = (P - t_k) @ R_k, (P - t_a) @ R_a
    return {
        "anchor cube only (lab code in git)": np.ones(len(P), bool),
        "+ step cube, step CPF axes": (np.abs(lk) < H).all(1),
        "+ step cube, anchor CPF axes": (np.abs((P - t_k) @ R_a) < H).all(1),
        "+ step cube, world axes": (np.abs(P - t_k) < H).all(1),
        "+ step sphere r=D/2": np.linalg.norm(P - t_k, axis=1) < H,
        "+ step sphere r=D/2*sqrt3": np.linalg.norm(P - t_k, axis=1) < H * np.sqrt(3),
        "+ step L1 |x|+|y|+|z| < D/2": np.abs(lk).sum(1) < H,
        "+ anchor sphere r=D/2": np.linalg.norm(la, axis=1) < H,
        "+ step depth z in (0, D/2)": (lk[:, 2] > 0) & (lk[:, 2] < H),
        "+ step depth |z| < D/2": np.abs(lk[:, 2]) < H,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("rec", "lab", "slam-root", "gaze-root", "cpf-root"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    T = np.array(json.load(open(os.path.join(a.cpf_root, f"{a.rec}.json")))["T_device_cpf"])
    s = read_aea_server(a.slam_root, a.gaze_root, a.rec, T)
    good = s.points_dist_std < 0.1
    gi = np.flatnonzero(good)
    good[gi[~statistical_outlier_mask(s.points_world[gi], 16, 2.0)]] = False
    lab = np.load(os.path.join(a.lab, "recs", f"{a.rec}.npz"))
    Xl = unpack_grids(lab["X"], (2, 5, R, R, R))[:, :, 4]                      # lab scene channel (n, 2, R, R, R)
    t_idx = np.searchsorted(s.t_us, lab["t0_us"])
    assert (s.t_us[t_idx] == lab["t0_us"]).all()
    names = list(extra_tests(np.zeros((1, 3)), np.zeros(3), np.eye(3), np.zeros(3), np.eye(3)))
    ours = {n: np.zeros_like(Xl) for n in names}
    for i, t0 in enumerate(t_idx):
        t_a, R_a = s.T_world_cpf[t0, :3, 3], s.T_world_cpf[t0, :3, :3]
        for rel, k in enumerate(range(t0 - 19, t0 + 1)):
            vis = s.visible(k)
            P = s.points_world[vis[good[vis]]].astype(np.float64)
            la = (P - t_a) @ R_a
            base = (np.abs(la) < H).all(1)
            ijk = np.clip(((la + H) / (D / R)).astype(int), 0, R - 1)
            for n, m in extra_tests(P, s.T_world_cpf[k, :3, 3], s.T_world_cpf[k, :3, :3], t_a, R_a).items():
                sel = base & m
                ours[n][i, rel // 10, ijk[sel, 0], ijk[sel, 1], ijk[sel, 2]] = True
    print(f"{'crop hypothesis':40s} {'ours':>8s} {'lab':>8s} {'only ours':>10s} {'only lab':>9s} {'differ':>8s}")
    for n in names:
        o = ours[n]
        print(f"{n:40s} {o.sum():8d} {Xl.sum():8d} {(o & ~Xl).sum():10d} {(~o & Xl).sum():9d} "
              f"{(o != Xl).sum() / max((o | Xl).sum(), 1):8.3%}")


if __name__ == "__main__":
    main()
