"""Recover each AEA recording's T_device_cpf (device frame -> central pupil frame) from the lab's built benchmark,
for raw copies that have the MPS SLAM zips but not the .vrs calibration (the lab server's /data/SS/Aria_data).

The lab stored, per sample, the CPF pose at the anchor (R_now, c_now = T_world_cpf) and its gaze timestamp (t0_us).
The closed-loop trajectory gives T_world_device at the same timestamp (nearest pose within 5 ms, as both builders do),
so T_device_cpf = T_world_device^-1 T_world_cpf. It is constant per recording; the spread over a recording's samples
is printed as a check (expect ~0). Writes <out>/<rec>.json (4x4) for scripts/build_dataset.py aea-server.

    python scripts/derive_device_cpf.py --lab /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria \
        --slam-root /data/SS/Aria_data/mps_slam --out data/aea_cpf --workers 8
"""
import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.aria import POSE_TOL_US, _read_csv     # noqa: E402
from egospan.data.streams import quat_to_rotmat           # noqa: E402

POSE_COLS = ["tracking_timestamp_us", "tx_world_device", "ty_world_device", "tz_world_device",
             "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]


def derive(job, slam_root, out):
    rec, t0, R_now, c_now = job
    try:
        Tr = _read_csv((os.path.join(slam_root, f"{rec}_mps_slam_trajectories.zip"), "closed_loop_trajectory.csv"),
                       usecols=POSE_COLS)
        tt = Tr["tracking_timestamp_us"].to_numpy(np.int64)
        j = np.clip(np.searchsorted(tt, t0), 1, len(tt) - 1)
        j -= (t0 - tt[j - 1]) < (tt[j] - t0)
        ok = np.abs(tt[j] - t0) < POSE_TOL_US
        if not ok.any():
            return rec, None, "no trajectory pose within 5 ms of the lab's anchors"
        P = Tr[POSE_COLS[1:]].to_numpy(float)[j[ok]]
        T_wd = np.tile(np.eye(4), (ok.sum(), 1, 1))
        T_wd[:, :3, :3], T_wd[:, :3, 3] = quat_to_rotmat(P[:, 3:]), P[:, :3]
        T_wc = np.tile(np.eye(4), (ok.sum(), 1, 1))
        T_wc[:, :3, :3], T_wc[:, :3, 3] = R_now[ok], c_now[ok]
        T_dc = np.linalg.inv(T_wd) @ T_wc                                    # (n, 4, 4), should all be equal
        U, _, Vt = np.linalg.svd(T_dc[:, :3, :3].sum(0))
        Rm = U @ Vt                                                          # mean rotation
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rm, T_dc[:, :3, 3].mean(0)
        rot_dev = np.degrees(np.arccos(np.clip((np.trace(Rm.T @ T_dc[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
        trans_dev = 1000 * np.linalg.norm(T_dc[:, :3, 3] - T[:3, 3], axis=1)
        stats = dict(samples=int(ok.sum()), rot_dev_deg_max=float(rot_dev.max()),
                     trans_dev_mm_max=float(trans_dev.max()))
        with open(os.path.join(out, f"{rec}.json"), "w") as f:
            json.dump({"T_device_cpf": T.tolist(), **stats}, f, indent=1)
        return rec, T, stats
    except Exception as e:                                                    # noqa: BLE001
        return rec, None, f"FAILED {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab", required=True, help="lab benchmark folder with {train,val,test}.npz")
    ap.add_argument("--slam-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Z = [np.load(os.path.join(a.lab, f"{p}.npz")) for p in ("train", "val", "test")]
    rec = np.concatenate([z["rec"] for z in Z]).astype(str)
    t0 = np.concatenate([z["t0_us"] for z in Z]).astype(np.int64)
    R_now, c_now = np.concatenate([z["R_now"] for z in Z]), np.concatenate([z["c_now"] for z in Z])
    jobs = [(r, t0[rec == r], R_now[rec == r], c_now[rec == r]) for r in sorted(set(rec))]
    fn = partial(derive, slam_root=a.slam_root, out=a.out)
    Ts, bad = [], []
    with Pool(a.workers) as pool:
        for r, T, info in pool.imap_unordered(fn, jobs):
            if T is None:
                bad.append(r)
                print(f"{r}: {info}", flush=True)
            else:
                Ts.append(T)
                if info["rot_dev_deg_max"] > 0.01 or info["trans_dev_mm_max"] > 0.1:
                    print(f"{r}: WARNING not constant over the recording: {info}", flush=True)
    Ts = np.stack(Ts)
    print(f"\nderived {len(Ts)} / {len(jobs)} recordings -> {a.out}  (failed: {len(bad)})")
    off = np.linalg.norm(Ts[:, :3, 3] - Ts[:, :3, 3].mean(0), axis=1) * 1000
    print(f"across recordings (different glasses / calibrations): translation spread up to {off.max():.1f} mm")


if __name__ == "__main__":
    main()
