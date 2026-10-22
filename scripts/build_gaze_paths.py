"""Past gaze / head-motion path for every sample of a converted lab benchmark (input of the gaze branch).

For each sample (recording, anchor time t0) of the lab's {train,val,test}.npz, in the converter's order:
    the 20 gaze steps up to and including t0 (10 Hz, MPS general_eye_gaze.csv) and the head rotation at each step
    (the lab's per-recording head rotations, data/spatial_audio_pilot/<rec>.npz, keys t [s] and R [3x3]).
Per step features (11):
    0:3   gaze direction in the anchor CPF frame (= the grid frame)       uses the head rotation
    3:6   gaze direction in the CPF at that step (eye-in-head)
    6:9   head rotation from that step to the anchor, as a rotation vector (rad)
    9     gaze valid        10   head valid
If the head rotation of a step is missing, 0:3 falls back to the eye-in-head direction and 6:9 is zero.

The head-rotation convention is checked against the lab's SLAM future gaze (g_fut, R_now): the convention with the
smaller angular error is used and its error is printed. Writes <out>/gaze_all.npz aligned with <out>/all.npz.

    python scripts/build_gaze_paths.py --src /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria \
        --gaze-root /data/SS/Aria_data/gaze \
        --head-root /data/SS/SS/Dynamic_Fusion_ALFA/data/spatial_audio_pilot --out data/fovs_aria_lab
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

PARTS = ("train", "val", "test")
STEPS, MAX_GAP_US, HEAD_TOL_S = 20, 150_000, 0.06
LAYOUT = {"vectors": [[0, 3], [3, 6]], "pseudovectors": [[6, 9]], "n_features": 11}


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def yaw_pitch_dir(yaw, pitch):
    return unit(np.stack([np.tan(yaw), np.tan(pitch), np.ones_like(yaw)], -1))


def rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


class Recording:
    """Gaze rows and head rotations of one recording, looked up by gaze timestamp."""

    def __init__(self, rec, gaze_root, head_root):
        G = pd.read_csv(os.path.join(gaze_root, rec, "mps", "eye_gaze", "general_eye_gaze.csv"))
        self.ts = G["tracking_timestamp_us"].to_numpy(np.int64)
        yaw = G["yaw_rads_cpf"].to_numpy() if "yaw_rads_cpf" in G else \
            0.5 * (G["left_yaw_rads_cpf"].to_numpy() + G["right_yaw_rads_cpf"].to_numpy())
        self.g = yaw_pitch_dir(yaw, G["pitch_rads_cpf"].to_numpy())
        hp = os.path.join(head_root, f"{rec}.npz") if head_root else None
        if hp and os.path.isfile(hp):
            H = np.load(hp)
            self.ht, self.hR = H["t"].astype(np.float64), H["R"].astype(np.float64)
        else:
            self.ht = self.hR = None

    def rows(self, t0_us, offsets):
        """row indices for anchor-relative step offsets (-1 if missing), or None if the anchor is not in the CSV."""
        k = np.searchsorted(self.ts, t0_us)
        if k >= len(self.ts) or self.ts[k] != t0_us:
            return None
        r = k + offsets
        ok = (r >= 0) & (r < len(self.ts))
        r = np.where(ok, r, -1)
        rt = self.ts[np.clip(r, 0, None)]
        ok &= np.abs(rt - (t0_us + offsets * 100_000)) < MAX_GAP_US       # contiguous 10 Hz around the anchor
        return np.where(ok, r, -1)

    def head(self, t_us):
        """(R (n,3,3), valid (n,)) nearest head rotation to each time."""
        if self.ht is None:
            return np.repeat(np.eye(3)[None], len(t_us), 0), np.zeros(len(t_us), bool)
        t = t_us / 1e6
        j = np.clip(np.searchsorted(self.ht, t), 1, len(self.ht) - 1)
        j -= (t - self.ht[j - 1]) < (self.ht[j] - t)
        return self.hR[j], np.abs(self.ht[j] - t) < HEAD_TOL_S


def relative(R0, Rk, convention):
    """rotation taking step-k CPF vectors into the anchor CPF, for the two possible meanings of the stored R."""
    if convention == "world_from_cpf":
        return np.einsum("ji,njk->nik", R0, Rk)            # R0^T Rk
    return np.einsum("ij,nkj->nik", R0, Rk)                # R0 Rk^T  (R stored as cpf_from_world)


def check_convention(Z, recs, max_samples=3000):
    """Angular error of head-rotated CSV future gaze vs the lab's SLAM future gaze, per convention."""
    errs = {"world_from_cpf": [], "cpf_from_world": []}
    for z in Z:
        n = len(z["rec"])
        for i in np.linspace(0, n - 1, min(n, max_samples // len(Z))).astype(int):
            r = recs[str(z["rec"][i])]
            if r.ht is None:
                continue
            t0 = int(z["t0_us"][i])
            rows = r.rows(t0, np.arange(1, STEPS + 1))
            if rows is None or (rows < 0).any():
                continue
            R0, ok0 = r.head(np.array([t0]))
            Rk, okk = r.head(r.ts[rows])
            if not (ok0[0] and okk.all()):
                continue
            truth = unit(np.einsum("ji,nj->ni", z["R_now"][i], z["g_fut"][i]))   # SLAM future gaze, anchor CPF
            for conv in errs:
                pred = np.einsum("nij,nj->ni", relative(R0[0], Rk, conv), r.g[rows])
                errs[conv].append(np.degrees(np.arccos(np.clip((pred * truth).sum(-1), -1, 1))))
    return {c: np.concatenate(e) if e else np.array([np.nan]) for c, e in errs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="lab benchmark folder with {train,val,test}.npz")
    ap.add_argument("--gaze-root", required=True)
    ap.add_argument("--head-root", default=None, help="folder with <rec>.npz head rotations (optional)")
    ap.add_argument("--out", required=True, help="converted benchmark folder (has all.npz)")
    a = ap.parse_args()
    Z = [np.load(os.path.join(a.src, f"{p}.npz")) for p in PARTS]
    rec_all = np.concatenate([z["rec"] for z in Z]).astype(str)
    conv_rec = np.load(os.path.join(a.out, "all.npz"))["rec"].astype(str)
    assert np.array_equal(rec_all, conv_rec), "sample order differs from all.npz: re-run convert_sensoryspotlight.py"
    recs = {r: Recording(r, a.gaze_root, a.head_root) for r in sorted(set(rec_all))}
    print(f"{len(recs)} recordings, head rotations for {sum(r.ht is not None for r in recs.values())}", flush=True)

    errs = check_convention(Z, recs)
    conv = min(errs, key=lambda c: np.nanmedian(errs[c]))
    for c, e in errs.items():
        print(f"head-rotation convention {c:15s}: future-gaze error median {np.nanmedian(e):6.2f} deg, "
              f"90th pct {np.nanpercentile(e, 90):6.2f} deg  ({len(e)} steps){'   <- used' if c == conv else ''}")
    if np.nanmedian(errs[conv]) > 5:
        print("WARNING: head rotations do not reproduce the SLAM future gaze well; consider --head-root none")

    feats, n_missing = [], 0
    offsets = np.arange(-(STEPS - 1), 1)                       # -19 .. 0
    for z in Z:
        for rec, t0 in zip(z["rec"].astype(str), z["t0_us"].astype(np.int64)):
            r = recs[rec]
            f = np.zeros((STEPS, LAYOUT["n_features"]), np.float32)
            rows = r.rows(t0, offsets)
            if rows is None:
                n_missing += 1
                feats.append(f)
                continue
            gv = rows >= 0
            gc = np.where(gv[:, None], r.g[np.clip(rows, 0, None)], 0.0)
            R0, ok0 = r.head(np.array([t0]))
            Rk, okk = r.head(t0 + offsets * 100_000)
            hv = okk & ok0[0]
            rel = relative(R0[0], Rk, conv)
            ga = np.where(hv[:, None], np.einsum("nij,nj->ni", rel, gc), gc)
            f[:, 0:3], f[:, 3:6] = ga, gc
            f[:, 6:9] = np.where(hv[:, None], rotvec(rel), 0.0)
            f[:, 9], f[:, 10] = gv, hv
            feats.append(f)
    feats = np.stack(feats)
    np.savez(os.path.join(a.out, "gaze_all.npz"), path=feats.astype(np.float16), rec=rec_all,
             meta=np.array(json.dumps({**LAYOUT, "steps": STEPS, "head_convention": conv,
                                       "head_error_median_deg": float(np.nanmedian(errs[conv]))})))
    print(f"wrote {a.out}/gaze_all.npz  {feats.shape}  | anchor missing from CSV: {n_missing} | "
          f"gaze valid {feats[..., 9].mean():.3f} | head valid {feats[..., 10].mean():.3f}")


if __name__ == "__main__":
    main()
