"""Camera features for an attention-shift benchmark (build.window.anchor = shift), from the WHERE suite's video cache.

    S=~/SensorySpotlight/data/traj_forecast/attention_shift/senses_suite
    python scripts/attach_shift_video.py --bench data/fovs_aria_shift_sph --suite $S          # V-JEPA (default)
    python scripts/attach_shift_video.py --bench data/fovs_aria_shift_sph --suite $S --dense $S/dense_aea_lag.npy

The suite (SensorySpotlight tools/attention_senses_suite.py) has one row per WHERE question in data_<tag>.npz and
its video tokens in dense_<tag>[_vjepa].npy (2 n, 256, D): row 2 i = 16 x 16 patch tokens at question i's time
(V-JEPA 2: the last 0.4 s of the 1.6 s clip ending there; DINOv2: the frame there). Our shift samples are anchored at
the same moment (0.3 s before the shift), so each is matched to the question of the same recording whose time is
within --tol s and gets its tokens as a (D, 16, 16) map; unmatched samples get zeros and has = 0. Writes
<bench>/video_all.npz {map, has, rec}; train with data.video_features=true (video_dim = D).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def match(bench_rec, bench_t, q_rec, q_t, tol):
    """for each bench sample the index of the question of its recording nearest in time (within tol), else -1."""
    by = {}
    for i, r in enumerate(q_rec):
        by.setdefault(r, []).append(i)
    out = np.full(len(bench_rec), -1)
    for k, (r, t) in enumerate(zip(bench_rec, bench_t)):
        cand = by.get(r)
        if cand:
            j = cand[int(np.argmin(np.abs(q_t[cand] - t)))]
            if abs(q_t[j] - t) <= tol:
                out[k] = j
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--suite", required=True, help="folder with data_<tag>.npz and the dense video cache")
    ap.add_argument("--tag", default="aea_lag")
    ap.add_argument("--dense", default=None, help="video cache (default <suite>/dense_<tag>_vjepa.npy)")
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()
    A = np.load(os.path.join(a.bench, "all.npz"))
    if "land_dir" not in A.files:
        raise SystemExit(f"{a.bench} is not an attention-shift benchmark (build.window.anchor=shift)")
    Q = np.load(os.path.join(a.suite, f"data_{a.tag}.npz"))
    dense = np.load(a.dense or os.path.join(a.suite, f"dense_{a.tag}_vjepa.npy"), mmap_mode="r")
    if len(dense) != 2 * len(Q["rec"]):
        raise SystemExit(f"the video cache has {len(dense)} rows, expected 2 x {len(Q['rec'])} questions")
    rec = A["rec"].astype(str)
    j = match(rec, A["t0_us"] / 1e6, Q["rec"].astype(str), Q["t"].astype(np.float64), a.tol)
    D = dense.shape[-1]
    vmap = np.zeros((len(rec), D, 16, 16), np.float16)
    for k in np.flatnonzero(j >= 0):
        vmap[k] = np.asarray(dense[2 * j[k]], np.float16).reshape(16, 16, D).transpose(2, 0, 1)
    has = (j >= 0).astype(np.float16)
    out = os.path.join(a.bench, "video_all.npz")
    np.savez(out, map=vmap, has=has, rec=rec)
    print(f"wrote {out}: {len(rec)} samples, {int(has.sum())} matched to a WHERE question ({has.mean():.1%}), "
          f"tokens {D} x 16 x 16")
    covered = len(np.unique(j[j >= 0]))
    print(f"agreement with the lab's events: {has.mean():.1%} of our shift samples are lab questions, "
          f"{covered / len(Q['rec']):.1%} of the lab's {len(Q['rec'])} questions are among our samples "
          f"({len(set(rec))} of our recordings, {len(set(Q['rec'].astype(str)))} of theirs)")


if __name__ == "__main__":
    main()
