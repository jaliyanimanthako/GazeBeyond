"""Attach the lab's per-sample camera features (V-JEPA 2 token maps) to one of our packed benchmarks.

    LAB=/data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria
    python scripts/attach_video_features.py --lab $LAB --bench data/fovs_aria_lab
    python scripts/attach_video_features.py --lab $LAB --bench data/rebuild_b4

Reads <lab>/vjepa_all.npz (map (n, 128, 16, 16), has (n,); rows in the order of the lab's train + val + test files)
and writes <bench>/video_all.npz in the benchmark's own sample order. Samples are matched by (recording, position of the
sample within its recording), so it works for the converted lab data and for our rebuilds alike; every benchmark sample
must be found. Then train with --set data.video_features=true.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sample_keys(rec):
    rec = np.asarray(rec).astype(str)
    seen, keys = {}, []
    for r in rec:
        p = seen.get(r, 0)
        seen[r] = p + 1
        keys.append(f"{r}#{p}")
    return np.array(keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab", required=True, help="lab benchmark folder: {train,val,test}.npz and vjepa_all.npz")
    ap.add_argument("--bench", required=True, help="our packed benchmark (all.npz)")
    ap.add_argument("--features", default="vjepa_all.npz")
    a = ap.parse_args()
    lab_rec = np.concatenate([np.load(os.path.join(a.lab, f"{p}.npz"))["rec"] for p in ("train", "val", "test")])
    V = np.load(os.path.join(a.lab, a.features))
    if len(V["map"]) != len(lab_rec):
        raise SystemExit(f"{a.features} has {len(V['map'])} rows but the lab's train+val+test have {len(lab_rec)}")
    row = {k: i for i, k in enumerate(sample_keys(lab_rec))}
    bench_rec = np.load(os.path.join(a.bench, "all.npz"))["rec"].astype(str)
    keys = sample_keys(bench_rec)
    missing = [k for k in keys if k not in row]
    if missing:
        raise SystemExit(f"{len(missing)} benchmark samples have no lab features, e.g. {missing[:3]}")
    idx = np.array([row[k] for k in keys])
    out = os.path.join(a.bench, "video_all.npz")
    np.savez(out, map=V["map"][idx], has=V["has"][idx], rec=bench_rec)
    print(f"wrote {out}: {len(idx)} samples, map {V['map'].shape[1:]}, with video {V['has'][idx].mean():.1%}")


if __name__ == "__main__":
    main()
