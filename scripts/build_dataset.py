"""Build a 3D visual span forecasting benchmark from raw recordings (Sec. 5, Appendix A.1).

Step 1 - extract: one .npz of bit-packed samples per recording into <out>/recs/   (needs projectaria_tools for AEA /
         Ego-Exo4D calibration; run it in the environment that has it)
Step 2 - pack: concatenate into <out>/all.npz and write <out>/splits.npz (train/val/test [+ location-wise CV folds])

    # FoVS-Aria (Aria Everyday Activities)
    python scripts/build_dataset.py aea    --config configs/fovs_aria.yaml --src /data/aea --out data/fovs_aria
    python scripts/build_dataset.py pack   --config configs/fovs_aria.yaml --out data/fovs_aria

    # FoVS-EgoExo (Ego-Exo4D; --takes-json gives categories, --segments keeps task-execution footage only)
    python scripts/build_dataset.py egoexo --config configs/fovs_egoexo.yaml --src /data/egoexo/takes \
        --takes-json /data/egoexo/takes.json --out data/fovs_egoexo
    python scripts/build_dataset.py pack   --config configs/fovs_egoexo.yaml --out data/fovs_egoexo \
        --set build.pack.split_file=data/fovs_egoexo/split.json

    # fake AEA download (same files as the real one; --out is the fake download folder), then build from it
    python scripts/build_dataset.py dummy-aea --out data/dummy_aea --n 15
    python scripts/build_dataset.py aea --config configs/dummy.yaml --src data/dummy_aea --out data/dummy_fovs

    # HD-EPIC and ADT on the lab's (SensorySpotlight) questions: --folds maps each recording to the lab's held-out group
    # (tools/export_gbf_folds.py there), used as its "location", so location-wise CV folds = the lab's folds
    python scripts/build_dataset.py hdepic --config configs/fovs_aria_shift.yaml --folds data/lab_q/hdepic_where_folds.json \
        --slam-root /data/disk2/hdepic_dinithi/HD-EPIC/SLAM-and-Gaze --src /data/disk2/hdepic/HD-EPIC/raw --out ...
    python scripts/build_dataset.py adt --config configs/fovs_aria_shift.yaml --folds data/lab_q/adt_where_folds.json \
        --src /data/disk2/adt --out ...

    # synthetic recordings built directly (skips the file format; for smoke tests)
    python scripts/build_dataset.py synthetic --out data/synthetic --n 12
    python scripts/build_dataset.py pack --out data/synthetic --set build.pack.min_points_sigma=null
"""
import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.config import load_config                                   # noqa: E402
from egospan.data.build import extract_samples, pack, save_recording   # noqa: E402


def _extract_one(job, build_cfg, out_dir):
    kind, name, arg = job
    path = os.path.join(out_dir, "recs", f"{name}.npz")
    if os.path.isfile(path):
        return name, "exists"
    try:
        if kind == "aea":
            from egospan.data.aria import read_aea
            streams = read_aea(build_cfg["src"], name)
        elif kind == "aea-server":
            from egospan.data.aria import read_aea_server
            cpf = os.path.join(build_cfg["cpf_root"], f"{name}.json")
            if not os.path.isfile(cpf):
                return name, f"SKIPPED: no {cpf} (run scripts/derive_device_cpf.py)"
            T = np.array(json.load(open(cpf))["T_device_cpf"])
            streams = read_aea_server(build_cfg["slam_root"], build_cfg["gaze_root"], name, T)
        elif kind == "hdepic":
            from egospan.data.aria import read_hdepic
            P, zpath, i, loc = arg
            streams = read_hdepic(build_cfg["src"], name, P, zpath, i, loc)
        elif kind == "adt":
            from egospan.data.aria import read_adt
            streams = read_adt(build_cfg["src"], name, arg)
        elif kind == "egoexo":
            from egospan.data.aria import read_egoexo, restrict_to_segments
            take_dir, meta, segments = arg
            streams = read_egoexo(take_dir, meta=meta)
            if segments:
                restrict_to_segments(streams, segments)
        else:
            from egospan.data.synthetic import make_recording
            seed, loc = arg
            streams = make_recording(name=name, seed=seed, location=loc, seconds=40)
        arr = extract_samples(streams, build_cfg["lift"], build_cfg["window"])
        if arr is None:
            return name, "no samples"
        save_recording(arr, path)
        return name, f"{len(arr['X'])} samples, {len(streams.points_world)} points"
    except Exception as e:                                                  # noqa: BLE001
        return name, f"FAILED {type(e).__name__}: {e}"


def _egoexo_jobs(src, takes_json, segments_json):
    meta = {}
    if takes_json:
        for t in json.load(open(takes_json)):
            meta[t["take_name"]] = {"category": t.get("parent_task_name", t.get("task_name", "")),
                                    "take_uid": t.get("take_uid", "")}
    segments = json.load(open(segments_json)) if segments_json else {}
    jobs = []
    for d in sorted(os.listdir(src)):
        if os.path.isdir(os.path.join(src, d)):
            jobs.append(("egoexo", d, (os.path.join(src, d), meta.get(d, {}), segments.get(d))))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["aea", "aea-server", "hdepic", "adt", "egoexo", "synthetic", "dummy-aea",
                                         "pack"])
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. build.lift.resolution=32")
    ap.add_argument("--src", help="AEA download folder, Ego-Exo4D takes/ folder, HD-EPIC raw/ folder or ADT folder")
    ap.add_argument("--slam-root", help="aea-server: folder with <rec>_mps_slam_*.zip; hdepic: SLAM-and-Gaze/")
    ap.add_argument("--folds", help="hdepic / adt: json {recording: the lab's held-out group}; others are skipped")
    ap.add_argument("--gaze-root", help="aea-server: folder with <rec>/mps/eye_gaze/ and <rec>/metadata.json")
    ap.add_argument("--cpf-root", help="aea-server: folder with <rec>.json from scripts/derive_device_cpf.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only the first N recordings (debugging)")
    ap.add_argument("--recs", nargs="+", help="only these recordings (names as in the source folder)")
    ap.add_argument("--takes-json", help="Ego-Exo4D takes.json (categories)")
    ap.add_argument("--segments", help="json {take_name: [[start_s, end_s], ...]} of task-execution footage")
    ap.add_argument("--n", type=int, default=12, help="synthetic / dummy-aea: number of recordings")
    ap.add_argument("--seconds", type=float, default=40.0, help="dummy-aea: length of each recording")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)

    if a.cmd == "dummy-aea":
        from egospan.data.dummy_aea import make_dummy_aea
        names = make_dummy_aea(a.out, a.n, seconds=a.seconds)
        print(f"wrote {len(names)} fake AEA recordings to {a.out}; build with: "
              f"build_dataset.py aea --src {a.out} --out <dataset dir>")
        return
    if a.cmd == "pack":
        pack(os.path.join(a.out, "recs"), a.out, cfg.build.pack)
        return
    if a.cmd == "aea":
        from egospan.data.aria import list_aea
        jobs = [("aea", r, None) for r in list_aea(a.src)]
    elif a.cmd == "aea-server":
        from egospan.data.aria import list_aea_server
        jobs = [("aea-server", r, None) for r in list_aea_server(a.slam_root)]
    elif a.cmd in ("hdepic", "adt"):
        if not (a.folds and a.src):
            ap.error(f"{a.cmd} needs --folds and --src")
        folds = json.load(open(a.folds))
        if a.cmd == "hdepic":
            from egospan.data.aria import list_hdepic
            jobs = [("hdepic", r, (P, z, i, folds[r])) for r, P, z, i in list_hdepic(a.slam_root) if r in folds]
        else:
            from egospan.data.aria import list_adt
            jobs = [("adt", r, folds[r]) for r in list_adt(a.src) if r in folds]
        print(f"{len(jobs)} of the {len(folds)} recordings in {a.folds} have their files", flush=True)
    elif a.cmd == "egoexo":
        jobs = _egoexo_jobs(a.src, a.takes_json, a.segments)
    else:
        jobs = [("synthetic", f"synth_loc{1 + i % 5}_{i:04d}", (i, 1 + i % 5)) for i in range(a.n)]
    if a.recs:
        jobs = [j for j in jobs if j[1] in set(a.recs)]
    jobs = jobs[:a.limit] if a.limit else jobs
    os.makedirs(os.path.join(a.out, "recs"), exist_ok=True)
    build_cfg = {"src": a.src, "slam_root": a.slam_root, "gaze_root": a.gaze_root, "cpf_root": a.cpf_root,
                 "lift": cfg.build.lift, "window": cfg.build.window}
    fn = partial(_extract_one, build_cfg=build_cfg, out_dir=a.out)
    if a.workers > 1:
        with Pool(a.workers) as pool:
            for name, msg in pool.imap_unordered(fn, jobs):
                print(f"{name}: {msg}", flush=True)
    else:
        for job in jobs:
            print("{}: {}".format(*fn(job)), flush=True)


if __name__ == "__main__":
    main()
