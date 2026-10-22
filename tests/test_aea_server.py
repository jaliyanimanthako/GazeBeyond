"""Raw AEA in the lab server's layout -> derive T_device_cpf from lab files -> aea-server build -> validation."""
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np

from egospan.data.build import WindowConfig, extract_samples, unpack_grids
from egospan.data.dummy_aea import T_DEVICE_CPF, write_aea_recording
from egospan.data.synthetic import make_recording
from egospan.lift import LiftConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB_LIFT = LiftConfig(grid_frame="local", crop_around_user=False, sor_neighbors=16, sor_std_ratio=2.0,
                      max_dist_std=0.1, sor_scope="global")


def run(*args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_server_layout_end_to_end(tmp_path):
    raw, slam, gaze, lab = tmp_path / "raw", tmp_path / "mps_slam", tmp_path / "gaze", tmp_path / "lab"
    for d in (slam, gaze, lab / "recs"):
        d.mkdir(parents=True)
    names, parts = [], {"train": [], "val": [], "test": []}
    for i, part in enumerate(("train", "train", "val", "test")):
        rec = f"AriaEverydayActivities_1.0.0_loc{[1, 2, 3, 4][i]}_script1_seq{i}_rec1"
        s = make_recording(name=rec, seconds=10, seed=i, location=[1, 2, 3, 4][i], n_wall=3000)
        s.points_dist_std = np.random.default_rng(i).uniform(0, 0.15, len(s.points_world)).astype(np.float32)
        write_aea_recording(str(raw), rec, s)
        for z in ("mps_slam_points.zip", "mps_slam_trajectories.zip"):           # server layout
            shutil.copy(raw / f"{rec}_{z}", slam / f"{rec}_{z}")
        (gaze / rec / "mps" / "eye_gaze").mkdir(parents=True)
        gz = gaze / rec / "mps" / "eye_gaze"
        zipfile.ZipFile(raw / f"{rec}_mps_eye_gaze.zip").extract("general_eye_gaze.csv", gz)
        zipfile.ZipFile(raw / f"{rec}_annotations.zip").extract("metadata.json", gaze / rec)
        a = extract_samples(s, LAB_LIFT, WindowConfig())                        # the "lab" build, from memory
        YF = unpack_grids(a["YF"], (2, 4, 16, 16, 16))
        np.savez(lab / "recs" / f"{rec}.npz", X=a["X"], t0_us=a["t0_us"],
                 Y=np.stack([np.packbits(y.any(0).reshape(-1)) for y in YF]))
        parts[part].append(dict(rec=np.array([rec] * len(a["X"])), t0_us=a["t0_us"],
                                R_now=a["T_world_grid"][:, :3, :3], c_now=a["T_world_grid"][:, :3, 3]))
        names.append(rec)
    for p, items in parts.items():
        np.savez(lab / f"{p}.npz", **{k: np.concatenate([it[k] for it in items]) for k in items[0]})

    out = run("scripts/derive_device_cpf.py", "--lab", str(lab), "--slam-root", str(slam),
              "--out", str(tmp_path / "cpf"), "--workers", "2")
    assert "derived 4 / 4" in out and "WARNING" not in out
    T = np.array(json.load(open(tmp_path / "cpf" / f"{names[0]}.json"))["T_device_cpf"])
    np.testing.assert_allclose(T, T_DEVICE_CPF, atol=1e-6)

    run("scripts/build_dataset.py", "aea-server", "--config", "configs/fovs_aria_lab_rebuild.yaml",
        "--slam-root", str(slam), "--gaze-root", str(gaze), "--cpf-root", str(tmp_path / "cpf"),
        "--out", str(tmp_path / "rebuild"), "--workers", "2")
    out = run("scripts/validate_against_lab.py", "--ours", str(tmp_path / "rebuild"), "--lab", str(lab))
    line = [ln for ln in out.splitlines() if ln.startswith("identical samples")][0]
    x_exact, y_exact = (float(v.rstrip("%,")) for v in line.split()[3::2])
    assert x_exact >= 95 and y_exact >= 95, out
    meta = json.loads(str(np.load(tmp_path / "rebuild" / "recs" / f"{names[0]}.npz")["meta"]))
    assert meta["recording"]["location"] == 1
