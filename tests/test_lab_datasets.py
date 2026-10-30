"""HD-EPIC and ADT readers (the lab's datasets): a fake AEA recording repackaged in each dataset's layout, with its
eye gaze at a higher rate, must come back on the lab's 10 Hz timeline with the same poses, gaze and points."""
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from egospan.data.aria import T_DEVICE_CPF_ADT, list_adt, list_hdepic, read_aea, read_adt, read_hdepic
from egospan.data.dummy_aea import T_DEVICE_CPF, make_dummy_aea
from egospan.video import CAMERA, camera_of


def _aea(tmp_path):
    src = tmp_path / "aea"
    rec = make_dummy_aea(str(src), n_recordings=1, seconds=8, n_wall=1500, seed=3)[0]
    member = lambda f, m: zipfile.ZipFile(src / f"{rec}_{f}").read(m)          # noqa: E731
    gaze = pd.read_csv(io.BytesIO(member("mps_eye_gaze.zip", "general_eye_gaze.csv")))
    files = {"traj": member("mps_slam_trajectories.zip", "closed_loop_trajectory.csv"),
             "points": member("mps_slam_points.zip", "semidense_points.csv.gz"),
             "obs": member("mps_slam_points.zip", "semidense_observations.csv.gz")}
    return read_aea(str(src), rec), gaze, files


def _upsample(gaze, k, lead=0):
    """gaze at k x the rate: row lead + k j is the original row j, the rows between are other directions; `lead`
    rows come first (without a depth hit)."""
    t = gaze["tracking_timestamp_us"].to_numpy()
    n = lead + k * len(gaze)
    tt = t[0] + (np.arange(n) - lead) * (100_000 // k)
    up = pd.DataFrame({"tracking_timestamp_us": tt, "yaw": 0.3, "pitch": -0.2, "depth_m": 1.0})
    up.loc[:lead - 1, "depth_m"] = 0.0
    keep = lead + k * np.arange(len(gaze))
    up.loc[keep, "tracking_timestamp_us"] = t
    up.loc[keep, "yaw"] = gaze["yaw_rads_cpf"].to_numpy()
    up.loc[keep, "pitch"] = gaze["pitch_rads_cpf"].to_numpy()
    return up


def _zip(path, members):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)


def test_hdepic_reader(tmp_path):
    ref, gaze, f = _aea(tmp_path)
    rec, P = "P03-20240101-000000", "P03"
    slam, raw = tmp_path / "SLAM-and-Gaze", tmp_path / "raw"
    multi = slam / P / "SLAM" / "multi"
    _zip(str(multi / "0.zip"), {"0/slam/closed_loop_trajectory.csv": f["traj"],
                                "0/slam/semidense_points.csv.gz": f["points"],
                                "0/slam/semidense_observations.csv.gz": f["obs"]})
    (multi / "0.zip.ok").write_text("")
    (multi / "vrs_to_multi_slam.json").write_text(json.dumps({f"/x/{rec}.vrs": 0, "/x/unfinished.vrs": 1}))
    up = _upsample(gaze, 6)                                                    # 60 Hz, newer MPS: per-eye yaw
    csv = pd.DataFrame({"tracking_timestamp_us": up.tracking_timestamp_us, "left_yaw_rads_cpf": up.yaw,
                        "right_yaw_rads_cpf": up.yaw, "pitch_rads_cpf": up.pitch}).to_csv(index=False)
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr(f"mps_{rec}_vrs/eye_gaze/general_eye_gaze.csv", csv)
    _zip(str(raw / "HD-EPIC_GAZE" / P / "GAZE_HAND.zip"), {f"GAZE_HAND/mps_{rec}_vrs.zip": inner.getvalue()})
    cal = raw / "HD-EPIC_INTERMEDIATE" / P / rec / "device_calibration.json"
    cal.parent.mkdir(parents=True)
    cal.write_text(json.dumps({"origin_transforms": {"T_device_cpf": T_DEVICE_CPF[:3].tolist()}}))   # 3 x 4 [R | t]

    jobs = list_hdepic(str(slam))
    assert [j[0] for j in jobs] == [rec]                                       # the unfinished zip is left out
    got = read_hdepic(str(raw), *jobs[0], location=3)
    np.testing.assert_array_equal(got.t_us, ref.t_us)
    np.testing.assert_allclose(got.gaze_cpf, ref.gaze_cpf, atol=1e-6)
    np.testing.assert_allclose(got.T_world_cpf, ref.T_world_cpf, atol=1e-6)
    np.testing.assert_array_equal(got.obs_ptr, ref.obs_ptr)
    assert len(got.points_world) == len(ref.points_world)
    assert got.meta["dataset"] == "hdepic" and got.meta["location"] == 3


def test_adt_reader(tmp_path):
    ref, gaze, f = _aea(tmp_path)
    seq = "Apartment_release_clean_seq131_M1292"
    d = tmp_path / "adt" / seq
    up = _upsample(gaze, 3, lead=4)                                            # 30 Hz, 4 samples before a depth hit
    # the ground truth lives in another world frame: SLAM = A x ground truth
    A = Rotation.from_euler("zyx", [30, -10, 5], degrees=True).as_matrix()
    gt = pd.read_csv(io.BytesIO(f["traj"]))
    q = ["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
    gt[q] = Rotation.from_matrix(A.T @ Rotation.from_quat(gt[q].to_numpy()).as_matrix()).as_quat()
    _zip(str(d / f"{seq}_main_groundtruth.zip"), {"eyegaze.csv": pd.DataFrame({
        "tracking_timestamp_us": up.tracking_timestamp_us, "yaw_rads_cpf": up.yaw, "pitch_rads_cpf": up.pitch,
        "depth_m": up.depth_m}).to_csv(index=False), "aria_trajectory.csv": gt.to_csv(index=False)})
    _zip(str(d / f"{seq}_mps_slam_trajectories.zip"), {"mps/slam/closed_loop_trajectory.csv": f["traj"]})
    _zip(str(d / f"{seq}_mps_slam_points.zip"), {"mps/slam/semidense_points.csv.gz": f["points"],
                                                 "mps/slam/semidense_observations.csv.gz": f["obs"]})
    (tmp_path / "adt" / "Apartment_release_no_points").mkdir()
    assert list_adt(str(tmp_path / "adt")) == [seq]
    got = read_adt(str(tmp_path / "adt"), seq, location=2)
    np.testing.assert_array_equal(got.t_us, ref.t_us)                          # every 3rd sample from the first hit
    np.testing.assert_allclose(got.gaze_cpf, ref.gaze_cpf, atol=1e-6)
    T_wd = ref.T_world_cpf @ np.linalg.inv(T_DEVICE_CPF)                       # the trajectory, with the lab's CPF
    np.testing.assert_allclose(got.T_world_cpf, T_wd @ T_DEVICE_CPF_ADT, atol=1e-6)
    np.testing.assert_array_equal(got.obs_ptr, ref.obs_ptr)
    assert got.meta["dataset"] == "adt" and got.meta["location"] == 2
    np.testing.assert_allclose(got.meta["R_slam_from_lab"], A, atol=1e-6)

    # the build command: the lab's fold becomes the location; recordings without a fold are skipped
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({seq: 2, "Apartment_release_other": 1}))
    out = tmp_path / "bench"
    for cmd in (["adt", "--folds", str(folds), "--src", str(tmp_path / "adt"), "--workers", "1"], ["pack"]):
        r = subprocess.run([sys.executable, os.path.join(root, "scripts", "build_dataset.py"), *cmd,
                            "--config", os.path.join(root, "configs", "dummy.yaml"), "--out", str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    z = np.load(out / "all.npz")
    assert (z["location"] == 2).all() and json.loads(str(z["meta"]))["dataset"] == "adt"


def test_camera_by_dataset():
    assert camera_of({"dataset": "hdepic"}) == CAMERA["hdepic"]
    assert camera_of({"dataset": "adt"}) == camera_of({}) == CAMERA["default"]
