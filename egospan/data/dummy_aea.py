"""Write a fake "Aria Everyday Activities" download: the same file names, zip archives and CSV columns that
`egospan.data.aria.read_aea` reads from the real dataset, filled with synthetic recordings.

Per recording `locL_scriptS_seqQ_recR` in the output folder:

    <rec>_annotations.zip            metadata.json                      {"location": L, ...}
    <rec>_mps_eye_gaze.zip           general_eye_gaze.csv               10 Hz, yaw/pitch in the CPF
    <rec>_mps_slam_trajectories.zip  closed_loop_trajectory.csv         device pose + gravity
    <rec>_mps_slam_points.zip        semidense_points.csv.gz            uid, px/py/pz_world, dist_std
                                     semidense_observations.csv.gz      uid, frame_tracking_timestamp_us
    <rec>_device_cpf.json            4x4 T_device_cpf   (stands in for the calibration inside the real .vrs file,
                                                         which cannot be faked; read_aea picks it up automatically)
"""
import gzip
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .streams import dir_to_yaw_pitch
from .synthetic import make_recording

# a fixed, slightly rotated and offset CPF relative to the device frame
T_DEVICE_CPF = np.eye(4)
T_DEVICE_CPF[:3, :3] = Rotation.from_euler("x", 5, degrees=True).as_matrix()
T_DEVICE_CPF[:3, 3] = [0.0, 0.02, 0.03]


def _zip_write(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)


def _csv(df):
    return df.to_csv(index=False)


def _csv_gz(df):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(df.to_csv(index=False).encode())
    return buf.getvalue()


def write_aea_recording(root, rec, streams, T_device_cpf=T_DEVICE_CPF, rng=None):
    """Serialise one RecordingStreams in the AEA / MPS layout (see module docstring)."""
    rng = rng or np.random.default_rng(0)
    os.makedirs(root, exist_ok=True)
    f = lambda suffix: os.path.join(root, f"{rec}_{suffix}")                       # noqa: E731
    t = streams.t_us
    meta = streams.meta

    _zip_write(f("annotations.zip"), {"metadata.json": json.dumps(
        {"location": int(meta.get("location", 1)), "script": 1, "sequence": 1, "recording": 1})})

    yaw, pitch = dir_to_yaw_pitch(streams.gaze_cpf)
    _zip_write(f("mps_eye_gaze.zip"), {"general_eye_gaze.csv": _csv(pd.DataFrame({
        "tracking_timestamp_us": t, "yaw_rads_cpf": yaw, "pitch_rads_cpf": pitch,
        "depth_m": 1.0, "yaw_low_rads_cpf": yaw - 0.01, "yaw_high_rads_cpf": yaw + 0.01,
        "pitch_low_rads_cpf": pitch - 0.01, "pitch_high_rads_cpf": pitch + 0.01}))})

    T_wd = streams.T_world_cpf @ np.linalg.inv(T_device_cpf)                       # world <- device
    q = Rotation.from_matrix(T_wd[:, :3, :3]).as_quat()                            # x, y, z, w
    g = np.asarray(streams.gravity_world if streams.gravity_world is not None else [0, 0, -9.81])
    traj = pd.DataFrame({
        "graph_uid": "dummy", "tracking_timestamp_us": t + 1000,                   # 1 ms off, within the 5 ms tolerance
        "utc_timestamp_ns": (t + 1000) * 1000,
        "tx_world_device": T_wd[:, 0, 3], "ty_world_device": T_wd[:, 1, 3], "tz_world_device": T_wd[:, 2, 3],
        "qx_world_device": q[:, 0], "qy_world_device": q[:, 1], "qz_world_device": q[:, 2],
        "qw_world_device": q[:, 3],
        "gravity_x_world": g[0], "gravity_y_world": g[1], "gravity_z_world": g[2]})
    _zip_write(f("mps_slam_trajectories.zip"), {"closed_loop_trajectory.csv": _csv(traj)})

    n = len(streams.points_world)
    uid = rng.permutation(n).astype(np.int64) * 3 + 1000                            # non-contiguous, unordered ids
    pts = pd.DataFrame({"uid": uid, "graph_uid": "dummy",
                        "px_world": streams.points_world[:, 0], "py_world": streams.points_world[:, 1],
                        "pz_world": streams.points_world[:, 2],
                        "inv_dist_std": 100.0, "dist_std": streams.points_dist_std})
    step = np.repeat(np.arange(len(t)), np.diff(streams.obs_ptr))
    jitter = rng.integers(-30_000, 30_000, len(step))                               # inside the +-50 ms bin
    obs = pd.DataFrame({"uid": uid[streams.obs_idx], "frame_tracking_timestamp_us": t[step] + jitter,
                        "camera_serial": "dummy", "u": 0.0, "v": 0.0})
    _zip_write(f("mps_slam_points.zip"), {"semidense_points.csv.gz": _csv_gz(pts),
                                          "semidense_observations.csv.gz": _csv_gz(obs)})
    with open(f("device_cpf.json"), "w") as fh:
        json.dump(np.asarray(T_device_cpf).tolist(), fh)


def make_dummy_aea(root, n_recordings=15, seconds=40.0, locations=5, seed=0, n_wall=6000):
    """Write `n_recordings` fake recordings spread over `locations` locations; returns their names."""
    names = []
    for i in range(n_recordings):
        loc = 1 + i % locations
        rec = f"loc{loc}_script{1 + i // locations}_seq1_rec1"
        streams = make_recording(name=rec, seconds=seconds, n_wall=n_wall, seed=seed + i, location=loc)
        write_aea_recording(root, rec, streams, rng=np.random.default_rng(seed + i))
        names.append(rec)
    return names
