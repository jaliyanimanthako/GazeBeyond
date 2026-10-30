"""Readers for Project Aria MPS outputs -> RecordingStreams.

Both source datasets of the paper are recorded with Aria glasses and ship Machine Perception Services (MPS) outputs:
closed-loop SLAM trajectory, semidense points + their observations, and eye gaze in the central pupil frame (CPF).

  Aria Everyday Activities (FoVS-Aria)    zip archives per recording       read_aea(root, rec)
  Ego-Exo4D (FoVS-EgoExo)                 plain directories per take       read_egoexo(take_dir)

Only `pandas` is needed to parse MPS; `projectaria_tools` is needed once per recording to read T_device_cpf from the
device calibration (VRS file or MPS online_calibration.jsonl). Ego-Exo4D layouts differ between releases: the file
patterns in read_egoexo are the common ones; pass explicit paths if yours differ.
"""
import glob
import io
import json
import os
import zipfile

import numpy as np

from .streams import RecordingStreams, bin_observations, quat_to_rotmat, yaw_pitch_to_dir

POSE_TOL_US = 5_000          # nearest closed-loop pose must be within 5 ms of the gaze timestamp


# ------------------------------------------------------------------------------------------------- calibration
def T_device_cpf_from_vrs(vrs_path):
    from projectaria_tools.core import data_provider
    p = data_provider.create_vrs_data_provider(vrs_path)
    return p.get_device_calibration().get_transform_device_cpf().to_matrix()


def T_device_cpf_from_online_calib(jsonl_path):
    from projectaria_tools.core.calibration import device_calibration_from_json_string
    with open(jsonl_path) as f:
        line = f.readline()
    return device_calibration_from_json_string(line).get_transform_device_cpf().to_matrix()


# ------------------------------------------------------------------------------------------------- MPS parsing
def _read_csv(src, usecols=None):
    """src: a path, (zip_path, member) for files inside an archive, or (zip_path, inner_zip, member) for a file inside
    an archive inside an archive (HD-EPIC's eye gaze)."""
    import pandas as pd
    if isinstance(src, tuple):
        zpath, *inner, member = src
        with zipfile.ZipFile(zpath) as z:
            for name in inner:
                z = zipfile.ZipFile(io.BytesIO(z.read(name)))
            buf = io.BytesIO(z.read(member))
        return pd.read_csv(buf, compression="gzip" if member.endswith(".gz") else None, usecols=usecols)
    return pd.read_csv(src, usecols=usecols)


def _zip_member(zpath, suffix):
    """the one member of the archive whose name ends with `suffix` (layouts differ between releases)."""
    with zipfile.ZipFile(zpath) as z:
        hits = [n for n in z.namelist() if n.endswith(suffix)]
    if len(hits) != 1:
        raise FileNotFoundError(f"{zpath}: {len(hits)} members ending with {suffix}")
    return hits[0]


def read_gaze_pose(gaze_src, trajectory_src, T_device_cpf, gaze_rows=None):
    """Eye gaze and head pose on the gaze timeline, without the (large) SLAM points. gaze_rows: optional function
    (gaze table -> row indices) that picks the gaze samples to keep, e.g. to bring 30 / 60 Hz gaze to 10 Hz.
    Returns (t_us, gaze_cpf (n, 3), T_world_cpf (n, 4, 4), pose_ok (n,), gravity_world or None)."""
    G = _read_csv(gaze_src)
    if gaze_rows is not None:
        G = G.iloc[gaze_rows(G)].reset_index(drop=True)
    tg = G["tracking_timestamp_us"].to_numpy(np.int64)
    if "yaw_rads_cpf" in G:
        yaw = G["yaw_rads_cpf"].to_numpy()
    else:                                                     # newer MPS: per-eye yaw, shared pitch
        yaw = 0.5 * (G["left_yaw_rads_cpf"].to_numpy() + G["right_yaw_rads_cpf"].to_numpy())
    gaze = yaw_pitch_to_dir(yaw, G["pitch_rads_cpf"].to_numpy())

    pose_cols = ["tx_world_device", "ty_world_device", "tz_world_device",
                 "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
    Tr = _read_csv(trajectory_src)
    tt = Tr["tracking_timestamp_us"].to_numpy(np.int64)
    j = np.clip(np.searchsorted(tt, tg), 1, len(tt) - 1)
    j -= (tg - tt[j - 1]) < (tt[j] - tg)
    pose_ok = np.abs(tt[j] - tg) < POSE_TOL_US
    P = Tr[pose_cols].to_numpy(float)[j]
    T_world_device = np.tile(np.eye(4), (len(tg), 1, 1))
    T_world_device[:, :3, :3] = quat_to_rotmat(P[:, 3:])
    T_world_device[:, :3, 3] = P[:, :3]
    T_world_cpf = T_world_device @ np.asarray(T_device_cpf, float)
    gravity = None
    if {"gravity_x_world", "gravity_y_world", "gravity_z_world"} <= set(Tr.columns):
        gravity = Tr[["gravity_x_world", "gravity_y_world", "gravity_z_world"]].to_numpy(float).mean(0)
    return tg, gaze, T_world_cpf, pose_ok, gravity


def read_mps(name, gaze_src, trajectory_src, points_src, observations_src, T_device_cpf, meta=None, gaze_rows=None):
    """Synchronise MPS streams on the eye-gaze timeline."""
    tg, gaze, T_world_cpf, pose_ok, gravity = read_gaze_pose(gaze_src, trajectory_src, T_device_cpf, gaze_rows)
    Pt = _read_csv(points_src)
    uids = Pt["uid"].to_numpy(np.int64)
    xyz = Pt[["px_world", "py_world", "pz_world"]].to_numpy(np.float32)
    dist_std = Pt["dist_std"].to_numpy(np.float32) if "dist_std" in Pt else None
    Ob = _read_csv(observations_src, usecols=["uid", "frame_tracking_timestamp_us"])
    order = np.argsort(uids)
    pos = np.searchsorted(uids[order], Ob["uid"].to_numpy(np.int64))
    pos = np.clip(pos, 0, len(uids) - 1)
    rows = order[pos]
    known = uids[rows] == Ob["uid"].to_numpy(np.int64)
    ptr, idx = bin_observations(tg, rows[known], Ob["frame_tracking_timestamp_us"].to_numpy(np.int64)[known], len(xyz))

    valid = pose_ok & np.isfinite(gaze).all(1) & np.isfinite(T_world_cpf).all((1, 2))
    return RecordingStreams(name=name, t_us=tg, T_world_cpf=T_world_cpf, gaze_cpf=gaze, valid=valid,
                            points_world=xyz, obs_ptr=ptr, obs_idx=idx, points_dist_std=dist_std,
                            gravity_world=gravity, meta=dict(meta or {}))


# ------------------------------------------------------------------------------------------------- AEA
def list_aea(root):
    files = glob.glob(os.path.join(root, "*_annotations.zip"))
    return sorted(os.path.basename(f)[:-len("_annotations.zip")] for f in files)


def read_aea(root, rec, T_device_cpf=None):
    """Aria Everyday Activities recording `rec` in `root` (the flat AEA download: per recording
    *_main_recording.vrs, *_annotations.zip, *_mps_eye_gaze.zip, *_mps_slam_trajectories.zip, *_mps_slam_points.zip)."""
    f = lambda suffix: os.path.join(root, f"{rec}_{suffix}")              # noqa: E731
    if T_device_cpf is None:
        sidecar = f("device_cpf.json")                    # written by egospan.data.dummy_aea in place of a .vrs
        if os.path.isfile(sidecar):
            T_device_cpf = np.array(json.load(open(sidecar)))
        else:
            T_device_cpf = T_device_cpf_from_vrs(f("main_recording.vrs"))
    meta = json.loads(zipfile.ZipFile(f("annotations.zip")).read("metadata.json"))
    meta = {"dataset": "aea", "location": int(meta["location"]), **{k: v for k, v in meta.items() if k != "location"}}
    return read_mps(rec,
                    gaze_src=(f("mps_eye_gaze.zip"), "general_eye_gaze.csv"),
                    trajectory_src=(f("mps_slam_trajectories.zip"), "closed_loop_trajectory.csv"),
                    points_src=(f("mps_slam_points.zip"), "semidense_points.csv.gz"),
                    observations_src=(f("mps_slam_points.zip"), "semidense_observations.csv.gz"),
                    T_device_cpf=T_device_cpf, meta=meta)


# ------------------------------------------------------------------------------------------------- AEA, lab server
def list_aea_server(slam_root):
    files = glob.glob(os.path.join(slam_root, "*_mps_slam_points.zip"))
    return sorted(os.path.basename(f)[:-len("_mps_slam_points.zip")] for f in files)


def read_aea_server(slam_root, gaze_root, rec, T_device_cpf):
    """AEA as stored on the lab server: SLAM zips in slam_root (<rec>_mps_slam_{points,trajectories}.zip, original AEA
    names), eye gaze and metadata unzipped in gaze_root/<rec>/ (mps/eye_gaze/general_eye_gaze.csv, metadata.json).
    There are no .vrs files, so T_device_cpf must be given (scripts/derive_device_cpf.py recovers it per recording)."""
    meta = json.load(open(os.path.join(gaze_root, rec, "metadata.json")))
    meta = {"dataset": "aea", "location": int(meta["location"]), **{k: v for k, v in meta.items() if k != "location"}}
    f = lambda suffix: os.path.join(slam_root, f"{rec}_{suffix}")              # noqa: E731
    return read_mps(rec,
                    gaze_src=os.path.join(gaze_root, rec, "mps", "eye_gaze", "general_eye_gaze.csv"),
                    trajectory_src=(f("mps_slam_trajectories.zip"), "closed_loop_trajectory.csv"),
                    points_src=(f("mps_slam_points.zip"), "semidense_points.csv.gz"),
                    observations_src=(f("mps_slam_points.zip"), "semidense_observations.csv.gz"),
                    T_device_cpf=T_device_cpf, meta=meta)


# ------------------------------------------------------------------------------------------------- Ego-Exo4D
def _first(patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"none of: {patterns}")


def read_egoexo(take_dir, meta=None, T_device_cpf=None, personalized_gaze=False):
    """One Ego-Exo4D take directory (takes/<take_name>/ with trajectory/ and eye_gaze/ MPS outputs)."""
    name = os.path.basename(take_dir.rstrip("/"))
    gz = "personalized_eye_gaze.csv" if personalized_gaze else "general_eye_gaze.csv"
    paths = dict(
        gaze_src=_first([os.path.join(take_dir, "eye_gaze", gz), os.path.join(take_dir, "**", gz)]),
        trajectory_src=_first([os.path.join(take_dir, "trajectory", "closed_loop_trajectory.csv"),
                               os.path.join(take_dir, "**", "closed_loop_trajectory.csv")]),
        points_src=_first([os.path.join(take_dir, "trajectory", "semidense_points.csv.gz"),
                           os.path.join(take_dir, "**", "semidense_points.csv.gz")]),
        observations_src=_first([os.path.join(take_dir, "trajectory", "semidense_observations.csv.gz"),
                                 os.path.join(take_dir, "**", "semidense_observations.csv.gz")]))
    if T_device_cpf is None:
        vrs = sorted(glob.glob(os.path.join(take_dir, "aria*.vrs")))
        if vrs:
            T_device_cpf = T_device_cpf_from_vrs(vrs[0])
        else:
            T_device_cpf = T_device_cpf_from_online_calib(
                _first([os.path.join(take_dir, "trajectory", "online_calibration.jsonl"),
                        os.path.join(take_dir, "**", "online_calibration.jsonl")]))
    return read_mps(name, T_device_cpf=T_device_cpf, meta={"dataset": "egoexo", **(meta or {})}, **paths)


def restrict_to_segments(streams, segments_s):
    """Invalidate steps outside [(start_s, end_s), ...] (seconds from the first gaze sample), e.g. to keep only the
    task-execution parts of an Ego-Exo4D take."""
    t = (streams.t_us - streams.t_us[0]) / 1e6
    inside = np.zeros(len(t), bool)
    for a, b in segments_s:
        inside |= (t >= a) & (t <= b)
    streams.valid = streams.valid & inside
    return streams


# ------------------------------------------------------------------------------------------------- HD-EPIC, ADT (lab)
# Both follow the lab's (SensorySpotlight) timelines exactly, so `build.window.anchor=lab` finds each question's step.
# Aria's nominal device -> CPF transform (the lab's, used where no per-recording calibration exists).
T_DEVICE_CPF_NOMINAL = np.array([[-0.0319, -0.9986, -0.0415, 0.005], [0.7934, 0.0, -0.6088, -0.0491],
                                 [0.6079, -0.0523, 0.7923, -0.0492], [0, 0, 0, 1.0]])
T_DEVICE_CPF_ADT = T_DEVICE_CPF_NOMINAL.copy()
T_DEVICE_CPF_ADT[:3, 3] = [0.005, -0.0515, -0.0499]            # the lab's ADT pipeline (adt_pipeline.py T_DEV_CPF)


def list_hdepic(slam_root):
    """(rec, participant, multi-session zip, session id) for every recording whose kitchen SLAM zip is complete
    (<slam_root>/P0*/SLAM/multi/vrs_to_multi_slam.json; <i>.zip.ok marks a finished download), as the lab's
    hdepic_fix3d.recordings()."""
    out = []
    for m in sorted(glob.glob(os.path.join(slam_root, "P0*", "SLAM", "multi", "vrs_to_multi_slam.json"))):
        P = m.split(os.sep)[-4]
        for vrs, i in json.load(open(m)).items():
            z = os.path.join(os.path.dirname(m), f"{i}.zip")
            if os.path.isfile(z + ".ok"):
                out.append((os.path.basename(vrs)[:-4], P, z, str(i)))
    return out


def read_hdepic(raw_root, rec, P, zpath, i, location):
    """HD-EPIC recording `rec` of participant P: eye gaze from raw/HD-EPIC_GAZE/<P>/GAZE_HAND.zip (a zip per recording
    inside), pose / points / observations from the kitchen's multi-session SLAM zip (session i), so all recordings of a
    kitchen share one world frame. Gaze at 60 Hz is brought to 10 Hz by keeping every k-th sample from the first, as
    the lab does; T_device_cpf from HD-EPIC_INTERMEDIATE/<P>/<rec>/device_calibration.json, nominal otherwise.
    location: the lab's held-out group (the kitchen)."""
    cal = os.path.join(raw_root, "HD-EPIC_INTERMEDIATE", P, rec, "device_calibration.json")
    T = np.array(json.load(open(cal))["origin_transforms"]["T_device_cpf"]) if os.path.isfile(cal) \
        else T_DEVICE_CPF_NOMINAL

    def every_100ms(G):
        tg = G["tracking_timestamp_us"].to_numpy()
        return np.arange(0, len(G), max(1, int(round(0.1 / np.median(np.diff(tg) / 1e6)))))
    return read_mps(rec,
                    gaze_src=(os.path.join(raw_root, "HD-EPIC_GAZE", P, "GAZE_HAND.zip"), f"GAZE_HAND/mps_{rec}_vrs.zip",
                              f"mps_{rec}_vrs/eye_gaze/general_eye_gaze.csv"),
                    trajectory_src=(zpath, f"{i}/slam/closed_loop_trajectory.csv"),
                    points_src=(zpath, f"{i}/slam/semidense_points.csv.gz"),
                    observations_src=(zpath, f"{i}/slam/semidense_observations.csv.gz"),
                    T_device_cpf=T, gaze_rows=every_100ms,
                    meta={"dataset": "hdepic", "location": int(location), "participant": P})


def list_adt(root):
    """ADT sequences in `root` (the lab's selective download, tools/adt_download.py) that have their SLAM points
    (--groups mps_slam_points; not part of the lab's default download)."""
    return sorted(s for s in os.listdir(root) if s.startswith("Apartment")
                  and glob.glob(os.path.join(root, s, "*mps_slam_points*.zip")))


def lab_to_slam_rotation(gt_src, trajectory_src, max_spread_deg=2.0):
    """ADT's ground truth (the lab's world frame) and MPS SLAM (ours) are different world frames: the rotation taking
    ground-truth world directions into the SLAM world, from the two device trajectories at the same timestamps
    (R_slam R_gt^T per pose, averaged on SO(3)). Raises if the poses disagree by more than max_spread_deg."""
    cols = ["tracking_timestamp_us", "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
    A, B = _read_csv(gt_src, usecols=cols), _read_csv(trajectory_src, usecols=cols)
    ta, tb = A["tracking_timestamp_us"].to_numpy(np.int64), B["tracking_timestamp_us"].to_numpy(np.int64)
    j = np.clip(np.searchsorted(ta, tb), 1, len(ta) - 1)
    j -= (tb - ta[j - 1]) < (ta[j] - tb)
    ok = np.abs(ta[j] - tb) < POSE_TOL_US
    if ok.sum() < 10:
        raise ValueError("too few ground-truth poses at the SLAM timestamps")
    Ra, Rb = quat_to_rotmat(A[cols[1:]].to_numpy(float)[j[ok]]), quat_to_rotmat(B[cols[1:]].to_numpy(float)[ok])
    Rs = Rb @ np.swapaxes(Ra, 1, 2)
    U, _, Vt = np.linalg.svd(Rs.mean(0))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    spread = np.degrees(np.arccos(np.clip((np.trace(np.swapaxes(Rs, 1, 2) @ R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    if np.median(spread) > max_spread_deg:
        raise ValueError(f"ground-truth and SLAM poses disagree: median {np.median(spread):.1f} deg after alignment")
    return R


def read_adt(root, seq, location):
    """ADT sequence `seq`: the ground-truth eye gaze (the lab's answers come from it), taken as the lab does: every 3rd
    30 Hz sample from the first with a depth hit; pose, points and observations from MPS SLAM (one consistent frame;
    the eye gaze is in the CPF, so it needs no world frame). T_device_cpf as the lab's ADT pipeline. The lab's world
    directions (ground-truth frame) are mapped into ours with meta["R_slam_from_lab"] (lab_to_slam_rotation).
    location: the lab's held-out group (the session fold)."""
    d = os.path.join(root, seq)
    one = lambda pat: _first([os.path.join(d, pat)])                     # noqa: E731
    gt, traj, pts = one("*_main_groundtruth.zip"), one("*mps_slam_trajectories*.zip"), one("*mps_slam_points*.zip")

    def lab_steps(G):
        k = np.flatnonzero(G["depth_m"].to_numpy() > 0)
        return np.arange(k[0] if len(k) else 0, len(G), 3)
    traj_src = (traj, _zip_member(traj, "closed_loop_trajectory.csv"))
    R = lab_to_slam_rotation((gt, "aria_trajectory.csv"), traj_src)
    return read_mps(seq,
                    gaze_src=(gt, "eyegaze.csv"),
                    trajectory_src=traj_src,
                    points_src=(pts, _zip_member(pts, "semidense_points.csv.gz")),
                    observations_src=(pts, _zip_member(pts, "semidense_observations.csv.gz")),
                    T_device_cpf=T_DEVICE_CPF_ADT, gaze_rows=lab_steps,
                    meta={"dataset": "adt", "location": int(location), "R_slam_from_lab": R.tolist()})
