"""Synthetic recordings with the structure of Aria MPS outputs, for tests and smoke runs without the real datasets.

A box-shaped room with textured walls and a few "objects" (point blobs); the wearer walks up to one object after
another, turns the head toward it, and fixates points on it with saccades. Visibility = inside a 110 deg camera cone
and < 5 m.
World frame is gravity-aligned with gravity along -z, as in MPS.
"""
import numpy as np

from .streams import RecordingStreams


def _look_at(forward, up=(0.0, 0.0, 1.0)):
    """CPF rotation (columns = CPF x, y, z axes in world) with z = forward, y ~ up, x = y cross z (left)."""
    z = forward / np.linalg.norm(forward)
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def make_recording(name="synth_0000", seconds=30.0, hz=10.0, n_wall=6000, n_objects=6, seed=0, location=1,
                   category=""):
    rng = np.random.default_rng(seed)
    room = np.array([5.0, 4.0, 2.6])
    # walls / floor / ceiling
    face = rng.integers(0, 6, n_wall)
    pts = rng.uniform(0, 1, (n_wall, 3)) * room
    ax, hi = face // 2, face % 2
    pts[np.arange(n_wall), ax] = hi * room[ax]
    # objects: dense blobs at table height
    centers = np.column_stack([rng.uniform(0.6, room[0] - 0.6, n_objects), rng.uniform(0.6, room[1] - 0.6, n_objects),
                               rng.uniform(0.7, 1.4, n_objects)])
    blobs = np.concatenate([c + rng.normal(0, 0.08, (400, 3)) for c in centers])
    xyz = np.concatenate([pts, blobs]).astype(np.float32)
    # wearer: walks up to a target object (stops ~0.8 m away) that changes every few seconds, head turned toward it
    T = int(seconds * hz)
    t = np.arange(T) / hz
    target_obj = (t // rng.uniform(2.0, 4.0)).astype(int) % n_objects
    away = room / 2 - centers                                       # stand on the room-centre side of the object
    away[:, 2] = 0
    stand = centers + 0.8 * away / (np.linalg.norm(away, axis=1, keepdims=True) + 1e-9)
    stand[:, 2] = 1.6
    pos = np.zeros((T, 3))
    pos[0] = [room[0] / 2, room[1] / 2, 1.6]
    for i in range(1, T):
        step = stand[target_obj[i]] - pos[i - 1]
        pos[i] = pos[i - 1] + step * min(1.0, 0.1 / (np.linalg.norm(step) + 1e-9))     # walk at <= 1 m/s
    look = centers[target_obj] - pos
    alpha = 0.15                                                   # smooth head turning
    for i in range(1, T):
        look[i] = (1 - alpha) * look[i - 1] + alpha * look[i]
    T_world_cpf = np.tile(np.eye(4), (T, 1, 1))
    for i in range(T):
        T_world_cpf[i, :3, :3] = _look_at(look[i])
        T_world_cpf[i, :3, 3] = pos[i]
    # gaze: fixate a random blob point of the target object, re-sampled at ~3 Hz saccades
    gaze_cpf = np.zeros((T, 3))
    fix = None
    for i in range(T):
        if fix is None or rng.random() < 0.3:
            o = target_obj[i]
            fix = blobs[o * 400 + rng.integers(0, 400)]
        v = T_world_cpf[i, :3, :3].T @ (fix - pos[i])
        gaze_cpf[i] = v / np.linalg.norm(v)
    # visibility: within 55 deg of the head forward axis, < 5 m, random dropout
    ptr, idx = [0], []
    for i in range(T):
        loc = (xyz - pos[i]) @ T_world_cpf[i, :3, :3]
        d = np.linalg.norm(loc, axis=1)
        vis = np.flatnonzero((loc[:, 2] / (d + 1e-9) > np.cos(np.radians(55))) & (d < 5.0) &
                             (rng.random(len(xyz)) < 0.5))
        idx.append(vis)
        ptr.append(ptr[-1] + len(vis))
    return RecordingStreams(name=name, t_us=(t * 1e6).astype(np.int64), T_world_cpf=T_world_cpf, gaze_cpf=gaze_cpf,
                            valid=np.ones(T, bool), points_world=xyz, obs_ptr=np.array(ptr, np.int64),
                            obs_idx=np.concatenate(idx).astype(np.int64),
                            points_dist_std=np.full(len(xyz), 0.01, np.float32),
                            gravity_world=np.array([0.0, 0.0, -9.81]),
                            meta={"dataset": "synthetic", "location": location, "category": category})
