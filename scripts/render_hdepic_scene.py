"""An HD-EPIC recording as a 3D scene for figures, in the same viewer as render_adt_scene.py: the kitchen's SLAM point
cloud (one session, or with --kitchen every session of the kitchen, one world frame), the moved objects of HD-EPIC's
scene-and-object-movements annotation as labelled dots where they rest at the chosen moment, the head path, the
fixations and, at that moment, the current gaze and the next attention shift (its landing, the object it lands on).
HD-EPIC has no ground-truth gaze depth: each fixation's 3D point is where its gaze ray meets the point cloud.

    python scripts/render_hdepic_scene.py --rec P01-20240202-110250 --kitchen --list       # teaser moments
    python scripts/render_hdepic_scene.py --rec P01-20240202-110250 --kitchen --out ~/GazeBeyond/figs/hdepic_scene

Paths default to the lab server's (--slam-root, --src, --ann). Recording names: render_scene.py hdepic --list.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from egospan.data.aria import read_gaze_pose                                    # noqa: E402
from render_adt_scene import find_shifts, fixations, rank_shifts, write_page    # noqa: E402
from render_scene import height_colors, load_points, sources                    # noqa: E402


def ray_depth(P, origins, dirs, cone_deg=1.5, chunk=16):
    """distance along each ray to the nearest surface of the point cloud P: the 10th percentile of the distances of the
    points within cone_deg of the ray (nan with fewer than 3 such points)."""
    tan2 = np.tan(np.radians(cone_deg)) ** 2
    out = np.full(len(origins), np.nan)
    for i in range(0, len(origins), chunk):
        v = P[None] - origins[i:i + chunk, None].astype(np.float32)
        along = np.einsum("knc,kc->kn", v, dirs[i:i + chunk].astype(np.float32))
        near = (along > 0.15) & ((v * v).sum(-1) - along ** 2 < along ** 2 * tan2)
        for j in range(len(v)):
            a = along[j][near[j]]
            if len(a) >= 3:
                out[i + j] = np.percentile(a, 10)
    return out


def moved_objects(ann, raw, rec, t_abs):
    """[(name, category, motion, R, xyz, size)] of the recording's moved objects resting at absolute time t_abs (s):
    at the first pick-up location before the first movement, then at each put-down; held objects (or after a movement
    with no put-down) are left out. As hdepic_next_object.py (SensorySpotlight)."""
    d = os.path.join(ann, "scene-and-object-movements")
    ft_file = os.path.join(raw, "HD-EPIC_INTERMEDIATE", rec[:3], rec, "framewise_info.jsonl")
    if not os.path.isfile(os.path.join(d, "assoc_info.json")) or not os.path.isfile(ft_file):
        print(f"no object annotation ({d}/assoc_info.json, {ft_file}): no objects")
        return []
    A = json.load(open(os.path.join(d, "assoc_info.json"))).get(rec, {})
    MI = json.load(open(os.path.join(d, "mask_info.json"))).get(rec, {})
    ft = np.array([json.loads(line)["tracking_timestamp_ns"] for line in open(ft_file)]) / 1e9
    fr = lambda m: ft[min(int(m["frame_number"]), len(ft) - 1)]                  # noqa: E731
    out = []
    for v in A.values():
        moves = []                                                               # (t up, t down, xyz up, xyz down)
        for tr in v.get("tracks", []):
            ms = sorted((MI[m] for m in tr.get("masks", []) if m in MI and MI[m].get("3d_location") is not None
                         and MI[m].get("frame_number") is not None), key=lambda m: m["frame_number"])
            if ms:
                end = ms[-1] if len(ms) > 1 else None
                moves.append((fr(ms[0]), fr(end) if end else np.inf, ms[0]["3d_location"],
                              end["3d_location"] if end else None))
        if not moves:
            continue
        moves.sort(key=lambda m: m[0])
        if t_abs < moves[0][0]:
            xyz = moves[0][2]
        else:
            up, down, _, x_down = [m for m in moves if m[0] <= t_abs][-1]
            xyz = x_down if (down <= t_abs and x_down is not None) else None
        if xyz is not None:
            out.append((v.get("name", "?"), "moved object", "", np.eye(3), np.asarray(xyz, float), np.full(3, 0.07)))
    print(f"{len(out)} annotated objects resting at that moment (of {len(A)})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--slam-root", default="/data/disk2/hdepic_dinithi/HD-EPIC/SLAM-and-Gaze")
    ap.add_argument("--src", default="/data/disk2/hdepic/HD-EPIC/raw", help="HD-EPIC raw/ (gaze, calibration, frames)")
    ap.add_argument("--ann", default="/data/disk2/hdepic_dinithi/annotations", help="HD-EPIC annotations folder")
    ap.add_argument("--kitchen", action="store_true", help="points of every session of the kitchen")
    ap.add_argument("--list", action="store_true", help="print the best teaser moments and exit")
    ap.add_argument("--t", type=float, help="the moment to open at (s from the start); default: the top --list shift")
    ap.add_argument("--fov-half", type=float, default=45.0, help="half-angle of the view cone (deg)")
    ap.add_argument("--max-points", type=int, default=800_000, help="random subset of the points for the page")
    ap.add_argument("--max-inv-dist-std", type=float, default=0.001)
    ap.add_argument("--max-dist-std", type=float, default=0.05)
    ap.add_argument("--out", default="~/GazeBeyond/figs/hdepic_scene")
    a = ap.parse_args()
    a.cmd = "hdepic"

    gaze_src, traj_src, pts_src, T, _ = sources(a)

    def every_33ms(G):                                         # 60 Hz gaze -> 30 Hz
        tg = G["tracking_timestamp_us"].to_numpy()
        return np.arange(0, len(G), max(1, int(round(1 / 30 / np.median(np.diff(tg) / 1e6)))))
    tus, g_cpf, T_wc, ok, gravity = read_gaze_pose(gaze_src, traj_src, T, every_33ms)
    t = (tus - tus[0]) / 1e6
    eye, wdir, fwd = T_wc[:, :3, 3], np.einsum("nij,nj->ni", T_wc[:, :3, :3], g_cpf), T_wc[:, :3, 2]
    ok &= np.isfinite(wdir).all(1)
    xyz = load_points(pts_src, a.max_inv_dist_std, a.max_dist_std)

    # each fixation's depth: its gaze ray against the point cloud (a subset is enough)
    P = xyz[np.random.default_rng(0).permutation(len(xyz))[:300_000]]
    F0 = fixations(t, wdir, np.full_like(eye, np.nan), ok)
    mid = np.searchsorted(t, [(f0 + f1) / 2 for f0, f1, _, _ in F0])
    depth = ray_depth(P, eye[mid], np.array([d for _, _, d, _ in F0]))
    p = np.full_like(eye, np.nan)
    for (f0, f1, _, _), dep in zip(F0, depth):
        if np.isfinite(dep):
            k = (t >= f0) & (t <= f1)
            p[k] = eye[k] + dep * wdir[k]
    print(f"fixation depth from the points: {np.isfinite(depth).mean():.0%} of {len(F0)} fixations")
    F, shifts = find_shifts(t, eye, fwd, wdir, p, ok, a.fov_half)

    t_abs = lambda s: tus[0] / 1e6 + s                                           # noqa: E731
    objects = moved_objects(a.ann, a.src, a.rec, t_abs(a.t if a.t is not None else t[-1] / 2))
    rank = rank_shifts(shifts, objects)
    if a.list:
        sys.exit()
    t_open = a.t
    if t_open is None:
        t_open = rank[0]["t"] - 0.3 if rank else t[-1] / 2
        objects = moved_objects(a.ann, a.src, a.rec, t_abs(t_open))
        rank_shifts(shifts, objects, show=0)

    up = -gravity if gravity is not None else None
    sub = np.random.default_rng(1).permutation(len(xyz))[:a.max_points]
    h = xyz[sub] @ (up if up is not None else np.array([0, 0, 1.0])).astype(np.float32)
    rgb = (0.55 * height_colors(h) + 0.45 * 244).astype(np.uint8)              # muted, towards the page background
    name = a.rec + ("_kitchen" if a.kitchen else "")
    write_page(os.path.join(os.path.abspath(os.path.expanduser(a.out)), f"{name}.html"), name, t, eye, fwd, wdir, p,
               ok, F, shifts, objects, t_open, a.fov_half, up=up, points=(xyz[sub], rgb), marker_objects=True)
