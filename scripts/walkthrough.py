"""Step-by-step walkthrough on one synthetic recording: raw streams -> EgoSpanLift (Eq. 1-3) -> forecasting sample ->
every stage of the forecasting network. Prints what happens at each step and saves a figure.

    python scripts/walkthrough.py --out walkthrough.png
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.build import WindowConfig, extract_samples, unpack_grids   # noqa: E402
from egospan.data.synthetic import make_recording                           # noqa: E402
from egospan.lift import CHANNELS, LEVELS, EgoSpanLift, LiftConfig, gravity_alignment   # noqa: E402
from egospan.models import SpanForecaster, count_parameters                 # noqa: E402


def title(s):
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="walkthrough.png")
    a = ap.parse_args()
    np.set_printoptions(precision=3, suppress=True)

    # ------------------------------------------------------------------------------------------- 0. raw streams
    title("0. RAW STREAMS (what Aria MPS gives us; here a synthetic recording)")
    s = make_recording(seconds=10, seed=3)
    print(f"timeline: {len(s)} steps at 10 Hz (gaze timestamps), {(s.t_us[-1] - s.t_us[0]) / 1e6:.1f} s")
    print(f"semidense keypoints in the world: {len(s.points_world)}  (N x 3, metres)")
    print(f"per step: CPF pose T_world_cpf {s.T_world_cpf.shape[1:]}, gaze direction in CPF {s.gaze_cpf.shape[1:]}, "
          f"and the ids of points seen in that 100 ms bin (mean {np.diff(s.obs_ptr).mean():.0f} points/step)")

    cfg, win = LiftConfig(), WindowConfig()
    lifter = EgoSpanLift(cfg, gravity_alignment(s.gravity_world))
    P = win.past_units * win.steps_per_unit           # 20 observed steps
    F = win.future_units * win.steps_per_unit         # 20 future steps
    t0 = P - 1                                        # first sample: last observed step = 19
    k = t0
    print(f"\nsample window: observed steps {t0 - P + 1}..{t0} (2 s), forecast steps {t0 + 1}..{t0 + F} (2 s)")
    print(f"anchor = user's pose at step {t0}; every grid of this sample is centred there")

    # ------------------------------------------------------------------------------------------- 1. EgoSpanLift
    title(f"1. EgoSpanLift ON ONE STEP (k = {k})")
    T, g = s.T_world_cpf[k], s.gaze_cpf[k]
    frame = lifter.anchor(s.T_world_cpf[t0])
    vis = s.visible(k)
    pts = s.points_world[vis]
    print(f"user position (world) {T[:3, 3]}, head forward (world) {T[:3, 2]}, gaze in CPF {g}")
    print(f"visible points at this step: {len(pts)}")

    print("\n-- Eq. 1  select: keep points inside a D = 3.2 m cube around the user, then statistical outlier removal")
    local = lifter.anchor(T).points(pts)
    in_cube = (np.abs(local) < cfg.cube_size / 2).all(1)
    keep = lifter.select(pts, T)
    print(f"   inside cube: {in_cube.sum()} / {len(pts)}   after outlier removal: {keep.sum()}")
    p = pts[keep]

    print("\n-- Eq. 2  classify: angle between each point (in the CPF) and the gaze ray / head-forward axis")
    lab = lifter.classify(p, T, g)
    loc_cpf = (p - T[:3, 3]) @ T[:3, :3]
    u = loc_cpf / np.linalg.norm(loc_cpf, axis=1, keepdims=True)
    ang_gaze = np.degrees(np.arccos(np.clip(u @ (g / np.linalg.norm(g)), -1, 1)))
    ang_head = np.degrees(np.arccos(np.clip(u[:, 2], -1, 1)))
    for c, name in enumerate(CHANNELS):
        rule = {0: "angle to head-forward < 55 deg", 1: "angle to gaze < 30 deg", 2: "angle to gaze < 8 deg",
                3: "angle to gaze < 2 deg", 4: "every selected point"}[c]
        print(f"   {name:12s} {rule:32s} -> {lab[c].sum():5d} points")
    j = np.argsort(ang_gaze)[:3]
    print(f"   e.g. the 3 points closest to the gaze ray: angles {ang_gaze[j]} deg (head: {ang_head[j]} deg)")

    print("\n-- Eq. 3  voxelise: 16^3 grid of 20 cm cells around the anchor;"
          " a cell is on if >= 1 labelled point is in it")
    grid = lifter.voxelize(p, lab, frame)
    print(f"   grid shape {grid.shape} = (channels, X, Y, Z)")
    for c, name in enumerate(CHANNELS):
        print(f"   {name:12s} {grid[c].sum():4d} occupied voxels of {grid[c].size}  ({100 * grid[c].mean():.2f} %)")
    ijk, _ = lifter.voxel_index(T[:3, 3][None], frame)
    print(f"   the user's own voxel at this step: {ijk[0]} (anchor = grid centre 8,8,8 at step {t0})")

    # ------------------------------------------------------------------------------------------- 2. sample
    title("2. FROM STEPS TO ONE TRAINING SAMPLE (union per 1 s unit)")
    X = np.zeros((win.past_units, 5, 16, 16, 16), bool)
    YF = np.zeros((win.future_units, 4, 16, 16, 16), bool)
    for rel, kk in enumerate(range(t0 - P + 1, t0 + F + 1)):
        q = s.points_world[s.visible(kk)]
        q = q[lifter.select(q, s.T_world_cpf[kk])]
        lb = lifter.classify(q, s.T_world_cpf[kk], s.gaze_cpf[kk])
        if rel < P:
            lifter.voxelize(q, lb, frame, out=X[rel // 10])            # steps 0-9 -> second 1, 10-19 -> second 2
        else:
            lifter.voxelize(q, lb[:4], frame, out=YF[(rel - P) // 10])
    Y = YF.any(0)                                                      # target = union over the whole 2 s future
    print("input X  (Tp=2, 5, 16, 16, 16): occupied voxels")
    for t in range(2):
        print(f"   second {t + 1}: " + "  ".join(f"{n} {X[t, c].sum():3d}" for c, n in enumerate(CHANNELS)))
    print("target Y (4, 16, 16, 16): occupied voxels")
    print("   future 2 s: " + "  ".join(f"{n} {Y[c].sum():3d}" for c, n in enumerate(LEVELS)))
    arr = extract_samples(s, cfg, win)
    Xb = unpack_grids(arr["X"][0], (2, 5, 16, 16, 16))
    Yb = unpack_grids(arr["YF"][0], (2, 4, 16, 16, 16)).any(0)
    print(f"\nsame as the dataset builder (extract_samples) produced? "
          f"X {np.array_equal(X, Xb)}, Y {np.array_equal(Y, Yb)}")
    print(f"this recording gives {len(arr['X'])} samples (one per second); stored bit-packed: "
          f"X {arr['X'].shape[1]} bytes + Y {arr['YF'].shape[1]} bytes per sample")

    # ------------------------------------------------------------------------------------------- 3. network
    title("3. FORECASTING NETWORK (paper size: C = 1024)")
    torch.manual_seed(0)
    net = SpanForecaster(num_steps=2).eval()
    print(f"parameters: {count_parameters(net) / 1e6:.1f} M")
    shapes = []
    hook = lambda name: (lambda m, i, o: shapes.append((name, tuple(o.shape))))   # noqa: E731
    for i, enc in enumerate(net.encoders):
        enc.register_forward_hook(hook(f"encoder {i + 1}"))
    net.transformer.register_forward_hook(hook("transformer"))
    for i, dec in enumerate(net.decoders):
        dec.register_forward_hook(hook(f"decoder {i + 1}"))
    net.final_conv.register_forward_hook(hook("final 1x1 conv"))
    x = torch.from_numpy(X[None].astype(np.float32))
    with torch.no_grad():
        logits = net(x)
    print(f"input x                {tuple(x.shape)}   batch, Tp observed seconds, 5 channels, 16^3 grid")
    print(f"+ global grid          {(1, 3, 5, 16, 16, 16)}   union of the 2 seconds appended as a 3rd grid (head)")
    print("all 3 grids go through the SAME encoder as a batch of 3:")
    for name, shp in shapes:
        if name.startswith("encoder"):
            print(f"  {name:20s} {shp}")
    print(f"tokens                 {(1, 3, 1024)}   deepest features (2^3) averaged -> one 1024-d vector per grid")
    print("causal mask (0 = may attend, -inf = blocked); rows/cols = [second 1, second 2, global head]:")
    print(net.causal_mask.numpy())
    print(f"{'transformer':22s} {dict(shapes)['transformer']}   -> keep the output at the head position: (1, 1024)")
    print("decoder: head vector broadcast to 2^3 and added to the deepest skip (1, 1024, 2, 2, 2), then upsampled:")
    for name, shp in shapes:
        if name.startswith("decoder") or name.startswith("final"):
            print(f"  {name:20s} {shp}")
    prob = torch.sigmoid(logits)[0].numpy()
    print(f"sigmoid -> soft occupancy {prob.shape}: one 16^3 map per level "
          f"(untrained, so ~{prob.mean():.2f} everywhere)")
    print("training compares it with Y using the Dice loss; prediction > 0.5 gives the forecast voxels")

    # ------------------------------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gp = frame.points(p)
    user = frame.points(T[:3, 3][None])[0]
    gaze_w = T[:3, :3] @ (g / np.linalg.norm(g))
    gaze_g = frame.dirs(gaze_w[None])[0]
    fig = plt.figure(figsize=(18, 5.5))
    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(gp[:, 0], gp[:, 1], s=2, c="#bbbbbb", label="scene (selected points)")
    for c, col in ((0, "#9db4c0"), (1, "#5c9ead"), (2, "#326273"), (3, "#e3542b")):
        ax.scatter(gp[lab[c], 0], gp[lab[c], 1], s=4 + 4 * c, c=col, label=CHANNELS[c])
    ax.arrow(user[0], user[1], 1.2 * gaze_g[0], 1.2 * gaze_g[1], color="k", width=0.01, label="gaze")
    ax.plot(*user[:2], "k^", ms=10, label="user")
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_title(f"Eq. 1-2: points at step {k}, top view (grid x-y, m)")
    ax.legend(fontsize=7, loc="lower left")
    for i, (G, ttl) in enumerate(((X.any(0), "Eq. 3 + union: input X (2 s observed)"),
                                  (np.concatenate([Y, X[:, 4:].any(0)]), "target Y (2 s future)"))):
        ax3 = fig.add_subplot(1, 3, 2 + i, projection="3d")
        ax3.voxels(G[4], facecolors="#00000012", shade=False)
        for c, col in ((1, "#5c9ead55"), (2, "#326273aa"), (3, "#e3542bff")):
            if G[c].any():
                ax3.voxels(G[c], facecolors=col, shade=False)
        ax3.set_title(ttl)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z (up)")
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"\nfigure: {a.out}")


if __name__ == "__main__":
    main()
