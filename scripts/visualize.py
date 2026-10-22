"""Render previous span / forecast / ground truth of test samples as 3D voxel plots (cf. Fig. 5).

    python scripts/visualize.py --run runs/fovs_aria/egospanlift_s0 --index 0 5 10 --out figs/
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.dataset import PackedBenchmark   # noqa: E402
from egospan.engine import load_run                # noqa: E402

COLORS = {"orientation": "#9db4c0", "peripheral": "#5c9ead", "central": "#326273", "foveal": "#e39774"}


def draw(ax, scene, spans, levels, title):
    ax.voxels(scene, facecolors="#00000010", shade=False)
    for lev in ("orientation", "peripheral", "central", "foveal"):
        if lev in levels:
            g = spans[levels.index(lev)]
            if g.any():
                ax.voxels(g, facecolors=COLORS[lev] + ("ff" if lev == "foveal" else "55"), shade=False)
    ax.set_title(title)
    ax.set_axis_off()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--index", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    _, cfg = load_run(a.run)
    bench = PackedBenchmark(cfg.data.root)
    idx = bench.indices("test", cfg.data.fold)
    prob = np.load(os.path.join(a.run, "probs_test.npy")).astype(np.float32)
    levels = cfg.data.levels
    os.makedirs(a.out, exist_ok=True)
    for i in a.index:
        x = bench.inputs(idx[i])
        y = bench.target(idx[i], cfg.data.horizon, levels)
        prev = x[:, [["orientation", "peripheral", "central", "foveal"].index(l) for l in levels]].any(0)
        scene = x[:, 4].any(0)
        fig = plt.figure(figsize=(15, 5))
        for k, (spans, title) in enumerate(((prev, "previous span"), (prob[i] > 0.5, "forecast"), (y, "ground truth"))):
            draw(fig.add_subplot(1, 3, k + 1, projection="3d"), scene, spans, levels, title)
        path = os.path.join(a.out, f"sample_{i:05d}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(path)
