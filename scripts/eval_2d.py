"""3D forecast -> 2D gaze anticipation (Sec. 5.3), scored by angular error against the true future gaze.

Uses <run>/probs_test.npy from scripts/evaluate.py (the run must include the foveal level).

    python scripts/eval_2d.py --run runs/fovs_aria/egospanlift_s0
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.dataset import PackedBenchmark                                        # noqa: E402
from egospan.engine import load_run                                                    # noqa: E402
from egospan.project2d import angular_error_deg, foveal_target_cpf, interpolate_gaze   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    _, cfg = load_run(a.run)
    bench = PackedBenchmark(cfg.data.root)
    idx = bench.indices("test", cfg.data.fold)
    prob = np.load(os.path.join(a.run, "probs_test.npy")).astype(np.float32)
    fov = prob[:, cfg.data.levels.index("foveal")]
    spu = bench.meta["window"]["steps_per_unit"]
    P, F = bench.past_units * spu, (cfg.data.horizon or bench.Tf) * spu
    anchor = bench.arrays["poses_grid"][idx, P - 1]
    g = bench.arrays["gaze_cpf"][idx]
    g_now, g_true = g[:, P - 1], g[:, P:P + F]
    pred = interpolate_gaze(g_now, foveal_target_cpf(fov, anchor, bench.meta["lift"]["cube_size"]), F)
    still = np.repeat(g_now[:, None], F, 1)
    center = np.zeros_like(g_true)
    center[..., 2] = 1
    for name, p in (("ours (3D -> 2D)", pred), ("current gaze held", still), ("center prior", center)):
        e = angular_error_deg(p, g_true)
        print(f"{name:20s} mean angular error {np.mean(e):6.2f} deg   at +{F / spu:.0f}s {np.mean(e[:, -1]):6.2f} deg")
