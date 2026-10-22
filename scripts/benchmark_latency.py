"""Latency of the online pipeline (Table 1): per-step point preprocessing (Eq. 1), span localisation (Eq. 2),
voxelisation of a 2 s window (Eq. 3) and one model forward pass. Runs on a synthetic recording.

    python scripts/benchmark_latency.py --config configs/fovs_aria.yaml --device cuda
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.config import load_config                 # noqa: E402
from egospan.data.synthetic import make_recording      # noqa: E402
from egospan.engine import resolve_device               # noqa: E402
from egospan.lift import EgoSpanLift, gravity_alignment  # noqa: E402
from egospan.models import build_model                  # noqa: E402


def stats(ms):
    return f"{np.mean(ms):8.3f} +- {np.std(ms):7.3f} ms"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    s = make_recording(seconds=30)
    lifter = EgoSpanLift(cfg.build.lift, gravity_alignment(s.gravity_world))
    spu, Tp = cfg.build.window.steps_per_unit, cfg.build.window.past_units
    pre, loc, vox, inf = [], [], [], []
    device = resolve_device(a.device)
    model = build_model(cfg.model, len(cfg.data.levels), Tp).to(device).eval()
    R = cfg.build.lift.resolution
    for i in range(a.n):
        k = Tp * spu + i
        t0 = time.perf_counter()
        p = s.points_world[s.visible(k)]
        keep = lifter.select(p, s.T_world_cpf[k])
        p = p[keep]
        t1 = time.perf_counter()
        lab = lifter.classify(p, s.T_world_cpf[k], s.gaze_cpf[k])
        t2 = time.perf_counter()
        pre.append(1e3 * (t1 - t0))
        loc.append(1e3 * (t2 - t1))
        frame = lifter.anchor(s.T_world_cpf[k])
        steps = range(k - Tp * spu + 1, k + 1)
        cached = [(s.points_world[s.visible(j)], j) for j in steps]
        cached = [(q, lifter.classify(q, s.T_world_cpf[j], s.gaze_cpf[j])) for q, j in cached]
        t3 = time.perf_counter()
        X = np.zeros((Tp, 5, R, R, R), bool)
        for r, (q, lb) in enumerate(cached):
            lifter.voxelize(q, lb, frame, out=X[r // spu])
        t4 = time.perf_counter()
        vox.append(1e3 * (t4 - t3))
        x = torch.from_numpy(X[None].astype(np.float32)).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t5 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inf.append(1e3 * (time.perf_counter() - t5))
    inf = inf[3:]                                                       # warm-up
    print(f"point preprocessing      {stats(pre)}\n3D visual span locali.  {stats(loc)}\n"
          f"voxelization            {stats(vox)}\nmodel inference ({device.type:4s})  {stats(inf)}\n"
          f"total (last step + voxelization + inference) "
          f"{np.mean(pre) + np.mean(loc) + np.mean(vox) + np.mean(inf):.3f} ms")
