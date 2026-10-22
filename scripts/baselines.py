"""Non-learned baselines on a packed benchmark (test split), cf. Table 2.

    python scripts/baselines.py --config configs/fovs_aria.yaml
    python scripts/baselines.py --config configs/fovs_aria.yaml --gaze-preds csts_test.npz --name CSTS

  Global Prior                      train-target voxel frequency, threshold tuned on val
  Persistence (last unit / all)     previous span copied forward
  2D Center Prior + EgoSpanLift     gaze fixed at the image centre, head poses extrapolated (GP), lifted on the scene
  <name> + EgoSpanLift              2D gaze anticipation predictions from another model (--gaze-preds): an .npz with
                                    `yaw_pitch` (n_test, future_steps, 2) in radians in the CPF, rows in test order
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.baselines import global_prior, lift_2d_gaze_predictions, persistence   # noqa: E402
from egospan.config import load_config                                              # noqa: E402
from egospan.data.dataset import PackedBenchmark                                    # noqa: E402
from egospan.data.streams import yaw_pitch_to_dir                                   # noqa: E402
from egospan.metrics import evaluate, format_report, tune_thresholds                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--pose", default="gp", choices=["gp", "velocity", "static"])
    ap.add_argument("--gaze-preds", help="npz with yaw_pitch (n_test, future_steps, 2)")
    ap.add_argument("--name", default="2D model")
    ap.add_argument("--out", help="json file for all reports")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    d = cfg.data
    bench = PackedBenchmark(d.root)
    levels, H = d.levels, d.horizon
    idx = {p: bench.indices(p, d.fold) for p in ("train", "val", "test")}
    Y = {p: bench.target(idx[p], H, levels) for p in idx}
    Xt = bench.inputs(idx["test"])
    lift = bench.meta["lift"]
    vox_cm = 100 * lift["cube_size"] / bench.R
    spu = bench.meta["window"]["steps_per_unit"]
    past_steps, fut_steps = bench.past_units * spu, (H or bench.Tf) * spu
    bins_per_unit = bench.Tp // bench.past_units                 # > 1 with finer temporal input bins
    reports = {}

    def report(name, prob_t, thr=0.5):
        rep = evaluate(prob_t.astype(np.float32), Y["test"], levels, thr, vox_cm, saliency=False)
        reports[name] = rep
        print(f"\n== {name}\n" + format_report(rep, levels), flush=True)

    prior = global_prior(Y["train"])
    thr = tune_thresholds(np.broadcast_to(prior, Y["val"].shape).astype(np.float32), Y["val"])
    report("Global Prior", np.broadcast_to(prior, Y["test"].shape), thr)
    report("Persistence (last unit)", persistence(Xt, levels, bins_per_unit).astype(np.float32))
    report("Persistence (all units)", persistence(Xt, levels, bench.Tp).astype(np.float32))

    if "poses_grid" not in bench.arrays:
        print("\n(no per-step poses in this benchmark, e.g. converted lab data: skipping the 2D-lift baselines)")
        if a.out:
            json.dump(reports, open(a.out, "w"), indent=1)
        return
    poses = bench.arrays["poses_grid"][idx["test"]]
    lift_kw = dict(cube_size=lift["cube_size"], orientation_deg=lift["orientation_deg"],
                   cone_deg=tuple(lift["cone_deg"]))
    sel = [["orientation", "peripheral", "central", "foveal"].index(l) for l in levels]
    center = np.zeros((len(Xt), fut_steps, 3))
    center[..., 2] = 1.0
    P = lift_2d_gaze_predictions(Xt, poses, center, past_steps, fut_steps, a.pose, **lift_kw)
    report(f"2D Center Prior + EgoSpanLift + pose[{a.pose}]", P[:, sel].astype(np.float32))
    if a.gaze_preds:
        yp = np.load(a.gaze_preds)["yaw_pitch"]
        assert yp.shape[:2] == (len(Xt), fut_steps), f"expected ({len(Xt)}, {fut_steps}, 2), got {yp.shape}"
        g = yaw_pitch_to_dir(yp[..., 0], yp[..., 1])
        P = lift_2d_gaze_predictions(Xt, poses, g, past_steps, fut_steps, a.pose, **lift_kw)
        report(f"{a.name} + EgoSpanLift + pose[{a.pose}]", P[:, sel].astype(np.float32))
    if a.out:
        json.dump(reports, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
