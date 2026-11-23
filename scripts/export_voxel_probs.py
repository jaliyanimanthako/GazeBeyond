"""EgoSpanLift's own 3D output at the lab's WHERE questions, so it can be read at objects' 3D locations (SensorySpotlight
tools/hdepic_next_object.py compare, NO_VOXELS). export_lab_probs.py where max-pools each direction's distance cells
into the lab's 960 directions; this keeps every cell: the paper's R^3 Cartesian voxels (20 cm in the 3.2 m cube around
the CPF) or the (azimuth, elevation, distance) cells of the spherical grid, whichever the benchmark was built on.

    S=~/SensorySpotlight/data/traj_forecast/attention_shift/senses_suite
    python scripts/export_voxel_probs.py --bench data/hdepic_labq_where --suite $S --tag hdepic_lag \
        --runs runs/hdepic_labq/ft_f{1,2,...} --out $S/voxels_gbf_retrained_novid_hdepic.npz

Writes qid (question rows of <suite>/data_<tag>.npz), p (questions, x, y, z voxels or azimuth, elevation, distance
cells) float16, the --level sigmoid per cell as predicted, resolution and cube size (Cartesian) or the grid's edges
and cube size (spherical), and M (signed permutation: grid frame = lab head frame
@ M.T, rows; checked as in export_lab_probs.py). Each question is scored by the run of its own CV fold.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from egospan.data.dataset import PackedBenchmark                    # noqa: E402
from egospan.engine import resolve_device                            # noqa: E402
from export_lab_probs import run_predictions                         # noqa: E402
from export_where_geometry import frame_map, from_azel               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="a WHERE benchmark built with build.window.anchor=lab")
    ap.add_argument("--runs", nargs="+", required=True, help="trained runs, one per CV fold")
    ap.add_argument("--suite", required=True, help="folder with the lab's data_<tag>.npz")
    ap.add_argument("--tag", default="hdepic_lag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="foveal")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-frame-err", type=float, default=10.0)
    a = ap.parse_args()
    bench = PackedBenchmark(a.bench)
    if "land_dir" not in bench.arrays or "qid" not in bench.arrays:
        raise SystemExit(f"{a.bench} is not a WHERE benchmark built with build.window.anchor=lab")
    Q = np.load(os.path.join(a.suite, f"data_{a.tag}.npz"))
    qid = bench.arrays["qid"].astype(int)
    if qid.max() >= len(Q["rec"]) or not np.array_equal(Q["rec"].astype(str)[qid], bench.arrays["rec"].astype(str)):
        raise SystemExit(f"the benchmark's questions are not rows of {a.suite}/data_{a.tag}.npz (another suite?)")
    M, err = frame_map(from_azel(*Q["truth"].T)[qid], bench.arrays["land_dir"].astype(np.float64))
    print(f"landing direction lab vs ours: median {err:.2f} deg with axes "
          f"{'unchanged' if np.allclose(M, np.eye(3)) else M.tolist()}")
    if err > a.max_frame_err:
        raise SystemExit(f"the frames disagree ({err:.1f} deg > {a.max_frame_err})")
    prob, covered = run_predictions(bench, a.runs, a.level, resolve_device(a.device), a.batch_size)
    s = np.flatnonzero(covered)
    if len(np.unique(qid[s])) != len(s):
        raise SystemExit("several samples per question: expected one")
    g = bench.grid
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = dict(qid=qid[s], p=prob[s].reshape(len(s), *bench.shape).astype(np.float16), M=M)
    if g is None:                                                      # the paper's R^3 voxels, centred on the CPF
        out |= dict(resolution=bench.R, cube_size=float(bench.meta["lift"]["cube_size"]))
    else:
        out |= dict(yaw_edges=np.asarray(g.yaw_edges), pitch_edges=np.asarray(g.pitch_edges),
                    depth_edges=np.asarray(g.depth_edges), cube_size=g.cube_size)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}: {len(s)} questions x {bench.shape} {'voxels' if g is None else 'cells'} ({a.level})")


if __name__ == "__main__":
    main()
