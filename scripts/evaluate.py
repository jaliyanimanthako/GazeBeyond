"""Score a trained run (writes <run>/metrics_<split>.json and <run>/probs_<split>.npy).

    python scripts/evaluate.py --run runs/fovs_aria/egospanlift_s0
    python scripts/evaluate.py --run runs/fovs_aria/egospanlift_s0 --threshold tune     # per-level thresholds on val
    python scripts/evaluate.py --run runs/... --data-root data/fovs_aria_corrupt        # e.g. SLAM sensitivity data
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.engine import evaluate_run   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", default=None, help="a number or 'tune' (default: the run's eval.threshold)")
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    thr = a.threshold if a.threshold in (None, "tune") else float(a.threshold)
    evaluate_run(a.run, split=a.split, threshold=thr, data_root=a.data_root)
