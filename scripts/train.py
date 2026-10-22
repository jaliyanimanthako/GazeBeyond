"""Train a 3D visual span forecaster, then score the best-validation epoch on the test split.

    python scripts/train.py --config configs/fovs_aria.yaml
    python scripts/train.py --config configs/fovs_aria.yaml --set train.seed=1 train.out_dir=runs/fovs_aria/full_s1

Ablations of Table 2 (one --set each):
    loss                    train.loss=bce
    w/o global embedding    model.head=none
    w/o previous span       data.drop_span_input=true
    single-task             data.levels=[foveal]          (repeat for each level)
Temporal window (Fig. 4-b)  data.horizon=1 | 2 | 3 | 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.config import load_config   # noqa: E402
from egospan.engine import train         # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides section.key=value")
    a = ap.parse_args()
    train(load_config(a.config, a.set))
