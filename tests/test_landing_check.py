"""landing_check: planted landings are counted inside / outside the right past regions, the chance level and model
recall are right, shift groups split correctly, and the script runs end to end on converted lab data."""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

from egospan.lift import CHANNELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("landing_check", os.path.join(ROOT, "scripts", "landing_check.py"))
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)
R = 16
O, P, C, F, S = (CHANNELS.index(c) for c in ("orientation", "peripheral", "central", "foveal", "scene"))


class FakeBench:
    """inputs (n, Tp, 5, R, R, R) and future foveal (n, R, R, R) given directly."""
    R, vertical_axis, meta = R, 1, {"lift": {"cube_size": 3.2}}

    def __init__(self, X, Yf):
        self.X, self.Yf = X, Yf

    def inputs(self, j):
        return self.X[j]

    def target(self, j, horizon=None, levels=("foveal",)):
        return self.Yf[j][:, None]


def planted():
    n = 2
    X = np.zeros((n, 2, 5, R, R, R), bool)
    Y = np.zeros((n, R, R, R), bool)
    a, b, c = (8, 8, 12), (8, 3, 12), (2, 2, 2)                        # c: behind the wearer
    # sample 0: looked at a (second 1), saw b out of the corner of the eye (second 2); then looks at a and b
    X[0, 0, F][a] = X[0, 0, C][a] = X[0, 0, P][a] = X[0, 0, O][a] = True
    X[0, 1, P][b] = X[0, 1, O][b] = X[0, :, S][:, b[0], b[1], b[2]] = True
    Y[0][a] = Y[0][b] = True
    # sample 1: the eyes land on c, which was in no past region
    X[1, 0, F][a] = True
    Y[1][c] = True
    return X, Y, a, b, c


def test_planted_coverage_chance_and_recall():
    X, Y, a, b, c = planted()
    pred = np.zeros((2, R, R, R), np.float16)
    pred[0][a] = 0.9                                                    # the model stays where it looked
    s = lc.check(FakeBench(X, Y), np.arange(2), probs=pred)["all"]
    assert s["samples"] == 2 and s["with_landing"] == 1.0
    cov = s["covered"]
    assert cov["past peripheral"] == 0.5 and cov["seen, not looked at"] == 0.5 and cov["past orientation"] == 0.5
    assert cov["past central"] == 0.0 and cov["past scene"] == 0.5
    assert cov["front half"] == 0.5                                     # b in front, c behind
    # the only candidate surface of sample 0 is b and it is hit: density 1; sample 1 has none
    assert s["candidate_density"]["seen, not looked at"] == 1.0
    assert s["candidates_median"]["past peripheral"] == 0.5
    r = s["model_recall"]
    assert r["start"] == 1.0 and r["landing in past peripheral"] == 0.0 and r["landing outside past peripheral"] == 0.0


def test_shift_groups_split_samples():
    X, Y, *_ = planted()
    shift = np.array([60.0, 5.0])
    out = lc.check(FakeBench(X, Y), np.arange(2), shift=shift)
    assert out["45-181 deg"]["samples"] == 1 and out["0-8 deg"]["samples"] == 1 and out["all"]["samples"] == 2
    assert out["45-181 deg"]["covered"]["past peripheral"] == 1.0
    assert out["0-8 deg"]["covered"]["past peripheral"] == 0.0
    assert "8-20 deg" not in out                                       # empty groups are left out
    assert all(np.isnan(v) for v in out["all"]["model_recall"].values())  # no model given


def test_chunks_give_the_same_answer():
    rng = np.random.default_rng(0)
    X = rng.random((7, 2, 5, R, R, R)) > 0.9
    Y = rng.random((7, R, R, R)) > 0.97
    full = lc.check(FakeBench(X, Y), np.arange(7))
    parts = lc.check(FakeBench(X, Y), np.arange(7), chunk=3)
    assert json.dumps(full, sort_keys=True) == json.dumps(parts, sort_keys=True)


def test_end_to_end_on_converted_lab_data(tmp_path):
    spec2 = importlib.util.spec_from_file_location("conv", os.path.join(ROOT, "tests", "test_convert_lab.py"))
    conv_test = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(conv_test)
    src, out = tmp_path / "lab", tmp_path / "out"
    src.mkdir()
    conv_test.fake_lab_dir(src)
    rng = np.random.default_rng(1)
    for p in ("train", "val", "test"):                                  # add the lab's gaze fields
        z = dict(np.load(src / f"{p}.npz"))
        n = len(z["rec"])
        z["g_now"], z["g_fut"] = rng.normal(size=(n, 3)), rng.normal(size=(n, 20, 3))
        np.savez(src / f"{p}.npz", **z)
    conv_test.convert.convert(str(src), str(out))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "landing_check.py"), "--bench", str(out),
                        "--part", "all", "--lab-src", str(src)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "seen, not looked at" in r.stdout and "45-181 deg" in r.stdout
    summary = json.load(open(out / "landing_check.json"))
    assert summary["all"]["samples"] == sum(len(np.load(src / f"{p}.npz")["rec"]) for p in ("train", "val", "test"))
