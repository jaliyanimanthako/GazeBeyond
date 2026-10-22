"""Attention-shift benchmark: one sample per shift anchored lag steps before the gaze leaves the fixation, target =
the landing fixation's cones, landing / current directions; fine-tuning from a trained run; the landing-direction
evaluation; camera features matched from the WHERE suite's cache."""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from egospan.data.build import (PackConfig, WindowConfig, extract_samples, extract_shift_samples, pack, save_recording,
                                unpack_grids)
from egospan.data.dataset import PackedBenchmark
from egospan.data.synthetic import make_recording
from egospan.lift import EgoSpanLift, LiftConfig
from egospan.shifts import ShiftConfig, attention_shifts, detect_fixations, unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CART = LiftConfig(grid_frame="local")
SPH = LiftConfig(grid_frame="local", grid="spherical")
SHIFT = WindowConfig(anchor="shift")


def angle(a, b):
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


@pytest.fixture(scope="module")
def rec():
    return make_recording(seconds=40, seed=1)


def test_samples_follow_the_shifts(rec):
    arr = extract_samples(rec, CART, SHIFT)
    w = unit(np.einsum("nij,nj->ni", rec.T_world_cpf[:, :3, :3], rec.gaze_cpf))
    fix = detect_fixations(w, rec.t_us, rec.valid)
    sh = attention_shifts(fix, rec.t_us)
    q = fix["end"][sh["src"]] + 1 - 3                                      # 0.3 s before the gaze leaves A
    ok = q >= 19                                                          # needs 2 s of history
    np.testing.assert_array_equal(arr["t0_us"], rec.t_us[q[ok]])
    assert len(arr["X"]) == ok.sum() > 5
    np.testing.assert_allclose(arr["shift_deg"], sh["angle_deg"][ok], rtol=1e-5)
    assert np.all(arr["shift_deg"] >= 30)
    # directions are in the anchor frame: the angle between them is the shift's angle (a rotation keeps angles)
    np.testing.assert_allclose(angle(arr["gaze_now"], arr["land_dir"]), arr["shift_deg"], atol=1e-2)
    np.testing.assert_array_equal(arr["land_t0_us"], rec.t_us[fix["start"][sh["dst"]][ok]])
    meta = json.loads(str(arr["meta"]))
    assert meta["window"]["future_units"] == 1 and meta["window"]["anchor"] == "shift"


def test_target_is_the_landing_fixation(rec):
    arr = extract_samples(rec, CART, SHIFT)
    lifter = EgoSpanLift(CART)
    from egospan.data.build import StepPoints
    steps = StepPoints(rec, CART, lifter)
    Y = unpack_grids(arr["YF"], (1, 4, 16, 16, 16))[:, 0]
    for i in range(min(4, len(Y))):
        frame = lifter.anchor(rec.T_world_cpf[np.searchsorted(rec.t_us, arr["t0_us"][i])])
        k0, k1 = (np.searchsorted(rec.t_us, arr[k][i]) for k in ("land_t0_us", "land_t1_us"))
        ref = np.zeros((4, 16, 16, 16), bool)
        for k in range(k0, k1 + 1):
            vis, lab = steps(k)
            lifter.voxelize(rec.points_world[vis], lab[:4], frame, out=ref)
        np.testing.assert_array_equal(Y[i], ref)
        # the inputs are the ordinary past window of that anchor: same as a window sample anchored there
    win = extract_samples(rec, CART, WindowConfig())
    common = np.intersect1d(arr["t0_us"], win["t0_us"])
    for t0 in common:
        np.testing.assert_array_equal(arr["X"][arr["t0_us"] == t0][0], win["X"][win["t0_us"] == t0][0])


def test_spherical_and_unknown_anchor(rec):
    arr = extract_shift_samples(rec, SPH, SHIFT)
    assert "YF_cart" in arr and unpack_grids(arr["YF"], (1, 4, 40, 24, 8)).any()
    with pytest.raises(ValueError):
        extract_samples(rec, CART, WindowConfig(anchor="sometimes"))


def _bench(root, cfg, win, names):
    for name in names:
        loc = int(name.split("_loc")[1].split("_")[0])
        s = make_recording(name=name, seed=int(name.rsplit("_", 1)[1]), location=loc, seconds=40)
        save_recording(extract_samples(s, cfg, win), os.path.join(root, "recs", f"{name}.npz"))
    pack(os.path.join(root, "recs"), root, PackConfig(test_locations=(3,), min_points_sigma=None))


def _cfg(root, out, **kw):
    from egospan.config import Config
    cfg = Config()
    cfg.data.root, cfg.data.num_workers = root, 0
    cfg.model.f_maps, cfg.model.layer_order, cfg.model.tf_layers, cfg.model.tf_heads = [8, 16, 32, 64], "cgr", 1, 2
    cfg.train.epochs, cfg.train.batch_size, cfg.train.device, cfg.train.progress = 1, 4, "cpu", False
    cfg.train.out_dir, cfg.eval.saliency = out, False
    for k, v in kw.items():
        sec, key = k.split("__")
        setattr(getattr(cfg, sec), key, v)
    return cfg


def test_finetune_eval_and_video(tmp_path):
    from egospan.engine import build_model, load_matching, train
    names = [f"synth_loc{1 + i % 3}_{i:04d}" for i in range(4)]
    wb, sb = str(tmp_path / "win"), str(tmp_path / "shift")
    _bench(wb, CART, WindowConfig(), names)
    _bench(sb, CART, SHIFT, names)
    b = PackedBenchmark(sb)
    assert b.Tf == 1 and b.target(b.indices("test")).shape[1:] == (4, 16, 16, 16)
    base = _cfg(wb, str(tmp_path / "base"))
    train(base)
    # fine-tuning loads every matching tensor; a model with a different input layer loads the rest
    ft = _cfg(sb, str(tmp_path / "ft"), train__init=str(tmp_path / "base"))
    rep = train(ft)
    assert np.isfinite(rep["mean"]["iou"])
    m = build_model(ft.model, n_out=4, num_steps=2)
    loaded, fresh = load_matching(m, str(tmp_path / "base"))
    assert loaded and not fresh
    m2 = build_model(ft.model, n_out=4, num_steps=2, in_channels=6)
    loaded2, fresh2 = load_matching(m2, str(tmp_path / "base"))
    assert fresh2 and set(fresh2) < set(loaded)                                 # only the first conv is new
    # landing evaluation, with a fake lab question file covering half the test samples
    idx = b.indices("test")
    lab = tmp_path / "data_aea_lag.npz"
    half = idx[::2]
    np.savez(lab, rec=b.arrays["rec"][half], t=b.arrays["t0_us"][half] / 1e6 + 0.02)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "landing_eval.py"), "--runs", str(tmp_path / "ft"),
                        "--lab-queries", str(lab)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.load(open(tmp_path / "ft" / "landing_eval.json"))
    assert res["n"] == len(half)
    stay = res["stay (current fixation)"]["median"]
    assert stay >= 30                                                           # every shift is >= 30 deg
    # camera features from a fake suite cache: every sample matched, tokens in (D, 16, 16) order
    n_all = len(b.arrays["rec"])
    np.savez(tmp_path / "data_aea_lag.npz", rec=b.arrays["rec"], t=b.arrays["t0_us"] / 1e6 + 0.01)
    dense = np.zeros((2 * n_all, 256, 8), np.float16)
    dense[0::2, :, 0] = np.arange(n_all)[:, None]                               # channel 0 = question number
    dense[0::2, 17, 1] = 1                                                      # patch (row 1, col 1)
    np.save(tmp_path / "dense_aea_lag_vjepa.npy", dense)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "attach_shift_video.py"), "--bench", sb,
                        "--suite", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    v = np.load(os.path.join(sb, "video_all.npz"))
    assert v["map"].shape == (n_all, 8, 16, 16) and v["has"].all()
    np.testing.assert_array_equal(v["map"][:, 0, 0, 0], np.arange(n_all))
    assert v["map"][0, 1, 1, 1] == 1 and v["map"][0, 1].sum() == 1
    vid = _cfg(sb, str(tmp_path / "vid"), data__video_features=True, train__init=str(tmp_path / "base"))
    assert np.isfinite(train(vid)["mean"]["iou"])


def test_perfect_prediction_points_at_the_landing():
    spec = importlib.util.spec_from_file_location("le", os.path.join(ROOT, "scripts", "landing_eval.py"))
    le = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(le)
    from egospan.spherical import spherical_grid
    g = spherical_grid({"grid": "spherical"})
    dirs = le.unit(g.centers().reshape(-1, 3))
    true = le.unit(np.array([np.sin(np.radians(-47)), 0.1, np.cos(np.radians(-47))]))
    prob = (dirs @ true > np.cos(np.radians(3))).astype(float)                  # cells around the landing
    pred = le.predict_dirs(prob, dirs)
    assert le.angle(pred[0], true) < 3
    prob2 = prob + 0.5 * (dirs @ le.unit(np.array([1.0, 0, 0.2])) > np.cos(np.radians(3)))   # a second, weaker peak
    p3 = le.predict_dirs(prob2, dirs)
    assert le.angle(p3[0], true) < 3 and le.angle(p3[1], np.array([1.0, 0, 0.2])) < 4
