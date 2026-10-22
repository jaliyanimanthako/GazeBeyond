"""shift_destination_analysis: causality of the features, planted effects, statistics, end to end on synthetic data."""
import importlib.util
import json
import os

import numpy as np

from egospan.shifts import ShiftConfig, attention_shifts, detect_fixations, unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sda", os.path.join(ROOT, "scripts", "shift_destination_analysis.py"))
sda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sda)
CFG = ShiftConfig()


def direction(yaw, pitch=0.0):
    y, p = np.radians(yaw), np.radians(pitch)
    return np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])


def make_arrays(seed=0, n_fix=60, spots=(0, 50, -50, 100), head_follow=False):
    """fixations of 4-12 samples on random spots, a saccade step between them; head forward drifts slowly."""
    rng = np.random.default_rng(seed)
    w = []
    for _ in range(n_fix):
        s = rng.choice(spots)
        w += [direction(s + rng.normal(0, 0.4), rng.normal(0, 0.4)) for _ in range(rng.integers(4, 13))]
    w = np.array(w)
    n = len(w)
    head_yaw = np.cumsum(rng.normal(0, 1.0, n))
    f = np.stack([direction(h) for h in head_yaw])
    g_cpf = np.stack([unit(wi - fi + np.array([0, 0, 1.0])) for wi, fi in zip(w, f)])
    return dict(w=w, f=f, g_cpf=g_cpf, t_us=np.arange(n) * 100_000, valid=np.ones(n, bool), loc=1)


def test_features_are_causal():
    arr = make_arrays(1)
    n = len(arr["w"])
    cut = n // 2
    res_a = sda.analyse(arr, CFG, stride=1)                                       # every step: any peek shows
    garbled = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in arr.items()}
    rng = np.random.default_rng(5)
    garbled["w"][cut:] = unit(rng.normal(size=(n - cut, 3)))                     # arbitrary future
    garbled["f"][cut:] = unit(rng.normal(size=(n - cut, 3)))
    garbled["g_cpf"][cut:] = unit(rng.normal(size=(n - cut, 3)))
    res_b = sda.analyse(garbled, CFG, stride=1)
    k0s = np.array([k for k in range(20, n - sda.HORIZON - 1, 1) if arr["valid"][k - 3:k + sda.HORIZON + 1].all()])
    known = k0s <= cut - 1                                                       # anchors whose past is intact
    assert known.sum() > 10
    np.testing.assert_allclose(res_a["X"][known], res_b["X"][known])
    # labels look into the future, so they are allowed to differ; features must not


def test_detected_counts_and_rates():
    arr = make_arrays(2)
    fix = detect_fixations(arr["w"], arr["t_us"], arr["valid"], CFG)
    sh = attention_shifts(fix, arr["t_us"], CFG)
    res = sda.analyse(arr, CFG)
    assert res["n_shifts"] == len(sh["dst"]) and res["n_fixations"] == len(fix["start"])
    assert res["n_shifts"] > 10 and np.all(res["shift_angles"] >= 30)
    assert res["X"].shape[1] == len(sda.FEATURES) and len(res["X"]) == len(res["y"])
    assert 0 < res["y"].mean() < 1


def test_revisit_planted():
    """gaze alternates between 3 fixed spots 60 deg apart: every destination was fixated before (within 30 s)."""
    arr = make_arrays(3, n_fix=80, spots=(0, 60, 120))
    ev = sda.analyse(arr, CFG)["events"]
    assert ev["n"] > 20
    assert np.mean(ev["revisit"][30] <= 10) > 0.9
    # a window that is too short to contain an earlier visit gives fewer hits
    assert np.mean(ev["revisit"][2] <= 10) < np.mean(ev["revisit"][30] <= 10)


def test_tangent_cosines():
    at = np.array([[0.0, 0.0, 1.0]])
    v = np.array([[1.0, 0.0, 0.3]])
    t = sda.tangent(v, at)
    np.testing.assert_allclose(t, [[1.0, 0.0, 0.0]], atol=1e-12)
    assert abs(float((t * sda.tangent(np.array([[2.0, 0.0, -1.0]]), at)).sum()) - 1) < 1e-12


def test_auc_and_logistic_leave_group_out():
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, len(sda.FEATURES)))
    groups = rng.integers(0, 4, n)
    y = (X[:, 0] + 0.5 * rng.normal(size=n)) > 0.3                                # feature 0 carries the signal
    a_all, single, scored, skipped = sda.out_of_fold_auc(X, y, groups)
    assert a_all > 0.8 and single[sda.FEATURES[0]] > 0.8 and 0.4 < single[sda.FEATURES[3]] < 0.6
    assert scored == 1.0 and skipped == []
    y0 = rng.random(n) < 0.3                                                      # no signal at all
    a0, _, _, _ = sda.out_of_fold_auc(X, y0, groups)
    assert 0.45 < a0 < 0.55
    assert abs(sda.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])) - 1) < 1e-12


def test_scene_visibility_and_report_on_synthetic_recording(capsys):
    from egospan.data.synthetic import make_recording
    s = make_recording(seconds=60, seed=3, n_wall=3000)
    R = s.T_world_cpf[:, :3, :3]
    arr = dict(t_us=s.t_us, g_cpf=s.gaze_cpf, w=unit(np.einsum("nij,nj->ni", R, s.gaze_cpf)), f=R[:, :, 2],
               valid=s.valid, loc=2)
    res = sda.analyse(arr, CFG, streams=s)
    assert "scene" in res or res["events"]["n"] <= 1
    if "scene" in res:
        for W in sda.WINDOWS_S:
            assert len(res["scene"][W]) == res["events"]["n"]
        assert res["scene"][2].mean() <= res["scene"][30].mean() + 1e-9        # a longer memory can only cover more
    sda.report([res, sda.analyse(make_arrays(4), CFG), sda.analyse(make_arrays(5), CFG)])
    out = capsys.readouterr().out
    assert "REVISITS" in out and "EYE STATE" in out and "attention shifts" in out
    assert isinstance(sda.jsonable([res]), list)


def test_too_little_data_is_reported_not_hidden():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, len(sda.FEATURES)))
    y = X[:, 0] > 0
    groups = np.repeat([0, 1, 2], 40)                         # each fold's training part has only 80 anchors: fine
    assert sda.out_of_fold_auc(X, y, groups)[2] == 1.0
    groups = np.repeat(np.arange(6), 20)                      # training part 100: fine; now make it too small
    a, _, scored, skipped = sda.out_of_fold_auc(X[:60], y[:60], np.repeat([0, 1, 2], 20))
    assert scored == 0.0 and sorted(skipped) == [0, 1, 2] and np.isnan(a)    # all skipped: NaN, not a fake 0.5


def test_json_round_trip():
    res = sda.analyse(make_arrays(6), CFG)
    text = json.dumps(sda.jsonable([res, res]))
    assert len(json.loads(text)) == 2
