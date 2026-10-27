"""Samples on the lab's own questions (build.window.anchor = lab, scripts/lab_questions.py) and their export in the
lab's format (scripts/export_lab_probs.py): given the shifts we detect ourselves as the lab's questions, the samples are
exactly the shift benchmark's; WHEN questions get the window target; the export maps cells onto the lab's candidates,
checks the frames, covers each fold's own location and drops candidates near the gaze for gbf30."""
import importlib.util
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from egospan.data.build import WindowConfig, extract_samples, pack, save_recording
from egospan.data.build import PackConfig, load_questions
from egospan.data.dataset import PackedBenchmark
from egospan.data.synthetic import make_recording
from egospan.lift import LiftConfig
from egospan.shifts import attention_shifts, detect_fixations, unit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_shift_benchmark import _cfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
CART = LiftConfig(grid_frame="local")
NAMES = [f"synth_loc{1 + i % 3}_{i:04d}" for i in range(6)]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, SCRIPTS)
    spec.loader.exec_module(m)
    return m


def azel(d):
    d = unit(d)
    return np.stack([np.degrees(np.arctan2(d[:, 0], d[:, 2])), np.degrees(np.arcsin(d[:, 1]))], -1)


def _recs():
    return [make_recording(name=n, seed=int(n.rsplit("_", 1)[1]), location=int(n.split("_loc")[1][0]), seconds=40)
            for n in NAMES]


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """fake lab files from our own shift detection: labels (fixations / events csv), the suite's data_aea_lag.npz (one
    question per shift sample, a shuffled order) and the forecast file; plus the shift benchmark's samples."""
    root = tmp_path_factory.mktemp("lab")
    fix_rows, ev_rows, q, shift = [], [], [], {}
    for s in _recs():
        t = s.t_us
        w = unit(np.einsum("nij,nj->ni", s.T_world_cpf[:, :3, :3], s.gaze_cpf))
        fix = detect_fixations(w, t, s.valid)
        sh = attention_shifts(fix, t)
        off = len(fix_rows)
        for a, b, d in zip(fix["start"], fix["end"], fix["direction"]):
            fix_rows.append(dict(rec=s.name, t0=t[a] / 1e6, t1=t[b] / 1e6, gx=d[0], gy=d[1], gz=d[2]))
        for a, b in zip(sh["src"], sh["dst"]):
            ev_rows.append(dict(rec=s.name, t_leave=t[fix["end"][a]] / 1e6 + 0.1, shift=1, i_from=off + a, i_to=off + b))
        arr = extract_samples(s, CART, WindowConfig(anchor="shift"))
        shift[s.name] = arr
        for k in range(len(arr["X"])):
            q.append((s.name, arr["t0_us"][k] / 1e6, *azel(arr["land_dir"][k:k + 1])[0], *azel(arr["gaze_now"][k:k + 1])[0]))
    pd.DataFrame(fix_rows).to_csv(root / "aea_fixations.csv", index=False)
    pd.DataFrame(ev_rows).to_csv(root / "aea_events.csv", index=False)
    order = np.random.default_rng(0).permutation(len(q))
    q = [q[i] for i in order]
    np.savez(root / "data_aea_lag.npz", rec=np.array([x[0] for x in q]), t=np.array([x[1] for x in q]),
             truth=np.array([x[2:4] for x in q]), g0=np.array([x[4:6] for x in q]), qi=np.arange(len(q)))
    recs = np.array(NAMES)                                       # forecast file: every 0.1 s step, y = anything
    ft, fg = [], []
    for g, s in enumerate(_recs()):
        ft.append(s.t_us[25:-25:3] / 1e6)
        fg.append(np.full(len(ft[-1]), g))
    ft, fg = np.concatenate(ft), np.concatenate(fg)
    np.savez(root / "forecast_aea.npz", t=ft, group=fg, recs=recs, y=(np.arange(len(ft)) % 3 == 0))
    return root, shift


def _questions(root, kind):
    out = str(root / f"q_{kind}.npz")
    args = (["where", "--suite", str(root), "--labels", str(root)] if kind == "where"
            else ["when", "--forecast", str(root / "forecast_aea.npz"), "--every", "2"])
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "lab_questions.py"), *args, "--out", out],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def test_where_questions_rebuild_the_shift_samples(lab):
    root, shift = lab
    qf = _questions(root, "where")
    Q = load_questions(qf)
    D = np.load(root / "data_aea_lag.npz")
    assert len(Q["rec"]) == len(D["rec"])                                 # every question matched to its shift
    np.testing.assert_array_equal(Q["rec"], D["rec"][Q["qid"]])
    win = WindowConfig(anchor="lab", questions=qf)
    for s in _recs()[:3]:
        arr, ref = extract_samples(s, CART, win), shift[s.name]
        np.testing.assert_array_equal(arr["t0_us"], ref["t0_us"])         # same anchors (questions sorted in time)
        for k in ("X", "YF", "land_t0_us", "land_t1_us"):
            np.testing.assert_array_equal(arr[k], ref[k])
        np.testing.assert_allclose(arr["land_dir"], ref["land_dir"], atol=1e-5)
        np.testing.assert_allclose(arr["gaze_now"], ref["gaze_now"], atol=1e-5)
        np.testing.assert_allclose(arr["shift_deg"], ref["shift_deg"], atol=1e-3)
        np.testing.assert_array_equal(D["rec"][arr["qid"]], s.name)
        np.testing.assert_allclose(D["t"][arr["qid"]], arr["t0_us"] / 1e6)


def test_questions_without_history_or_step_are_skipped(lab, tmp_path):
    root, shift = lab
    s = _recs()[0]
    t = s.t_us / 1e6
    t_ok = shift[s.name]["t0_us"][0] / 1e6                                 # a time with 2 s of valid history
    np.savez(tmp_path / "q.npz", rec=np.array([s.name] * 3), t=np.array([t[5], t[-1] + 1.0, t_ok + 0.01]),
             qid=np.array([7, 8, 9]), y=np.array([True, False, True]))
    arr = extract_samples(s, CART, WindowConfig(anchor="lab", questions=str(tmp_path / "q.npz")))
    np.testing.assert_array_equal(arr["qid"], [9])                       # no 2 s history / no gaze step within 50 ms
    assert arr["t0_us"][0] == shift[s.name]["t0_us"][0]


def test_when_questions_get_the_window_target(lab):
    root, _ = lab
    qf = _questions(root, "when")
    s = _recs()[0]
    arr = extract_samples(s, CART, WindowConfig(anchor="lab", questions=qf))
    ref = extract_samples(s, CART, WindowConfig())                        # window samples every 1 s
    common = np.intersect1d(arr["t0_us"], ref["t0_us"])
    assert len(common) >= 3 and "land_dir" not in arr
    for t0 in common:
        a, b = np.flatnonzero(arr["t0_us"] == t0)[0], np.flatnonzero(ref["t0_us"] == t0)[0]
        np.testing.assert_array_equal(arr["X"][a], ref["X"][b])
        np.testing.assert_array_equal(arr["YF"][a], ref["YF"][b])


@pytest.mark.parametrize("grid", ["sph", "lab"])
def test_cells_to_candidates_peaks_where_the_cells_do(grid):
    elp = _load("export_lab_probs")
    from egospan.spherical import spherical_grid
    dirs = unit(spherical_grid({"grid": "spherical"}).centers().reshape(-1, 3))
    conv = elp.CellsToCandidates(dirs, grid)
    C, area = conv.C, conv.area
    true = unit(np.array([np.sin(np.radians(-47)), 0.1, np.cos(np.radians(-47))]))
    prob = (dirs @ true > np.cos(np.radians(4))).astype(np.float32)[None]
    p = conv(prob)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-9)
    assert elp.angle(elp.refine(p, C, area)[0], true) < 6
    if grid == "sph":                                                      # our cells are the lab's: one per candidate
        assert len(conv.empty) == 0 and np.all(np.bincount(elp.assign(dirs, "sph"), minlength=960) == 8)


def test_export_where_and_when(lab, tmp_path):
    from egospan.engine import train
    root, _ = lab
    load_questions.cache_clear()
    benches = {}
    for kind in ("where", "when"):
        b = str(tmp_path / kind)
        win = WindowConfig(anchor="lab", questions=_questions(root, kind))
        for s in _recs():
            save_recording(extract_samples(s, CART, win), os.path.join(b, "recs", f"{s.name}.npz"))
        pack(os.path.join(b, "recs"), b, PackConfig(test_locations=(3,), min_points_sigma=None))
        benches[kind] = b
    run = str(tmp_path / "run_f1")
    train(_cfg(benches["where"], run, data__fold=1))
    cmd = [sys.executable, os.path.join(SCRIPTS, "export_lab_probs.py"), "where", "--bench", benches["where"],
           "--suite", str(root), "--name", "gbf_test", "--runs", run, "--device", "cpu"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "axes unchanged" in r.stdout
    r = subprocess.run(cmd + ["--compare", "probs_gbf_test.npz,probs_gbf_test.npz:gbf"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr                                     # itself, "two seed files" pooled
    assert re.search(r"gbf - probs_gbf_test\.npz,probs_gbf_test\.npz:gbf \([\d.]+\): \+0\.0 \[[+-]0\.0, [+-]0\.0\]",
                     r.stdout)
    P = np.load(root / "probs_gbf_test.npz")
    D = np.load(root / "data_aea_lag.npz")
    loc1 = np.char.find(D["rec"].astype(str), "_loc1_") >= 0
    np.testing.assert_array_equal(P["has"], loc1)                         # fold 1 = location 1, all of its questions
    assert P["gbf"].shape == (1, len(D["rec"]), 960)
    elp = _load("export_lab_probs")
    C = elp.candidates("sph")[0]
    G = elp.from_azel(*D["g0"].T)
    near = (G @ C.T) > np.cos(np.radians(30))
    assert np.all(P["gbf30"][0][near] == 0) and np.allclose(P["gbf30"].sum(-1), 1, atol=1e-5)
    out = str(tmp_path / "when.npz")
    ours = tmp_path / "ours.npy"
    np.save(ours, np.random.default_rng(0).random(10_000))
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "export_lab_probs.py"), "when", "--bench",
                        benches["when"], "--runs", run, "--out", out, "--device", "cpu", "--ours", str(ours)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    W = np.load(out)
    assert len(W["qid"]) > 10 and np.all((W["score_share"] >= 0) & (W["score_share"] <= 1))
    assert "AUC" in r.stdout
