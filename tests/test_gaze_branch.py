"""Gaze-path branch: vector augmentation, path extraction from (fake) lab files, dataset + model end to end."""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from egospan.data.augment import SpanAugment, transform_vectors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("conv", os.path.join(ROOT, "tests", "test_convert_lab.py"))
conv_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv_test)


def test_vectors_follow_the_grid():
    """A voxel at offset d from the centre of a centred grid must end up at transform_vectors(d)."""
    R = 9                                                    # odd size: voxel 4 is exactly the centre
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = rng.integers(-4, 5, 3)
        g = np.zeros((R, R, R), bool)
        g[tuple(d + 4)] = True
        aug = SpanAugment(vertical_axis=1, max_shift=0, seed=int(rng.integers(1e6)))
        perm, flips, shift = aug.sample()
        moved = np.argwhere(SpanAugment.apply(g, perm, flips, shift))[0] - 4
        np.testing.assert_array_equal(moved, transform_vectors(d[None].astype(float), perm, flips, [[0, 3]])[0])


def test_rotation_vectors_follow_the_grid():
    """Rotations conjugate by the grid map M: rotvec' must equal rotvec(M R M^T)."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        r = rng.normal(size=3) * 0.5
        aug = SpanAugment(vertical_axis=1, max_shift=0, seed=int(rng.integers(1e6)))
        perm, flips, _ = aug.sample()
        M = np.eye(3)[list(perm)]
        for f in flips:
            M[f] *= -1
        want = Rotation.from_matrix(M @ Rotation.from_rotvec(r).as_matrix() @ M.T).as_rotvec()
        got = transform_vectors(r[None], perm, flips, pseudo_slices=[[0, 3]])[0]
        np.testing.assert_allclose(got, want, atol=1e-9)


def make_lab_with_gaze(tmp_path):
    """Fake lab benchmark + gaze CSVs + head rotations (world_from_cpf) consistent with its SLAM future gaze."""
    src, out, groot, hroot = tmp_path / "lab", tmp_path / "out", tmp_path / "gaze", tmp_path / "head"
    src.mkdir()
    conv_test.fake_lab_dir(src)
    rng = np.random.default_rng(0)
    truth = {}
    for p in ("train", "val", "test"):
        z = dict(np.load(src / f"{p}.npz"))
        n = len(z["rec"])
        t0s, Rn, gn, gf = [], [], [], []
        for rec in z["rec"]:
            T, k0 = 60, 30                                   # 6 s of gaze, anchor at row 30
            ts = 1_000_000 + np.arange(T) * 100_000
            yaw, pitch = rng.uniform(-0.3, 0.3, T), rng.uniform(-0.2, 0.2, T)
            d = os.path.join(groot, str(rec), "mps", "eye_gaze")
            os.makedirs(d, exist_ok=True)
            pd.DataFrame({"tracking_timestamp_us": ts, "yaw_rads_cpf": yaw, "pitch_rads_cpf": pitch}).to_csv(
                os.path.join(d, "general_eye_gaze.csv"), index=False)
            Rw = Rotation.from_rotvec(np.cumsum(rng.normal(0, 0.05, (T, 3)), 0)).as_matrix()   # world_from_cpf
            os.makedirs(hroot, exist_ok=True)
            np.savez(hroot / f"{rec}.npz", t=ts / 1e6, R=Rw)
            g = np.stack([np.tan(yaw), np.tan(pitch), np.ones(T)], 1)
            g /= np.linalg.norm(g, axis=1, keepdims=True)
            gw = np.einsum("nij,nj->ni", Rw, g)
            t0s.append(ts[k0])
            Rn.append(Rw[k0])
            gn.append(gw[k0])
            gf.append(gw[k0 + 1:k0 + 21])
            truth[str(rec)] = (Rw, g, k0)
        z.update(t0_us=np.array(t0s), R_now=np.stack(Rn), g_now=np.stack(gn), g_fut=np.stack(gf))
        assert len(t0s) == n
        np.savez(src / f"{p}.npz", **z)
    conv_test.convert.convert(str(src), str(out))
    return src, out, groot, hroot, truth


def test_build_gaze_paths(tmp_path):
    src, out, groot, hroot, truth = make_lab_with_gaze(tmp_path)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_gaze_paths.py"), "--src", str(src),
                        "--gaze-root", str(groot), "--head-root", str(hroot), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "world_from_cpf" in [ln.split()[2] for ln in r.stdout.splitlines() if "<- used" in ln]
    g = np.load(out / "gaze_all.npz")
    rec = str(g["rec"][0])
    Rw, gc, k0 = truth[rec]
    path = g["path"][0].astype(np.float64)
    assert path.shape == (20, 11) and path[:, 9:].all()
    want = np.einsum("ji,njk,nk->ni", Rw[k0], Rw[k0 - 19:k0 + 1], gc[k0 - 19:k0 + 1])   # R0^T Rk g_k
    np.testing.assert_allclose(path[:, 0:3], want, atol=2e-3)                            # float16 storage
    np.testing.assert_allclose(path[:, 3:6], gc[k0 - 19:k0 + 1], atol=2e-3)
    np.testing.assert_allclose(path[-1, 6:9], 0, atol=1e-3)                              # no rotation at the anchor


def test_gaze_dataset_and_model(tmp_path):
    from egospan.config import Config, ModelConfig
    from egospan.engine import make_datasets, predict
    from egospan.losses import dice_loss
    from egospan.models import build_model
    src, out, groot, hroot, _ = make_lab_with_gaze(tmp_path)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_gaze_paths.py"), "--src", str(src),
                    "--gaze-root", str(groot), "--head-root", str(hroot), "--out", str(out)], check=True,
                   capture_output=True)
    cfg = Config()
    cfg.data.root, cfg.data.gaze_paths, cfg.data.crop_front = str(out), True, True
    bench, ds = make_datasets(cfg)
    item = ds["train"][0]
    assert item["g"].shape == (20, 11) and ds["train"].cue_dim == 11
    net = build_model(ModelConfig(f_maps=[8, 16, 32, 64], layer_order="cgr", tf_layers=1, tf_heads=2), 4, 2,
                      cue_dim=11)
    loss = dice_loss(net(item["x"][None], item["g"][None]), item["y"][None])
    loss.backward()
    assert net.cue_encoder.weight_ih_l0.grad.abs().sum() > 0                 # the branch receives gradients
    p, y = predict(net, ds["test"], torch.device("cpu"), batch_size=4, amp=False, progress=False)
    assert p.shape == y.shape
