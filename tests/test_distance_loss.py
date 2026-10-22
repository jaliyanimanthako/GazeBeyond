"""Distance-aware loss: values, one-way vs bidirectional, empty targets, gradient direction, training integration."""
import numpy as np
import pytest
import torch

from egospan.losses import build_loss, dice_loss, distance_loss, distance_terms

R = 16
OFF = -40.0                                                  # logit of "no probability" (sigmoid ~ 4e-18)


def logit(p):
    return float(np.log(p / (1 - p)))


def grid(points, values=None):
    g = torch.full((R, R, R), OFF)
    for k, ijk in enumerate(points):
        g[ijk] = logit(values[k]) if values is not None else 40.0
    return g


def target(points):
    t = torch.zeros(R, R, R)
    for ijk in points:
        t[ijk] = 1
    return t


def test_expected_distance_example():
    """mass 0.7 / 0.2 / 0.1 at 1 / 2 / 3 voxels from the target -> 0.7*1 + 0.2*2 + 0.1*3 = 1.4 voxels."""
    t = target([(8, 8, 8)])
    p = torch.sigmoid(grid([(8, 8, 9), (8, 8, 10), (8, 8, 11)], [0.7, 0.2, 0.1]))
    fwd, _ = distance_terms(p, t)
    assert abs(fwd.item() - 1.4) < 1e-3
    far = torch.sigmoid(grid([(8, 8, 9), (8, 8, 10), (8, 8, 11)], [0.1, 0.2, 0.7]))
    assert distance_terms(far, t)[0].item() > fwd.item()                 # mass further away costs more
    on = torch.sigmoid(grid([(8, 8, 8)], [0.9]))
    assert distance_terms(on, t)[0].item() < 1e-6                        # on the target: zero


def test_bidirectional_needs_every_spot_covered():
    t = target([(4, 8, 8), (12, 8, 8)])                                  # two spots 8 voxels apart
    one = torch.sigmoid(grid([(4, 8, 8)], [0.9]))
    both = torch.sigmoid(grid([(4, 8, 8), (12, 8, 8)], [0.9, 0.9]))
    f1, r1 = distance_terms(one, t)
    f2, r2 = distance_terms(both, t)
    assert f1.item() < 1e-6 and f2.item() < 1e-6                         # one-way: both look perfect...
    assert r1.item() > 3 and r2.item() < 0.01                            # ...the reverse term sees the missed spot


def test_empty_targets_and_levels():
    logits = torch.zeros(2, 4, R, R, R)
    y = torch.zeros(2, 4, R, R, R)
    assert distance_loss(logits, y, channels=[2, 3]).item() == 0.0       # nothing to measure against
    loss = build_loss("dice+dist", dist_weight=0.1)
    assert abs(loss(logits, y).item() - dice_loss(logits, y).item()) < 1e-6
    with pytest.raises(ValueError):
        build_loss("dice+dist", levels=["orientation"], dist_levels=["foveal"])
    with pytest.raises(ValueError):
        build_loss("dice+dist", dist_mode="sideways")


def test_gradient_pulls_mass_toward_target():
    y = torch.zeros(1, 4, R, R, R)
    y[0, 3, 8, 8, 8] = 1
    logits = torch.full((1, 4, R, R, R), -3.0, requires_grad=True)
    loss = distance_loss(logits, y, channels=[3])
    loss.backward()
    g = logits.grad[0, 3]
    # d L_fwd / d p_i = (D_i - L) / sum p: raise probability where a voxel is closer than the current mean distance
    # of the predicted mass, lower it where it is further away, the more so the further
    assert g[8, 8, 8] < 0 and g[8, 8, 8] < g[8, 8, 10] < g[0, 0, 0]
    assert g[0, 0, 0] > 0                                                 # the far corner is pushed down
    for mode in ("oneway", "bidirectional"):
        total = build_loss("dice+dist", dist_mode=mode)(logits, y)
        assert torch.isfinite(total)


def test_trains_with_dice_dist(tmp_path):
    from egospan.config import Config
    from egospan.data.build import PackConfig, WindowConfig, extract_samples, pack, save_recording
    from egospan.data.synthetic import make_recording
    from egospan.engine import train
    from egospan.lift import LiftConfig
    for i in range(6):
        s = make_recording(name=f"r{i}", seconds=10, seed=i, location=1 + i % 3)
        save_recording(extract_samples(s, LiftConfig(grid_frame="local"), WindowConfig()),
                       str(tmp_path / "recs" / f"r{i}.npz"))
    pack(str(tmp_path / "recs"), str(tmp_path / "bench"), PackConfig(test_locations=(3,), min_points_sigma=None))
    cfg = Config()
    cfg.data.root, cfg.data.num_workers, cfg.data.crop_front = str(tmp_path / "bench"), 0, True
    cfg.model.f_maps, cfg.model.layer_order, cfg.model.tf_layers, cfg.model.tf_heads = [8, 16, 32, 64], "cgr", 1, 2
    cfg.train.epochs, cfg.train.batch_size, cfg.train.device, cfg.train.progress = 1, 4, "cpu", False
    cfg.train.loss, cfg.train.dist_mode = "dice+dist", "bidirectional"
    cfg.train.out_dir = str(tmp_path / "run")
    cfg.eval.saliency = False
    rep = train(cfg)
    assert np.isfinite(rep["mean"]["iou"]) and "dist_avg" in rep["foveal"]
