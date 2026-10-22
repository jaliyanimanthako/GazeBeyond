import numpy as np

from egospan.metrics import evaluate, overlap_scores, region_distance, saliency_scores, tune_thresholds


def test_overlap_scores():
    P = np.zeros((2, 1, 4, 4, 4), bool)
    T = np.zeros_like(P)
    P[0, 0, 0, 0, :2] = True
    T[0, 0, 0, 0, 1:3] = True                      # inter 1, union 3
    s = overlap_scores(P, T)
    assert np.isclose(s["iou"][0, 0], 1 / 3) and np.isclose(s["f1"][0, 0], 0.5)
    assert np.isclose(s["precision"][0, 0], 0.5) and np.isclose(s["recall"][0, 0], 0.5)
    assert np.isnan(s["iou"][1, 0]) and np.isnan(s["precision"][1, 0])       # empty target / prediction


def test_region_distance_and_fallback():
    prob = np.zeros((2, 8, 8, 8), np.float32)
    tgt = np.zeros((2, 8, 8, 8), bool)
    tgt[:, 4, 4, 4] = True
    prob[0, 4, 4, 5] = 0.9                         # one voxel away
    prob[1, 4, 4, 7] = 0.2                         # below threshold -> arg-max fallback, 3 voxels away
    d = region_distance(prob, tgt, 0.5, voxel_size_cm=20)
    np.testing.assert_allclose(d[:, 1], [20.0, 60.0])


def test_saliency_perfect():
    t = np.zeros((1, 1, 4, 4, 4), bool)
    t[0, 0, 1, 1, 1] = True
    s = saliency_scores(t.astype(np.float32), t)
    assert np.isclose(s["auc"][0, 0], 1.0) and np.isclose(s["cc"][0, 0], 1.0)


def test_evaluate_and_tuning():
    rng = np.random.default_rng(0)
    y = rng.random((5, 4, 6, 6, 6)) > 0.8
    prob = np.where(y, 0.7, 0.3).astype(np.float32)
    rep = evaluate(prob, y, ["orientation", "peripheral", "central", "foveal"])
    assert np.isclose(rep["mean"]["iou"], 1.0) and rep["foveal"]["dist_avg"] == 0.0
    thr = tune_thresholds(prob, y)
    assert ((thr > 0.3) & (thr < 0.7)).all()
