"""Fixation / attention-shift detector on planted gaze sequences with known answers."""
import numpy as np

from egospan.shifts import angle_deg, attention_shifts, causal_run_start, detect_fixations, step_links


def direction(yaw_deg, pitch_deg=0.0):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])


def series(segments, jitter_deg=0.5, seed=0):
    """segments: [(yaw_deg, n_samples), ...] at 10 Hz; jitter well below the 75 deg/s detector threshold."""
    rng = np.random.default_rng(seed)
    w = np.concatenate([np.stack([direction(y + rng.normal(0, jitter_deg), rng.normal(0, jitter_deg))
                                  for _ in range(n)]) for y, n in segments])
    t = np.arange(len(w)) * 100_000
    return w, t, np.ones(len(w), bool)


def test_planted_shifts():
    # A(5) --40--> B(5) --10--> C(5) --50--> D(1, passing through) --40--> E(5), then a 5 deg step into F(5):
    # 5 deg per 100 ms is 50 deg/s < 75, so E and F are ONE fixation; 10 deg per 100 ms (B->C) is 100 deg/s, a saccade
    w, t, v = series([(0, 5), (40, 5), (50, 5), (100, 1), (140, 5), (145, 5)])
    fix = detect_fixations(w, t, v)
    assert len(fix["start"]) == 4                         # A, B, C, E+F; D (1 sample = 100 ms) is not held
    assert fix["start"].tolist() == [0, 5, 10, 16] and fix["end"].tolist() == [4, 9, 14, 25]
    sh = attention_shifts(fix, t)
    # pairs: A-B 40 (shift), B-C 10, C-E 90 (D skipped)
    assert np.allclose(sh["angle_deg"], [40, 92.5], atol=4.0)       # E+F merged: mean direction ~142.5
    assert sh["t_dst"].tolist() == [5, 16]


def test_threshold_boundaries():
    w, t, v = series([(0, 4), (29, 4)], jitter_deg=0.0)   # 29 deg: not a shift
    assert len(attention_shifts(detect_fixations(w, t, v), t)["dst"]) == 0
    w, t, v = series([(0, 4), (31, 4)], jitter_deg=0.0)   # 31 deg: a shift
    assert len(attention_shifts(detect_fixations(w, t, v), t)["dst"]) == 1
    # a drift of 5 deg per 100 ms = 50 deg/s stays one fixation; 9 deg per 100 ms = 90 deg/s breaks it
    slow = np.stack([direction(5 * k) for k in range(8)])
    fast = np.stack([direction(9 * k) for k in range(8)])
    t8, v8 = np.arange(8) * 100_000, np.ones(8, bool)
    assert len(detect_fixations(slow, t8, v8)["start"]) == 1
    assert len(detect_fixations(fast, t8, v8)["start"]) == 0       # every sample is its own (unheld) run


def test_gaps_and_invalid_samples():
    w, t, v = series([(0, 4), (60, 4)], jitter_deg=0.0)
    t = t.copy()
    t[4:] += 2_000_000                                    # 2 s of missing gaze between the fixations
    assert len(attention_shifts(detect_fixations(w, t, v), t)["dst"]) == 0
    w, t, v = series([(0, 6), (60, 6)], jitter_deg=0.0)
    v[3] = False                                          # an invalid sample splits the first fixation
    fix = detect_fixations(w, t, v)
    assert (fix["end"] - fix["start"] + 1).tolist() == [3, 2, 6]


def test_causal_run_start_and_links():
    w, t, v = series([(0, 4), (50, 4)], jitter_deg=0.0)
    link, speed = step_links(w, t, v)
    assert link.tolist() == [False, True, True, True, False, True, True, True]
    assert abs(speed[4] - 500.0) < 1e-6                   # 50 deg in 0.1 s
    assert causal_run_start(link).tolist() == [0, 0, 0, 0, 4, 4, 4, 4]


def test_angle():
    assert abs(angle_deg(direction(0), direction(30)) - 30) < 1e-9
    assert abs(angle_deg(direction(0), direction(0, 30)) - 30) < 1e-9
