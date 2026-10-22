"""Fixations and attention shifts, following the lab's definition ("Our Ground Truth for attention shifts"):

    fixation          a run of gaze samples in which every step turns slower than 75 deg/s (I-VT, in the WORLD frame:
                      eye direction rotated by the head pose, so eye and head movement both count)
    held fixation     one lasting at least 150 ms (at 10 Hz one sample is 100 ms, so at least 2 samples)
    attention shift   >= 30 deg between the mean directions of two consecutive HELD fixations; shorter fixations in
                      between are skipped (the eye passing through); pairs more than 1 s apart (tracker loss) are
                      skipped

Angles are seen from the person, not distances. Everything is vectorised over a single recording's 10 Hz series.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ShiftConfig:
    vel_deg_s: float = 75.0           # fixation detector: a step slower than this stays in the fixation
    min_hold_s: float = 0.15          # held fixation: duration = samples * dt >= this
    min_shift_deg: float = 30.0       # attention shift: angle between consecutive held fixations
    max_gap_s: float = 1.0            # skip pairs separated by more than this much missing gaze


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def angle_deg(a, b):
    """angle between direction arrays (..., 3) in degrees."""
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


def step_links(w, t_us, valid, cfg=None):
    """link[k] (k >= 1): sample k continues the fixation of sample k-1 (both valid, contiguous in time, slower than
    the velocity threshold). speed[k] is the angular speed (deg/s) of that step, NaN where undefined."""
    cfg = cfg or ShiftConfig()
    n = len(w)
    dt = np.diff(t_us) / 1e6
    nominal = np.median(dt) if n > 1 else 0.1
    speed = np.full(n, np.nan)
    ok = valid[1:] & valid[:-1] & (dt > 0) & (dt <= 1.5 * nominal)
    ang = angle_deg(w[1:], w[:-1])
    speed[1:][ok] = ang[ok] / dt[ok]
    link = np.zeros(n, bool)
    link[1:] = ok & (ang / np.maximum(dt, 1e-9) < cfg.vel_deg_s)
    return link, speed


def causal_run_start(link):
    """start index of the (so far) fixation run containing each sample, using only samples up to that index."""
    start = np.arange(len(link))
    for k in range(1, len(link)):
        if link[k]:
            start[k] = start[k - 1]
    return start


def detect_fixations(w, t_us, valid, cfg=None):
    """held fixations -> dict(start, end (inclusive sample indices), direction (n, 3) mean direction, dur_s)."""
    cfg = cfg or ShiftConfig()
    link, _ = step_links(w, t_us, valid, cfg)
    dt = np.median(np.diff(t_us)) / 1e6 if len(t_us) > 1 else 0.1
    starts, ends = [], []
    k, n = 0, len(w)
    while k < n:
        if not valid[k]:
            k += 1
            continue
        j = k
        while j + 1 < n and link[j + 1]:
            j += 1
        starts.append(k)
        ends.append(j)
        k = j + 1
    starts, ends = np.array(starts, int), np.array(ends, int)
    keep = (ends - starts + 1) * dt >= cfg.min_hold_s - 1e-9
    starts, ends = starts[keep], ends[keep]
    dirs = np.stack([unit(w[s:e + 1].mean(0)) for s, e in zip(starts, ends)]) if len(starts) else np.zeros((0, 3))
    return dict(start=starts, end=ends, direction=dirs, dur_s=(ends - starts + 1) * dt)


def attention_shifts(fix, t_us, cfg=None):
    """consecutive held fixations >= min_shift_deg apart -> dict(src, dst (fixation indices), angle_deg, t_dst (the
    destination's first sample index))."""
    cfg = cfg or ShiftConfig()
    n = len(fix["start"])
    src, dst, ang = [], [], []
    for i in range(n - 1):
        gap = (t_us[fix["start"][i + 1]] - t_us[fix["end"][i]]) / 1e6
        if gap > cfg.max_gap_s:
            continue
        a = float(angle_deg(fix["direction"][i], fix["direction"][i + 1]))
        if a >= cfg.min_shift_deg:
            src.append(i)
            dst.append(i + 1)
            ang.append(a)
    src, dst = np.array(src, int), np.array(dst, int)
    return dict(src=src, dst=dst, angle_deg=np.array(ang), t_dst=fix["start"][dst] if len(dst) else np.zeros(0, int))
