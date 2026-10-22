"""Is there information, anywhere, that could predict WHERE the gaze goes at an attention shift? (no training)

Uses the raw Aria recordings and the lab's attention-shift definition (egospan/shifts.py: 75 deg/s fixations, held
150 ms, >= 30 deg between consecutive held fixations, in the world frame). Four questions, answered with real data:

  1. Revisits       Does a shift land on a place fixated before? Fraction of destinations within 10 / 20 deg (world
                    direction) of some held fixation in the previous 2 / 10 / 30 s (counted up to the anchor, 1 s
                    before the destination fixation starts), against a control that swaps in another destination
                    from the same recording.
  2. Head motion    Does the head's turning direction (last 0.3 s, at the anchor) point toward the shift's direction?
                    Same for the eyes' offset from head-forward. Mean cosine and share > 0, against a control that
                    pairs each shift with another event's cue (random cue: cosine ~ 0, share ~ 50 %).
  3. Eye state      Can simple eye-state features at an anchor predict whether an attention shift happens in the next
                    2 s? Features: time in the current fixation, time since the last completed shift, gaze speed (1 and
                    3 steps), head speed, eye eccentricity in the head. Out-of-fold AUC (leave-one-location-out),
                    each feature alone and all together. 0.5 = no information.
  4. Scene memory   (--scene, loads the SLAM points) Was the destination direction already covered by a scene point
                    observed in the last 2 / 10 / 30 s before the anchor (point within 5 deg of the destination ray from
                    the glasses' position at arrival, 0.3-6 m away)? Control: a random destination direction.

    python scripts/shift_destination_analysis.py \
        --slam-root /data/SS/Aria_data/mps_slam --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf \
        --out results/shift_destination.json [--limit 20] [--scene] [--workers 8]

Anchors are every 0.5 s (10 Hz gaze), the 2 s forecast window follows each anchor, like the benchmark's samples.
Caveats: directions are compared in the world frame without correcting for walking (a revisit seen from another spot
is not exactly the same direction); eye-state features are strictly causal (only samples up to the anchor, and only
shifts whose destination fixation had already ended).
"""
import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.shifts import (ShiftConfig, angle_deg, attention_shifts, causal_run_start, detect_fixations,   # noqa: E402
                            step_links, unit)

LAG = 10                                  # steps: the anchor is 1 s before the destination fixation starts
WINDOWS_S = (2, 10, 30)
HORIZON = 20                              # steps: the 2 s forecast window
FEATURES = ["fix_dur_s", "since_shift_s", "gaze_speed_1", "gaze_speed_3", "head_speed_3", "eye_ecc_deg"]
SINCE_CAP_S = 10.0


# ---------------------------------------------------------------------------------------------- per recording
def tangent(v, at):
    """component of v perpendicular to the unit vector `at`, normalised (rows)."""
    t = v - (v * at).sum(-1, keepdims=True) * at
    return unit(t)


def anchor_features(w, f, g_cpf, t_us, valid, shifts, fix, k0s, cfg):
    """causal eye-state features and the 'shift in the next 2 s' label at each anchor k0 (only samples <= k0 are used,
    and only shifts whose destination fixation ended before k0)."""
    link, speed = step_links(w, t_us, valid, cfg)
    run_start = causal_run_start(link)
    dt = np.median(np.diff(t_us)) / 1e6
    k0s = np.asarray(k0s)
    end_d = fix["end"][shifts["dst"]] if len(shifts["dst"]) else np.zeros(0, int)
    start_d = shifts["t_dst"]
    X = np.zeros((len(k0s), len(FEATURES)))
    y = np.zeros(len(k0s), bool)
    for i, k0 in enumerate(k0s):
        done = start_d[end_d < k0] if len(start_d) else start_d
        since = (k0 - done.max()) * dt if len(done) else SINCE_CAP_S
        X[i] = [(k0 - run_start[k0] + 1) * dt, min(since, SINCE_CAP_S), np.nan_to_num(speed[k0]),
                float(angle_deg(w[k0 - 3], w[k0])) / (3 * dt), float(angle_deg(f[k0 - 3], f[k0])) / (3 * dt),
                float(angle_deg(g_cpf[k0], np.array([0.0, 0.0, 1.0])))]
        y[i] = bool(((start_d > k0) & (start_d <= k0 + HORIZON)).any())
    return X, y


def min_angles_to_past(fix, a, dirs, dt):
    """for each window W: min angle (deg) from each direction in `dirs` to any held fixation ended <= step a and
    at most W seconds before it (inf if there is none)."""
    out = {}
    for W in WINDOWS_S:
        sel = (fix["end"] <= a) & ((a - fix["end"]) * dt <= W)
        out[W] = (angle_deg(dirs[:, None, :], fix["direction"][sel][None]).min(1) if sel.any()
                  else np.full(len(dirs), np.inf))
    return out


def shift_events(w, f, t_us, valid, shifts, fix, dt):
    """per attention shift (with a usable anchor): revisit min-angles, head and eye cosines, and the same under the
    control pairing. Returns dict of arrays (one row per shift)."""
    keep = [i for i, sd in enumerate(shifts["t_dst"]) if sd - LAG >= 3 and valid[sd - LAG - 3:sd + 1].all()]
    n = len(keep)
    res = dict(n=n, revisit={W: [] for W in WINDOWS_S}, revisit_ctl={W: [] for W in WINDOWS_S},
               head_cos=[], head_cos_ctl=[], eye_cos=[], eye_cos_ctl=[])
    if n == 0:
        return res
    dest = fix["direction"][shifts["dst"][keep]]
    anchor = shifts["t_dst"][keep] - LAG
    shift_vec, head_vec, eye_vec, head_ok = [], [], [], []
    for j in range(n):
        a = anchor[j]
        past = min_angles_to_past(fix, a, np.concatenate([dest[j:j + 1], dest[(j + 1) % n:(j + 1) % n + 1]]), dt)
        for W in WINDOWS_S:
            res["revisit"][W].append(past[W][0])
            res["revisit_ctl"][W].append(past[W][1] if n > 1 else np.inf)
        wn = w[a]
        shift_vec.append(tangent(dest[j] - (dest[j] @ wn) * wn, wn))
        head_vec.append(tangent(f[a] - f[a - 3], wn))
        eye_vec.append(tangent(wn - f[a], wn))
        head_ok.append(float(angle_deg(f[a - 3], f[a])) / (3 * dt) >= 5.0)
    S, H, E, ok = np.array(shift_vec), np.array(head_vec), np.array(eye_vec), np.array(head_ok)
    perm = np.roll(np.arange(n), 1)
    res["head_cos"], res["head_cos_ctl"] = (S * H).sum(1)[ok], (S[perm] * H).sum(1)[ok]
    res["eye_cos"], res["eye_cos_ctl"] = (S * E).sum(1), (S[perm] * E).sum(1)
    for key in ("revisit", "revisit_ctl"):
        res[key] = {W: np.array(v) for W, v in res[key].items()}
    res["anchor"], res["dest"], res["t_dst"] = anchor, dest, shifts["t_dst"][keep]
    return res


def scene_visibility(streams, events, shifts_t, rng):
    """fraction of destinations covered by points observed in the last W seconds before the anchor (+ control)."""
    out = {W: [] for W in WINDOWS_S}
    ctl = {W: [] for W in WINDOWS_S}
    n = len(events["anchor"])
    for j in range(n):
        a, sd = events["anchor"][j], shifts_t[j]
        pos = streams.T_world_cpf[sd, :3, 3]
        for W in WINDOWS_S:
            lo = max(0, a - 10 * W)
            idx = np.unique(np.concatenate([streams.visible(k) for k in range(lo, a + 1)]))
            if len(idx) > 400_000:
                idx = rng.choice(idx, 400_000, replace=False)
            v = streams.points_world[idx].astype(np.float64) - pos
            d = np.linalg.norm(v, axis=1)
            near = (d > 0.3) & (d < 6.0)
            v, d = v[near], d[near]
            for dirs, store in ((events["dest"][j], out), (events["dest"][(j + 1) % n], ctl)):
                store[W].append(bool(((v @ dirs) / d > np.cos(np.radians(5.0))).any()) if len(v) else False)
    return {W: np.array(v) for W, v in out.items()}, {W: np.array(v) for W, v in ctl.items()}


def analyse(arr, cfg, streams=None, stride=5):
    w, f, g_cpf, t_us, valid = arr["w"], arr["f"], arr["g_cpf"], arr["t_us"], arr["valid"]
    n = len(w)
    dt = np.median(np.diff(t_us)) / 1e6
    fix = detect_fixations(w, t_us, valid, cfg)
    shifts = attention_shifts(fix, t_us, cfg)
    k0s = [k for k in range(20, n - HORIZON - 1, stride) if valid[k - 3:k + HORIZON + 1].all()]
    X, y = anchor_features(w, f, g_cpf, t_us, valid, shifts, fix, k0s, cfg) if k0s else (np.zeros((0, 6)), np.zeros(0))
    ev = shift_events(w, f, t_us, valid, shifts, fix, dt)
    res = dict(minutes=n * dt / 60, n_shifts=len(shifts["dst"]), n_fixations=len(fix["start"]),
               shift_angles=shifts["angle_deg"], X=X, y=y, events=ev, loc=arr.get("loc", -1))
    if streams is not None and ev["n"] > 1:
        res["scene"], res["scene_ctl"] = scene_visibility(streams, ev, ev["t_dst"], np.random.default_rng(0))
    return res


# ---------------------------------------------------------------------------------------------- loading
def load_recording(rec, slam_root, gaze_root, cpf_root, scene=False):
    from egospan.data.aria import read_aea_server, read_gaze_pose
    T_dc = np.array(json.load(open(os.path.join(cpf_root, f"{rec}.json")))["T_device_cpf"])
    meta = json.load(open(os.path.join(gaze_root, rec, "metadata.json")))
    if scene:
        s = read_aea_server(slam_root, gaze_root, rec, T_dc)
        t_us, g_cpf, T, ok = s.t_us, s.gaze_cpf, s.T_world_cpf, s.valid
    else:
        s = None
        t_us, g_cpf, T, ok, _ = read_gaze_pose(
            os.path.join(gaze_root, rec, "mps", "eye_gaze", "general_eye_gaze.csv"),
            (os.path.join(slam_root, f"{rec}_mps_slam_trajectories.zip"), "closed_loop_trajectory.csv"), T_dc)
    valid = ok & np.isfinite(g_cpf).all(1) & np.isfinite(T).all((1, 2))
    R = T[:, :3, :3]
    return dict(t_us=t_us, g_cpf=g_cpf, w=unit(np.einsum("nij,nj->ni", R, g_cpf)), f=R[:, :, 2], valid=valid,
                loc=int(meta["location"])), s


def run_one(rec, slam_root, gaze_root, cpf_root, scene, cfg):
    try:
        arr, streams = load_recording(rec, slam_root, gaze_root, cpf_root, scene)
        return rec, analyse(arr, cfg, streams)
    except Exception as e:                                                            # noqa: BLE001
        return rec, f"FAILED {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------------------------- statistics
def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def fit_logreg(X, y, l2=1.0, iters=30):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    beta = np.zeros(Z.shape[1])
    reg = l2 * np.eye(Z.shape[1])
    reg[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ beta))
        g = Z.T @ (p - y) + reg @ beta
        H = (Z * (p * (1 - p))[:, None]).T @ Z + reg + 1e-8 * np.eye(Z.shape[1])
        beta -= np.linalg.solve(H, g)
    return lambda Xn: np.hstack([(Xn - mu) / sd, np.ones((len(Xn), 1))]) @ beta


def out_of_fold_auc(X, y, groups):
    """pooled leave-one-group-out AUC of the logistic model on all features, over the anchors that were scored (a group
    is skipped when its training part has under 50 anchors or no positives); per-feature AUC (orientation-free).
    Returns (auc, per-feature AUC, scored fraction, skipped groups)."""
    score = np.zeros(len(y))
    scored = np.zeros(len(y), bool)
    skipped = []
    for g in np.unique(groups):
        te = groups == g
        if te.all() or (~te).sum() < 50 or y[~te].sum() == 0:
            skipped.append(g.item() if hasattr(g, "item") else g)
            continue
        score[te] = fit_logreg(X[~te], y[~te].astype(float))(X[te])
        scored[te] = True
    single = {name: auc(y, X[:, i]) for i, name in enumerate(FEATURES)}
    a = auc(y[scored], score[scored]) if scored.any() else float("nan")
    return a, {k: max(v, 1 - v) for k, v in single.items()}, float(scored.mean()), skipped


# ---------------------------------------------------------------------------------------------- report
def pct(a):
    return f"{100 * np.mean(a):5.1f}%" if len(a) else "  n/a "


def report(results):
    minutes = sum(r["minutes"] for r in results)
    n_sh = sum(r["n_shifts"] for r in results)
    ang = np.concatenate([r["shift_angles"] for r in results])
    print(f"\n== recordings {len(results)}, {minutes:.0f} min, {n_sh} attention shifts "
          f"({n_sh / minutes:.1f} per minute; the notes: ~14.4), mean size {ang.mean():.0f} deg, "
          f"median {np.median(ang):.0f} deg")
    X = np.vstack([r["X"] for r in results])
    y = np.concatenate([r["y"] for r in results])
    print(f"   anchors {len(y)}, with a shift in the next 2 s: {100 * y.mean():.1f}%")

    print("\n== 1. REVISITS: destination within X deg of a held fixation of the previous W s "
          "(anchor = 1 s before arrival)")
    print(f"   {'window':>8s} {'within 10 deg':>14s} {'control':>9s} {'within 20 deg':>14s} {'control':>9s}")
    for W in WINDOWS_S:
        obs = np.concatenate([r["events"]["revisit"][W] for r in results if r["events"]["n"]])
        ctl = np.concatenate([r["events"]["revisit_ctl"][W] for r in results if r["events"]["n"]])
        print(f"   {W:>6d} s {pct(obs <= 10):>14s} {pct(ctl <= 10):>9s} {pct(obs <= 20):>14s} {pct(ctl <= 20):>9s}")

    print("\n== 2. HEAD / EYE CUES: cosine between the cue and the shift's direction "
          "(tangent plane at the current gaze)")
    ev = [r["events"] for r in results if r["events"]["n"]]
    for name, key in (("head turning (>= 5 deg/s)", "head_cos"), ("eye offset from head-forward", "eye_cos")):
        o, c = np.concatenate([e[key] for e in ev]), np.concatenate([e[key + "_ctl"] for e in ev])
        print(f"   {name:30s} n {len(o):5d}   cue: mean cos {o.mean():+.3f}, share > 0 {pct(o > 0)}   "
              f"control: mean {c.mean():+.3f}, share {pct(c > 0)}")

    print("\n== 3. EYE STATE -> will an attention shift happen in the next 2 s? "
          "(leave-one-location-out AUC, 0.5 = none)")
    groups = np.concatenate([np.full(len(r["y"]), r["loc"]) for r in results])
    if len(np.unique(groups)) < 2:
        groups = np.arange(len(y)) // max(len(y) // 5, 1)
    all_auc, single, scored, skipped = out_of_fold_auc(X, y, groups)
    for name in FEATURES:
        print(f"   {name:16s} alone: AUC {single[name]:.3f}")
    print(f"   {'all features':16s} logistic: AUC {all_auc:.3f}   (scored {100 * scored:.0f}% of anchors"
          + (f"; groups skipped for lack of training data: {skipped}" if skipped else "") + ")")

    if any("scene" in r for r in results):
        print("\n== 4. SCENE MEMORY: destination direction covered by a scene point observed in the last W s")
        print(f"   {'window':>8s} {'destination':>12s} {'control':>9s}")
        for W in WINDOWS_S:
            o = np.concatenate([r["scene"][W] for r in results if "scene" in r])
            c = np.concatenate([r["scene_ctl"][W] for r in results if "scene" in r])
            print(f"   {W:>6d} s {pct(o):>12s} {pct(c):>9s}")


def jsonable(results):
    def conv(o):
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    return conv([{k: v for k, v in r.items() if k not in ("X", "y")} for r in results])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slam-root", required=True)
    ap.add_argument("--gaze-root", required=True)
    ap.add_argument("--cpf-root", required=True)
    ap.add_argument("--recs", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--scene", action="store_true", help="also test scene memory (loads the SLAM points: slow)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from egospan.data.aria import list_aea_server
    recs = a.recs or list_aea_server(a.slam_root)
    recs = recs[:a.limit] if a.limit else recs
    fn = partial(run_one, slam_root=a.slam_root, gaze_root=a.gaze_root, cpf_root=a.cpf_root, scene=a.scene,
                 cfg=ShiftConfig())
    results = []
    with Pool(a.workers) as pool:
        for rec, r in pool.imap_unordered(fn, recs):
            if isinstance(r, str):
                print(f"{rec}: {r}", flush=True)
            else:
                results.append(r)
                print(f"{rec[-30:]}: {r['n_shifts']} shifts in {r['minutes']:.1f} min", flush=True)
    report(results)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(jsonable(results), open(a.out, "w"))
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
