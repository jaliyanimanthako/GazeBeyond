"""Our forecasters on the lab's own questions (a benchmark built with build.window.anchor=lab, scripts/lab_questions.py),
exported in the lab's format so they are scored with the lab's WHERE table and paired against the lab's models.

    S=~/SensorySpotlight/data/traj_forecast/attention_shift/senses_suite
    # WHERE, one run per CV fold (each scores only its own held-out location's questions):
    python scripts/export_lab_probs.py where --bench data/fovs_aria_labq_sph --suite $S --name gbf_astrained \
        --runs runs/fovs_aria_sph/cv/sph_f{1,2,3,4,5} [--compare probs_<our run>.npz:mem+video]
    # WHEN (a benchmark built from the WHEN questions):
    python scripts/export_lab_probs.py when --bench data/fovs_aria_labq_when_sph --runs ... --out results/gbf_when_aea.npz \
        [--ours ~/SensorySpotlight/data/traj_forecast/attention_shift/baselines/when_ours_aea.npy] \
        [--forecast ~/SensorySpotlight/data/traj_forecast/attention_shift/forecast/aea.npz] [--reuse]
    # --forecast adds the lead-time AUC; --reuse rescores the scores already in --out without running the model

WHERE. The run's --level map (foveal by default; sigmoid per cell, seen from the anchor = the eye at the question)
becomes a probability over the lab's candidate directions (--grid sph: the 960 spherical cells, elevation-major as
SensorySpotlight's WS_GRID=sph; lab: the 468 10-deg directions): each candidate takes the highest cell probability among
the cells whose direction falls in it (the nearest cell's for a candidate no cell centre falls in), times its solid
angle, normalised. Variants written to <suite>/probs_<name>.npz, shape (1, questions, candidates) like the lab's runs:
    gbf      as predicted
    gbf30    candidates within 30 deg of the lab's current gaze (g0) removed: conditioned on a shift, as the lab reads
             Ryan et al. as trained (first predicted fixation >= 30 deg away)
    has      (questions,) the question has a sample (questions without one get a uniform guess: score only has rows;
             attention_where_metrics.py WM_EXTRA does)
The frames are checked first (the lab's landing direction vs ours, as export_where_geometry.py). A short table follows
(median [95% CI over recordings], best of 3, stay), and with --compare the paired difference to a lab-format run on the
same questions.

WHEN. Score per question = the highest --level probability among cells >= 30 deg from the current gaze (the gaze at
the anchor), and the share of the map's probability there; AUC against the lab's y (the questions' file), with a
bootstrap over recordings, next to --ours (the lab's WHEN model, one score per forecast row) on the same questions.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from egospan.data.dataset import PackedBenchmark, SpanDataset      # noqa: E402
from egospan.engine import load_run, predict, resolve_device         # noqa: E402
from egospan.spherical import default_edges                          # noqa: E402
from export_where_geometry import CAZ, CEL, angle, frame_map, from_azel, unit   # noqa: E402
from landing_eval import cell_dirs                                   # noqa: E402

BAND = 4 * np.pi * np.sin(np.radians(65))      # solid angle the lab's 468 directions stand for (attention_where_metrics)


def candidates(grid):
    """-> (directions (C, 3), solid angles (C,), az edges or None, el edges or None) in the lab's head frame."""
    if grid == "lab":
        return from_azel(CAZ, CEL), np.full(len(CAZ), BAND / len(CAZ)), None, None
    az_e, el_e = (np.asarray(e, float) for e in default_edges()[:2])
    az, el = (az_e[:-1] + az_e[1:]) / 2, (el_e[:-1] + el_e[1:]) / 2
    caz, cel = (x.ravel() for x in np.meshgrid(az, el))                  # elevation-major, as WS_GRID=sph
    area = np.outer(np.diff(np.sin(np.radians(el_e))), np.radians(np.diff(az_e))).ravel()
    return from_azel(caz, cel), area, az_e, el_e


def assign(dirs_lab, grid):
    """candidate of every cell direction (lab frame)."""
    C, _, az_e, el_e = candidates(grid)
    if grid == "lab":
        return np.argmax(dirs_lab @ C.T, 1)
    d = unit(dirs_lab)
    az = np.degrees(np.arctan2(d[:, 0], d[:, 2]))
    el = np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))
    ia = np.clip(np.searchsorted(az_e, az, "right") - 1, 0, len(az_e) - 2)
    ie = np.clip(np.searchsorted(el_e, el, "right") - 1, 0, len(el_e) - 2)
    return ie * (len(az_e) - 1) + ia


class CellsToCandidates:
    """(n, cells) probabilities -> (n, C) probability over the candidates (see the module docstring)."""

    def __init__(self, dirs_lab, grid):
        self.C, self.area, _, _ = candidates(grid)
        cb = assign(dirs_lab, grid)
        self.order = np.argsort(cb, kind="stable")
        self.have, self.starts = np.unique(cb[self.order], return_index=True)
        empty = np.setdiff1d(np.arange(len(self.C)), self.have)
        self.empty, self.fill = empty, np.argmax(self.C[empty] @ unit(dirs_lab).T, 1) if len(empty) else empty

    def __call__(self, prob):
        s = np.zeros((len(prob), len(self.C)), np.float64)
        s[:, self.have] = np.maximum.reduceat(prob[:, self.order], self.starts, axis=1)
        s[:, self.empty] = prob[:, self.fill]
        p = s * self.area + 1e-6 * self.area
        return p / p.sum(1, keepdims=True)


def run_predictions(bench, runs, level, device, batch_size, eval_folds=None):
    """-> (probabilities (samples, cells) of `level`, covered (samples,) bool): every run predicts the test split of
    its own CV fold, so each sample is scored by the model that never saw its location. eval_folds: a model trained on
    another dataset (e.g. AEA's, on HD-EPIC / ADT) never saw any of this benchmark: one run then predicts the test
    split of each of these folds of the benchmark."""
    out = np.zeros((len(bench.arrays["X"]), int(np.prod(bench.shape))), np.float32)
    covered = np.zeros(len(out), bool)
    if eval_folds and len(runs) != 1:
        raise SystemExit("--eval-folds takes one run (a model trained on another dataset)")
    jobs = [(runs[0], f) for f in eval_folds] if eval_folds else [(r, None) for r in runs]
    for run, fold in jobs:
        model, cfg = load_run(run, device)
        if level not in cfg.data.levels:
            raise SystemExit(f"{run} does not predict the {level} level ({cfg.data.levels})")
        fold = cfg.data.fold if fold is None else fold
        ds = SpanDataset(bench, "test", fold=fold, levels=cfg.data.levels, crop_front=cfg.data.crop_front,
                         crop_target=False, gaze_paths=cfg.data.gaze_paths, video=cfg.data.video_features)
        p, _ = predict(model, ds, device, batch_size, cfg.train.amp, desc=os.path.basename(run))
        if tuple(p.shape[2:]) != tuple(bench.shape):
            raise SystemExit(f"{run} predicts {p.shape[2:]} cells, the benchmark has {bench.shape}: build it with the "
                             "run's grid settings")
        if covered[ds.idx].any():
            raise SystemExit(f"{run} (fold {fold}) overlaps another run's test samples")
        out[ds.idx] = p[:, cfg.data.levels.index(level)].reshape(len(p), -1)
        covered[ds.idx] = True
        print(f"{run}: fold {fold}, {len(ds.idx)} samples", flush=True)
    return out, covered


def boot_ci(stat, groups, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ug))]
    vals = [stat(np.concatenate([members[k] for k in rng.choice(len(ug), len(ug))])) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def refine(p, C, area):
    """the lab's best guess: highest density, refined to the probability-weighted mean within 15 deg."""
    best = C[(p / area).argmax(1)]
    near = (C[None] * best[:, None]).sum(-1) > np.cos(np.radians(15))
    return unit(((p * near)[..., None] * C[None]).sum(1))


def top3(p, C, area):
    q, picks = p / area, []
    for _ in range(3):
        j = q.argmax(1)
        picks.append(C[j])
        q = np.where((C[j] @ C.T) > np.cos(np.radians(30)), 0, q)
    return picks


def where(a, bench, device):
    if "land_dir" not in bench.arrays or "qid" not in bench.arrays:
        raise SystemExit(f"{a.bench} is not a WHERE benchmark built with build.window.anchor=lab")
    Q = np.load(os.path.join(a.suite, f"data_{a.tag}.npz"))
    n_q = len(Q["rec"])
    qid = bench.arrays["qid"].astype(int)
    if qid.max() >= n_q or not np.array_equal(Q["rec"].astype(str)[qid], bench.arrays["rec"].astype(str)):
        raise SystemExit(f"the benchmark's questions are not rows of {a.suite}/data_{a.tag}.npz (another suite?)")
    T, G = from_azel(*Q["truth"].T), from_azel(*Q["g0"].T)
    M, err = frame_map(T[qid], bench.arrays["land_dir"].astype(np.float64))
    print(f"{len(qid)} samples for {n_q} questions ({len(qid) / n_q:.1%}); landing direction lab vs ours: median "
          f"{err:.2f} deg with axes {'unchanged' if np.allclose(M, np.eye(3)) else M.tolist()}")
    if err > a.max_frame_err:
        raise SystemExit(f"the frames disagree ({err:.1f} deg > {a.max_frame_err})")
    prob, covered = run_predictions(bench, a.runs, a.level, device, a.batch_size, a.eval_folds)
    conv = CellsToCandidates(cell_dirs(bench, bench.shape) @ M, a.grid)  # ours = M @ lab -> lab = ours @ M (rows)
    C, area = conv.C, conv.area
    raw = np.tile(area / area.sum(), (n_q, 1))
    has = np.zeros(n_q, bool)
    s = np.flatnonzero(covered)
    raw[qid[s]] = conv(prob[s])
    has[qid[s]] = True
    far = np.where((G @ C.T) > np.cos(np.radians(30)), 0.0, raw)
    p30 = np.where(far.sum(1, keepdims=True) > 0, far / np.maximum(far.sum(1, keepdims=True), 1e-30), raw)
    path = os.path.join(a.suite, f"probs_{a.name}.npz")
    np.savez_compressed(path, gbf=raw[None].astype(np.float32), gbf30=p30[None].astype(np.float32), has=has)
    print(f"wrote {path}: gbf / gbf30 {raw.shape} on the {a.grid} grid, {has.sum()} questions covered")

    h = np.flatnonzero(has)
    recs = Q["rec"].astype(str)[h]
    rows = {"stay (current gaze)": (G[h], None), "gbf": (refine(raw[h], C, area), raw[h]),
            "gbf, >= 30 deg": (refine(p30[h], C, area), p30[h])}
    print(f"\n{len(h)} questions, {len(np.unique(recs))} recordings (lab's answers)\n"
          f"{'':>24} {'median':>7} {'[95% CI]':>14} {'best of 3':>10}")
    errs = {}
    for name, (d, p) in rows.items():
        e = angle(d, T[h])
        errs[name] = e
        ci = boot_ci(lambda b: np.median(e[b]), recs)
        b3 = "–" if p is None else f"{np.median(np.min([angle(x, T[h]) for x in top3(p, C, area)], 0)):.1f}"
        print(f"{name:>24} {np.median(e):7.1f} [{ci[0]:5.1f}, {ci[1]:5.1f}] {b3:>10}")
    for spec in a.compare:
        f, k = spec.rsplit(":", 1)
        po = np.concatenate([np.load(os.path.join(a.suite, x))[k] for x in f.split(",")]).astype(np.float64).mean(0)[h]
        g = "sph" if po.shape[1] == 960 else "lab"
        Co, Ao, _, _ = candidates(g)
        eo = angle(refine(po, Co, Ao), T[h])
        for name in ("gbf", "gbf, >= 30 deg"):
            e = errs[name]
            ci = boot_ci(lambda b: np.median(e[b]) - np.median(eo[b]), recs)
            print(f"  {name} - {f}:{k} ({np.median(eo):.1f}): {np.median(e) - np.median(eo):+.1f} "
                  f"[{ci[0]:+.1f}, {ci[1]:+.1f}] deg on the same questions")


def auc(y, s):
    """area under the ROC curve (Mann-Whitney, ties counted half)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def when(a, bench, device):
    if "land_dir" in bench.arrays or "qid" not in bench.arrays:
        raise SystemExit(f"{a.bench} is not a WHEN benchmark built with build.window.anchor=lab")
    qf = a.questions or bench.meta["window"].get("questions")
    if not qf or not os.path.isfile(qf):
        raise SystemExit(f"the WHEN questions file ({qf}) is missing: pass --questions")
    if a.reuse and os.path.isfile(a.out):                               # scores already written: rescore only
        R = np.load(a.out)
        qid, recs, y, score_max, score_share = (R[k] for k in ("qid", "rec", "y", "score_max", "score_share"))
        print(f"read {a.out}: {len(qid)} WHEN questions, {y.mean():.1%} positive")
    else:
        if not a.runs:
            raise SystemExit("when needs --runs (or --reuse with an existing --out)")
        Q = np.load(qf)
        y_of = dict(zip(Q["qid"].astype(int), Q["y"]))
        prob, covered = run_predictions(bench, a.runs, a.level, device, a.batch_size, a.eval_folds)
        s = np.flatnonzero(covered)
        dirs = cell_dirs(bench, bench.shape)
        g = unit(bench.arrays["gaze_grid"][s, -1].astype(np.float64))  # the gaze at the anchor, anchor frame
        away = (g @ dirs.T) <= np.cos(np.radians(30))
        p = prob[s].astype(np.float64)
        score_max = np.where(away, p, 0).max(1)
        score_share = np.where(away, p, 0).sum(1) / np.maximum(p.sum(1), 1e-12)
        qid = bench.arrays["qid"][s].astype(int)
        y = np.array([y_of[q] for q in qid], bool)
        recs = bench.arrays["rec"].astype(str)[s]
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.savez(a.out, qid=qid, rec=recs, y=y, score_max=score_max, score_share=score_share)
        print(f"wrote {a.out}: {len(s)} WHEN questions, {y.mean():.1%} positive")
    rows = {"gbf, max >= 30 deg away": score_max, "gbf, share >= 30 deg away": score_share}
    if a.ours:                                                         # .npy, or <file>.npz:<key>
        f, _, key = a.ours.partition(".npz:")
        rows["ours (--ours)"] = (np.load(f + ".npz")[key] if key else np.load(a.ours))[qid]
    lead = None
    if a.forecast:                                                     # the lab's lead-time AUC: a shift 0.3-1.3 s
        D = np.load(a.forecast)                                        # ahead, on moments with none in the next 0.3 s
        lt = D["y_near"][qid] == 0
        lead = (lt, D["y_far"][qid].astype(bool))
        print(f"  lead time: {lt.sum()} of {len(qid)} questions with no shift in the next 0.3 s, "
              f"{lead[1][lt].mean():.1%} positive")
    for name, sc in rows.items():
        ci = boot_ci(lambda b: auc(y[b], sc[b]), recs, n=200)
        line = f"  {name:>28}: AUC {auc(y, sc):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        if lead is not None:
            lt, yf = lead
            k = np.flatnonzero(lt)
            ci = boot_ci(lambda b: auc(yf[k][b], sc[k][b]), recs[k], n=200)
            line += f", lead-time AUC {auc(yf[k], sc[k]):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("where", "when"))
    ap.add_argument("--bench", required=True, help="a benchmark built with build.window.anchor=lab")
    ap.add_argument("--runs", nargs="+", help="trained runs, one per CV fold")
    ap.add_argument("--level", default="foveal")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--suite", help="where: folder with the lab's data_<tag>.npz; the output goes there")
    ap.add_argument("--tag", default="aea_lag")
    ap.add_argument("--name", default="gbf", help="where: writes <suite>/probs_<name>.npz")
    ap.add_argument("--grid", choices=("sph", "lab"), default="sph", help="where: the lab's candidates to map onto")
    ap.add_argument("--max-frame-err", type=float, default=10.0)
    ap.add_argument("--compare", nargs="*", default=[],
                    help="where: <probs file(s) in suite, comma-separated: seeds pooled>:<variant>, paired")
    ap.add_argument("--questions", help="when: the WHEN questions file (default: the one the benchmark was built from)")
    ap.add_argument("--ours", help="when: the lab's WHEN scores, one per forecast row (.npy, or <file>.npz:<key>, e.g. "
                    "forecast/when_v4_aea.npz:gru_ens = the final model)")
    ap.add_argument("--eval-folds", type=int, nargs="+",
                    help="one run trained on another dataset (e.g. AEA's) scores the test split of each of these folds")
    ap.add_argument("--out", help="when: output .npz")
    ap.add_argument("--forecast", help="when: the lab's forecast/<ds>.npz, adds the lead-time AUC (y_near / y_far)")
    ap.add_argument("--reuse", action="store_true", help="when: rescore the scores already in --out (no model run)")
    a = ap.parse_args()
    if a.kind == "where" and not a.suite:
        ap.error("where needs --suite")
    if a.kind == "where" and not a.runs:
        ap.error("where needs --runs")
    if a.kind == "when" and not a.out:
        ap.error("when needs --out")
    bench = PackedBenchmark(a.bench)
    (where if a.kind == "where" else when)(a, bench, resolve_device(a.device))


if __name__ == "__main__":
    main()
