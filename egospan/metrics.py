"""Evaluation protocol (Sec. 5.1, Appendix B.1).

Each span level is scored separately. Overlap metrics (IoU primary, F1, precision, recall) are computed per sample
and averaged over samples, so the mean F1 is not the harmonic mean of the mean precision and recall (Appendix B.1).
Samples whose target is empty are skipped for IoU / F1 / recall; samples with an empty prediction are skipped for
precision ("different patterns of invalid values across metrics").

Foveal metric distance (Table 3): per sample, the distance from every predicted voxel to the nearest ground-truth
voxel; we report the mean over samples of the per-sample min / mean / max (cm). An empty prediction falls back to the
arg-max voxel. Saliency metrics (Table 6): Pearson CC and ROC-AUC between the soft prediction and the binary target.
"""
import numpy as np


def overlap_scores(pred, target):
    """pred, target: bool (n, C, ...) -> dict of (n, C) float arrays with NaN where undefined."""
    P = pred.reshape(*pred.shape[:2], -1).astype(bool)
    T = target.reshape(*target.shape[:2], -1).astype(bool)
    inter = (P & T).sum(-1).astype(float)
    ps, ts = P.sum(-1).astype(float), T.sum(-1).astype(float)
    has_t, has_p = ts > 0, ps > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"iou": np.where(has_t, inter / np.maximum(ps + ts - inter, 1), np.nan),
                "f1": np.where(has_t, 2 * inter / np.maximum(ps + ts, 1), np.nan),
                "precision": np.where(has_p, inter / np.maximum(ps, 1), np.nan),
                "recall": np.where(has_t, inter / np.maximum(ts, 1), np.nan)}


def tune_thresholds(prob, target, grid=np.linspace(0.05, 0.95, 19)):
    """Per-channel threshold maximising mean IoU (use on the validation split only)."""
    out = []
    for c in range(prob.shape[1]):
        ious = [np.nanmean(overlap_scores(prob[:, c:c + 1] > t, target[:, c:c + 1])["iou"]) for t in grid]
        out.append(float(grid[int(np.nanargmax(ious))]))
    return np.array(out)


def region_distance(prob, target, threshold=0.5, voxel_size_cm=20.0):
    """(n, 3) per-sample [min, mean, max] distance (cm) from predicted voxels to the nearest target voxel; NaN rows
    where the target is empty. prob, target: (n, R, R, R)."""
    from scipy.ndimage import distance_transform_edt
    n = len(prob)
    out = np.full((n, 3), np.nan)
    for i in range(n):
        t = target[i].astype(bool)
        if not t.any():
            continue
        p = prob[i] > threshold
        if not p.any():
            p = np.zeros_like(t)
            p[np.unravel_index(np.argmax(prob[i]), p.shape)] = True
        d = distance_transform_edt(~t)[p] * voxel_size_cm          # distance of each predicted voxel to the target
        out[i] = d.min(), d.mean(), d.max()
    return out


def saliency_scores(prob, target):
    """dict of (n, C) arrays: Pearson CC and ROC-AUC per sample and channel (NaN when undefined)."""
    from scipy.stats import rankdata
    n, C = prob.shape[:2]
    S = prob.reshape(n, C, -1).astype(np.float64)
    T = target.reshape(n, C, -1).astype(bool)
    cc, auc = np.full((n, C), np.nan), np.full((n, C), np.nan)
    for i in range(n):
        for c in range(C):
            s, t = S[i, c], T[i, c]
            n1 = t.sum()
            n0 = t.size - n1
            if n1 == 0 or n0 == 0:
                continue
            if s.std() > 0:
                cc[i, c] = np.corrcoef(s, t.astype(float))[0, 1]
            r = rankdata(s)
            auc[i, c] = (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {"cc": cc, "auc": auc}


def evaluate(prob, target, levels, thresholds=0.5, voxel_size_cm=20.0, saliency=True, distance=True):
    """Full report: {level: {iou, f1, precision, recall[, cc, auc][, dist_min/avg/max (foveal)]}, "mean": {...}}."""
    thr = np.broadcast_to(np.asarray(thresholds, float), (prob.shape[1],))
    pred = prob > thr[None, :, None, None, None]
    per = overlap_scores(pred, target)
    if saliency:
        per.update(saliency_scores(prob, target))
    report = {lev: {k: float(np.nanmean(v[:, c])) for k, v in per.items()} for c, lev in enumerate(levels)}
    for c, lev in enumerate(levels):
        report[lev]["threshold"] = float(thr[c])
    if distance and "foveal" in levels:
        c = list(levels).index("foveal")
        d = region_distance(prob[:, c], target[:, c], thr[c], voxel_size_cm)
        report["foveal"].update(dist_min=float(np.nanmean(d[:, 0])), dist_avg=float(np.nanmean(d[:, 1])),
                                dist_max=float(np.nanmean(d[:, 2])))
    keys = list(per)
    report["mean"] = {k: float(np.mean([report[l][k] for l in levels])) for k in keys}
    return report


def format_report(report, levels, metrics=("iou", "f1")):
    head = "".join(f"{l[:11]:>{12 * len(metrics)}s}" for l in levels)
    sub = "".join("".join(f"{m:>12s}" for m in metrics) for _ in levels)
    row = "".join("".join(f"{report[l].get(m, float('nan')):12.4f}" for m in metrics) for l in levels)
    return f"{head}\n{sub}\n{row}"
