"""The lab's questions (SensorySpotlight) as a file for `build.window.anchor=lab`, so our forecasters are built and
scored on exactly the lab's attention-shift questions and answers instead of the shifts we detect ourselves.

    S=~/SensorySpotlight/data/traj_forecast/attention_shift          # labels/, forecast/, senses_suite/
    # WHERE: one question per row of the suite's data_<tag>.npz (0.3 s before each shift), in the same order
    python scripts/lab_questions.py where --suite $S/senses_suite --labels $S/labels --out data/lab_q/aea_where.npz
    # WHEN: every --every-th query of the lab's forecast file (5 = every 0.5 s, as the Ryan et al. baseline)
    python scripts/lab_questions.py when --forecast $S/forecast/aea.npz --out data/lab_q/aea_when.npz

WHERE file: rec, t (s), qid (the row in data_<tag>.npz), truth / g0 (the lab's landing / current gaze, az / el in the
head frame at t), and from the lab's labels the shift's two held fixations: land_t0 / land_t1 (s, the landing
fixation's span), land_world / from_world (their mean world gaze directions). A question is matched to the shift
(events.csv, shift = 1) of its recording whose t_leave follows it by 0.3-0.4 s (the lab's rule: the shift's step is the
last sample at or before t_leave, the question 3 steps earlier); questions with no such shift are left out (reported).
WHEN file: rec, t (s), qid (the row in forecast/<ds>.npz), y (a shift starts within the lab's horizon).
"""
import argparse
import os

import numpy as np


def where(a):
    import pandas as pd
    Q = np.load(os.path.join(a.suite, f"data_{a.tag}.npz"))
    ds = a.tag.split("_")[0]
    E = pd.read_csv(os.path.join(a.labels, f"{ds}_events.csv"))
    F = pd.read_csv(os.path.join(a.labels, f"{ds}_fixations.csv"))
    E = E[E["shift"] == 1]
    by_rec = {r: g for r, g in E.groupby("rec")}
    rec, t = Q["rec"].astype(str), Q["t"].astype(np.float64)
    keep, i_from, i_to = [], [], []
    for i, (r, tq) in enumerate(zip(rec, t)):
        g = by_rec.get(r)
        if g is None:
            continue
        lead = g["t_leave"].to_numpy() - tq
        ok = (lead >= a.lead[0]) & (lead <= a.lead[1])
        if not ok.any():
            continue
        j = np.flatnonzero(ok)[np.argmin(np.abs(lead[ok] - np.mean(a.lead)))]
        keep.append(i)
        i_from.append(int(g["i_from"].iloc[j]))
        i_to.append(int(g["i_to"].iloc[j]))
    keep, i_from, i_to = map(np.array, (keep, i_from, i_to))
    gw = F[["gx", "gy", "gz"]].to_numpy(np.float64)
    out = dict(rec=rec[keep], t=t[keep], qid=keep, truth=Q["truth"][keep], g0=Q["g0"][keep],
               land_t0=F["t0"].to_numpy(np.float64)[i_to], land_t1=F["t1"].to_numpy(np.float64)[i_to],
               land_world=gw[i_to], from_world=gw[i_from])
    if "loc" in Q.files:
        out["loc"] = Q["loc"][keep]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {len(keep)} of {len(rec)} WHERE questions matched to a shift of the labels "
          f"({len(rec) - len(keep)} left out), {len(np.unique(rec[keep]))} recordings")


def when(a):
    D = np.load(a.forecast)
    q = np.arange(0, len(D["y"]), a.every)
    out = dict(rec=D["recs"][D["group"][q]].astype(str), t=D["t"][q].astype(np.float64), qid=q, y=D["y"][q])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {len(q)} WHEN questions (every {a.every}th), {out['y'].mean():.1%} positive, "
          f"{len(np.unique(out['rec']))} recordings")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("where", "when"))
    ap.add_argument("--suite", help="where: folder with data_<tag>.npz")
    ap.add_argument("--tag", default="aea_lag")
    ap.add_argument("--labels", help="where: folder with <ds>_events.csv and <ds>_fixations.csv")
    ap.add_argument("--lead", type=float, nargs=2, default=(0.25, 0.5),
                    help="where: t_leave - question time allowed (s; the lab's rule gives 0.3-0.4)")
    ap.add_argument("--forecast", help="when: the lab's forecast/<ds>.npz")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.kind == "where" and not (a.suite and a.labels):
        ap.error("where needs --suite and --labels")
    if a.kind == "when" and not a.forecast:
        ap.error("when needs --forecast")
    (where if a.kind == "where" else when)(a)


if __name__ == "__main__":
    main()
