"""Compare model variants over seeds: test IoU / F1 (mean ± std), FLOPs and inference latency.
Optional: --memory (GPU memory) and --e2e (end-to-end latency including voxelisation).

    R=runs/fovs_aria_lab
    python scripts/compare_variants.py --group full $R/egospanlift_s0 $R/egospanlift_s1 $R/egospanlift_s2 \
        --group front $R/front_s0 $R/front_s1 $R/front_s2 --device cuda --out results/full_vs_front

Accuracy   recomputed for every run from <run>/probs_test.npy at threshold 0.5 on the full 16^3 test grid (so runs
           scored earlier with --threshold tune are treated the same); std over seeds with ddof=1.
FLOPs      one forward pass, batch 1 (torch.utils.flop_counter).
Memory     (--memory) peak allocated CUDA memory, weights included: inference (batch 1) and one training step
           (the run's batch size, forward + backward + Adam step).
Latency    batch 1, after warm-up, median and p90 over --iters runs, with the run's bf16 autocast setting.
End to end (--e2e) per new observation, as in the paper's Table 1: preprocess the newest 100 ms step (cube crop +
           outlier removal), localise its spans (cone test), voxelise the 2 s window (earlier steps already
           preprocessed), copy to the GPU, run the model. The raw SLAM points are not on the server, so this runs
           on a synthetic recording calibrated to --points-per-step visible points (the lab's extract.log: ~3.5k-6k).
Run it on an otherwise idle GPU: latency and memory are affected by other jobs.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.dataset import PackedBenchmark          # noqa: E402
from egospan.data.synthetic import make_recording         # noqa: E402
from egospan.engine import _autocast, load_run            # noqa: E402
from egospan.lift import N_CHANNELS, EgoSpanLift, LiftConfig, rasterize_ray   # noqa: E402
from egospan.losses import loss_from_config               # noqa: E402
from egospan.metrics import overlap_scores                # noqa: E402
from egospan.models import count_parameters               # noqa: E402

_BENCH = {}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ------------------------------------------------------------------------------------------------------ accuracy
def accuracy(run, cfg):
    root = cfg.data.root
    if root not in _BENCH:
        _BENCH[root] = PackedBenchmark(root)
    b = _BENCH[root]
    Y = b.target(b.indices("test", cfg.data.fold), cfg.data.horizon, cfg.data.levels)
    path = os.path.join(run, "probs_test.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} missing: run  python scripts/evaluate.py --run {run}")
    P = np.load(path) > 0.5
    assert P.shape == Y.shape, f"{run}: predictions {P.shape} vs targets {Y.shape}"
    s = overlap_scores(P, Y)
    return {"iou": np.nanmean(s["iou"], 0), "f1": np.nanmean(s["f1"], 0)}


# ------------------------------------------------------------------------------------------------------ model costs
def input_shape(cfg, tp, n_ch=N_CHANNELS, R=16):
    return (1, tp, n_ch, R, R, R // 2 if cfg.data.crop_front else R)


def dummy_cues(model, batch, tp, device):
    """zero gaze-path input for timing a model with the gaze branch (None otherwise)."""
    d = getattr(model, "cue_dim", 0)
    return torch.zeros(batch, tp * 10, d, device=device) if d else None


def dummy_video(model, batch, shape, device):
    """zero camera input (features + in-view mask) for timing a model with camera features (None otherwise)."""
    d = getattr(model, "video_dim", 0)
    return torch.zeros(batch, d + 1, *shape[-3:], device=device) if d else None


def model_costs(model, cfg, tp, device, iters, memory=False):
    from torch.utils.flop_counter import FlopCounterMode
    shape = input_shape(cfg, tp, model.in_channels)
    x = (torch.rand(shape) < 0.03).float()
    model = model.to(device).eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x.to(device), dummy_cues(model, 1, tp, device), dummy_video(model, 1, shape, device))
    out = {"params_M": count_parameters(model) / 1e6, "gflops": fc.get_total_flops() / 1e9}
    xd, cd, vd = x.to(device), dummy_cues(model, 1, tp, device), dummy_video(model, 1, shape, device)
    with torch.no_grad():
        for _ in range(max(10, iters // 10)):                                      # warm-up
            with _autocast(device, cfg.train.amp):
                model(xd, cd, vd)
        sync(device)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            with _autocast(device, cfg.train.amp):
                model(xd, cd, vd)
            sync(device)
            ts.append(1e3 * (time.perf_counter() - t0))
    out["infer_ms_median"], out["infer_ms_p90"] = float(np.median(ts)), float(np.percentile(ts, 90))
    if memory and device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad(), _autocast(device, cfg.train.amp):
            model(xd, cd, vd)
        sync(device)
        out["infer_mem_GB"] = torch.cuda.max_memory_allocated(device) / 1e9
        # one training step at the run's batch size, with the run's optimizer type
        bs = cfg.train.batch_size
        xb = (torch.rand(bs, *shape[1:], device=device) < 0.03).float()
        yb = (torch.rand(bs, len(cfg.data.levels), *shape[3:], device=device) < 0.01).float()
        opt = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[cfg.train.optimizer](model.parameters(), lr=0)
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with _autocast(device, cfg.train.amp):
            logits = model(xb, dummy_cues(model, bs, tp, device), dummy_video(model, bs, shape, device))
        loss_from_config(cfg)(logits, yb).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync(device)
        out["train_step_mem_GB"] = torch.cuda.max_memory_allocated(device) / 1e9
        model.eval()
        del opt, xb, yb, logits
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------------------ end to end
def calibrated_recording(points_per_step, seconds):
    n_wall = 6000
    for _ in range(8):                                                   # visible points grow ~linearly with n_wall
        s = make_recording(seconds=seconds, n_wall=n_wall, seed=0)
        vis = np.diff(s.obs_ptr).mean()
        if abs(vis - points_per_step) / points_per_step < 0.1:
            break
        n_wall = int(n_wall * points_per_step / vis)
    return s, float(vis)


def voxelize_window(lifter, cached, frame, tp, spu, crop, n_ch=N_CHANNELS):
    """(Tp, n_ch, R, R, R or R/2) grid from the cached (points, labels, pose, gaze) of the window's steps; with
    n_ch = 6 the gaze ray of every step is drawn too (from that step's glasses position, in the anchor frame)."""
    R = lifter.cfg.resolution
    z0 = R // 2 if crop else 0
    X = np.zeros((tp, n_ch, R, R, R - z0), bool)
    for r, (p, lab, T, g) in enumerate(cached):
        ijk, inside = lifter.voxel_index(p, frame)
        inside &= ijk[:, 2] >= z0
        for c in range(N_CHANNELS):
            m = lab[c] & inside
            X[r // spu, c, ijk[m, 0], ijk[m, 1], ijk[m, 2] - z0] = True
        if n_ch > N_CHANNELS:
            ray = rasterize_ray(frame.points(T[:3, 3]), frame.dirs(T[:3, :3] @ g), lifter.cfg.cube_size, R)
            X[r // spu, N_CHANNELS] |= ray[..., z0:]
    return X


def end_to_end(model, cfg, tp, device, n, s):
    lifter = EgoSpanLift(LiftConfig(grid_frame="local"))                 # the lab data's frame, for both variants
    P = 20                                                               # 2 s observed at 10 Hz
    spu = P // tp                                                        # steps per input bin (hard bins)
    cache = {}

    def prepare(k):
        p = s.points_world[s.visible(k)]
        p = p[lifter.select(p, s.T_world_cpf[k])]
        return p, lifter.classify(p, s.T_world_cpf[k], s.gaze_cpf[k]), s.T_world_cpf[k], s.gaze_cpf[k]

    for k in range(P - 1):
        cache[k] = prepare(k)
    rows = []
    model = model.to(device).eval()
    for k in range(P - 1, min(len(s), P - 1 + n + 5)):
        t0 = time.perf_counter()
        p = s.points_world[s.visible(k)]
        keep = lifter.select(p, s.T_world_cpf[k])
        p = p[keep]
        t1 = time.perf_counter()
        lab = lifter.classify(p, s.T_world_cpf[k], s.gaze_cpf[k])
        t2 = time.perf_counter()
        cache[k] = (p, lab, s.T_world_cpf[k], s.gaze_cpf[k])
        cache.pop(k - P, None)
        X = voxelize_window(lifter, [cache[j] for j in range(k - P + 1, k + 1)], lifter.anchor(s.T_world_cpf[k]),
                            tp, spu, cfg.data.crop_front, model.in_channels)
        t3 = time.perf_counter()
        with torch.no_grad():
            x = torch.from_numpy(X[None].astype(np.float32)).to(device)
            with _autocast(device, cfg.train.amp):
                cues, vid = dummy_cues(model, 1, tp, device), dummy_video(model, 1, x.shape, device)
                torch.sigmoid(model(x, cues, vid).float())
            sync(device)
        t4 = time.perf_counter()
        rows.append([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0])
    rows = 1e3 * np.array(rows[5:])                                      # first iterations warm up caches / CUDA
    keys = ["preprocess_ms", "localize_ms", "voxelize_ms", "transfer_infer_ms", "e2e_ms"]
    out = {k: float(np.median(rows[:, i])) for i, k in enumerate(keys)}
    out["e2e_ms_p90"] = float(np.percentile(rows[:, 4], 90))
    return out


# ------------------------------------------------------------------------------------------------------ report
def pm(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", nargs="+", action="append", required=True, metavar=("NAME", "RUN"),
                    help="a variant name followed by its run directories (one per seed); repeat per variant")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=200, help="latency repetitions")
    ap.add_argument("--e2e-samples", type=int, default=100)
    ap.add_argument("--points-per-step", type=float, default=5000)
    ap.add_argument("--out", default="results/compare")
    ap.add_argument("--memory", action="store_true", help="also measure peak GPU memory")
    ap.add_argument("--e2e", action="store_true", help="also measure end-to-end latency including voxelisation")
    a = ap.parse_args()
    device = torch.device(a.device)
    rec = vis = None
    if a.e2e:
        rec, vis = calibrated_recording(a.points_per_step, (a.e2e_samples + 40) / 10)
        print(f"end-to-end recording: {vis:.0f} visible points per step (target {a.points_per_step:.0f})", flush=True)

    results = {}
    for name, *runs in a.group:
        print(f"\n== {name}: {len(runs)} runs", flush=True)
        acc, levels, model, cfg, tp = [], None, None, None, None
        for r in runs:
            model, cfg = load_run(r, "cpu")
            tp = model.num_steps
            acc.append(accuracy(r, cfg))
            levels = cfg.data.levels
            print(f"   {r}: mean IoU {acc[-1]['iou'].mean():.4f}  F1 {acc[-1]['f1'].mean():.4f}", flush=True)
        costs = model_costs(model, cfg, tp, device, a.iters, a.memory)      # architecture is the same across seeds
        if a.e2e:
            costs.update(end_to_end(model, cfg, tp, device, a.e2e_samples, rec))
        model.cpu()
        iou, f1 = np.stack([x["iou"] for x in acc]), np.stack([x["f1"] for x in acc])
        results[name] = {"runs": runs, "levels": levels, "input": list(input_shape(cfg, tp, model.in_channels)[1:]),
                         "iou_per_run": iou.tolist(), "f1_per_run": f1.tolist(), **costs}

    lines = [f"# {' vs '.join(results)}", "",
             "Test split, threshold 0.5, full 16^3 grid; mean ± std over seeds (ddof=1).", "",
             "| Variant | Seeds | " + " | ".join(f"{l} IoU" for l in levels) + " | Mean IoU | Mean F1 |",
             "|---|---|" + "---|" * len(levels) + "---|---|"]
    for name, r in results.items():
        iou, f1 = np.array(r["iou_per_run"]), np.array(r["f1_per_run"])
        lines.append(f"| {name} | {len(iou)} | " + " | ".join(pm(iou[:, i]) for i in range(iou.shape[1]))
                     + f" | {pm(iou.mean(1))} | {pm(f1.mean(1))} |")
    lines += ["", f"Cost (batch 1, device {a.device})", "",
              "| Variant | Input | Params | GFLOPs | Inference latency ms (median / p90) |", "|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {'x'.join(map(str, r['input']))} | {r['params_M']:.1f} M | {r['gflops']:.1f} | "
                     f"{r['infer_ms_median']:.2f} / {r['infer_ms_p90']:.2f} |")
    if a.memory and "infer_mem_GB" in next(iter(results.values())):
        lines += ["", "| Variant | Inference memory (batch 1) | Training-step memory (batch size of the run) |",
                  "|---|---|---|"]
        lines += [f"| {n} | {r['infer_mem_GB']:.2f} GB | {r['train_step_mem_GB']:.2f} GB |" for n, r in results.items()]
    if a.e2e:
        lines += ["", f"End-to-end latency, ms (median; synthetic recording, {vis:.0f} visible points/step)", "",
                  "| Variant | Preprocess | Localize | Voxelize | Transfer + infer | Total (median / p90) |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {n} | {r['preprocess_ms']:.2f} | {r['localize_ms']:.2f} | {r['voxelize_ms']:.2f} | "
                  f"{r['transfer_infer_ms']:.2f} | {r['e2e_ms']:.2f} / {r['e2e_ms_p90']:.2f} |"
                  for n, r in results.items()]
    lines += ["", "Per-run mean IoU: " + "; ".join(
        f"{n}: " + ", ".join(f"{v:.4f}" for v in np.array(r["iou_per_run"]).mean(1)) for n, r in results.items())]
    if torch.cuda.is_available() and device.type == "cuda":
        lines.append(f"GPU: {torch.cuda.get_device_name(device)}")
    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".md", "w") as f:
        f.write(text + "\n")
    with open(a.out + ".json", "w") as f:
        json.dump({"device": a.device, "e2e_points_per_step": vis, "variants": results}, f, indent=1)
    print(f"\nsaved {a.out}.md and {a.out}.json")


if __name__ == "__main__":
    main()
