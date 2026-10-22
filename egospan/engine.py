"""Training and evaluation loops (Appendix A.2: Adam 1e-4, no scheduler / weight decay, batch 16, 50 epochs, keep
the epoch with the best mean validation IoU over all span levels)."""
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import Config, _from_dict, save_config, to_dict
from .data.dataset import PackedBenchmark, SpanDataset, worker_init_fn
from .lift import LEVELS
from .losses import loss_from_config
from .metrics import evaluate, format_report, overlap_scores, tune_thresholds
from .models import build_model, count_parameters


def resolve_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_datasets(cfg, bench=None):
    d = cfg.data
    bench = bench or PackedBenchmark(d.root)
    if cfg.eval.space not in ("cartesian", "native"):
        raise ValueError(f"eval.space {cfg.eval.space!r}: cartesian | native")
    on_voxels = cfg.eval.space == "cartesian"          # spherical runs: map predictions to R^3 and score there
    kw = dict(fold=d.fold, horizon=d.horizon, levels=d.levels, drop_span_input=d.drop_span_input,
              crop_front=d.crop_front, gaze_paths=d.gaze_paths, video=d.video_features)
    return bench, {                         # val / test targets stay full-size, so scores are on the full grid
        "train": SpanDataset(bench, "train", augment=d.augment, max_shift=d.max_shift, seed=cfg.train.seed,
                             categories=d.train_categories, **kw),
        "val": SpanDataset(bench, "val", categories=d.train_categories, crop_target=False, eval_targets=on_voxels, **kw),
        "test": SpanDataset(bench, "test", categories=d.test_categories, crop_target=False, eval_targets=on_voxels,
                            **kw)}


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


@torch.no_grad()
def predict(model, dataset, device, batch_size=32, amp=True, num_workers=0, desc=None, progress=True):
    """Soft occupancy (n, L, R, R, R) float16 and targets (n, L, R, R, R) bool, in dataset order. Predictions of a
    spherical model are mapped to the R^3 Cartesian grid first (dataset.to_eval), so every run is scored alike."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs, targets = [], []
    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not progress):
        with _autocast(device, amp):
            logits = model(batch["x"].to(device, non_blocking=True), _cues(batch, device), _video(batch, device))
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        to_eval = getattr(dataset, "to_eval", None)
        if to_eval is not None:             # spherical model: cells -> R^3 voxels
            prob = to_eval(prob)
        missing = batch["y"].shape[-1] - prob.shape[-1]
        if missing > 0:                     # front-half model: the back half of the grid is predicted empty
            prob = np.pad(prob, [(0, 0)] * (prob.ndim - 1) + [(missing, 0)])
        probs.append(prob.astype(np.float16))
        targets.append(batch["y"].numpy().astype(bool))
    return np.concatenate(probs), np.concatenate(targets)


def _cues(batch, device):
    return batch["g"].to(device, non_blocking=True) if "g" in batch else None


def _video(batch, device):
    return batch["v"].to(device, non_blocking=True) if "v" in batch else None


def mean_iou(prob, target, threshold=0.5):
    return float(np.nanmean(np.nanmean(overlap_scores(prob > threshold, target)["iou"], 0)))


def load_matching(model, src):
    """copy the weights of a trained run (folder -> best.pt, or a .pt file) into `model` wherever name and shape
    match; -> (loaded, skipped) name lists. Skipped = new or resized layers, which keep their fresh initialisation."""
    path = os.path.join(src, "best.pt") if os.path.isdir(src) else src
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    own = model.state_dict()
    take = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(take, strict=False)
    return sorted(take), sorted(k for k in own if k not in take)


def build_optimizer(model, tcfg, steps_per_epoch):
    opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[tcfg.optimizer]
    opt = opt_cls(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    total = max(1, tcfg.epochs * steps_per_epoch)
    if tcfg.scheduler == "none":
        sched = None
    elif tcfg.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg.lr, total_steps=total, pct_start=0.1)
    elif tcfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    else:
        raise ValueError(tcfg.scheduler)
    return opt, sched


def train(cfg):
    t = cfg.train
    seed_everything(t.seed)
    device = resolve_device(t.device)
    os.makedirs(t.out_dir, exist_ok=True)
    save_config(cfg, os.path.join(t.out_dir, "config.yaml"))
    bench, ds = make_datasets(cfg)
    cue_dim, video_dim = ds["train"].cue_dim, ds["train"].video_dim
    copy_bins = max(1, bench.Tp // bench.past_units)                    # input bins of the last observed second
    model = build_model(cfg.model, n_out=len(cfg.data.levels), num_steps=bench.Tp, cue_dim=cue_dim,
                        in_channels=bench.in_channels, video_dim=video_dim, copy_bins=copy_bins,
                        copy_channels=[LEVELS.index(l) for l in cfg.data.levels]).to(device)
    if t.init:
        loaded, fresh = load_matching(model, t.init)
        print(f"init from {t.init}: {len(loaded)} tensors loaded, {len(fresh)} new"
              + (f" ({', '.join(fresh[:6])}{' ...' if len(fresh) > 6 else ''})" if fresh else ""), flush=True)
    print(f"{cfg.model.name}: {count_parameters(model) / 1e6:.2f} M parameters | device {device} | "
          f"train {len(ds['train'])} val {len(ds['val'])} test {len(ds['test'])}", flush=True)
    loader = DataLoader(ds["train"], batch_size=t.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.data.num_workers, worker_init_fn=worker_init_fn,
                        pin_memory=device.type == "cuda", persistent_workers=cfg.data.num_workers > 0)
    opt, sched = build_optimizer(model, t, len(loader))
    loss_fn = loss_from_config(cfg)
    best, log_path = -1.0, os.path.join(t.out_dir, "log.jsonl")
    open(log_path, "w").close()
    for epoch in range(1, t.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        bar = tqdm(loader, desc=f"epoch {epoch}/{t.epochs} train", leave=False, dynamic_ncols=True,
                   disable=not t.progress)
        for it, batch in enumerate(bar):
            x, y = batch["x"].to(device, non_blocking=True), batch["y"].to(device, non_blocking=True)
            with _autocast(device, t.amp):
                logits = model(x, _cues(batch, device), _video(batch, device))
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if t.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            total, n = total + loss.item(), n + 1
            bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total / n:.4f}")
            if not t.progress and t.log_every and it % t.log_every == 0:
                print(f"  epoch {epoch} it {it}/{len(loader)} loss {loss.item():.4f}", flush=True)
        pv, yv = predict(model, ds["val"], device, cfg.eval.batch_size, t.amp, desc=f"epoch {epoch}/{t.epochs} val",
                         progress=t.progress)
        viou = mean_iou(pv, yv, t.select_threshold)
        rec = dict(epoch=epoch, loss=total / max(n, 1), val_mean_iou=viou, sec=round(time.time() - t0, 1))
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        ck = {"model": model.state_dict(), "epoch": epoch, "num_steps": bench.Tp, "cue_dim": cue_dim,
              "in_channels": bench.in_channels, "video_dim": video_dim, "copy_bins": copy_bins,
              "config": to_dict(cfg)}
        torch.save(ck, os.path.join(t.out_dir, "last.pt"))
        if viou > best:
            best = viou
            torch.save(ck, os.path.join(t.out_dir, "best.pt"))
    return evaluate_run(t.out_dir, cfg=cfg, bench=bench, device=device)


def load_run(run_dir, device="cpu", which="best.pt"):
    """-> (model in eval mode, Config) from a run directory written by `train`."""
    ck = torch.load(os.path.join(run_dir, which), map_location="cpu", weights_only=False)
    cfg = _from_dict(Config, ck["config"])
    model = build_model(cfg.model, n_out=len(cfg.data.levels), num_steps=ck["num_steps"], cue_dim=ck.get("cue_dim", 0),
                        in_channels=ck.get("in_channels", 5), video_dim=ck.get("video_dim", 0),
                        copy_bins=ck.get("copy_bins", 1), copy_channels=[LEVELS.index(l) for l in cfg.data.levels])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


def evaluate_run(run_dir, cfg=None, bench=None, device=None, split="test", threshold=None, data_root=None):
    """Score a run on `split`; thresholds fixed (default 0.5) or tuned per level on val. Writes metrics_<split>.json."""
    device = device or resolve_device("auto")
    model, saved = load_run(run_dir, device)
    cfg = cfg or saved
    if data_root:
        cfg.data.root = data_root
    bench, ds = make_datasets(cfg, bench)
    thr = cfg.eval.threshold if threshold is None else threshold
    levels = cfg.data.levels
    if thr == "tune":
        pv, yv = predict(model, ds["val"], device, cfg.eval.batch_size, cfg.train.amp, desc="val (tune thresholds)",
                         progress=cfg.train.progress)
        thr = tune_thresholds(pv.astype(np.float32), yv)
    p, y = predict(model, ds[split], device, cfg.eval.batch_size, cfg.train.amp, desc=split,
                   progress=cfg.train.progress)
    vox_cm = 100 * bench.meta["lift"]["cube_size"] / bench.R
    native = bench.grid is not None and cfg.eval.space == "native"    # spherical cells: no cm distance on them
    rep = evaluate(p.astype(np.float32), y, levels, thr, vox_cm, cfg.eval.saliency, cfg.eval.distance and not native)
    rep["space"] = "spherical cells" if native else f"{bench.R}^3 voxels"
    with open(os.path.join(run_dir, f"metrics_{split}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    np.save(os.path.join(run_dir, f"probs_{split}.npy"), p)
    print(f"[{split}] {run_dir}\n" + format_report(rep, levels), flush=True)
    return rep
