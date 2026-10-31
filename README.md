# Gaze Beyond the Frame: 3D visual span forecasting

An unofficial codebase for *Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span* (Yun et al., NeurIPS 2025).
The authors have not released their code.

The task: given 2 s of a person's past gaze, head pose and SLAM keypoints, predict where they will look in 3D over the
next 2 s, as voxel regions at four levels: head orientation (55°), peripheral (30°), central (8°) and foveal (2°).

Sources:
- **EgoSpanLift and the benchmark builder** are refactored from the lab's `aria-3d-analysis` branch of
  SensorySpotlight (`tools/fovs_build.py`, `tools/fovs_models.py`). A test checks that this code reproduces their grids
  exactly when given their settings.
- **The 3D U-Net blocks** are vendored from [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet). The paper's
  appendix says the authors built their model with that library.
- **Everything else** (the forecaster, losses, metrics, baselines, 2D projection) follows the paper and its appendix.

## Layout

```
egospan/
  lift.py            EgoSpanLift (Sec. 3): Eq. 1 point selection, Eq. 2 cone classification, Eq. 3 voxelisation
  data/
    aria.py          Aria MPS readers: Aria Everyday Activities (zips) and Ego-Exo4D (take folders)
    streams.py       RecordingStreams: gaze-timeline-synchronised poses, gaze, points, visibility
    build.py         windows -> bit-packed samples per recording; pack -> all.npz + splits.npz (+ CV folds)
    dataset.py       torch Dataset (keeps grids bit-packed in RAM), target horizon / level selection
    augment.py       axis permutation + flips that never turn the scene upside down, ±2 voxel translation
    synthetic.py     fake Aria-like recordings for tests and smoke runs
  models/
    blocks.py        pytorch-3dunet building blocks (MIT)
    forecaster.py    SpanForecaster (Sec. 4) and the flat 3D U-Net baseline
  losses.py          Dice (Eq. 4), BCE
  metrics.py         IoU / F1 / precision / recall per sample, foveal distance (Tab. 3), CC / AUC (Tab. 6)
  engine.py          training loop, model selection on val IoU, evaluation
  baselines.py       global prior, persistence, 2D gaze + GP pose forecast + voxel lift
  project2d.py       3D forecast -> 2D gaze anticipation (Sec. 5.3)
configs/             fovs_aria.yaml (paper), fovs_egoexo.yaml, fovs_aria_sensoryspotlight.yaml (lab settings), smoke.yaml
scripts/             build_dataset, convert_sensoryspotlight, train, evaluate, baselines, eval_2d, benchmark_latency, visualize
tests/               38 tests, including exact equivalence with the lab's fovs_build.py
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e ".[viz,dev]"
pytest -q                                # 38 passed
```

You need `projectaria-tools` only to build the benchmark from raw recordings, because the device calibration lives in
the VRS file. The lab keeps it in a separate environment (`pip install projectaria-tools==1.5.7`). The `egospan.data`
package does not import torch, so the builder runs in that environment too.

## Quick start without any downloads

```bash
python scripts/build_dataset.py synthetic --config configs/smoke.yaml --out data/synthetic --n 10
python scripts/build_dataset.py pack      --config configs/smoke.yaml --out data/synthetic
python scripts/train.py                   --config configs/smoke.yaml        # ~30 s on a laptop CPU
python scripts/baselines.py               --config configs/smoke.yaml
python scripts/visualize.py --run runs/smoke --index 0 --out runs/smoke/figs
```

## FoVS-Aria

**1. Download** [Aria Everyday Activities](https://www.projectaria.com/datasets/aea/) into one folder. You need five
files per recording: `*_main_recording.vrs`, `*_annotations.zip`, `*_mps_eye_gaze.zip`,
`*_mps_slam_trajectories.zip` and `*_mps_slam_points.zip`. The lab server already holds the raw recordings, so ask
before downloading them again.

**2. Build** (run in the environment with projectaria-tools):

```bash
python scripts/build_dataset.py aea  --config configs/fovs_aria.yaml --src /data/aea --out data/fovs_aria --workers 8
python scripts/build_dataset.py pack --config configs/fovs_aria.yaml --out data/fovs_aria
```

The test split is every recording from location 4, as in the paper. `pack` also writes location-wise cross-validation
folds; train on one with `--set data.fold=2`. The paper also curated some recordings by hand (phone/TV-only and poor
SLAM). List those recordings, one per line, and pass the file with `--set build.pack.exclude_file=excluded.txt`.

**3. Train and evaluate:**

```bash
python scripts/train.py --config configs/fovs_aria.yaml                     # Adam 1e-4, bs 16, 50 epochs
python scripts/evaluate.py --run runs/fovs_aria/egospanlift_s0              # IoU/F1/Pr/Re, CC/AUC, foveal distance
python scripts/baselines.py --config configs/fovs_aria.yaml                 # global prior, persistence, center prior
python scripts/eval_2d.py --run runs/fovs_aria/egospanlift_s0               # Sec. 5.3
```

## Rebuild FoVS-Aria from the raw files on the lab server

`/data/SS/Aria_data` has the raw AEA SLAM zips (`mps_slam/`) and eye gaze + metadata (`gaze/<rec>/`), but no `.vrs`
files. The head-to-eye transform that normally comes from the `.vrs` is recovered from the lab's built benchmark
(T_device_cpf = T_world_device^-1 T_world_cpf at the lab's anchors; constant per recording, checked).

```bash
python scripts/derive_device_cpf.py --lab /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria \
    --slam-root /data/SS/Aria_data/mps_slam --out data/aea_cpf
python scripts/build_dataset.py aea-server --config configs/fovs_aria_lab_rebuild.yaml \
    --slam-root /data/SS/Aria_data/mps_slam --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf \
    --out data/fovs_aria_rebuild --limit 5                      # first a few recordings ...
python scripts/validate_against_lab.py --ours data/fovs_aria_rebuild --lab /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria
# ... then all of them (drop --limit; finished recordings are skipped), and pack
python scripts/build_dataset.py pack --config configs/fovs_aria_lab_rebuild.yaml --out data/fovs_aria_rebuild
```

`configs/fovs_aria_lab_rebuild.yaml` uses the lab's settings (anchor-CPF frame, crop around the anchor, `dist_std <
0.1`, one outlier pass over the whole cloud, no sample filter), so the rebuild matches their data sample for sample
(tested against their build code). Build the temporal-bin / gaze-ray variants with the same config plus `--set
build.window...` into separate `--out` folders; they then share samples, split and CV folds with the lab data.

## Train on the lab's already-built FoVS-Aria (no raw recordings)

On the lab server the built benchmark is at `/data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria/`. It has the span grids but
not the raw SLAM points, so it cannot be rebuilt with other settings, but it is enough to train and evaluate.

```bash
python scripts/convert_sensoryspotlight.py --src /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria --out data/fovs_aria_lab
python scripts/train.py --config configs/fovs_aria_lab.yaml
```

The lab stores the target as four 0.5 s bins. The converter keeps them as 4 future units, so the paper's 2 s union is the
default and `--set data.horizon=1` (...`4`) gives the shorter windows. Grids use the head's axes (vertical axis 1). Per-step
poses are not stored, so `baselines.py`'s 2D-lift rows and `eval_2d.py` need the raw recordings. The built data and
cross-validation folds (`--set data.fold=2`) carry over.

## Front-half variant (our addition, not in the paper)

On head-aligned grids (the converted lab data), almost all spans lie in front of the wearer, so the model can see and
predict only the front half of the cube: 16 x 16 x 8 instead of 16 x 16 x 16.

```bash
python scripts/train.py --config configs/fovs_aria_lab.yaml --set data.crop_front=true train.out_dir=runs/fovs_aria_lab/front_s0
```

Validation and test are still scored on the full 16^3 grid (the back half is predicted empty), so the scores compare
directly with the full-cube model. The option refuses to run on world-aligned grids, which have no front. On the lab's
FoVS-Aria (location-4 test, 3 seeds) it gave mean IoU 0.4229 +- 0.0003 against 0.4190 +- 0.0012 for the full cube, with
half the FLOPs (19 against 38 GFLOPs per sample). Latency at batch size 1 was the same (about 2.4 ms on the lab GPU).

Helper scripts:
- `scripts/run_seed_comparison.sh 0 1`: trains seeds 0-2 of both variants (skipping finished runs), then runs the comparison.
- `scripts/compare_variants.py --group NAME RUN... --group NAME RUN...`: IoU / F1 mean +- std over seeds, FLOPs and
  inference latency (`--memory` and `--e2e` add GPU memory and end-to-end latency).
- `scripts/inspect_data.py --root data/fovs_aria_lab`: stored sizes, one sample's shapes, occupancy statistics and
  consistency checks of a packed benchmark.

## Gaze-path branch (our addition, not in the paper)

The span grids merge the ten 10 Hz steps of each second into one binary union, which loses the order, speed and
direction of gaze movement and where the gaze points now. This branch adds that back: the past 2 s of gaze (20 steps,
expressed in the grid frame using the head rotations), the eye-in-head gaze and the head rotation per step go through a
small GRU, whose output becomes the first token of the causal transformer: [v_gaze, v_1, v_2, v_head].

```bash
# once: per-sample gaze/head paths from the lab's files (checks the head-rotation convention against SLAM future gaze)
python scripts/build_gaze_paths.py --src /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria \
    --gaze-root /data/SS/Aria_data/gaze \
    --head-root /data/SS/SS/Dynamic_Fusion_ALFA/data/spatial_audio_pilot --out data/fovs_aria_lab
python scripts/train.py --config configs/fovs_aria_lab.yaml \
    --set data.gaze_paths=true data.crop_front=true train.out_dir=runs/fovs_aria_lab/front_gaze_s0
```

Augmentation mirrors the gaze and head vectors together with the grid. The branch adds well under 1M parameters.

## Temporal input bins (our addition, not in the paper; needs the raw recordings)

The paper merges the 10 observations of each second into one binary grid, so order, recency, dwell time and speed of
gaze movement are lost before the model sees them. `build.window.time_bins` and `build.window.soft` keep them:

| Setting | Input per sample | Meaning |
|---|---|---|
| `time_bins: 2` (default) | 2 binary grids | the paper's 1 s units |
| `time_bins: 4` | 4 binary grids | 0.5 s units (hard bins) |
| `time_bins: 4, soft: true` | 4 grids with values in [0, 1] | each 10 Hz step is lifted to a binary mask, then shared between its two nearest bins by linear interpolation; a value is the fraction of the bin's weighted time the voxel was active (dwell), and the bin order carries recency / direction |
| `ray_channel: true` (with any of the above) | + a 6th channel per bin | the voxels along each step's gaze ray, from that step's glasses position (in the anchor frame) to the grid edge; needs no SLAM points, so it shows where the eyes pointed even at bare walls; direction evidence, not foveal occupancy (no depth) |

```bash
python scripts/build_dataset.py aea  --config configs/fovs_aria.yaml --src /data/aea --out data/fovs_aria_b4soft \
    --set build.window.time_bins=4 build.window.soft=true            # add build.window.ray_channel=true for rays
python scripts/build_dataset.py pack --config configs/fovs_aria.yaml --out data/fovs_aria_b4soft
python scripts/train.py --config configs/fovs_aria.yaml --set data.root=data/fovs_aria_b4soft train.out_dir=runs/b4soft_s0
python scripts/shift_breakdown.py --runs runs/b2_s0 runs/b4soft_s0     # IoU by gaze-shift size
```

The target is unchanged (binary union over the forecast window), so scores stay comparable. The model takes one
transformer token per bin. Soft grids are stored as uint8 (FoVS-Aria: ~2.6 GB). Hard bins with `time_bins: 2`
reproduce the default inputs exactly (tested). The lab's already-built data cannot be re-binned: it only stores
per-second unions.

## Distance-aware loss (our addition, not in the paper)

Dice only rewards overlap, so a foveal guess one voxel off costs as much as one a metre off. `train.loss=dice+dist` adds,
for `train.dist_levels` (default central + foveal), the expected distance (in voxels) of the predicted probability
mass from the target, and with `train.dist_mode=bidirectional` also how far each target voxel is from the predicted
mass (so a target with several spots must be covered everywhere). Empty targets get no distance term.

```bash
python scripts/train.py --config configs/fovs_aria_lab.yaml --set data.crop_front=true data.fold=1 \
    train.loss=dice+dist train.dist_weight=0.1 train.dist_mode=oneway train.out_dir=runs/.../front_dist0.1_f1
```

Start with `dist_weight` 0.05 / 0.1 / 0.2. Judge it on central / foveal IoU and the foveal `dist_avg` in
`metrics_test.json`. It costs ~10-15 ms per batch.

## Why large gaze shifts are hard: what could predict them (analysis, no training)

`scripts/shift_destination_analysis.py` uses the lab's attention-shift definition (`egospan/shifts.py`: 75 deg/s
fixations, held 150 ms, >= 30 deg between consecutive held fixations, in the world frame) on the raw recordings and asks
whether any information could predict where the gaze goes at a shift: revisits of earlier fixations, head and eye cues
for the direction, whether eye-state features predict that a shift is coming, and (`--scene`) whether the destination
was already in the scene seen earlier. Each is compared with a control. Run it before building new inputs:

```bash
python scripts/shift_destination_analysis.py --slam-root /data/SS/Aria_data/mps_slam \
    --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf --out results/shift_destination.json --limit 20
```

## Camera features (the SensorySpotlight `vid` input)

The analysis above suggests the destination of a large gaze shift depends on what the objects are, which geometry
cannot show. The lab's frozen V-JEPA 2 features of the last ~0.4 s of video can be lifted into the grid (every voxel
takes the feature of the image cell its centre projects into; `egospan/video.py`, identical to the lab's mapping),
reduced by a 1x1x1 conv to 16 channels and added, with an in-view mask, to every input grid.

```bash
python scripts/attach_video_features.py --lab /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria --bench data/fovs_aria_lab
python scripts/train.py --config configs/fovs_aria_lab.yaml \
    --set data.video_features=true data.crop_front=true data.fold=1 train.out_dir=runs/fovs_aria_lab/cv/front_vid_f1
```

Needs grids in the anchor-CPF frame (the lab data and our lab rebuild). Samples without video have zero features.

## Inter-span attention (our addition, not in the paper)

The paper averages the deepest encoder level into one token per grid, so the transformer never sees where things were,
and the decoder decides each voxel alone. `model.span_attention` taps encoder level 2 (8 x 8 x 4 cells of 40 cm in the
front half, about the size of where the eyes land after a large shift) of all input grids: every cell attends to every
cell of its own and earlier grids, `bias` adds a learned score from each cell's past span shares (peripheral, "seen,
not looked at", in view, ...). The zero-initialised update is added to the level-2 skip, so a fresh model equals the
one without it. +1.7 M parameters.

```bash
for m in bias attn; do
  python scripts/train.py --config configs/fovs_aria_lab.yaml --set data.video_features=true data.crop_front=true \
      data.fold=4 model.span_attention=$m train.out_dir=runs/fovs_aria_lab/cv/front_vid_sa_${m}_f4
done
```

## Direction-adaptive spherical grid (our addition, not in the paper; needs the raw recordings)

`build.lift.grid=spherical` bins the spans into azimuth x elevation x distance cells around the wearer: 5 deg within
55 deg of head forward, 10 deg at the sides, ~22 deg behind (40 x 24 x 8 cells, `egospan/spherical.py`), covering all
360 deg, from the same points as the Cartesian build (which is also stored, as `YF_cart`).

`configs/fovs_aria_sph.yaml` trains and scores on the spherical cells themselves (`eval.space: native`). Those IoUs are
on a different scale from 20 cm voxel IoUs (cell size changes IoU), so compare spherical runs with each other;
`cv_compare.py` refuses to mix the two. `eval.space: cartesian` maps predictions back to the 20 cm voxels instead,
which has a ceiling (cells don't line up with voxels): `scripts/spherical_ceiling.py` measures it.

```bash
python scripts/build_dataset.py aea-server --config configs/fovs_aria_sph.yaml \
    --slam-root /data/SS/Aria_data/mps_slam --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf \
    --out data/fovs_aria_sph
python scripts/build_dataset.py pack --config configs/fovs_aria_sph.yaml --out data/fovs_aria_sph
python scripts/attach_video_features.py --lab /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria --bench data/fovs_aria_sph
python scripts/train.py --config configs/fovs_aria_sph.yaml --set data.fold=4 train.out_dir=runs/fovs_aria_sph/cv/sph_f4
```

### Comparing box and spherical runs: point-level scores

Voxel IoU and cell IoU are different scales. Both grids predict the same thing, which scene points the person will
look at, so score both on the future SLAM points (`egospan/point_metrics.py`): a point is predicted if its voxel /
cell is on, and true if it really was in the span. Each grid also gets its "perfect grid" row (the true target in that
grid: its ceiling, below 1 because a cell lumps true and other points together) and "copy last 1 s"; plus
persistence on the exact points ("[points] copy last 1 s") and the coverage: how many future true points were already
observed, i.e. what a point-level refinement on the known map could work on.

```bash
python scripts/build_future_points.py --bench data/fovs_aria_sph --part test --fold 4 \
    --slam-root /data/SS/Aria_data/mps_slam --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf
python scripts/point_score.py --points data/fovs_aria_sph/points_test_f4.npz \
    --runs runs/fovs_aria_lab/cv/front_vid_f4 runs/fovs_aria_sph/cv/sph_vid_f4
```

## Attention-shift benchmark: where does attention land? (our addition)

`build.window.anchor=shift` builds one sample per attention shift (`egospan/shifts.py`: 75 deg/s fixations held
>= 150 ms, >= 30 deg between consecutive held fixations), anchored 0.3 s before the gaze leaves the fixation (the
lab's WHERE `lag` questions). Input = the usual 2 s of span grids; target = the landing fixation's span cones; also
stored: the landing and current-fixation directions. `train.init` fine-tunes from a trained forecaster (same fold);
`scripts/landing_eval.py` gives the WHERE metric (angle error of the predicted landing direction, in / out of view,
best of 3, vs "stay"), optionally on exactly the lab's questions; `scripts/attach_shift_video.py` takes the camera
tokens from the WHERE suite's cache (V-JEPA 2 or DINOv2) and reports how many events agree with the lab's.

```bash
python scripts/build_dataset.py aea-server --config configs/fovs_aria_shift_sph.yaml \
    --slam-root /data/SS/Aria_data/mps_slam --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf \
    --out data/fovs_aria_shift_sph
python scripts/build_dataset.py pack --config configs/fovs_aria_shift_sph.yaml --out data/fovs_aria_shift_sph
python scripts/attach_shift_video.py --bench data/fovs_aria_shift_sph \
    --suite ~/SensorySpotlight/data/traj_forecast/attention_shift/senses_suite
python scripts/train.py --config configs/fovs_aria_shift_sph.yaml --set data.fold=4 \
    train.init=runs/fovs_aria_sph/cv/sph_f4 train.out_dir=runs/fovs_aria_shift_sph/cv/ft_f4
python scripts/landing_eval.py --runs runs/fovs_aria_shift_sph/cv/ft_f4 \
    --lab-queries /data/SS/SS/Dynamic_Fusion_ALFA/data/traj_forecast/attention_shift/senses_suite/data_aea_lag.npz
```

### On the lab's own questions (baseline rows for the lab's WHERE / WHEN tables)

`build.window.anchor=lab` builds one sample per question of the lab's (from `scripts/lab_questions.py`), so the model
is scored on exactly the lab's shifts and answers, not on the ones we detect (only ~37% of those match the lab's).
`scripts/export_lab_probs.py` runs one model per CV fold on its own held-out location, maps the foveal map onto the
lab's candidate directions and writes `probs_<name>.npz` in the lab's format (`gbf`, `gbf30` = nothing within 30 deg of
the current gaze, `has` = covered questions), for `attention_where_metrics.py` (`WM_EXTRA`) and paired tests;
`when` scores the lab's WHEN questions (AUC).

```bash
S=~/SensorySpotlight/data/traj_forecast/attention_shift
python scripts/lab_questions.py where --suite $S/senses_suite --labels $S/labels --out data/lab_q/aea_where.npz
python scripts/lab_questions.py when --forecast $S/forecast/aea.npz --out data/lab_q/aea_when.npz
for k in where when; do
  python scripts/build_dataset.py aea-server --config configs/fovs_aria_shift_sph.yaml \
      --set build.window.anchor=lab build.window.questions=data/lab_q/aea_$k.npz \
      --slam-root /data/SS/Aria_data/mps_slam --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf \
      --out data/fovs_aria_labq_${k}_sph
  python scripts/build_dataset.py pack --config configs/fovs_aria_shift_sph.yaml --out data/fovs_aria_labq_${k}_sph
done
# as trained (the forecasters of fovs_aria_sph.yaml, one per fold) / retrained: fine-tuned on the WHERE questions
python scripts/export_lab_probs.py where --bench data/fovs_aria_labq_where_sph --suite $S/senses_suite \
    --name gbf_astrained --runs runs/fovs_aria_sph/cv/sph_f{1,2,3,4,5}
python scripts/export_lab_probs.py when --bench data/fovs_aria_labq_when_sph --out results/gbf_when_aea.npz \
    --runs runs/fovs_aria_sph/cv/sph_f{1,2,3,4,5} --ours $S/baselines/when_ours_aea.npy
```

**ADT and HD-EPIC** (`build_dataset.py adt | hdepic`, readers in `egospan/data/aria.py`) follow the lab's timelines
(HD-EPIC gaze at 10 Hz as the lab subsamples it; ADT's ground-truth gaze, every 3rd 30 Hz sample) and take the lab's
held-out groups as locations (`--folds`, from SensorySpotlight's `tools/export_gbf_folds.py`), so the location-wise CV
folds are the lab's. ADT needs its MPS SLAM points (not in the lab's download); its ground-truth world frame is mapped
onto the SLAM frame per sequence. Video tokens use HD-EPIC's own camera mapping. `export_lab_probs.py --eval-folds`
scores a model trained on another dataset (the AEA model, "as released") on every fold; `when --forecast` adds the
lead-time AUC, `--ours <file>.npz:<key>` compares with the lab's final WHEN model, `--reuse` rescores saved scores.
`scripts/run_lab_more.sh` runs all of it (ADT / HD-EPIC WHERE and WHEN, WHEN retrained on all three datasets);
`scripts/run_where.sh` only the WHERE rows and table, skipping whatever is already done (ADT once its points are in).
`scripts/render_scene.py` renders one recording's SLAM points as a 3D scene (an HTML viewer with the head trajectory and
gaze ray over time, and a .ply); `hdepic --kitchen` merges every session of the kitchen.
`scripts/render_adt_scene.py` draws an ADT sequence from its ground truth alone (object boxes, head path, 3D fixations,
the next shift's landing) for figures, with a "Save PNG" at 3x resolution; `scripts/render_hdepic_scene.py` the same for
HD-EPIC, on the kitchen's point cloud with the annotated moved objects (fixation depth from the points).

### 3D scene features for the lab's WHERE model

`scripts/export_where_geometry.py` turns a spherical shift benchmark into a per-direction input for the lab's WHERE
suite: for each question and each of its 468 candidate directions, the fraction of that 10 deg sky patch holding each
span / seen surfaces in each of the 2 s, and how near the closest surface is (only what was observed up to the
question). It first checks that the lab's landing directions agree with ours (same head frame). Use the room-scale
config (8 m cube, distances to 6.9 m) so surfaces across the room count.

```bash
python scripts/build_dataset.py aea-server --config configs/fovs_aria_shift_sph_far.yaml \
    --slam-root /data/SS/Aria_data/mps_slam --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf \
    --out data/fovs_aria_shift_sph_far
python scripts/build_dataset.py pack --config configs/fovs_aria_shift_sph_far.yaml --out data/fovs_aria_shift_sph_far
python scripts/export_where_geometry.py --bench data/fovs_aria_shift_sph_far --suite $S --out $S/geo3d_aea_lag_far.npy
# SensorySpotlight: WS_VARIANTS=...,3d,mem+3d WS_GEO3D_SUF=_far ... attention_senses_suite.py train aea lag location residual ...
```

## Copy path (our addition, not in the paper)

Copying the last observed second is hard to beat when the gaze stays put. `model.copy_path` feeds the last second's
spans straight to the output: `add` (logits + w * last, w per level from 0) or `gate` (p = g * last + (1 - g) *
forecast, g per cell from the decoder features, starting near 0), so the network keeps the recent region where gaze is
steady and only has to learn the changes.

```bash
for m in gate add; do
  python scripts/train.py --config configs/fovs_aria_sph.yaml --set data.fold=4 data.video_features=true \
      model.copy_path=$m train.out_dir=runs/fovs_aria_sph/cv/sph_vid_copy${m}_f4
done
```

## Which inputs does a trained model use? (no retraining)

Re-scores the test split with one input removed at a time: zeroed, or taken from another test sample ("shuffle", the
fairer test). Inputs: each past span channel, scene, the gaze ray, all four spans, camera features, gaze path.

```bash
python scripts/input_ablation.py --run runs/fovs_aria_lab/cv/front_vid_f4 \
    --lab-src /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria       # also split central / foveal by gaze-shift size
```

## Where do the eyes land after a shift? (data only; optional model recall)

Landing voxels = future foveal voxels that were not foveal in the past. Reports, by gaze-shift size, how many of them
lay in the past orientation / peripheral span ("seen, not looked at"), the scene, the camera view; how large those
candidate regions are (and the chance level of guessing inside them); with `--run`, the model's recall on start vs
landing voxels.

```bash
python scripts/landing_check.py --bench data/fovs_aria_lab --part all --lab-src /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria
python scripts/landing_check.py --run runs/fovs_aria_lab/cv/front_vid_f4 --lab-src /data/SS/SS/Dynamic_Fusion_ALFA/data/fovs_aria
```

## Reproducing the paper's experiments

| Paper | How |
|---|---|
| Tab. 2 full model | `train.py --config configs/fovs_aria.yaml` |
| Tab. 2 L_BCE | `--set train.loss=bce` |
| Tab. 2 w/o global embedding | `--set model.head=none` |
| Tab. 2 w/o previous span | `--set data.drop_span_input=true` |
| Tab. 2 single-task | `--set "data.levels=[foveal]"`, one run per level |
| Tab. 2 global prior, 2D center prior | `scripts/baselines.py` |
| Tab. 2 GLC / CSTS + EgoSpanLift + GP | `baselines.py --gaze-preds preds.npz --name CSTS` (the 2D models are not included) |
| Tab. 3 foveal metric distance | `dist_min/avg/max` in `metrics_test.json` |
| Tab. 4 FoVS-EgoExo | `configs/fovs_egoexo.yaml`; D/R variants: `--set build.lift.cube_size=6.4 build.lift.resolution=32` at build time |
| Fig. 4-a cross-category | `--set "data.train_categories=[Cooking]" "data.test_categories=[Music]"` (names from takes.json) |
| Fig. 4-b temporal window | `--set data.horizon=1` (…`4`) |
| Tab. 1 latency | `scripts/benchmark_latency.py --device cuda` |
| Tab. 6 CC / AUC | `cc`, `auc` in `metrics_test.json` |
| Tab. 5 2D anticipation | `scripts/eval_2d.py` reports angular error; pixel F1 needs `project2d.project_to_image` with Aria calibration |

The paper's FoVS-Aria results (test split, IoU), for reference:

| | Orientation | Peripheral | Central | Foveal |
|---|---|---|---|---|
| Ours (full) | 0.5838 | 0.4886 | 0.3513 | 0.2836 |
| EgoChoir | 0.4959 | 0.4302 | 0.2612 | 0.1987 |
| CSTS + EgoSpanLift + GP | – | 0.4567 | 0.2342 | 0.1388 |
| Global prior | 0.1359 | 0.1048 | 0.0331 | 0.0146 |

Our rebuilt dataset will not match the authors' curation exactly (their manual exclusions are not published), so treat
these numbers as a scale, not a target. The lab's rebuild has 26.6k samples; the paper reports 23.2k.

## Choices the paper does not pin down

Every one of these is a config option. The default is the reading that seems most faithful to the text.

| Topic | Default | Reasoning / alternative |
|---|---|---|
| Grid orientation | `grid_frame: world` (gravity-aligned, centred on the user) | Eq. 3 voxelises `p − t`, which keeps world axes, and "excluding upside-down" augmentation only makes sense with a fixed vertical axis. `local` uses the anchor CPF axes, as the lab rebuild does |
| Common anchor | user pose at the last observed step | Input and target must share a frame |
| Outlier filter | statistical outlier removal per step (k=20, 2σ), no `dist_std` filter | Sec. 3.2 explicitly prefers neighbour-based filtering over confidence-based filtering. The lab used `dist_std<0.1` plus one global SOR pass |
| Time steps T_p | 1 s units, so T_p = 2 on Aria | From Appendix A.1 ("group into 1-second units") |
| Global embedding | encoder applied to the union of the T_p grids | "an embedding that encodes all visual spans within the duration". `head: learned` gives a learned token (lab) |
| Decoder entry | head embedding broadcast and added to the deepest skip (`bottom: add`) | `bottom: conv` (concatenate + DoubleConv) adds about 85M parameters. The paper-spec model has 116M parameters with `add` |
| Skips | encoder features of the union grid (`skip: head`) | `concat` (all steps, 1×1 conv) or `last` |
| Transformer | 4 layers, 8 heads, FFN 4C, pre-norm, GELU | Not given |
| Dice | `1 − 2ΣŶY / (ΣŶ + ΣY + 1)` per sample and level | Eq. 4 as written. The lab version adds +1 to the numerator |
| Binarisation | 0.5 (`eval.threshold`), or `tune` per level on val | Not given |
| Sample filter | drop samples whose orientation-span point count in the forecast window is below mean − 2σ | "keypoints of interest" is not defined |
| Foveal distance | mean over samples of min/mean/max distance from predicted voxels to the nearest GT voxel | Our reading of Tab. 3 |
| GP pose forecast | numpy exact GP (RBF on a linear trend), independent outputs | The paper used a GPyTorch multi-task GP |
| FoVS-EgoExo past window | 4 s | Only the 4 s forecast is stated |

## Differences from the SensorySpotlight reimplementation

`configs/fovs_aria_sensoryspotlight.yaml` approximates the lab's settings: local frame, `dist_std` filter, C = 256,
learned head token, concatenated skips, AdamW 1e-3 one-cycle, batch 64, 30 epochs and thresholds tuned on val.
`tests/test_reference_equivalence.py` shows that, with the lab's lift settings, this builder produces exactly the
lab's `X`/`Y` grids. Three differences remain:
- Their blocks use Conv-GroupNorm-GELU with a 2×2×2 transposed conv. Here the pytorch-3dunet equivalents are used.
- Their SOR ran once on the whole cloud. Here it runs per step.
- Their `.npz` format differs. Rebuild with `scripts/build_dataset.py` rather than reusing their files.

## Not included

- 3D localisation and affordance baselines: OccFormer, VoxFormer, IAG, EgoChoir. Use their official repositories.
- 2D gaze models: GLC, CSTS. Export their predictions as `yaw_pitch` (n_test, steps, 2) and use `baselines.py`.
- The SLAM corruption study (Tab. 7). Corrupt the `RecordingStreams` before `extract_samples`, then evaluate with
  `evaluate.py --data-root`.
- Ego-Exo4D file layouts change between releases. `read_egoexo` searches for the usual MPS files; check it on one
  take before building the whole set.
