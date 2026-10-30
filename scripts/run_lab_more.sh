#!/usr/bin/env bash
# EgoSpanLift on the lab's (SensorySpotlight) questions: everything missing after the AEA run (run_gbf_lab.sh).
#   0  AEA WHEN, as released: lead-time AUC and the final WHEN model on the same questions (rescoring, no model run)
#   1  ADT and HD-EPIC: the lab's folds, questions and benchmarks (WHERE and WHEN), camera tokens for WHERE
#   2  ADT and HD-EPIC WHERE: as released (the AEA model on every fold) and retrained (fine-tuned per fold + V-JEPA)
#   3  WHEN retrained, all three datasets: each fold's model fine-tuned on that fold's WHEN questions (no video: the
#      lab encoded camera tokens only for the WHERE questions)
#   4  WHEN export: as released (ADT, HD-EPIC) and retrained (all three), AUC and lead-time AUC next to the final model
#   5  the lab's WHERE table with the new rows (median, within 30 deg, in / out of view, ...)
# Needs ADT's SLAM points first (not in the lab's ADT download), e.g.
#   cd ~/SensorySpotlight && python3 tools/adt_download.py --urls <ADT_download_urls.json> --out /data/disk2/adt \
#       --groups mps_slam_points
# STEP=k starts at step k (earlier steps' outputs are reused), LAST=k stops after step k; DATASETS (default "adt hdepic"):
# the new datasets, e.g. DATASETS=hdepic while ADT's points download; G0 / G1: the two GPUs; WHEN_EPOCHS (default 10).
set -euo pipefail
source ~/gb_env.sh
cd ~/GazeBeyond
L=${L:-/data/SS/SS/Dynamic_Fusion_ALFA/data/traj_forecast/attention_shift}
FC=${FC:-$L/forecast}
SS=~/SensorySpotlight/data/traj_forecast/attention_shift/senses_suite
G0=${G0:-0}; G1=${G1:-1}; STEP=${STEP:-0}; LAST=${LAST:-5}; DS=${DATASETS:-adt hdepic}
CFG=configs/fovs_aria_shift.yaml
AEA0=runs/fovs_aria_lab/egospanlift_s0                     # the AEA model of the default split (test location 4)
HDEPIC_SLAM=/data/disk2/hdepic_dinithi/HD-EPIC/SLAM-and-Gaze
HDEPIC_RAW=/data/disk2/hdepic/HD-EPIC/raw
mkdir -p data/lab_q results logs

for f in $(for ds in aea $DS; do echo $FC/$ds.npz $FC/when_v4_$ds.npz; done); do
  [ -f "$f" ] || { echo "missing $f: set FC=<the lab's forecast folder>"; exit 1; }
done

bench() { [ $1 = aea ] && echo data/fovs_aria_labq_$2 || echo data/$1_labq_$2; }
folds() { [ $1 = aea ] && echo 1 2 3 4 5 ||
          $PY -c "import json,sys; print(*sorted(set(json.load(open(sys.argv[1])).values())))" data/lab_q/$1_$2_folds.json; }
final() { case $1 in aea) s=location; v=causalfix;; adt) s=session; v=causalfix;; hdepic) s=location; v=cachefix;; esac
          p=probs_$1_lag_${s}_residual_pc_sph; echo ${p}_${v}_sphgeo.npz,${p}_seeds10_${v}_sphgeo.npz:mem+video; }
lanes() {                                                     # lanes <function> <job> ...: two GPUs, jobs alternating
  local fn=$1 i=0 a=() b=(); shift
  for j in "$@"; do if (( i++ % 2 )); then b+=("$j"); else a+=("$j"); fi; done
  ( for j in ${a[@]+"${a[@]}"}; do $fn $G0 $j; done ) & local p0=$!
  ( for j in ${b[@]+"${b[@]}"}; do $fn $G1 $j; done ) & local p1=$!
  wait $p0 && wait $p1 || { echo "a $fn job failed: see logs/"; exit 1; }
}

if (( STEP <= 0 && LAST >= 0 )); then
  echo "== 0. AEA WHEN, as released: lead time + final model"
  $PY scripts/export_lab_probs.py when --bench $(bench aea when) --out results/gbf_when_astrained_aea.npz --reuse \
      --forecast $FC/aea.npz --ours $FC/when_v4_aea.npz:gru_ens | tee results/gbf_when_astrained_aea_lead.txt
fi

if (( STEP <= 1 && LAST >= 1 )); then
  echo "== 1. ADT, HD-EPIC: folds, questions, benchmarks"
  for ds in $DS; do
    for k in where when; do
      (cd ~/SensorySpotlight && $PY tools/export_gbf_folds.py $ds $k --out ~/GazeBeyond/data/lab_q/${ds}_${k}_folds.json)
    done
    $PY scripts/lab_questions.py where --suite $SS --tag ${ds}_lag --labels $L/labels --out data/lab_q/${ds}_where.npz
    $PY scripts/lab_questions.py when --forecast $FC/$ds.npz --out data/lab_q/${ds}_when.npz
  done
  for k in where when; do
    q="--set build.window.anchor=lab build.window.questions=data/lab_q"
    for ds in $DS; do
      # HD-EPIC recordings are long (up to ~50 min of SLAM observations each): few at a time
      case $ds in adt) src="--workers 8 --src /data/disk2/adt";;
                  hdepic) src="--workers ${HDEPIC_WORKERS:-2} --slam-root $HDEPIC_SLAM --src $HDEPIC_RAW";; esac
      $PY scripts/build_dataset.py $ds --config $CFG $src $q/${ds}_$k.npz --folds data/lab_q/${ds}_${k}_folds.json \
          --out $(bench $ds $k) > logs/build_${ds}_$k.log 2>&1
    done
    for ds in $DS; do
      echo "$ds $k: $(grep -c FAILED logs/build_${ds}_$k.log || true) recordings failed (logs/build_${ds}_$k.log)"
      $PY scripts/build_dataset.py pack --config $CFG --out $(bench $ds $k)
    done
  done
  for ds in $DS; do
    $PY scripts/attach_shift_video.py --bench $(bench $ds where) --suite $SS --tag ${ds}_lag
  done
fi

ftw() { local ds=${2%:*} f=${2#*:}                           # WHERE: the AEA model fine-tuned on one fold + V-JEPA
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/train.py --config $CFG --set data.root=$(bench $ds where) data.fold=$f \
      data.video_features=true train.init=$AEA0 train.progress=false train.out_dir=runs/${ds}_labq/ft_vid_f$f \
      > logs/ft_vid_${ds}_f$f.log 2>&1; }
if (( STEP <= 2 && LAST >= 2 )); then
  echo "== 2. ADT, HD-EPIC WHERE"
  export CUDA_VISIBLE_DEVICES=$G0
  for ds in $DS; do
    $PY scripts/export_lab_probs.py where --bench $(bench $ds where) --suite $SS --tag ${ds}_lag \
        --name gbf_astrained_$ds --runs $AEA0 --eval-folds $(folds $ds where) --compare $(final $ds) \
        | tee results/gbf_where_astrained_$ds.txt
  done
  unset CUDA_VISIBLE_DEVICES
  lanes ftw $(for ds in $DS; do for f in $(folds $ds where); do echo $ds:$f; done; done)
  export CUDA_VISIBLE_DEVICES=$G0
  for ds in $DS; do
    $PY scripts/export_lab_probs.py where --bench $(bench $ds where) --suite $SS --tag ${ds}_lag \
        --name gbf_retrained_$ds --runs $(for f in $(folds $ds where); do echo runs/${ds}_labq/ft_vid_f$f; done) \
        --compare $(final $ds) | tee results/gbf_where_retrained_$ds.txt
  done
  unset CUDA_VISIBLE_DEVICES
fi

ftn() { local ds=${2%:*} f=${2#*:} init=$AEA0                # WHEN: fine-tuned on one fold's WHEN questions
  [ $ds = aea ] && [ $f != 4 ] && init=runs/fovs_aria_lab/cv/egospanlift_f$f   # AEA: the same fold's model
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/train.py --config $CFG --set data.root=$(bench $ds when) data.fold=$f \
      train.init=$init train.epochs=${WHEN_EPOCHS:-10} train.progress=false train.out_dir=runs/${ds}_labq_when/ft_f$f \
      > logs/ft_when_${ds}_f$f.log 2>&1; }
if (( STEP <= 3 && LAST >= 3 )); then
  echo "== 3. WHEN retrained"
  lanes ftn $(for ds in aea $DS; do for f in $(folds $ds when); do echo $ds:$f; done; done)
fi

if (( STEP <= 4 && LAST >= 4 )); then
  echo "== 4. WHEN export"
  export CUDA_VISIBLE_DEVICES=$G0
  for ds in $DS; do
    $PY scripts/export_lab_probs.py when --bench $(bench $ds when) --runs $AEA0 --eval-folds $(folds $ds when) \
        --out results/gbf_when_astrained_$ds.npz --forecast $FC/$ds.npz --ours $FC/when_v4_$ds.npz:gru_ens \
        | tee results/gbf_when_astrained_$ds.txt
  done
  for ds in aea $DS; do
    $PY scripts/export_lab_probs.py when --bench $(bench $ds when) \
        --runs $(for f in $(folds $ds when); do echo runs/${ds}_labq_when/ft_f$f; done) \
        --out results/gbf_when_retrained_$ds.npz --forecast $FC/$ds.npz --ours $FC/when_v4_$ds.npz:gru_ens \
        | tee results/gbf_when_retrained_$ds.txt
  done
  unset CUDA_VISIBLE_DEVICES
fi

if (( STEP <= 5 && LAST >= 5 )); then
  echo "== 5. the lab's WHERE table with the new rows"
  X=""
  for ds in $DS; do
    for w in astrained:"as released" retrained:"retrained (+ V-JEPA)"; do
      f=probs_gbf_${w%%:*}_$ds.npz; n="EgoSpanLift, ${w#*:}"
      X+="$ds|$n|sph|$f|gbf;$ds|$n, >= 30 deg|sph|$f|gbf30;"
    done
    X+="$ds|Ours: final, 6 seeds|sph|$(final $ds | sed 's/:/|/');"
  done
  (cd ~/SensorySpotlight && WM_DATASETS=${DS// /,} WM_EXTRA="${X%;}" WM_OUT=$HOME/GazeBeyond/results/where_metrics_gbf_more.md \
      $PY tools/attention_where_metrics.py)
  echo "WHERE table: ~/GazeBeyond/results/where_metrics_gbf_more.md"
fi
echo "LAB MORE DONE"
