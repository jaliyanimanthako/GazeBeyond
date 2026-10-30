#!/usr/bin/env bash
# EgoSpanLift WHERE rows for the lab's table, in one go: for each dataset its folds, questions and benchmark, the camera
# tokens, as released (the AEA model on every fold), retrained (fine-tuned per fold + V-JEPA), then the lab's WHERE
# table with AEA's rows (from the AEA run) and the new ones. Every stage whose output already exists is skipped (a fold
# counts as trained once it has metrics_test.json), so a stopped run just starts again; delete an output to redo it.
# ADT is left out until its SLAM points are downloaded (not in the lab's ADT download).
#   nohup ./scripts/run_where.sh > ~/run_where.log 2>&1 &        then: tail -f ~/run_where.log
# DATASETS (default "hdepic adt"), G0 / G1: the two GPUs, NW: data loader workers for the fine-tunes (config's if unset).
set -euo pipefail
source ~/gb_env.sh
cd ~/GazeBeyond
L=${L:-/data/SS/SS/Dynamic_Fusion_ALFA/data/traj_forecast/attention_shift}
FC=${FC:-$L/forecast}
SS=~/SensorySpotlight/data/traj_forecast/attention_shift/senses_suite
G0=${G0:-0}; G1=${G1:-1}
CFG=configs/fovs_aria_shift.yaml
AEA0=runs/fovs_aria_lab/egospanlift_s0                     # the AEA model of the default split (test location 4)
HDEPIC_SLAM=/data/disk2/hdepic_dinithi/HD-EPIC/SLAM-and-Gaze
HDEPIC_RAW=/data/disk2/hdepic/HD-EPIC/raw
ADT_SRC=/data/disk2/adt
mkdir -p data/lab_q results logs

DS=""
for ds in ${DATASETS:-hdepic adt}; do
  if [ $ds = adt ] && ! ls $ADT_SRC/*/*mps_slam_points* > /dev/null 2>&1; then
    echo "-- adt left out: no SLAM points in $ADT_SRC yet"
  else
    DS+="$ds "
  fi
done

bench() { echo data/$1_labq_where; }
folds() { $PY -c "import json,sys; print(*sorted(set(json.load(open(sys.argv[1])).values())))" data/lab_q/$1_where_folds.json; }
final() { case $1 in aea) s=location; v=causalfix;; adt) s=session; v=causalfix;; hdepic) s=location; v=cachefix;; esac
          p=probs_$1_lag_${s}_residual_pc_sph; echo ${p}_${v}_sphgeo.npz,${p}_seeds10_${v}_sphgeo.npz:mem+video; }
lanes() {                                                     # lanes <function> <job> ...: two GPUs, jobs alternating
  local fn=$1 i=0 a=() b=(); shift
  for j in "$@"; do if (( i++ % 2 )); then b+=("$j"); else a+=("$j"); fi; done
  ( for j in ${a[@]+"${a[@]}"}; do $fn $G0 $j; done ) & local p0=$!
  ( for j in ${b[@]+"${b[@]}"}; do $fn $G1 $j; done ) & local p1=$!
  wait $p0 && wait $p1 || { echo "a $fn job failed: see logs/"; exit 1; }
}
ftw() { local ds=${2%:*} f=${2#*:} out=runs/${2%:*}_labq/ft_vid_f${2#*:}   # the AEA model fine-tuned on one fold + V-JEPA
  [ -f $out/metrics_test.json ] && { echo "   $ds fold $f: trained"; return; }
  echo "   $ds fold $f: training on GPU $1 ($(date +%H:%M))"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/train.py --config $CFG --set data.root=$(bench $ds) data.fold=$f \
      data.video_features=true ${NW:+data.num_workers=$NW} train.init=$AEA0 train.progress=false train.out_dir=$out \
      > logs/ft_vid_${ds}_f$f.log 2>&1
  echo "   $ds fold $f: done ($(date +%H:%M))"
}

for ds in $DS; do
  echo "== $ds: folds, questions, benchmark, camera tokens"
  q=data/lab_q/${ds}_where
  [ -f ${q}_folds.json ] || (cd ~/SensorySpotlight && $PY tools/export_gbf_folds.py $ds where --forecast $FC \
                                 --out ~/GazeBeyond/${q}_folds.json)
  [ -f $q.npz ] || $PY scripts/lab_questions.py where --suite $SS --tag ${ds}_lag --labels $L/labels --out $q.npz
  if [ ! -f $(bench $ds)/all.npz ]; then
    # HD-EPIC recordings are long (up to ~50 min of SLAM observations each): few at a time
    case $ds in adt) src="--workers 8 --src $ADT_SRC";;
                hdepic) src="--workers ${HDEPIC_WORKERS:-2} --slam-root $HDEPIC_SLAM --src $HDEPIC_RAW";; esac
    $PY scripts/build_dataset.py $ds --config $CFG $src --set build.window.anchor=lab build.window.questions=$q.npz \
        --folds ${q}_folds.json --out $(bench $ds) > logs/build_${ds}_where.log 2>&1
    echo "$ds: $(grep -c FAILED logs/build_${ds}_where.log || true) recordings failed (logs/build_${ds}_where.log)"
    $PY scripts/build_dataset.py pack --config $CFG --out $(bench $ds)
  fi
  if [ ! -f $(bench $ds)/video_all.npz ]; then
    v=$SS/dense_${ds}_lag_vjepa.npy; [ -f $v ] || v=$SS/dense_${ds}_lag_vjepalab.npy   # the same V-JEPA cache, renamed
    $PY scripts/attach_shift_video.py --bench $(bench $ds) --suite $SS --tag ${ds}_lag --dense $v
  fi

  echo "== $ds: as released"
  if [ ! -f $SS/probs_gbf_astrained_$ds.npz ]; then
    CUDA_VISIBLE_DEVICES=$G0 $PY scripts/export_lab_probs.py where --bench $(bench $ds) --suite $SS --tag ${ds}_lag \
        --name gbf_astrained_$ds --runs $AEA0 --eval-folds $(folds $ds) --compare $(final $ds) \
        | tee results/gbf_where_astrained_$ds.txt
  else
    echo "   done: results/gbf_where_astrained_$ds.txt"
  fi
done

echo "== retrained: $(for ds in $DS; do echo -n "$ds $(folds $ds | wc -w) folds  "; done)(50 epochs each, ~3 h a fold)"
lanes ftw $(for ds in $DS; do for f in $(folds $ds); do echo $ds:$f; done; done)
for ds in $DS; do
  [ -f $SS/probs_gbf_retrained_$ds.npz ] && { echo "   $ds retrained: done (results/gbf_where_retrained_$ds.txt)"; continue; }
  CUDA_VISIBLE_DEVICES=$G0 $PY scripts/export_lab_probs.py where --bench $(bench $ds) --suite $SS --tag ${ds}_lag \
      --name gbf_retrained_$ds --runs $(for f in $(folds $ds); do echo runs/${ds}_labq/ft_vid_f$f; done) \
      --compare $(final $ds) | tee results/gbf_where_retrained_$ds.txt
done

echo "== the lab's WHERE table"
row() { local ds=$1 n=$2 f=$3
  X+="$ds|$n|sph|$f|gbf;$ds|$n, >= 30 deg|sph|$f|gbf30;"; }
X=""; W="aea"
for w in astrained:"as released" retrained:"retrained (+ V-JEPA)"; do      # AEA's rows from the AEA run (run before this)
  f=""; for c in probs_gbf_${w%%:*}_aea.npz probs_gbf_${w%%:*}.npz; do [ -f $SS/$c ] && { f=$c; break; }; done
  [ -n "$f" ] && row aea "EgoSpanLift, ${w#*:}" $f || echo "-- aea ${w%%:*}: no probs_gbf_${w%%:*}[_aea].npz in $SS"
done
X+="aea|Ours: final, 6 seeds|sph|$(final aea | sed 's/:/|/');"
for ds in $DS; do
  W+=",$ds"
  for w in astrained:"as released" retrained:"retrained (+ V-JEPA)"; do row $ds "EgoSpanLift, ${w#*:}" probs_gbf_${w%%:*}_$ds.npz; done
  X+="$ds|Ours: final, 6 seeds|sph|$(final $ds | sed 's/:/|/');"
done
(cd ~/SensorySpotlight && WM_DATASETS=$W WM_EXTRA="${X%;}" WM_OUT=$HOME/GazeBeyond/results/where_metrics_gbf.md \
    $PY tools/attention_where_metrics.py)
echo "WHERE DONE: ~/GazeBeyond/results/where_metrics_gbf.md"
