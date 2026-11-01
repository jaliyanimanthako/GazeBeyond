#!/usr/bin/env bash
# EgoSpanLift WHEN rows for the lab's table (AUROC and lead-time AUROC, next to the lab's final WHEN model on the same
# questions), in one go: for each dataset the lab's WHEN folds and questions, the benchmark, as released (the AEA model
# on every fold; AEA's own from run_lab_more.sh step 0), retrained (each fold's model fine-tuned on that fold's WHEN
# questions; no video: the lab encoded camera tokens for the WHERE questions only), then a summary. Every stage whose
# output already exists is skipped (a fold counts as trained once it has metrics_test.json), as run_where.sh.
# The lab's forecast files (<ds>.npz, when_v4_<ds>.npz) are copied once to data/lab_q/fc/ and only the copies are read,
# so a rebuild of the lab's files mid-run cannot mix two question sets; delete data/lab_q/fc/*<ds>.npz to take the
# lab's current ones (and the outputs built from them).
#   ./scripts/run_when.sh 2>&1 | tee -a ~/run_when.log          (in tmux)
# DATASETS (default "aea adt hdepic"), G0 / G1: the two GPUs (G0=G1: two folds at a time on one GPU), LANES=1: one fold
# at a time, NW: data loader workers, UPDATES: optimiser steps per fine-tune (default 14000, as the WHERE fine-tunes:
# epochs = UPDATES x batch 16 / the fold's training samples, at least 1).
set -euo pipefail
source ~/gb_env.sh
cd ~/GazeBeyond
L=${L:-/data/SS/SS/Dynamic_Fusion_ALFA/data/traj_forecast/attention_shift}
FC=${FC:-$L/forecast}
FCL=$HOME/GazeBeyond/data/lab_q/fc                         # our copies of the lab's forecast files
G0=${G0:-0}; G1=${G1:-1}
CFG=configs/fovs_aria_shift.yaml
AEA0=runs/fovs_aria_lab/egospanlift_s0                     # the AEA model of the default split (test location 4)
HDEPIC_SLAM=/data/disk2/hdepic_dinithi/HD-EPIC/SLAM-and-Gaze
HDEPIC_RAW=/data/disk2/hdepic/HD-EPIC/raw
ADT_SRC=/data/disk2/adt
DS=${DATASETS:-aea adt hdepic}
mkdir -p data/lab_q $FCL results logs

bench() { [ $1 = aea ] && echo data/fovs_aria_labq_when || echo data/$1_labq_when; }
folds() { [ $1 = aea ] && echo 1 2 3 4 5 ||
          $PY -c "import json,sys; print(*sorted(set(json.load(open(sys.argv[1])).values())))" data/lab_q/$1_when_folds.json; }
epochs() { $PY -c "
import sys; sys.path.insert(0, '.')
from egospan.data.dataset import PackedBenchmark
n = len(PackedBenchmark(sys.argv[1]).indices('train', int(sys.argv[2])))
print(max(1, round(int(sys.argv[3]) * 16 / n)))" $(bench $1) $2 ${UPDATES:-14000}; }
lanes() {                                                     # lanes <function> <job> ...: two GPUs, jobs alternating
  local fn=$1 i=0 a=() b=(); shift
  for j in "$@"; do if (( i++ % ${LANES:-2} )); then b+=("$j"); else a+=("$j"); fi; done
  ( for j in ${a[@]+"${a[@]}"}; do $fn $G0 $j; done ) & local p0=$!
  ( for j in ${b[@]+"${b[@]}"}; do $fn $G1 $j; done ) & local p1=$!
  wait $p0 && wait $p1 || { echo "a $fn job failed: see logs/"; exit 1; }
}
ftn() { local ds=${2%:*} f=${2#*:} init=$AEA0 out=runs/${2%:*}_labq_when/ft_f${2#*:}   # one fold's WHEN fine-tune
  [ -f $out/metrics_test.json ] && { echo "   $ds fold $f: trained"; return; }
  [ $ds = aea ] && [ $f != 4 ] && init=runs/fovs_aria_lab/cv/egospanlift_f$f   # AEA: the same fold's model
  local ep; ep=$(epochs $ds $f)
  echo "   $ds fold $f: training on GPU $1, $ep epochs ($(date +%H:%M))"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/train.py --config $CFG --set data.root=$(bench $ds) data.fold=$f \
      ${NW:+data.num_workers=$NW} train.init=$init train.epochs=$ep train.progress=false train.out_dir=$out \
      > logs/ft_when_${ds}_f$f.log 2>&1
  echo "   $ds fold $f: done ($(date +%H:%M))"
}

for ds in $DS; do
  echo "== $ds: the lab's forecast files, folds, questions, benchmark"
  for f in $ds.npz when_v4_$ds.npz; do
    [ -f $FCL/$f ] || { cp $FC/$f $FCL/$f; echo "   copied $FC/$f ($(date -r $FC/$f +%F' '%T))"; }
  done
  if [ $ds = aea ]; then
    [ -f $(bench aea)/all.npz ] || { echo "no $(bench aea): build it first (scripts/run_gbf_lab.sh)"; exit 1; }
    continue
  fi
  q=data/lab_q/${ds}_when
  [ -f ${q}_folds.json ] || (cd ~/SensorySpotlight && $PY tools/export_gbf_folds.py $ds when --forecast $FCL \
                                 --out ${q}_folds.json)
  [ -f $q.npz ] || $PY scripts/lab_questions.py when --forecast $FCL/$ds.npz --out $q.npz
  if [ ! -f $(bench $ds)/all.npz ]; then
    # HD-EPIC recordings are long (up to ~50 min of SLAM observations each): few at a time
    case $ds in adt) src="--workers 8 --src $ADT_SRC";;
                hdepic) src="--workers ${HDEPIC_WORKERS:-2} --slam-root $HDEPIC_SLAM --src $HDEPIC_RAW";; esac
    echo "   building $(bench $ds) ($(date +%H:%M))"
    $PY scripts/build_dataset.py $ds --config $CFG $src --set build.window.anchor=lab build.window.questions=$q.npz \
        --folds ${q}_folds.json --out $(bench $ds) > logs/build_${ds}_when.log 2>&1
    echo "   $ds: $(grep -c FAILED logs/build_${ds}_when.log || true) recordings failed (logs/build_${ds}_when.log)"
    $PY scripts/build_dataset.py pack --config $CFG --out $(bench $ds)
  fi

  echo "== $ds: as released"
  if [ ! -f results/gbf_when_astrained_$ds.npz ]; then
    CUDA_VISIBLE_DEVICES=$G0 $PY scripts/export_lab_probs.py when --bench $(bench $ds) --runs $AEA0 \
        --eval-folds $(folds $ds) --out results/gbf_when_astrained_$ds.npz --forecast $FCL/$ds.npz \
        --ours $FCL/when_v4_$ds.npz:gru_ens | tee results/gbf_when_astrained_$ds.txt
  else
    echo "   done: results/gbf_when_astrained_$ds.txt"
  fi
done

echo "== retrained: $(for ds in $DS; do echo -n "$ds $(folds $ds | wc -w) folds  "; done)(${UPDATES:-14000} updates each)"
lanes ftn $(for ds in $DS; do for f in $(folds $ds); do echo $ds:$f; done; done)
for ds in $DS; do
  [ -f results/gbf_when_retrained_$ds.npz ] && { echo "   $ds retrained: done (results/gbf_when_retrained_$ds.txt)"; continue; }
  CUDA_VISIBLE_DEVICES=$G0 $PY scripts/export_lab_probs.py when --bench $(bench $ds) \
      --runs $(for f in $(folds $ds); do echo runs/${ds}_labq_when/ft_f$f; done) \
      --out results/gbf_when_retrained_$ds.npz --forecast $FCL/$ds.npz --ours $FCL/when_v4_$ds.npz:gru_ens \
      | tee results/gbf_when_retrained_$ds.txt
done

echo "== WHEN summary"
for ds in $DS; do
  for w in astrained retrained; do
    f=results/gbf_when_${w}_$ds.txt; [ $ds = aea ] && [ $w = astrained ] && f=results/gbf_when_astrained_aea_lead.txt
    echo "--- $ds, $w ($f)"; [ -f $f ] && cat $f || echo "   (missing)"
  done
done
echo "WHEN DONE"
