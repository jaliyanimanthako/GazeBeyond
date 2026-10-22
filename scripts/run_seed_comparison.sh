#!/usr/bin/env bash
# Full cube vs front half, 3 seeds each, then the comparison table (IoU / F1 ± std, FLOPs, inference latency).
#
#   bash scripts/run_seed_comparison.sh 0 1      # full-cube runs on GPU 0, front-half runs on GPU 1, in parallel
#   bash scripts/run_seed_comparison.sh 0        # everything on GPU 0, one after another
#
# Runs that already finished (they have probs_test.npy) are skipped, so seed 0 of each is reused and the script can
# be restarted. Training logs: runs/fovs_aria_lab/logs/. Results: results/full_vs_front.{md,json}.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU_FULL=${1:-0}
GPU_FRONT=${2:-$GPU_FULL}
CFG=configs/fovs_aria_lab.yaml
RUNS=runs/fovs_aria_lab
mkdir -p "$RUNS/logs" results

train() {                                  # train <out_dir> <gpu> [overrides...]
  local out=$1 gpu=$2
  shift 2
  if [ -f "$out/probs_test.npy" ]; then echo "skip $out (already trained)"; return; fi
  echo "train $out on GPU $gpu  (log: $RUNS/logs/$(basename "$out").log)"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/train.py --config "$CFG" \
    --set train.out_dir="$out" train.progress=false "$@" > "$RUNS/logs/$(basename "$out").log" 2>&1
  echo "done  $out"
}
run_full()  { for s in 0 1 2; do train "$RUNS/egospanlift_s$s" "$GPU_FULL" train.seed=$s; done; }
run_front() { for s in 0 1 2; do train "$RUNS/front_s$s" "$GPU_FRONT" train.seed=$s data.crop_front=true; done; }

if [ "$GPU_FULL" != "$GPU_FRONT" ]; then
  run_full & run_front & wait
else
  run_full
  run_front
fi

# measured after all training has finished, so the latency is not disturbed by the training jobs
CUDA_VISIBLE_DEVICES=$GPU_FULL python scripts/compare_variants.py \
  --group full  "$RUNS"/egospanlift_s0 "$RUNS"/egospanlift_s1 "$RUNS"/egospanlift_s2 \
  --group front "$RUNS"/front_s0 "$RUNS"/front_s1 "$RUNS"/front_s2 \
  --device cuda --out results/full_vs_front
