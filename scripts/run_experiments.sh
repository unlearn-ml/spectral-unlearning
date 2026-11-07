#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo "Usage: bash scripts/run_experiments.sh MODEL SPLIT METHOD"
    echo "       bash scripts/run_experiments.sh --list"
    exit 0
fi
if [[ $# -eq 1 && "$1" == --list ]]; then
    python scripts/get_learning_rate.py --list
    exit 0
fi
if (( $# != 3 )); then
    echo "Usage: bash scripts/run_experiments.sh MODEL SPLIT METHOD (or --list)" >&2
    exit 2
fi
MODEL=$1
SPLIT=$2
METHOD=$3
LR=$(python scripts/get_learning_rate.py "$MODEL" "$SPLIT" "$METHOD")
export SEED=${SEED:-0}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

python - "$MODEL" "$SPLIT" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

model, split, seed = sys.argv[1:]
if not seed.isascii() or not seed.isdecimal() or int(seed) > 2**32 - 1:
    raise SystemExit("SEED must be an integer between 0 and 4294967295")
retain = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}[split]
for path in (
    Path(f"saves/eval/tofu_{model}_{retain}/TOFU_EVAL.json"),
    Path(f"saves/eval/tofu_{model}_full/evals_{split}/TOFU_EVAL.json"),
):
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run: python setup_data.py --eval_logs")
    with path.open() as file:
        if not isinstance(json.load(file), dict):
            raise SystemExit(f"Invalid evaluation log: {path}")

import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Select one available GPU with CUDA_VISIBLE_DEVICES")
PY

RUN_ROOT=${RUN_ROOT:-"saves/experiments/${MODEL}_${SPLIT}_${METHOD}_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p -- "$(dirname -- "$RUN_ROOT")"
if ! mkdir -- "$RUN_ROOT"; then
    echo "Choose a new RUN_ROOT; existing results are not overwritten." >&2
    exit 1
fi
RUNS_DIR="$RUN_ROOT/runs"
mkdir -p -- "$RUNS_DIR" "$RUN_ROOT/logs"
printf "model|split|trainer|lr\n%s|%s|%s|%s\n" "$MODEL" "$SPLIT" "$METHOD" "$LR" > "$RUN_ROOT/plan.txt"
printf "seed=%s\nCUDA_VISIBLE_DEVICES=%s\n" "$SEED" "$CUDA_VISIBLE_DEVICES" > "$RUN_ROOT/settings.txt"

TASK="tofu_${MODEL}_${SPLIT}_${METHOD}_lr${LR}_s${SEED}"
CKPT="$RUNS_DIR/$TASK"
TRAIN_LOG="$RUN_ROOT/logs/train.log"
EVAL_LOG="$RUN_ROOT/logs/eval.log"
echo "Train: $MODEL $SPLIT $METHOD, lr=$LR, seed=$SEED"
if ! RUNS_DIR="$RUNS_DIR" bash scripts/quickstart_train.sh "$METHOD" "$MODEL" "$SPLIT" "$LR" > "$TRAIN_LOG" 2>&1; then
    tail -n 40 "$TRAIN_LOG" >&2
    exit 1
fi
echo "Evaluate: $CKPT"
if ! bash scripts/quickstart_eval.sh "$CKPT" "$MODEL" "$SPLIT" > "$EVAL_LOG" 2>&1; then
    tail -n 40 "$EVAL_LOG" >&2
    exit 1
fi
test -s "$CKPT/evals_${SPLIT}/TOFU_EVAL.json"

python scripts/aggregate_results.py \
    --saves-dir "$RUNS_DIR" --seeds "$SEED" --splits "$SPLIT" \
    --compact --output "$RUN_ROOT/results.tex"
echo "Results: $RUN_ROOT/results.tsv"
