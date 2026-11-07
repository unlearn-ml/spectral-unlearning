#!/usr/bin/env bash

set -euo pipefail

CKPT_DIR=${1:?"checkpoint directory required"}
MODEL=${2:-Llama-3.2-3B-Instruct}
FORGET_SPLIT=${3:-forget10}

case "$FORGET_SPLIT" in
    forget01) RETAIN_SPLIT=retain99 ; HOLDOUT_SPLIT=holdout01 ;;
    forget05) RETAIN_SPLIT=retain95 ; HOLDOUT_SPLIT=holdout05 ;;
    forget10) RETAIN_SPLIT=retain90 ; HOLDOUT_SPLIT=holdout10 ;;
    *) echo "Unknown FORGET_SPLIT: $FORGET_SPLIT" ; exit 1 ;;
esac

HF_TOK="open-unlearning/tofu_${MODEL}_full"
TASK_NAME="$(basename "${CKPT_DIR}")_eval_${FORGET_SPLIT}"
OUT_DIR="${CKPT_DIR}/evals_${FORGET_SPLIT}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python src/eval.py \
    experiment=eval/tofu/default.yaml \
    model="${MODEL}" \
    forget_split="${FORGET_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    model.model_args.pretrained_model_name_or_path="${CKPT_DIR}" \
    model.tokenizer_args.pretrained_model_name_or_path="${HF_TOK}" \
    paths.output_dir="${OUT_DIR}" \
    task_name="${TASK_NAME}" \
    retain_logs_path="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json" \
    init_logs_path="saves/eval/tofu_${MODEL}_full/evals_${FORGET_SPLIT}/TOFU_EVAL.json" \
    eval.tofu.overwrite=true
