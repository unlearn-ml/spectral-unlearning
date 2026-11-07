#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

METHOD=${1:-SPNPO_bf16_fast}
MODEL=${2:-Llama-3.2-3B-Instruct}
FORGET_SPLIT=${3:-forget10}
LR=${4:-$(python scripts/get_learning_rate.py "$MODEL" "$FORGET_SPLIT" "$METHOD")}
SEED=${SEED:-0}

case "$FORGET_SPLIT" in
    forget01) RETAIN_SPLIT=retain99 ; HOLDOUT_SPLIT=holdout01 ;;
    forget05) RETAIN_SPLIT=retain95 ; HOLDOUT_SPLIT=holdout05 ;;
    forget10) RETAIN_SPLIT=retain90 ; HOLDOUT_SPLIT=holdout10 ;;
    *) echo "Unknown FORGET_SPLIT: $FORGET_SPLIT (use forget01/forget05/forget10)" ; exit 1 ;;
esac

case "$MODEL" in
    Llama-3.2-1B-Instruct) DEFAULT_BSZ=8; DEFAULT_CHECKPOINTING=false ;;
    Llama-3.2-3B-Instruct) DEFAULT_BSZ=4; DEFAULT_CHECKPOINTING=true ;;
    *) echo "Unsupported MODEL: $MODEL" ; exit 1 ;;
esac

HF_BASE="open-unlearning/tofu_${MODEL}_full"
PER_DEVICE_BSZ=${PER_DEVICE_BSZ:-$DEFAULT_BSZ}
GRAD_ACCUM=${GRAD_ACCUM:-4}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-$DEFAULT_CHECKPOINTING}
NUM_EPOCHS=${NUM_EPOCHS:-10}

TASK="tofu_${MODEL}_${FORGET_SPLIT}_${METHOD}_lr${LR}_s${SEED}"
OUT_DIR="${RUNS_DIR:-saves/unlearn}/${TASK}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml \
    trainer="${METHOD}" \
    model="${MODEL}" \
    forget_split="${FORGET_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    model.model_args.pretrained_model_name_or_path="${HF_BASE}" \
    model.tokenizer_args.pretrained_model_name_or_path="${HF_BASE}" \
    paths.output_dir="${OUT_DIR}" \
    task_name="${TASK}" \
    trainer.args.per_device_train_batch_size="${PER_DEVICE_BSZ}" \
    trainer.args.gradient_accumulation_steps="${GRAD_ACCUM}" \
    trainer.args.learning_rate="${LR}" \
    trainer.args.num_train_epochs="${NUM_EPOCHS}" \
    trainer.args.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    trainer.args.save_strategy=no \
    trainer.args.eval_on_start=False \
    trainer.args.eval_strategy=no \
    trainer.args.do_eval=False \
    trainer.args.seed="${SEED}"
