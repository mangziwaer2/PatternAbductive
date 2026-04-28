#!/usr/bin/env bash
set -euo pipefail

MODELNAME="${MODELNAME:-GPT2_6_act_nt}"
DATA_ROOT="${DATA_ROOT:-./sampled_data_abduction_traced/}"
BATCH_SIZE="${BATCH_SIZE:-12}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
STAGE1_EPOCH="${STAGE1_EPOCH:-25}"
STAGE2_EPOCH="${STAGE2_EPOCH:-25}"

python training.py \
  --batch_size "${BATCH_SIZE}" \
  --data_root "${DATA_ROOT}" \
  --modelname "${MODELNAME}" \
  --train_stage logic \
  --accelerate \
  --mixed_precision "${MIXED_PRECISION}" \
  --experiment_name stage1-logic \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_pin_memory true \
  --dataloader_persistent_workers true \
  --train_log_every 2000

python training.py \
  --batch_size "${BATCH_SIZE}" \
  --data_root "${DATA_ROOT}" \
  --modelname "${MODELNAME}" \
  --train_stage stage2_loop \
  --resume_epoch "${STAGE1_EPOCH}" \
  --accelerate \
  --mixed_precision "${MIXED_PRECISION}" \
  --experiment_name stage2-action-loop \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_pin_memory true \
  --dataloader_persistent_workers true \
  --train_log_every 2000

python scripts/run_stage3_rollout_train.py \
  --data_root "${DATA_ROOT}" \
  --split train \
  --modelname "${MODELNAME}" \
  --scale default \
  --resume_epoch "${STAGE2_EPOCH}" \
  --max-steps 100 \
  --print-every 10
