#!/usr/bin/env bash
set -euo pipefail

MODELNAME="${MODELNAME:-GPT2_6_act_nt}"
DATA_ROOT="${DATA_ROOT:-./sampled_data_abduction/}"
BATCH_SIZE="${BATCH_SIZE:-12}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
STAGE1_NEPOCH="${STAGE1_NEPOCH:-2}"
STAGE2_NEPOCH="${STAGE2_NEPOCH:-3}"
STAGE1_EPOCH="${STAGE1_EPOCH:-${STAGE1_NEPOCH}}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-}"
STAGE2_EXTRA_EPOCHS="${STAGE2_EXTRA_EPOCHS:-${STAGE2_NEPOCH}}"
STAGE2_FINAL_EPOCH="${STAGE2_FINAL_EPOCH:-$((STAGE1_EPOCH + STAGE2_EXTRA_EPOCHS))}"
STAGE2_EPOCH="${STAGE2_EPOCH:-${STAGE2_FINAL_EPOCH}}"
STAGE2_CHECKPOINT_PATH="${STAGE2_CHECKPOINT_PATH:-}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
MAX_VALID_ROWS="${MAX_VALID_ROWS:-0}"
MAX_STAGE1_BATCHES="${MAX_STAGE1_BATCHES:-0}"
MAX_STAGE2_BATCHES="${MAX_STAGE2_BATCHES:-0}"
MAX_VALID_BATCHES="${MAX_VALID_BATCHES:-0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-1}"
RESULT_TOP_K="${RESULT_TOP_K:-3}"
LR="${LR:-}"
WARM_UP="${WARM_UP:-}"
RUN_STAGE3="${RUN_STAGE3:-0}"
STAGE3_MAX_STEPS="${STAGE3_MAX_STEPS:-100}"
USE_PEFT="${USE_PEFT:-0}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
LORA_MODULES_TO_SAVE="${LORA_MODULES_TO_SAVE:-none}"
DISABLE_TEXT_EXTRA_TOKENS="${DISABLE_TEXT_EXTRA_TOKENS:-${USE_PEFT}}"

PEFT_ARGS=()
if [[ "${USE_PEFT}" == "1" ]]; then
  PEFT_ARGS+=(--use_peft)
  PEFT_ARGS+=(--lora_r "${LORA_R}")
  PEFT_ARGS+=(--lora_alpha "${LORA_ALPHA}")
  PEFT_ARGS+=(--lora_dropout "${LORA_DROPOUT}")
  PEFT_ARGS+=(--lora_modules_to_save "${LORA_MODULES_TO_SAVE}")
  if [[ -n "${LORA_TARGET_MODULES}" ]]; then
    PEFT_ARGS+=(--lora_target_modules "${LORA_TARGET_MODULES}")
  fi
fi
if [[ "${DISABLE_TEXT_EXTRA_TOKENS}" == "1" ]]; then
  PEFT_ARGS+=(--disable_text_extra_tokens)
fi

TRAIN_OVERRIDE_ARGS=()
if [[ -n "${LR}" ]]; then
  TRAIN_OVERRIDE_ARGS+=(--override_lr "${LR}")
fi
if [[ -n "${WARM_UP}" ]]; then
  TRAIN_OVERRIDE_ARGS+=(--override_warm_up "${WARM_UP}")
fi

STAGE2_RESUME_ARGS=()
if [[ -n "${STAGE1_CHECKPOINT_PATH}" ]]; then
  STAGE2_RESUME_ARGS+=(--checkpoint-path "${STAGE1_CHECKPOINT_PATH}")
fi

STAGE3_RESUME_ARGS=()
if [[ -n "${STAGE2_CHECKPOINT_PATH}" ]]; then
  STAGE3_RESUME_ARGS+=(--checkpoint-path "${STAGE2_CHECKPOINT_PATH}")
fi

python training.py \
  --batch_size "${BATCH_SIZE}" \
  --data_root "${DATA_ROOT}" \
  --modelname "${MODELNAME}" \
  --train_stage logic \
  --override_nepoch "${STAGE1_NEPOCH}" \
  --max_train_rows "${MAX_TRAIN_ROWS}" \
  --max_valid_rows "${MAX_VALID_ROWS}" \
  --max_train_batches "${MAX_STAGE1_BATCHES}" \
  --max_valid_batches "${MAX_VALID_BATCHES}" \
  --result_top_k "${RESULT_TOP_K}" \
  --accelerate \
  --mixed_precision "${MIXED_PRECISION}" \
  --experiment_name stage1-logic \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_pin_memory true \
  --dataloader_persistent_workers true \
  --train_log_every "${TRAIN_LOG_EVERY}" \
  --save_frequency "${SAVE_FREQUENCY}" \
  "${TRAIN_OVERRIDE_ARGS[@]}" \
  "${PEFT_ARGS[@]}"

python training.py \
  --batch_size "${BATCH_SIZE}" \
  --data_root "${DATA_ROOT}" \
  --modelname "${MODELNAME}" \
  --train_stage stage2_loop \
  --resume_epoch "${STAGE1_EPOCH}" \
  "${STAGE2_RESUME_ARGS[@]}" \
  --override_nepoch "${STAGE2_EPOCH}" \
  --max_train_rows "${MAX_TRAIN_ROWS}" \
  --max_valid_rows "${MAX_VALID_ROWS}" \
  --max_train_batches "${MAX_STAGE2_BATCHES}" \
  --max_valid_batches "${MAX_VALID_BATCHES}" \
  --result_top_k "${RESULT_TOP_K}" \
  --accelerate \
  --mixed_precision "${MIXED_PRECISION}" \
  --experiment_name stage2-action-loop \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_pin_memory true \
  --dataloader_persistent_workers true \
  --train_log_every "${TRAIN_LOG_EVERY}" \
  --save_frequency "${SAVE_FREQUENCY}" \
  "${TRAIN_OVERRIDE_ARGS[@]}" \
  "${PEFT_ARGS[@]}"

if [[ "${RUN_STAGE3}" == "1" ]]; then
  python training.py \
    --mode optimizing \
    --data_root "${DATA_ROOT}" \
    --train_stage stage2_loop \
    --modelname "${MODELNAME}" \
    --resume_epoch "${STAGE2_EPOCH}" \
    "${STAGE3_RESUME_ARGS[@]}" \
    --batch_size "${BATCH_SIZE}" \
    --result_top_k "${RESULT_TOP_K}" \
    --rl_max_steps "${STAGE3_MAX_STEPS}" \
    --rl_logging_steps 10 \
    --experiment_name stage3-rollout-rl \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --dataloader_pin_memory true \
    --dataloader_persistent_workers true \
    "${PEFT_ARGS[@]}"
fi
