# PatternAbductive

Minimal runnable pipeline for KG abduction with executable DSL and tool-calling supervision.

## Data

Convert the existing surface dataset without re-sampling:

```bash
conda run -n patternabductive python scripts/convert_surface_to_abduction.py ^
  --input-root ./sampled_data_surface/ ^
  --output-root ./sampled_data_abduction/ ^
  --splits train,valid,test ^
  --result-top-k 3 ^
  --max-observation-entities 8 ^
  --trace-mode lazy ^
  --overwrite
```

For fast Stage 2 SFT, prefill `stage2_trace` once and train from the traced root:

```bash
conda run -n patternabductive python scripts/hydrate_stage2_traces.py ^
  --input-root ./sampled_data_abduction/ ^
  --output-root ./sampled_data_abduction_traced/ ^
  --splits train,valid,test ^
  --result-top-k 3 ^
  --trace-cache-size 100000 ^
  --overwrite
```

Or sample the target format directly:

```bash
conda run -n patternabductive python sampling.py ^
  --data_root ./sampled_data_abduction/ ^
  --result-top-k 3 ^
  --max-answer-size 8 ^
  --restart
```

The target JSONL keeps only:

```text
pattern_str
observation_text
logic_dsl
stage2_trace
```

`sampled_data_abduction/` may keep `stage2_trace` empty for quick conversion. `sampled_data_abduction_traced/` stores the same compact rows with precomputed ACTION/RESULT trace, so Stage 2 no longer calls the KG during SFT preprocessing.

## Stage 1

```bash
conda run -n patternabductive python training.py ^
  --data_root ./sampled_data_abduction_traced/ ^
  --train_stage logic ^
  --modelname GPT2_6_act_nt ^
  --dataset_num_proc 8 ^
  --dataloader_num_workers 4 ^
  --dataloader_pin_memory true
```

## Stage 2

```bash
conda run -n patternabductive python training.py ^
  --data_root ./sampled_data_abduction_traced/ ^
  --train_stage stage2_loop ^
  --resume_epoch <stage1_epoch> ^
  --modelname GPT2_6_act_nt ^
  --dataset_num_proc 8 ^
  --dataloader_num_workers 4 ^
  --dataloader_pin_memory true
```

## Stage 3

```bash
conda run -n patternabductive python scripts/run_stage3_rollout_train.py ^
  --data_root ./sampled_data_abduction_traced/ ^
  --split train ^
  --modelname GPT2_6_act_nt ^
  --scale <stage2_scale> ^
  --resume_epoch <stage2_epoch> ^
  --max-steps 100
```

Stage 3 does full rollout RL from `OBS`; it does not use dataset action traces as targets.

Rollout evaluation:

```bash
conda run -n patternabductive python scripts/run_stage3_rollout_eval.py ^
  --data_root ./sampled_data_abduction_traced/ ^
  --resume_epoch <stage2_or_rl_epoch> ^
  --max-rows 20
```

## Training Speed

The current minimal speed path is:

```text
1. Hydrate stage2_trace once.
2. Let utils/dataloader.py expand trace into prefix-to-next-step samples.
3. Save the expanded HuggingFace dataset cache under dataset_cache/.
4. Reuse the processed cache on later runs with the same data_root, train_stage, top_k, and max_rows.
5. Use dataset_num_proc for HF map and dataloader_num_workers/pin_memory for batch loading.
6. Skip KG loading automatically during SFT when logic_dsl/stage2_trace already exist.
7. Use --accelerate --mixed_precision fp16 or bf16 on cloud GPUs that support it.
```

Kaggle notebook cells can call `training.py` directly. Use `--lora_modules_to_save none` and `--disable_text_extra_tokens` for lightweight LoRA; otherwise embedding/lm_head can dominate the trainable parameter count.

Stage 1:

```bash
!python training.py \
  --batch_size 4 \
  --data_root "/kaggle/input/datasets/mangziwaer2/abductive-sampled/sampled_data_abduction_traced" \
  --modelname "Qwen2.5-0.5B" \
  --train_stage logic \
  --override_nepoch 1 \
  --max_train_rows 0 \
  --max_valid_rows 0 \
  --max_train_batches 2000 \
  --max_valid_batches 100 \
  --result_top_k 3 \
  --accelerate \
  --mixed_precision "fp16" \
  --experiment_name stage1-logic \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory true \
  --dataloader_persistent_workers true \
  --train_log_every 100 \
  --save_frequency 1 \
  --use_peft \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_modules_to_save none \
  --disable_text_extra_tokens \
  --override_lr 1e-4 \
  --override_warm_up 100
```

Stage 2:

```bash
!python training.py \
  --batch_size 4 \
  --data_root "/kaggle/input/datasets/mangziwaer2/abductive-sampled/sampled_data_abduction_traced" \
  --modelname "Qwen2.5-0.5B" \
  --train_stage stage2_loop \
  --resume_epoch 1 \
  --override_nepoch 2 \
  --max_train_rows 0 \
  --max_valid_rows 0 \
  --max_train_batches 3000 \
  --max_valid_batches 100 \
  --result_top_k 3 \
  --accelerate \
  --mixed_precision "fp16" \
  --experiment_name stage2-action-loop \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory true \
  --dataloader_persistent_workers true \
  --train_log_every 100 \
  --save_frequency 1 \
  --use_peft \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_modules_to_save none \
  --disable_text_extra_tokens \
  --override_lr 1e-4 \
  --override_warm_up 100
```

Set `--max_train_batches 0` only when intentionally running the full split.

## Model Switch

Local smoke tests use `GPT2_6_act_nt`. Cloud runs can switch models through:

```bash
--modelname Qwen2.5-0.5B --config-model configs/config-model.yml
```

Model entries live in `configs/config-model.yml`.
