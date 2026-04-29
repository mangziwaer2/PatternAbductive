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
  --rebuild ^
  --overwrite
```

Validate that each gold `logic_dsl` can explain its `OBS` on the matching KG split:

```bash
conda run -n patternabductive python scripts/validate_abduction_dataset.py ^
  --data-root ./sampled_data_abduction_traced/ ^
  --splits train,valid,test ^
  --min-obs-recall 1.0 ^
  --diagnose-splits train,valid,test ^
  --overwrite
```

To write a filtered copy for reward-sensitive Stage 3 experiments, add an output root:

```bash
conda run -n patternabductive python scripts/validate_abduction_dataset.py ^
  --data-root ./sampled_data_abduction_traced/ ^
  --output-root ./sampled_data_abduction_checked/ ^
  --splits train,valid,test ^
  --min-obs-recall 1.0 ^
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

Current ACTION schema:

```text
ACTION FIND_COMMON TARGETS [...] TOP_K k
ACTION FIND_ALTERNATIVE TARGETS [...] TOP_K k
ACTION FIND_EXCLUSION TARGETS [...] TOP_K k
ACTION EXPAND TARGETS [...] DIRECTION backward|forward TOP_K k
ACTION CHECK_COVERAGE CANDIDATES [...] OBS [...] TOP_K k
```

Multi-hop supervision is represented by repeated one-hop actions. For example, a two-hop branch is trained as `FIND_* -> EXPAND -> CHECK_COVERAGE -> DSL`, not as a single oracle `MAX_HOPS 2` request.

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
conda run -n patternabductive python training.py ^
  --mode optimizing ^
  --data_root ./sampled_data_abduction_traced/ ^
  --train_stage stage2_loop ^
  --modelname GPT2_6_act_nt ^
  --resume_epoch <stage2_epoch> ^
  --rl_max_steps 100 ^
  --rl_logging_steps 10
```

Stage 3 now enters through `training.py --mode optimizing`. For `--train_stage stage2_loop`, it does full rollout RL from `OBS`; it does not use dataset action traces as targets. If the model emits `ACTION`, KG is called and `RESULT` is appended. If it emits `DSL`, the rollout stops and the executable DSL is scored against the input OBS.

Rollout evaluation:

```bash
conda run -n patternabductive python scripts/run_stage3_rollout_eval.py ^
  --data_root ./sampled_data_abduction_traced/ ^
  --resume_epoch <stage2_or_rl_epoch> ^
  --max-rows 20
```

## Checkpoint Inference

Test a downloaded checkpoint locally with a single observation. Stage 1 checkpoints generate `PATTERN + DSL` directly:

```bash
conda run -n patternabductive python scripts/test_checkpoint_inference.py ^
  --stage logic ^
  --modelname Qwen2.5-0.5B ^
  --checkpoint-path ./downloaded_ckpt/DBpedia50-default-8-1-text2text.pth ^
  --observation "OBS [Augustin de Lespinasse]" ^
  --disable-text-extra-tokens
```

Stage 2 checkpoints run the tool loop. The model emits `ACTION`, the script executes the KG tool, appends `RESULT`, and repeats until `DSL` or `--max-action-steps`:

```bash
conda run -n patternabductive python scripts/test_checkpoint_inference.py ^
  --stage stage2_loop ^
  --modelname Qwen2.5-0.5B ^
  --checkpoint-path ./downloaded_ckpt/DBpedia50-default-8-2-text2text.pth ^
  --observation "OBS [Augustin de Lespinasse]" ^
  --graph-split train ^
  --max-action-steps 3 ^
  --disable-text-extra-tokens ^
  --print-raw
```

For LoRA checkpoints, keep the `.pth.adapter/` directory next to the `.pth` file.

## Training Speed

The current minimal speed path is:

```text
1. Hydrate stage2_trace once.
2. Let utils/dataloader.py expand trace into prefix-to-next-step samples.
3. Save the expanded HuggingFace dataset cache under dataset_cache/.
4. Reuse the processed cache on later runs with the same data_root, train_stage, ACTION schema, top_k, and max_rows.
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
  --checkpoint-path "/kaggle/input/stage1-logic-ckpt/DBpedia50-default-8-1-text2text.pth" \
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

`--checkpoint-path` is optional when the checkpoint is already under `./ckpt/<modelname>/`.
On Kaggle it is useful after downloading/uploading Stage 1 output as a dataset under `/kaggle/input`.
For LoRA checkpoints, upload both the `.pth` file and the sibling `.pth.adapter/` directory.

Set `--max_train_batches 0` only when intentionally running the full split.

## Model Switch

Local smoke tests use `GPT2_6_act_nt`. Cloud runs can switch models through:

```bash
--modelname Qwen2.5-0.5B --config-model configs/config-model.yml
```

Model entries live in `configs/config-model.yml`.
