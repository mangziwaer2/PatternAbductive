# PatternAbductive

This project trains a text-to-text abductive reasoning model over a KG. The current codebase keeps only the three-step pipeline:

1. sample compact abductive data with graph-tool traces;
2. Stage 1 SFT: `OBS -> PATTERN + DSL`;
3. Stage 2 SFT: `OBS + ACTION/RESULT history -> next ACTION or final DSL`;
4. RL: stream from `OBS`, execute KG tools when a complete ACTION block appears, append RESULT, and reward the final DSL.

## Text Format

Training and rollout use explicit boundaries:

```text
OBS [entity_a] [entity_b]
<ACTION>
ACTION FIND_COMMON TARGETS [entity_a] [entity_b] TOP_K 3
</ACTION>
<RESULT>
[candidate] --[+relation]--> [entity_a]
[candidate] --[+relation]--> [entity_b]
</RESULT>
<DSL>
AND(PROJ([+relation], ENT([entity_a])), PROJ([+relation], ENT([entity_b])))
</DSL>
```

`RESULT` is deliberately compact: only subgraph edge lines are kept inside the result tags. It does not include coverage, candidate summaries, mode fields, or an extra `RESULT` line.

## Sampling

Generate the dataset directly with `sampling.py`:

```bash
python sampling.py \
  --data_root ./sampled_data_abduction/ \
  --dataname DBpedia50 \
  --result-top-k 3 \
  --condition-samples-per-query 0 \
  --restart
```

For a small local preview:

```bash
python sampling.py \
  --data_root ./sampled_data_smoke/ \
  --dataname DBpedia50 \
  --max-patterns-per-split 2 \
  --train-samples-per-pattern 2 \
  --valid-samples-per-pattern 1 \
  --test-samples-per-pattern 1 \
  --condition-samples-per-query 0 \
  --restart
```

Each JSONL row contains:

- `pattern_str`
- `observation_text`
- `logic_dsl`
- `stage2_trace`

## Stage 1 SFT

Stage 1 teaches the model the symbolic output format. The target includes both pattern abstraction and executable DSL:

```bash
python training.py \
  --mode training \
  --train_stage logic \
  --data_root ./sampled_data_abduction/ \
  --dataname DBpedia50 \
  --modelname Qwen2.5-0.5B \
  --batch_size 4 \
  --override_nepoch 1 \
  --use_peft \
  --lora_modules_to_save none \
  --disable_text_extra_tokens
```

## Stage 2 SFT

Stage 2 trains tool-use continuation. The dataloader expands each `stage2_trace` into prefix-to-next-step samples:

```bash
python training.py \
  --mode training \
  --train_stage stage2 \
  --data_root ./sampled_data_abduction/ \
  --dataname DBpedia50 \
  --modelname Qwen2.5-0.5B \
  --resume_epoch 1 \
  --batch_size 4 \
  --override_nepoch 1 \
  --use_peft \
  --lora_modules_to_save none \
  --disable_text_extra_tokens
```

The training task is still one pipeline: given the current context, predict the next model segment. During inference/RL, the tool loop runs after a complete `</ACTION>` is generated.

## Rollout RL

RL starts from `OBS` only. The model streams tokens until `</ACTION>` or `</DSL>`:

- `</ACTION>`: parse ACTION, execute KG tool, append `<RESULT>...</RESULT>`, then continue.
- `</DSL>`: stop rollout and score the DSL by parse/execution quality and answer-set overlap.

```bash
python training.py \
  --mode rl \
  --train_stage stage2 \
  --data_root ./sampled_data_abduction/ \
  --dataname DBpedia50 \
  --modelname Qwen2.5-0.5B \
  --resume_epoch 1 \
  --rl_max_steps 100 \
  --rl_max_action_steps 3 \
  --rl_max_completion_length 128
```

## Testing

Use `testing.py` for a quick local checkpoint check.

Stage 1:

```bash
python testing.py \
  --stage logic \
  --modelname Qwen2.5-0.5B \
  --checkpoint-path /path/to/checkpoint-or-kaggle-input-dir \
  --obs "[entity_a] [entity_b]" \
  --disable_text_extra_tokens \
  --use_peft
```

Stage 2:

```bash
python testing.py \
  --stage stage2 \
  --modelname Qwen2.5-0.5B \
  --checkpoint-path /path/to/checkpoint-or-kaggle-input-dir \
  --obs "[entity_a] [entity_b]" \
  --max_action_steps 3 \
  --disable_text_extra_tokens \
  --use_peft
```

If the model emits a complete `<ACTION>...</ACTION>` block, the script executes the KG action, prints compact `<RESULT>...</RESULT>`, and continues until a compact `<DSL>...</DSL>` block is produced or the step limit is reached.

## Notes

- `--checkpoint-path` can point directly to a downloaded Kaggle checkpoint or checkpoint directory.
- `--max_train_rows` limits the first rows loaded for a run; it is not per-epoch random resampling.
- `--max_train_batches` is useful for smoke experiments.
- Keep `ckpt/` for local checkpoints; generated `results/` and `dataset_cache/` can be removed when you need a clean run.
