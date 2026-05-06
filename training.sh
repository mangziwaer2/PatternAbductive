export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python training.py \
  --batch_size 1 \
  --checkpoint-path "/root/autodl-tmp/Qwen2.5-0.5B" \
  --data_root "/root/autodl-tmp/sampled_data_abduction" \
  --modelname "Qwen2.5-0.5B" \
  --train_stage stage2 \
  --mode training \
  --resume_epoch 1 \
  --override_nepoch 3 \
  --max_train_rows 0 \
  --max_valid_rows 0 \
  --max_train_batches 0 \
  --max_valid_batches 0 \
  --result_top_k 3 \
  --accelerate \
  --mixed_precision "fp16" \
  --experiment_name stage2-action \
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
  --override_warm_up 100 \
  --intra_epoch_comparison_every 5000 \
&& \
python training.py \
  --batch_size 1 \
  --checkpoint-path latest \
  --data_root "/root/autodl-tmp/sampled_data_abduction"  \
  --modelname "Qwen2.5-0.5B" \
  --train_stage stage2 \
  --mode rl \
  --resume_epoch 3 \
  --max_train_rows 0 \
  --result_top_k 5 \
  --experiment_name stage3-rl \
  --dataset_num_proc 4 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory true \
  --use_peft \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_modules_to_save none \
  --disable_text_extra_tokens \
  --rl_lr 1e-6 \
  --rl_epochs 1 \
  --rl_max_steps 50000 \
  --rl_max_action_steps 6 \
  --rl_max_completion_length 128 \
  --rl_logging_steps 1000 \
  --rl_save_steps 1000


  
