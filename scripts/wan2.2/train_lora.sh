#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="/mnt/DataPart/jianghongda/"
export DATASET_META_NAME="datasets/all_video_dataset_prompts.json"
export VALIDATION_DATA_DIR="/mnt/DataPart/jianghongda/VideoX-Fun-dev/test_data/livephoto_test"
export PRETRAIN_LORA="/mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun-Single8/output_dir_wan2.2_lora_8_trajectory/checkpoint-1600.safetensors"

# Only enable these for multi-node runs without RDMA/P2P support.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch \
  --gpu_ids 4,5 \
  --num_processes 2 \
  --mixed_precision bf16 \
  scripts/wan2.2/train_lora.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --transformer_path="$PRETRAIN_LORA" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --image_sample_size=960 \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=121 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=400 \
  --checkpoints_total_limit=3 \
  --learning_rate=1e-4 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_lora_8_trajectory" \
  --report_to=tensorboard \
  --logging_dir=logs \
  --tracker_project_name="wan2.2-trajectory-lora" \
  --gradient_checkpointing \
  --mixed_precision=bf16 \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type=full \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode=ti2v \
  --ti2v_condition_probability=0.95 \
  --validation_data_dir="$VALIDATION_DATA_DIR" \
  --validation_steps=500 \
  --validation_epochs=0 \
  --validation_cases_per_process=1 \
  --validation_sample_size=720 \
  --validation_num_inference_steps=8 \
  --validation_guidance_scale=6.0 \
  --low_vram
