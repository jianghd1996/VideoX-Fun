export MODEL_NAME="/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-Control"
export DATASET_NAME="/mnt/DataPart/jianghongda/dataset/livephoto"
export DATASET_META_NAME="datasets/dataset.json"

# Reduce CUDA memory fragmentation (prevents OOM during validation decode)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Increase NCCL timeout to handle large video loading differences between ranks
export NCCL_TIMEOUT=1800  # 30 minutes instead of default 10 minutes

accelerate launch --gpu-ids 4,5 --num_processes=2 --mixed_precision="bf16" scripts/wan2.2_fun/train_control_lora.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=720 \
  --video_sample_size=720 \
  --token_sample_size=720 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --validation_steps=200 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_5b_control_lora" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_inpaint_info \
  --control_mask_ratio=0.3 \
  --add_full_ref_image_in_self_attention \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --low_vram \
  --validation_samples 4 \
  --validation_n_frames 81 \
  --validation_sample_size 720 \
  --resume_from_checkpoint="output_dir_wan2.2_5b_control_lora/checkpoint-2400.safetensors"
