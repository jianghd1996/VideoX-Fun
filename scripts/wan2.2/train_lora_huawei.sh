#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-/cache/02_model/Wan2.2-TI2V-5B}"
DATASET_NAME="${DATASET_NAME:-/cache/00_data}"
DATASET_META_NAME="${DATASET_META_NAME:-/cache/00_data/theta30_phi5_self_render_cloud.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/cache/03_output/trajectory_lora_theta30_phi5}"

GPU_IDS="${GPU_IDS:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-200}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=(--initial_global_step="${INITIAL_GLOBAL_STEP:-0}")
if [[ -n "${PRETRAIN_LORA:-}" ]]; then
  EXTRA_ARGS+=(--transformer_path="${PRETRAIN_LORA}")
fi
if [[ "${LOW_VRAM:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--low_vram)
fi

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Model:       ${MODEL_NAME}"
echo "Dataset:     ${DATASET_NAME}"
echo "Metadata:    ${DATASET_META_NAME}"
echo "Output:      ${OUTPUT_DIR}"
echo "GPU IDs:     ${GPU_IDS}"
echo "Processes:   ${NUM_PROCESSES}"
echo "Master port: ${MAIN_PROCESS_PORT}"

accelerate launch \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --gpu_ids "${GPU_IDS}" \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision="bf16" \
  scripts/wan2.2/train_lora.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path="${MODEL_NAME}" \
  --train_data_dir="${DATASET_NAME}" \
  --train_data_meta="${DATASET_META_NAME}" \
  --image_sample_size=960 \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=121 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  --num_train_epochs="${NUM_TRAIN_EPOCHS}" \
  --checkpointing_steps=400 \
  --checkpoints_total_limit=3 \
  --learning_rate=5e-5 \
  --seed=42 \
  --output_dir="${OUTPUT_DIR}" \
  --logging_dir="${OUTPUT_DIR}/tensorboard" \
  --report_to="tensorboard" \
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
  --boundary_type="full" \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="ti2v" \
  --trainable_modules="." \
  --random_ratio_crop \
  --random_frame_crop \
  --random_hw_adapt \
  --keep_all_node_same_token_length \
  --use_dynamic_cfg \
  --cfg_skip_ratio=0.1 \
  --ti2v_condition_dropout_ratio=0.05 \
  "${EXTRA_ARGS[@]}"
