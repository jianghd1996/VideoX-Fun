"""
Batch inference script for Wan2.2-Fun-5B-Control with LoRA weights.
Reads a JSON file and generates videos for each entry.
"""
import os
import sys
import json
import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import AutoencoderKLWan3_8, Wan2_2Transformer3DModel, WanT5EncoderModel
from videox_fun.pipeline import Wan2_2FunControlPipeline
from videox_fun.utils.utils import (get_image_to_video_latent, get_video_to_video_latent,
                                    save_videos_grid, filter_kwargs)
from videox_fun.utils.lora_utils import merge_lora
from transformers import AutoTokenizer

# ==================== Configuration ====================
# Model paths
config_path = "config/wan2.2/wan_civitai_5b.yaml"
model_name = "/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-Control/"

# LoRA checkpoint (trained with mask encoder)
lora_path = "/mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun/output_dir_wan2.2_fun_control_lora/checkpoint-25000.safetensors"
lora_weight = 1.0  # LoRA weight strength

# Input JSON file (same format as training dataset)
input_json = "/mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun-ori/datasets/dataset1+2.json"
data_root = "/mnt/DataPart/jianghongda/dataset/livephoto"

# Output directory
output_dir = "samples/batch_inference"

# Generation parameters
video_length = 81  # Number of frames
sample_size = [960, 960]  # [height, width] - 720P
fps = 24
num_inference_steps = 50
guidance_scale = 6.0
seed = 42

# Device and dtype
device = "cuda:0"
weight_dtype = torch.bfloat16

# ==================== Load Models ====================
print("Loading models...")
config = OmegaConf.load(config_path)
boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

# Load Transformer
transformer = Wan2_2Transformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Load VAE
vae = AutoencoderKLWan3_8.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

# Load Tokenizer and Text Encoder
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Load Scheduler
scheduler = FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Create Pipeline
pipeline = Wan2_2FunControlPipeline(
    transformer=transformer,
    transformer_2=None,  # 5B model has only one transformer
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)

# Enable sequential CPU offload to save VRAM
pipeline.enable_sequential_cpu_offload(device=device)
print("Models loaded and moved to device.")

# ==================== Load LoRA Weights ====================
print(f"Loading LoRA weights from {lora_path}...")
from safetensors.torch import load_file
state_dict = load_file(lora_path, device="cpu")

# Separate LoRA and mask encoder weights
lora_state_dict = {}
mask_encoder_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("control_mask_encoder."):
        mask_encoder_state_dict[k.replace("control_mask_encoder.", "")] = v
    else:
        lora_state_dict[k] = v

print(f"Loaded {len(lora_state_dict)} LoRA weights and {len(mask_encoder_state_dict)} mask encoder weights")

# Merge LoRA into pipeline
pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
print("LoRA weights merged.")

# Note: Mask encoder is not used in inference (only for training)
# The model has learned to handle black regions in control videos

# ==================== Load Dataset ====================
print(f"Loading dataset from {input_json}...")
with open(input_json, 'r') as f:
    dataset = json.load(f)
print(f"Loaded {len(dataset)} entries.")

# ==================== Batch Inference ====================
os.makedirs(output_dir, exist_ok=True)

for idx, data_info in enumerate(tqdm(dataset, desc="Processing")):
    gt_video_path = data_info['file_path']
    control_video_path = data_info.get('control_file_path', '')
    prompt = data_info.get('text', '')
    
    # Resolve paths
    if data_root:
        gt_video_full_path = os.path.join(data_root, gt_video_path)
        control_video_full_path = os.path.join(data_root, control_video_path) if control_video_path else ''
    else:
        gt_video_full_path = gt_video_path
        control_video_full_path = control_video_path
    
    # Check if files exist
    if not os.path.exists(gt_video_full_path):
        print(f"[{idx+1}/{len(dataset)}] GT video not found: {gt_video_full_path}, skipping...")
        continue
    if control_video_full_path and not os.path.exists(control_video_full_path):
        print(f"[{idx+1}/{len(dataset)}] Control video not found: {control_video_full_path}, skipping...")
        continue
    
    # Extract first and last frames from GT video
    gt_cap = cv2.VideoCapture(gt_video_full_path)
    gt_total_frames = int(gt_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gt_width = int(gt_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gt_height = int(gt_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    gt_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret_first, first_frame = gt_cap.read()
    gt_cap.set(cv2.CAP_PROP_POS_FRAMES, gt_total_frames - 1)
    ret_last, last_frame = gt_cap.read()
    gt_cap.release()
    
    if not ret_first or not ret_last:
        print(f"[{idx+1}/{len(dataset)}] Failed to read frames from {gt_video_full_path}, skipping...")
        continue
    
    # Convert to RGB
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
    
    # Calculate target size (maintain aspect ratio)
    target_h = sample_size[0]
    target_w = int(target_h * gt_width / gt_height)
    target_w = target_w - (target_w % 16)  # Ensure divisible by 16
    
    # Resize frames
    first_frame_pil = Image.fromarray(first_frame_rgb).resize((target_w, target_h))
    last_frame_pil = Image.fromarray(last_frame_rgb).resize((target_w, target_h))
    
    # Prepare inpaint video (first/last frames as constraints)
    inpaint_video, inpaint_video_mask, clip_image = get_image_to_video_latent(
        [first_frame_pil], [last_frame_pil], video_length=video_length, sample_size=[target_h, target_w]
    )
    
    # Load control video
    input_video, input_video_mask, _, _ = get_video_to_video_latent(
        control_video_full_path, video_length=video_length, sample_size=[target_h, target_w]
    )
    
    # Generate video
    print(f"[{idx+1}/{len(dataset)}] Generating video for: {os.path.basename(gt_video_path)}")
    
    generator = torch.Generator(device=device).manual_seed(seed + idx)
    
    with torch.no_grad():
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            height=target_h,
            width=target_w,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            video=inpaint_video,
            mask_video=inpaint_video_mask,
            control_video=input_video,
            boundary=boundary,
        ).videos
    
    # Save output
    output_filename = f"{idx:05d}_{os.path.splitext(os.path.basename(gt_video_path))[0]}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    save_videos_grid(sample, output_path, fps=fps)
    print(f"Saved to: {output_path}")

print(f"\nBatch inference completed! Results saved to: {output_dir}")
