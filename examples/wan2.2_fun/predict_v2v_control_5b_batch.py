"""
Batch inference script for Wan2.2-Fun-5B-Control with LoRA weights.
Reads test data from structured directories and generates videos.

Test data structure:
    background_test/
        1/
            image/           # First and last frame images
                xxx.png      # First frame
                yyy.png      # Last frame
            3dgs_render/     # Control signal video
                gs_render.mp4
            gen_index.json   # Generation configuration
            prompt.txt       # Text prompt
        2/
            ...
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

# Test data directory
test_dir = "/mnt/DataPart/jianghongda/test/background_test"

# Output directory
output_dir = "samples/background_test_results"

# Generation parameters
frames_per_segment = 81  # Number of frames per segment (will be adjusted to multiple of 4)
target_height = 1080     # 1080P resolution, width will be calculated to maintain aspect ratio
fps = 24
num_inference_steps = 50
guidance_scale = 6.0
seed = 42

def adjust_frames_to_4n_plus_1(num_frames):
    """Adjust frame count to satisfy (n-1) % 4 == 0, i.e., n % 4 == 1.
    Valid frame counts: 1, 5, 9, 13, 17, 21, ..., 81, 85, 89, ...
    """
    if (num_frames - 1) % 4 == 0:
        return num_frames
    # Round up to next valid frame count
    return ((num_frames - 1) // 4 + 1) * 4 + 1

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
from safetensors.torch import load_file, save_file
import tempfile

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

# Create a temporary file with only LoRA weights (excluding mask encoder)
with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
    tmp_path = tmp.name
    save_file(lora_state_dict, tmp_path)

# Merge LoRA into pipeline using the filtered weights
pipeline = merge_lora(pipeline, tmp_path, lora_weight, device=device, dtype=weight_dtype)
print("LoRA weights merged.")

# Clean up temporary file
import os
os.unlink(tmp_path)

# Note: Mask encoder is not used in inference (only for training)
# The model has learned to handle black regions in control videos

# ==================== Helper Functions ====================
def get_control_video_dimensions(video_path):
    """Get video dimensions and frame count."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, total_frames

def calculate_target_size(orig_width, orig_height, target_h):
    """Calculate target size maintaining aspect ratio."""
    # Ensure target_h is divisible by 16
    target_h = (target_h // 16) * 16
    
    target_w = int(target_h * orig_width / orig_height)
    target_w = target_w - (target_w % 16)  # Ensure divisible by 16
    return target_h, target_w

# ==================== Process Test Cases ====================
os.makedirs(output_dir, exist_ok=True)

# Get all test case directories (1, 2, 3, 4, 5, 6)
test_cases = sorted([d for d in os.listdir(test_dir) 
                     if os.path.isdir(os.path.join(test_dir, d)) and d.isdigit()])

print(f"Found {len(test_cases)} test cases: {test_cases}")

for case_idx, case_name in enumerate(tqdm(test_cases, desc="Processing test cases")):
    case_dir = os.path.join(test_dir, case_name)
    
    # Load prompt
    prompt_path = os.path.join(case_dir, "prompt.txt")
    if not os.path.exists(prompt_path):
        print(f"[{case_idx+1}/{len(test_cases)}] Prompt file not found: {prompt_path}, skipping...")
        continue
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()
    
    # Load gen_index.json (located in 3dgs_render directory)
    gen_index_path = os.path.join(case_dir, "3dgs_render", "gen_index.json")
    if not os.path.exists(gen_index_path):
        print(f"[{case_idx+1}/{len(test_cases)}] gen_index.json not found: {gen_index_path}, skipping...")
        continue
    with open(gen_index_path, 'r') as f:
        gen_index = json.load(f)
    
    # Get image files (sorted by name)
    image_dir = os.path.join(case_dir, "image")
    if not os.path.exists(image_dir):
        print(f"[{case_idx+1}/{len(test_cases)}] Image directory not found: {image_dir}, skipping...")
        continue
    image_files = sorted([f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if len(image_files) < 2:
        print(f"[{case_idx+1}/{len(test_cases)}] Need at least 2 images, found {len(image_files)}, skipping...")
        continue
    
    # Load control video
    control_video_path = os.path.join(case_dir, "3dgs_render", "gs_render.mp4")
    if not os.path.exists(control_video_path):
        print(f"[{case_idx+1}/{len(test_cases)}] Control video not found: {control_video_path}, skipping...")
        continue
    
    # Get control video dimensions
    ctrl_width, ctrl_height, ctrl_total_frames = get_control_video_dimensions(control_video_path)
    target_h, target_w = calculate_target_size(ctrl_width, ctrl_height, target_height)
    
    print(f"\n[{case_idx+1}/{len(test_cases)}] Processing case {case_name}")
    print(f"  Prompt: {prompt[:50]}...")
    print(f"  Images: {image_files}")
    print(f"  Control video: {ctrl_total_frames} frames, {ctrl_width}x{ctrl_height}")
    print(f"  Target size: {target_w}x{target_h}")
    print(f"  Gen index: {gen_index}")
    
    # Parse gen_index to get segments
    # Format: [[[first_img_idx, first_frame_idx], [last_img_idx, last_frame_idx]], ...]
    segments = gen_index[:-1]  # Last element is not a segment
    
    all_generated_frames = []
    
    for seg_idx, segment in enumerate(segments):
        first_img_idx, first_frame_idx = segment[0]
        last_img_idx, last_frame_idx = segment[1]
        
        # Load first and last frame images
        first_img_path = os.path.join(image_dir, image_files[first_img_idx])
        last_img_path = os.path.join(image_dir, image_files[last_img_idx])
        
        first_frame = cv2.imread(first_img_path)
        last_frame = cv2.imread(last_img_path)
        
        if first_frame is None or last_frame is None:
            print(f"  [Segment {seg_idx+1}] Failed to load images, skipping segment...")
            continue
        
        # Convert to RGB
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        
        # Resize frames to target size
        first_frame_pil = Image.fromarray(first_frame_rgb).resize((target_w, target_h))
        last_frame_pil = Image.fromarray(last_frame_rgb).resize((target_w, target_h))
        
        # Extract control video segment
        start_frame = first_frame_idx
        end_frame = last_frame_idx
        
        # Calculate actual frame count (must satisfy (n-1) % 4 == 0)
        actual_frame_count = end_frame - start_frame + 1
        adjusted_frame_count = adjust_frames_to_4n_plus_1(actual_frame_count)
        
        print(f"    [DEBUG] Frame range: {start_frame}-{end_frame}, actual: {actual_frame_count}, adjusted: {adjusted_frame_count}")
        
        # Prepare inpaint video (first/last frames as constraints)
        inpaint_video, inpaint_video_mask, clip_image = get_image_to_video_latent(
            [first_frame_pil], [last_frame_pil], 
            video_length=adjusted_frame_count, 
            sample_size=[target_h, target_w]
        )
        print(f"    [DEBUG] inpaint_video shape: {inpaint_video.shape}")
        
        # Calculate actual frame count (must be multiple of 4)
        actual_frame_count = end_frame - start_frame + 1
        adjusted_frame_count = adjust_frames_to_multiple_of_4(actual_frame_count)
        
        # Read control video and extract specific frame range
        cap = cv2.VideoCapture(control_video_path)
        control_frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if start_frame <= frame_idx <= end_frame:
                frame = cv2.resize(frame, (target_w, target_h))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                control_frames.append(frame_rgb)
            frame_idx += 1
            if frame_idx > end_frame:
                break
        cap.release()
        
        # Pad frames if necessary to satisfy (n-1) % 4 == 0
        while len(control_frames) < adjusted_frame_count:
            control_frames.append(control_frames[-1])  # Repeat last frame
        print(f"    [DEBUG] After padding: {len(control_frames)} frames")
        
        # Convert to tensor format [1, C, F, H, W]
        control_frames_array = np.array(control_frames)
        input_video = torch.from_numpy(control_frames_array).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        input_video_mask = torch.zeros_like(input_video[:, :1])
        
        print(f"    [DEBUG] control_video shape: {input_video.shape}")
        
        # Update frames_per_segment to match actual frame count
        actual_frames_per_segment = len(control_frames)
        
        # Generate video segment
        print(f"  [Segment {seg_idx+1}/{len(segments)}] Generating frames {start_frame}-{end_frame} (adjusted to {actual_frames_per_segment} frames)...")
        
        generator = torch.Generator(device=device).manual_seed(seed + case_idx * 100 + seg_idx)
        
        with torch.no_grad():
            sample = pipeline(
                prompt,
                num_frames=actual_frames_per_segment,
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
        
        # Collect frames (sample shape: [1, C, F, H, W])
        segment_frames = sample[0].permute(1, 0, 2, 3)  # [F, C, H, W]
        all_generated_frames.append(segment_frames)
    
    if not all_generated_frames:
        print(f"  No segments generated, skipping...")
        continue
    
    # Concatenate all segments
    final_video = torch.cat(all_generated_frames, dim=0)  # [F_total, C, H, W]
    final_video = final_video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, F_total, H, W]
    
    # Save output
    output_filename = f"{case_name}_generated.mp4"
    output_path = os.path.join(output_dir, output_filename)
    save_videos_grid(final_video, output_path, fps=fps)
    print(f"  Saved to: {output_path}")

print(f"\nBatch inference completed! Results saved to: {output_dir}")
