import os
import sys
import gc
from typing import List, Tuple

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                               AutoTokenizer, CLIPModel,
                               Wan2_2Transformer3DModel, WanT5EncoderModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2FunControlPipeline
from videox_fun.utils import (register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)

# ==================== 配置参数 ====================

# GPU内存模式
GPU_memory_mode = "sequential_cpu_offload"

# 多GPU配置
ulysses_degree = 1
ring_degree = 1
fsdp_dit = False
fsdp_text_encoder = True
compile_dit = False

# TeaCache配置
enable_teacache = True
teacache_threshold = 0.10
num_skip_start_steps = 5
teacache_offload = False

# CFG跳过比例
cfg_skip_ratio = 0

# Riflex配置
enable_riflex = False
riflex_k = 6

# 配置文件和模型路径
config_path = "config/wan2.2/wan_civitai_5b.yaml"
model_name = "models/Diffusion_Transformer/Wan2.2-Fun-5B-Control/"

# 采样器配置
sampler_name = "Flow"
shift = 5

# 预训练模型路径（可选）
transformer_path = None
transformer_high_path = None
vae_path = None
lora_path = None
lora_high_path = None

# 推理参数
sample_size = [1280, 704]
segment_length = 81  # 每段帧数
fps = 24

# 数据类型
weight_dtype = torch.bfloat16

# 推理参数
negative_prompt = ""  # 无需negative prompt
guidance_scale = 0.0  # CFG设为0
seed = 43
num_inference_steps = 40
lora_weight = 0.55
lora_high_weight = 0.55

# ==================== 输入配置 ====================

# 控制视频路径（总帧数应为N*81）
control_video = "asset/pose.mp4"

# 图片列表（K张图片）
# 示例：images = ["img1.png", "img2.png", "img3.png"]
images = []

# 每段的首尾帧索引配置
# 格式：[(start_img_idx, end_img_idx), ...] 表示每段使用images中的哪两张作为首尾帧
# 示例：[(0, 1), (1, 2), (2, 3)] 表示3段，分别使用(0,1), (1,2), (2,3)作为首尾帧
segment_configs = []

# 输出保存路径
save_path = "samples/multi-segment-control"

# ==================== 主逻辑 ====================

# 图片直接传路径给 get_image_to_video_latent，不需要预加载

def split_control_video(control_video_path: str, segment_length: int, sample_size: List[int], fps: int):
    """将控制视频按segment_length切分成多段"""
    import cv2
    
    cap = cv2.VideoCapture(control_video_path)
    all_frames = []
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = 1 if fps is None else max(1, int(original_fps // fps))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
            all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
    
    cap.release()
    
    # 转换为tensor
    all_frames = torch.from_numpy(np.array(all_frames))  # [T, H, W, C]
    total_frames = len(all_frames)
    
    # 计算段数
    num_segments = total_frames // segment_length
    if total_frames % segment_length != 0:
        print(f"Warning: Total frames ({total_frames}) is not divisible by segment_length ({segment_length})")
        print(f"Will process {num_segments} segments, ignoring last {total_frames % segment_length} frames")
    
    # 切分段
    segments = []
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = all_frames[start_idx:end_idx]  # [81, H, W, C]
        segment = segment.permute([3, 0, 1, 2]).unsqueeze(0) / 255  # [1, C, T, H, W]
        segments.append(segment)
    
    return segments

def main():
    # 验证输入
    assert len(images) > 0, "Please provide at least one image path in 'images' list"
    assert len(segment_configs) > 0, "Please provide segment configurations in 'segment_configs' list"
    
    device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    config = OmegaConf.load(config_path)
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)
    
    # ==================== 加载模型 ====================
    print("Loading models...")
    
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    
    if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None
    
    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    if transformer_2 is not None and transformer_high_path is not None:
        print(f"From checkpoint: {transformer_high_path}")
        if transformer_high_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(transformer_high_path)
        else:
            state_dict = torch.load(transformer_high_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer_2.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    # VAE
    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)
    
    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    
    # Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()
    
    # Scheduler
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    
    # Pipeline
    pipeline = Wan2_2FunControlPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    
    # 多GPU设置
    if ulysses_degree > 1 or ring_degree > 1:
        from functools import partial
        transformer.enable_multi_gpus_inference()
        if transformer_2 is not None:
            transformer_2.enable_multi_gpus_inference()
        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            if transformer_2 is not None:
                pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
            print("Add FSDP DIT")
        if fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            print("Add FSDP TEXT ENCODER")
    
    if compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
        if transformer_2 is not None:
            for i in range(len(pipeline.transformer_2.blocks)):
                pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
        print("Add Compile")
    
    # 内存优化
    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(transformer_2, ["modulation",], device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_group_offload":
        register_auto_device_hook(pipeline.transformer)
        if transformer_2 is not None:
            register_auto_device_hook(pipeline.transformer_2)
        safe_enable_group_offload(pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)
    
    # TeaCache
    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )
        if transformer_2 is not None:
            pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)
    
    if cfg_skip_ratio is not None:
        print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
        if transformer_2 is not None:
            pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)
    
    # LoRA
    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
        if transformer_2 is not None:
            pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")
    
    # ==================== 准备输入数据 ====================
    print(f"Using {len(images)} images")
    
    print("Splitting control video...")
    control_segments = split_control_video(control_video, segment_length, sample_size, fps)
    
    num_segments = len(control_segments)
    assert len(segment_configs) == num_segments, f"segment_configs length ({len(segment_configs)}) must match number of segments ({num_segments})"
    
    video_length = int((segment_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
    
    if enable_riflex:
        pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
        if transformer_2 is not None:
            pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)
    
    # ==================== 多段推理 ====================
    all_samples = []
    
    for seg_idx in range(num_segments):
        print(f"\n{'='*50}")
        print(f"Processing segment {seg_idx + 1}/{num_segments}")
        print(f"{'='*50}")
        
        start_img_idx, end_img_idx = segment_configs[seg_idx]
        start_image_path = images[start_img_idx]
        end_image_path = images[end_img_idx]
        control_video_segment = control_segments[seg_idx].to(device)
        
        print(f"  Start image: {start_image_path}, End image: {end_image_path}")
        
        # 准备inpaint latent（传路径字符串，函数内部会加载）
        inpaint_video, inpaint_video_mask, clip_image = get_image_to_video_latent(
            start_image_path, end_image_path, video_length=video_length, sample_size=sample_size
        )
        
        # 准备control video
        input_video = control_video_segment
        input_video_mask = torch.zeros_like(input_video[:, :1])
        input_video_mask[:, :, :] = 255
        
        # 生成随机种子
        generator = torch.Generator(device=device).manual_seed(seed + seg_idx)
        
        # 推理
        with torch.no_grad():
            sample = pipeline(
                "",  # 空prompt
                num_frames=video_length,
                negative_prompt=negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                video=inpaint_video,
                mask_video=inpaint_video_mask,
                control_video=input_video,
                control_camera_video=None,
                ref_image=None,
                boundary=boundary,
                shift=shift,
            ).videos
        
        all_samples.append(sample.cpu())
        
        # 清理显存
        del inpaint_video, inpaint_video_mask, clip_image, input_video, input_video_mask, sample
        gc.collect()
        torch.cuda.empty_cache()
    
    # ==================== 拼接视频 ====================
    print(f"\n{'='*50}")
    print("Concatenating all segments...")
    print(f"{'='*50}")
    
    # all_samples: List of [1, C, T, H, W]
    # 拼接成 [1, C, N*T, H, W]
    final_video = torch.cat(all_samples, dim=2)
    
    # 保存
    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
        if transformer_2 is not None:
            pipeline = unmerge_lora(pipeline, lora_high_path, lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")
    
    def save_results():
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        index = len([path for path in os.listdir(save_path)]) + 1
        prefix = str(index).zfill(8)
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(final_video, video_path, fps=fps)
        print(f"Saved to: {video_path}")
    
    if ulysses_degree * ring_degree > 1:
        import torch.distributed as dist
        if dist.get_rank() == 0:
            save_results()
    else:
        save_results()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
