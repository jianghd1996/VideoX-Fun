"""Persistent TI2V adapter based on examples/wan2.2/predict_ti2v.py.

Place this file at SingleRecon/VideoX-Fun/predict_ti2v.py.
Use video_adapter='predict_ti2v:Wan_5b'.
"""
import json
from dataclasses import dataclass
import os
import sys
from datetime import datetime

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
video_code_root = os.path.dirname(current_file_path)
if video_code_root not in sys.path:
    sys.path.insert(0, video_code_root)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                               AutoTokenizer, Wan2_2Transformer3DModel,
                               WanT5EncoderModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline.pipeline_wan2_2_ti2v import Wan2_2TI2VPipeline
from videox_fun.utils import (register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils_new import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                    save_videos_grid)

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, model_group_offload, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_group_offload transfers internal layer groups between CPU/CUDA, 
# balancing memory efficiency and speed between full-module and leaf-level offloading methods.
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "model_full_load"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = int(os.environ.get("SINGLERECON_WORLD_SIZE", "1"))
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = False
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# TeaCache config
enable_teacache     = False
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold |
# | Wan2.2-T2V-A14B     | 0.10~0.15 | Wan2.2-I2V-A14B     | 0.15~0.20 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = os.path.join(video_code_root, "config/wan2.2/wan_civitai_5b.yaml")
# model path
model_name          = os.path.join(os.path.dirname(video_code_root), "Cloud_Models/Wan2.2-TI2V-5B")

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow_Unipc"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
shift               = 5

# Load pretrained model if need
# The transformer_path is used for low noise model, the transformer_high_path is used for high noise model.
# Since Wan2.2-5b consists of only one model, only transformer_path is used.
transformer_path        = None
transformer_high_path   = None
vae_path                = None
use_lightvae            = os.environ.get("SINGLERECON_USE_LIGHTVAE", "0") == "1"
lightvae_path           = os.environ.get("SINGLERECON_LIGHTVAE_PATH")
# Load lora model if need
# The lora_path is used for low noise model, the lora_high_path is used for high noise model.
# Since Wan2.2-5b consists of only one model, only lora_path is used.
lora_path               = os.environ.get("SINGLERECON_LORA_PATH")
lora_high_path          = None

# Batch test configuration
test_data_dir       = "/mnt/DataPart/jianghongda/VideoX-Fun-dev/test_data/livephoto_test"
prompt_json_path    = None  # Auto-detect the unique JSON file in test_data_dir.
target_short_edge   = 1088
align_to             = 32
video_length        = 121
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16.
weight_dtype        = torch.bfloat16
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
guidance_scale      = 0.0
seed                = 42
num_inference_steps = 8
# PEFT LoRA checkpoints are evaluated at their trained strength.
lora_weight         = float(os.environ.get("SINGLERECON_LORA_WEIGHT", "0.55"))
lora_high_weight    = 1.0
checkpoint_name     = os.path.splitext(os.path.basename(lora_path))[0] if lora_path else "base"
save_path           = os.path.join(
    "samples",
    "trajectory_lora_test",
    f"{checkpoint_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)



class Wan_5b:
    def __init__(self):
        self.pipeline = None
        self.load()

    def load(self):
        if self.pipeline is not None:
            return
        # Retain model, VAE, tokenizer and LoRA on GPU across all cases.
        device = set_multi_gpus_devices(ulysses_degree, ring_degree)
        config = OmegaConf.load(config_path)
        boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)
        
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
                from safetensors.torch import load_file, safe_open
                state_dict = load_file(transformer_path)
            else:
                state_dict = torch.load(transformer_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        
            m, u = transformer.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        
        if transformer_2 is not None:
            if transformer_high_path is not None:
                print(f"From checkpoint: {transformer_high_path}")
                if transformer_high_path.endswith("safetensors"):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(transformer_high_path)
                else:
                    state_dict = torch.load(transformer_high_path, map_location="cpu")
                state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        
                m, u = transformer_2.load_state_dict(state_dict, strict=False)
                print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        
        # Get VAE: optionally use the teacher encoder + pruned LightVAE decoder.
        if use_lightvae:
            if not lightvae_path or not os.path.isfile(lightvae_path):
                raise FileNotFoundError(f"LightVAE checkpoint not found: {lightvae_path!r}")
            try:
                from videox_fun.models.wan_vae_first_encoder_lightvae_decoder import AutoencoderKLWan3_8Stream
            except ImportError as exc:
                raise ImportError(
                    "LightVAE requires videox_fun/models/wan_vae_first_encoder_lightvae_decoder.py "
                    "and its dependencies in your local VideoX-Fun checkout."
                ) from exc
            print(f"[TI2V] Loading LightVAE decoder: {lightvae_path}", flush=True)
            vae = AutoencoderKLWan3_8Stream.from_teacher_and_lightvae(
                teacher_vae_path=os.path.join(model_name, "Wan2.2_VAE.pth"),
                lightvae_decoder_path=lightvae_path,
                additional_kwargs={
                    "latent_channels": 48,
                    "c_dim": 160,
                    "dim_mult": [1, 2, 4, 4],
                    "temperal_downsample": [False, True, True],
                },
                decoder_pruning_rate=None,
            ).eval().requires_grad_(False).to(dtype=weight_dtype)
        else:
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

        # Get Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )
        
        # Optional precomputed embedding benchmark: skip loading T5 entirely.
        precomputed_path = os.environ.get("SINGLERECON_PROMPT_EMBEDS_PATH")
        if precomputed_path:
            text_encoder = None
            tokenizer = None
        else:
            text_encoder = WanT5EncoderModel.from_pretrained(
                os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
                additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=weight_dtype,
            )
        
        # Get Scheduler
        Chosen_Scheduler = scheduler_dict = {
            "Flow": FlowMatchEulerDiscreteScheduler,
            "Flow_Unipc": FlowUniPCMultistepScheduler,
            "Flow_DPM++": FlowDPMSolverMultistepScheduler,
        }[sampler_name]
        if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
            config['scheduler_kwargs']['shift'] = 1
        scheduler = Chosen_Scheduler(
            **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        
        # Get Pipeline
        pipeline = Wan2_2TI2VPipeline(
            transformer=transformer,
            transformer_2=transformer_2,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )
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
        
        if lora_path is not None:
            if not os.path.isfile(lora_path):
                raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
            print(f"[TI2V] Loading LoRA: {lora_path} (weight={lora_weight})", flush=True)
            pipeline = merge_lora(
                pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype
            )
            if transformer_2 is not None:
                pipeline = merge_lora(
                    pipeline,
                    lora_high_path,
                    lora_high_weight,
                    device=device,
                    dtype=weight_dtype,
                    sub_transformer_name="transformer_2",
                )
        
        
        self.pipeline = pipeline
        self.device = device
        self.boundary = boundary
        self.vae = vae

    @torch.no_grad()
    def infer(self, args):
        """Infer one case from TI2VParams or a compatible argument object."""
        image_path = args.image_path
        output_path = args.output_path
        num_frames = int(args.view_num)
        if self.pipeline is None:
            raise RuntimeError("Call video_generator.load() before inference")
        image_path, output_path = str(image_path), str(output_path)
        precomputed_path = os.environ.get("SINGLERECON_PROMPT_EMBEDS_PATH")
        prompt_file = os.path.join(os.path.dirname(os.path.dirname(image_path)), "prompt.txt")
        if os.path.isfile(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = os.environ.get("SINGLERECON_DEFAULT_PROMPT", "")
        if not prompt and not precomputed_path:
            raise ValueError(f"Missing prompt.txt and SINGLERECON_DEFAULT_PROMPT for {image_path}")
        if (num_frames - 1) % self.vae.config.temporal_compression_ratio:
            raise ValueError("Frame count must match VAE temporal compression")
        with Image.open(image_path) as im:
            w, h = im.size
        scale = target_short_edge / min(w, h)
        width = max(align_to, int(round(w * scale / align_to)) * align_to)
        height = max(align_to, int(round(h * scale / align_to)) * align_to)
        input_video, input_video_mask, _ = get_image_to_video_latent(
            image_path, None, video_length=num_frames, sample_size=[height, width])
        generator = torch.Generator(device=self.device).manual_seed(seed)
        precomputed_kwargs = {}
        if precomputed_path:
            embeddings = torch.load(precomputed_path, map_location="cpu", weights_only=True)
            embedding = embeddings["prompt"]
            if embedding.ndim == 2 and embedding.shape[0] == 1 and embedding.shape[1] == 4096:
                # Pooled [1,4096] vector is NOT a proper T5 token sequence.
                # One-token conversion is strictly for latency benchmarking.
                print("[BENCHMARK WARNING] pooled [1,4096] prompt expanded to one token; generation semantics differ from T5.", flush=True)
                embedding = embedding.unsqueeze(1)
            if embedding.ndim != 3 or embedding.shape[0] != 1 or embedding.shape[-1] != 4096:
                raise ValueError(f"Expected prompt [1,seq,4096], got {tuple(embedding.shape)}")
            precomputed_kwargs["prompt_embeds"] = [embedding[0].to(device=self.device, dtype=weight_dtype)]
        sample = self.pipeline(
            None if precomputed_path else prompt, num_frames=num_frames, negative_prompt=negative_prompt,
            height=height, width=width, generator=generator,
            guidance_scale=guidance_scale, num_inference_steps=int(getattr(args, "num_inference_steps", num_inference_steps)),
            boundary=self.boundary, video=input_video,
            mask_video=input_video_mask, shift=shift, **precomputed_kwargs,
        ).videos
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # In sequence-parallel inference all ranks execute the model, but only
        # rank zero writes the shared video file.
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            save_videos_grid(sample, output_path, fps=int(getattr(args, "fps", 30)))
        del sample
        return output_path


@dataclass
class TI2VParams:
    image_path: str
    output_path: str
    view_num: int = 121
    num_inference_steps: int = 8
    fps: int = 30


# The object is constructed by SparseRecon once, never at module import.

