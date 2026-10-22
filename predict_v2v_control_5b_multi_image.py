"""Single-GPU, persistent Wan2.2-Fun-5B-Control adapter for multi-image reconstruction.

Copy to CodexRecon/VideoX-Fun/predict_v2v_control_5b_multi_image.py.
Called by MultiImagePipeline with:
  (control_video, start_image, end_image, output_video, prompt, num_frames, fps)

This is the non-mask-encoder Control model. The END input view is used both as
an end-frame constraint and as ref_image (requires transformer add_ref_conv).
All model weights are loaded only once per adapter instance.
"""
from __future__ import annotations

import os
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from videox_fun.models import AutoencoderKLWan3_8, Wan2_2Transformer3DModel, WanT5EncoderModel
from videox_fun.pipeline.pipeline_wan2_2_fun_control import Wan2_2FunControlPipeline
from videox_fun.utils.utils import (
    filter_kwargs, get_image_latent, get_image_to_video_latent, save_videos_grid,
)

cur_py_path = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(cur_py_path), "Cloud_Models/Wan2.2-Fun-5B-Control/")
CONFIG_PATH = os.path.join(cur_py_path, "config/wan2.2/wan_civitai_5b.yaml")
LIGHTVAE_PATH = os.path.join(
    os.path.dirname(cur_py_path), "Cloud_Models/transformer_w_skybox_sla/MG-LightVAE.pth")
# Wan2.2 VAE teacher uses the same pretrained weights as the single-image TI2V pipeline.
DEFAULT_TEACHER_VAE_PATH = os.path.join(
    os.path.dirname(cur_py_path), "Cloud_Models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，毁容的，静止不动的画面"
)


def _sla_primitives():
    """Load installed TurboDiffusion SLA kernels and quantized operators."""
    sla_root = os.path.join(cur_py_path, "turbodiffusion")
    if not os.path.isdir(sla_root):
        raise FileNotFoundError(
            f"Missing TurboDiffusion root {sla_root}; "
            "expected SLA/, ops/ and rcm/ under this directory.")
    if sla_root not in sys.path:
        sys.path.insert(0, sla_root)
    # This must run BEFORE importing wan_transformer3d_sla, which imports rcm.
    from SLA import SparseLinearAttention, SageSparseLinearAttention
    from ops import FastLayerNorm, FastRMSNorm, Int8Linear
    return {
        "SparseLinearAttention": SparseLinearAttention,
        "SageSparseLinearAttention": SageSparseLinearAttention,
        "FastLayerNorm": FastLayerNorm,
        "FastRMSNorm": FastRMSNorm,
        "Int8Linear": Int8Linear,
    }


def _load_merged_transformer_weights(transformer, path):
    """Load the actual SLA merged checkpoint, not a LoRA adapter."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "transformer"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected state_dict-like transformer checkpoint, got {type(checkpoint)}")
    if checkpoint and all(k.startswith("transformer.") for k in checkpoint):
        checkpoint = {k[len("transformer."):]: v for k, v in checkpoint.items()}
    missing, unexpected = transformer.load_state_dict(checkpoint, strict=False)
    # Verify the actual quantized module parameters were loaded; missing
    # parameters would make the performance/quality comparison invalid.
    param_keys = set(dict(transformer.named_parameters()))
    serious_missing = sorted(set(missing) & param_keys)
    if serious_missing or unexpected:
        raise RuntimeError(
            f"Unexpected merged SLA checkpoint structure: "
            f"missing_parameters={serious_missing[:12]} ({len(serious_missing)}), "
            f"unexpected_keys={unexpected[:12]} ({len(unexpected)})")
    print(f"[MULTI SLA] merged transformer loaded: {path} "
          f"(tensors={len(checkpoint)}, missing_buffers={len(missing)})", flush=True)


def _video_rgb(path, expected_frames: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open GS control video: {path}")
    frames = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) != expected_frames:
        raise ValueError(
            f"GS control frame mismatch: {path}: expected {expected_frames}, got {len(frames)}"
        )
    return frames


def _target_resolution(width: int, height: int, short_edge: int):
    # Wan VAE needs aligned spatial dimensions; keep original orientation.
    scale = short_edge / min(width, height)
    new_w = max(32, int(round(width * scale / 32)) * 32)
    new_h = max(32, int(round(height * scale / 32)) * 32)
    return new_h, new_w


class WanControl5B:
    """Load the Control pipeline once; __call__ generates one trajectory video."""

    def __init__(self):
        self.model_name = Path(os.environ.get(
            "SINGLERECON_CONTROL_MODEL_PATH", MODEL_DIR)).expanduser().resolve()
        self.config_path = Path(os.environ.get(
            "SINGLERECON_CONTROL_CONFIG_PATH", CONFIG_PATH)).expanduser().resolve()
        # LoRA must be a checkpoint file, not the model directory. Explicitly
        # configuring it avoids silently choosing the wrong training iteration.
        self.lora_path = os.environ.get("SINGLERECON_CONTROL_LORA_PATH") or None
        self.lora_weight = float(os.environ.get("SINGLERECON_CONTROL_LORA_WEIGHT", "0.55"))
        self.short_edge = int(os.environ.get("SINGLERECON_CONTROL_SHORT_EDGE", "1088"))
        self.steps = int(os.environ.get("SINGLERECON_CONTROL_INFER_STEPS",
                                        os.environ.get("SINGLERECON_CONTROL_STEPS", "8")))
        self.sla_topk = float(os.environ.get("SINGLERECON_CONTROL_SLA_TOPK", "1.0"))
        self.transformer_path = os.environ.get("SINGLERECON_CONTROL_TRANSFORMER_PATH") or None
        if not 0.0 < self.sla_topk <= 1.0:
            raise ValueError(f"Invalid SLA topk={self.sla_topk}")
        if self.transformer_path and self.lora_path:
            raise ValueError("Merged Transformer checkpoint must not be mixed with LoRA")
        self.cfg = float(os.environ.get("SINGLERECON_CONTROL_CFG", "0.0"))
        self.seed = int(os.environ.get("SINGLERECON_CONTROL_SEED", "42"))
        self.use_lightvae = os.environ.get("SINGLERECON_CONTROL_USE_LIGHTVAE", "1") == "1"
        self.lightvae_path = Path(os.environ.get(
            "SINGLERECON_CONTROL_LIGHTVAE_PATH", LIGHTVAE_PATH)).expanduser().resolve()
        self.teacher_vae_path = Path(os.environ.get(
            "SINGLERECON_CONTROL_TEACHER_VAE_PATH", DEFAULT_TEACHER_VAE_PATH)).expanduser().resolve()
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.pipeline = None
        self.boundary = None
        self.last_timing = {}
        self.load_seconds = 0.0
        self.load()

    def load(self):
        if self.pipeline is not None:
            return
        load_started = time.perf_counter()
        if not torch.cuda.is_available():
            raise RuntimeError("Wan2.2-Fun-5B-Control requires a CUDA GPU")
        if not self.model_name.is_dir():
            raise FileNotFoundError(f"Missing Control model directory: {self.model_name}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Missing Wan config: {self.config_path}")
        config = OmegaConf.load(str(self.config_path))
        self.boundary = float(config["transformer_additional_kwargs"].get("boundary", 0.875))
        model_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
        if self.sla_topk <= 0.9:
            # Initialize TurboDiffusion import path before SLA Transformer
            # import: wan_transformer3d_sla depends on rcm.utils.a2a_cp.
            sla_ops = _sla_primitives()
            from videox_fun.models.wan_transformer3d_sla import Wan2_2Transformer3DModel_SLA
            transformer_class = Wan2_2Transformer3DModel_SLA
        else:
            transformer_class = Wan2_2Transformer3DModel
        transformer = transformer_class.from_pretrained(
            str(self.model_name / model_kwargs.get("transformer_low_noise_model_subpath", "transformer")),
            transformer_additional_kwargs=model_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
        )
        if not transformer.config.get("add_ref_conv", False):
            raise RuntimeError(
                "Control transformer add_ref_conv=False; tail-frame ref_image is not supported "
                "by this checkpoint/config."
            )
        if model_kwargs.get("transformer_combination_type", "single") == "moe":
            raise RuntimeError("Expected single-transformer Wan2.2-Fun-5B-Control model")

        if self.sla_topk <= 0.9:
            # Substitute attention and accelerated quantized linear/norm
            # modules only after the merged tensor state has been loaded.
            from videox_fun.models.wan_transformer3d_sla import (
                WanSelfAttention, WanRMSNorm, WanLayerNorm)
            ops = sla_ops
            changed_attention = 0
            for module in transformer.modules():
                if type(module) is WanSelfAttention:
                    module.attn_op.local_attn = ops["SparseLinearAttention"](
                        head_dim=module.dim // module.num_heads,
                        topk=self.sla_topk, BLKQ=128, BLKK=64)
                    changed_attention += 1
            if changed_attention == 0:
                raise RuntimeError("SLA model had no WanSelfAttention modules")
            replacements = {}
            for name, module in transformer.blocks.named_modules():
                if not name:
                    continue
                if isinstance(module, torch.nn.Linear) and "proj_l" not in name:
                    replacements[name] = ops["Int8Linear"].from_linear(module, True)
                elif isinstance(module, WanRMSNorm):
                    replacements[name] = ops["FastRMSNorm"].from_rmsnorm(module)
                elif isinstance(module, WanLayerNorm):
                    replacements[name] = ops["FastLayerNorm"].from_layernorm(module)
            for name, replacement in replacements.items():
                parent = transformer.blocks
                parts = name.split(".")
                for key in parts[:-1]:
                    parent = parent[int(key)] if key.isdigit() else getattr(parent, key)
                setattr(parent, parts[-1], replacement)
            if not replacements:
                raise RuntimeError("SLA quantization replaced no modules")
            print(f"[MULTI SLA] topk={self.sla_topk} attention={changed_attention} "
                  f"linear/norm={len(replacements)}", flush=True)

        # The merged SLA checkpoint contains int8_weight/scale tensors and
        # attention local_attn.proj_l. These keys exist only AFTER replacing
        # attention and Linear/Norm modules. Loading beforehand silently
        # mismatches ~300 fp weight keys and ~660 quantized keys.
        if self.transformer_path:
            merged = Path(self.transformer_path).expanduser().resolve()
            if not merged.is_file():
                raise FileNotFoundError(f"Merged SLA transformer not found: {merged}")
            _load_merged_transformer_weights(transformer, merged)

        if self.use_lightvae:
            if not self.lightvae_path.is_file():
                raise FileNotFoundError(f"LightVAE decoder checkpoint missing: {self.lightvae_path}")
            if not self.teacher_vae_path.is_file():
                # Some Control model layouts place the teacher checkpoint in their own directory.
                candidate = self.model_name / "Wan2.2_VAE.pth"
                if candidate.is_file():
                    self.teacher_vae_path = candidate
                else:
                    raise FileNotFoundError(
                        f"Teacher VAE checkpoint missing: {self.teacher_vae_path}; "
                        "set SINGLERECON_CONTROL_TEACHER_VAE_PATH to an existing Wan2.2_VAE.pth")
            try:
                from videox_fun.models.wan_vae_first_encoder_lightvae_decoder import AutoencoderKLWan3_8Stream
            except ImportError as exc:
                raise ImportError(
                    "LightVAE implementation missing: "
                    "SingleRecon/VideoX-Fun/videox_fun/models/"
                    "wan_vae_first_encoder_lightvae_decoder.py. "
                    "Copy the exact implementation used by the working single-image pipeline, "
                    "or set video_use_lightvae=False."
                ) from exc
            print(f"[MULTI CONTROL] LightVAE decoder={self.lightvae_path} "
                  f"teacher={self.teacher_vae_path}", flush=True)
            vae = AutoencoderKLWan3_8Stream.from_teacher_and_lightvae(
                teacher_vae_path=str(self.teacher_vae_path),
                lightvae_decoder_path=str(self.lightvae_path),
                additional_kwargs={
                    "latent_channels": 48,
                    "c_dim": 160,
                    "dim_mult": [1, 2, 4, 4],
                    "temperal_downsample": [False, True, True],
                },
                decoder_pruning_rate=None,
            ).eval().requires_grad_(False).to(dtype=self.dtype)
        else:
            vae = AutoencoderKLWan3_8.from_pretrained(
                str(self.model_name / config["vae_kwargs"].get("vae_subpath", "vae")),
                additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
            ).to(self.dtype)
        print(f"[MULTI CONTROL] VAE={'LightVAE' if self.use_lightvae else 'teacher'}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_name / config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")))
        text_encoder = WanT5EncoderModel.from_pretrained(
            str(self.model_name / config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True, torch_dtype=self.dtype,
        ).eval()
        scheduler = FlowMatchEulerDiscreteScheduler(**filter_kwargs(
            FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"])))

        pipeline = Wan2_2FunControlPipeline(
            transformer=transformer, transformer_2=None, vae=vae,
            tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler,
        )
        # Single GPU. No ControlMaskEncoder, control_mask, FSDP or sequence parallel.
        pipeline.to(self.device)
        if self.lora_path:
            lora = Path(self.lora_path).expanduser().resolve()
            if not lora.is_file():
                raise FileNotFoundError(f"LoRA must be a .safetensors file: {lora}")
            try:
                from videox_fun.utils.lora_utils_new import merge_lora
            except ImportError:
                from videox_fun.utils.lora_utils import merge_lora
            pipeline = merge_lora(
                pipeline, str(lora), self.lora_weight,
                device=self.device, dtype=self.dtype,
            )
            print(f"[MULTI CONTROL] LoRA={lora} weight={self.lora_weight}", flush=True)
        else:
            print("[MULTI CONTROL] LoRA disabled (set SINGLERECON_CONTROL_LORA_PATH to enable)", flush=True)
        self.pipeline = pipeline
        torch.cuda.synchronize(self.device)
        self.load_seconds = round(time.perf_counter() - load_started, 3)
        print(f"[MULTI CONTROL] loaded {self.model_name} on {self.device}; "
              f"steps={self.steps} sla_topk={self.sla_topk} "
              f"merged_transformer={self.transformer_path or 'none'} "
              f"model_load={self.load_seconds:.3f}s", flush=True)

    @torch.inference_mode()
    def __call__(self, *, control_video, start_image, end_image, output_video,
                 prompt, num_frames, fps=30, sampled_frame_indices=None,
                 sampled_frames_dir=None, save_mp4=True):
        if self.pipeline is None:
            self.load()
        num_frames = int(num_frames)
        if (num_frames - 1) % 4:
            raise ValueError(f"Wan 5B requires 4n+1 frames, received {num_frames}")
        # Empty positive prompt is intentional for the text-free SLA workflow.
        prompt = "" if prompt is None else str(prompt)
        torch.cuda.synchronize(self.device)
        segment_started = time.perf_counter()
        frames = _video_rgb(control_video, num_frames)
        height, width = _target_resolution(frames[0].shape[1], frames[0].shape[0], self.short_edge)
        resized = np.stack([
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            for frame in frames], axis=0)
        # Control video: RGB, 0..1, [B,C,T,H,W].
        input_video = torch.from_numpy(resized.copy()).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        with Image.open(start_image) as image:
            first = image.convert("RGB").resize((width, height))
        with Image.open(end_image) as image:
            last = image.convert("RGB").resize((width, height))
        # Constrain BOTH actual DA3 input endpoint images, not GS-rendered endpoints.
        inpaint_video, inpaint_mask, _ = get_image_to_video_latent(
            [first], [last], video_length=num_frames, sample_size=[height, width])
        # Use the end reference image as requested, independently from end constraint.
        ref_image = get_image_latent(last, sample_size=[height, width])
        random_gen = torch.Generator(device=self.device).manual_seed(self.seed)
        torch.cuda.synchronize(self.device)
        preprocess_seconds = time.perf_counter() - segment_started
        inference_started = time.perf_counter()

        print(f"[MULTI CONTROL] {Path(control_video).name} {num_frames} frames "
              f"{width}x{height} steps={self.steps} cfg={self.cfg}, "
              f"start={Path(start_image).name}, end/ref={Path(end_image).name}",
              flush=True)
        video = self.pipeline(
            str(prompt), num_frames=num_frames, negative_prompt=NEGATIVE_PROMPT,
            height=height, width=width,
            generator=random_gen, guidance_scale=self.cfg,
            num_inference_steps=self.steps,
            video=inpaint_video, mask_video=inpaint_mask,
            control_video=input_video, ref_image=ref_image,
            boundary=self.boundary,
        ).videos
        torch.cuda.synchronize(self.device)
        inference_seconds = time.perf_counter() - inference_started
        save_started = time.perf_counter()
        destination = Path(output_video)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sampled_frame_indices = sorted(set(int(i) for i in (sampled_frame_indices or [])))
        if any(i < 0 or i >= num_frames for i in sampled_frame_indices):
            raise ValueError(f"Sample frame outside video length {num_frames}: {sampled_frame_indices}")
        jpg_started = time.perf_counter()
        if sampled_frame_indices:
            frame_dir = Path(sampled_frames_dir or destination.parent / "sampled_frames")
            frame_dir.mkdir(parents=True, exist_ok=True)
            if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[0] != 1:
                raise ValueError(f"Expected Wan output tensor [1,C,T,H,W], got {type(video)} "
                                 f"{getattr(video, 'shape', None)}")
            # Wan save_videos_grid uses x * 255 then uint8 for output. Match
            # its RGB image convention but avoid MP4 lossy encode/decode.
            for i in sampled_frame_indices:
                rgb = video[0, :, i].permute(1, 2, 0).detach().float().cpu()
                rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8).numpy()
                target = frame_dir / f"f{i:04d}.jpg"
                if not cv2.imwrite(str(target), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise IOError(f"Cannot save direct Wan JPG {target}")
        jpg_seconds = time.perf_counter() - jpg_started
        mp4_started = time.perf_counter()
        if save_mp4:
            save_videos_grid(video, str(destination), fps=int(fps))
        mp4_seconds = time.perf_counter() - mp4_started if save_mp4 else 0.0
        del video
        torch.cuda.synchronize(self.device)
        save_seconds = time.perf_counter() - save_started
        self.last_timing = {
            "preprocess": round(preprocess_seconds, 3),
            "inference": round(inference_seconds, 3),
            "save": round(save_seconds, 3),
            "save_jpg": round(jpg_seconds, 3),
            "save_mp4": round(mp4_seconds, 3),
            "total": round(time.perf_counter() - segment_started, 3),
        }
        print(f"[MULTI VIDEO TIMING] {destination.parent.name}: "
              + " ".join(f"{key}={value:.3f}s" for key, value in self.last_timing.items()),
              flush=True)
        if save_mp4 and (not destination.is_file() or destination.stat().st_size == 0):
            raise RuntimeError(f"Video was not written: {destination}")
        return str(destination)
