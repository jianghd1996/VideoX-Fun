import os
import sys

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf


current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    AutoTokenizer,
    Wan2_2Transformer3DModel,
    WanT5EncoderModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2FunInpaintPipeline
from videox_fun.utils.utils import filter_kwargs, save_videos_grid


# =============================================================================
# User settings
# =============================================================================
model_name = "/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-InP"

input_video_path = "/mnt/DataPart/jianghongda/related_work/LanPaint/gs_render.mp4"
input_mask_path = "/mnt/DataPart/jianghongda/related_work/LanPaint/mask.mp4"

save_path = "samples/wan2.2-fun-5b-video-inpaint"

# Wan2.2-Fun-5B-InP is trained with 121 frames at 24 FPS. Only the first
# `video_length` decoded frames from both files are used.
video_length = 121
fps = 24
# Automatically use 1280x704 for landscape videos and 704x1280 for portrait videos.
# Set this to [height, width] to force a specific resolution instead.
sample_size = None

# Mask convention for this script:
#   True:  black pixels are generated, non-black pixels are preserved.
#   False: white pixels are generated, black pixels are preserved.
black_is_inpaint = True
mask_threshold = 128

prompt = (
    "Complete the missing background naturally and consistently across the video. "
    "Keep the visible subject, geometry, lighting, texture, and camera motion coherent."
)
negative_prompt = (
    "flicker, temporal inconsistency, color shift, distorted geometry, blurry, low quality, "
    "artifacts, stripes, duplicated objects, text, watermark"
)

guidance_scale = 6.0
num_inference_steps = 30
seed = 43
weight_dtype = torch.bfloat16

# This test script intentionally uses full model loading (no CPU/model offload).
enable_teacache = True
teacache_threshold = 0.10
num_skip_start_steps = 5
teacache_offload = False

sampler_name = "Flow"  # This standalone script currently uses Flow.
shift = 5
config_path = "config/wan2.2/wan_civitai_5b.yaml"


def validate_settings(target_sample_size):
    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video does not exist: {input_video_path}")
    if not os.path.isfile(input_mask_path):
        raise FileNotFoundError(f"Input mask video does not exist: {input_mask_path}")
    if not os.path.isdir(model_name):
        raise FileNotFoundError(f"Model directory does not exist: {model_name}")
    if video_length != 1 and (video_length - 1) % 4 != 0:
        raise ValueError("video_length must satisfy 4n+1 for the Wan VAE, for example 121.")
    if target_sample_size[0] % 16 != 0 or target_sample_size[1] % 16 != 0:
        raise ValueError(
            f"sample_size must be divisible by 16, got {target_sample_size}."
        )


def resolve_sample_size(video_path):
    if sample_size is not None:
        return list(sample_size)

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open input video: {video_path}")
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Failed to read input video dimensions: width={width}, height={height}."
        )

    target_sample_size = [1280, 704] if height > width else [704, 1280]
    print(
        f"Input resolution: {width}x{height}; selected inference resolution: "
        f"{target_sample_size[1]}x{target_sample_size[0]}"
    )
    return target_sample_size


def load_video_and_mask(video_path, mask_path, length, size):
    """Decode paired frames without FPS resampling and convert them to pipeline tensors."""
    height, width = size
    video_capture = cv2.VideoCapture(video_path)
    mask_capture = cv2.VideoCapture(mask_path)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open input video: {video_path}")
    if not mask_capture.isOpened():
        video_capture.release()
        raise RuntimeError(f"Failed to open input mask video: {mask_path}")

    source_fps = video_capture.get(cv2.CAP_PROP_FPS)
    mask_fps = mask_capture.get(cv2.CAP_PROP_FPS)
    source_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    mask_frame_count = int(mask_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"Input video: frames={source_frame_count}, fps={source_fps:.3f}; "
        f"mask: frames={mask_frame_count}, fps={mask_fps:.3f}"
    )
    if source_fps > 0 and mask_fps > 0 and abs(source_fps - mask_fps) > 1e-3:
        print(
            "Warning: video and mask FPS differ. Frames are paired by decoded frame index; "
            "no temporal resampling is performed."
        )

    video_frames = []
    mask_frames = []
    try:
        for frame_index in range(length):
            video_ok, video_frame = video_capture.read()
            mask_ok, mask_frame = mask_capture.read()
            if not video_ok or not mask_ok:
                raise ValueError(
                    f"Both files must contain at least {length} decodable frames. "
                    f"Stopped at frame {frame_index}: video_ok={video_ok}, mask_ok={mask_ok}."
                )

            video_frame = cv2.resize(video_frame, (width, height), interpolation=cv2.INTER_AREA)
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

            mask_frame = cv2.resize(mask_frame, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            if black_is_inpaint:
                # The pipeline expects 1/white for the region to generate.
                mask_frame = mask_frame < mask_threshold
            else:
                mask_frame = mask_frame >= mask_threshold

            video_frames.append(video_frame)
            mask_frames.append(mask_frame.astype(np.float32))
    finally:
        video_capture.release()
        mask_capture.release()

    video = torch.from_numpy(np.stack(video_frames)).permute(3, 0, 1, 2).unsqueeze(0)
    video = video.to(dtype=torch.float32) / 255.0
    mask = torch.from_numpy(np.stack(mask_frames)).unsqueeze(0).unsqueeze(0)
    mask = mask.to(dtype=torch.float32)

    masked_ratio = mask.mean().item()
    print(
        f"Using the first {length} frames at {width}x{height}; "
        f"inpaint area={masked_ratio * 100:.2f}%"
    )
    if masked_ratio == 0:
        raise ValueError(
            "The processed mask contains no inpaint pixels. Check black_is_inpaint and mask_threshold."
        )
    if masked_ratio == 1:
        print("Warning: the processed mask marks the entire video for generation.")
    return video, mask


def load_pipeline(device):
    config = OmegaConf.load(config_path)
    transformer_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    vae_kwargs = OmegaConf.to_container(config["vae_kwargs"])
    text_encoder_kwargs = OmegaConf.to_container(config["text_encoder_kwargs"])

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            model_name,
            config["transformer_additional_kwargs"].get(
                "transformer_low_noise_model_subpath", "transformer"
            ),
        ),
        transformer_additional_kwargs=transformer_kwargs,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    if config["transformer_additional_kwargs"].get("transformer_combination_type", "single") == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(
                model_name,
                config["transformer_additional_kwargs"].get(
                    "transformer_high_noise_model_subpath", "transformer"
                ),
            ),
            transformer_additional_kwargs=transformer_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None

    autoencoder_class = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = autoencoder_class.from_pretrained(
        os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=vae_kwargs,
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            model_name,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            model_name,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=text_encoder_kwargs,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    if sampler_name != "Flow":
        raise ValueError(f"Unsupported sampler_name in this script: {sampler_name}")
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(
            FlowMatchEulerDiscreteScheduler,
            OmegaConf.to_container(config["scheduler_kwargs"]),
        )
    )
    pipeline = Wan2_2FunInpaintPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    pipeline.to(device=device)

    if enable_teacache:
        coefficients = get_teacache_coefficients(model_name)
        if coefficients is None:
            print("Warning: TeaCache coefficients were not found; TeaCache is disabled.")
        else:
            pipeline.transformer.enable_teacache(
                coefficients,
                num_inference_steps,
                teacache_threshold,
                num_skip_start_steps=num_skip_start_steps,
                offload=teacache_offload,
            )
            if transformer_2 is not None:
                pipeline.transformer_2.share_teacache(transform=pipeline.transformer)
            print(f"TeaCache enabled with threshold {teacache_threshold}.")

    boundary = config["transformer_additional_kwargs"].get("boundary", 0.900)
    return pipeline, boundary


def save_results(input_video, mask_video, generated_video):
    os.makedirs(save_path, exist_ok=True)
    generated_video = generated_video.cpu().float().clamp(0, 1)
    input_video = input_video.cpu().float().clamp(0, 1)
    mask_video = mask_video.cpu().float().clamp(0, 1)

    # Preserve known pixels exactly in an additional practical output.
    composited_video = generated_video * mask_video + input_video * (1 - mask_video)
    mask_visualization = mask_video.repeat(1, 3, 1, 1, 1)
    comparison_top = torch.cat([input_video, mask_visualization], dim=4)
    comparison_bottom = torch.cat([generated_video, composited_video], dim=4)
    comparison = torch.cat([comparison_top, comparison_bottom], dim=3)

    generated_path = os.path.join(save_path, "generated_raw.mp4")
    composited_path = os.path.join(save_path, "generated_composited.mp4")
    comparison_path = os.path.join(save_path, "comparison_input_mask_raw_composited.mp4")
    save_videos_grid(generated_video, generated_path, fps=fps)
    save_videos_grid(composited_video, composited_path, fps=fps)
    save_videos_grid(comparison, comparison_path, fps=fps)
    print(f"Saved raw model output to: {generated_path}")
    print(f"Saved exact-known-region composite to: {composited_path}")
    print(f"Saved comparison video to: {comparison_path}")


def main():
    target_sample_size = resolve_sample_size(input_video_path)
    validate_settings(target_sample_size)
    device = set_multi_gpus_devices(1, 1)
    input_video, input_mask = load_video_and_mask(
        input_video_path,
        input_mask_path,
        video_length,
        target_sample_size,
    )
    pipeline, boundary = load_pipeline(device)
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        generated = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=target_sample_size[0],
            width=target_sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            boundary=boundary,
            video=input_video,
            mask_video=input_mask,
            shift=shift,
        ).videos

    save_results(input_video, input_mask, generated)


if __name__ == "__main__":
    main()
