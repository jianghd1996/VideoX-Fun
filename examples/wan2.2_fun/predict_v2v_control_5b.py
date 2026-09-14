#!/usr/bin/env python
# coding=utf-8
"""Batch inference for the mask-aware Wan2.2-Fun 5B Control LoRA.

Each child directory of --cases_dir must contain:
  image.jpg, prompt.txt, gs_render.mp4

The script creates gs_render_mask.mp4, runs every consecutive 81-frame
control segment, saves each generated segment, and concatenates them.
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file
from PIL import Image
from transformers import AutoTokenizer

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
    Wan2_2Transformer3DModel,
    WanT5EncoderModel,
)
from videox_fun.pipeline import Wan2_2FunControlPipeline
from videox_fun.utils.utils import (
    filter_kwargs,
    get_image_latent,
    get_image_to_video_latent,
    get_video_to_video_latent,
    save_videos_grid,
)


DEFAULT_MODEL = (
    "/mnt/DataPart/jianghongda/VideoX-Fun/models/"
    "Diffusion_Transformer/Wan2.2-Fun-5B-Control"
)
DEFAULT_CHECKPOINT = (
    "/mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun-mask/"
    "output_dir_wan2.2_fun_control_lora/checkpoint-20000.safetensors"
)
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases_dir",
        type=Path,
        required=True,
        help="Root containing one subdirectory per inference case.",
    )
    parser.add_argument("--model_path", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument(
        "--checkpoint_path", type=Path, default=Path(DEFAULT_CHECKPOINT)
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("config/wan2.2/wan_civitai_5b.yaml"),
    )
    parser.add_argument("--frames_per_segment", type=int, default=81)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help=(
            "Parent output directory. Default: "
            "<VideoX-Fun>/inference_results."
        ),
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=[6.0, 0.0],
        help="CFG values generated for every case (default: 6.0 0.0).",
    )
    parser.add_argument(
        "--lora_weight",
        type=float,
        default=0.55,
        help="Extra multiplier applied to the loaded PEFT LoRA output.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--mask_crf", type=int, default=18)
    parser.add_argument("--mask_preset", default="medium")
    parser.add_argument("--ffmpeg_threads", type=int, default=4)
    parser.add_argument("--overwrite_mask", action="store_true")
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--fps", type=float, default=None)
    return parser.parse_args()


def make_mask(
    video: Path,
    destination: Path,
    threshold: int,
    crf: int,
    preset: str,
    ffmpeg_threads: int,
    overwrite: bool,
) -> tuple[str, Path, float, str]:
    if destination.exists() and not overwrite:
        return "skipped", video, 0.0, "output exists"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + f".tmp-{os.getpid()}-{threading.get_ident()}.mp4"
    )
    graph = (
        "[0:v]format=gbrp,extractplanes=r+g+b[r][g][b];"
        "[r][g]blend=all_expr='max(A,B)'[rg];"
        f"[rg][b]blend=all_expr='if(lte(max(A,B),{threshold}),0,255)',"
        "format=yuv420p[out]"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-threads", str(ffmpeg_threads), "-i", str(video),
        "-filter_complex", graph, "-map", "[out]", "-an",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            temporary.unlink(missing_ok=True)
            return (
                "failed", video, time.monotonic() - started,
                result.stderr.strip(),
            )
        os.replace(temporary, destination)
        return "done", video, time.monotonic() - started, str(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        return "failed", video, time.monotonic() - started, str(exc)


def expand_patch_embedding(transformer):
    """Append four mask channels after all pretrained input channels."""
    old = transformer.patch_embedding
    old_channels = old.in_channels
    expanded = torch.nn.Conv3d(
        old_channels + 4,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, :old_channels].copy_(old.weight)
        if old.bias is not None:
            expanded.bias.copy_(old.bias)

    if not torch.equal(expanded.weight[:, :old_channels], old.weight):
        raise RuntimeError("Pretrained Patchify channel prefix changed.")
    if torch.count_nonzero(expanded.weight[:, old_channels:]).item() != 0:
        raise RuntimeError("New Patchify mask channels are not zero initialized.")

    transformer.patch_embedding = expanded
    transformer.in_dim = old_channels + 4
    return old_channels


def load_mask_aware_checkpoint(
    transformer, checkpoint_path: Path, lora_weight: float
):
    old_channels = expand_patch_embedding(transformer)
    state_dict = load_file(str(checkpoint_path), device="cpu")

    patch_state = {
        key.replace("patch_embedding.", ""): state_dict.pop(key)
        for key in list(state_dict.keys())
        if key.startswith("patch_embedding.")
    }
    if set(patch_state) != {"weight", "bias"}:
        raise RuntimeError(
            "Checkpoint must contain patch_embedding.weight and "
            f"patch_embedding.bias; got {sorted(patch_state)}"
        )
    expected_shape = transformer.patch_embedding.weight.shape
    if patch_state["weight"].shape != expected_shape:
        raise RuntimeError(
            "Patchify checkpoint shape mismatch: "
            f"checkpoint={tuple(patch_state['weight'].shape)}, "
            f"model={tuple(expected_shape)}"
        )
    transformer.patch_embedding.load_state_dict(patch_state, strict=True)

    lora_keys = [key for key in state_dict if "lora_" in key]
    if not lora_keys:
        raise RuntimeError("No PEFT LoRA weights found in checkpoint.")
    result = set_peft_model_state_dict(
        transformer, state_dict, adapter_name="default"
    )

    # Match merge_lora(..., multiplier=lora_weight) from the original example.
    scaled_layers = 0
    for module in transformer.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and "default" in scaling:
            scaling["default"] *= lora_weight
            scaled_layers += 1
    if scaled_layers == 0:
        raise RuntimeError("No active PEFT LoRA layers were found to scale.")

    unexpected = list(getattr(result, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint keys while loading LoRA: "
            + ", ".join(unexpected[:20])
        )

    print(
        f"Loaded {checkpoint_path}: Patchify {old_channels} -> "
        f"{old_channels + 4}, LoRA tensors={len(lora_keys)}, "
        f"LoRA weight={lora_weight}, scaled layers={scaled_layers}"
    )


def read_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return frame_count, width, height, fps


def select_sample_size(source_width: int, source_height: int):
    """Return [height, width] using the requested aspect-ratio buckets."""
    img_size = [source_width, source_height]
    ratio = min(img_size) / max(img_size)

    if img_size[0] <= img_size[1]:  # portrait: width <= height
        if ratio > 0.7:
            sample_size = [1440, 1088]
        elif ratio > 0.65:
            sample_size = [1600, 1088]
        else:
            sample_size = [1920, 1088]
    else:  # landscape
        if ratio > 0.7:
            sample_size = [1088, 1440]
        elif ratio > 0.65:
            sample_size = [1088, 1600]
        else:
            sample_size = [1088, 1920]

    if img_size[0] == img_size[1]:
        sample_size = [1280, 1280]

    return sample_size


def iter_aligned_segments(
    control_path: Path, mask_path: Path, frames_per_segment: int
):
    control_cap = cv2.VideoCapture(str(control_path))
    mask_cap = cv2.VideoCapture(str(mask_path))
    segment_index = 0
    try:
        while True:
            control_frames = []
            mask_frames = []
            for _ in range(frames_per_segment):
                ok_control, control_bgr = control_cap.read()
                ok_mask, mask_bgr = mask_cap.read()
                if not ok_control and not ok_mask:
                    break
                if ok_control != ok_mask:
                    raise RuntimeError(
                        "Control and mask videos have different frame counts."
                    )
                control_frames.append(
                    cv2.cvtColor(control_bgr, cv2.COLOR_BGR2RGB)
                )
                mask_frames.append(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))

            if not control_frames:
                return
            if len(control_frames) != frames_per_segment:
                raise RuntimeError(
                    f"Trailing segment has {len(control_frames)} frames; "
                    f"expected exactly {frames_per_segment}."
                )
            yield segment_index, control_frames, mask_frames
            segment_index += 1
    finally:
        control_cap.release()
        mask_cap.release()


def ffconcat_escape(path: Path):
    return str(path.resolve()).replace("'", "'\\''")


def concatenate_segments(segment_paths, destination: Path, overwrite: bool):
    if destination.exists() and not overwrite:
        print(f"Skip existing output: {destination}")
        return
    concat_file = destination.with_suffix(".concat.txt")
    concat_file.write_text(
        "".join(f"file '{ffconcat_escape(path)}'\n" for path in segment_paths),
        encoding="utf-8",
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", "-movflags", "+faststart", str(destination),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    concat_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to concatenate segments: {result.stderr.strip()}"
        )


def build_pipeline(args, device):
    config = OmegaConf.load(args.config_path)
    model_path = str(args.model_path)
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            model_path,
            config["transformer_additional_kwargs"].get(
                "transformer_low_noise_model_subpath", "transformer"
            ),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q", "k", "v", "ffn.0", "ffn.2"],
    )
    transformer = inject_adapter_in_model(lora_config, transformer)
    load_mask_aware_checkpoint(
        transformer, args.checkpoint_path, args.lora_weight
    )
    transformer.eval()

    vae_cls = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = vae_cls.from_pretrained(
        os.path.join(
            model_path, config["vae_kwargs"].get("vae_subpath", "vae")
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            model_path,
            config["text_encoder_kwargs"].get(
                "tokenizer_subpath", "tokenizer"
            ),
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            model_path,
            config["text_encoder_kwargs"].get(
                "text_encoder_subpath", "text_encoder"
            ),
        ),
        additional_kwargs=OmegaConf.to_container(
            config["text_encoder_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(
            FlowMatchEulerDiscreteScheduler,
            OmegaConf.to_container(config["scheduler_kwargs"]),
        )
    )
    pipeline = Wan2_2FunControlPipeline(
        transformer=transformer,
        transformer_2=None,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    # Full-load mode: keep all modules resident on the selected GPU.
    pipeline.to(device=device)
    return pipeline, vae, config


def run_case(
    case_dir: Path,
    case_index: int,
    args,
    pipeline,
    vae,
    config,
    device,
    run_dir: Path,
):
    image_path = case_dir / "image.jpg"
    prompt_path = case_dir / "prompt.txt"
    control_path = case_dir / "gs_render.mp4"
    for required in (image_path, prompt_path, control_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing required case file: {required}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Empty prompt: {prompt_path}")

    mask_path = run_dir / f"{case_dir.name}_mask.mp4"
    status, _, elapsed, detail = make_mask(
        control_path,
        mask_path,
        args.threshold,
        args.mask_crf,
        args.mask_preset,
        args.ffmpeg_threads,
        args.overwrite_mask,
    )
    if status == "failed":
        raise RuntimeError(f"Mask generation failed for {case_dir}: {detail}")
    print(f"[{case_dir.name}] mask {status} ({elapsed:.2f}s): {mask_path}")

    frame_count, source_w, source_h, source_fps = read_video_info(control_path)
    mask_count, _, _, _ = read_video_info(mask_path)
    if frame_count != mask_count:
        raise RuntimeError(
            f"Control/mask frame mismatch: {frame_count} vs {mask_count}"
        )
    if frame_count % args.frames_per_segment != 0:
        raise RuntimeError(
            f"{control_path} has {frame_count} frames; expected a multiple "
            f"of {args.frames_per_segment}."
        )

    sample_size = select_sample_size(source_w, source_h)
    target_h, target_w = sample_size

    # All configured buckets are expected to align with VAE compression and
    # Transformer patches; fail loudly if a future bucket violates this.
    spatial_ratio = vae.config.spatial_compression_ratio
    patch_h = pipeline.transformer.config.patch_size[1]
    patch_w = pipeline.transformer.config.patch_size[2]
    if (
        target_h % (spatial_ratio * patch_h) != 0
        or target_w % (spatial_ratio * patch_w) != 0
    ):
        raise RuntimeError(
            f"Sample size {sample_size} is not aligned with VAE/patch sizes."
        )
    output_fps = args.fps or source_fps
    boundary = config["transformer_additional_kwargs"].get(
        "boundary", 0.900
    )

    image = Image.open(image_path).convert("RGB")
    start_end_video, start_end_mask, _ = get_image_to_video_latent(
        [image],
        [image],
        video_length=args.frames_per_segment,
        sample_size=sample_size,
    )
    ref_image = get_image_latent(str(image_path), sample_size=sample_size)

    def cfg_tag(value):
        return f"cfg{value:.1f}".replace(".", "p").replace("-", "m")

    cfg_variants = [(value, cfg_tag(value)) for value in args.guidance_scales]
    segment_paths_by_cfg = {tag: [] for _, tag in cfg_variants}

    for segment_index, control_frames, mask_frames in iter_aligned_segments(
        control_path, mask_path, args.frames_per_segment
    ):
        control_video, _, _, _ = get_video_to_video_latent(
            control_frames,
            video_length=args.frames_per_segment,
            sample_size=sample_size,
        )
        mask_video, _, _, _ = get_video_to_video_latent(
            mask_frames,
            video_length=args.frames_per_segment,
            sample_size=sample_size,
        )
        control_mask = mask_video.amax(dim=1, keepdim=True)
        segment_seed = args.seed + case_index * 10000 + segment_index

        for guidance_scale, tag in cfg_variants:
            segment_dir = run_dir / "segments" / case_dir.name / tag
            segment_dir.mkdir(parents=True, exist_ok=True)
            segment_path = segment_dir / f"segment_{segment_index:04d}.mp4"
            segment_paths_by_cfg[tag].append(segment_path)
            if segment_path.exists() and not args.overwrite_output:
                print(
                    f"[{case_dir.name}] skip segment {segment_index} {tag}"
                )
                continue

            # Use identical initial noise for the CFG comparison.
            generator = torch.Generator(device=device).manual_seed(
                segment_seed
            )
            print(
                f"[{case_dir.name}] generating segment {segment_index}, "
                f"{tag}, size={target_w}x{target_h}, "
                f"frames={args.frames_per_segment}"
            )
            with torch.no_grad():
                sample = pipeline(
                    prompt,
                    num_frames=args.frames_per_segment,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=target_h,
                    width=target_w,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    num_inference_steps=args.steps,
                    video=start_end_video,
                    mask_video=start_end_mask,
                    control_video=control_video,
                    control_mask=control_mask,
                    ref_image=ref_image,
                    boundary=boundary,
                ).videos
            save_videos_grid(sample, str(segment_path), fps=output_fps)
            del sample
            torch.cuda.empty_cache()

    for _, tag in cfg_variants:
        final_path = run_dir / f"{case_dir.name}_{tag}.mp4"
        concatenate_segments(
            segment_paths_by_cfg[tag], final_path, args.overwrite_output
        )
        print(f"[{case_dir.name}] done: {final_path}")


def main():
    args = parse_args()
    if not args.cases_dir.is_dir():
        raise NotADirectoryError(args.cases_dir)
    if not args.model_path.is_dir():
        raise NotADirectoryError(args.model_path)
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(args.checkpoint_path)

    case_dirs = sorted(
        path for path in args.cases_dir.iterdir() if path.is_dir()
    )
    if not case_dirs:
        raise RuntimeError(f"No case directories found in {args.cases_dir}")

    project_root = Path(__file__).resolve().parents[2]
    output_root = (
        args.output_root
        if args.output_root is not None
        else project_root / "inference_results"
    )
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Inference results: {run_dir}")

    device = set_multi_gpus_devices(1, 1)
    pipeline, vae, config = build_pipeline(args, device)
    failures = []
    for case_index, case_dir in enumerate(case_dirs):
        try:
            run_case(
                case_dir,
                case_index,
                args,
                pipeline,
                vae,
                config,
                device,
                run_dir,
            )
        except Exception as exc:
            failures.append((case_dir.name, str(exc)))
            print(f"[{case_dir.name}] FAILED: {exc}", file=sys.stderr)

    if failures:
        details = "\n".join(f"  {name}: {error}" for name, error in failures)
        raise RuntimeError(f"{len(failures)} case(s) failed:\n{details}")


if __name__ == "__main__":
    main()
