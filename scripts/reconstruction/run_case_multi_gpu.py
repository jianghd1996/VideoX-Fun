#!/usr/bin/env python3
"""Two-GPU sequence-parallel latency benchmark for the SAME case.

Copy to SingleRecon/run_case_multi_gpu.py and run:
    CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 run_case_multi_gpu.py

Video generation only; requires a nonempty prompt.txt in each case.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from .multistage_pipeline import find_input_image
except ImportError:
    from multistage_pipeline import find_input_image


WORK_DIR = Path("/home/z00566689/dev/mnt/jiang_test/00_1021_test")


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--video-root", type=Path, default=root / "VideoX-Fun")
    parser.add_argument("--lora-path", type=Path, default=root / "Cloud_Models/Wan2.2-TI2V-5B/lora_ckpts_single8/checkpoint-9600.safetensors")
    parser.add_argument("--lightvae-path", type=Path, default=root / "Cloud_Models/transformer_w_skybox_sla/MG-LightVAE.pth")
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    work_dir = WORK_DIR.expanduser().resolve()
    if (work_dir / "image").is_dir():
        cases = [work_dir]
    else:
        cases = [p for p in sorted(work_dir.iterdir())
                 if p.is_dir() and (p / "image").is_dir()
                 and p.name not in {"collect", "test"} and "." not in p.name]
    if not cases:
        raise ValueError(f"No cases found under {work_dir}")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise RuntimeError("Expected two processes: torchrun --nproc_per_node=2")

    os.environ["SINGLERECON_WORLD_SIZE"] = str(world_size)
    os.environ["SINGLERECON_LORA_PATH"] = str(args.lora_path.resolve())
    os.environ["SINGLERECON_LORA_WEIGHT"] = "0.55"
    os.environ["SINGLERECON_USE_LIGHTVAE"] = "1"
    os.environ["SINGLERECON_LIGHTVAE_PATH"] = str(args.lightvae_path.resolve())
    sys.path.insert(0, str(args.video_root.resolve()))

    import torch
    import torch.distributed as dist
    torch.cuda.set_device(local_rank)
    from predict_ti2v import Wan_5b, TI2VParams

    preload_start = time.perf_counter()
    model = Wan_5b()
    torch.cuda.synchronize()
    preload = time.perf_counter() - preload_start
    if not dist.is_initialized():
        raise RuntimeError("Wan sequence parallelism did not initialize torch.distributed")

    try:
        for case in cases:
            prompt_path = case / "prompt.txt"
            if not prompt_path.is_file() or not prompt_path.read_text(encoding="utf-8").strip():
                raise FileNotFoundError(f"Provide a nonempty {prompt_path}")
            image_path = find_input_image(case / "image")

            dist.barrier()
            output = case / "reconstruction" / "generated.mp4"
            if rank == 0:
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists():
                    output.unlink()
            dist.barrier()

            torch.cuda.synchronize()
            start = time.perf_counter()
            model.infer(TI2VParams(
                image_path=str(image_path), output_path=str(output),
                view_num=args.frames, num_inference_steps=args.steps, fps=args.fps,
            ))
            torch.cuda.synchronize()
            dist.barrier()
            elapsed = time.perf_counter() - start
            durations = [None] * world_size
            dist.all_gather_object(durations, elapsed)
            if rank == 0:
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError(f"Video was not written: {output}")
                result = {
                    "mode": "same_case_sequence_parallel",
                    "gpus": world_size,
                    "use_lightvae": True,
                    "frames": args.frames,
                    "steps": args.steps,
                    "preload_seconds_rank0": round(preload, 3),
                    "video_seconds_per_rank": [round(v, 3) for v in durations],
                    "video_wall_seconds": round(max(durations), 3),
                    "output": str(output),
                }
                report = output.parent / "multi_gpu_timing.json"
                report.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print("[MULTI GPU BENCHMARK] " + json.dumps(result, indent=2), flush=True)
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
