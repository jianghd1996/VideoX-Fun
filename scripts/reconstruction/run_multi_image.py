#!/usr/bin/env python3
"""Run custom DA3 -> dense 3DGS on one or more multi-image cases.

Edit CONFIG and WORK_DIR below, or pass --config / --work-dir.
"""
from __future__ import annotations

import argparse
import json
import time
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_image_pipeline import MultiImagePipeline

CONFIG = {
    "external_code_root": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/DA3_custom",
    "dense_code_root": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon",
    "da3_model_path": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/Cloud_Models/Saliency-Face-DA3/besthumanmodel_lr1e_6_ckpt2.pt",
    "run_3dgs_iter": True,  # full pipeline: generated videos -> COLMAP -> Reconstruction_full
    "dense_generated_frames": 80,  # globally sample 80 novel camera poses across all trajectories
    "resume_stages": False,
    "crop_ratio": 1.0,
    "mode": "scene",
    "circle": False,
    "max_frames": 121,
    "render_trajectories": True,
    "trajectory_extra": False,
    "trajectory_view_num": 81,
    "trajectory_interpolation": "bezier",  # Keep fixed until E2E pipeline works
    "run_gen_video": True,
    "run_captioner": True,
    "captioner_code_root": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon",
    "captioner_model_path": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/Cloud_Models/Qwen3-VL-8B-Instruct",
    "captioner_cpu_swap": True,
    "captioner_num_gpus": 1,
    "video_adapter": "predict_v2v_control_5b_multi_image:WanControl5B",
    "video_code_root": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/VideoX-Fun",
    "video_lora_path": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/Cloud_Models/Wan2.2-Fun-5B-Control/12000_lora.safetensors",
    "video_lora_weight": 0.55,
    "video_cfg": 0.0,
    "video_short_edge": 1088,
    # Fast LightVAE decode, same encoder+decoder combination as single-image TI2V.
    "video_use_lightvae": True,  # False restores original full VAE
    "video_lightvae_path": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/Cloud_Models/transformer_w_skybox_sla/MG-LightVAE.pth",
    "video_teacher_vae_path": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/Cloud_Models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    "video_fps": 30,
    # Automatically dispatch: one ID -> sequential, multiple -> parallel workers.
    # GPU IDs are PHYSICAL CUDA device IDs; 4 trajectories + 4 GPUs => one each.
    # With one GPU, use [0] and select physical GPU via CUDA_VISIBLE_DEVICES=4.
    "video_gpu_ids": [4, 5, 6, 7],
    "video_resume": True,  # reuse previously generated MP4s when testing dense stage
}
WORK_DIR = "/home/z00566689/dev/mnt/jiang_test/00_1022_test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(WORK_DIR))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--da3-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    opts = dict(CONFIG)
    if args.config:
        opts.update(json.loads(args.config.read_text(encoding="utf-8")))
    if args.da3_only:
        opts["run_3dgs_iter"] = False
    if args.resume:
        opts["resume_stages"] = True
    root = args.work_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    cases = ([root] if (root / "image").is_dir() else
             sorted(p for p in root.iterdir() if p.is_dir() and (p / "image").is_dir()))
    if not cases:
        raise ValueError(f"No case/image folders found in {root}")
    wall_started = time.perf_counter()
    pipeline = MultiImagePipeline(**opts)
    print(f"[MULTI PRELOAD] initial={pipeline.load_seconds:.3f}s", flush=True)
    summary = {}
    for case in cases:
        print(f"[MULTI CASE] {case}", flush=True)
        try:
            result = pipeline(case)
            summary[case.name] = {"status": "ok", **result["timings"]}
        except Exception as exc:
            summary[case.name] = {"status": "failed", "error": str(exc)}
            traceback.print_exc()
        (root / "multi_image_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[MULTI END-TO-END TIMING SUMMARY] (seconds)", flush=True)
    keys = ("image_preprocess", "caption", "da3", "video_model_load",
            "video_generation", "video_inference", "trajectory_video",
            "dense_prepare", "dense", "total")
    print(f"{'Case':<35} {'Status':<8} " + " ".join(f"{key:>19}" for key in keys), flush=True)
    for name, record in summary.items():
        print(f"{name[:35]:<35} {record['status']:<8} "
              + " ".join(f"{record[key]:>18.3f}s" if key in record else f"{'-':>19}"
                         for key in keys), flush=True)
    print(f"[MULTI ALL CASES] wall={time.perf_counter() - wall_started:.3f}s "
          f"initial_model_preload={pipeline.load_seconds:.3f}s", flush=True)
    if any(row["status"] == "failed" for row in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
