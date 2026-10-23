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
    "force_rerun_dense": False,  # --rerun-dense backs up old output and retrains with PLY
    "crop_ratio": 1.0,
    "mode": "scene",
    "circle": False,
    "max_frames": 121,
    "render_trajectories": True,
    "trajectory_extra": False,
    "trajectory_view_num": 49,  # 49-frame speed/quality ablation; use --trajectory-frames 81 for baseline
    "trajectory_mode": "four_paths",  # switch with --roundtrip-bab-bcb
    "roundtrip_center_index": 0,
    "roundtrip_target_indices": [1, 2],
    "roundtrip_pitch_deg": 10.0,
    "roundtrip_elevation_profile": "broad_sine",  # continuously moving wide arc
    "roundtrip_edge_ramp": 0.2,  # fraction of B->A/B->C travel used for each smooth ramp
    "trajectory_interpolation": "bezier",  # Keep fixed until E2E pipeline works
    "run_gen_video": True,
    "run_captioner": False,  # no caption model; positive prompt defaults to empty
    "captioner_code_root": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon",
    "captioner_model_path": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/Cloud_Models/Qwen3-VL-8B-Instruct",
    "captioner_cpu_swap": True,
    "captioner_num_gpus": 1,
    "video_adapter": "predict_v2v_control_5b_multi_image:WanControl5B",
    "video_code_root": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/VideoX-Fun",
    "video_lora_path": None,
    "video_transformer_path": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/Cloud_Models/transformer_w_skybox_sla/transformer_lora_merge_weight_sla_quant.pt",
    "video_sla_topk": 0.5,
    "video_infer_steps": 6,
    "video_lora_weight": 0.0,
    "video_cfg": 0.0,
    "video_short_edge": 1088,
    # Fast LightVAE decode, same encoder+decoder combination as single-image TI2V.
    "video_use_lightvae": True,  # False restores original full VAE
    "video_lightvae_path": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/Cloud_Models/transformer_w_skybox_sla/MG-LightVAE.pth",
    "video_teacher_vae_path": "/home/z00566689/dev/mnt/jiang_dev/CodexRecon/Cloud_Models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    "video_fps": 30,
    # Automatically dispatch: one ID -> sequential, multiple -> parallel workers.
    # GPU IDs are PHYSICAL CUDA device IDs; 4 trajectories on GPUs 6 and 7.
    # With one GPU, use [0] and select physical GPU via CUDA_VISIBLE_DEVICES=4.
    "video_gpu_ids": [6, 7],
    "video_cpu_threads": 16,  # per Wan worker; compare with 32
    "video_positive_prompt": "",  # do not load prompt.txt
    "video_prompt_embeds_path": None,  # optional: precomputed prompt.pt (skip T5 entirely)
    "video_save_mp4": True,  # also archive complete MP4; selected JPGs always saved from tensor
    "video_resume": False,  # full E2E benchmark: regenerate all videos
    "video_preload_workers": True,  # preload Wan on GPU 6,7 at pipeline startup
}
WORK_DIR = "/home/z00566689/dev/mnt/jiang_test/00_1022_test/cover2_e2e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(WORK_DIR))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--da3-only", action="store_true")
    parser.add_argument("--trajectory-frames", type=int,
                        help="Frames per GS/Wan trajectory (4n+1), default 49 for this ablation")
    parser.add_argument("--roundtrip-bab-bcb", action="store_true",
                        help="Two smooth 81-frame BAB and BCB loops; B is center index")
    parser.add_argument("--roundtrip-center-index", type=int, default=0)
    parser.add_argument("--roundtrip-target-indices", type=str, default="1,2",
                        help="A and C indices in sorted normalized input images")
    parser.add_argument("--roundtrip-pitch-deg", type=float, default=10.0)
    parser.add_argument("--roundtrip-elevation-profile", choices=("broad_sine", "plateau", "sine"),
                        default="broad_sine", help="Smooth wide elevated arc; plateau may look jerky, sine is legacy")
    parser.add_argument("--roundtrip-edge-ramp", type=float, default=0.2,
                        help="Fraction of each leg used to ease into/out of elevation [0.05, 0.5]")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-save-mp4", action="store_true",
                        help="Skip full generated MP4; still save pose-selected JPGs from Wan tensor")
    parser.add_argument("--video-prompt-embeds", type=Path,
                        help="Use a precomputed Wan prompt.pt and skip loading T5 text encoder")
    parser.add_argument("--video-cpu-threads", type=int, choices=(8, 16, 32),
                        help="CPU threads per Wan worker; default 16, try 32 as A/B")
    parser.add_argument("--video-gpus", type=str,
                        help="Physical GPUs for Wan workers, e.g. 6,7 (does not change DA3/dense GPU)")
    parser.add_argument("--benchmark-fresh", action="store_true",
                        help="Force complete fresh stages; refuses existing reconstruction outputs")
    parser.add_argument("--rerun-dense", action="store_true",
                        help="Reuse DA3/videos but back up and redo dense reconstruction")
    args = parser.parse_args()
    opts = dict(CONFIG)
    if args.config:
        opts.update(json.loads(args.config.read_text(encoding="utf-8")))
    if args.trajectory_frames is not None:
        opts["trajectory_view_num"] = args.trajectory_frames
    if args.roundtrip_bab_bcb:
        opts["trajectory_mode"] = "roundtrip_bab_bcb"
        opts["trajectory_view_num"] = args.trajectory_frames or 81
        opts["roundtrip_center_index"] = args.roundtrip_center_index
        try:
            opts["roundtrip_target_indices"] = tuple(
                int(x.strip()) for x in args.roundtrip_target_indices.split(","))
        except ValueError:
            parser.error("--roundtrip-target-indices must be comma-separated integers")
        opts["roundtrip_pitch_deg"] = args.roundtrip_pitch_deg
        opts["roundtrip_elevation_profile"] = args.roundtrip_elevation_profile
        opts["roundtrip_edge_ramp"] = args.roundtrip_edge_ramp
        if sorted([opts["roundtrip_center_index"], *opts["roundtrip_target_indices"]]) != [0,1,2]:
            parser.error("BAB/BCB requires distinct center and A/C indices [0,1,2]")
    if opts["trajectory_view_num"] < 5 or (opts["trajectory_view_num"] - 1) % 4:
        parser.error("--trajectory-frames must be >= 5 and equal to 4n+1 (e.g. 49 or 81)")
    if args.video_prompt_embeds is not None:
        opts["video_prompt_embeds_path"] = str(args.video_prompt_embeds.expanduser().resolve())
    if args.no_save_mp4:
        opts["video_save_mp4"] = False
    if args.video_cpu_threads is not None:
        opts["video_cpu_threads"] = args.video_cpu_threads
    if args.video_gpus:
        opts["video_gpu_ids"] = [int(x.strip()) for x in args.video_gpus.split(",") if x.strip()]
        if len(opts["video_gpu_ids"]) != len(set(opts["video_gpu_ids"])) or not opts["video_gpu_ids"]:
            parser.error("--video-gpus must contain distinct nonempty GPU IDs")
    if args.benchmark_fresh:
        if args.resume or args.rerun_dense:
            parser.error("--benchmark-fresh cannot be used with --resume or --rerun-dense")
        opts["resume_stages"] = False
        opts["video_resume"] = False
        opts["force_rerun_dense"] = False
    if args.da3_only:
        opts["run_3dgs_iter"] = False
    if args.resume:
        opts["resume_stages"] = True
    if args.rerun_dense:
        opts["resume_stages"] = True
        opts["video_resume"] = True
        opts["force_rerun_dense"] = True
    root = args.work_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    cases = ([root] if (root / "image").is_dir() else
             sorted(p for p in root.iterdir() if p.is_dir() and (p / "image").is_dir()))
    if not cases:
        raise ValueError(f"No case/image folders found in {root}")
    if args.benchmark_fresh:
        collisions = [str(case / "reconstruction") for case in cases
                      if (case / "reconstruction").exists()]
        if collisions:
            raise FileExistsError(
                "Fresh benchmark would reuse/overwrite existing reconstruction. "
                "Create a separate test case containing only image/: " + ", ".join(collisions))
    print(f"[BENCHMARK] trajectory_mode={opts['trajectory_mode']} "
          f"trajectory_frames={opts['trajectory_view_num']} "
          f"gpus={opts['video_gpu_ids']} "
          f"worker_cpu_threads={opts['video_cpu_threads']} "
          f"captioner={opts['run_captioner']} positive_prompt={opts['video_positive_prompt']!r} "
          f"prompt_embeds={opts['video_prompt_embeds_path'] or 'T5'} "
          f"save_mp4={opts['video_save_mp4']} "
          f"fresh={args.benchmark_fresh} "
          f"video_resume={opts['video_resume']} resume_stages={opts['resume_stages']}", flush=True)
    wall_started = time.perf_counter()
    pipeline = MultiImagePipeline(**opts)
    print(f"[MULTI PRELOAD] initial={pipeline.load_seconds:.3f}s "
          f"wan_worker_pool={pipeline.video_worker_preload_seconds:.3f}s", flush=True)
    summary = {}
    try:
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
    finally:
        pipeline.close()
    print("\n[MULTI END-TO-END TIMING SUMMARY] (seconds)", flush=True)
    keys = ("image_preprocess", "caption", "da3", "video_model_load",
            "video_generation", "video_inference", "video_text_condition_prepare_sum",
            "video_text_embed_load_per_worker", "trajectory_video",
            "dense_prepare", "dense", "video_save_jpg_sum",
            "video_save_mp4_sum", "video_wall_no_mp4_estimate",
            "total_no_mp4_estimate", "total")
    print(f"{'Case':<35} {'Status':<8} " + " ".join(f"{key:>19}" for key in keys), flush=True)
    for name, record in summary.items():
        print(f"{name[:35]:<35} {record['status']:<8} "
              + " ".join(f"{record[key]:>18.3f}s" if key in record else f"{'-':>19}"
                         for key in keys), flush=True)
    wall = time.perf_counter() - wall_started
    print(f"[MULTI ALL CASES] wall={wall:.3f}s "
          f"initial_model_preload={pipeline.load_seconds:.3f}s "
          f"wan_worker_preload={pipeline.video_worker_preload_seconds:.3f}s", flush=True)
    for name, record in summary.items():
        if record.get("status") != "ok":
            continue
        print(f"[BENCHMARK {name}] stage_total={record.get('stages_total', float('nan')):.3f}s "
              f"case_total={record.get('total', float('nan')):.3f}s "
              f"initial_preload_once={pipeline.load_seconds:.3f}s "
              f"wan_preload_once={pipeline.video_worker_preload_seconds:.3f}s "
              f"video_wall={record.get('trajectory_video', float('nan')):.3f}s "
              f"video_sum_segments={record.get('video_generation', float('nan')):.3f}s "
              f"video_no_mp4_est={record.get('video_wall_no_mp4_estimate', float('nan')):.3f}s "
              f"total_no_mp4_est={record.get('total_no_mp4_estimate', float('nan')):.3f}s "
              f"dense={record.get('dense', float('nan')):.3f}s", flush=True)
    if any(row["status"] == "failed" for row in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
