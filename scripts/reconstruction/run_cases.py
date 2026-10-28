#!/usr/bin/env python3
"""SparseRecon-style sequential case runner with reusable preloaded backends.

Set adapter specs in JSON or edit DEFAULT_ARGS. External models are not vendored.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import traceback
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_ARGS = {
    "run_DA3": True,
    "use_enhance_crop": True,
    "use_view_extrapolation": False,
    "run_gen_video": False,
    "run_captioner": False,  # benchmark: skip image captioning
    "captioner_cpu_swap": True,
    "captioner_model_path": None,
    "captioner_code_root": None,
    "view_num": 121,
    "recon_view_num": 40,
    "gen_vid_infer_step": 8,
    "gen_vid_fps": 30,
    "world_size": 1,
    "multi_gpu": False,  # True: same-case sequence parallelism via torchrun (2 GPUs)
    "run_3dgs_iter": True,
    "mode": "scene",
    "crop_ratio": 1.0,
    "max_frames": 121,
    "is_livephoto": False,
    "reuse_video": False,
    "resume_stages": False,
    "video_adapter": None,
    "lora_path": str(Path(__file__).resolve().parent / "Cloud_Models/Wan2.2-TI2V-5B/lora_ckpts_single8/checkpoint-9600.safetensors"),
    "lora_weight": 0.55,
    "use_lightvae": True,
    "condition_vae_fast_encode": True,  # False: original full 121-frame condition VAE encode
    "condition_vae_encode_frames": 49,  # only used when condition_vae_fast_encode=True
    "prompt_embeds_path": None,  # use T5 text encoder for benchmark
    "lightvae_path": str(Path(__file__).resolve().parent / "Cloud_Models/transformer_w_skybox_sla/MG-LightVAE.pth"),
    "external_code_root": str(Path(__file__).resolve().parent / "DA3_custom"),
    "dense_code_root": str(Path(__file__).resolve().parent),
    "da3_model_path": str(Path(__file__).resolve().parent / "Cloud_Models/Saliency-Face-DA3/besthumanmodel_lr1e_6_ckpt2.pt"),
    "da3_factory": None,
    "dense_adapter": None,
    "run_list": [],
    "skip_list": ["collect", "test"],
}


class Tee:
    """Mirror console output into a line-buffered log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def _format_seconds(value):
    return f"{value:.3f}s" if isinstance(value, (int, float)) else "-"


def _print_summary(summary, preload_seconds=None):
    stage_names = ("caption", "video_and_sampling", "da3", "dense", "total")
    print("\\n" + "=" * 105, flush=True)
    print("[FINAL TIMING SUMMARY] (seconds; total excludes model preload)", flush=True)
    if preload_seconds is not None:
        print(f"Model preload: {preload_seconds:.3f}s (once for the entire batch)", flush=True)
    print(f"{'Case':<38} {'Status':<8} " + " ".join(f"{n:>15}" for n in stage_names), flush=True)
    for case_name, row in summary.items():
        print(f"{case_name[:38]:<38} {row.get('status', '?'):<8} " +
              " ".join(f"{_format_seconds(row.get(n)):>15}" for n in stage_names), flush=True)
        if row.get("error"):
            print(f"  ERROR: {row['error']}", flush=True)
    totals = {name: sum(row.get(name, 0) for row in summary.values()
                        if isinstance(row.get(name), (int, float))) for name in stage_names}
    print(f"{'SUM (recorded)':<38} {'':8} " +
          " ".join(f"{totals[n]:>14.3f}s" for n in stage_names), flush=True)
    print("=" * 105, flush=True)


def run_cases(work_dir: Path, options: dict) -> None:
    from pipeline_reconstruction import SparseRecon

    work_dir.mkdir(parents=True, exist_ok=True)
    rank_for_log = int(os.environ.get("RANK", "0"))
    log_path = work_dir / ("reconstruction_run_rank1.log" if rank_for_log else "reconstruction_run.log")
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            print(f"\\n{'=' * 40} RUN {datetime.now().isoformat(timespec='seconds')} {'=' * 40}", flush=True)
            print(f"[LOG] {log_path}", flush=True)
            opts = {**DEFAULT_ARGS, **options}
            da3_root = Path(opts["external_code_root"]).expanduser().resolve()
            if not (da3_root / "DA3" / "run_da3_full_crop_dev.py").is_file():
                raise FileNotFoundError(f"DA3_custom runner not found: {da3_root}")
            for import_root in (da3_root / "src", da3_root):
                if str(import_root) not in sys.path:
                    sys.path.insert(0, str(import_root))
            import DA3.run_da3_full_crop_dev as da3_module
            print(f"[DA3 IMPORT] {da3_module.__file__}", flush=True)
            if not Path(da3_module.__file__).resolve().is_relative_to(da3_root):
                raise ImportError("DA3 resolved outside DA3_custom")
            opts["da3_backend"] = "legacy"
            opts["da3_factory"] = None
            cases = [work_dir] if (work_dir / "image").is_dir() else [
                p for p in sorted(work_dir.iterdir()) if p.is_dir() and (p / "image").is_dir()
            ]
            run_list, skip_list = set(opts["run_list"]), set(opts["skip_list"])
            cases = [p for p in cases if (not run_list or p.name in run_list)
                     and p.name not in skip_list and "." not in p.name]
            if not cases:
                raise ValueError(f"No matching cases under {work_dir}")

            rank = int(os.environ.get("RANK", "0"))
            distributed_requested = bool(opts.get("multi_gpu", False))
            launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if distributed_requested:
                if launched_world_size != 2:
                    raise RuntimeError("multi_gpu=True requires torchrun --nproc_per_node=2")
                opts["world_size"] = launched_world_size
                # Both ranks must enter video inference for every case.
                opts["resume_stages"] = False
                opts["reuse_video"] = False
                import torch
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            elif launched_world_size > 1:
                raise RuntimeError("torchrun detected but multi_gpu=False; enable the multi_gpu switch")
            if distributed_requested and rank != 0:
                # Secondary rank participates only in Wan sequence-parallel video inference.
                # Never writes the case journal, sampled frames, or reconstruction outputs.
                worker_opts = {**opts, "run_captioner": False, "run_DA3": False,
                               "run_3dgs_iter": False}
                recon_worker = SparseRecon(json.dumps(worker_opts))
                for case in cases:
                    print(f"[GPU WORKER rank={rank}] {case}", flush=True)
                    recon_worker.pipeline.set_case(case)
                    recon_worker.pipeline.stage_video(reuse=False)
                return

            if not opts["run_gen_video"]:
                for case in cases:
                    frame_dir = case / "reconstruction" / "image"
                    if not frame_dir.is_dir() or not any(frame_dir.glob("*.jpg")):
                        raise FileNotFoundError(f"DA3-only needs sampled JPG frames: {frame_dir}")
            summary = {}
            recon = None
            try:
                # One object, one model initialization, many cases.
                recon = SparseRecon(json.dumps(opts))
                for case in cases:
                    print(f"\\n[CASE] {case}", flush=True)
                    try:
                        result = recon(work_dir=case, export_dir=case / "export",
                                       mode=opts["mode"])
                        summary[case.name] = {**result["timings"], "status": "ok"}
                    except Exception as exc:
                        timings_file = case / "reconstruction" / "timings.json"
                        try:
                            timings = json.loads(timings_file.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            timings = {}
                        summary[case.name] = {**timings, "status": "failed", "error": str(exc)}
                        print(f"[ERROR] {case.name}: {exc}", flush=True)
                        traceback.print_exc()
                    finally:
                        # Keep a usable summary even if the process is interrupted later.
                        (work_dir / "reconstruction_summary.json").write_text(
                            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            finally:
                _print_summary(summary, getattr(recon, "load_seconds", None))
                (work_dir / "reconstruction_summary.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[LOG SAVED] {log_path}", flush=True)
            if any(row["status"] != "ok" for row in summary.values()):
                raise SystemExit(1)


if __name__ == "__main__":
    # Edit these paths directly, then run: python scripts/reconstruction/run_cases.py
    WORK_DIR = "/home/z00566689/dev/mnt/jiang_test/00_1021_test"
    CONFIG_PATH = None  # Optional JSON file; None uses DEFAULT_ARGS above.

    config = (json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
              if CONFIG_PATH else {})
    run_cases(Path(WORK_DIR).expanduser().resolve(), config)
