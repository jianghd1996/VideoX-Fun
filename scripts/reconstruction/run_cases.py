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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_ARGS = {
    "run_DA3": True,
    "use_enhance_crop": True,
    "use_view_extrapolation": True,
    "run_gen_video": True,
    "run_captioner": True,
    "captioner_cpu_swap": True,
    "captioner_model_path": None,
    "captioner_code_root": None,
    "view_num": 121,
    "recon_view_num": 40,
    "gen_vid_infer_step": 8,
    "gen_vid_fps": 30,
    "world_size": 1,
    "run_3dgs_iter": True,
    "mode": "scene",
    "crop_ratio": 1.0,
    "max_frames": 121,
    "is_livephoto": True,
    "reuse_video": False,
    "resume_stages": False,
    "video_adapter": None,
    "lora_path": str(Path(__file__).resolve().parent / "Cloud_Models/Wan2.2-TI2V-5B/lora_ckpts_single8/checkpoint-9600.safetensors"),
    "lora_weight": 0.55,
    "use_lightvae": False,
    "lightvae_path": str(Path(__file__).resolve().parent / "Cloud_Models/transformer_w_skybox_sla/MG-LightVAE.pth"),
    "external_code_root": None,
    "da3_model_path": None,
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
    log_path = work_dir / "reconstruction_run.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            print(f"\\n{'=' * 40} RUN {datetime.now().isoformat(timespec='seconds')} {'=' * 40}", flush=True)
            print(f"[LOG] {log_path}", flush=True)
            opts = {**DEFAULT_ARGS, **options}
            cases = [work_dir] if (work_dir / "image").is_dir() else [
                p for p in sorted(work_dir.iterdir()) if p.is_dir() and (p / "image").is_dir()
            ]
            run_list, skip_list = set(opts["run_list"]), set(opts["skip_list"])
            cases = [p for p in cases if (not run_list or p.name in run_list)
                     and p.name not in skip_list and "." not in p.name]
            if not cases:
                raise ValueError(f"No matching cases under {work_dir}")

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
