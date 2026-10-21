#!/usr/bin/env python3
"""SparseRecon-style sequential case runner with reusable preloaded backends.

Set adapter specs in JSON or edit DEFAULT_ARGS. External models are not vendored.
"""
from __future__ import annotations

import argparse
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
    "video_adapter": None,
    "external_code_root": None,
    "da3_model_path": None,
    "da3_factory": None,
    "dense_adapter": None,
    "run_list": [],
    "skip_list": ["collect", "test"],
}


def run_cases(work_dir: Path, options: dict) -> None:
    from pipeline_reconstruction import SparseRecon

    opts = {**DEFAULT_ARGS, **options}
    cases = [work_dir] if (work_dir / "image").is_dir() else [
        p for p in sorted(work_dir.iterdir()) if p.is_dir() and (p / "image").is_dir()
    ]
    run_list, skip_list = set(opts["run_list"]), set(opts["skip_list"])
    cases = [p for p in cases if (not run_list or p.name in run_list)
             and p.name not in skip_list and "." not in p.name]
    if not cases:
        raise ValueError(f"No matching cases under {work_dir}")

    # One object, one model initialization, many cases.
    recon = SparseRecon(json.dumps(opts))
    summary = {}
    for case in cases:
        print(f"\\n[CASE] {case}", flush=True)
        try:
            result = recon(work_dir=case, export_dir=case / "export",
                           mode=opts["mode"])
            summary[case.name] = {**result["timings"], "status": "ok"}
        except Exception as exc:
            summary[case.name] = {"status": "failed", "error": str(exc)}
            print(f"[ERROR] {case.name}: {exc}", flush=True)
    (work_dir / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    if any(row["status"] != "ok" for row in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    # Edit these paths directly, then run: python scripts/reconstruction/run_cases.py
    WORK_DIR = "/home/z00566689/dev/mnt/jiang_test/00_0825_test"
    CONFIG_PATH = None  # Optional JSON file; None uses DEFAULT_ARGS above.

    config = (json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
              if CONFIG_PATH else {})
    run_cases(Path(WORK_DIR).expanduser().resolve(), config)
