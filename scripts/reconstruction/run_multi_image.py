#!/usr/bin/env python3
"""Run custom DA3 -> dense 3DGS on one or more multi-image cases.

Edit CONFIG and WORK_DIR below, or pass --config / --work-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_image_pipeline import MultiImagePipeline

CONFIG = {
    "external_code_root": "/home/z00566689/dev/mnt/jiang_dev/SingleRecon/DA3_custom",
    "dense_code_root": "/home/z00566689/dev/mnt/jiang_dev/SingleRecon",
    "da3_model_path": "/home/z00566689/dev/mnt/jiang_dev/SingleRecon/Cloud_Models/Saliency-Face-DA3/besthumanmodel_lr1e_6_ckpt2.pt",
    "run_3dgs_iter": False,  # trajectory inspection: DA3 + GS render only
    "resume_stages": False,
    "crop_ratio": 1.0,
    "mode": "scene",
    "circle": False,
    "max_frames": 121,
    "render_trajectories": True,
    "trajectory_extra": False,
    "trajectory_view_num": 81,
    "trajectory_interpolation": "bezier",  # Keep fixed until E2E pipeline works
    "run_gen_video": True,  # enable after connecting a control-video Wan adapter
    "video_adapter": "predict_v2v_control_5b_multi_image:WanControl5B",
    "video_code_root": "/home/z00566689/dev/mnt/jiang_dev/SingleRecon/VideoX-Fun",
    "video_lora_path": "/home/z00566689/dev/mnt/jiang_dev/SingleRecon/Cloud_Models/Wan2.2-Fun-5B-Control/12000_lora.safetensors",
    "video_lora_weight": 0.55,
    "video_fps": 30,
    "video_resume": True,
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
    pipeline = MultiImagePipeline(**opts)
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
    if any(row["status"] == "failed" for row in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
