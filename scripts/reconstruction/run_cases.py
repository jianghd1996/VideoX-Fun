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
from multistage_pipeline import ReconstructionConfig, ReconstructionPipeline, load_callable

DEFAULT_ARGS = {
    "run_DA3": True,
    "use_enhance_crop": True,
    "use_view_extrapolation": True,
    "run_gen_video": True,
    "view_num": 40,
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
    opts = {**DEFAULT_ARGS, **options}
    if int(opts["world_size"]) < 1:
        raise ValueError("world_size must be >= 1")
    # world_size is passed through for future stage-level distributed inference;
    # this entrypoint intentionally does NOT spawn extra ranks.
    cases = [work_dir] if (work_dir / "image").is_dir() else [
        p for p in sorted(work_dir.iterdir()) if p.is_dir() and (p / "image").is_dir()
    ]
    run_list, skip_list = set(opts["run_list"]), set(opts["skip_list"])
    cases = [p for p in cases if (not run_list or p.name in run_list)
             and p.name not in skip_list and "." not in p.name]
    if not cases:
        raise ValueError(f"No matching case directories under {work_dir}")
    cfg = ReconstructionConfig(
        case_dir=cases[0], view_num=int(opts["view_num"]),
        crop_ratio=float(opts["crop_ratio"]),
        use_view_extrapolation=bool(opts["use_view_extrapolation"]),
        is_livephoto=bool(opts["is_livephoto"]),
        mode=str(opts["mode"]), max_frames=int(opts["max_frames"]),
        external_code_root=Path(opts["external_code_root"]) if opts["external_code_root"] else None,
        da3_model_path=Path(opts["da3_model_path"]) if opts["da3_model_path"] else None,
    )
    video = load_callable(opts["video_adapter"]) if opts["video_adapter"] else None
    da3 = load_callable(opts["da3_factory"])() if opts["da3_factory"] else None
    dense = load_callable(opts["dense_adapter"]) if opts["dense_adapter"] else None
    pipeline = ReconstructionPipeline(cfg, generate_video=video, run_DA3=da3,
                                      recon_iter_full=dense)
    # Explicitly load only enabled models, once before the first case.
    load_start = time.perf_counter()
    if opts["run_gen_video"]:
        pipeline._preload_video()
    if opts["run_DA3"]:
        pipeline._preload_da3()
    if opts["run_3dgs_iter"]:
        pipeline._preload_dense()
    print(f"[PRELOAD] all enabled models: {time.perf_counter() - load_start:.2f}s", flush=True)

    summary = {}
    for case in cases:
        pipeline.set_case(case)
        timings = {}
        start = time.perf_counter()
        print(f"\n[CASE] {case}", flush=True)
        try:
            for enabled, label, stage in (
                (opts["run_gen_video"], "video_and_sampling",
                 lambda: pipeline.stage_video(reuse=opts["reuse_video"])),
                (opts["run_DA3"], "da3", pipeline.stage_da3),
                (opts["run_3dgs_iter"], "dense", pipeline.stage_dense),
            ):
                if not enabled:
                    continue
                t0 = time.perf_counter()
                try:
                    stage()
                finally:
                    from multistage_pipeline import _synchronize_cuda
                    _synchronize_cuda()
                    timings[label] = round(time.perf_counter() - t0, 3)
                    print(f"[TIME] {label}: {timings[label]:.3f}s", flush=True)
            status = "ok"
        except Exception as exc:
            status = "failed"
            print(f"[ERROR] {case.name}: {exc}", flush=True)
        finally:
            from multistage_pipeline import _synchronize_cuda
            _synchronize_cuda()
            timings["total"] = round(time.perf_counter() - start, 3)
            timings["status"] = status
            pipeline.gen_vid_path.mkdir(parents=True, exist_ok=True)
            (pipeline.gen_vid_path / "timings.json").write_text(
                json.dumps(timings, indent=2), encoding="utf-8")
            summary[case.name] = timings
            print(f"[TIME] {case.name} TOTAL excluding preload: {timings['total']:.3f}s", flush=True)
    (work_dir / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    if any(v["status"] != "ok" for v in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--config", type=Path, help="JSON file with SparseRecon-style options")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    run_cases(args.work_dir.resolve(), config)
