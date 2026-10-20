#!/usr/bin/env python3
"""Run independent reconstruction cases concurrently, one worker per GPU.

Each worker receives CUDA_VISIBLE_DEVICES=<physical GPU> and sees it as cuda:0.
External adapters remain responsible for their own model initialization.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def find_cases(root: Path) -> list[Path]:
    if (root / "image").is_dir():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "image").is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Single case or parent directory of cases")
    parser.add_argument("--gpus", default="0,1", help="Physical GPU IDs, e.g. 4,5")
    parser.add_argument("--video-adapter")
    parser.add_argument("--da3-factory")
    parser.add_argument("--dense-adapter")
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=3)
    parser.add_argument("--reuse-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--view-num", type=int, default=40)
    parser.add_argument("--crop-ratio", type=float, default=1.0)
    parser.add_argument("--use-view-extrapolation", action="store_true")
    parser.add_argument("--no-livephoto", action="store_true")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--circle", action="store_true")
    parser.add_argument("--max-frames", type=int, default=121)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Directory not found: {root}")
    cases = find_cases(root)
    if not cases:
        parser.error(f"No cases with image/ found in {root}")
    gpus = [s.strip() for s in args.gpus.split(",") if s.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        parser.error("--gpus must contain distinct GPU IDs")

    script = Path(__file__).with_name("multistage_pipeline.py")
    common = ["--start-stage", str(args.start_stage), "--end-stage", str(args.end_stage),
              "--view-num", str(args.view_num), "--crop-ratio", str(args.crop_ratio),
              "--mode", args.mode, "--max-frames", str(args.max_frames)]
    for flag, value in (("--video-adapter", args.video_adapter),
                        ("--da3-factory", args.da3_factory),
                        ("--dense-adapter", args.dense_adapter)):
        if value:
            common.extend([flag, value])
    for flag, enabled in (("--reuse-video", args.reuse_video),
                          ("--use-view-extrapolation", args.use_view_extrapolation),
                          ("--no-livephoto", args.no_livephoto),
                          ("--circle", args.circle),
                          ("--dry-run", args.dry_run)):
        if enabled:
            common.append(flag)

    def worker(gpu: str, assigned: list[Path]) -> list[tuple[Path, int]]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        outcomes = []
        for case in assigned:
            command = [sys.executable, str(script), str(case), *common]
            print(f"[GPU {gpu}] START {case}", flush=True)
            result = subprocess.run(command, env=env, cwd=str(script.parents[2]))
            print(f"[GPU {gpu}] {'DONE' if result.returncode == 0 else 'FAILED'} {case} (exit={result.returncode})", flush=True)
            outcomes.append((case, result.returncode))
        return outcomes

    assignments = [cases[i::len(gpus)] for i in range(len(gpus))]
    results = []
    with ThreadPoolExecutor(max_workers=min(len(cases), len(gpus))) as pool:
        futures = [pool.submit(worker, gpu, assigned)
                   for gpu, assigned in zip(gpus, assignments) if assigned]
        for future in as_completed(futures):
            results.extend(future.result())
    failed = [str(case) for case, code in results if code != 0]
    print(f"Completed: {len(results) - len(failed)}/{len(results)} cases; failures: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
