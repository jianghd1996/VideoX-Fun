#!/usr/bin/env python3
"""DA3 + Reconstruction_full on pre-existing images (no video generation).

Reuses the single-image pipeline's *same* DA3 runner and dense reconstruction
backend. This is an A/B experiment with multi-image known-pose reconstruction.

Examples (run from SingleRecon after copying this script there):
    CUDA_VISIBLE_DEVICES=4 python run_existing_images_da3_dense.py \
        --work-dir /home/z00566689/dev/mnt/jiang_test/00_1022_test/cover2_compare

Image inputs may be either <work-dir>/image/* or <work-dir>/*.
Outputs are isolated in <work-dir>/reconstruction/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multistage_pipeline import (
    DA3Params, IMAGE_SUFFIXES, _synchronize_cuda,
    create_da3, create_full_reconstruction,
)

# Same external implementations and checkpoints as the original single-image
# pipeline. Override these from CLI when testing a different checkout.
SINGLERECON_ROOT = Path("/home/z00566689/dev/mnt/jiang_dev/SingleRecon")
DA3_ROOT = SINGLERECON_ROOT / "DA3_custom"
DA3_MODEL = SINGLERECON_ROOT / "Cloud_Models/Saliency-Face-DA3/besthumanmodel_lr1e_6_ckpt2.pt"


def _source_images(root: Path) -> list[Path]:
    image_dir = root / "image" if (root / "image").is_dir() else root
    files = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(files) < 2:
        raise ValueError(f"Expected 2+ existing image files at {image_dir}; got {len(files)}")
    return files


def _prepare_frames(files: list[Path], dest: Path, *, reuse: bool) -> None:
    """Copy image inputs into a normalized JPG folder for the unchanged DA3."""
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    expected = [{"path": str(p.resolve()), "size": p.stat().st_size,
                 "mtime_ns": p.stat().st_mtime_ns} for p in files]
    provenance = dest.parent / "source_images_manifest.json"
    if reuse and provenance.is_file():
        previous = json.loads(provenance.read_text(encoding="utf-8"))
        existing = sorted(dest.glob("*.jpg"))
        if previous == expected and len(existing) == len(files):
            print(f"[COMPARE] Reuse normalized {len(files)} images", flush=True)
            return

    # Avoid silently using stale JPGs from an earlier experiment.
    for old in dest.iterdir():
        if old.is_file() and old.suffix.lower() in IMAGE_SUFFIXES:
            old.unlink()
    for i, file in enumerate(files):
        # DA3 runner historically globs '*.jpg' and sorts filenames.
        with Image.open(file) as img:
            img.convert("RGB").save(dest / f"{i:06d}.jpg", quality=98)
    provenance.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print(f"[COMPARE] Normalized {len(files)} images -> {dest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(
        "/home/z00566689/dev/mnt/jiang_test/00_1022_test/cover2_compare"))
    parser.add_argument("--da3-root", type=Path, default=DA3_ROOT)
    parser.add_argument("--dense-root", type=Path, default=SINGLERECON_ROOT)
    parser.add_argument("--da3-model", type=Path, default=DA3_MODEL)
    parser.add_argument("--crop-ratio", type=float, default=1.0)
    parser.add_argument("--mode", default="scene")
    parser.add_argument("--max-frames", type=int, default=121)
    parser.add_argument("--skip-da3", action="store_true",
                        help="Reuse prior COLMAP if already complete")
    parser.add_argument("--skip-dense", action="store_true",
                        help="Run DA3 only (useful for testing 83-frame capacity)")
    parser.add_argument("--dense-export", type=Path,
                        help="Optional dense output path (default reconstruction/dense)")
    args = parser.parse_args()

    work = args.work_dir.expanduser().resolve()
    if not work.is_dir():
        raise FileNotFoundError(work)
    images = _source_images(work)
    workspace = work / "reconstruction"
    image_dir = workspace / "image"
    sparse = workspace / "sparse" / "0"
    gs_render = workspace / "gs_render.mp4"
    dense_out = (args.dense_export.expanduser().resolve() if args.dense_export
                 else workspace / "dense")
    workspace.mkdir(parents=True, exist_ok=True)

    times = {}
    all_started = time.perf_counter()
    image_started = time.perf_counter()
    _prepare_frames(images, image_dir, reuse=args.skip_da3)
    times["input_prepare"] = round(time.perf_counter() - image_started, 3)

    if args.skip_da3:
        if not any((sparse / f"images.{ext}").is_file() for ext in ("bin", "txt")):
            raise FileNotFoundError(f"--skip-da3 needs preexisting COLMAP under {sparse}")
        print(f"[COMPARE] Skip DA3, use {sparse}", flush=True)
    else:
        load_started = time.perf_counter()
        da3 = create_da3(args.da3_root, args.da3_model)
        times["da3_model_load"] = round(time.perf_counter() - load_started, 3)
        sparse.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f"[COMPARE] DA3 input={len(images)} images; crop_ratio={args.crop_ratio}", flush=True)
        try:
            result = da3.infer(args=DA3Params(
                data_path=str(image_dir),
                colmap_path=str(sparse),
                gs_render_vid_path=str(gs_render),
                view_num=len(images),
                crop_ratio=float(args.crop_ratio),
                extra_view=[False, False],
            ))
            if result is False:
                raise RuntimeError("DA3 returned False")
        finally:
            _synchronize_cuda()
            times["da3"] = round(time.perf_counter() - started, 3)
            print(f"[COMPARE] DA3 time={times['da3']:.3f}s", flush=True)

    if not args.skip_dense:
        started = time.perf_counter()
        dense = create_full_reconstruction(args.dense_root)
        times["dense_model_load"] = round(time.perf_counter() - started, 3)
        dense_out.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f"[COMPARE] Reconstruction_full images={len(images)} "
              f"work_dir={workspace} export_dir={dense_out}", flush=True)
        try:
            result = dense(work_dir=workspace, export_dir=dense_out,
                           mode=args.mode, circle=False,
                           max_frames=args.max_frames)
            if result is False:
                raise RuntimeError("Reconstruction_full returned False")
        finally:
            _synchronize_cuda()
            times["dense"] = round(time.perf_counter() - started, 3)
            print(f"[COMPARE] Dense time={times['dense']:.3f}s", flush=True)

    times["total"] = round(time.perf_counter() - all_started, 3)
    summary = dict(input_images=len(images), pipeline="DA3+Reconstruction_full",
                   input_dir=str((work / "image") if (work / "image").is_dir() else work),
                   sparse_dir=str(sparse), dense_dir=str(dense_out), timings=times)
    (workspace / "compare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[COMPARE SUMMARY] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
