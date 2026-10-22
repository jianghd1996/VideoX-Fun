"""Multi-image reconstruction: custom DA3 -> (reserved video stage) -> dense 3DGS.

The DA3 and dense backends are the SAME as the single-image pipeline.
No novel-view generation or trajectory planning is performed in this version.
"""
from __future__ import annotations

import json
import os
import sys
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .multistage_pipeline import (
        DA3Params, IMAGE_SUFFIXES, ReconstructionConfig, ReconstructionPipeline,
        _synchronize_cuda, create_da3, create_full_reconstruction,
    )
except ImportError:
    from multistage_pipeline import (
        DA3Params, IMAGE_SUFFIXES, ReconstructionConfig, ReconstructionPipeline,
        _synchronize_cuda, create_da3, create_full_reconstruction,
    )


def input_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing multi-image input directory: {image_dir}")
    images = sorted(p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < 2:
        raise ValueError(f"Multi-image reconstruction needs >=2 images in {image_dir}; got {len(images)}")
    return images


def _colmap_valid(folder: Path) -> bool:
    for ext in ("bin", "txt"):
        files = [folder / f"{name}.{ext}" for name in ("cameras", "images", "points3D")]
        if all(p.is_file() for p in files) and files[1].stat().st_size > (8 if ext == "bin" else 0):
            return True
    return False


class MultiImagePipeline:
    """Persistent multi-case pipeline. Construct once to preload DA3/dense models."""

    def __init__(self, *, external_code_root: str | Path,
                 dense_code_root: str | Path | None = None,
                 da3_model_path: str | Path | None = None,
                 run_3dgs_iter: bool = True,
                 resume_stages: bool = False,
                 crop_ratio: float = 1.0,
                 mode: str = "scene",
                 circle: bool = False,
                 max_frames: int = 121,
                 da3_instance: Any = None,
                 dense_instance: Any = None,
                 render_trajectories: bool = True,
                 trajectory_extra: bool = False,
                 trajectory_view_num: int = 81,
                 trajectory_corner_smoothing: float = 0.15):
        root = Path(external_code_root).expanduser().resolve()
        dense_root = Path(dense_code_root or root).expanduser().resolve()
        self.run_3dgs_iter = bool(run_3dgs_iter)
        self.render_trajectories = bool(render_trajectories)
        self.trajectory_extra = bool(trajectory_extra)
        self.trajectory_view_num = int(trajectory_view_num)
        self.trajectory_corner_smoothing = float(trajectory_corner_smoothing)
        self.resume_stages = bool(resume_stages)
        self.crop_ratio = float(crop_ratio)
        self.mode = mode
        self.circle = bool(circle)
        self.max_frames = int(max_frames)
        for import_root in (root / "src", root):
            if str(import_root) not in sys.path:
                __import__("sys").path.insert(0, str(import_root))
        start = time.perf_counter()
        self.da3 = da3_instance if da3_instance is not None else create_da3(
            root, Path(da3_model_path) if da3_model_path else None)
        self.dense = (dense_instance if dense_instance is not None
                      else create_full_reconstruction(dense_root)) if self.run_3dgs_iter else None
        self.load_seconds = round(time.perf_counter() - start, 3)
        print(f"[MULTI PRELOAD] {self.load_seconds:.3f}s (excluded from case timings)", flush=True)


    def _da3_with_trajectory(self, image_dir: Path, colmap: Path, gs_video: Path):
        """One DA3 inference, then render each planned path from its predicted Gaussians."""
        import glob
        import numpy as np
        import torch
        import cv2
        from camera_trajectory import plan_trajectories, save_trajectories
        from depth_anything_3.utils.export.colmap_init import save_sparserecon
        from depth_anything_3.utils.export.gs import export_to_gs_video

        images = sorted(glob.glob(str(image_dir / "*.jpg")))
        device = self.da3.device
        self.da3.model.to(device=device)
        try:
            prediction = self.da3.model.inference(
                images, infer_gs=True, use_ray_pose=True, process_res=840)
            # COLMAP exporter applies the same metric scale conversion as the
            # original DA3 runner. Do not modify prediction or Gaussian positions.
            recon = save_sparserecon(prediction, images, colmap)
            poses_for_colmap = np.asarray(prediction.extrinsics).copy()
            if prediction.is_metric and prediction.scale_factor is not None:
                poses_for_colmap[:, :3, 3] /= float(prediction.scale_factor)
            paths, info = plan_trajectories(
                poses_for_colmap, extra=self.trajectory_extra,
                view_num=self.trajectory_view_num, circle=self.circle,
                corner_smoothing=self.trajectory_corner_smoothing)
            if not paths:
                raise RuntimeError("No up/down trajectories generated")
            out = gs_video.parent / "trajectories"
            manifest = save_trajectories(paths, info, out)
            rendered_paths = []
            for i, path in enumerate(paths):
                name = manifest["trajectories"][i]["file"].removesuffix(".npy")
                poses = path["extrinsics"].copy()
                # Poses already match the COLMAP and DA3 GS-renderer scale.
                a, b = path["start_index"], path["end_index"]
                Ka = np.asarray(prediction.intrinsics[a], dtype=np.float32)
                Kb = np.asarray(prediction.intrinsics[b], dtype=np.float32)
                weights = np.linspace(0, 1, len(poses), dtype=np.float32)
                Ks = np.stack([(1-w)*Ka+w*Kb for w in weights])
                render_kwargs = dict(
                    prediction=prediction, export_dir=str(out),
                    extrinsics=torch.from_numpy(poses).to(prediction.gaussians.means).unsqueeze(0),
                    intrinsics=torch.from_numpy(Ks).to(prediction.gaussians.means).unsqueeze(0),
                    trj_mode="original", vis_depth=None, output_name=name,
                    chunk_size=4)
                print(f"[TRAJECTORY] {name}: {len(poses)} frames, input {a}->{b}", flush=True)
                export_to_gs_video(**render_kwargs)
                rendered = out / "gs_video" / f"{name}.mp4"
                if not rendered.is_file():
                    raise FileNotFoundError(rendered)
                rendered_paths.append(rendered)
            # A single inspection video concatenates all paths; each clip is
            # also retained independently to identify its source endpoints.
            writer = None
            try:
                for path in rendered_paths:
                    cap = cv2.VideoCapture(str(path))
                    if not cap.isOpened():
                        raise RuntimeError(f"Cannot open GS trajectory video: {path}")
                    try:
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            h, w = frame.shape[:2]
                            if writer is None:
                                writer = cv2.VideoWriter(str(gs_video),
                                    cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
                                if not writer.isOpened():
                                    raise RuntimeError(f"Cannot write {gs_video}")
                            writer.write(frame)
                    finally:
                        cap.release()
            finally:
                if writer is not None:
                    writer.release()
            if not gs_video.is_file() or gs_video.stat().st_size == 0:
                raise RuntimeError("GS trajectory video was not created")
            print(f"[TRAJECTORY VIDEO] {gs_video} ({len(rendered_paths)} paths)", flush=True)
            return recon
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.da3.model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def stage_video(self, case_dir: Path) -> None:
        """Reserved for trajectory-conditioned generation; intentionally disabled."""
        raise NotImplementedError("Multi-image video generation is not implemented yet")

    def __call__(self, work_dir: str | Path, export_dir: str | Path | None = None) -> dict[str, Any]:
        case = Path(work_dir).expanduser().resolve()
        sources = input_images(case / "image")
        reconstruction = case / "reconstruction"
        target_images = reconstruction / "image"
        colmap = reconstruction / "sparse" / "0"
        gs_video = reconstruction / "gs_render.mp4"
        dense_export = Path(export_dir).expanduser().resolve() if export_dir else reconstruction / "dense"
        journal_path = reconstruction / "stage_status.json"
        reconstruction.mkdir(parents=True, exist_ok=True)

        # Do not reuse single-image generated frames as multi-image DA3 inputs.
        target_images.mkdir(parents=True, exist_ok=True)
        manifest_path = reconstruction / "multi_input_manifest.json"
        manifest = [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                    for p in sources]
        old_manifest = None
        if manifest_path.exists():
            try:
                old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if old_manifest != manifest:
            # Stale DA3 poses must never be reused with a different set of images.
            if self.resume_stages and (old_manifest is not None or _colmap_valid(colmap)):
                raise RuntimeError("Input images changed since prior multi-image run; "
                                   "disable resume_stages or use a clean reconstruction directory")
            for p in target_images.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                    p.unlink()
        for src in sources:
            dst = target_images / src.name
            if src.resolve() != dst.resolve():
                if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # The current custom DA3 glob handles JPG or PNG, not mixed extensions.
        # Normalize names/extensions for deterministic DA3/COLMAP correspondence.
        from PIL import Image
        normalized = reconstruction / "da3_input"
        normalized.mkdir(parents=True, exist_ok=True)
        for old in normalized.glob("*.jpg"):
            old.unlink()
        for i, source in enumerate(sources):
            with Image.open(source) as im:
                im.convert("RGB").save(normalized / f"{i:06d}.jpg", quality=95)

        # Dense uses reconstruction/image; align its filenames with COLMAP.
        for p in target_images.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                p.unlink()
        for p in sorted(normalized.glob("*.jpg")):
            shutil.copy2(p, target_images / p.name)

        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.exists() else {}
        except (OSError, ValueError):
            journal = {}
        stages = journal.setdefault("stages", {})
        timings: dict[str, float] = {}
        results: dict[str, Any] = {"input_views": len(sources), "video_generation": "disabled",
                                   "trajectory_render": self.render_trajectories}

        def persist() -> None:
            journal["pipeline"] = "multi_image"
            journal["updated_at"] = datetime.now(timezone.utc).isoformat()
            tmp = journal_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(journal, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(journal_path)

        def da3_stage():
            colmap.mkdir(parents=True, exist_ok=True)
            if self.render_trajectories:
                return self._da3_with_trajectory(target_images, colmap, gs_video)
            return self.da3.infer(args=DA3Params(
                data_path=str(target_images), colmap_path=str(colmap),
                gs_render_vid_path=str(gs_video), view_num=len(sources),
                crop_ratio=self.crop_ratio, extra_view=[False, False]))

        def dense_stage():
            if not _colmap_valid(colmap):
                raise RuntimeError(f"Missing DA3 COLMAP initialization: {colmap}")
            dense_export.mkdir(parents=True, exist_ok=True)
            result = self.dense(work_dir=reconstruction, export_dir=dense_export,
                                mode=self.mode, circle=self.circle, max_frames=self.max_frames)
            if result is False:
                raise RuntimeError("Dense reconstruction returned False")
            (dense_export / ".dense_complete").write_text("ok\n", encoding="utf-8")
            return result

        operations = [("da3", da3_stage)]
        if self.run_3dgs_iter:
            operations.append(("dense", dense_stage))
        _synchronize_cuda()
        start = time.perf_counter()
        try:
            for label, operation in operations:
                valid = (_colmap_valid(colmap) and gs_video.is_file() and gs_video.stat().st_size > 0
                         if label == "da3" else (dense_export / ".dense_complete").is_file())
                if self.resume_stages and label == "da3" and self.render_trajectories:
                    valid = valid and (reconstruction / "trajectories" / "manifest.json").is_file()
                if self.resume_stages and valid and (
                    label != "dense" or stages.get(label, {}).get("status") == "ok"
                ):
                    print(f"[MULTI RESUME] {case.name}: skip {label}", flush=True)
                    stages[label] = {"status": "ok", "reused": True}
                    persist()
                    continue
                stages[label] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
                persist()
                _synchronize_cuda()
                t0 = time.perf_counter()
                try:
                    value = operation()
                    if label == "da3":
                        if not _colmap_valid(colmap):
                            raise RuntimeError(f"DA3 did not write COLMAP poses: {colmap}")
                        if not gs_video.is_file() or gs_video.stat().st_size == 0:
                            raise RuntimeError(f"DA3 did not write GS video: {gs_video}")
                    else:
                        results["dense_result"] = value
                    stages[label] = {"status": "ok", "finished_at": datetime.now(timezone.utc).isoformat()}
                except BaseException as exc:
                    stages[label] = {"status": "failed", "error": str(exc),
                                     "finished_at": datetime.now(timezone.utc).isoformat()}
                    raise
                finally:
                    _synchronize_cuda()
                    timings[label] = round(time.perf_counter() - t0, 3)
                    stages[label]["seconds"] = timings[label]
                    persist()
                    print(f"[MULTI {case.name}] {label}: {timings[label]:.3f}s", flush=True)
        finally:
            _synchronize_cuda()
            timings["total"] = round(time.perf_counter() - start, 3)
            (reconstruction / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
        results["timings"] = timings
        return results
