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
                 dense_generated_frames: int = 80,
                 resume_stages: bool = False,
                 force_rerun_dense: bool = False,
                 crop_ratio: float = 1.0,
                 mode: str = "scene",
                 circle: bool = False,
                 max_frames: int = 121,
                 da3_instance: Any = None,
                 dense_instance: Any = None,
                 render_trajectories: bool = True,
                 trajectory_extra: bool = False,
                 trajectory_view_num: int = 81,
                 trajectory_interpolation: str = "bezier",
                 run_gen_video: bool = False,
                 video_adapter: str | None = None,
                 video_code_root: str | Path | None = None,
                 video_lora_path: str | Path | None = None,
                 video_lora_weight: float = 0.55,
                 video_cfg: float = 0.0,
                 video_short_edge: int = 1088,
                 video_use_lightvae: bool = True,
                 video_lightvae_path: str | Path | None = None,
                 video_teacher_vae_path: str | Path | None = None,
                 video_fps: int = 30,
                 video_gpu_ids: list[int] | None = None,
                 video_resume: bool = False,
                 run_captioner: bool = True,
                 captioner_code_root: str | Path | None = None,
                 captioner_model_path: str | Path | None = None,
                 captioner_cpu_swap: bool = True,
                 captioner_num_gpus: int = 1):
        root = Path(external_code_root).expanduser().resolve()
        dense_root = Path(dense_code_root or root).expanduser().resolve()
        self.run_3dgs_iter = bool(run_3dgs_iter)
        self.dense_generated_frames = int(dense_generated_frames)
        self.force_rerun_dense = bool(force_rerun_dense)
        if self.dense_generated_frames < 1:
            raise ValueError("dense_generated_frames must be positive")
        self.render_trajectories = bool(render_trajectories)
        self.trajectory_extra = bool(trajectory_extra)
        self.trajectory_view_num = int(trajectory_view_num)
        self.trajectory_interpolation = trajectory_interpolation
        self.run_gen_video = bool(run_gen_video)
        self.video_fps = int(video_fps)
        # One GPU: existing sequential adapter; multiple: one process per GPU.
        self.video_gpu_ids = list(video_gpu_ids if video_gpu_ids is not None else [0])
        if not self.video_gpu_ids or len(set(self.video_gpu_ids)) != len(self.video_gpu_ids):
            raise ValueError(f"video_gpu_ids must contain distinct GPU IDs: {self.video_gpu_ids}")
        self.video_code_root = Path(video_code_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
        self.video_adapter = video_adapter
        self.video_environment = {
            "SINGLERECON_CONTROL_LORA_PATH": str(Path(video_lora_path).expanduser().resolve()) if video_lora_path else "",
            "SINGLERECON_CONTROL_LORA_WEIGHT": str(float(video_lora_weight)),
            "SINGLERECON_CONTROL_CFG": str(float(video_cfg)),
            "SINGLERECON_CONTROL_SHORT_EDGE": str(int(video_short_edge)),
            "SINGLERECON_CONTROL_USE_LIGHTVAE": "1" if video_use_lightvae else "0",
            "SINGLERECON_CONTROL_LIGHTVAE_PATH": str(
                Path(video_lightvae_path or
                     (dense_root / "Cloud_Models/transformer_w_skybox_sla/MG-LightVAE.pth"))
                .expanduser().resolve()),
            "SINGLERECON_CONTROL_TEACHER_VAE_PATH": str(
                Path(video_teacher_vae_path or
                     (dense_root / "Cloud_Models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"))
                .expanduser().resolve()),
        }
        self.video_model_load_seconds = 0.0
        self.video_resume = bool(video_resume)
        self.run_captioner = bool(run_captioner)
        self.captioner_cpu_swap = bool(captioner_cpu_swap)
        self.captioner_num_gpus = int(captioner_num_gpus)
        self.captioner_code_root = Path(captioner_code_root or dense_root).expanduser().resolve()
        self.captioner_model_path = Path(captioner_model_path or (dense_root / "Cloud_Models/Qwen3-VL-8B-Instruct")).expanduser().resolve()
        self.img_captioner = None
        self.video_generator = None
        if self.run_gen_video:
            if not self.render_trajectories:
                raise ValueError("run_gen_video requires render_trajectories=True")
            if not video_adapter:
                raise ValueError("run_gen_video requires a control-video-capable video_adapter")
            try:
                from .multistage_pipeline import load_callable
            except ImportError:
                from multistage_pipeline import load_callable
            if not self.video_code_root.is_dir():
                raise FileNotFoundError(f"Missing Wan video code root: {self.video_code_root}")
            # Multi-GPU workers import Wan only after setting CUDA_VISIBLE_DEVICES.
            if len(self.video_gpu_ids) == 1:
                if str(self.video_code_root) not in sys.path:
                    sys.path.insert(0, str(self.video_code_root))
                for key, value in self.video_environment.items():
                    os.environ[key] = value
                self.video_generator_factory = load_callable(video_adapter)
            print(f"[MULTI VIDEO MODE] {('multi_gpu' if len(self.video_gpu_ids) > 1 else 'single_gpu')} "
                  f"GPU IDs={self.video_gpu_ids}", flush=True)
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
                interpolation=self.trajectory_interpolation)
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

    def stage_caption(self, case_dir: Path) -> str:
        """Use the same Qwen3-VL ImagesCaptioner as the single-image pipeline.

        Caption the original multi-image folder once per case. Reuse prompt.txt
        if it is already nonempty; never regenerate for individual trajectories.
        """
        prompt_file = case_dir / "prompt.txt"
        if prompt_file.is_file():
            prompt = prompt_file.read_text(encoding="utf-8").strip()
            if prompt:
                print(f"[MULTI CAPTION] reuse {prompt_file}", flush=True)
                return prompt
        if not self.run_captioner:
            raise FileNotFoundError(
                f"Missing nonempty {prompt_file}; enable run_captioner or provide prompt.txt"
            )
        if str(self.captioner_code_root) not in sys.path:
            sys.path.insert(0, str(self.captioner_code_root))
        if self.img_captioner is None:
            from qwen_vl3_images_caption_parallel import ImagesCaptioner
            self.img_captioner = ImagesCaptioner(
                model_path=str(self.captioner_model_path),
                num_gpus=self.captioner_num_gpus,
            )
        import torch
        model = self.img_captioner.model
        if self.captioner_cpu_swap:
            model.to("cuda")
        try:
            # Legacy captioner globs JPG only. This folder contains normalized
            # copies of all original PNG/JPG/etc inputs (no generated frames).
            caption_dir = case_dir / "reconstruction" / "da3_input"
            if not caption_dir.is_dir():
                raise FileNotFoundError(f"Missing normalized input images: {caption_dir}")
            prompt = self.img_captioner.read_image_caption(str(caption_dir))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Empty image caption for {case_dir}")
            prompt = prompt.strip()
            prompt_file.write_text(prompt, encoding="utf-8")
            print(f"[MULTI CAPTION] {case_dir.name}: {prompt}", flush=True)
            return prompt
        finally:
            if self.captioner_cpu_swap:
                model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def stage_video(self, case_dir: Path) -> list[dict[str, Any]]:
        """Dispatch based on configured GPU count; both paths return same records."""
        try:
            from .multi_image_video import run_trajectory_videos
        except ImportError:
            from multi_image_video import run_trajectory_videos

        prompt_file = case_dir / "prompt.txt"
        prompt = prompt_file.read_text(encoding="utf-8").strip() if prompt_file.is_file() else ""
        if not prompt:
            raise FileNotFoundError(f"Missing nonempty {prompt_file}")
        arguments = dict(
            reconstruction_dir=case_dir / "reconstruction",
            source_images=input_images(case_dir / "image"),
            fps=self.video_fps, prompt=prompt, resume=self.video_resume)

        if len(self.video_gpu_ids) > 1:
            try:
                from .multi_image_video_multi_gpu import run_trajectory_videos_multi_gpu
            except ImportError:
                from multi_image_video_multi_gpu import run_trajectory_videos_multi_gpu
            results = run_trajectory_videos_multi_gpu(
                **arguments,
                adapter_spec=self.video_adapter,
                video_code_root=self.video_code_root,
                gpu_ids=self.video_gpu_ids,
                env_overrides=self.video_environment)
            summary_path = case_dir / "reconstruction" / "generated_trajectories" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            loads = summary.get("model_load_by_gpu", {})
            # Critical-path model load (the workers load concurrently).
            self.video_model_load_seconds = max(loads.values(), default=0.0)
            return results

        if self.video_generator is None:
            factory = self.video_generator_factory
            load_started = time.perf_counter()
            self.video_generator = factory() if isinstance(factory, type) else factory
            preload = getattr(self.video_generator, "load", None)
            if callable(preload):
                preload()
            _synchronize_cuda()
            self.video_model_load_seconds = round(time.perf_counter() - load_started, 3)
            print(f"[MULTI VIDEO MODEL LOAD] {self.video_model_load_seconds:.3f}s", flush=True)
        return run_trajectory_videos(**arguments, generator=self.video_generator)

    def __call__(self, work_dir: str | Path, export_dir: str | Path | None = None) -> dict[str, Any]:
        case_started = time.perf_counter()
        case = Path(work_dir).expanduser().resolve()
        sources = input_images(case / "image")
        reconstruction = case / "reconstruction"
        target_images = reconstruction / "image"
        colmap = reconstruction / "sparse" / "0"
        gs_video = reconstruction / "gs_render.mp4"
        dense_export = Path(export_dir).expanduser().resolve() if export_dir else reconstruction / "dense"
        journal_path = reconstruction / "stage_status.json"
        reconstruction.mkdir(parents=True, exist_ok=True)

        image_prep_started = time.perf_counter()
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

        image_preprocess_seconds = round(time.perf_counter() - image_prep_started, 3)
        print(f"[MULTI IMAGE PREPROCESS] {case.name}: {image_preprocess_seconds:.3f}s", flush=True)

        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.exists() else {}
        except (OSError, ValueError):
            journal = {}
        stages = journal.setdefault("stages", {})
        timings: dict[str, float] = {"image_preprocess": image_preprocess_seconds}
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

        def dense_prepare_stage():
            try:
                from .multi_image_dense_input import prepare_dense_input
            except ImportError:
                from multi_image_dense_input import prepare_dense_input
            if not (reconstruction / "generated_trajectories" / "manifest.json").is_file():
                raise FileNotFoundError("Dense reconstruction requires completed generated trajectories")
            return prepare_dense_input(
                reconstruction_dir=reconstruction,
                sample_count=self.dense_generated_frames)

        def dense_stage():
            dense_input = reconstruction / "dense_input"
            dense_colmap = dense_input / "sparse" / "0"
            if not _colmap_valid(dense_colmap):
                raise RuntimeError(f"Missing merged generated-video COLMAP: {dense_colmap}")
            # Reconstruction_full may initialize from sparse/0/points3D.ply
            # instead of COLMAP points3D.bin. Never run with missing/empty PLY.
            from pathlib import Path as _Path
            cloud = dense_colmap / "points3D.ply"
            if not cloud.is_file() or cloud.stat().st_size < 256:
                raise RuntimeError(f"Missing or invalid 3DGS initialization PLY: {cloud}")
            if not any((dense_input / "image").glob("*.jpg")):
                raise RuntimeError(f"No sampled generated frames in {dense_input / 'image'}")
            if self.force_rerun_dense and dense_export.exists() and any(dense_export.iterdir()):
                # Keep the previous reconstruction as a backup while forcing a
                # clean run; rename is cheap on the same filesystem.
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                backup = dense_export.with_name(dense_export.name + "_before_ply_fix_" + stamp)
                if backup.exists():
                    raise FileExistsError(f"Dense backup already exists: {backup}")
                dense_export.rename(backup)
                print(f"[DENSE RERUN] preserved previous results at {backup}", flush=True)
            dense_export.mkdir(parents=True, exist_ok=True)
            result = self.dense(work_dir=dense_input, export_dir=dense_export,
                                mode=self.mode, circle=self.circle, max_frames=self.max_frames)
            if result is False:
                raise RuntimeError("Dense reconstruction returned False")
            (dense_export / ".dense_complete").write_text("ok\n", encoding="utf-8")
            return result

        operations = []
        if self.run_gen_video:
            operations.append(("caption", lambda: self.stage_caption(case)))
        operations.append(("da3", da3_stage))
        if self.run_gen_video:
            operations.append(("trajectory_video", lambda: self.stage_video(case)))
        if self.run_3dgs_iter:
            operations.append(("dense_prepare", dense_prepare_stage))
            operations.append(("dense", dense_stage))
        _synchronize_cuda()
        start = time.perf_counter()
        try:
            for label, operation in operations:
                if label == "caption":
                    p = case / "prompt.txt"
                    valid = p.is_file() and bool(p.read_text(encoding="utf-8").strip())
                elif label == "da3":
                    valid = (_colmap_valid(colmap) and gs_video.is_file() and gs_video.stat().st_size > 0)
                elif label == "trajectory_video":
                    # Video resume is enforced inside the video stage, per MP4.
                    valid = False
                elif label == "dense_prepare":
                    # Regenerate the input mapping to avoid stale frame selections.
                    valid = False
                else:
                    valid = ((dense_export / ".dense_complete").is_file()
                             and not self.force_rerun_dense)
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
                    elif label == "caption":
                        results["prompt"] = value
                    elif label == "dense_prepare":
                        results["dense_input"] = value
                        timings["dense_input_views"] = value["total_views"]
                    elif label == "trajectory_video":
                        results["trajectory_videos"] = value
                        timings["video_model_load"] = self.video_model_load_seconds
                        timings["video_generation"] = round(sum(
                            float(item["seconds"]) for item in value), 3)
                        timings["video_gpu_count"] = len(self.video_gpu_ids)
                        timings["video_inference"] = round(sum(
                            float(item.get("inference_timing", {}).get("inference", 0))
                            for item in value), 3)
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
            timings["stages_total"] = round(time.perf_counter() - start, 3)
            timings["total"] = round(time.perf_counter() - case_started, 3)
            timings["model_preload_once"] = self.load_seconds
            (reconstruction / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
            print(f"[MULTI TIMING] {case.name}: "
                  + " ".join(f"{key}={value:.3f}s" for key, value in timings.items()),
                  flush=True)
        results["timings"] = timings
        return results
