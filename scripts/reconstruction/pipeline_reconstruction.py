"""String-configured, reusable SparseRecon-compatible reconstruction entrypoint."""
from __future__ import annotations

import json
from datetime import datetime, timezone
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .multistage_pipeline import (
        ReconstructionConfig, ReconstructionPipeline, FRAME_INDICES, load_callable, _synchronize_cuda
    )
except ImportError:
    from multistage_pipeline import (
        ReconstructionConfig, ReconstructionPipeline, FRAME_INDICES, load_callable, _synchronize_cuda
    )


class SparseRecon:
    """Accept JSON text, a JSON-file path, or an already-decoded dict.

    Initialize models once, then call the same object for multiple cases.
    """

    def __init__(self, json_args: str | dict[str, Any]):
        if isinstance(json_args, str):
            maybe_file = Path(json_args)
            if json_args.lstrip().startswith("{"):
                json_args = json.loads(json_args)
            elif maybe_file.is_file():
                json_args = json.loads(maybe_file.read_text(encoding="utf-8"))
            else:
                raise ValueError("Expected a JSON object string or an existing JSON file")
        if not isinstance(json_args, dict):
            raise TypeError("SparseRecon expects JSON string or dict")
        self.args = json_args.copy()
        root = self.args.get("external_code_root") or str(Path(__file__).resolve().parent)
        video_root = Path(self.args.get("video_code_root") or
                          (Path(__file__).resolve().parent / "VideoX-Fun")).resolve()
        if not video_root.is_dir():
            raise FileNotFoundError(f"VideoX-Fun directory not found: {video_root}")
        if str(video_root) not in sys.path:
            sys.path.insert(0, str(video_root))
        os.environ["SINGLERECON_WORLD_SIZE"] = str(self.args.get("world_size", 1))
        os.environ["SINGLERECON_FPS"] = str(self.args.get("gen_vid_fps", 30))
        if self.args.get("lora_path"):
            os.environ["SINGLERECON_LORA_PATH"] = str(self.args["lora_path"])
        if self.args.get("lora_weight") is not None:
            os.environ["SINGLERECON_LORA_WEIGHT"] = str(self.args["lora_weight"])
        os.environ["SINGLERECON_USE_LIGHTVAE"] = "1" if self.args.get("use_lightvae", False) else "0"
        condition_frames = int(self.args.get("condition_vae_encode_frames", 49)) if self.args.get("condition_vae_fast_encode", False) else 0
        os.environ["SINGLERECON_CONDITION_PREFIX_FRAMES"] = str(condition_frames)
        print(f"[CONDITION VAE] fast_encode={condition_frames > 0}, encode_frames={condition_frames or 121}", flush=True)
        if self.args.get("lightvae_path"):
            os.environ["SINGLERECON_LIGHTVAE_PATH"] = str(self.args["lightvae_path"])
        if self.args.get("prompt_embeds_path"):
            os.environ["SINGLERECON_PROMPT_EMBEDS_PATH"] = str(Path(self.args["prompt_embeds_path"]).resolve())
        else:
            os.environ.pop("SINGLERECON_PROMPT_EMBEDS_PATH", None)
        video_spec = self.args.get("video_adapter") or (
            "predict_ti2v:Wan_5b" if self.args.get("run_gen_video", True) else None
        )
        da3_spec = self.args.get("da3_factory")
        da3_backend = self.args.get("da3_backend", "legacy")
        if da3_backend not in ("legacy", "official"):
            raise ValueError(f"Unsupported da3_backend: {da3_backend}")
        if da3_backend == "official" and da3_spec:
            raise ValueError("Use either da3_backend=official or da3_factory, not both")
        da3_instance = None
        if da3_backend == "official" and self.args.get("run_DA3", True):
            try:
                from .da3_official_adapter import create_official_da3
            except ImportError:
                from da3_official_adapter import create_official_da3
            da3_repo_root = self.args.get("da3_repo_root")
            da3_model_path = self.args.get("da3_model_path")
            if not da3_repo_root or not da3_model_path:
                raise ValueError("official DA3 requires da3_repo_root and da3_model_path")
            da3_instance = create_official_da3(
                repo_root=da3_repo_root,
                model_path=da3_model_path,
                process_res=int(self.args.get("da3_process_res", 504)),
                process_res_method=self.args.get("da3_process_res_method", "upper_bound_resize"),
                preprocess_workers=int(self.args.get("da3_preprocess_workers", 8)),
                profile=bool(self.args.get("da3_profile", True)),
            )
        dense_spec = self.args.get("dense_adapter")
        self.pipeline = ReconstructionPipeline(
            ReconstructionConfig(
                case_dir=Path("."),
                view_num=int(self.args.get("view_num", 121)),
                recon_view_num=int(self.args.get("recon_view_num", 40)),
                crop_ratio=float(self.args.get("crop_ratio", 1.0)),
                use_view_extrapolation=bool(self.args.get("use_view_extrapolation", False)),
                is_livephoto=bool(self.args.get("is_livephoto", True)),
                mode=str(self.args.get("mode", "scene")),
                circle=bool(self.args.get("circle", False)),
                max_frames=int(self.args.get("max_frames", 121)),
                external_code_root=Path(root),
                dense_code_root=Path(self.args.get("dense_code_root") or Path(__file__).resolve().parent),
                da3_model_path=(Path(self.args["da3_model_path"])
                                if self.args.get("da3_model_path") else None),
            ),
            generate_video=(load_callable(video_spec)() if video_spec else None),
            run_DA3=da3_instance if da3_instance is not None else (load_callable(da3_spec)() if da3_spec else None),
            recon_iter_full=load_callable(dense_spec) if dense_spec else None,
        )
        self.pipeline.captioner_cpu_swap = bool(self.args.get("captioner_cpu_swap", False))
        self.pipeline.caption_force_regenerate = bool(self.args.get("caption_force_regenerate", False))
        captioner_enabled = bool(self.args.get("run_captioner", True))
        if captioner_enabled:
            captioner_root = Path(self.args.get("captioner_code_root") or
                                  Path(__file__).resolve().parent).resolve()
            if str(captioner_root) not in sys.path:
                sys.path.insert(0, str(captioner_root))
            from qwen_vl3_images_caption_parallel import ImagesCaptioner
            captioner_model = Path(self.args.get("captioner_model_path") or
                                   (Path(__file__).resolve().parent /
                                    "Cloud_Models/Qwen3-VL-8B-Instruct"))
            gpu_count = int(self.args.get("captioner_num_gpus", self.args.get("world_size", 1)))
            self.pipeline.img_captioner = ImagesCaptioner(
                model_path=str(captioner_model),
                num_gpus=max(1, min(gpu_count, 4)),
            )
        self.pipeline.video_infer_steps = int(self.args.get("gen_vid_infer_step", 8))
        self.pipeline.video_fps = int(self.args.get("gen_vid_fps", 30))
        start = time.perf_counter()
        if self.args.get("run_gen_video", True):
            self.pipeline._preload_video()
        if self.args.get("run_DA3", True):
            self.pipeline._preload_da3()
        if self.args.get("run_3dgs_iter", True):
            self.pipeline._preload_dense()
        self.load_seconds = time.perf_counter() - start
        print(f"[PRELOAD] {self.load_seconds:.3f}s (excluded from case timings)", flush=True)

    def __call__(self, work_dir: str | Path, export_dir: str | Path | None = None,
                 mode: str = "scene", circle: bool | None = None) -> dict[str, Any]:
        pipeline = self.pipeline
        pipeline.set_case(Path(work_dir))
        pipeline.args.mode = mode
        if circle is not None:
            pipeline.args.circle = circle
        if export_dir is not None:
            pipeline.export_dir = Path(export_dir).resolve()
            pipeline.export_dir.mkdir(parents=True, exist_ok=True)
        # Persistent per-case stage journal; interrupted runs resume at the first
        # incomplete stage, rather than repeating costly video inference.
        journal_path = pipeline.gen_vid_path / "stage_status.json"
        pipeline.gen_vid_path.mkdir(parents=True, exist_ok=True)
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.exists() else {}
        except (ValueError, OSError):
            journal = {}
        stages = journal.setdefault("stages", {})
        timings: dict[str, float] = {}
        results: dict[str, Any] = {}

        def persist():
            journal["updated_at"] = datetime.now(timezone.utc).isoformat()
            tmp = journal_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(journal, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(journal_path)

        def colmap_valid():
            base = pipeline.colmap_path
            for folder in (base,):
                if all((folder / f"{name}.txt").is_file() for name in ("cameras", "images", "points3D")):
                    if (folder / "images.txt").stat().st_size > 0:
                        return True
                if all((folder / f"{name}.bin").is_file() for name in ("cameras", "images", "points3D")):
                    if (folder / "images.bin").stat().st_size > 8:
                        return True
            return False

        def video_valid():
            return pipeline.video_path.is_file() and pipeline.video_path.stat().st_size > 0 and all(
                (pipeline.img_path / f"{i:06d}.jpg").is_file() for i in FRAME_INDICES
            )

        def stage_valid(name):
            if name == "caption":
                p = pipeline.case_dir / "prompt.txt"
                return p.is_file() and bool(p.read_text(encoding="utf-8").strip())
            if name == "video_and_sampling":
                return video_valid()
            if name == "da3":
                return colmap_valid()
            if name == "dense":
                return (pipeline.export_dir / ".dense_complete").is_file()
            return False

        operations = (
            (bool(self.args.get("run_captioner", True) and self.args.get("run_gen_video", True)), "caption", pipeline.stage_caption),
            (bool(self.args.get("run_gen_video", True)), "video_and_sampling", lambda: pipeline.stage_video(reuse=bool(self.args.get("resume_stages", False)) and video_valid())),
            (bool(self.args.get("run_DA3", True)), "da3", pipeline.stage_da3),
            (bool(self.args.get("run_3dgs_iter", True)), "dense", pipeline.stage_dense),
        )
        _synchronize_cuda()
        started = time.perf_counter()
        try:
            for enabled, label, operation in operations:
                if not enabled:
                    continue
                if self.args.get("resume_stages", False) and stage_valid(label) and (stages.get(label, {}).get("status") == "ok" or label != "dense"):
                    print(f"[RESUME] {pipeline.case_dir.name}: skip completed {label}", flush=True)
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
                        if not colmap_valid():
                            raise RuntimeError(f"DA3 returned but COLMAP poses missing/invalid: {pipeline.colmap_path}")
                        results["gen_index"] = value
                    elif label == "dense":
                        # Some backends log an error without raising; their return
                        # value cannot be trusted as a success signal alone.
                        if not colmap_valid():
                            raise RuntimeError(f"COLMAP poses missing before/after dense stage: {pipeline.colmap_path}")
                        if value is False:
                            raise RuntimeError("Dense reconstruction returned False")
                        results["dense_result"] = value
                        (pipeline.export_dir / ".dense_complete").write_text("ok\\n", encoding="utf-8")
                    stages[label] = {"status": "ok", "finished_at": datetime.now(timezone.utc).isoformat()}
                except BaseException as exc:
                    stages[label] = {"status": "failed", "error": str(exc), "finished_at": datetime.now(timezone.utc).isoformat()}
                    raise
                finally:
                    _synchronize_cuda()
                    timings[label] = round(time.perf_counter() - t0, 3)
                    stages[label]["seconds"] = timings[label]
                    persist()
                    print(f"[{pipeline.case_dir.name}] {label}: {timings[label]:.3f}s", flush=True)
        finally:
            _synchronize_cuda()
            timings["total"] = round(time.perf_counter() - started, 3)
            (pipeline.gen_vid_path / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
            print(f"[{pipeline.case_dir.name}] TOTAL excluding model preload: {timings['total']:.3f}s", flush=True)
        results["timings"] = timings
        return results
