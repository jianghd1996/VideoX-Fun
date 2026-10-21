"""String-configured, reusable SparseRecon-compatible reconstruction entrypoint."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .multistage_pipeline import (
        ReconstructionConfig, ReconstructionPipeline, load_callable, _synchronize_cuda
    )
except ImportError:
    from multistage_pipeline import (
        ReconstructionConfig, ReconstructionPipeline, load_callable, _synchronize_cuda
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
        root = self.args.get("external_code_root")
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
        video_spec = self.args.get("video_adapter") or (
            "predict_ti2v:Wan_5b" if self.args.get("run_gen_video", True) else None
        )
        da3_spec = self.args.get("da3_factory")
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
                external_code_root=Path(root) if root else None,
                da3_model_path=(Path(self.args["da3_model_path"])
                                if self.args.get("da3_model_path") else None),
            ),
            generate_video=(load_callable(video_spec)() if video_spec else None),
            run_DA3=load_callable(da3_spec)() if da3_spec else None,
            recon_iter_full=load_callable(dense_spec) if dense_spec else None,
        )
        self.pipeline.captioner_cpu_swap = bool(self.args.get("captioner_cpu_swap", False))
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
            gpu_count = int(self.args.get("world_size", 1))
            self.pipeline.img_captioner = ImagesCaptioner(
                model_path=str(captioner_model),
                num_gpus=min(gpu_count, 4),
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
        timings: dict[str, float] = {}
        results: dict[str, Any] = {}
        _synchronize_cuda()
        started = time.perf_counter()
        try:
            for enabled, label, operation in (
                (self.args.get("run_captioner", True) and
                 self.args.get("run_gen_video", True) and
                 not (self.args.get("reuse_video", False) and pipeline.video_path.is_file()),
                 "caption", pipeline.stage_caption),
                (self.args.get("run_gen_video", True), "video_and_sampling",
                 lambda: pipeline.stage_video(reuse=bool(self.args.get("reuse_video", False)))),
                (self.args.get("run_DA3", True), "da3", pipeline.stage_da3),
                (self.args.get("run_3dgs_iter", True), "dense", pipeline.stage_dense),
            ):
                if not enabled:
                    continue
                _synchronize_cuda()
                t0 = time.perf_counter()
                try:
                    value = operation()
                    if label == "da3":
                        results["gen_index"] = value
                    elif label == "dense":
                        results["dense_result"] = value
                finally:
                    _synchronize_cuda()
                    timings[label] = round(time.perf_counter() - t0, 3)
                    print(f"[{pipeline.case_dir.name}] {label}: {timings[label]:.3f}s", flush=True)
        finally:
            _synchronize_cuda()
            timings["total"] = round(time.perf_counter() - started, 3)
            pipeline.gen_vid_path.mkdir(parents=True, exist_ok=True)
            (pipeline.gen_vid_path / "timings.json").write_text(
                json.dumps(timings, indent=2), encoding="utf-8")
            print(f"[{pipeline.case_dir.name}] TOTAL excluding model preload: {timings['total']:.3f}s",
                  flush=True)
        results["timings"] = timings
        return results
