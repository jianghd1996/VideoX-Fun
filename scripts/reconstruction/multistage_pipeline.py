#!/usr/bin/env python3
"""Three-stage single-image generative reconstruction orchestration.

External model integrations are intentionally injected, not imported here.
Run with --dry-run to inspect paths before wiring model implementations.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

FRAME_INDICES = (0, 6, 14, 16, 17, 18, 19, 20, 21, 23, 28, 30, 34, 39,
                 40, 41, 42, 43, 44, 45, 54, 75, 76, 77, 78, 79, 80, 81,
                 82, 88, 91, 95, 99, 100, 101, 102, 103, 104, 105, 106)
VIDEO_FRAMES = 121
RECON_VIEW_NUM = len(FRAME_INDICES)


@dataclass
class DA3Params:
    data_path: str
    colmap_path: str
    gs_render_vid_path: str
    view_num: int
    crop_ratio: float
    extra_view: list[bool]


@dataclass
class ReconstructionConfig:
    case_dir: Path
    view_num: int = VIDEO_FRAMES
    recon_view_num: int = RECON_VIEW_NUM
    crop_ratio: float = 1.0
    use_view_extrapolation: bool = False
    is_livephoto: bool = True
    mode: str = "default"
    circle: bool = False
    max_frames: int = 121
    external_code_root: Path | None = None
    da3_model_path: Path | None = None


def load_callable(spec: str) -> Callable[..., Any]:
    """Import a user-supplied adapter, e.g. my_adapters:generate_video."""
    module, sep, name = spec.partition(":")
    if not sep or not module or not name:
        raise ValueError("Adapter must use module:function notation")
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError(f"Adapter is not callable: {spec}")
    return fn


def _external_import_root(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"External code root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def create_da3(root: Path, model_path: Path | None = None) -> Any:
    """Initialize the external DA3 implementation only when stage 2 runs."""
    root = _external_import_root(root)
    from DA3.run_da3_full_crop_dev import run_DA3
    checkpoint = (Path(model_path).expanduser().resolve() if model_path else
                  root / "Cloud_Models/Saliency-Face-DA3/besthumanmodel_lr1e_6_ckpt2.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DA3 checkpoint not found: {checkpoint}")
    return run_DA3(str(checkpoint))


def create_full_reconstruction(root: Path) -> Callable[..., Any]:
    """Initialize the camera-optimizing dense reconstruction backend."""
    _external_import_root(root)
    from Reconstruction_full import Reconstruction
    return Reconstruction(preload=True, dense_init_scene=False)


def create_vanilla_reconstruction(root: Path) -> Callable[..., Any]:
    """Optional vanilla backend for later stages."""
    _external_import_root(root)
    from Reconstruction import Reconstruction
    return Reconstruction(preload=True, dense_init_scene=False)


class ReconstructionPipeline:
    def __init__(self, config: ReconstructionConfig, *,
                 generate_video: Callable[..., Any] | None = None,
                 run_DA3: Any = None,
                 recon_iter_full: Callable[..., Any] | None = None):
        self.args = config
        self.case_dir = Path(config.case_dir).resolve()
        self.source_img_path = self.case_dir / "image"
        self.gen_vid_path = self.case_dir / "reconstruction"
        self.img_path = self.gen_vid_path / "image"
        self.video_path = self.gen_vid_path / "generated.mp4"
        self.colmap_path = self.gen_vid_path / "sparse" / "0"
        self.gs_render_vid_path = self.gen_vid_path / "gs_render.mp4"
        self.export_dir = self.gen_vid_path / "dense"
        self.max_frames = config.max_frames
        self.generate_video = generate_video
        self.img_captioner = None
        self.captioner_cpu_swap = False
        self.run_DA3 = run_DA3
        self.recon_iter_full = recon_iter_full

    def set_case(self, case_dir: Path) -> None:
        """Switch case paths without reloading any model."""
        self.case_dir = Path(case_dir).resolve()
        self.source_img_path = self.case_dir / "image"
        self.gen_vid_path = self.case_dir / "reconstruction"
        self.img_path = self.gen_vid_path / "image"
        self.video_path = self.gen_vid_path / "generated.mp4"
        self.colmap_path = self.gen_vid_path / "sparse" / "0"
        self.gs_render_vid_path = self.gen_vid_path / "gs_render.mp4"
        self.export_dir = self.gen_vid_path / "dense"

    def preload(self, *, start_stage: int = 1, end_stage: int = 3) -> dict[str, float]:
        """Load all requested backends before case timing starts."""
        times = {}
        for name, enabled, load in (
            ("video", start_stage <= 1 <= end_stage, self._preload_video),
            ("da3", start_stage <= 2 <= end_stage, self._preload_da3),
            ("dense", start_stage <= 3 <= end_stage, self._preload_dense),
        ):
            if enabled:
                t0 = time.perf_counter()
                load()
                times[name] = time.perf_counter() - t0
                print(f"[PRELOAD] {name}: {times[name]:.2f}s", flush=True)
        return times

    def _preload_video(self) -> None:
        if self.generate_video is None:
            raise RuntimeError("Video adapter not provided")
        # An adapter can expose load() to initialize and retain its model once.
        preload = getattr(self.generate_video, "load", None)
        if callable(preload):
            preload()

    def _preload_da3(self) -> None:
        if self.run_DA3 is None and self.args.external_code_root is not None:
            self.run_DA3 = create_da3(self.args.external_code_root, self.args.da3_model_path)

    def _preload_dense(self) -> None:
        if self.recon_iter_full is None and self.args.external_code_root is not None:
            self.recon_iter_full = create_full_reconstruction(self.args.external_code_root)

    def input_image(self) -> Path:
        images = sorted(p for p in self.source_img_path.iterdir()
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
        if len(images) != 1:
            raise ValueError(f"Expected exactly one JPG in {self.source_img_path}, got {len(images)}")
        return images[0]

    def paths(self) -> dict[str, str]:
        return {k: str(v) for k, v in {
            "input_image": self.input_image(), "video": self.video_path,
            "sampled_images": self.img_path, "colmap": self.colmap_path,
            "gs_render_video": self.gs_render_vid_path, "dense": self.export_dir,
        }.items()}

    def stage_caption(self) -> str:
        """Reuse an existing prompt or generate it once per case."""
        prompt_path = self.case_dir / "prompt.txt"
        if prompt_path.is_file():
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            if prompt:
                return prompt
        if self.img_captioner is None:
            raise RuntimeError("Missing prompt.txt and captioner is not initialized")
        import torch
        if self.captioner_cpu_swap:
            self.img_captioner.model.to("cuda")
        try:
            prompt = self.img_captioner.read_image_caption(str(self.source_img_path))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("ImagesCaptioner returned an empty caption")
            prompt = prompt.strip()
            prompt_path.write_text(prompt, encoding="utf-8")
            print(f"[CAPTION] {self.case_dir.name}: {prompt}", flush=True)
            return prompt
        finally:
            if self.captioner_cpu_swap:
                self.img_captioner.model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def stage_video(self, *, reuse: bool = False) -> None:
        if self.args.view_num != VIDEO_FRAMES or self.args.recon_view_num != RECON_VIEW_NUM:
            raise ValueError("Current fixed sampling requires view_num=121 and recon_view_num=40")
        self.gen_vid_path.mkdir(parents=True, exist_ok=True)
        import torch.distributed as dist
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        if distributed:
            # Rank 1 waits here while rank 0 finishes captioning.
            dist.barrier()
        if not (reuse and self.video_path.is_file()):
            if self.generate_video is None:
                raise RuntimeError("Video generator missing; inject generate_video(image_path, output_path, num_frames)")
            if hasattr(self.generate_video, "infer"):
                from predict_ti2v import TI2VParams
                self.generate_video.infer(args=TI2VParams(
                    image_path=str(self.input_image()),
                    output_path=str(self.video_path),
                    view_num=self.args.view_num,
                    num_inference_steps=getattr(self, "video_infer_steps", 8),
                    fps=getattr(self, "video_fps", 30),
                ))
            else:
                self.generate_video(image_path=self.input_image(),
                                    output_path=self.video_path, num_frames=self.args.view_num)
        if distributed:
            # Both ranks finish DiT/LightVAE before rank 0 samples frames.
            dist.barrier()
            if dist.get_rank() != 0:
                return
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video generator did not create {self.video_path}")
        self.extract_frames()

    def extract_frames(self) -> None:
        import cv2
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open generated video: {self.video_path}")
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count != self.args.view_num:
                raise ValueError(f"Expected {self.args.view_num} frames, got {count}")
            self.img_path.mkdir(parents=True, exist_ok=True)
            for index in FRAME_INDICES:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode frame {index}")
                target = self.img_path / f"{index:06d}.jpg"
                if not cv2.imwrite(str(target), frame):
                    raise IOError(f"Failed to save {target}")
        finally:
            cap.release()
        (self.gen_vid_path / "sampled_frame_indices.json").write_text(
            json.dumps(list(FRAME_INDICES), indent=2), encoding="utf-8")

    def stage_da3(self) -> Any:
        if self.run_DA3 is None:
            raise RuntimeError("DA3 adapter missing; inject object with .infer(args=DA3Params(...))")
        if len(list(self.img_path.glob("*.jpg"))) != len(FRAME_INDICES):
            raise RuntimeError("40 sampled JPG frames are required before DA3")
        self.colmap_path.mkdir(parents=True, exist_ok=True)
        crop_ratio = self.args.crop_ratio
        return self.run_DA3.infer(args=DA3Params(
            data_path=str(self.img_path),
            colmap_path=str(self.colmap_path),
            gs_render_vid_path=str(self.gs_render_vid_path),
            view_num=self.args.recon_view_num,
            crop_ratio=crop_ratio,
            extra_view=[self.args.use_view_extrapolation, self.args.is_livephoto],
        ))

    def stage_dense(self) -> Any:
        if self.recon_iter_full is None:
            raise RuntimeError("Dense reconstruction adapter missing; inject recon_iter_full")
        if not self.img_path.is_dir() or not self.colmap_path.is_dir():
            raise RuntimeError("Sampled images and COLMAP directory are required")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        return self.recon_iter_full(
            work_dir=Path(self.gen_vid_path), export_dir=Path(self.export_dir),
            mode=self.args.mode, circle=self.args.circle, max_frames=self.max_frames)

    def run(self, *, start_stage: int = 1, end_stage: int = 3,
            reuse_video: bool = False) -> dict[str, Any]:
        if not 1 <= start_stage <= end_stage <= 3:
            raise ValueError("Stages must satisfy 1 <= start_stage <= end_stage <= 3")
        results: dict[str, Any] = {}
        timings: dict[str, float] = {}
        stages = [
            ("video_and_sampling", lambda: self.stage_video(reuse=reuse_video)),
            ("da3", self.stage_da3),
            ("dense", self.stage_dense),
        ]
        total_start = time.perf_counter()
        try:
            for number, (name, operation) in enumerate(stages, 1):
                if start_stage <= number <= end_stage:
                    started = time.perf_counter()
                    try:
                        value = operation()
                        if number == 2:
                            results["gen_index"] = value
                        elif number == 3:
                            results["dense_result"] = value
                    finally:
                        _synchronize_cuda()
                        timings[name] = time.perf_counter() - started
                        print(f"[{self.case_dir.name}] {name}: {timings[name]:.2f}s", flush=True)
        finally:
            _synchronize_cuda()
            timings["total"] = time.perf_counter() - total_start
            print(f"[{self.case_dir.name}] TOTAL (excluding preload): {timings['total']:.2f}s", flush=True)
            self.gen_vid_path.mkdir(parents=True, exist_ok=True)
            (self.gen_vid_path / "timings.json").write_text(
                json.dumps(timings, indent=2), encoding="utf-8")
        results["timings"] = timings
        return results


def _synchronize_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, nargs="?", help="Case directory; optional with --cases-root")
    parser.add_argument("--video-adapter", help="module:object with __call__ and optional load() for preload")
    parser.add_argument("--da3-factory", help="module:function() -> object with infer(args=...)")
    parser.add_argument("--dense-adapter", help="module:function(work_dir, export_dir, mode, circle, max_frames)")
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=3)
    parser.add_argument("--reuse-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--view-num", type=int, default=121, help="Generated video frame count")
    parser.add_argument("--recon-view-num", type=int, default=40, help="DA3 input view count (currently fixed sampling indices)")
    parser.add_argument("--crop-ratio", type=float, default=1.0)
    parser.add_argument("--use-view-extrapolation", action="store_true")
    parser.add_argument("--no-livephoto", action="store_true")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--circle", action="store_true")
    parser.add_argument("--max-frames", type=int, default=121)
    parser.add_argument("--external-code-root", type=Path, help="Root containing DA3 and Reconstruction_full.py")
    parser.add_argument("--cases-root", type=Path, help="Run all immediate child cases, reusing preloaded models")
    parser.add_argument("--da3-model-path", type=Path, help="Override default DA3 checkpoint")
    args = parser.parse_args()
    if args.case_dir is None and args.cases_root is None:
        parser.error("Provide case_dir or --cases-root")
    cfg = ReconstructionConfig(
        case_dir=args.case_dir or args.cases_root, view_num=args.view_num,
        recon_view_num=args.recon_view_num, crop_ratio=args.crop_ratio,
        use_view_extrapolation=args.use_view_extrapolation,
        is_livephoto=not args.no_livephoto, mode=args.mode,
        circle=args.circle, max_frames=args.max_frames,
        external_code_root=args.external_code_root, da3_model_path=args.da3_model_path)
    if args.dry_run:
        pipeline = ReconstructionPipeline(cfg)
        print(json.dumps({"paths": pipeline.paths(), "frame_indices": FRAME_INDICES,
                          "start_stage": args.start_stage, "end_stage": args.end_stage},
                         indent=2))
        return
    pipeline = ReconstructionPipeline(
        cfg,
        generate_video=load_callable(args.video_adapter) if args.video_adapter else None,
        run_DA3=load_callable(args.da3_factory)() if args.da3_factory else None,
        recon_iter_full=load_callable(args.dense_adapter) if args.dense_adapter else None,
    )
    cases = ([p for p in sorted(args.cases_root.iterdir())
              if p.is_dir() and (p / "image").is_dir()]
             if args.cases_root else [args.case_dir])
    if not cases:
        parser.error("No cases found")
    # External video adapter must initialize its model when loaded; a plain
    # function that reloads weights on every call cannot guarantee reuse.
    preload_times = pipeline.preload(start_stage=args.start_stage, end_stage=args.end_stage)
    print(f"[PRELOAD] total: {sum(preload_times.values()):.2f}s", flush=True)
    for case in cases:
        pipeline.set_case(case)
        pipeline.run(start_stage=args.start_stage, end_stage=args.end_stage,
                     reuse_video=args.reuse_video)


if __name__ == "__main__":
    main()
