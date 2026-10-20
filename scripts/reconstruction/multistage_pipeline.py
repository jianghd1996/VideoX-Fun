#!/usr/bin/env python3
"""Three-stage single-image generative reconstruction orchestration.

External model integrations are intentionally injected, not imported here.
Run with --dry-run to inspect paths before wiring model implementations.
"""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

FRAME_INDICES = (0, 6, 14, 16, 17, 18, 19, 20, 21, 23, 28, 30, 34, 39,
                 40, 41, 42, 43, 44, 45, 54, 75, 76, 77, 78, 79, 80, 81,
                 82, 88, 91, 95, 99, 100, 101, 102, 103, 104, 105, 106)
VIDEO_FRAMES = 121


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
    view_num: int = 40
    crop_ratio: float = 1.0
    use_view_extrapolation: bool = False
    is_livephoto: bool = True
    mode: str = "default"
    circle: bool = False
    max_frames: int = 121


def load_callable(spec: str) -> Callable[..., Any]:
    """Import a user-supplied adapter, e.g. my_adapters:generate_video."""
    module, sep, name = spec.partition(":")
    if not sep or not module or not name:
        raise ValueError("Adapter must use module:function notation")
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError(f"Adapter is not callable: {spec}")
    return fn


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
        self.colmap_path = self.gen_vid_path / "sparse"
        self.gs_render_vid_path = self.gen_vid_path / "gs_render.mp4"
        self.export_dir = self.gen_vid_path / "dense"
        self.max_frames = config.max_frames
        self.generate_video = generate_video
        self.run_DA3 = run_DA3
        self.recon_iter_full = recon_iter_full

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

    def stage_video(self, *, reuse: bool = False) -> None:
        self.gen_vid_path.mkdir(parents=True, exist_ok=True)
        if not (reuse and self.video_path.is_file()):
            if self.generate_video is None:
                raise RuntimeError("Video generator missing; inject generate_video(image_path, output_path, num_frames)")
            self.generate_video(image_path=self.input_image(),
                                output_path=self.video_path, num_frames=VIDEO_FRAMES)
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
            if count != VIDEO_FRAMES:
                raise ValueError(f"Expected {VIDEO_FRAMES} frames, got {count}")
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
            view_num=self.args.view_num,
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
        if start_stage <= 1 <= end_stage:
            self.stage_video(reuse=reuse_video)
        if start_stage <= 2 <= end_stage:
            results["gen_index"] = self.stage_da3()
        if start_stage <= 3 <= end_stage:
            results["dense_result"] = self.stage_dense()
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--video-adapter", help="module:function(image_path, output_path, num_frames)")
    parser.add_argument("--da3-factory", help="module:function() -> object with infer(args=...)")
    parser.add_argument("--dense-adapter", help="module:function(work_dir, export_dir, mode, circle, max_frames)")
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
    cfg = ReconstructionConfig(
        case_dir=args.case_dir, view_num=args.view_num, crop_ratio=args.crop_ratio,
        use_view_extrapolation=args.use_view_extrapolation,
        is_livephoto=not args.no_livephoto, mode=args.mode,
        circle=args.circle, max_frames=args.max_frames)
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
    print(pipeline.run(start_stage=args.start_stage, end_stage=args.end_stage,
                       reuse_video=args.reuse_video))


if __name__ == "__main__":
    main()
