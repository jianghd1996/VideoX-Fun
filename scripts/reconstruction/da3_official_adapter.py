"""Adapter between VideoX-Fun reconstruction and persistent Depth Anything 3.

Select with da3_backend=official. The legacy DA3 backend remains default.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


class OfficialDA3Adapter:
    def __init__(self, *, repo_root: str, model_path: str, process_res: int = 504,
                 process_res_method: str = "upper_bound_resize",
                 preprocess_workers: int = 8, profile: bool = True):
        root = Path(repo_root).expanduser().resolve()
        if not (root / "src" / "depth_anything_3").is_dir():
            raise FileNotFoundError(f"Depth Anything 3 repository not found: {root}")
        if str(root / "src") not in sys.path:
            sys.path.insert(0, str(root / "src"))
        from depth_anything_3.services.persistent_inference import PersistentDA3
        self.engine = PersistentDA3(
            model_path, process_res=process_res,
            process_res_method=process_res_method,
            preprocess_workers=preprocess_workers, profile=profile,
        )

    def infer(self, *, args):
        image_dir = Path(args.data_path)
        images = sorted(str(path) for path in image_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(images) != args.view_num:
            raise ValueError(f"Expected {args.view_num} images, got {len(images)}")
        if args.crop_ratio != 1.0:
            raise ValueError("Official DA3 adapter currently requires crop_ratio=1.0")
        if any(args.extra_view):
            raise ValueError("Official DA3 adapter does not implement legacy extra_view geometry")
        colmap_dir = Path(args.colmap_path)
        colmap_dir.mkdir(parents=True, exist_ok=True)
        self.engine.infer(
            images,
            export_dir=str(colmap_dir),
            export_format="colmap",
            timing_path=str(colmap_dir.parent.parent / "da3_timings.json"),
        )
        # Legacy code interprets this return as an optional generated frame index.
        return None


def create_official_da3(*, repo_root: str, model_path: str, process_res: int = 504,
                        process_res_method: str = "upper_bound_resize",
                        preprocess_workers: int = 8, profile: bool = True):
    return OfficialDA3Adapter(
        repo_root=repo_root, model_path=model_path, process_res=process_res,
        process_res_method=process_res_method,
        preprocess_workers=preprocess_workers, profile=profile,
    )
