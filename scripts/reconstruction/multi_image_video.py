"""Generate videos for DA3-planned multi-image paths and preserve frame-to-pose IDs.

The generator is injectable: callable(control_video, start_image, end_image,
output_video, prompt, num_frames, fps) -> optional output path.
It MUST actually condition on the rendered control video (do not use TI2V).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _read_video_frames(path, expected):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    try:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) != expected:
        raise ValueError(f"{path}: expected {expected} frames, decoded {len(frames)}")
    return frames


def extract_generated_frames(video_path, poses_path, dest, *, indices=None):
    """Extract exact frames and matched DA3 w2c poses, no pose re-estimation."""
    import cv2
    poses = np.load(poses_path)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected (T,4,4) poses in {poses_path}, got {poses.shape}")
    frames = _read_video_frames(video_path, len(poses))
    selected = range(len(frames)) if indices is None else indices
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for i in selected:
        if not 0 <= i < len(frames):
            raise IndexError(f"Frame index {i} outside video with {len(frames)} frames")
        name = f"{i:06d}.jpg"
        if not cv2.imwrite(str(dest / name), frames[i]):
            raise IOError(f"Unable to write {dest / name}")
        saved.append({"frame": int(i), "image": name, "pose_index": int(i)})
    np.save(dest / "w2c.npy", poses[[v["frame"] for v in saved]])
    (dest / "frame_poses.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return saved


def run_trajectory_videos(*, reconstruction_dir, source_images, generator,
                          fps=30, prompt="", resume=False,
                          sample_plan=None, save_mp4=True):
    """Generate MP4s and preserve references to already planned per-frame poses.

    No image extraction in this stage; retain the original NPY to align frames
    if/when a later reconstruction stage requests sampling.
    """
    root = Path(reconstruction_dir)
    manifest_path = root / "trajectories" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing planned trajectories: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest["trajectories"]
    if not paths:
        raise ValueError(f"No planned trajectories in {manifest_path}")
    sources = list(source_images)
    generated_root = root / "generated_trajectories"
    generated_root.mkdir(parents=True, exist_ok=True)
    indices = (sample_plan or {}).get("selected_indices")
    if indices is None:
        raise ValueError("Wan generation requires pose-based FPS sample plan")
    summary = []
    phase_started = time.perf_counter()
    for item in paths:
        stem = Path(item["file"]).stem
        poses_path = root / "trajectories" / item["file"]
        poses = np.load(poses_path)
        if len(poses) != item["frames"]:
            raise ValueError(f"Pose length mismatch for {stem}")
        control_video = root / "trajectories" / "gs_video" / f"{stem}.mp4"
        if not control_video.is_file():
            raise FileNotFoundError(f"Missing per-trajectory GS control video: {control_video}")
        start_index, end_index = int(item["start_index"]), int(item["end_index"])
        if not (0 <= start_index < len(sources) and 0 <= end_index < len(sources)):
            raise IndexError(f"Input image indices invalid for {stem}")
        target = generated_root / stem
        target.mkdir(parents=True, exist_ok=True)
        output_video = target / "generated.mp4"
        selected = list(map(int, indices[stem]))
        sample_dir = target / "sampled_frames"
        segment_started = time.perf_counter()
        frames_ready = all((sample_dir / f"f{i:04d}.jpg").is_file() for i in selected)
        reused = bool(resume and frames_ready and
                      (not save_mp4 or (output_video.is_file() and output_video.stat().st_size)))
        if not reused:
            if generator is None:
                raise RuntimeError("Missing a control-video-capable Wan generator adapter")
            sample_dir.mkdir(parents=True, exist_ok=True)
            for old in sample_dir.glob("f*.jpg"):
                old.unlink()
            if not save_mp4 and output_video.is_file():
                output_video.unlink()
            produced = generator(
                control_video=str(control_video),
                start_image=str(sources[start_index]),
                end_image=str(sources[end_index]),
                output_video=str(output_video),
                prompt=str(prompt),
                num_frames=int(len(poses)),
                fps=int(fps), sampled_frame_indices=selected,
                sampled_frames_dir=str(sample_dir), save_mp4=save_mp4,
            )
            if save_mp4 and produced is not None and Path(produced).resolve() != output_video.resolve():
                raise RuntimeError(f"Adapter returned {produced}; expected {output_video}")
        if save_mp4 and (not output_video.is_file() or not output_video.stat().st_size):
            raise RuntimeError(f"No generated MP4 from adapter: {output_video}")
        if not all((sample_dir / f"f{i:04d}.jpg").is_file() for i in selected):
            raise RuntimeError(f"Direct JPG output incomplete: {sample_dir}")
        segment_seconds = round(time.perf_counter() - segment_started, 3)
        infer_timing = {} if reused else dict(getattr(generator, "last_timing", {}))
        summary.append({
            "name": stem, "video": str(output_video), "frames": len(poses),
            "reused": reused, "seconds": segment_seconds, "inference_timing": infer_timing,
            "sampled_frame_indices": selected, "sampled_frames_dir": str(sample_dir),
            "save_mp4": bool(save_mp4),
            "pose_file": str(poses_path),
            "start_image": str(sources[start_index]), "end_image": str(sources[end_index]),
            "control_video": str(control_video),
        })
        (generated_root / "manifest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[MULTI VIDEO] {stem}: {len(poses)} frames, "
              f"elapsed={segment_seconds:.3f}s, reused={reused}", flush=True)
    total = round(time.perf_counter() - phase_started, 3)
    print(f"[MULTI VIDEO SUMMARY] segments={len(summary)}, total={total:.3f}s, "
          f"generated={sum(not x['reused'] for x in summary)}", flush=True)
    return summary
