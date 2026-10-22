#!/usr/bin/env python3
"""Select reconstruction frames approximately uniformly in camera viewing angle.

Writes image/%05d.png (1-based names) and frame_indices.txt (0-based source indices).
The latter MUST be passed to refine_figure8_extrinsics(frame_indices=...).
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from refine_figure8_extrinsics import figure8_angles, canonical_orbit_cameras


def select_indices(total, count, start, elevation, theta, phi):
    if count > total:
        raise ValueError(f"Requested {count} frames from {total}")
    angles = figure8_angles(total, start, elevation, theta, phi)
    centers, _ = canonical_orbit_cameras(angles)
    dots = np.clip(np.sum(centers[1:] * centers[:-1], axis=1), -1.0, 1.0)
    distances = np.arccos(dots)
    cumulative = np.r_[0.0, np.cumsum(distances)]
    if cumulative[-1] <= 1e-10:
        return np.rint(np.linspace(0, total - 1, count)).astype(int)
    targets = np.linspace(0, cumulative[-1], count)
    # Monotonic, unique indices, including both endpoints.
    result = np.empty(count, dtype=int)
    result[0], result[-1] = 0, total - 1
    for k in range(1, count - 1):
        guess = int(np.searchsorted(cumulative, targets[k]))
        options = [max(result[k-1]+1, guess-1), max(result[k-1]+1, guess)]
        maximum = total - (count-k)
        options = [min(maximum, x) for x in options]
        result[k] = min(options, key=lambda j: abs(cumulative[j]-targets[k]))
    if not np.all(np.diff(result) > 0):
        raise RuntimeError("Could not select unique frames")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=Path, required=True)
    p.add_argument("--dst-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--elevation", type=float, default=0)
    p.add_argument("--theta", type=float, default=30)
    p.add_argument("--phi", type=float, default=5)
    args = p.parse_args()
    videos = sorted(args.src_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 files in {args.src_dir}")
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if len(frames) < args.count:
            raise ValueError(f"{video}: only {len(frames)} frames")
        indices = select_indices(len(frames), args.count, args.start, args.elevation, args.theta, args.phi)
        case = args.dst_dir / video.stem
        folder = case / "image"
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.png"):
            old.unlink()
        for output_idx, source_idx in enumerate(indices, 1):
            if not cv2.imwrite(str(folder / f"{output_idx:05d}.png"), frames[source_idx]):
                raise RuntimeError("Image write failed")
        np.savetxt(case / "frame_indices.txt", indices, fmt="%d")
        print(f"[done] {video.name}: {len(frames)} -> {len(indices)}; indices={indices.tolist()}", flush=True)


if __name__ == "__main__":
    main()
