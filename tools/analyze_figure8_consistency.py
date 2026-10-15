#!/usr/bin/env python3
"""Visualize and measure repeated-pose consistency in a generated figure-8 video."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


Pose = Tuple[float, float]


def smootherstep(u: float) -> float:
    """Quintic smoothstep with zero velocity and acceleration at both ends."""
    return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def figure8_angles(
    frames: int,
    start: float,
    elevation: float,
    theta: float = 30.0,
    phi: float = 5.0,
) -> List[Pose]:
    if frames < 2:
        raise ValueError("frames must be at least 2")

    waypoints = [
        (0.0, 0.0),
        (0.0, phi),
        (theta, phi),
        (theta, -phi),
        (0.0, -phi),
        (0.0, 0.0),
        (0.0, phi),
        (-theta, phi),
        (-theta, -phi),
        (0.0, -phi),
        (0.0, 0.0),
    ]

    result: List[Pose] = []
    for index in range(frames):
        time = 10.0 * index / (frames - 1)
        leg = min(9, int(time))
        u = time - leg
        eased = smootherstep(u)

        a, b = waypoints[leg], waypoints[leg + 1]
        azimuth = start + a[0] + (b[0] - a[0]) * eased
        pitch = elevation + a[1] + (b[1] - a[1]) * eased
        result.append((azimuth, pitch))
    return result


def angular_distance_degrees(a: Pose, b: Pose) -> float:
    azimuth_delta = abs((a[0] - b[0] + 180.0) % 360.0 - 180.0)
    pitch_delta = abs(a[1] - b[1])
    return max(azimuth_delta, pitch_delta)


def find_repeated_pose_groups(
    poses: Sequence[Pose],
    tolerance_degrees: float = 1e-6,
) -> List[List[int]]:
    """Cluster equal poses and return only groups containing multiple frames."""
    groups: List[List[int]] = []
    representatives: List[Pose] = []

    for frame_index, pose in enumerate(poses):
        matched_group = None
        for group_index, representative in enumerate(representatives):
            if angular_distance_degrees(pose, representative) <= tolerance_degrees:
                matched_group = group_index
                break
        if matched_group is None:
            representatives.append(pose)
            groups.append([frame_index])
        else:
            groups[matched_group].append(frame_index)

    repeated = [group for group in groups if len(group) > 1]
    repeated.sort(key=lambda group: group[0])
    return repeated


def read_video_frames(video_path: Path, expected_frames: int) -> Tuple[List[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    if len(frames) != expected_frames:
        raise ValueError(
            f"Expected exactly {expected_frames} decoded frames, got {len(frames)} "
            f"from {video_path}"
        )
    return frames, fps


def image_metrics(reference: np.ndarray, comparison: np.ndarray) -> Dict[str, float]:
    ref = reference.astype(np.float32) / 255.0
    cmp = comparison.astype(np.float32) / 255.0
    error = ref - cmp

    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error * error))
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))

    # Standard local SSIM, averaged over channels and pixels.
    kernel_size = 11
    sigma = 1.5
    c1 = 0.01**2
    c2 = 0.03**2
    mu_ref = cv2.GaussianBlur(ref, (kernel_size, kernel_size), sigma)
    mu_cmp = cv2.GaussianBlur(cmp, (kernel_size, kernel_size), sigma)
    sigma_ref = cv2.GaussianBlur(ref * ref, (kernel_size, kernel_size), sigma) - mu_ref**2
    sigma_cmp = cv2.GaussianBlur(cmp * cmp, (kernel_size, kernel_size), sigma) - mu_cmp**2
    sigma_cross = cv2.GaussianBlur(ref * cmp, (kernel_size, kernel_size), sigma) - mu_ref * mu_cmp
    ssim_map = (
        (2.0 * mu_ref * mu_cmp + c1) * (2.0 * sigma_cross + c2)
        / ((mu_ref**2 + mu_cmp**2 + c1) * (sigma_ref + sigma_cmp + c2))
    )
    ssim = float(np.mean(ssim_map))
    return {"mae": mae, "psnr": psnr, "ssim": ssim}


def fit_thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def add_header(tile: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    header_height = 52
    output = np.full(
        (tile.shape[0] + header_height, tile.shape[1], 3),
        245,
        dtype=np.uint8,
    )
    output[header_height:] = tile
    cv2.putText(
        output,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            output,
            subtitle,
            (8, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
    return output


def difference_heatmap(
    reference: np.ndarray,
    comparison: np.ndarray,
    amplification: float,
) -> np.ndarray:
    difference = cv2.absdiff(reference, comparison)
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    amplified = np.clip(
        difference_gray.astype(np.float32) * amplification,
        0,
        255,
    ).astype(np.uint8)
    return cv2.applyColorMap(amplified, cv2.COLORMAP_TURBO)


def build_contact_sheet(
    frames: Sequence[np.ndarray],
    poses: Sequence[Pose],
    groups: Sequence[Sequence[int]],
    thumbnail_width: int,
    thumbnail_height: int,
    amplification: float,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    max_columns = max(
        len(group) + math.comb(len(group), 2)
        for group in groups
    )
    rows: List[np.ndarray] = []
    report_groups: List[Dict[str, object]] = []

    for group in groups:
        pose = poses[group[0]]
        tiles: List[np.ndarray] = []
        comparisons: List[Dict[str, object]] = []

        for frame_index in group:
            tiles.append(
                add_header(
                    fit_thumbnail(
                        frames[frame_index],
                        thumbnail_width,
                        thumbnail_height,
                    ),
                    f"frame {frame_index}",
                    f"az={pose[0]:.6f}, pitch={pose[1]:.6f}",
                )
            )

        for first_index, second_index in combinations(group, 2):
            first = frames[first_index]
            second = frames[second_index]
            metrics = image_metrics(first, second)
            metric_text = (
                f"SSIM={metrics['ssim']:.4f}  "
                f"PSNR={metrics['psnr']:.2f}  MAE={metrics['mae']:.4f}"
            )
            heatmap = difference_heatmap(first, second, amplification)
            tiles.append(
                add_header(
                    fit_thumbnail(heatmap, thumbnail_width, thumbnail_height),
                    f"|frame {first_index} - frame {second_index}|",
                    metric_text,
                )
            )
            comparisons.append(
                {
                    "first_frame": first_index,
                    "second_frame": second_index,
                    **metrics,
                }
            )

        blank = np.full_like(tiles[0], 235)
        while len(tiles) < max_columns:
            tiles.append(blank.copy())

        row = cv2.hconcat(tiles)
        border = np.full((8, row.shape[1], 3), 80, dtype=np.uint8)
        rows.extend([row, border])
        report_groups.append(
            {
                "frames": list(group),
                "azimuth": pose[0],
                "pitch": pose[1],
                "comparisons": comparisons,
            }
        )

    return cv2.vconcat(rows[:-1]), report_groups

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="One generated 121-frame video.")
    source.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing MP4 videos. Subdirectories are searched recursively.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figure8_consistency"))
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--theta", type=float, default=30.0)
    parser.add_argument("--phi", type=float, default=5.0)
    parser.add_argument("--pose-tolerance", type=float, default=1e-6)
    parser.add_argument("--thumbnail-width", type=int, default=320)
    parser.add_argument("--thumbnail-height", type=int, default=240)
    parser.add_argument("--diff-amplification", type=float, default=4.0)
    return parser.parse_args()


def analyze_video(
    video_path: Path,
    result_dir: Path,
    args: argparse.Namespace,
    poses: Sequence[Pose],
    repeated_groups: Sequence[Sequence[int]],
) -> Dict[str, object]:
    result_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = read_video_frames(video_path, args.frames)

    sheet, group_report = build_contact_sheet(
        frames,
        poses,
        repeated_groups,
        args.thumbnail_width,
        args.thumbnail_height,
        args.diff_amplification,
    )
    sheet_path = result_dir / "figure8_repeated_pose_comparison.png"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise RuntimeError(f"Failed to save contact sheet: {sheet_path}")

    all_comparisons = [
        comparison
        for group in group_report
        for comparison in group["comparisons"]
    ]
    mean_mae = float(np.mean([item["mae"] for item in all_comparisons]))
    finite_psnr = [
        item["psnr"] for item in all_comparisons if math.isfinite(item["psnr"])
    ]
    mean_psnr = float(np.mean(finite_psnr)) if finite_psnr else float("inf")
    mean_ssim = float(np.mean([item["ssim"] for item in all_comparisons]))
    consistency_score = float(np.clip(mean_ssim, 0.0, 1.0) * 100.0)

    report: Dict[str, object] = {
        "video": str(video_path.resolve()),
        "decoded_frames": len(frames),
        "fps": fps,
        "trajectory": {
            "frames": args.frames,
            "start": args.start,
            "elevation": args.elevation,
            "theta": args.theta,
            "phi": args.phi,
            "pose_tolerance_degrees": args.pose_tolerance,
        },
        "repeated_pose_group_count": len(repeated_groups),
        "comparison_count": len(all_comparisons),
        "repeated_pose_groups": group_report,
        "summary": {
            "mean_mae": mean_mae,
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
            "consistency_score_0_to_100": consistency_score,
        },
        "contact_sheet": str(sheet_path.resolve()),
    }
    report_path = result_dir / "figure8_consistency_metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(
        f"[done] {video_path}: score={consistency_score:.2f}, "
        f"SSIM={mean_ssim:.4f}, PSNR={mean_psnr:.2f}, MAE={mean_mae:.4f}",
        flush=True,
    )
    return report


def collect_videos(args: argparse.Namespace) -> List[Path]:
    if args.video is not None:
        if not args.video.is_file():
            raise FileNotFoundError(f"Video does not exist: {args.video}")
        return [args.video]

    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {args.input_dir}")

    videos = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not videos:
        raise FileNotFoundError(f"No MP4 videos found under {args.input_dir}")
    return videos


def write_batch_summary(
    reports: Sequence[Dict[str, object]],
    output_dir: Path,
) -> None:
    rows = []
    for report in reports:
        summary = report["summary"]
        rows.append(
            {
                "video": report["video"],
                "consistency_score": summary["consistency_score_0_to_100"],
                "mean_ssim": summary["mean_ssim"],
                "mean_psnr": summary["mean_psnr"],
                "mean_mae": summary["mean_mae"],
                "contact_sheet": report["contact_sheet"],
            }
        )
    rows.sort(key=lambda row: row["consistency_score"])

    summary_json = output_dir / "batch_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "video_count": len(rows),
                "mean_consistency_score": float(
                    np.mean([row["consistency_score"] for row in rows])
                ),
                "videos_sorted_worst_first": rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    summary_csv = output_dir / "batch_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Batch JSON: {summary_json}")
    print(f"Batch CSV: {summary_csv}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    poses = figure8_angles(
        args.frames,
        args.start,
        args.elevation,
        args.theta,
        args.phi,
    )
    repeated_groups = find_repeated_pose_groups(poses, args.pose_tolerance)
    videos = collect_videos(args)

    print(f"Repeated pose groups ({len(repeated_groups)}):")
    print(" ".join("/".join(map(str, group)) for group in repeated_groups))
    print(f"Found {len(videos)} video(s).", flush=True)

    reports = []
    failures = []
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] Analyzing {video_path}", flush=True)
        if args.input_dir is None:
            result_dir = args.output_dir
        else:
            relative_path = video_path.relative_to(args.input_dir)
            result_dir = args.output_dir / relative_path.parent / relative_path.stem
        try:
            reports.append(
                analyze_video(
                    video_path,
                    result_dir,
                    args,
                    poses,
                    repeated_groups,
                )
            )
        except Exception as error:
            failures.append({"video": str(video_path), "error": str(error)})
            print(f"[failed] {video_path}: {error}", flush=True)

    if reports:
        write_batch_summary(reports, args.output_dir)
    if failures:
        failure_path = args.output_dir / "batch_failures.json"
        with failure_path.open("w", encoding="utf-8") as handle:
            json.dump(failures, handle, ensure_ascii=False, indent=2)
        print(f"Failures: {len(failures)}; details: {failure_path}")

    if not reports:
        raise RuntimeError("All videos failed analysis.")


if __name__ == "__main__":
    main()
