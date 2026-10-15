#!/usr/bin/env python3
"""Visualize and measure repeated-pose consistency in a generated figure-8 video."""

from __future__ import annotations

import argparse
import json
import math
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
    # A group can contain three repeated frames (0/60/120). Other groups are pairs.
    max_comparisons = max(len(group) - 1 for group in groups)
    column_count = 1 + max_comparisons * 2
    rows: List[np.ndarray] = []
    report_groups: List[Dict[str, object]] = []

    for group in groups:
        reference_index = group[0]
        reference = frames[reference_index]
        pose = poses[reference_index]
        tiles: List[np.ndarray] = [
            add_header(
                fit_thumbnail(reference, thumbnail_width, thumbnail_height),
                f"reference frame {reference_index}",
                f"az={pose[0]:.6f}, pitch={pose[1]:.6f}",
            )
        ]
        comparisons: List[Dict[str, object]] = []

        for comparison_index in group[1:]:
            comparison = frames[comparison_index]
            metrics = image_metrics(reference, comparison)
            metric_text = (
                f"SSIM={metrics['ssim']:.4f}  "
                f"PSNR={metrics['psnr']:.2f}  MAE={metrics['mae']:.4f}"
            )
            tiles.append(
                add_header(
                    fit_thumbnail(comparison, thumbnail_width, thumbnail_height),
                    f"comparison frame {comparison_index}",
                    metric_text,
                )
            )
            heatmap = difference_heatmap(reference, comparison, amplification)
            tiles.append(
                add_header(
                    fit_thumbnail(heatmap, thumbnail_width, thumbnail_height),
                    f"|frame {reference_index} - frame {comparison_index}|",
                    f"diff amplified x{amplification:g}",
                )
            )
            comparisons.append(
                {
                    "reference_frame": reference_index,
                    "comparison_frame": comparison_index,
                    **metrics,
                }
            )

        blank = np.full_like(tiles[0], 235)
        while len(tiles) < column_count:
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
    parser.add_argument("--video", type=Path, required=True, help="Generated 121-frame video.")
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
    frames, fps = read_video_frames(args.video, args.frames)

    sheet, group_report = build_contact_sheet(
        frames,
        poses,
        repeated_groups,
        args.thumbnail_width,
        args.thumbnail_height,
        args.diff_amplification,
    )
    sheet_path = args.output_dir / "figure8_repeated_pose_comparison.png"
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

    report = {
        "video": str(args.video.resolve()),
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
    }
    report_path = args.output_dir / "figure8_consistency_metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Repeated pose groups ({len(repeated_groups)}):")
    print(" ".join("/".join(map(str, group)) for group in repeated_groups))
    print(f"Mean MAE: {mean_mae:.6f} (lower is better)")
    print(f"Mean PSNR: {mean_psnr:.3f} dB (higher is better)")
    print(f"Mean SSIM: {mean_ssim:.6f} (higher is better)")
    print(f"Consistency score: {consistency_score:.2f}/100")
    print(f"Contact sheet: {sheet_path}")
    print(f"Metrics JSON: {report_path}")


if __name__ == "__main__":
    main()
