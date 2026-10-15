#!/usr/bin/env python3
"""Refine DA3 camera extrinsics with a known 121-frame figure-8 trajectory prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def smootherstep(u: float) -> float:
    return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def figure8_angles(
    frames: int,
    start: float,
    elevation: float,
    theta: float = 30.0,
    phi: float = 5.0,
) -> np.ndarray:
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
    result = []
    for index in range(frames):
        time = 10.0 * index / (frames - 1)
        leg = min(9, int(time))
        eased = smootherstep(time - leg)
        first, second = waypoints[leg], waypoints[leg + 1]
        result.append(
            (
                start + first[0] + (second[0] - first[0]) * eased,
                elevation + first[1] + (second[1] - first[1]) * eased,
            )
        )
    return np.asarray(result, dtype=np.float64)


def canonical_orbit_cameras(angles_degrees: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create unit-sphere camera centers and OpenCV-style c2w rotations.

    Camera axes are x=right, y=down, z=forward. Every camera looks at the origin.
    A later global alignment absorbs the arbitrary canonical coordinate system.
    """
    angles = np.deg2rad(angles_degrees)
    azimuth = angles[:, 0]
    pitch = angles[:, 1]
    centers = np.stack(
        [
            np.cos(pitch) * np.sin(azimuth),
            np.sin(pitch),
            np.cos(pitch) * np.cos(azimuth),
        ],
        axis=1,
    )

    rotations = []
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    fallback_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for center in centers:
        forward = -center / np.linalg.norm(center)
        up = world_up
        if abs(np.dot(forward, up)) > 0.99:
            up = fallback_up
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)
        rotations.append(np.stack([right, down, forward], axis=1))
    return centers, np.stack(rotations)


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    return u @ correction @ vt


def extrinsics_to_c2w(
    extrinsics: np.ndarray,
    convention: str,
) -> Tuple[np.ndarray, np.ndarray]:
    rotations = np.asarray(extrinsics[:, :3, :3], dtype=np.float64)
    translations = np.asarray(extrinsics[:, :3, 3], dtype=np.float64)
    rotations = np.stack([project_to_rotation(rotation) for rotation in rotations])

    if convention == "w2c":
        c2w_rotations = np.transpose(rotations, (0, 2, 1))
        centers = -np.einsum("nij,nj->ni", c2w_rotations, translations)
    elif convention == "c2w":
        c2w_rotations = rotations
        centers = translations.copy()
    else:
        raise ValueError("convention must be 'w2c' or 'c2w'")
    return c2w_rotations, centers


def c2w_to_extrinsics(
    rotations: np.ndarray,
    centers: np.ndarray,
    convention: str,
    dtype: np.dtype,
) -> np.ndarray:
    output = np.zeros((len(rotations), 3, 4), dtype=np.float64)
    if convention == "c2w":
        output[:, :3, :3] = rotations
        output[:, :3, 3] = centers
    else:
        w2c_rotations = np.transpose(rotations, (0, 2, 1))
        output[:, :3, :3] = w2c_rotations
        output[:, :3, 3] = -np.einsum("nij,nj->ni", w2c_rotations, centers)
    return output.astype(dtype, copy=False)


def weighted_umeyama(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Fit target ~= scale * rotation @ source + translation."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    source_mean = np.sum(source * weights[:, None], axis=0)
    target_mean = np.sum(target * weights[:, None], axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = (target_centered * weights[:, None]).T @ source_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt

    source_variance = np.sum(weights * np.sum(source_centered**2, axis=1))
    if source_variance < 1e-12:
        raise ValueError("Degenerate trajectory: source camera centers have no variance.")
    scale = float(np.sum(singular_values * sign) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def robust_sim3_alignment(
    source: np.ndarray,
    target: np.ndarray,
    iterations: int = 5,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    weights = np.ones(len(source), dtype=np.float64)
    for _ in range(iterations):
        scale, rotation, translation = weighted_umeyama(source, target, weights)
        predicted = scale * (source @ rotation.T) + translation
        residuals = np.linalg.norm(predicted - target, axis=1)
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        robust_scale = max(1.4826 * mad, 1e-8)
        huber_delta = 1.5 * robust_scale
        weights = np.ones_like(residuals)
        outliers = residuals > huber_delta
        weights[outliers] = huber_delta / np.maximum(residuals[outliers], 1e-12)
    return scale, rotation, translation, weights


def blend_rotations(
    estimated: np.ndarray,
    prior: np.ndarray,
    strength: float,
    max_change_degrees: Optional[float] = None,
) -> np.ndarray:
    if strength <= 0.0:
        return estimated.copy()

    blended = []
    max_change_radians = (
        None if max_change_degrees is None else np.deg2rad(max_change_degrees)
    )
    for estimated_rotation, prior_rotation in zip(estimated, prior):
        relative = estimated_rotation.T @ prior_rotation
        rotation_vector, _ = cv2.Rodrigues(relative)
        step = rotation_vector * strength
        step_angle = float(np.linalg.norm(step))
        if (
            max_change_radians is not None
            and step_angle > max_change_radians
            and step_angle > 1e-12
        ):
            step *= max_change_radians / step_angle
        incremental, _ = cv2.Rodrigues(step)
        blended.append(project_to_rotation(estimated_rotation @ incremental))
    return np.stack(blended)


def infer_frame_indices(
    camera_count: int,
    original_frame_count: int,
    frame_indices: Optional[Sequence[int]],
) -> np.ndarray:
    if frame_indices is not None:
        indices = np.asarray(frame_indices, dtype=np.int64)
        if indices.shape != (camera_count,):
            raise ValueError(
                f"Expected {camera_count} frame indices, got shape {indices.shape}."
            )
    elif camera_count == 40 and original_frame_count == 121:
        # Matches: select='not(mod(n,floor(121/40)))', -vframes 40
        indices = np.arange(0, 120, 3, dtype=np.int64)
    elif camera_count == 61 and original_frame_count == 121:
        indices = np.arange(0, 121, 2, dtype=np.int64)
    elif camera_count == original_frame_count:
        indices = np.arange(original_frame_count, dtype=np.int64)
    else:
        raise ValueError(
            "Cannot safely infer the downsample mapping. Pass frame_indices explicitly. "
            "For 40 cameras sampled by the provided ffmpeg command, the default "
            "mapping is 0,3,...,117."
        )

    if np.any(indices < 0) or np.any(indices >= original_frame_count):
        raise ValueError("frame_indices contains an index outside the original video.")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("frame_indices must be strictly increasing.")
    return indices


def repeated_pose_error(
    centers: np.ndarray,
    rotations: np.ndarray,
    selected_angles: np.ndarray,
    tolerance_degrees: float = 1e-7,
) -> Dict[str, float]:
    center_errors = []
    rotation_errors = []
    for first in range(len(selected_angles)):
        for second in range(first + 1, len(selected_angles)):
            if np.max(np.abs(selected_angles[first] - selected_angles[second])) > tolerance_degrees:
                continue
            center_errors.append(float(np.linalg.norm(centers[first] - centers[second])))
            relative = rotations[first].T @ rotations[second]
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            rotation_errors.append(float(np.rad2deg(np.arccos(cosine))))

    return {
        "pair_count": len(center_errors),
        "mean_center_error": float(np.mean(center_errors)) if center_errors else 0.0,
        "max_center_error": float(np.max(center_errors)) if center_errors else 0.0,
        "mean_rotation_error_degrees": (
            float(np.mean(rotation_errors)) if rotation_errors else 0.0
        ),
        "max_rotation_error_degrees": (
            float(np.max(rotation_errors)) if rotation_errors else 0.0
        ),
    }


def refine_figure8_extrinsics(
    extrinsics: np.ndarray,
    *,
    convention: str = "w2c",
    original_frame_count: int = 121,
    frame_indices: Optional[Sequence[int]] = None,
    start: float = 0.0,
    elevation: float = 0.0,
    theta: float = 30.0,
    phi: float = 5.0,
    position_strength: float = 0.1,
    rotation_strength: float = 0.1,
    max_position_shift_ratio: Optional[float] = 0.5,
    max_rotation_change_degrees: Optional[float] = 2.0,
    robust_alignment_iterations: int = 5,
    return_report: bool = False,
):
    """Correct DA3 extrinsics using the known figure-8 trajectory.

    Args:
        extrinsics: Array shaped (N, 3, 4), in w2c or c2w convention.
        convention: Input/output convention, either "w2c" or "c2w".
        frame_indices: Original 121-frame indices represented by the N cameras.
            For N=40, defaults to [0, 3, ..., 117], matching the provided
            ffmpeg command. N=61 still defaults to [0, 2, ..., 120].
        position_strength: 0 keeps DA3 centers; 1 uses the aligned trajectory prior.
        rotation_strength: Fraction of the rotation correction to apply.
        max_position_shift_ratio: Cap each position update relative to the median
            distance between consecutive DA3 camera centers. None disables the cap.
        max_rotation_change_degrees: Cap each camera rotation update. None disables it.
        return_report: Return (corrected_extrinsics, diagnostics) when True.
    """
    input_array = np.asarray(extrinsics)
    if input_array.ndim != 3 or input_array.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsics shaped (N,3,4), got {input_array.shape}.")
    if not 0.0 <= position_strength <= 1.0:
        raise ValueError("position_strength must be in [0,1].")
    if not 0.0 <= rotation_strength <= 1.0:
        raise ValueError("rotation_strength must be in [0,1].")

    indices = infer_frame_indices(
        len(input_array),
        original_frame_count,
        frame_indices,
    )
    all_angles = figure8_angles(
        original_frame_count,
        start,
        elevation,
        theta,
        phi,
    )
    selected_angles = all_angles[indices]
    canonical_centers, canonical_rotations = canonical_orbit_cameras(selected_angles)

    estimated_rotations, estimated_centers = extrinsics_to_c2w(
        input_array,
        convention,
    )
    scale, world_rotation, translation, alignment_weights = robust_sim3_alignment(
        canonical_centers,
        estimated_centers,
        robust_alignment_iterations,
    )
    prior_centers = scale * (canonical_centers @ world_rotation.T) + translation

    # Fit one constant camera-local rotation to absorb OpenCV/OpenGL axis choices
    # and any constant roll offset in the estimated camera convention.
    world_aligned_prior_rotations = np.einsum(
        "ij,njk->nik",
        world_rotation,
        canonical_rotations,
    )
    local_alignment_sum = np.zeros((3, 3), dtype=np.float64)
    for weight, prior_rotation, estimated_rotation in zip(
        alignment_weights,
        world_aligned_prior_rotations,
        estimated_rotations,
    ):
        local_alignment_sum += weight * (prior_rotation.T @ estimated_rotation)
    local_alignment = project_to_rotation(local_alignment_sum)
    prior_rotations = np.einsum(
        "nij,jk->nik",
        world_aligned_prior_rotations,
        local_alignment,
    )

    position_updates = position_strength * (prior_centers - estimated_centers)
    consecutive_steps = np.linalg.norm(np.diff(estimated_centers, axis=0), axis=1)
    median_camera_step = float(np.median(consecutive_steps))
    if max_position_shift_ratio is not None:
        maximum_shift = max_position_shift_ratio * median_camera_step
        update_norms = np.linalg.norm(position_updates, axis=1)
        clipped = update_norms > maximum_shift
        position_updates[clipped] *= (
            maximum_shift / np.maximum(update_norms[clipped], 1e-12)
        )[:, None]
    corrected_centers = estimated_centers + position_updates
    corrected_rotations = blend_rotations(
        estimated_rotations,
        prior_rotations,
        rotation_strength,
        max_rotation_change_degrees,
    )
    corrected = c2w_to_extrinsics(
        corrected_rotations,
        corrected_centers,
        convention,
        input_array.dtype,
    )

    report = {
        "convention": convention,
        "frame_indices": indices.tolist(),
        "sim3_scale": scale,
        "sim3_rotation": world_rotation.tolist(),
        "sim3_translation": translation.tolist(),
        "position_strength": position_strength,
        "rotation_strength": rotation_strength,
        "median_camera_step": median_camera_step,
        "max_position_shift_ratio": max_position_shift_ratio,
        "max_rotation_change_degrees": max_rotation_change_degrees,
        "camera_changes": camera_change_metrics(
            estimated_centers,
            estimated_rotations,
            corrected_centers,
            corrected_rotations,
        ),
        "alignment_center_rmse": float(
            np.sqrt(np.mean(np.sum((prior_centers - estimated_centers) ** 2, axis=1)))
        ),
        "before_repeated_pose_error": repeated_pose_error(
            estimated_centers,
            estimated_rotations,
            selected_angles,
        ),
        "after_repeated_pose_error": repeated_pose_error(
            corrected_centers,
            corrected_rotations,
            selected_angles,
        ),
    }
    return (corrected, report) if return_report else corrected


def camera_change_metrics(
    original_centers: np.ndarray,
    original_rotations: np.ndarray,
    corrected_centers: np.ndarray,
    corrected_rotations: np.ndarray,
) -> Dict[str, float]:
    position_changes = np.linalg.norm(corrected_centers - original_centers, axis=1)
    rotation_changes = []
    for original, corrected in zip(original_rotations, corrected_rotations):
        relative = original.T @ corrected
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        rotation_changes.append(np.rad2deg(np.arccos(cosine)))
    return {
        "mean_position_change": float(np.mean(position_changes)),
        "max_position_change": float(np.max(position_changes)),
        "mean_rotation_change_degrees": float(np.mean(rotation_changes)),
        "max_rotation_change_degrees": float(np.max(rotation_changes)),
    }


def visualize_extrinsics_comparison(
    original_extrinsics: np.ndarray,
    corrected_extrinsics: np.ndarray,
    output_path: Path,
    *,
    convention: str = "w2c",
    points: Optional[np.ndarray] = None,
    forward_axis: str = "z",
    max_points: int = 20000,
) -> None:
    """Save before/after/overlay camera-center and orientation visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_rotations, original_centers = extrinsics_to_c2w(
        original_extrinsics, convention
    )
    corrected_rotations, corrected_centers = extrinsics_to_c2w(
        corrected_extrinsics, convention
    )
    axis_index = {"x": 0, "y": 1, "z": 2, "-x": 0, "-y": 1, "-z": 2}[forward_axis]
    axis_sign = -1.0 if forward_axis.startswith("-") else 1.0
    original_forward = axis_sign * original_rotations[:, :, axis_index]
    corrected_forward = axis_sign * corrected_rotations[:, :, axis_index]

    all_centers = np.concatenate([original_centers, corrected_centers], axis=0)
    extent = max(float(np.ptp(all_centers, axis=0).max()), 1e-6)
    arrow_length = 0.08 * extent

    plot_points = None
    if points is not None:
        plot_points = np.asarray(points)
        if plot_points.ndim != 2 or plot_points.shape[1] < 3:
            raise ValueError("points must have shape (M,3) or (M,>=3).")
        plot_points = plot_points[:, :3]
        if len(plot_points) > max_points:
            selection = np.linspace(0, len(plot_points) - 1, max_points).astype(int)
            plot_points = plot_points[selection]

    figure = plt.figure(figsize=(18, 6))
    axes = [
        figure.add_subplot(131, projection="3d"),
        figure.add_subplot(132, projection="3d"),
        figure.add_subplot(133, projection="3d"),
    ]

    def draw_scene(axis, centers, directions, color, title):
        if plot_points is not None:
            axis.scatter(
                plot_points[:, 0], plot_points[:, 1], plot_points[:, 2],
                s=0.3, c="0.7", alpha=0.2,
            )
        axis.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=color)
        axis.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=12, c=color)
        axis.quiver(
            centers[:, 0], centers[:, 1], centers[:, 2],
            directions[:, 0], directions[:, 1], directions[:, 2],
            length=arrow_length, normalize=True, color=color, linewidth=0.7,
        )
        for index, center in enumerate(centers):
            if index % 4 == 0 or index in (0, len(centers) - 1):
                axis.text(*center, str(index), fontsize=7)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")

    draw_scene(axes[0], original_centers, original_forward, "tab:blue", "Before")
    draw_scene(axes[1], corrected_centers, corrected_forward, "tab:red", "After")

    if plot_points is not None:
        axes[2].scatter(
            plot_points[:, 0], plot_points[:, 1], plot_points[:, 2],
            s=0.3, c="0.7", alpha=0.2,
        )
    axes[2].plot(
        original_centers[:, 0], original_centers[:, 1], original_centers[:, 2],
        color="tab:blue", label="before",
    )
    axes[2].plot(
        corrected_centers[:, 0], corrected_centers[:, 1], corrected_centers[:, 2],
        color="tab:red", label="after",
    )
    for before, after in zip(original_centers, corrected_centers):
        axes[2].plot(
            [before[0], after[0]], [before[1], after[1]], [before[2], after[2]],
            color="0.5", linewidth=0.5, alpha=0.6,
        )
    axes[2].quiver(
        original_centers[:, 0], original_centers[:, 1], original_centers[:, 2],
        original_forward[:, 0], original_forward[:, 1], original_forward[:, 2],
        length=arrow_length, normalize=True, color="tab:blue", linewidth=0.5,
    )
    axes[2].quiver(
        corrected_centers[:, 0], corrected_centers[:, 1], corrected_centers[:, 2],
        corrected_forward[:, 0], corrected_forward[:, 1], corrected_forward[:, 2],
        length=arrow_length, normalize=True, color="tab:red", linewidth=0.5,
    )
    axes[2].set_title("Overlay (gray lines are corrections)")
    axes[2].legend()

    combined = all_centers if plot_points is None else np.concatenate(
        [all_centers, plot_points], axis=0
    )
    center = (combined.min(axis=0) + combined.max(axis=0)) / 2.0
    radius = max(float(np.ptp(combined, axis=0).max()) / 2.0, 1e-6)
    for axis in axes:
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_frame_indices(value: Optional[str]) -> Optional[Sequence[int]]:
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        text = value
    return [int(token) for token in text.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input .npy shaped (N,3,4).")
    parser.add_argument("--output", type=Path, required=True, help="Output corrected .npy.")
    parser.add_argument("--convention", choices=["w2c", "c2w"], default="w2c")
    parser.add_argument("--frame-indices", help="Comma list or text file of source frame indices.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--theta", type=float, default=30.0)
    parser.add_argument("--phi", type=float, default=5.0)
    parser.add_argument("--position-strength", type=float, default=0.1)
    parser.add_argument("--rotation-strength", type=float, default=0.1)
    parser.add_argument("--max-position-shift-ratio", type=float, default=0.5)
    parser.add_argument("--max-rotation-change-degrees", type=float, default=2.0)
    parser.add_argument("--points", type=Path, help="Optional .npy point cloud for visualization.")
    parser.add_argument("--forward-axis", choices=["x", "y", "z", "-x", "-y", "-z"], default="z")
    args = parser.parse_args()

    extrinsics = np.load(args.input)
    corrected, report = refine_figure8_extrinsics(
        extrinsics,
        convention=args.convention,
        frame_indices=parse_frame_indices(args.frame_indices),
        start=args.start,
        elevation=args.elevation,
        theta=args.theta,
        phi=args.phi,
        position_strength=args.position_strength,
        rotation_strength=args.rotation_strength,
        max_position_shift_ratio=args.max_position_shift_ratio,
        max_rotation_change_degrees=args.max_rotation_change_degrees,
        return_report=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, corrected)
    visualization_path = args.output.with_name(
        args.output.stem + "_comparison.png"
    )
    points = np.load(args.points) if args.points is not None else None
    visualize_extrinsics_comparison(
        extrinsics,
        corrected,
        visualization_path,
        convention=args.convention,
        points=points,
        forward_axis=args.forward_axis,
    )

    report_path = args.output.with_suffix(".json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Input shape: {extrinsics.shape}")
    print(f"Output shape: {corrected.shape}")
    print(f"Corrected extrinsics: {args.output}")
    print(f"Diagnostics: {report_path}")
    print(f"Visualization: {visualization_path}")
    print("Camera changes:", report["camera_changes"])
    print("Repeated-pose error before:", report["before_repeated_pose_error"])
    print("Repeated-pose error after:", report["after_repeated_pose_error"])


if __name__ == "__main__":
    main()
