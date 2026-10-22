"""Input-anchored spherical camera trajectories for DA3 Gaussian rendering.

All extrinsics are world-to-camera, OpenCV convention. Geometry is kept in
DA3's native prediction coordinates; no camera or Gaussian is rescaled here.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _unit(x, eps=1e-9):
    x = np.asarray(x, dtype=np.float64)
    return x / max(float(np.linalg.norm(x)), eps)


def _w2c44(extrinsics):
    a = np.asarray(extrinsics, dtype=np.float64)
    if a.shape[-2:] == (3, 4):
        out = np.broadcast_to(np.eye(4), (*a.shape[:-2], 4, 4)).copy()
        out[..., :3, :] = a
        return out
    if a.shape[-2:] != (4, 4):
        raise ValueError(f"Expected Nx3x4 or Nx4x4 w2c poses; got {a.shape}")
    return a.copy()


def _look_at(center, target, up):
    z = _unit(target - center)  # OpenCV camera forward
    x = _unit(np.cross(z, up))
    if np.linalg.norm(np.cross(z, up)) < 1e-6:
        raise ValueError("Look-at direction parallel to up")
    y = _unit(np.cross(z, x))  # OpenCV down
    R = np.stack((x, y, z), axis=0)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = -R @ center
    return pose


def _centers(poses):
    return -np.einsum("nji,nj->ni", poses[:, :3, :3], poses[:, :3, 3])


def _estimate_center(centers, directions):
    # Solve sum ||(I-dd^T)(O-C)||^2. Robust to non-intersecting sight lines.
    projectors = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    A = projectors.sum(axis=0)
    b = np.einsum("nij,nj->i", projectors, centers)
    if np.linalg.cond(A) > 1e7:
        raise ValueError("Cannot estimate common sight center: input rays nearly parallel")
    return np.linalg.solve(A, b)


def _estimate_scene_up(poses):
    """Fit common gravity-up from camera right axes, oriented by camera ups.

    Camera right should be perpendicular to the world-up axis. This suppresses
    inconsistent camera roll better than averaging the camera up vectors alone.
    Near-degenerate (nearly identical right axes) configurations use the prior.
    Absolute gravity remains unobservable if all input cameras share a roll.
    """
    R = poses[:, :3, :3]
    prior = _unit(-R[:, 1, :].mean(axis=0))  # OpenCV camera Y points down
    rights = R[:, 0, :]
    gram = rights.T @ rights
    eigenvalues, vectors = np.linalg.eigh(gram)
    if eigenvalues[1] - eigenvalues[0] < 1e-5 * max(float(eigenvalues[-1]), 1.0):
        # Nearly parallel horizontal axes cannot determine unique world-up.
        up = prior
    else:
        up = vectors[:, 0]
        if np.dot(up, prior) < 0:
            up = -up
    return _unit(up)


def _corner_round_positions(positions, knots, times, fraction):
    """Localized C1 corner rounding without global spline overshoot.

    Only a small window adjacent to each interior control pose is modified.
    All control positions stay fixed; the path outside the windows is linear.
    """
    base = np.column_stack([np.interp(times, knots, positions[:, j])
                            for j in range(3)])
    if fraction <= 0 or len(positions) < 3:
        return base
    fraction = min(float(fraction), 0.24)

    def hermite(p0, p1, v0, v1, t0, t1, ts):
        dt = t1 - t0
        u = ((ts - t0) / dt)[:, None]
        return ((2*u**3 - 3*u**2 + 1)*p0
                + (u**3 - 2*u**2 + u)*dt*v0
                + (-2*u**3 + 3*u**2)*p1
                + (u**3 - u**2)*dt*v1)

    for j in range(1, len(positions)-1):
        prev_dt = knots[j] - knots[j-1]
        next_dt = knots[j+1] - knots[j]
        left = knots[j] - fraction * prev_dt
        right = knots[j] + fraction * next_dt
        v_before = (positions[j] - positions[j-1]) / prev_dt
        v_after = (positions[j+1] - positions[j]) / next_dt
        # Time-weighted common tangent; avoids speed jumps at the corner.
        tangent = (v_before * next_dt + v_after * prev_dt) / (prev_dt + next_dt)
        p_left = positions[j] - (knots[j]-left)*v_before
        p_right = positions[j] + (right-knots[j])*v_after
        first = (times >= left) & (times <= knots[j])
        second = (times > knots[j]) & (times <= right)
        if np.any(first):
            base[first] = hermite(p_left, positions[j], v_before, tangent,
                                  left, knots[j], times[first])
        if np.any(second):
            base[second] = hermite(positions[j], p_right, tangent, v_after,
                                   knots[j], right, times[second])
    return base


def _interp_poses(controls, count, *, corner_smoothing=0.15):
    """Linear/SLERP baseline with optional gentle local position corner rounding.

    No global spline and no rotation-spline smoothing. Exact end input poses.
    corner_smoothing=0 reproduces the previous trajectory exactly.
    """
    positions = _centers(controls)
    rotations = Rotation.from_matrix(controls[:, :3, :3])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    angles = (rotations[:-1].inv() * rotations[1:]).magnitude()
    lengths = np.maximum(steps + 0.25 * angles *
                         max(np.linalg.norm(positions, axis=1).mean(), 1e-6), 1e-5)
    knots = np.concatenate(([0.0], np.cumsum(lengths)))
    times = np.linspace(0, knots[-1], count)
    cam_centers = _corner_round_positions(positions, knots, times, corner_smoothing)
    rot = Slerp(knots, rotations)(times).as_matrix()
    result = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    result[:, :3, :3] = rot
    result[:, :3, 3] = -np.einsum("nij,nj->ni", rot, cam_centers)
    result[0], result[-1] = controls[0], controls[-1]
    return result


def plan_trajectories(extrinsics, *, extra=False, extra_pitch_deg=5.0,
                      view_num=81, yaw_threshold=45.0, pitch_threshold=10.0,
                      circle=False, corner_smoothing=0.15):
    if view_num < 2:
        raise ValueError("view_num must be >= 2")
    poses = _w2c44(extrinsics)
    if len(poses) < 2:
        raise ValueError("At least two input poses required")
    centers = _centers(poses)
    forward = poses[:, 2, :3]
    forward = np.array([_unit(d) for d in forward])
    origin = _estimate_center(centers, forward)
    # Camera down is +Y in OpenCV. Recover scene up from all camera orientations.
    up = _estimate_scene_up(poses)
    z = centers[0] - origin
    z = _unit(z - np.dot(z, up) * up)
    if np.linalg.norm(z) < 1e-6:
        raise ValueError("First camera lies on estimated up axis")
    x = _unit(np.cross(up, z))
    z = _unit(np.cross(x, up))
    Q = np.stack((x, up, z), axis=1)  # normalized -> original
    local = (centers - origin) @ Q
    radius = float(np.linalg.norm(local, axis=1).mean())
    if radius < 1e-6:
        raise ValueError("Estimated camera sphere has near-zero radius")
    yaws = np.degrees(np.arctan2(-local[:, 0], local[:, 2]))
    pitches = np.degrees(np.arcsin(np.clip(local[:, 1] / np.linalg.norm(local, axis=1), -1, 1)))
    # unwrap only when the input range crosses +/-180; avoid a spurious 350-degree span
    yaw_unwrapped = np.degrees(np.unwrap(np.radians(yaws)))
    if np.ptp(yaw_unwrapped) > 180 and not circle:
        raise ValueError("Input yaw spans >180 degrees; inspect circle flag or input pose quality")
    order = np.argsort(yaw_unwrapped)
    ordered = [(float(yaw_unwrapped[i]), int(i)) for i in order]
    if circle:
        ordered.append((ordered[0][0] + 360.0, ordered[0][1]))
    lo = float(pitches.min() - (extra_pitch_deg if extra else 0.0))
    hi = float(pitches.max() + (extra_pitch_deg if extra else 0.0))

    def sphere_pose(pitch, yaw):
        a, b = np.radians([pitch, yaw])
        loc = radius * np.array([-np.sin(b)*np.cos(a), np.sin(a), np.cos(b)*np.cos(a)])
        world_center = origin + Q @ loc
        return _look_at(world_center, origin, up)

    trajectories = []
    for mode, limit in (("up", hi), ("down", lo)):
        current = 0
        while current < len(ordered) - 1:
            possible = [j for j in range(current + 1, len(ordered))
                        if ordered[j][0] - ordered[current][0] <= yaw_threshold]
            if not possible:
                possible = [current + 1]
            pitch_ref = (max if mode == "up" else min)(pitches[ordered[j][1]] for j in possible)
            possible = [j for j in possible
                        if abs(pitches[ordered[j][1]] - pitch_ref) <= pitch_threshold]
            target = max(possible)
            a_yaw, a = ordered[current]
            b_yaw, b = ordered[target]
            controls = [poses[a], sphere_pose(limit, a_yaw)]
            if b_yaw - a_yaw > 180:
                controls.append(sphere_pose(limit, (a_yaw+b_yaw)/2))
            controls.extend([sphere_pose(limit, b_yaw), poses[b]])
            # Collapse duplicate control points; Slerp requires strictly ascending knots.
            segment = _interp_poses(np.stack(controls), view_num,
                                    corner_smoothing=corner_smoothing)
            trajectories.append({"mode": mode, "start_index": a, "end_index": b,
                                 "start_yaw": a_yaw, "end_yaw": b_yaw,
                                 "extrinsics": segment})
            current = target
    info = {"interpolation": "local_corner_rounding_slerp" if corner_smoothing > 0 else "piecewise_linear_slerp",
            "corner_smoothing": float(corner_smoothing), "center": origin.tolist(), "estimated_scene_up": up.tolist(), "rotation_normalized_to_world": Q.tolist(),
            "radius": radius, "yaws": yaw_unwrapped.tolist(), "pitches": pitches.tolist(),
            "yaw_min": float(min(yaw_unwrapped)), "yaw_max": float(max(yaw_unwrapped)),
            "pitch_min": lo, "pitch_max": hi, "extra": bool(extra), "circle": bool(circle),
            "trajectory_count": len(trajectories)}
    return trajectories, info


def save_trajectories(trajectories, info, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(info)
    manifest["trajectories"] = []
    for i, item in enumerate(trajectories):
        name = f"{i:02d}_{item['mode']}_{item['start_index']:03d}_{item['end_index']:03d}"
        np.save(out / f"{name}.npy", item["extrinsics"])
        manifest["trajectories"].append({k: v for k, v in item.items() if k != "extrinsics"} |
                                        {"file": f"{name}.npy", "frames": len(item["extrinsics"])})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
