"""Audit pose, scale and intrinsic consistency of multi-image dense input.

Run on the reconstruction directory:
    python scripts/reconstruction/audit_multi_image_alignment.py /path/to/case/reconstruction

This is intentionally read-only and does not require CUDA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .multi_image_dense_input import _read_cameras, _read_images, _camera_fx_fy_cx_cy
except ImportError:
    from multi_image_dense_input import _read_cameras, _read_images, _camera_fx_fy_cx_cy


def _pose(record):
    q, t = record[0], record[1]
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    pose[:3, 3] = np.asarray(t, dtype=np.float64)
    return pose


def _center(w2c):
    r, t = w2c[:3, :3], w2c[:3, 3]
    return -r.T @ t


def _difference(a, b):
    relative = a[:3, :3] @ b[:3, :3].T
    deg = float(np.degrees(Rotation.from_matrix(relative).magnitude()))
    return float(np.linalg.norm(_center(a) - _center(b))), deg


def audit(root):
    root = Path(root).resolve()
    orig_sparse = root / "sparse" / "0"
    dense_sparse = root / "dense_input" / "sparse" / "0"
    traj_manifest = json.loads((root / "trajectories" / "manifest.json").read_text(encoding="utf-8"))
    dense_manifest = json.loads((root / "dense_input" / "input_manifest.json").read_text(encoding="utf-8"))
    original_images = _read_images(orig_sparse)
    original_cameras = _read_cameras(orig_sparse)
    dense_images = _read_images(dense_sparse)
    dense_cameras = _read_cameras(dense_sparse)
    original_by_name = {record[3]: _pose(record) for record in original_images.values()}
    normalized = sorted((root / "da3_input").glob("*.jpg"))
    points = np.array([_center(p) for p in original_by_name.values()])
    baseline = float(np.linalg.norm(points[:, None] - points[None], axis=-1).max())
    baseline = max(baseline, 1e-8)
    errors = []
    notices = []
    trajectory_checks = []
    for trajectory in traj_manifest["trajectories"]:
        name = Path(trajectory["file"]).stem
        poses = np.asarray(np.load(root / "trajectories" / trajectory["file"]))
        if poses.shape != (int(trajectory["frames"]), 4, 4):
            errors.append(f"{name}: trajectory shape {poses.shape}")
            continue
        for frame, source_index in [(0, int(trajectory["start_index"])),
                                    (len(poses)-1, int(trajectory["end_index"]))]:
            original_name = normalized[source_index].name
            if original_name not in original_by_name:
                errors.append(f"COLMAP source image not found: {original_name}")
                continue
            translation, rotation = _difference(poses[frame], original_by_name[original_name])
            normalized_error = translation / baseline
            row = dict(trajectory=name, frame=frame, source=original_name,
                       center_delta=translation, baseline_relative_delta=normalized_error,
                       rotation_delta_deg=rotation)
            trajectory_checks.append(row)
            if normalized_error > 0.01 or rotation > 1:
                errors.append(f"trajectory endpoint does NOT match DA3 COLMAP: {row}")
    dense_name_to_record = {record[3]: record for record in dense_images.values()}
    original_name_to_record = {record[3]: record for record in original_images.values()}
    for name, record in original_name_to_record.items():
        if name not in dense_name_to_record:
            errors.append(f"Original image lost in dense COLMAP: {name}")
            continue
        dc, dr = dense_name_to_record[name], record
        pos, angle = _difference(_pose(dc), _pose(dr))
        if pos > 1e-7 or angle > 0.001:
            errors.append(f"Original pose modified during merge: {name} center={pos}, angle={angle}")
        if tuple(dc[0]) != tuple(dr[0]) or tuple(dc[1]) != tuple(dr[1]):
            notices.append(f"Original pose serialized with slightly different floating precision: {name}")

    generated_checks = []
    for item in dense_manifest["generated"]:
        name = item["image"]
        if name not in dense_name_to_record:
            errors.append(f"Missing dense COLMAP image {name}")
            continue
        record = dense_name_to_record[name]
        pose = _pose(record)
        source_pose = np.load(item["pose_file"])[int(item["frame"])]
        position, angle = _difference(pose, source_pose)
        path = root / "dense_input" / "image" / name
        image = cv2.imread(str(path))
        if image is None:
            errors.append(f"Dense image missing/corrupt: {name}")
            continue
        h, w = image.shape[:2]
        camera = dense_cameras[record[2]]
        _, cw, ch, params = camera
        if cw != w or ch != h:
            errors.append(f"Camera dimensions mismatch: {name}, image={w}x{h} camera={cw}x{ch}")
        fx, fy, cx, cy, _, _ = _camera_fx_fy_cx_cy(camera)
        if not (fx > 0 and fy > 0 and 0 <= cx <= w and 0 <= cy <= h):
            errors.append(f"Invalid intrinsics for {name}: {params}")
        if position > 1e-6 or angle > 0.001:
            errors.append(f"Generated COLMAP pose differs from planned path: {name}, "
                          f"center={position}, rot={angle}")
        generated_checks.append(dict(image=name, width=w, height=h,
                                     fx=fx, fy=fy, cx=cx, cy=cy,
                                     center_delta=position, rotation_delta_deg=angle))
    original_sizes = []
    for name, record in original_name_to_record.items():
        image = cv2.imread(str(root / "dense_input" / "image" / name))
        if image is None:
            errors.append(f"Original normalized JPG missing: {name}")
            continue
        camera = original_cameras[record[2]]
        if (camera[1], camera[2]) != (image.shape[1], image.shape[0]):
            errors.append(f"Original camera dimensions mismatch: {name}, "
                          f"image={image.shape[1]}x{image.shape[0]}, "
                          f"COLMAP={camera[1]}x{camera[2]}")
        original_sizes.append(dict(name=name, image_width=image.shape[1], image_height=image.shape[0],
                                   camera_width=camera[1], camera_height=camera[2]))
    if len(generated_checks) != int(dense_manifest["sampled_generated_views"]):
        errors.append("Generated views do not match dense input manifest")
    if len(dense_images) != len(original_images) + len(generated_checks):
        errors.append(f"Dense COLMAP view count {len(dense_images)} differs from "
                      f"{len(original_images)} originals + {len(generated_checks)} generated")

    # The DA3 GS renderer receives prediction.gaussians in its prediction
    # coordinate system; the trajectory planner receives adjusted translations.
    # Without access to the DA3 exporter code/scale_factor, we cannot prove these
    # are the same frame. Flag for inspection even if endpoint tests pass.
    notices.append(
        "GS coordinate-scale check requires DA3 prediction.scale_factor and "
        "export_to_gs_video source. Endpoint consistency alone cannot verify "
        "Gaussian-vs-camera rendering scale.")
    report = dict(status="PASS" if not errors else "FAIL",
                  original_count=len(original_images), generated_count=len(generated_checks),
                  colmap_count=len(dense_images), source_baseline=baseline,
                  endpoint_checks=trajectory_checks,
                  original_sizes=original_sizes, generated_intrinsics=generated_checks,
                  errors=errors, notices=notices)
    out = root / "dense_input" / "alignment_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ALIGNMENT AUDIT] {report['status']} originals={len(original_images)} "
          f"generated={len(generated_checks)} baseline={baseline:.6f} errors={len(errors)}")
    for row in trajectory_checks:
        print(f"  {row['trajectory']} frame={row['frame']} "
              f"center/baseline={row['baseline_relative_delta']:.5f} "
              f"rot={row['rotation_delta_deg']:.4f}deg")
    for issue in errors:
        print("[ERROR]", issue)
    print(f"[ALIGNMENT AUDIT] saved {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reconstruction_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.reconstruction_dir)
    raise SystemExit(0 if result["status"] == "PASS" else 1)
