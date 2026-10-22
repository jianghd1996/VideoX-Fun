"""Prepare a dense 3DGS workspace from generated multi-view trajectory videos.

Preserve DA3's original COLMAP camera poses, feature observations and sparse
point tracks. Add sampled intermediate video frames with their known w2c poses.
Does not re-estimate camera poses or overwrite reconstruction/sparse/0.
"""
from __future__ import annotations

import json
import math
import shutil
import struct
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# Standard COLMAP camera models; number of double parameters.
MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_IDS = {name: idx for idx, (name, _) in MODELS.items()}


def _read_cameras(base):
    binary = base / "cameras.bin"
    out = {}
    if binary.is_file():
        with binary.open("rb") as file:
            (count,) = struct.unpack("<Q", file.read(8))
            for _ in range(count):
                camera_id, model_id, width, height = struct.unpack("<iiQQ", file.read(24))
                if model_id not in MODELS:
                    raise ValueError(f"Unsupported COLMAP camera model ID {model_id}")
                size = MODELS[model_id][1]
                params = struct.unpack("<" + "d" * size, file.read(8 * size))
                out[camera_id] = (model_id, int(width), int(height), tuple(params))
    else:
        path = base / "cameras.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Neither cameras.bin nor cameras.txt found at {base}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cols = line.split()
            out[int(cols[0])] = (MODEL_IDS[cols[1]], int(cols[2]),
                                int(cols[3]), tuple(float(v) for v in cols[4:]))
    if not out:
        raise ValueError(f"No COLMAP cameras in {base}")
    return out


def _read_cstr(file):
    chars = bytearray()
    while True:
        char = file.read(1)
        if not char:
            raise EOFError("Unexpected end of COLMAP image name")
        if char == b"\0":
            return chars.decode("utf-8")
        chars.extend(char)


def _read_images(base):
    binary = base / "images.bin"
    out = {}
    if binary.is_file():
        with binary.open("rb") as file:
            (count,) = struct.unpack("<Q", file.read(8))
            for _ in range(count):
                unpacked = struct.unpack("<i7di", file.read(64))
                image_id = unpacked[0]
                qvec = unpacked[1:5]
                tvec = unpacked[5:8]
                camera_id = unpacked[8]
                name = _read_cstr(file)
                (points_count,) = struct.unpack("<Q", file.read(8))
                observations = file.read(points_count * 24)
                if len(observations) != points_count * 24:
                    raise EOFError(f"Truncated observations in {name}")
                out[image_id] = (qvec, tvec, camera_id, name, observations)
    else:
        path = base / "images.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Neither images.bin nor images.txt found at {base}")
        lines = path.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 10:
                raise ValueError(f"Invalid COLMAP images.txt header: {line}")
            image_id = int(fields[0])
            qvec = tuple(map(float, fields[1:5]))
            tvec = tuple(map(float, fields[5:8]))
            camera_id = int(fields[8])
            name = fields[9]
            # Every COLMAP image record has a subsequent feature-observation line,
            # which may be blank for unregistered points.
            points_line = lines[i].strip() if i < len(lines) else ""
            i += 1
            values = points_line.split()
            if len(values) % 3:
                raise ValueError(f"Invalid COLMAP 2D points for {name}")
            observations = b"".join(
                struct.pack("<ddq", float(values[k]), float(values[k+1]), int(values[k+2]))
                for k in range(0, len(values), 3))
            out[image_id] = (qvec, tvec, camera_id, name, observations)
    if not out:
        raise ValueError(f"No COLMAP image records in {base}")
    return out


def _write_binary_model(base, cameras, images, source_sparse):
    base.mkdir(parents=True, exist_ok=True)
    with (base / "cameras.bin").open("wb") as file:
        file.write(struct.pack("<Q", len(cameras)))
        for cid, (mid, width, height, params) in sorted(cameras.items()):
            file.write(struct.pack("<iiQQ", cid, mid, width, height))
            file.write(struct.pack("<" + "d" * len(params), *params))
    with (base / "images.bin").open("wb") as file:
        file.write(struct.pack("<Q", len(images)))
        for iid, (qvec, tvec, cid, name, observations) in sorted(images.items()):
            file.write(struct.pack("<i7di", iid, *qvec, *tvec, cid))
            file.write(name.encode("utf-8") + b"\0")
            file.write(struct.pack("<Q", len(observations) // 24))
            file.write(observations)
    source_points = source_sparse / "points3D.bin"
    if source_points.is_file():
        shutil.copy2(source_points, base / "points3D.bin")
    else:
        source_points_txt = source_sparse / "points3D.txt"
        if not source_points_txt.is_file():
            raise FileNotFoundError("Missing DA3 COLMAP points3D.bin/txt")
        # COLMAP expects all three sparse model files in the same format.
        # Convert text points (including tracks) to binary instead of writing
        # an empty point cloud, retaining valid input-image feature associations.
        entries = []
        for line in source_points_txt.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = line.split()
            pid = int(values[0])
            xyz = [float(v) for v in values[1:4]]
            rgb = [int(v) for v in values[4:7]]
            error = float(values[7])
            track = [int(v) for v in values[8:]]
            if len(track) % 2:
                raise ValueError(f"Invalid points3D track for {pid}")
            entries.append((pid, xyz, rgb, error, track))
        with (base / "points3D.bin").open("wb") as file:
            file.write(struct.pack("<Q", len(entries)))
            for pid, xyz, rgb, error, track in entries:
                file.write(struct.pack("<QdddBBBdQ", pid, *xyz, *rgb,
                                       error, len(track) // 2))
                for j in range(0, len(track), 2):
                    file.write(struct.pack("<ii", track[j], track[j+1]))


def _camera_fx_fy_cx_cy(camera):
    mid, width, height, params = camera
    model = MODELS[mid][0]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                 "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        f, cx, cy = params[:3]
        return f, f, cx, cy, width, height
    # PINHOLE, OPENCV, FULL_OPENCV, FOV, THIN_PRISM_FISHEYE and OPENCV_FISHEYE
    fx, fy, cx, cy = params[:4]
    return fx, fy, cx, cy, width, height


def _sample_indices(lengths, total):
    """Allocate an approximately equal number of interior frames per trajectory."""
    available = np.array([max(0, n - 2) for n in lengths], dtype=int)
    total = min(int(total), int(available.sum()))
    if total <= 0:
        raise ValueError("No usable intermediate generated frames")
    allocations = np.zeros(len(lengths), dtype=int)
    for _ in range(total):
        ratios = (allocations + 1) / np.maximum(available, 1)
        ratios[available <= allocations] = float("inf")
        # Bias towards trajectories with more frames if ratios are tied.
        choice = int(np.argmin(ratios))
        allocations[choice] += 1
    indices = []
    for n, count in zip(lengths, allocations):
        if count == 0:
            indices.append([])
        else:
            indices.append(np.linspace(1, n - 2, count).round().astype(int).tolist())
    return indices


def prepare_dense_input(*, reconstruction_dir, sample_count=40, source_images=None):
    """Create dense_input/image and dense_input/sparse/0 without modifying DA3 outputs."""
    started = time.perf_counter()
    root = Path(reconstruction_dir)
    source_sparse = root / "sparse" / "0"
    output = root / "dense_input"
    image_dir = output / "image"
    sparse_dir = output / "sparse" / "0"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "generated_trajectories" / "manifest.json"
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not items:
        raise ValueError(f"No generated trajectories in {manifest_path}")
    cameras = _read_cameras(source_sparse)
    images = _read_images(source_sparse)
    names = {record[3] for record in images.values()}

    # DA3 source images are normalized, indexed JPGs. Preserve the original
    # images and IDs to maintain 3D point tracks from DA3.
    normalized = sorted((root / "da3_input").glob("*.jpg"))
    if len(normalized) != len(names):
        raise ValueError(f"Source images mismatch: {len(normalized)} JPGs, {len(names)} COLMAP views")
    for p in normalized:
        if p.name not in names:
            raise ValueError(f"DA3 COLMAP image {p.name} not found among original image records")
        shutil.copy2(p, image_dir / p.name)
    name_to_camera = {record[3]: record[2] for record in images.values()}

    frame_choices = _sample_indices([int(item["frames"]) for item in items], sample_count)
    last_camera_id = max(cameras)
    last_image_id = max(images)
    generated = []
    for item, selected in zip(items, frame_choices):
        if not selected:
            continue
        video = Path(item["video"])
        pose_file = Path(item["pose_file"])
        poses = np.load(pose_file)
        if poses.shape != (int(item["frames"]), 4, 4):
            raise ValueError(f"Pose mismatch for {video}")
        start_index, end_index = int(item["start_index"]), int(item["end_index"])
        a = _camera_fx_fy_cx_cy(cameras[name_to_camera[normalized[start_index].name]])
        b = _camera_fx_fy_cx_cy(cameras[name_to_camera[normalized[end_index].name]])
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot decode generated video: {video}")
        needed = set(selected)
        frame_index = 0
        written = set()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index in needed:
                    h, w = frame.shape[:2]
                    alpha = frame_index / (len(poses) - 1)
                    ka = np.array([a[0]*w/a[4], a[1]*h/a[5], a[2]*w/a[4], a[3]*h/a[5]])
                    kb = np.array([b[0]*w/b[4], b[1]*h/b[5], b[2]*w/b[4], b[3]*h/b[5]])
                    fx, fy, cx, cy = (1-alpha)*ka + alpha*kb
                    last_camera_id += 1
                    cameras[last_camera_id] = (1, w, h, (float(fx), float(fy), float(cx), float(cy)))
                    pose = poses[frame_index]
                    q_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
                    q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
                    tvec = tuple(float(v) for v in pose[:3, 3])
                    name = f"{item['name']}_f{frame_index:04d}.jpg"
                    if not cv2.imwrite(str(image_dir / name), frame):
                        raise IOError(f"Failed to save {name}")
                    last_image_id += 1
                    images[last_image_id] = (q_wxyz, tvec, last_camera_id, name, b"")
                    generated.append({"image": name, "trajectory": item["name"],
                                      "frame": frame_index, "pose_file": str(pose_file)})
                    written.add(frame_index)
                frame_index += 1
        finally:
            cap.release()
        if frame_index != len(poses):
            raise ValueError(f"Video {video} frames {frame_index}, expected {len(poses)}")
        if written != needed:
            raise RuntimeError(f"Missing frames {sorted(needed - written)} in {video}")
        print(f"[DENSE INPUT] {item['name']}: sampled {len(written)} from {frame_index}", flush=True)

    _write_binary_model(sparse_dir, cameras, images, source_sparse)
    summary = {"input_views": len(normalized), "sampled_generated_views": len(generated),
               "total_views": len(images), "sample_budget": int(sample_count),
               "generated": generated, "work_dir": str(output),
               "seconds": round(time.perf_counter() - started, 3)}
    (output / "input_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DENSE INPUT] original={len(normalized)} generated={len(generated)} "
          f"total={len(images)} time={summary['seconds']:.3f}s", flush=True)
    return summary
