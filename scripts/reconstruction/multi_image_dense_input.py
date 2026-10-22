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



def _validate_ply(path):
    """Require a nonempty PLY with xyz properties; fail before expensive 3DGS."""
    with Path(path).open("rb") as fp:
        if fp.readline().strip() != b"ply":
            raise ValueError(f"Invalid PLY header: {path}")
        count = None
        properties = []
        in_vertex = False
        for _ in range(256):
            line = fp.readline()
            if not line:
                break
            row = line.decode("ascii", errors="replace").strip()
            if row.startswith("element "):
                words = row.split()
                in_vertex = words[1] == "vertex"
                if in_vertex:
                    count = int(words[2])
            elif row.startswith("property ") and in_vertex:
                properties.append(row.split()[-1])
            elif row == "end_header":
                break
        else:
            raise ValueError(f"PLY header too long: {path}")
    if count is None or count <= 0:
        raise ValueError(f"PLY initialization has no points: {path}")
    if not {"x", "y", "z"}.issubset(properties):
        raise ValueError(f"PLY is missing xyz properties: {path}")
    if Path(path).stat().st_size < 256:
        raise ValueError(f"PLY is suspiciously small: {path}")
    return count


def _colmap_points_to_ply(points_bin, ply_path):
    """Convert COLMAP points3D.bin to conventional binary PLY xyz/normals/RGB.

    Matches common 3DGS fetchPly requirements: x,y,z,nx,ny,nz,red,green,blue.
    The COLMAP point coordinates are already in the same world frame as the
    original three camera poses, so there is no additional scale transform.
    """
    with Path(points_bin).open("rb") as source:
        (count,) = struct.unpack("<Q", source.read(8))
        if count == 0:
            raise ValueError(f"Cannot initialize 3DGS from empty COLMAP points: {points_bin}")
        with Path(ply_path).open("wb") as target:
            target.write((
                "ply\\nformat binary_little_endian 1.0\\n"
                f"element vertex {count}\\n"
                "property float x\\nproperty float y\\nproperty float z\\n"
                "property float nx\\nproperty float ny\\nproperty float nz\\n"
                "property uchar red\\nproperty uchar green\\nproperty uchar blue\\n"
                "end_header\\n"
            ).replace("\\n", "\n").encode("ascii"))
            vertex = struct.Struct("<ffffffBBB")
            for _ in range(count):
                fixed = source.read(43)
                if len(fixed) != 43:
                    raise EOFError(f"Truncated COLMAP point: {points_bin}")
                pid, x, y, z, red, green, blue, error, track_length = struct.unpack(
                    "<QdddBBBdQ", fixed)
                target.write(vertex.pack(float(x), float(y), float(z),
                                         0., 0., 0., red, green, blue))
                source.seek(int(track_length) * 8, 1)
    return count


def _prepare_initialization_cloud(source_sparse, sparse_dir):
    """Always provide a valid points3D.ply for Reconstruction_full."""
    source_ply = source_sparse / "points3D.ply"
    output_ply = sparse_dir / "points3D.ply"
    if source_ply.is_file():
        # Prefer DA3's own exported cloud; retain its geometry and colors.
        shutil.copy2(source_ply, output_ply)
        origin = "DA3 points3D.ply"
    else:
        _colmap_points_to_ply(sparse_dir / "points3D.bin", output_ply)
        origin = "converted COLMAP points3D.bin"
    count = _validate_ply(output_ply)
    print(f"[DENSE INIT] {origin}: {count} points -> {output_ply}", flush=True)
    return {"path": str(output_ply), "points": count, "source": origin}


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


def _camera_features(w2cs, radius):
    """Camera centers and forward directions in the common COLMAP world frame.

    The angular term is equivalent to comparing viewing rays on a sphere with
    radius = 0.5 * scene camera baseline; this avoids collapsing roll/yaw-only
    changes at approximately the same center.
    """
    poses = np.asarray(w2cs, dtype=np.float64)
    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3]
    centers = -np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), translations)
    forward = np.einsum("nij,j->ni", rotations.transpose(0, 2, 1),
                        np.array([0., 0., 1.]))
    return np.concatenate([centers, forward * (0.5 * radius)], axis=1)


def _global_pose_samples(items, source_images, total, min_per_trajectory=4):
    """Global FPS of generated poses, seeded by already retained source views.

    Avoid segment endpoints (original images), reserve a small minimum across
    the trajectories, then greedily maximize nearest-neighbor pose distance
    across all remaining generated candidates. Deterministic tie-breaking.
    """
    original = []
    for image_id, (q, t, _, name, _) in sorted(source_images.items()):
        rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = t
        original.append(pose)

    trajectories = []
    candidates = []
    affiliations = []
    for segment, item in enumerate(items):
        poses = np.asarray(np.load(item["pose_file"]), dtype=np.float64)
        expected = int(item["frames"])
        if poses.shape != (expected, 4, 4):
            raise ValueError(f"Invalid trajectory pose shape {poses.shape} for {item['name']}")
        trajectories.append(poses)
        for frame in range(1, expected - 1):
            candidates.append(poses[frame])
            affiliations.append((segment, frame))
    if not candidates:
        raise ValueError("There are no intermediate generated frames to sample")

    poses = np.concatenate([np.asarray(original), np.asarray(candidates)])
    all_centers = -np.einsum("nij,nj->ni",
                             poses[:, :3, :3].transpose(0, 2, 1),
                             poses[:, :3, 3])
    extents = np.linalg.norm(all_centers[:, None] - all_centers[None], axis=2) if len(all_centers) <= 500 else None
    baseline = float(np.max(extents)) if extents is not None else float(np.linalg.norm(np.ptp(all_centers, axis=0)))
    if baseline < 1e-6:
        baseline = 1.0
    features = _camera_features(poses, baseline) / baseline
    source_count = len(original)
    available = len(candidates)
    count = min(int(total), available)
    if count <= 0:
        raise ValueError("dense_generated_frames must be >0")
    candidate_features = features[source_count:]
    distances = np.full(available, np.inf, dtype=np.float64)
    for anchor in features[:source_count]:
        distances = np.minimum(distances, np.sum((candidate_features - anchor) ** 2, axis=1))

    selected = []
    taken = np.zeros(available, dtype=bool)

    def choose(index):
        taken[index] = True
        selected.append(index)
        new_dist = np.sum((candidate_features - candidate_features[index]) ** 2, axis=1)
        np.minimum(distances, new_dist, out=distances)
        distances[taken] = -np.inf

    # Coverage guarantee: up to 4 useful novel views per trajectory; seed using
    # exactly the same farthest-point score while enforcing segment membership.
    members = [np.array([i for i, (segment, _) in enumerate(affiliations) if segment == j],
                        dtype=int) for j in range(len(items))]
    reserve = min(int(min_per_trajectory), count // max(len(items), 1))
    for _ in range(reserve):
        for ids in members:
            allowed = ids[~taken[ids]]
            if len(allowed):
                choose(int(allowed[np.argmax(distances[allowed])]))
    while len(selected) < count:
        choose(int(np.argmax(distances)))

    result = [[] for _ in items]
    for idx in selected:
        segment, frame = affiliations[idx]
        result[segment].append(int(frame))
    for frames in result:
        frames.sort()
    return result



def prepare_dense_input(*, reconstruction_dir, sample_count=80, source_images=None):
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
    frame_choices = _global_pose_samples(items, images, sample_count)
    print("[DENSE POSE FPS] " + ", ".join(
        f"{item['name']}={len(indices)} frames" for item, indices in zip(items, frame_choices)),
        flush=True)
    # Remove stale JPGs from earlier sampling strategies before rebuilding.
    retained = {p.name for p in normalized}
    for old in image_dir.glob("*.jpg"):
        if old.name not in retained:
            old.unlink()
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
    initialization = _prepare_initialization_cloud(source_sparse, sparse_dir)
    summary = {"input_views": len(normalized), "sampled_generated_views": len(generated),
               "total_views": len(images), "sample_budget": int(sample_count),
               "sampling_strategy": "global_pose_fps_v1",
               "selected_indices": {item["name"]: ids for item, ids in zip(items, frame_choices)},
               "generated": generated, "work_dir": str(output),
               "initialization_cloud": initialization,
               "seconds": round(time.perf_counter() - started, 3)}
    (output / "input_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DENSE INPUT] original={len(normalized)} generated={len(generated)} "
          f"total={len(images)} time={summary['seconds']:.3f}s", flush=True)
    return summary
