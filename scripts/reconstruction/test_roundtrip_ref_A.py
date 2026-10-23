#!/usr/bin/env python3
"""Experiment A: B -> A -> B, fixed B endpoints, *reference* A only.

No frame-40 inpaint condition. This deliberately tests whether add_ref_conv
alone helps an intermediate target view. Does not change production pipeline.
Run from CodexRecon root:
  CUDA_VISIBLE_DEVICES=2 python test_roundtrip_ref_A.py --gpu 2 --a-index 1 --b-index 0

Outputs: <case>/reconstruction/ablation_A_BAB/{poses.npy,control.mp4,
         generated.mp4,sampled_frames/,experiment.json}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE_DEFAULT = Path("/home/z00566689/dev/mnt/jiang_test/00_1022_test/cover2_e2e")


def _smoothstep(x):
    return x * x * (3 - 2 * x)


def _slerp_unit(a, b, t):
    dot = float(np.clip(np.dot(a, b), -1, 1))
    angle = float(np.arccos(dot))
    if angle < 1e-5:
        x = (1 - t) * a + t * b
    elif np.pi - angle < 1e-5:
        raise ValueError("B and A camera directions almost opposite; cannot interpolate reliably")
    else:
        x = (np.sin((1 - t) * angle) * a + np.sin(t * angle) * b) / np.sin(angle)
    return x / np.linalg.norm(x)


def plan_roundtrip(extrinsics, a_index, b_index, pitch_deg, frame_count=81):
    """Two hemisphere detours with exact input W2C at frames 0, mid and last."""
    from camera_trajectory import _w2c44, _centers, _estimate_center, _estimate_scene_up, _look_at
    if frame_count < 5 or frame_count % 4 != 1 or frame_count % 2 != 1:
        raise ValueError("Wan requires 4n+1 odd frame count")
    poses = _w2c44(extrinsics)
    center = _estimate_center(_centers(poses), poses[:, 2, :3])
    up = _estimate_scene_up(poses)
    centers = _centers(poses)
    a, b = centers[a_index], centers[b_index]
    da, db = a - center, b - center
    ra, rb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    da, db = da / ra, db / rb
    mid = (frame_count - 1) // 2
    out = np.tile(np.eye(4, dtype=np.float64), (frame_count, 1, 1))
    # Half 1 B->A over upper hemisphere; half 2 A->B over lower hemisphere.
    # smoothstep gives zero pose velocity at A and B; squared sine makes the
    # pitch deviation and its derivative zero at the true view anchors.
    for i in range(frame_count):
        if i <= mid:
            u = i / mid
            src, dst, r0, r1, sign = db, da, rb, ra, 1
        else:
            u = (i - mid) / mid
            src, dst, r0, r1, sign = da, db, ra, rb, -1
        t = _smoothstep(u)
        direction = _slerp_unit(src, dst, t)
        deviation = sign * np.tan(np.deg2rad(pitch_deg)) * (np.sin(np.pi * u) ** 2)
        direction = direction + deviation * up
        direction = direction / np.linalg.norm(direction)
        radius = (1 - t) * r0 + t * r1
        camera_center = center + radius * direction
        out[i] = _look_at(camera_center, center, up)
    # The raw look-at orientation differs slightly from DA3's true camera
    # orientation. Simply overwriting frames 0/mid/last creates an abrupt
    # rotation at 0->1, mid-1->mid->mid+1 and last-1->last. Correct the
    # ENTIRE path's rotations, interpolating SO(3) anchor residuals with
    # zero velocity at anchors. Keep spherical positions unchanged.
    from scipy.spatial.transform import Rotation
    anchors = (0, mid, frame_count - 1)
    source_indices = (b_index, a_index, b_index)
    residual = [
        Rotation.from_matrix(poses[src, :3, :3] @ out[frame, :3, :3].T)
        for frame, src in zip(anchors, source_indices)
    ]
    for seg in range(2):
        lo, hi = anchors[seg], anchors[seg + 1]
        start, end = residual[seg], residual[seg + 1]
        # Lie-group interpolation. End-to-end relative rotation is smooth
        # because the derivative of smoothstep is zero at each anchor.
        relative = start.inv() * end
        rotvec = relative.as_rotvec()
        for i in range(lo, hi + 1):
            u = (i - lo) / (hi - lo)
            corr = start * Rotation.from_rotvec(_smoothstep(u) * rotvec)
            out[i, :3, :3] = corr.as_matrix() @ out[i, :3, :3]
    # Force exact poses for camera intrinsic/pose alignment at real frames.
    # They now agree with the interpolation up to floating point precision.
    out[0] = poses[b_index]
    out[mid] = poses[a_index]
    out[-1] = poses[b_index]
    np.testing.assert_allclose(out[0], poses[b_index], atol=1e-8)
    np.testing.assert_allclose(out[mid], poses[a_index], atol=1e-8)
    np.testing.assert_allclose(out[-1], poses[b_index], atol=1e-8)
    return out



@contextmanager
def _gs_writer_mode(mode, output_path, expected_frames):
    """Use OpenCV to write the DA3 GS clip without a MoviePy ffmpeg subprocess.

    Only patches the active render subprocess; production DA3 remains unchanged.
    """
    if mode == "moviepy":
        yield
        return
    if mode != "opencv":
        raise ValueError(f"Unknown writer mode: {mode}")
    import cv2
    from moviepy.video.VideoClip import VideoClip

    original = VideoClip.write_videofile

    def write_opencv(clip, filename, *args, **kwargs):
        fps = kwargs.get("fps") or getattr(clip, "fps", None)
        if fps is None:
            raise ValueError("DA3 moviepy clip has no fps; cannot write GS control")
        fps = float(fps)
        if fps <= 0:
            raise ValueError(f"Invalid GS FPS: {fps}")
        # DA3 may pass export_dir (a directory) as MoviePy's filename.  In the
        # experimental OpenCV mode always use our explicit MP4 file path.
        destination = Path(output_path)
        if destination.suffix.lower() != ".mp4":
            raise ValueError(f"Expected an explicit .mp4 path: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ABLATION A] GS writer destination={destination} "
              f"(DA3 requested {filename!s})", flush=True)
        writer = None
        frame_count = 0
        dimensions = None
        try:
            # Use explicit pose indices, not MoviePy duration iteration:
            # iter_frames may include an extra terminal frame for N/fps clips.
            for frame_idx in range(expected_frames):
                rgb = np.asarray(clip.get_frame(frame_idx / fps)).astype(np.uint8)
                if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
                    raise ValueError(f"Expected RGB or RGBA video frame, got {rgb.shape}")
                if writer is None:
                    h, w = rgb.shape[:2]
                    # Most MP4 encoders require even dimensions. Pad at most 1px
                    # and keep intrinsics aligned with the original top-left origin.
                    out_w, out_h = w + (w % 2), h + (h % 2)
                    writer = cv2.VideoWriter(
                        str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (out_w, out_h))
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"OpenCV cannot open MP4 writer for {destination}; "
                            "check codec support and free disk space")
                    dimensions = (out_w, out_h)
                    print(f"[ABLATION A] GS OpenCV writer: fps={fps}, "
                          f"input={w}x{h}, output={out_w}x{out_h}", flush=True)
                if rgb.shape[:2] != (h, w):
                    raise ValueError(f"GS frame dimensions changed: {rgb.shape[:2]}")
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR if rgb.shape[2] == 4
                                   else cv2.COLOR_RGB2BGR)
                if dimensions != (w, h):
                    bgr = cv2.copyMakeBorder(bgr, 0, dimensions[1]-h,
                                              0, dimensions[0]-w,
                                              cv2.BORDER_REPLICATE)
                writer.write(bgr)
                frame_count += 1
        finally:
            if writer is not None:
                writer.release()
        if frame_count == 0 or not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"OpenCV GS encoding produced empty video: {destination}")
        reader = cv2.VideoCapture(str(destination))
        if not reader.isOpened():
            raise RuntimeError(f"Encoded GS video cannot be decoded: {destination}")
        actual = 0
        try:
            while True:
                ok, _ = reader.read()
                if not ok:
                    break
                actual += 1
        finally:
            reader.release()
        if actual != expected_frames or frame_count != expected_frames:
            raise RuntimeError(
                f"GS video written={frame_count} but decoded={actual}, {destination}")
        print(f"[ABLATION A] GS encoded {actual} frames successfully: {destination}",
              flush=True)

    VideoClip.write_videofile = write_opencv
    try:
        yield
    finally:
        VideoClip.write_videofile = original



def diagnose_da3_gaussians(prediction, poses, images, output):
    """Render ALL original DA3 input cameras directly, independent of BAB video.

    Detect whether Gaussian geometry lacks an input viewpoint before debugging
    camera interpolation or Wan. Same renderer and same metric-scale convention
    as export_to_gs_video. Saves images and a compact JSON summary.
    """
    import cv2
    import torch
    from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode

    folder = output / "gs_diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    gs = prediction.gaussians
    shapes = {}
    for name in ("means", "scales", "rotations", "opacities", "harmonics"):
        x = getattr(gs, name, None)
        shapes[name] = list(x.shape) if x is not None else None
    depth_shape = list(np.asarray(prediction.depth).shape)
    # DA3 variants may retain the full tensor shape after internally selecting
    # a subset of views. Per-view Gaussian block stats help detect empty or
    # repeated blocks that aggregate counts alone cannot expose.
    pixels_per_view = int(np.prod(depth_shape[-2:]))
    if gs.means.shape[1] == len(images) * pixels_per_view:
        for view_idx in range(len(images)):
            sl = slice(view_idx*pixels_per_view, (view_idx+1)*pixels_per_view)
            v_means = gs.means[0, sl].detach().float()
            v_opacity = gs.opacities[0, sl].detach().float()
            finite = torch.isfinite(v_means).all(dim=-1)
            print(f"[BAB GS BLOCK] view={view_idx} count={pixels_per_view} "
                  f"finite_means={finite.float().mean().item():.3f} "
                  f"mean_xyz={v_means.nan_to_num().mean(dim=0).tolist()} "
                  f"xyz_std={v_means.nan_to_num().std(dim=0).tolist()} "
                  f"opacity_mean={v_opacity.nan_to_num().mean().item():.6f} "
                  f"opacity_range=({v_opacity.nan_to_num().min().item():.4f},"
                  f"{v_opacity.nan_to_num().max().item():.4f})", flush=True)
    else:
        print(f"[BAB GS BLOCK] No contiguous per-view block assumption: "
              f"N={gs.means.shape[1]} versus {len(images)}*{pixels_per_view}", flush=True)
    print(f"[BAB GS DIAG] inputs={len(images)} depth={depth_shape} "
          f"gaussians={shapes} is_metric={prediction.is_metric} "
          f"scale_factor={prediction.scale_factor}", flush=True)

    poses = np.asarray(poses)
    Ks = np.asarray(prediction.intrinsics)
    means = gs.means
    # render input viewpoints in one scene, not 3 scene batches
    if means.shape[0] != 1:
        print(f"[BAB GS DIAG WARNING] Gaussian scene batch={means.shape[0]} (expected 1); "
              "renderer treats each batch element as a separate scene", flush=True)
    extr = torch.from_numpy(poses).to(means).unsqueeze(0)
    intr = torch.from_numpy(Ks).to(means).unsqueeze(0)
    with torch.inference_mode():
        rendered = run_renderer_in_chunk_w_trj_mode(
            gaussians=gs, extrinsics=extr, intrinsics=intr,
            image_shape=tuple(np.asarray(prediction.depth).shape[-2:]),
            chunk_size=3, trj_mode="original",
            use_sh=True, color_mode="RGB+ED", enable_tqdm=False,
        )
    # Different DA3 forks return (color, depth), (color, depth, extras),
    # or a mapping. Inspect without unpacking away additional values.
    if isinstance(rendered, (tuple, list)):
        if len(rendered) < 2:
            raise RuntimeError(f"GS renderer returned only {len(rendered)} outputs")
        color, depth = rendered[:2]
        extra_desc = [tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__
                      for x in rendered[2:]]
        print(f"[BAB GS DIAG] renderer returned {len(rendered)} items; "
              f"extras={extra_desc}", flush=True)
    elif isinstance(rendered, dict):
        color = rendered.get("color", rendered.get("colors"))
        depth = rendered.get("depth", rendered.get("depths"))
        print(f"[BAB GS DIAG] renderer dictionary keys={list(rendered)}", flush=True)
    else:
        raise TypeError(f"Unexpected GS renderer return type: {type(rendered).__name__}")
    if not isinstance(color, torch.Tensor) or not isinstance(depth, torch.Tensor):
        raise TypeError(f"Renderer color/depth must be tensors; "
                        f"got {type(color).__name__}/{type(depth).__name__}")
    print(f"[BAB GS DIAG] renderer color={tuple(color.shape)} "
          f"depth={tuple(depth.shape)}", flush=True)
    if color.ndim != 5 or color.shape[:3] != (1, len(images), 3):
        raise ValueError(f"Expected renderer color [1,{len(images)},3,H,W], "
                         f"got {tuple(color.shape)}")
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 4 or depth.shape[:2] != (1, len(images)):
        raise ValueError(f"Expected depth [1,{len(images)},H,W], "
                         f"got {tuple(depth.shape)}")
    rgb_batch = color[0].detach().clamp(0, 1).mul(255).byte().cpu().numpy()
    depth_batch = depth[0].detach().float().cpu().numpy()
    records = []
    panels = []
    for i, source in enumerate(images):
        # Render tensor format [V,3,H,W], save with the same color orientation
        # as export_to_gs_video.
        render_rgb = np.transpose(rgb_batch[i], (1, 2, 0))
        render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
        render_path = folder / f"view_{i}_gs.png"
        cv2.imwrite(str(render_path), render_bgr)

        gt_bgr = cv2.imread(str(source))
        if gt_bgr is None:
            raise RuntimeError(f"Cannot read source image {source}")
        h, w = render_bgr.shape[:2]
        gt_bgr = cv2.resize(gt_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(folder / f"view_{i}_input.png"), gt_bgr)
        absdiff = cv2.absdiff(gt_bgr, render_bgr)
        cv2.imwrite(str(folder / f"view_{i}_diff_x3.png"),
                    np.minimum(absdiff.astype(np.uint16) * 3, 255).astype(np.uint8))
        mse = float(np.mean((gt_bgr.astype(np.float32) -
                             render_bgr.astype(np.float32)) ** 2))
        psnr = float(10 * np.log10(255.0**2 / max(mse, 1e-12)))
        depth_i = np.asarray(depth_batch[i])
        finite_depth = np.isfinite(depth_i)
        positive_depth = finite_depth & (depth_i > 1e-5)
        black = (np.max(render_rgb, axis=-1) <= 5)
        metric = {
            "view": i,
            "source": str(source),
            "psnr_db": round(psnr, 3),
            "near_black_fraction": round(float(black.mean()), 5),
            "positive_depth_fraction": round(float(positive_depth.mean()), 5),
            "gs_file": str(render_path),
        }
        records.append(metric)
        print(f"[BAB GS DIAG] view={i}: psnr={psnr:.2f}dB "
              f"near_black={metric['near_black_fraction']:.1%} "
              f"valid_depth={metric['positive_depth_fraction']:.1%}", flush=True)
        scale = min(1, 480 / max(w, 1))
        tile_size = (int(w * scale), int(h * scale))
        pair = np.hstack([cv2.resize(gt_bgr, tile_size),
                          cv2.resize(render_bgr, tile_size)])
        cv2.putText(pair, f"View {i}: input | GS", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(pair)
    if panels:
        cv2.imwrite(str(folder / "comparison.jpg"), np.vstack(panels),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    data = {
        "input_count": len(images), "gaussian_shapes": shapes,
        "depth_shape": depth_shape, "is_metric": bool(prediction.is_metric),
        "scale_factor": (float(prediction.scale_factor)
                         if prediction.scale_factor is not None else None),
        "views": records,
        "caution": "Near-black fraction includes genuinely black objects and is only a proxy for missing geometry.",
    }
    (folder / "diagnostics.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[BAB GS DIAG] comparison={folder / 'comparison.jpg'}", flush=True)
    return data


def _count_decoded_frames(video_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open control video: {video_path}")
    count = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    return count


def _encode_vscode_compatible_control(control_mp4, expected_frames, codec="h264"):
    """Re-encode OpenCV mp4v to browser/VS Code-compatible H.264 yuv420p.

    Encode to a sibling temporary MP4 and replace the source ONLY after all
    frames can be decoded. Does not rerender GS or alter camera-frame indexing.
    """
    if codec == "mp4v":
        return
    if codec != "h264":
        raise ValueError(f"Unknown control codec: {codec}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or use --control-codec mp4v")
    source = Path(control_mp4)
    if not source.is_file():
        raise FileNotFoundError(source)
    temporary = source.with_name(source.stem + ".h264_tmp.mp4")
    if temporary.exists():
        temporary.unlink()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-frames:v", str(expected_frames),
        "-movflags", "+faststart", str(temporary),
    ]
    print("[ABLATION A] encoding control.mp4 as H.264/yuv420p for VS Code", flush=True)
    try:
        subprocess.run(cmd, check=True)
        actual = _count_decoded_frames(temporary)
        if actual != expected_frames:
            raise RuntimeError(f"H.264 encoded control has {actual} frames, expected {expected_frames}")
        temporary.replace(source)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[ABLATION A] H.264 control ready ({expected_frames} frames): {source}", flush=True)


def _activate_local_da3(root):
    """Import DA3_custom only, isolating its namespace from installed DA3 copies."""
    import importlib
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DA3_custom root missing: {root}")
    wrapper = root / "DA3" / "run_da3_full_crop_dev.py"
    if not wrapper.is_file():
        raise FileNotFoundError(f"Local wrapper missing: {wrapper}")

    # Looking for the implementation file is safer than requiring __init__.py:
    # depth_anything_3 can be a PEP-420 namespace package.
    package_dirs = sorted({
        p.parents[2].resolve()
        for p in root.rglob("depth_anything_3/utils/export/gs.py")
        if p.is_file() and ".git" not in p.parts and "site-packages" not in p.parts
    }, key=lambda p: (p != root / "src" / "depth_anything_3", len(p.parts), str(p)))
    if not package_dirs:
        raise FileNotFoundError(
            f"No local depth_anything_3/utils/export/gs.py under {root}; "
            "refusing pip-installed DA3 fallback")
    package_dir = package_dirs[0]
    import_root = package_dir.parent
    if package_dir.name != "depth_anything_3":
        raise RuntimeError(f"Bad DA3 package candidate: {package_dir}")
    print(f"[BAB DA3 SOURCE] local_package={package_dir}, "
          f"import_root={import_root}, wrapper={wrapper}", flush=True)

    # Imports performed before path setup cannot be repaired safely.
    bad = []
    for name, module in tuple(sys.modules.items()):
        if name == "depth_anything_3" or name.startswith("depth_anything_3."):
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(root):
                bad.append((name, str(Path(file).resolve())))
    if bad:
        raise RuntimeError(f"DA3 imports from other trees already cached: {bad[:8]}")
    # First wrapper source root, then actual implementation root.
    for folder in (root, import_root):
        if str(folder) in sys.path:
            sys.path.remove(str(folder))
        sys.path.insert(0, str(folder))
    importlib.invalidate_caches()
    import depth_anything_3

    package_file = getattr(depth_anything_3, "__file__", None)
    if package_file is not None and not Path(package_file).resolve().is_relative_to(root):
        raise RuntimeError(f"Refusing pip DA3 package at {package_file}")
    if package_file is None:
        # Namespace packages may include directories from SingleRecon or pip.
        # Restrict lookup to the explicitly selected local source tree, so
        # submodules cannot silently come from another installation.
        depth_anything_3.__path__ = [str(package_dir)]
        depth_anything_3.__spec__.submodule_search_locations = [str(package_dir)]
    resolved_dirs = [Path(p).resolve() for p in depth_anything_3.__path__]
    if resolved_dirs != [package_dir]:
        raise RuntimeError(f"Mixed DA3 package paths: {resolved_dirs}; expected {package_dir}")

    import depth_anything_3.utils.export.gs as local_gs
    actual_gs = Path(local_gs.__file__).resolve()
    if not actual_gs.is_relative_to(package_dir):
        raise RuntimeError(f"Refusing nonlocal DA3 GS module: {actual_gs}")
    import DA3.run_da3_full_crop_dev as local_wrapper
    actual_wrapper = Path(local_wrapper.__file__).resolve()
    if not actual_wrapper.is_relative_to(root):
        raise RuntimeError(f"Refusing nonlocal DA3 wrapper: {actual_wrapper}")
    print(f"[BAB DA3 SOURCE] gs={actual_gs}; wrapper={actual_wrapper}", flush=True)
    return root

def render(args, output):
    import torch
    root = _activate_local_da3(args.da3_root)
    # Order is important: local DA3 is resolved before multistage_pipeline
    # can import and cache the wrong package.
    from depth_anything_3.utils.export.gs import export_to_gs_video
    import depth_anything_3.utils.export.gs as local_gs
    from multistage_pipeline import create_da3
    from camera_trajectory import _w2c44
    gs_path = Path(local_gs.__file__).resolve()
    if not gs_path.is_relative_to(root):
        raise RuntimeError(f"GS renderer imported outside DA3_custom: {gs_path}")
    print(f"[BAB DA3 SOURCE] export_to_gs_video={gs_path}", flush=True)

    images = sorted((args.case / "reconstruction" / "da3_input").glob("*.jpg"))
    if len(images) != 3:
        raise ValueError(f"Expected three normalized DA3 JPGs at reconstruction/da3_input; got {len(images)}")
    if args.a_index == args.b_index or not (0 <= args.a_index < 3 and 0 <= args.b_index < 3):
        raise ValueError("Set distinct --a-index and --b-index in [0,2]")
    da3 = create_da3(root, Path(args.da3_checkpoint))
    da3.model.to(device=da3.device)
    started = time.perf_counter()
    prediction = da3.model.inference(
        [str(p) for p in images], infer_gs=True, use_ray_pose=True, process_res=840)
    camera_poses = _w2c44(np.asarray(prediction.extrinsics).copy())
    if prediction.is_metric and prediction.scale_factor is not None:
        camera_poses[:, :3, 3] /= float(prediction.scale_factor)
    diagnose_da3_gaussians(prediction, camera_poses, images, output)
    if args.phase == "diagnose":
        print("[BAB GS DIAG] geometry-only diagnosis finished, skipping BAB video and Wan", flush=True)
        return
    poses = plan_roundtrip(camera_poses, args.a_index, args.b_index,
                           args.pitch_deg, args.frames)
    np.save(output / "poses.npy", poses)
    # Report W2C per-frame angular changes and position steps. Large spikes at
    # A/B anchors indicate remaining discontinuities rather than GS holes.
    from scipy.spatial.transform import Rotation
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    angular = (rotations[:-1].inv() * rotations[1:]).magnitude() * (180 / np.pi)
    camera_centers = -np.einsum("nji,nj->ni", poses[:, :3, :3], poses[:, :3, 3])
    distances = np.linalg.norm(np.diff(camera_centers, axis=0), axis=1)
    print(f"[BAB POSE] angular max={angular.max():.3f}deg "
          f"p95={np.percentile(angular,95):.3f}deg, "
          f"max_step={int(np.argmax(angular))}->{int(np.argmax(angular))+1}; "
          f"translation max={distances.max():.6f} "
          f"p95={np.percentile(distances,95):.6f}", flush=True)
    mid = (args.frames - 1) // 2
    Ka = np.asarray(prediction.intrinsics[args.a_index], dtype=np.float32)
    Kb = np.asarray(prediction.intrinsics[args.b_index], dtype=np.float32)
    Ks = []
    for i in range(args.frames):
        alpha = i / mid if i <= mid else (args.frames - 1 - i) / mid
        Ks.append((1-alpha)*Kb + alpha*Ka)
    Ks = np.stack(Ks)
    render_dir = output / "gs_render"
    render_dir.mkdir(parents=True, exist_ok=True)
    control_mp4 = output / "control.mp4"
    if args.gs_writer == "opencv" and control_mp4.exists():
        control_mp4.unlink()  # never reuse an incomplete file from an earlier run
    import inspect
    export_signature = inspect.signature(export_to_gs_video)
    required = ("prediction", "export_dir", "extrinsics", "intrinsics",
                "trj_mode", "vis_depth", "output_name", "chunk_size")
    missing = [key for key in required if key not in export_signature.parameters]
    if missing and not any(p.kind == inspect.Parameter.VAR_KEYWORD
                           for p in export_signature.parameters.values()):
        raise RuntimeError(
            f"Local DA3 export_to_gs_video lacks required arguments {missing}: "
            f"{export_signature}")
    print(f"[BAB DA3 SOURCE] export_to_gs_video signature={export_signature}", flush=True)
    with _gs_writer_mode(args.gs_writer, control_mp4, args.frames):
        export_to_gs_video(
            prediction=prediction, export_dir=str(render_dir),
            extrinsics=torch.from_numpy(poses).to(prediction.gaussians.means).unsqueeze(0),
            intrinsics=torch.from_numpy(Ks).to(prediction.gaussians.means).unsqueeze(0),
            trj_mode="original", vis_depth=None, output_name="ablation_A_BAB",
            chunk_size=4)
    if args.gs_writer == "moviepy":
        src = render_dir / "gs_video" / "ablation_A_BAB.mp4"
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, control_mp4)
    if not control_mp4.is_file() or control_mp4.stat().st_size == 0:
        raise RuntimeError(f"Control video was not created: {control_mp4}")
    actual = _count_decoded_frames(control_mp4)
    if actual != args.frames:
        raise RuntimeError(f"Control must have exactly {args.frames} decoded frames, got {actual}; "
                           "rerun with --gs-writer opencv --rerender")
    _encode_vscode_compatible_control(control_mp4, args.frames, args.control_codec)
    (output / "experiment.json").write_text(json.dumps({
        "experiment": "A", "trajectory": "B->A->B",
        "input_images": [str(x) for x in images],
        "a_index": args.a_index, "b_index": args.b_index,
        "fixed_frames": {"0": "B", str(mid): "NOT_FIXED", str(args.frames-1): "B"},
        "reference_image": "A", "pitch_deg": args.pitch_deg,
        "frames": args.frames, "gs_writer": args.gs_writer,
        "control_codec": args.control_codec,
        "render_seconds": round(time.perf_counter()-started, 3),
        "note": "Frame mid pose is precisely A but its pixels are NOT in the condition.",
    }, ensure_ascii=False, indent=2))
    print(f"[ABLATION A] GS rendered to {output / 'control.mp4'}", flush=True)


def wan(args, output):
    import torch
    wan_root = Path(args.wan_root).expanduser().resolve()
    if str(wan_root) not in sys.path:
        sys.path.insert(0, str(wan_root))
    from predict_v2v_control_5b_multi_image import WanControl5B
    # Fail before the expensive SLA model load if a stale nested adapter is used.
    import inspect
    adapter_path = Path(inspect.getfile(WanControl5B)).resolve()
    adapter_params = inspect.signature(WanControl5B.__call__).parameters
    if "reference_image" not in adapter_params or (args.midpoint_keyframe and "keyframe_images" not in adapter_params):
        raise RuntimeError(
            "Wan adapter is outdated: WanControl5B.__call__ lacks reference_image. "
            f"Loaded from: {adapter_path}. Sync the branch version of "
            "predict_v2v_control_5b_multi_image.py into CodexRecon/VideoX-Fun/ "
            "and rerun with --phase wan (or omit --phase to reuse control.mp4).")
    print(f"[ABLATION {'B' if args.midpoint_keyframe else 'A'}] "
          f"verified Wan adapter: {adapter_path}", flush=True)
    spec = json.loads((output / "experiment.json").read_text())
    images = list(map(Path, spec["input_images"]))
    a, b = images[args.a_index], images[args.b_index]
    os.environ["SINGLERECON_CONTROL_TRANSFORMER_PATH"] = str(Path(args.merged_transformer).resolve())
    os.environ["SINGLERECON_CONTROL_SLA_TOPK"] = str(args.sla_topk)
    os.environ["SINGLERECON_CONTROL_INFER_STEPS"] = str(args.steps)
    os.environ["SINGLERECON_CONTROL_LORA_PATH"] = ""
    os.environ["SINGLERECON_CONTROL_CFG"] = "0.0"
    os.environ["SINGLERECON_CONTROL_PROMPT_EMBEDS_PATH"] = ""  # explicitly restore T5 mode
    os.environ["SINGLERECON_CONTROL_SHORT_EDGE"] = str(args.short_edge)
    os.environ["SINGLERECON_CONTROL_USE_LIGHTVAE"] = "1"
    os.environ["SINGLERECON_CONTROL_LIGHTVAE_PATH"] = str(Path(args.lightvae))
    os.environ["SINGLERECON_CONTROL_TEACHER_VAE_PATH"] = str(Path(args.teacher_vae))
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    # Reuse the established empty-prompt T5 path for a clean ablation.
    # Do not silently inherit a prompt.pt experiment from the shell.
    generator = WanControl5B()
    middle = (args.frames - 1)//2
    samples = sorted(set(np.linspace(0, args.frames - 1, 9).round().astype(int).tolist()
                         + [middle]))
    # B mode uses the EXACT same control as A mode; only the inpainting
    # condition differs. Keep independent outputs for fair A/B comparison.
    is_b = args.midpoint_keyframe
    generated_name = "generated_midA.mp4" if is_b else "generated.mp4"
    sampled_name = "sampled_frames_midA" if is_b else "sampled_frames"
    keyframes = ({0: str(b), middle: str(a), args.frames - 1: str(b)}
                 if is_b else None)
    print(f"[ABLATION {'B' if is_b else 'A'}] control={output / 'control.mp4'} "
          f"keyframes={keyframes}; generated={generated_name}", flush=True)
    started = time.perf_counter()
    generator(
        control_video=str(output / "control.mp4"),
        start_image=str(b),
        end_image=str(b),
        reference_image=str(a),
        keyframe_images=keyframes,
        output_video=str(output / generated_name),
        prompt="", num_frames=args.frames, fps=30,
        sampled_frame_indices=samples,
        sampled_frames_dir=str(output / sampled_name), save_mp4=True)
    elapsed = time.perf_counter() - started
    # Evaluate whether reference A is actually reproduced at the unconstrained midpoint.
    import cv2
    generated_mid = output / sampled_name / f"f{middle:04d}.jpg"
    observed = cv2.imread(str(generated_mid))
    reference = cv2.imread(str(a))
    if observed is None or reference is None:
        raise RuntimeError(f"Cannot compare midpoint {generated_mid} with A {a}")
    reference = cv2.resize(reference, (observed.shape[1], observed.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    diff = (observed.astype(np.float32) - reference.astype(np.float32))
    mse = float(np.mean(diff ** 2))
    psnr = float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))
    metrics = {
        "comparison": "generated midpoint versus actual A (same output resolution)",
        "midpoint_frame": middle,
        "midpoint_psnr_db": round(psnr, 4),
        "midpoint_mse": round(mse, 4),
        "wan_wall_seconds": round(elapsed, 3),
        "wan_timing": generator.last_timing,
        "warning": "This is a diagnostic pixel metric, not a geometry/perceptual quality score.",
    }
    (output / ("comparison_midA.json" if is_b else "comparison.json")).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if is_b:
        (output / "experiment_midA.json").write_text(
            json.dumps({
                "experiment": "B", "trajectory": "B->A->B",
                "control_video": str(output / "control.mp4"),
                "output_video": str(output / generated_name),
                "fixed_keyframes": {"0": str(b), str(middle): str(a),
                                    str(args.frames - 1): str(b)},
                "reference_image": str(a),
                "note": "Middle A provided as sparse inpaint condition; VAE conditioning "
                        "does not guarantee pixel-exact reproduction in decoded output.",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    suffix = "_midA" if is_b else ""
    cv2.imwrite(str(output / f"midpoint_A_reference{suffix}.jpg"), reference)
    cv2.imwrite(str(output / f"midpoint_generated{suffix}.jpg"), observed)
    cv2.imwrite(str(output / f"midpoint_difference{suffix}.jpg"),
                np.clip(np.abs(diff) * 3, 0, 255).astype(np.uint8))
    print(f"[ABLATION {'B' if is_b else 'A'}] midpoint PSNR to real A={psnr:.3f}dB; "
          f"generated={generated_mid}", flush=True)
    print(f"[ABLATION {'B' if is_b else 'A'}] complete; timing={generator.last_timing}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, default=CASE_DEFAULT)
    p.add_argument("--gpu", type=int, default=2, help="Physical GPU for this single-GPU experiment")
    p.add_argument("--a-index", type=int, default=1, help="Index within sorted da3_input JPGs (default 1)")
    p.add_argument("--b-index", type=int, default=0, help="Index within sorted da3_input JPGs (default 0)")
    p.add_argument("--pitch-deg", type=float, default=10.0)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--cpu-threads", type=int, default=16)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--sla-topk", type=float, default=0.5)
    p.add_argument("--short-edge", type=int, default=1088)
    p.add_argument("--da3-root", default=str(HERE / "DA3_custom"))
    p.add_argument("--da3-checkpoint", default=str(HERE / "Cloud_Models/Saliency-Face-DA3/besthumanmodel_lr1e_6_ckpt2.pt"))
    p.add_argument("--wan-root", default=str(HERE / "VideoX-Fun"))
    p.add_argument("--merged-transformer", default=str(HERE / "Cloud_Models/transformer_w_skybox_sla/transformer_lora_merge_weight_sla_quant.pt"))
    p.add_argument("--lightvae", default=str(HERE / "Cloud_Models/transformer_w_skybox_sla/MG-LightVAE.pth"))
    p.add_argument("--teacher-vae", default=str(HERE / "Cloud_Models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"))
    p.add_argument("--midpoint-keyframe", action="store_true",
                   help="B ablation: constrain true A at frame midpoint; reuse existing control.mp4")
    p.add_argument("--rerender", action="store_true")
    p.add_argument("--control-codec", choices=("h264", "mp4v"), default="h264",
                   help="Final control.mp4 encoding; H.264/yuv420p plays in VS Code")
    p.add_argument("--gs-writer", choices=("opencv", "moviepy"), default="opencv",
                   help="Writer for DA3 GS control; OpenCV avoids ffmpeg subprocess BrokenPipe")
    p.add_argument("--phase", choices=("all","render","wan","diagnose","transcode"), default="all",
                   help="all/render/wan/diagnose/transcode; transcode only converts existing control.mp4")
    args = p.parse_args()
    args.case = args.case.expanduser().resolve()
    if args.frames < 5 or (args.frames - 1) % 4:
        p.error("--frames must be 4n+1, e.g. 49 or 81")
    if args.phase == "wan" and args.rerender:
        p.error("--rerender cannot be combined with --phase wan")
    output = args.case / "reconstruction" / "ablation_A_BAB"
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "all":
        # Separate DA3 and Wan CUDA processes so neither model consumes GPU
        # memory during the other's execution.
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            env[k] = str(args.cpu_threads)
        base = [sys.executable, str(Path(__file__).resolve())]
        provided = sys.argv[1:]
        if "--phase" in provided:
            raise ValueError("Cannot pass --phase inside automatic phase management")
        if args.rerender or not (output / "control.mp4").is_file():
            subprocess.run(base + provided + ["--phase", "render"], env=env, check=True)
        else:
            print("[ABLATION A] reuse existing control.mp4 (use --rerender to regenerate)", flush=True)
        # --rerender applies only to the GS render subprocess, never the Wan
        # subprocess. Passing it through trips the direct-Wan safety check.
        wan_args = [arg for arg in provided if arg != "--rerender"]
        subprocess.run(base + wan_args + ["--phase", "wan"], env=env, check=True)
    elif args.phase in ("render", "diagnose"):
        render(args, output)
    elif args.phase == "transcode":
        _encode_vscode_compatible_control(output / "control.mp4", args.frames, args.control_codec)
    else:
        if not (output / "control.mp4").is_file():
            raise FileNotFoundError(output / "control.mp4")
        wan(args, output)


if __name__ == "__main__":
    main()
