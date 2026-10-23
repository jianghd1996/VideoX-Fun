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
        color, depth = run_renderer_in_chunk_w_trj_mode(
            gaussians=gs, extrinsics=extr, intrinsics=intr,
            image_shape=tuple(np.asarray(prediction.depth).shape[-2:]),
            chunk_size=3, trj_mode="original",
            use_sh=True, color_mode="RGB+ED", enable_tqdm=False,
        )
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


def render(args, output):
    import torch
    from multistage_pipeline import create_da3
    from depth_anything_3.utils.export.gs import export_to_gs_video

    root = Path(args.da3_root).expanduser().resolve()
    for path in (root / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from camera_trajectory import _w2c44

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
    with _gs_writer_mode(args.gs_writer, control_mp4, args.frames):
        export_to_gs_video(
            prediction=prediction, export_dir=str(render_dir),
            extrinsics=torch.from_numpy(poses).to(prediction.gaussians.means).unsqueeze(0),
            intrinsics=torch.from_numpy(Ks).to(prediction.gaussians.means).unsqueeze(0),
            trj_mode="original", vis_depth=None, output_name="ablation_A_BAB",
            crop_ratio=1.0, view_num=args.frames, extra_view=[False, False],
            chunk_size=4)
    if args.gs_writer == "moviepy":
        src = render_dir / "gs_video" / "ablation_A_BAB.mp4"
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, control_mp4)
    if not control_mp4.is_file() or control_mp4.stat().st_size == 0:
        raise RuntimeError(f"Control video was not created: {control_mp4}")
    import cv2
    reader = cv2.VideoCapture(str(control_mp4))
    actual = 0
    try:
        while True:
            ok, _ = reader.read()
            if not ok:
                break
            actual += 1
    finally:
        reader.release()
    if actual != args.frames:
        raise RuntimeError(f"Control must have exactly {args.frames} decoded frames, got {actual}; "
                           "rerun with --gs-writer opencv --rerender")
    (output / "experiment.json").write_text(json.dumps({
        "experiment": "A", "trajectory": "B->A->B",
        "input_images": [str(x) for x in images],
        "a_index": args.a_index, "b_index": args.b_index,
        "fixed_frames": {"0": "B", str(mid): "NOT_FIXED", str(args.frames-1): "B"},
        "reference_image": "A", "pitch_deg": args.pitch_deg,
        "frames": args.frames, "gs_writer": args.gs_writer,
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
    if "reference_image" not in inspect.signature(WanControl5B.__call__).parameters:
        raise RuntimeError(
            "Wan adapter is outdated: WanControl5B.__call__ lacks reference_image. "
            f"Loaded from: {adapter_path}. Sync the branch version of "
            "predict_v2v_control_5b_multi_image.py into CodexRecon/VideoX-Fun/ "
            "and rerun with --phase wan (or omit --phase to reuse control.mp4).")
    print(f"[ABLATION A] verified reference_image adapter: {adapter_path}", flush=True)
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
    started = time.perf_counter()
    generator(
        control_video=str(output / "control.mp4"),
        start_image=str(b),
        end_image=str(b),
        reference_image=str(a),
        output_video=str(output / "generated.mp4"),
        prompt="", num_frames=args.frames, fps=30,
        sampled_frame_indices=samples,
        sampled_frames_dir=str(output / "sampled_frames"), save_mp4=True)
    elapsed = time.perf_counter() - started
    # Evaluate whether reference A is actually reproduced at the unconstrained midpoint.
    import cv2
    generated_mid = output / "sampled_frames" / f"f{middle:04d}.jpg"
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
    (output / "comparison.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    cv2.imwrite(str(output / "midpoint_A_reference.jpg"), reference)
    cv2.imwrite(str(output / "midpoint_generated.jpg"), observed)
    cv2.imwrite(str(output / "midpoint_difference.jpg"),
                np.clip(np.abs(diff) * 3, 0, 255).astype(np.uint8))
    print(f"[ABLATION A] midpoint PSNR to real A={psnr:.3f}dB; "
          f"generated={generated_mid}", flush=True)
    print(f"[ABLATION A] complete; timing={generator.last_timing}", flush=True)


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
    p.add_argument("--rerender", action="store_true")
    p.add_argument("--gs-writer", choices=("opencv", "moviepy"), default="opencv",
                   help="Writer for DA3 GS control; OpenCV avoids ffmpeg subprocess BrokenPipe")
    p.add_argument("--phase", choices=("all","render","wan","diagnose"), default="all",
                   help="all/render/wan/diagnose; diagnose runs DA3 and renders 3 original GS views only")
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
    else:
        if not (output / "control.mp4").is_file():
            raise FileNotFoundError(output / "control.mp4")
        wan(args, output)


if __name__ == "__main__":
    main()
