#!/usr/bin/env python3
"""Compare planned multi-image COLMAP cameras with DA3 re-estimation on identical 83 images.

Only the original input cameras (normally 3) determine a Sim(3) B->A mapping.
Generated frames are NEVER included in alignment, and form the held-out evaluation.
Requires no GPU and does not modify either reconstruction.

Example:
 python compare_multi_image_poses.py \
   --original /home/z00566689/dev/mnt/jiang_test/00_1022_test/cover2 \
   --compare /home/z00566689/dev/mnt/jiang_test/00_1022_test/cover2_compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .multi_image_dense_input import _read_cameras, _read_images, _camera_fx_fy_cx_cy
except ImportError:
    from multi_image_dense_input import _read_cameras, _read_images, _camera_fx_fy_cx_cy


def _record_to_camera(record, camera_models):
    q, t, cid, name, _ = record
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() # world -> camera
    c = -r.T @ np.asarray(t, dtype=float)
    fx, fy, cx, cy, width, height = _camera_fx_fy_cx_cy(camera_models[cid])
    return {"center": c, "w2c_rotation": r, "c2w_rotation": r.T,
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "width": int(width), "height": int(height), "image": name}


def _umeyama(b, a):
    """Fit A ~= scale * Q @ B + shift; proper rotation, no reflection."""
    if len(a) < 3:
        raise ValueError("Sim(3) requires >=3 distinct original images")
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    aa, bb = a - mu_a, b - mu_b
    var_b = np.sum(bb ** 2) / len(b)
    if var_b < 1e-12:
        raise ValueError("DA3 input camera centers collapsed; cannot align scale")
    cross = aa.T @ bb / len(b)
    u, singular, vt = np.linalg.svd(cross)
    diag = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        diag[2, 2] = -1
    q = u @ diag @ vt
    scale = float(np.sum(singular * np.diag(diag)) / var_b)
    if scale <= 0:
        raise ValueError("Sim(3) fitted invalid scale")
    shift = mu_a - scale * q @ mu_b
    rank = np.linalg.matrix_rank(bb, tol=max(np.linalg.norm(bb), 1) * 1e-8)
    return scale, q, shift, rank


def _match_sources(b_images, source_manifest, a_names):
    """Map B COLMAP names to original filenames, never assume their image_id matches."""
    b_names = {rec[3] for rec in b_images.values()}
    mapping = {}
    if source_manifest.is_file():
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        # The comparison runner produces 000000.jpg ... by sorted source order.
        for i, entry in enumerate(manifest):
            generated_name = f"{i:06d}.jpg"
            input_name = Path(entry["path"]).name
            if generated_name not in b_names:
                raise ValueError(f"B COLMAP missing {generated_name} from source_images_manifest.json")
            if input_name not in a_names:
                raise ValueError(f"B input file {input_name} is not present in A COLMAP. "
                                 "Use the identical 83 image files, preserving their filenames.")
            if input_name in mapping:
                raise ValueError(f"Duplicate original image filename in compare input: {input_name}")
            mapping[input_name] = generated_name
        extra = b_names - set(mapping.values())
        if extra:
            raise ValueError(f"B COLMAP includes {len(extra)} names not in source manifest: {sorted(extra)[:5]}")
    else:
        if b_names != a_names:
            raise ValueError(
                "Cannot establish B image identities: source_images_manifest.json missing "
                "and COLMAP image names differ. Do not match by image_id/order.")
        mapping = {name: name for name in a_names}
    missing = a_names - set(mapping)
    if missing:
        raise ValueError(f"Missing {len(missing)} images in B: {sorted(missing)[:8]}")
    return mapping


def _summary(rows):
    if not rows:
        return None
    def stats(field):
        values = np.asarray([row[field] for row in rows], dtype=float)
        return dict(mean=float(values.mean()), median=float(np.median(values)),
                    p90=float(np.percentile(values, 90)), max=float(values.max()))
    return {"count": len(rows),
            "center_error": stats("center_error"),
            "center_error_relative_to_baseline": stats("center_error_relative"),
            "rotation_error_degrees": stats("rotation_error_deg"),
            "relative_fx_difference": stats("fx_relative_difference"),
            "relative_fy_difference": stats("fy_relative_difference")}


def _plot(rows, directory):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[POSE COMPARE] matplotlib unavailable; saved JSON only", flush=True)
        return []
    by_trajectory = defaultdict(list)
    for row in rows:
        if row["type"] == "generated":
            by_trajectory[row["trajectory"]].append(row)
    if not by_trajectory:
        return []
    result = []
    for field, ylabel, filename in (
        ("center_error_relative", "Camera-center error / original baseline", "position_error.png"),
        ("rotation_error_deg", "Rotation difference (degrees)", "rotation_error.png"),
        ("fx_relative_difference", "Relative focal-length difference (fx)", "focal_error.png"),
    ):
        fig, ax = plt.subplots(figsize=(11, 5))
        for name, values in sorted(by_trajectory.items()):
            values.sort(key=lambda r: r["frame"])
            ax.plot([v["frame"] for v in values], [v[field] for v in values],
                    marker=".", markersize=4, label=name)
        ax.set_xlabel("Generated video frame index")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        dest = directory / filename
        fig.savefig(dest, dpi=160)
        plt.close(fig)
        result.append(str(dest))
    return result


def compare(original, compare_dir, output):
    a_root, b_root = original / "reconstruction", compare_dir / "reconstruction"
    a_sparse = a_root / "dense_input" / "sparse" / "0"
    b_sparse = b_root / "sparse" / "0"
    a_cameras = _read_cameras(a_sparse)
    a_images = _read_images(a_sparse)
    b_cameras = _read_cameras(b_sparse)
    b_images = _read_images(b_sparse)
    a_by_name = {rec[3]: _record_to_camera(rec, a_cameras) for rec in a_images.values()}
    b_by_name = {rec[3]: _record_to_camera(rec, b_cameras) for rec in b_images.values()}
    matching = _match_sources(b_images, b_root / "source_images_manifest.json", set(a_by_name))
    dense_manifest = json.loads((a_root / "dense_input" / "input_manifest.json").read_text(encoding="utf-8"))
    generated = {entry["image"]: entry for entry in dense_manifest["generated"]}
    anchors = sorted(set(a_by_name) - set(generated))
    if len(anchors) != int(dense_manifest["input_views"]):
        raise ValueError(f"Expected {dense_manifest['input_views']} anchors; got {len(anchors)}")
    a_anchor_centers = np.array([a_by_name[name]["center"] for name in anchors])
    b_anchor_centers = np.array([b_by_name[matching[name]]["center"] for name in anchors])
    scale, q, shift, rank = _umeyama(b_anchor_centers, a_anchor_centers)
    pairwise = np.linalg.norm(a_anchor_centers[:, None] - a_anchor_centers[None], axis=-1)
    baseline = max(float(pairwise.max()), 1e-12)
    rows = []
    for name, a in sorted(a_by_name.items()):
        b = b_by_name[matching[name]]
        center_aligned = scale * (q @ b["center"]) + shift
        center_error = float(np.linalg.norm(center_aligned - a["center"]))
        # Map B camera->world orientation into A world coordinates.
        b_c2w_aligned = q @ b["c2w_rotation"]
        rotation_error = float(np.degrees(Rotation.from_matrix(
            a["c2w_rotation"].T @ b_c2w_aligned).magnitude()))
        # Intrinsic ratio corrected for resolution: dimensionless fx / width.
        fx_a, fx_b = a["fx"] / a["width"], b["fx"] / b["width"]
        fy_a, fy_b = a["fy"] / a["height"], b["fy"] / b["height"]
        entry = generated.get(name)
        rows.append({
            "name": name, "b_colmap_name": matching[name],
            "type": "generated" if entry else "original",
            "trajectory": entry["trajectory"] if entry else "source",
            "frame": int(entry["frame"]) if entry else None,
            "center_error": center_error, "center_error_relative": center_error / baseline,
            "rotation_error_deg": rotation_error,
            "fx_relative_difference": float(abs(fx_b / fx_a - 1)) if fx_a else float("nan"),
            "fy_relative_difference": float(abs(fy_b / fy_a - 1)) if fy_a else float("nan"),
            "fx_over_width_A": fx_a, "fx_over_width_B": fx_b,
            "fy_over_height_A": fy_a, "fy_over_height_B": fy_b,
            "A_center": a["center"].tolist(), "B_center_aligned": center_aligned.tolist(),
        })
    output.mkdir(parents=True, exist_ok=True)
    originals = [r for r in rows if r["type"] == "original"]
    generated_rows = [r for r in rows if r["type"] == "generated"]
    by_trajectory = {key: _summary([r for r in generated_rows if r["trajectory"] == key])
                     for key in sorted({r["trajectory"] for r in generated_rows})}
    report = {
        "reference": "A = known planned poses, B = 83-image DA3 re-estimate",
        "alignment": {
            "fit_uses": "original input views only",
            "anchors": anchors, "center_rank": int(rank),
            "scale_B_to_A": scale, "rotation_B_to_A": q.tolist(),
            "translation_B_to_A": shift.tolist(),
            "original_baseline_A": baseline,
            "warning": ("Three original centers are collinear; global rotation around "
                        "the anchor baseline is weakly constrained." if rank < 2 else None),
        },
        "original": _summary(originals), "generated": _summary(generated_rows),
        "by_trajectory": by_trajectory, "per_image": rows,
        "caution": ("B DA3 camera estimates are NOT ground truth; discrepancy alone does not "
                    "prove A is wrong. A/B additionally differ in DA3 initial geometry.")
    }
    report_file = output / "pose_comparison.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plots = _plot(rows, output)
    print(f"[POSE COMPARE] A={len(a_by_name)} B={len(b_by_name)} anchors={len(anchors)} "
          f"generated={len(generated_rows)}", flush=True)
    print(f"[SIM3] scale(B->A)={scale:.6f}, anchor_baseline={baseline:.6f}, rank={rank}", flush=True)
    for label, summary in [("anchors", _summary(originals)), ("generated", _summary(generated_rows))]:
        if summary:
            print(f"[{label}] center/baseline mean={summary['center_error_relative_to_baseline']['mean']:.6f}, "
                  f"median={summary['center_error_relative_to_baseline']['median']:.6f}; "
                  f"rotation mean={summary['rotation_error_degrees']['mean']:.3f} deg", flush=True)
    for name, summary in by_trajectory.items():
        print(f"  {name}: count={summary['count']}, "
              f"center/baseline median={summary['center_error_relative_to_baseline']['median']:.5f}, "
              f"rotation median={summary['rotation_error_degrees']['median']:.3f} deg, "
              f"fx diff median={summary['relative_fx_difference']['median']:.4f}", flush=True)
    print(f"[POSE COMPARE] report={report_file}", flush=True)
    for plot in plots:
        print(f"[POSE COMPARE] plot={plot}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path(
        "/home/z00566689/dev/mnt/jiang_test/00_1022_test/cover2"))
    parser.add_argument("--compare", type=Path, default=Path(
        "/home/z00566689/dev/mnt/jiang_test/00_1022_test/cover2_compare"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    original = args.original.expanduser().resolve()
    comparison = args.compare.expanduser().resolve()
    result_dir = args.output.expanduser().resolve() if args.output else comparison / "pose_comparison"
    compare(original, comparison, result_dir)


if __name__ == "__main__":
    main()
