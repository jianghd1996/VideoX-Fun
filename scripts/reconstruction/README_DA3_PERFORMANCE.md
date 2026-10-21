# DA3 opt-in performance path

The existing reconstruction backend remains `legacy` by default. This branch
adds an **experimental**, persistent Depth-Anything-3 backend that runs directly
from the separately maintained DA3 repository.

## Checkout

```bash
cd /mnt/DataPart/jianghongda/VideoX-Fun
git fetch origin
git switch feat/da3-inference-performance-20261021

cd /mnt/DataPart/jianghongda/Depth-Anything-3
git fetch origin
git switch feat/da3-inference-performance-20261021
pip install -e .
```

## SparseRecon JSON override

```json
{
  "da3_backend": "official",
  "da3_repo_root": "/mnt/DataPart/jianghongda/Depth-Anything-3",
  "da3_model_path": "/ABSOLUTE/PATH/TO/OFFICIAL/DA3/CHECKPOINT",
  "da3_process_res": 504,
  "da3_process_res_method": "upper_bound_resize",
  "da3_preprocess_workers": 8,
  "da3_profile": true,
  "crop_ratio": 1.0,
  "use_view_extrapolation": false,
  "is_livephoto": false
}
```

Use these keys in your existing JSON passed to `SparseRecon` instead of
replacing your other video/dense reconstruction settings.

**Important:** the old custom `DA3.run_da3_full_crop_dev` is a different
backend with a separate checkpoint and `extra_view` behavior. Do **not** assume
an old custom model checkpoint works with the official `DepthAnything3`
loader. The official adapter **does not yet reproduce** the custom GS-render
video, cropping/extrapolation, or generated-frame-index outputs. It exports
COLMAP, but is not yet feature-compatible with all legacy pipeline modes.
Do not use this experimental backend when downstream expects `gs_render.mp4`
or `gen_index` from legacy DA3.

Logs report `preprocess`, `transfer`, `forward`, `postprocess`,
`export`, and `total`. Per-case timings are written to
`reconstruction/da3_timings.json`; regular pipeline times remain in
`reconstruction/timings.json`.

## Comparison

Run the same 40 sampled JPGs with the same official checkpoint, resolution,
and image order before comparing performance or output. First execution
includes CUDA warmup, so compare repeated steady-state runs; record
wall-clock total and peak VRAM. Check COLMAP view count, camera convention,
intrinsics, depth distribution, and pose differences before enabling in production.

Revert simply by setting `da3_backend: legacy` (default).
