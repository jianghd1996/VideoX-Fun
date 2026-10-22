# Reconstruction pipeline

This directory implements the existing single-image reconstruction workflow. Multi-image reconstruction is planned but is **not** implemented here yet.

## Current files and entrypoints

| File | Purpose |
| --- | --- |
| `pipeline_reconstruction.py` | `SparseRecon` public API: preloading, case runs, stage journal and timing |
| `multistage_pipeline.py` | Single-image stages: video generation, fixed-frame sampling, DA3, dense reconstruction |
| `da3_official_adapter.py` | Experimental persistent official DA3 adapter (COLMAP output only) |
| `run_cases.py` | Configurable sequential case runner; preloads models once |
| `run_e2e.py` | Editable single-/two-GPU end-to-end launcher, delegates to torchrun |
| `run_end_to_end_multi_gpu.py` | torchrun worker: Wan sequence parallelism, rank-0 DA3/dense reconstruction |

For ordinary single-GPU/batch reconstruction, edit `DEFAULT_ARGS` and `WORK_DIR` in `run_cases.py`, then run:

```bash
python scripts/reconstruction/run_cases.py
```

For editable single-/two-GPU E2E runs, edit `run_e2e.py`, then run:

```bash
python scripts/reconstruction/run_e2e.py
```

`run_batch.py` (legacy process-per-case launcher) and `run_case_multi_gpu.py` (video-only latency benchmark) were removed during cleanup. Their historic implementations remain accessible in git history. Use the supported entrypoints above instead.

---

## Experimental official DA3 performance path

The existing reconstruction backend remains `legacy` by default. This branch
adds an **experimental**, persistent Depth-Anything-3 backend that runs directly
from the separately maintained DA3 repository.

## Checkout

```bash
cd /mnt/DataPart/jianghongda/VideoX-Fun
git fetch origin
git switch chore/reconstruction-cleanup-20261022

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

---

## End-to-end benchmark: parallel video, single-GPU DA3

Use the cleanup branch (or a descendant containing it) in VideoX-Fun, and feat/da3-inference-performance-20261021 in the companion DA3 repository.
DA3 internals are unchanged. Both ranks generate the same video using Wan
sequence parallelism. Rank 0 samples frames, runs DA3, exports GS/COLMAP and
optionally runs dense reconstruction. Rank 1 waits before the next case.
The launcher also supports one GPU for the baseline. Resume is disabled so
benchmarks actually rerun every stage. Existing generated outputs are overwritten;
use separate copies of the same case when comparing outputs.

Use your existing SparseRecon JSON with correct video_code_root,
da3_model_path, dense_code_root, LoRA and LightVAE paths. video_code_root must
point to the updated VideoX-Fun checkout (includes the process-group reuse fix).
Set run_gen_video=true, run_DA3=true, run_captioner=true. Configure your existing
captioner_code_root and captioner_model_path when they differ from the defaults.
run_3dgs_iter=false measures video + DA3 only; true includes your dense backend.

If you currently edit DEFAULT_ARGS instead of using JSON, export your settings:

```bash
cd /PATH/TO/YOUR/SingleRecon
python - <<'PY'
import json
from run_cases import DEFAULT_ARGS
cfg = dict(DEFAULT_ARGS)
cfg.update(run_gen_video=True, run_DA3=True, run_captioner=True, caption_force_regenerate=True)
with open('e2e_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
PY
```

From the updated VideoX-Fun checkout (replace config/case/DA3 paths):

```bash
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
  scripts/reconstruction/run_end_to_end_multi_gpu.py \
  --config /PATH/TO/SingleRecon/e2e_config.json \
  --work-dir /PATH/TO/single_gpu_cases \
  --da3-root /PATH/TO/Depth-Anything-3

CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 \
  scripts/reconstruction/run_end_to_end_multi_gpu.py \
  --config /PATH/TO/SingleRecon/e2e_config.json \
  --work-dir /PATH/TO/two_gpu_cases \
  --da3-root /PATH/TO/Depth-Anything-3
```

CPU thread pools are explicitly limited before numerical-library imports:
Torch intra-op, OpenMP, MKL, OpenBLAS, NumExpr, BLIS and OpenCV default to 32
threads per rank; Torch inter-op is one. Override with --cpu-threads 16 if needed.
The launcher prints the configured limit for every rank. This is independent of
DA3 image-preprocessing worker counts and does not disable GPU parallelism.

Reports: work-dir/end_to_end_1gpu.json and end_to_end_2gpu.json;
per-case reconstruction/timings.json and stage_status.json retain stage details.
Model preload is excluded. Compare repeated runs after the first kernel warmup.

Validation: Python compilation and CLI help passed. A mocked two-rank
orchestration test checks frame-sampling/DA3/dense barriers and primary-only
DA3 execution. No GPU, NCCL, model weights or complete external dense backend
were available on the build host; full inference/performance needs local testing.

### Editable standalone launcher

Edit the settings at the top of `scripts/reconstruction/run_e2e.py`, then:

```bash
python scripts/reconstruction/run_e2e.py
```

Defaults: GPU_IDS="4,5", CPU_THREADS=32, WORK_DIR=/home/z00566689/dev/mnt/jiang_test/00_1021_test,
DA3_ROOT=/home/z00566689/dev/mnt/jiang_dev/SingleRecon/DA3_custom. Verify your actual paths.
Set CONFIG_PATH to your working JSON or BASE_RUN_CASES to the existing
SingleRecon/run_cases.py containing your edited DEFAULT_ARGS. The latter is
used by default, pointing to the sibling run_cases.py. Its model/dense paths
are inherited, while video_code_root is validated before torchrun starts. VIDEO_ROOT=None first
uses config.video_code_root, then checks a sibling VideoX-Fun directory and
enclosing checkouts. An explicit VIDEO_ROOT overrides these choices. Both
predict_ti2v.py and videox_fun/ must exist in the chosen directory. RUN_CAPTIONER=True enables rank-0 caption inference; RUN_DENSE toggles the dense stage.
DRY_RUN=True prints the generated config and command without inference.
Change GPU_IDS="4" for the single-GPU baseline; use copies of the same cases
if both generated outputs should be retained.

Thread default follows the historical experiment shared by the user:
https://chatgpt.com/share/6ad88d69-c734-83ec-b881-a970965789bd

| CPU threads per rank | Historical Total Infer (s) |
| --- | ---: |
| 2 | 71.056 |
| 4 | 67.042 |
| 8 | 63.149 |
| 16 | 61.438 |
| 32 | 60.629 |

These are previous video-inference measurements, not new end-to-end DA3 results.
32 was the fastest measured configuration; the small 16-to-32 difference needs
repeat runs to distinguish load/encoding variation. The previous measurements
set OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS. This launcher also
controls the other pools listed above; it has not yet been GPU-benchmarked.

When copying run_e2e.py into SingleRecon, the sibling layout is supported:
SingleRecon/run_e2e.py and SingleRecon/VideoX-Fun/predict_ti2v.py. Keep
VIDEO_ROOT=None for automatic discovery, or set it explicitly:

```python
VIDEO_ROOT = Path(__file__).resolve().parent / "VideoX-Fun"
```

The previous parents[2] assumption only applied inside the repository and was
removed because it selected an unrelated ancestor when the script was copied.

### Caption performance measurement

The editable launcher defaults to RUN_CAPTIONER=True and
CAPTION_FORCE_REGENERATE=True, so repeated benchmarks include real caption
inference even if prompt.txt already exists. The caption is overwritten by the
new generated text. Set CAPTION_FORCE_REGENERATE=False to reuse existing prompts
and generate only missing/empty prompts during ordinary batch processing.
Rank 0 uses one GPU for captioning; rank 1 does not load the captioner and waits
at the video-stage barrier. Caption device transfers and inference are measured
by the caption stage, which is included in total and end_to_end_wall. Model
construction/preload remains excluded, as for the video and DA3 backends.
Update all four Python files together when copying them into SingleRecon:
run_e2e.py, run_end_to_end_multi_gpu.py, pipeline_reconstruction.py,
and multistage_pipeline.py.

Checks include actual caption cache/regeneration/missing-prompt behavior and
SparseRecon timing output, plus mocked one/two-rank orchestration with new cases.
Full Qwen captioner inference and GPU-memory behavior require local validation.
