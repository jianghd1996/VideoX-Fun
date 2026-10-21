# End-to-end benchmark: parallel video, single-GPU DA3

Use feat/da3-inference-performance-20261021 in both repositories.
DA3 internals are unchanged. Both ranks generate the same video using Wan
sequence parallelism. Rank 0 samples frames, runs DA3, exports GS/COLMAP and
optionally runs dense reconstruction. Rank 1 waits before the next case.
The launcher also supports one GPU for the baseline. Resume is disabled so
benchmarks actually rerun every stage. Existing generated outputs are overwritten;
use separate copies of the same case when comparing outputs.

Use your existing SparseRecon JSON with correct video_code_root,
da3_model_path, dense_code_root, LoRA and LightVAE paths. video_code_root must
point to the updated VideoX-Fun checkout (includes the process-group reuse fix).
Set run_gen_video=true, run_DA3=true, run_captioner=false and provide prompt.txt.
run_3dgs_iter=false measures video + DA3 only; true includes your dense backend.

If you currently edit DEFAULT_ARGS instead of using JSON, export your settings:

```bash
cd /PATH/TO/YOUR/SingleRecon
python - <<'PY'
import json
from run_cases import DEFAULT_ARGS
cfg = dict(DEFAULT_ARGS)
cfg.update(run_gen_video=True, run_DA3=True, run_captioner=False)
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

## Editable standalone launcher

Edit the settings at the top of `scripts/reconstruction/run_e2e.py`, then:

```bash
python scripts/reconstruction/run_e2e.py
```

Defaults: GPU_IDS="4,5", CPU_THREADS=32, WORK_DIR=/home/z00566689/dev/mnt/jiang_test/00_1021_test,
DA3_ROOT=/mnt/DataPart/jianghongda/Depth-Anything-3. Verify your actual paths.
Set CONFIG_PATH to your working JSON or BASE_RUN_CASES to the existing
SingleRecon/run_cases.py containing your edited DEFAULT_ARGS. The latter is
used by default, pointing to the sibling run_cases.py. Its model/dense paths
are inherited, while video_code_root is set to the current checkout. Prompt
captioning is disabled (provide prompt.txt); RUN_DENSE toggles the dense stage.
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
