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

Reports: work-dir/end_to_end_1gpu.json and end_to_end_2gpu.json;
per-case reconstruction/timings.json and stage_status.json retain stage details.
Model preload is excluded. Compare repeated runs after the first kernel warmup.

Validation: Python compilation and CLI help passed. A mocked two-rank
orchestration test checks frame-sampling/DA3/dense barriers and primary-only
DA3 execution. No GPU, NCCL, model weights or complete external dense backend
were available on the build host; full inference/performance needs local testing.
