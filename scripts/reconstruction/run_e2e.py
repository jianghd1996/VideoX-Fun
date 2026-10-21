#!/usr/bin/env python3
"""Edit the settings below, then run: python scripts/reconstruction/run_e2e.py.

Video uses both GPUs; DA3 and dense reconstruction use the primary GPU.
CPU_THREADS=32 follows the user's measured thread sweep; 16 is an alternative
for a busy server. This is a per-process thread limit, not a CPU process count.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

# ===================== EDIT THESE SETTINGS =====================
GPU_IDS = "4,5"                 # "4" for the single-GPU baseline
CPU_THREADS = 32               # per rank; use 16 on a busy server
WORK_DIR = Path("/home/z00566689/dev/mnt/jiang_test/00_1021_test")
DA3_ROOT = Path("/mnt/DataPart/jianghongda/Depth-Anything-3")
VIDEO_ROOT = None              # Auto: config, sibling VideoX-Fun, or enclosing checkout
# Explicit example when copied into SingleRecon:
# VIDEO_ROOT = Path(__file__).resolve().parent / "VideoX-Fun"

# Reuse an existing JSON, or leave None to import your edited DEFAULT_ARGS.
CONFIG_PATH = None             # e.g. Path("/path/to/SingleRecon/e2e_config.json")
BASE_RUN_CASES = Path(__file__).with_name("run_cases.py")
# If you keep the working configuration in SingleRecon/run_cases.py, change
# BASE_RUN_CASES to that file so model/dense paths are copied from your setup.
RUN_DENSE = True               # False measures only video + DA3
DRY_RUN = False                # True prints config/command without loading models
# ===============================================================


def resolve_video_root(script_path, config, explicit=None):
    """Support both checkout/scripts/reconstruction and copied SingleRecon scripts."""
    base = Path(script_path).resolve().parent
    def normalize(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base / path).resolve()
    def valid(path):
        return (path / 'predict_ti2v.py').is_file() and (path / 'videox_fun').is_dir()
    for label, value in (('VIDEO_ROOT', explicit),
                         ('config.video_code_root', config.get('video_code_root'))):
        if value is not None:
            path = normalize(value)
            if not valid(path):
                raise FileNotFoundError(f'{label}={path}: expected predict_ti2v.py and videox_fun/. '
                                        'Set VIDEO_ROOT to your VideoX-Fun checkout.')
            return path
    candidates = [base / 'VideoX-Fun', base, *base.parents]
    for candidate in candidates:
        if valid(candidate):
            return candidate
    raise FileNotFoundError('Cannot locate VideoX-Fun. Set VIDEO_ROOT to the directory '
                            f'containing predict_ti2v.py and videox_fun/; script directory: {base}')


def main():
    if not isinstance(CPU_THREADS, int) or CPU_THREADS < 1:
        raise ValueError("CPU_THREADS must be a positive integer")
    gpus = [value.strip() for value in GPU_IDS.split(',') if value.strip()]
    if len(gpus) not in (1, 2) or len(set(gpus)) != len(gpus):
        raise ValueError('GPU_IDS must contain one or two distinct GPU IDs')
    launcher = Path(__file__).with_name('run_end_to_end_multi_gpu.py').resolve()
    if CONFIG_PATH is not None:
        config = json.loads(Path(CONFIG_PATH).expanduser().read_text(encoding='utf-8'))
    else:
        source = Path(BASE_RUN_CASES).expanduser().resolve()
        spec = importlib.util.spec_from_file_location('e2e_existing_settings', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = dict(module.DEFAULT_ARGS)
    video_root = resolve_video_root(__file__, config, VIDEO_ROOT)
    if not launcher.is_file():
        raise FileNotFoundError(f'End-to-end launcher missing: {launcher}')
    config.update(video_code_root=str(video_root),
                  run_gen_video=True, run_DA3=True, run_captioner=False,
                  run_3dgs_iter=RUN_DENSE, world_size=len(gpus), multi_gpu=len(gpus) > 1,
                  external_code_root=str(DA3_ROOT.expanduser().resolve()))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = str(CPU_THREADS)
    print(f'[RUN] GPUs={gpus}, CPU threads per rank={CPU_THREADS}, DA3=single GPU', flush=True)
    print('[CONFIG] ' + json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    # Temp configuration is local to this invocation, preventing concurrent
    # baseline/two-GPU runs from changing one another's settings.
    with tempfile.TemporaryDirectory(prefix='videox_e2e_') as temporary:
        cfg_path = Path(temporary) / 'config.json'
        cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc_per_node={len(gpus)}', str(launcher),
                   '--config', str(cfg_path), '--work-dir', str(WORK_DIR.expanduser().resolve()),
                   '--da3-root', str(DA3_ROOT.expanduser().resolve()),
                   '--cpu-threads', str(CPU_THREADS)]
        print('[COMMAND] ' + shlex.join(command), flush=True)
        if not DRY_RUN:
            subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
