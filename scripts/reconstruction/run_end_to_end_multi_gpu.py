#!/usr/bin/env python3
"""Same-case parallel Wan followed by single-GPU DA3 reconstruction.

python run_end_to_end_multi_gpu.py --help
Launch with torchrun --standalone --nproc_per_node=2 (or 1 for baseline).
Uses the legacy-compatible DA3 runner, preserving GS/COLMAP exports.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from datetime import timedelta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Existing SparseRecon JSON config')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--da3-root', type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=1,
                        help='CPU threads per rank for Torch/BLAS/OpenMP/OpenCV (default: 1)')
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be >= 1')
    # Set before importing numerical libraries; each torchrun rank otherwise
    # creates independent CPU thread pools, which can oversubscribe the host.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.cpu_threads)
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    import cv2
    cv2.setNumThreads(args.cpu_threads)
    import torch.distributed as dist
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    size = int(os.environ.get('WORLD_SIZE', '1'))
    if size not in (1, 2):
        raise ValueError('This launcher supports one or two GPUs; Wan path is validated for two ranks')
    print(f'[CPU THREADS rank={rank}] torch/BLAS/OpenMP/OpenCV={args.cpu_threads}, interop=1', flush=True)
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    if size > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    os.environ['DA3_SEQUENCE_PARALLEL'] = '0'
    da3_root = args.da3_root.expanduser().resolve()
    if not (da3_root / 'DA3/run_da3_full_crop_dev.py').is_file():
        raise FileNotFoundError(f'Compatible DA3 runner missing: {da3_root}')
    sys.path.insert(0, str(da3_root))
    sys.path.insert(0, str(da3_root / 'src'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline_reconstruction import SparseRecon
    options = json.loads(args.config.read_text())
    options.update(external_code_root=str(da3_root), da3_backend='legacy',
                   da3_factory=None, world_size=size, multi_gpu=size > 1,
                   resume_stages=False, reuse_video=False)
    options.setdefault('run_gen_video', True)
    options.setdefault('run_DA3', True)
    options.setdefault('run_captioner', False)
    if options['run_captioner']:
        raise ValueError('Prepare prompt.txt first and set run_captioner=false for this benchmark')
    if not options['run_gen_video'] or not options['run_DA3']:
        raise ValueError('End-to-end benchmark requires run_gen_video=true and run_DA3=true')
    root = args.work_dir.expanduser().resolve()
    cases = [root] if (root / 'image').is_dir() else [
        p for p in sorted(root.iterdir()) if p.is_dir() and (p / 'image').is_dir()
        and p.name not in set(options.get('skip_list', ['collect', 'test']))]
    run_list = set(options.get('run_list', []))
    cases = [p for p in cases if not run_list or p.name in run_list]
    if not cases:
        raise ValueError(f'No cases in {root}')
    # Validate shared inputs on every rank before entering model collectives.
    if not options['run_captioner']:
        for case in cases:
            prompt = case / 'prompt.txt'
            if not prompt.is_file() or not prompt.read_text().strip():
                raise FileNotFoundError(f'Provide a nonempty {prompt}')
    worker_options = dict(options)
    if rank:
        worker_options.update(run_captioner=False, run_3dgs_iter=False, run_DA3=False)
    def barrier():
        if size > 1:
            dist.barrier()
    try:
        recon = SparseRecon(worker_options)
        import DA3.run_da3_full_crop_dev as runner_module
        if not Path(runner_module.__file__).resolve().is_relative_to(da3_root):
            raise ImportError(f'Unexpected DA3 import: {runner_module.__file__}')
        print(f'[E2E rank={rank}] DA3={runner_module.__file__} DA3=single_gpu', flush=True)
        original_da3 = recon.pipeline.stage_da3
        def joint_da3():
            # Rank 0 must finish frame extraction before any DA3 rank reads files.
            barrier()
            value = original_da3() if rank == 0 else None
            # Includes primary-only GS rendering and COLMAP export.
            barrier()
            return value
        recon.pipeline.stage_da3 = joint_da3
        barrier()
        summary = {}
        for case in cases:
            start = time.perf_counter()
            if rank == 0:
                result = recon(work_dir=case, export_dir=case / 'export', mode=options.get('mode', 'scene'))
            else:
                recon.pipeline.set_case(case)
                recon.pipeline.stage_video(reuse=False)
                joint_da3()
            barrier()  # Wait for primary-only dense reconstruction before next case.
            if rank == 0:
                summary[case.name] = dict(result['timings'], end_to_end_wall=time.perf_counter()-start,
                                          gpus=size, da3_parallel=False)
                report = root / f'end_to_end_{size}gpu.json'
                report.write_text(json.dumps(summary, indent=2))
                print('[E2E] ' + json.dumps(summary[case.name]), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
