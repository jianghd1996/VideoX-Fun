"""Independent-GPU Wan Control generation for DA3-planned trajectories.

One spawned worker per physical GPU, each owns a persistent WanControl5B model.
Workers pull whole trajectory tasks from a shared queue; no NCCL/FSDP.
Run only from the Python script entry point (multiprocessing 'spawn').
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from pathlib import Path


def _worker(gpu_id, video_code_root, adapter_spec, environment, tasks, events):
    # Bind CUDA before importing either torch or the Wan adapter.
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        for key, value in environment.items():
            os.environ[key] = str(value)
        # Limit every spawned worker BEFORE importing PyTorch, numpy/OpenCV,
        # tokenizer or SLA modules. Parent DA3/3DGS threading is unchanged.
        cpu_threads = int(environment.get("SINGLERECON_CONTROL_CPU_THREADS", "16"))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[name] = str(cpu_threads)
        import torch
        import cv2
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        print(f"[VIDEO GPU {gpu_id}] cpu_threads={torch.get_num_threads()} "
              f"interop={torch.get_num_interop_threads()} "
              f"opencv={cv2.getNumThreads()}", flush=True)
        if video_code_root not in sys.path:
            sys.path.insert(0, video_code_root)
        module, symbol = adapter_spec.split(":", 1)
        factory = getattr(importlib.import_module(module), symbol)
        start = time.perf_counter()
        generator = factory() if isinstance(factory, type) else factory
        load = getattr(generator, "load", None)
        if callable(load):
            load()
        events.put({"kind": "ready", "gpu": gpu_id,
                    "seconds": round(time.perf_counter() - start, 3)})
        while True:
            item = tasks.get()
            if item is None:
                break
            start = time.perf_counter()
            try:
                result = generator(
                    control_video=item["control_video"],
                    start_image=item["start_image"],
                    end_image=item["end_image"],
                    output_video=item["video"],
                    prompt=item["prompt"], num_frames=item["frames"], fps=item["fps"])
                output = Path(item["video"])
                if result is not None and Path(result).resolve() != output.resolve():
                    raise RuntimeError(f"Adapter returned {result}, expected {output}")
                if not output.is_file() or output.stat().st_size == 0:
                    raise FileNotFoundError(f"Generated video missing/empty: {output}")
                record = {key: value for key, value in item.items() if key != "prompt"}
                record.update(gpu_id=gpu_id, reused=False,
                              seconds=round(time.perf_counter() - start, 3),
                              inference_timing=dict(getattr(generator, "last_timing", {})))
                events.put({"kind": "done", "record": record})
                print(f"[VIDEO GPU {gpu_id}] done {item['name']} in "
                      f"{record['seconds']:.3f}s", flush=True)
            except BaseException:
                events.put({"kind": "error", "gpu": gpu_id, "task": item["name"],
                            "traceback": traceback.format_exc()})
                return
    except BaseException:
        events.put({"kind": "error", "gpu": gpu_id, "task": "<initialize>",
                    "traceback": traceback.format_exc()})



class WanControlWorkerPool:
    """Spawn persistent GPU-bound workers once and reuse across cases.

    No CUDA/model import in parent. Call close() from a finally block.
    """

    def __init__(self, *, adapter_spec, video_code_root, gpu_ids, env_overrides=None):
        self.ids = list(map(int, gpu_ids))
        if len(self.ids) < 2 or len(self.ids) != len(set(self.ids)):
            raise ValueError(f"Need >=2 unique physical GPUs: {self.ids}")
        self.ctx = mp.get_context("spawn")
        self.tasks = self.ctx.Queue()
        self.events = self.ctx.Queue()
        self.workers = []
        self.loads = {}
        self.closed = False
        self.started_at = time.perf_counter()
        try:
            for gpu in self.ids:
                proc = self.ctx.Process(
                    target=_worker,
                    args=(gpu, str(Path(video_code_root).resolve()), adapter_spec,
                          dict(env_overrides or {}), self.tasks, self.events),
                    name=f"wan-control-gpu-{gpu}")
                proc.start()
                self.workers.append(proc)
                print(f"[MULTI GPU PRELOAD] GPU {gpu}, pid={proc.pid}", flush=True)
            while len(self.loads) < len(self.ids):
                event = self._next_event()
                if event["kind"] == "error":
                    raise RuntimeError(
                        f"GPU {event['gpu']} preload failed:\n{event['traceback']}")
                if event["kind"] != "ready":
                    raise RuntimeError(f"Unexpected event during preload: {event['kind']}")
                self.loads[str(event["gpu"])] = event["seconds"]
                print(f"[MULTI GPU PRELOAD] GPU {event['gpu']} ready, "
                      f"model_load={event['seconds']:.3f}s", flush=True)
        except BaseException:
            self.close(terminate=True)
            raise
        self.preload_wall_seconds = round(time.perf_counter() - self.started_at, 3)
        print(f"[MULTI GPU PRELOAD] all ready in {self.preload_wall_seconds:.3f}s",
              flush=True)

    def _next_event(self):
        while True:
            try:
                return self.events.get(timeout=3)
            except queue.Empty:
                failed = [(p.name, p.exitcode) for p in self.workers
                          if p.exitcode not in (None, 0)]
                if failed or not any(p.is_alive() for p in self.workers):
                    raise RuntimeError(f"Wan workers unavailable: {failed}")

    def generate(self, pending, completed, persist):
        if self.closed:
            raise RuntimeError("Wan pool already closed")
        start = time.perf_counter()
        for item in pending:
            self.tasks.put(item)
        remaining = len(pending)
        while remaining:
            event = self._next_event()
            if event["kind"] == "error":
                raise RuntimeError(
                    f"GPU {event['gpu']}, task {event['task']} failed:\n{event['traceback']}")
            if event["kind"] != "done":
                raise RuntimeError(f"Unexpected worker event: {event['kind']}")
            record = event["record"]
            completed[record["name"]] = record
            remaining -= 1
            persist()
            print(f"[MULTI GPU] {len(pending)-remaining}/{len(pending)} "
                  f"{record['name']} GPU {record['gpu_id']} "
                  f"{record['seconds']:.3f}s", flush=True)
        return round(time.perf_counter() - start, 3)

    def close(self, terminate=False):
        if self.closed:
            return
        self.closed = True
        if not terminate:
            for _ in self.workers:
                self.tasks.put(None)
            for proc in self.workers:
                proc.join(timeout=15)
        for proc in self.workers:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        self.tasks.close()
        self.events.close()


def run_trajectory_videos_multi_gpu(*, reconstruction_dir, source_images,
                                     adapter_spec, video_code_root, gpu_ids,
                                     prompt, fps=30, resume=False,
                                     env_overrides=None, worker_pool=None):
    """Generate one MP4 per path; save the same result schema as single GPU.

    summary.json additionally stores worker loads, GPU assignment and wall time.
    """
    import numpy as np

    root = Path(reconstruction_dir)
    manifest_file = root / "trajectories" / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    sources = [Path(p) for p in source_images]
    generated_root = root / "generated_trajectories"
    generated_root.mkdir(parents=True, exist_ok=True)
    ids = [int(i) for i in gpu_ids]
    if len(ids) < 2 or len(set(ids)) != len(ids) or min(ids) < 0:
        raise ValueError(f"Expected >=2 distinct physical GPU IDs: {gpu_ids}")
    paths = manifest["trajectories"]
    if not paths:
        raise ValueError(f"No trajectories in {manifest_file}")

    pending, completed = [], {}
    for item in paths:
        name = Path(item["file"]).stem
        pose_file = root / "trajectories" / item["file"]
        poses = np.load(pose_file, mmap_mode="r")
        n = int(item["frames"])
        if poses.shape != (n, 4, 4):
            raise ValueError(f"Trajectory {name} poses {poses.shape}, expected {(n, 4, 4)}")
        a, b = int(item["start_index"]), int(item["end_index"])
        if not 0 <= a < len(sources) or not 0 <= b < len(sources):
            raise IndexError(f"Invalid input view indexes for {name}: {a}->{b}")
        control = root / "trajectories" / "gs_video" / f"{name}.mp4"
        if not control.is_file():
            raise FileNotFoundError(control)
        dest = generated_root / name / "generated.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "name": name, "video": str(dest), "frames": n,
            "start_index": a, "end_index": b,
            "start_image": str(sources[a]), "end_image": str(sources[b]),
            "control_video": str(control), "pose_file": str(pose_file),
        }
        if resume and dest.is_file() and dest.stat().st_size:
            completed[name] = dict(record, reused=True, seconds=0.0,
                                   inference_timing={}, gpu_id=None)
        else:
            pending.append(dict(record, prompt=prompt, fps=int(fps)))

    def persist():
        # Keep manifest ordering deterministic regardless of worker completion order.
        ordered = [completed[Path(p["file"]).stem] for p in paths
                   if Path(p["file"]).stem in completed]
        target = generated_root / "manifest.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)

    persist()
    if not pending:
        reused_output = [completed[Path(p["file"]).stem] for p in paths]
        (generated_root / "summary.json").write_text(json.dumps({
            "mode": "multi_gpu", "gpu_ids": [], "model_load_by_gpu": {},
            "wall_seconds": 0.0, "sum_segment_seconds": 0.0,
            "trajectories": len(reused_output), "videos_generated": 0,
            "videos_reused": len(reused_output),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[MULTI GPU] All videos reused; no workers launched", flush=True)
        return reused_output

    if worker_pool is not None:
        if ids != worker_pool.ids:
            raise ValueError(f"GPU mismatch: requested {ids}, pool {worker_pool.ids}")
        wall = worker_pool.generate(pending, completed, persist)
        output = [completed[Path(p["file"]).stem] for p in paths]
        details = {"mode": "multi_gpu_preloaded", "gpu_ids": ids,
                   "model_load_by_gpu": worker_pool.loads,
                   "model_preload_wall_seconds": worker_pool.preload_wall_seconds,
                   "wall_seconds": wall,
                   "sum_segment_seconds": round(sum(x["seconds"] for x in output), 3),
                   "trajectories": len(output),
                   "videos_generated": len(pending)}
        (generated_root / "summary.json").write_text(
            json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[MULTI GPU PRELOADED SUMMARY] {len(pending)} jobs on {ids}; "
              f"video_wall={wall:.3f}s; preloaded_before_case=True", flush=True)
        return output

    # Never import the adapter in the parent process: it initializes CUDA.
    ctx = mp.get_context("spawn")
    tasks = ctx.Queue()
    events = ctx.Queue()
    active_ids = ids[:min(len(ids), len(pending))]
    for item in pending:
        tasks.put(item)
    for _ in active_ids:
        tasks.put(None)

    workers = []
    started = time.perf_counter()
    loads = {}
    try:
        for gpu in active_ids:
            proc = ctx.Process(
                target=_worker,
                args=(gpu, str(Path(video_code_root).resolve()), adapter_spec,
                      dict(env_overrides or {}), tasks, events),
                name=f"wan-control-gpu-{gpu}")
            proc.start()
            workers.append(proc)
            print(f"[MULTI GPU] launched worker GPU {gpu}, pid={proc.pid}", flush=True)
        remaining = len(pending)
        while remaining:
            try:
                event = events.get(timeout=3)
            except queue.Empty:
                failed = [(active_ids[i], p.exitcode) for i, p in enumerate(workers)
                          if p.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(f"GPU worker(s) crashed before completion: {failed}")
                if all(not p.is_alive() for p in workers):
                    raise RuntimeError(
                        f"All workers exited but {remaining} video results are missing")
                continue
            kind = event["kind"]
            if kind == "error":
                raise RuntimeError(
                    f"GPU {event['gpu']}, task {event['task']} failed:\n{event['traceback']}")
            if kind == "ready":
                loads[str(event["gpu"])] = event["seconds"]
                print(f"[MULTI GPU] GPU {event['gpu']} ready, "
                      f"model_load={event['seconds']:.3f}s", flush=True)
            elif kind == "done":
                record = event["record"]
                completed[record["name"]] = record
                remaining -= 1
                persist()
                print(f"[MULTI GPU] {len(pending)-remaining}/{len(pending)} "
                      f"{record['name']} GPU {record['gpu_id']} "
                      f"{record['seconds']:.3f}s", flush=True)
        for proc in workers:
            proc.join(timeout=15)
        bad = [(p.name, p.exitcode) for p in workers if p.exitcode not in (0,)]
        if bad:
            raise RuntimeError(f"GPU workers exited abnormally: {bad}")
    finally:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join(timeout=5)
        tasks.close()
        events.close()

    output = [completed[Path(p["file"]).stem] for p in paths]
    wall = round(time.perf_counter() - started, 3)
    details = {"mode": "multi_gpu", "gpu_ids": active_ids,
               "model_load_by_gpu": loads,
               "wall_seconds": wall,
               "sum_segment_seconds": round(sum(x["seconds"] for x in output), 3),
               "trajectories": len(output),
               "videos_generated": len(pending)}
    (generated_root / "summary.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[MULTI GPU SUMMARY] {len(pending)} jobs on {active_ids}; "
          f"wall={wall:.3f}s, summed_job_time={details['sum_segment_seconds']:.3f}s",
          flush=True)
    return output
