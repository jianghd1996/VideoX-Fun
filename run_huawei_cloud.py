#!/usr/bin/env python3
"""Huawei Cloud/ModelArts bootstrap for VideoX-Fun trajectory LoRA training."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

import moxing as mox


CACHE_ROOT = Path(os.getenv("CACHE_ROOT", "/cache"))
DATA_ROOT = CACHE_ROOT / "training_data"
TEST_DATA_ROOT = CACHE_ROOT / "testing_data"
CODE_ROOT = CACHE_ROOT / "code"
MODEL_ROOT = CACHE_ROOT / "models"
OUTPUT_ROOT = CACHE_ROOT / "output"

REPO_DIR = CODE_ROOT / "VideoX-Fun-Single8"
TOOLS_DIR = CODE_ROOT / "tools"
DATASET_DIR = DATA_ROOT / "theta30_phi5_self_render_1019"
VALIDATION_DATA_DIR = TEST_DATA_ROOT / "livephoto_test"
MODEL_DIR = MODEL_ROOT / "Wan2.2-TI2V-5B"
OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        str(OUTPUT_ROOT / "trajectory_lora_theta30_phi5_self_render_1019"),
    )
)

CODE_OBS_URI = os.getenv(
    "CODE_OBS_URI",
    "s3://bucket-si-wulan/jianghognda/01_code/VideoX-Fun-Single8/",
)
TOOLS_OBS_URI = os.getenv(
    "TOOLS_OBS_URI",
    "s3://bucket-si-wulan/jianghognda/01_code/tools",
)
DATA_OBS_URI = os.getenv(
    "DATA_OBS_URI",
    "s3://bucket-si-wulan/jianghognda/00_data/theta30_phi5_self_render_1019/",
)
TEST_DATA_OBS_URI = os.getenv(
    "TEST_DATA_OBS_URI",
    "s3://bucket-si-wulan/jianghognda/00_data_test/livephoto_test/",
)
MODEL_OBS_URI = os.getenv(
    "MODEL_OBS_URI",
    "s3://bucket-si-wulan/jianghognda/02_models/Wan2.2-TI2V-5B/",
)
OUTPUT_OBS_URI = os.getenv(
    "OUTPUT_OBS_URI",
    "s3://bucket-si-wulan/jianghognda/03_output/"
    "trajectory_lora_theta30_phi5_self_render_1019/",
)

FORCE_COPY = os.getenv("FORCE_COPY", "0").lower() in {"1", "true", "yes"}
STRICT_DATASET = os.getenv("STRICT_DATASET", "0").lower() in {"1", "true", "yes"}
SYNC_INTERVAL_SECONDS = max(60, int(os.getenv("SYNC_INTERVAL_SECONDS", "300")))


def copy_once(source: str, destination: Path) -> None:
    """Copy an OBS directory once and mark only completed copies."""
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".mox_copy_complete"

    if marker.exists() and not FORCE_COPY:
        print(f"[cache] Skip existing completed copy: {destination}", flush=True)
        return

    print(f"[copy] {source} -> {destination}", flush=True)
    mox.file.copy_parallel(source, str(destination))
    marker.write_text(source + "\n", encoding="utf-8")
    print(f"[copy] Completed: {destination}", flush=True)


def resolve_video(item: Dict[str, Any]) -> Path:
    relative_path = str(item.get("relative_path") or "").strip()
    video_name = str(item.get("video_name") or "").strip()

    candidates: List[Path] = []
    if relative_path:
        clean_relative = relative_path.lstrip("/")
        candidates.append(DATA_ROOT / clean_relative)
        candidates.append(DATASET_DIR / clean_relative)
    if video_name:
        candidates.append(DATASET_DIR / video_name)
    if item.get("video_path"):
        candidates.append(DATASET_DIR / Path(str(item["video_path"])).name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    shown = ", ".join(str(path) for path in candidates) or "<no usable path>"
    raise FileNotFoundError(f"Cannot find video. Tried: {shown}")


def build_training_metadata() -> Path:
    """Convert video_path/caption records to VideoX-Fun file_path/text records."""
    source_json = DATASET_DIR / "theta30_phi5_self_render_1019.json"
    target_json = DATA_ROOT / "theta30_phi5_self_render_1019_cloud.json"

    if not source_json.is_file():
        raise FileNotFoundError(f"Dataset metadata does not exist: {source_json}")

    with source_json.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {source_json}, got {type(records).__name__}")

    converted: List[Dict[str, Any]] = []
    errors: List[str] = []
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            errors.append(f"record {index}: expected object")
            continue
        caption = item.get("caption", item.get("text", ""))
        try:
            video_path = resolve_video(item)
            video_path.relative_to(DATA_ROOT.resolve())
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"record {index}: {exc}")
            continue

        converted.append(
            {
                "file_path": str(video_path),
                "text": str(caption),
                "type": "video",
            }
        )

    if errors:
        preview = "\n".join(errors[:10])
        message = (
            f"{len(errors)} dataset records cannot be resolved and will be skipped. "
            f"First errors:\n{preview}"
        )
        if STRICT_DATASET:
            raise RuntimeError(message)
        print(f"[dataset] WARNING: {message}", file=sys.stderr, flush=True)
    if not converted:
        raise RuntimeError("No valid videos were found in the training metadata.")

    temporary = target_json.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, target_json)
    print(
        f"[dataset] Wrote {len(converted)} valid samples to {target_json}; "
        f"skipped {len(errors)} missing samples",
        flush=True,
    )
    return target_json


def patch_cloud_python39_compatibility() -> None:
    """Remove an unused datasets import that breaks with some cloud Python 3.9 images."""
    train_script = REPO_DIR / "scripts/wan2.2/train_lora.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"Training script does not exist: {train_script}")

    source = train_script.read_text(encoding="utf-8")
    patched = source.replace("\nimport datasets\n", "\n")
    patched = patched.replace(
        "        datasets.utils.logging.set_verbosity_warning()\n",
        "",
    )
    patched = patched.replace(
        "        datasets.utils.logging.set_verbosity_error()\n",
        "",
    )
    if patched != source:
        train_script.write_text(patched, encoding="utf-8")
        print(
            f"[compat] Removed unused Hugging Face datasets import from {train_script}",
            flush=True,
        )
    else:
        print("[compat] Python 3.9 datasets workaround already applied", flush=True)

    # Several unrelated model modules use PEP 604 annotations (X | None).
    # Python 3.9 evaluates those annotations eagerly and fails while importing
    # videox_fun.models, so postpone annotation evaluation for model modules.
    patched_annotation_files = 0
    future_import = "from __future__ import annotations\n"
    package_dir = REPO_DIR / "videox_fun"
    for python_file in package_dir.rglob("*.py"):
        module_source = python_file.read_text(encoding="utf-8")
        if future_import not in module_source:
            python_file.write_text(
                future_import + module_source,
                encoding="utf-8",
            )
            patched_annotation_files += 1
    print(
        f"[compat] Future annotations enabled in "
        f"{patched_annotation_files} videox_fun package files",
        flush=True,
    )


def remove_broken_optional_datasets() -> None:
    """Disable a broken optional HF datasets install that Accelerate auto-imports."""
    if importlib.util.find_spec("datasets") is None:
        print("[python] Optional datasets package is not installed", flush=True)
        return
    try:
        __import__("datasets")
        print("[python] Optional datasets package imports successfully", flush=True)
    except BaseException as error:
        print(
            f"[python] Broken optional datasets package detected: {error}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[python] Uninstalling datasets; VideoX-Fun uses its own PyTorch dataset",
            flush=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "datasets"],
            check=True,
        )


def install_training_environment() -> None:
    """Install training dependencies without replacing the server's CUDA PyTorch."""
    import shutil
    if shutil.which("ffmpeg") is None:
        print("[env] WARNING: ffmpeg executable is missing; install it via the system package manager if needed", flush=True)
    try:
        import torch
    except ImportError:
        # A new cloud image may lack PyTorch; install a CUDA wheel.
        torch_version = os.getenv("TORCH_VERSION", "2.6.0")
        cuda_tag = os.getenv("TORCH_CUDA_TAG", "cu124")
        torch_index = os.getenv("TORCH_INDEX_URL", f"https://download.pytorch.org/whl/{cuda_tag}")
        print(f"[env] Installing PyTorch {torch_version} from {torch_index}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--index-url", torch_index, f"torch=={torch_version}"],
            check=True,
        )
        import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"PyTorch {torch.__version__} cannot access CUDA. "
            "Check nvidia-smi, GPU drivers and TORCH_CUDA_TAG/TORCH_INDEX_URL."
        )
    print(f"[env] PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPUs {torch.cuda.device_count()}", flush=True)

    requirements = REPO_DIR / "requirements.txt"
    if not requirements.is_file():
        raise FileNotFoundError(f"Missing repository requirements: {requirements}")
    # The cloud image supplies CUDA PyTorch. Avoid replacing it and skip the
    # optional 'datasets' package, which breaks on some Python 3.9 images.
    excluded = {"torch", "datasets"}
    filtered = []
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].strip().lower().replace("_", "-")
        if name not in excluded:
            filtered.append(line)
    print(f"[env] Installing {len(filtered)} repository dependencies", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *filtered],
        check=True,
    )
    # These are used by the training/validation code and may not be in requirements.txt.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "peft", "protobuf"],
        check=True,
    )
    remove_broken_optional_datasets()


def ensure_python_package(import_name: str, pip_name: str) -> None:
    if importlib.util.find_spec(import_name) is not None:
        print(f"[python] {pip_name} is already installed", flush=True)
        return
    print(f"[python] Installing {pip_name}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)


def sync_output(reason: str) -> None:
    if not OUTPUT_OBS_URI or not OUTPUT_DIR.exists():
        return
    print(
        f"[upload:{reason}] {OUTPUT_DIR} -> {OUTPUT_OBS_URI}",
        flush=True,
    )
    mox.file.copy_parallel(str(OUTPUT_DIR), OUTPUT_OBS_URI)
    print(f"[upload:{reason}] Completed", flush=True)


def run_training(launcher: Path, environment: Dict[str, str]) -> None:
    """Run training and periodically upload completed outputs while it is alive."""
    process = subprocess.Popen(
        ["bash", str(launcher)],
        cwd=str(REPO_DIR),
        env=environment,
    )
    try:
        while True:
            try:
                return_code = process.wait(timeout=SYNC_INTERVAL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                try:
                    sync_output("periodic")
                except BaseException as upload_error:
                    # A transient OBS failure must not kill an otherwise healthy training job.
                    print(
                        f"[upload:periodic] Failed; will retry: {upload_error}",
                        file=sys.stderr,
                        flush=True,
                    )
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        raise

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process.args)


def main() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in (DATA_ROOT, TEST_DATA_ROOT, CODE_ROOT, MODEL_ROOT, OUTPUT_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    os.chdir(CACHE_ROOT)

    copy_once(CODE_OBS_URI, REPO_DIR)
    copy_once(TOOLS_OBS_URI, TOOLS_DIR)
    copy_once(DATA_OBS_URI, DATASET_DIR)
    copy_once(TEST_DATA_OBS_URI, VALIDATION_DATA_DIR)
    copy_once(MODEL_OBS_URI, MODEL_DIR)

    metadata_path = build_training_metadata()
    install_training_environment()
    patch_cloud_python39_compatibility()
    ensure_python_package("accelerate", "accelerate")
    ensure_python_package("tensorboard", "tensorboard")

    launcher = REPO_DIR / "scripts/wan2.2/train_lora_huawei.sh"
    if not launcher.is_file():
        raise FileNotFoundError(
            f"Cloud launcher is missing: {launcher}. "
            "Upload the latest repository code to CODE_OBS_URI first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "MODEL_NAME": str(MODEL_DIR),
            "DATASET_NAME": str(DATA_ROOT),
            "DATASET_META_NAME": str(metadata_path),
            "VALIDATION_DATA_DIR": str(VALIDATION_DATA_DIR),
            "OUTPUT_DIR": str(OUTPUT_DIR),
            "PYTHONUNBUFFERED": "1",
        }
    )

    training_error = None
    try:
        run_training(launcher, environment)
    except BaseException as exc:
        training_error = exc
        raise
    finally:
        try:
            sync_output("final")
        except BaseException as upload_error:
            if training_error is None:
                raise
            print(
                f"[upload:final] Failed after training error: {upload_error}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
