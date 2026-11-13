#!/usr/bin/env python3
"""Huawei Cloud/ModelArts bootstrap for first-frame LivePhoto frame-cat LoRA training."""

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
VALIDATION_DATA_DIR = TEST_DATA_ROOT
VALIDATION_DATASETS = {
    "livephoto_test": TEST_DATA_ROOT / "livephoto_test",
    "B007_物": TEST_DATA_ROOT / "B007_物",
    "B007_人": TEST_DATA_ROOT / "B007_人",
}
MODEL_DIR = MODEL_ROOT / "Wan2.2-TI2V-5B"
OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        str(OUTPUT_ROOT / "livephoto_frame_cat_lora_theta30_phi5_self_render_1019"),
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
LIVEPHOTO_TEST_OBS_URI = os.getenv(
    "LIVEPHOTO_TEST_OBS_URI",
    os.getenv("TEST_DATA_OBS_URI",
    "s3://bucket-si-wulan/jianghognda/00_data_test/livephoto_test/",
    ),
)
B007_OBJECT_TEST_OBS_URI = os.getenv(
    "B007_OBJECT_TEST_OBS_URI",
    "s3://bucket-si-wulan/jianghognda/00_data_test/B007_物/",
)
B007_PERSON_TEST_OBS_URI = os.getenv(
    "B007_PERSON_TEST_OBS_URI",
    "s3://bucket-si-wulan/jianghognda/00_data_test/B007_人/",
)
MODEL_OBS_URI = os.getenv(
    "MODEL_OBS_URI",
    "s3://bucket-si-wulan/jianghognda/02_models/Wan2.2-TI2V-5B/",
)
OUTPUT_OBS_URI = os.getenv(
    "OUTPUT_OBS_URI",
    "s3://bucket-si-wulan/jianghognda/03_output/"
    "livephoto_frame_cat_lora_theta30_phi5_self_render_1019/",
)

FORCE_COPY = os.getenv("FORCE_COPY", "0").lower() in {"1", "true", "yes"}
FORCE_CODE_COPY = os.getenv("FORCE_CODE_COPY", "1").lower() in {"1", "true", "yes"}
STRICT_DATASET = os.getenv("STRICT_DATASET", "0").lower() in {"1", "true", "yes"}
SYNC_INTERVAL_SECONDS = max(60, int(os.getenv("SYNC_INTERVAL_SECONDS", "300")))


def copy_once(source: str, destination: Path, force: bool = False) -> None:
    """Copy an OBS directory once and mark only completed copies."""
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".mox_copy_complete"

    if marker.exists() and not FORCE_COPY and not force:
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


def build_validation_metadata() -> Path:
    """Merge the three image/caption test sets into one relative-path index."""
    combined: Dict[str, str] = {}
    for dataset_name, dataset_dir in VALIDATION_DATASETS.items():
        preferred = dataset_dir / "caption.json"
        if preferred.is_file():
            prompt_json = preferred
        else:
            json_files = sorted(dataset_dir.glob("*.json"))
            if len(json_files) != 1:
                raise RuntimeError(
                    f"Expected caption.json or exactly one JSON in {dataset_dir}; "
                    f"found {[p.name for p in json_files]}"
                )
            prompt_json = json_files[0]

        with prompt_json.open("r", encoding="utf-8") as handle:
            prompt_map = json.load(handle)
        if not isinstance(prompt_map, dict) or not prompt_map:
            raise RuntimeError(f"Validation captions must be a non-empty object: {prompt_json}")

        loaded = 0
        for image_name, caption in prompt_map.items():
            if not isinstance(caption, str) or not caption.strip():
                continue
            raw_path = Path(str(image_name))
            candidates = [dataset_dir / str(image_name), dataset_dir / raw_path.name]
            image_path = next((path for path in candidates if path.is_file()), None)
            if image_path is None:
                print(
                    f"[validation] WARNING: missing image for {dataset_name}: {image_name}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            relative_path = image_path.relative_to(TEST_DATA_ROOT).as_posix()
            combined[relative_path] = caption.strip()
            loaded += 1
        print(f"[validation] {dataset_name}: loaded {loaded} cases", flush=True)

    if len(combined) < 5:
        raise RuntimeError(f"Need at least 5 validation cases, found {len(combined)}")
    target_json = TEST_DATA_ROOT / "validation_prompts_combined.json"
    temporary = target_json.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, target_json)
    print(f"[validation] Wrote {len(combined)} combined cases to {target_json}", flush=True)
    return target_json


def patch_cloud_python39_compatibility() -> None:
    """Remove an unused datasets import that breaks with some cloud Python 3.9 images."""
    train_script = REPO_DIR / "scripts/wan2.2/train_livephoto_lora.py"
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
        print(f"[env] Installing PyTorch {torch_version} from the configured pip index", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"torch=={torch_version}"],
            check=True,
        )
        import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"PyTorch {torch.__version__} cannot access CUDA. "
            "Check nvidia-smi, GPU drivers and the installed PyTorch CUDA build."
        )
    print(f"[env] PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPUs {torch.cuda.device_count()}", flush=True)

    # Cloud training only: use the user's known-working versions for the HF
    # stack and OpenCV. Do not install the full requirements.txt: it contains
    # optional UI packages and unconstrained dependency families.
    from importlib import metadata

    packages = [
        ("accelerate", "accelerate", "1.10.1"),
        ("opencv-python", "cv2", "4.11.0.86"),
        ("diffusers", "diffusers", "0.36.0"),
        ("transformers", "transformers", "4.57.6"),
        ("huggingface-hub", "huggingface_hub", "0.36.2"),
        ("peft", "peft", "0.17.1"),
        ("protobuf", "google.protobuf", "3.20.3"),
        ("einops", "einops", "0.8.2"),
        ("safetensors", "safetensors", "0.7.0"),
        ("decord", "decord", "0.6.0"),
        ("omegaconf", "omegaconf", "2.3.0"),
        ("imageio", "imageio", "2.37.2"),
        ("imageio-ffmpeg", "imageio_ffmpeg", "0.6.0"),
        ("av", "av", "15.1.0"),
        ("ftfy", "ftfy", "6.3.1"),
        ("tensorboard", "tensorboard", "2.20.0"),
        # torch 2.6.x pairs with torchvision 0.21.x. Install without
        # dependencies so pip cannot replace the cloud CUDA torch wheel.
        ("torchvision", "torchvision", "0.21.0"),
        ("timm", "timm", "1.0.22"),
        ("tomesd", "tomesd", None),
        ("torchdiffeq", "torchdiffeq", None),
        ("torchsde", "torchsde", None),
        ("sentencepiece", "sentencepiece", None),
        ("albumentations", "albumentations", None),
        ("scikit-image", "skimage", None),
        ("librosa", "librosa", "0.11.0"),
        ("func-timeout", "func_timeout", "4.3.5"),
    ]
    constraints = CACHE_ROOT / "torch_constraints.txt"
    constraints.write_text(
        f"torch=={torch.__version__}\n"
        "opencv-python==4.11.0.86\n"
        "opencv-contrib-python==4.11.0.86\n"
        "opencv-python-headless==4.11.0.86\n"
        "opencv-contrib-python-headless==4.11.0.86\n",
        encoding="utf-8",
    )
    for package, module, version in packages:
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = None
        enforce_exact = package == "torchvision"
        if installed is not None and (not enforce_exact or installed == version):
            print(f"[env] Keeping {package}=={installed}", flush=True)
            continue
        if installed is not None:
            print(
                f"[env] Replacing incompatible {package}=={installed} with {version}",
                flush=True,
            )
        spec = f"{package}=={version}" if version else package
        print(f"[env] Installing missing {spec}", flush=True)
        command = [
            sys.executable, "-m", "pip", "install",
            "--prefer-binary", "--upgrade-strategy", "only-if-needed",
            "-c", str(constraints),
        ]
        # The Huawei mirror may not expose the already installed CUDA torch
        # wheel to pip's resolver. PEFT's torch>=1.13 requirement is satisfied
        # by our checked torch 2.6; do not make pip resolve it again.
        if package in {"torchvision", "peft", "timm", "tomesd", "torchdiffeq", "torchsde"}:
            command.append("--no-deps")
        command.append(spec)
        subprocess.run(command, check=True)
    import torchvision
    print(
        f"[env] torchvision {torchvision.__version__} imports successfully",
        flush=True,
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

    # Code is small and should follow the newly uploaded branch even when a
    # persistent /cache contains an old completion marker. Large data/model
    # directories keep their normal copy-once behavior.
    copy_once(CODE_OBS_URI, REPO_DIR, force=FORCE_CODE_COPY)
    copy_once(TOOLS_OBS_URI, TOOLS_DIR)
    copy_once(DATA_OBS_URI, DATASET_DIR)
    copy_once(LIVEPHOTO_TEST_OBS_URI, VALIDATION_DATASETS["livephoto_test"])
    copy_once(B007_OBJECT_TEST_OBS_URI, VALIDATION_DATASETS["B007_物"])
    copy_once(B007_PERSON_TEST_OBS_URI, VALIDATION_DATASETS["B007_人"])
    copy_once(MODEL_OBS_URI, MODEL_DIR)

    metadata_path = build_training_metadata()
    validation_prompt_json = build_validation_metadata()
    install_training_environment()
    patch_cloud_python39_compatibility()
    ensure_python_package("accelerate", "accelerate")
    ensure_python_package("tensorboard", "tensorboard")

    launcher = REPO_DIR / "scripts/wan2.2/train_livephoto_lora_huawei.sh"
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
            "VALIDATION_PROMPT_JSON": str(validation_prompt_json),
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
