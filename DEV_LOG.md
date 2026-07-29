# VideoX-Fun Development Log

## Remote Dev Workflow

- **Local**: Edit code here (Kilo session)
- **Remote**: Pull from GitHub, run training/inference on GPU server
- **Branch**: `main` (direct development)
- **Push**: After each meaningful change, commit & push to `origin/main`

---

## Current State

**Base Commit**: `804a425` - Update Flash Head && Update Readmes (#491)
- Date: 2026-05-12
- Author: Bubbliiiing (original repo)
- This is the last upstream commit before any local modifications

**Previous Local Changes**: 82 commits discarded (reset on 2026-07-29)
- Training infrastructure adaptations
- Validation system improvements
- Mask Adapter / Control training modifications
- OOM/NCCL fixes
- Dataset enhancements
- Performance optimizations

**Stash**: `horizontal_flip + aider ignore` changes still preserved
- `git stash list` to see
- `git stash pop` to restore when needed

---

## Project Overview

VideoX-Fun: Video generation & editing framework supporting multiple models:
- CogVideoX-Fun (v1/v1.1/v1.5)
- Wan2.1/2.2 Fun (including 5B)
- Z-Image / Qwen-Image / Flux2
- VACE, Phantom, FlashHead, LTX-2, etc.

---

## Session Notes

> Add notes here for future sessions to pick up context.

### 2026-07-29 (Session 2)
- Updated `train_control_lora.sh` for Wan2.2-Fun-5B-Control training
- Key changes:
  - Config: `wan_civitai_5b.yaml` (5B single-Transformer) instead of `wan_civitai_i2v.yaml` (A14B dual-Transformer)
  - `boundary_type="full"` for 5B single-Transformer (was `"low"` for A14B)
  - Multi-GPU: `--gpu-ids 6,7 --num_processes=2 --main_process_port=29501`
  - Paths updated to user's remote server paths
- Dataset: Relative paths in JSON, base dir `/mnt/DataPart/jianghongda/dataset/livephoto`
- Metadata: `/mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun-ori/datasets/dataset1+2.json`
- Note: Metadata JSON lacks width/height fields, will be read from video files at runtime (slower)

### 2026-07-29 (Session 3)
- Changed to single GPU: `--gpu-ids 7 --num_processes=1`
- Implemented new `log_validation` function:
  - Randomly samples from training data (not from validation_prompts/paths)
  - Extracts first/last frames from GT video as start/end images
  - Uses control video as control signal
  - Generates video using pipeline
  - Saves 3-panel concat: GT | Control | Generated (n_rows=3)
  - Supports multi-GPU: each card runs its own samples
  - 2 samples per GPU per validation (configurable via `--validation_samples_per_gpu`)
  - Validation runs at step 0 and every 500 steps
  - Validation uses 21 frames to avoid OOM (configurable via `--validation_n_frames`)
- Added new args:
  - `--validation_samples_per_gpu`: Number of validation samples per GPU (default 2)
  - `--validation_n_frames`: Number of frames for validation video (default 21)
- Changed `checkpointing_steps` from 50 to 500
- Changed `validation_steps` from 2000 to 500
- Removed dependency on `--validation_prompts` and `--validation_paths` for validation

### 2026-07-29 (Session 1)
- Initial setup: configured remote dev workflow
- Created DEV_LOG.md
- First push to GitHub
- **Reset to upstream** `804a425` due to issues with previous local changes
- Force pushed to remote
- Stash preserved for horizontal_flip feature
