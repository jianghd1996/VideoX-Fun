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
- Changed to dual GPU: `--gpu-ids 6,7 --num_processes=2`
- Implemented new `log_validation` function:
  - Randomly samples from training data (not from validation_prompts/paths)
  - Extracts first/last frames from GT video as start/end images
  - Uses control video as control signal
  - Generates video using pipeline
  - Saves 4-panel concat: GT | Control | Mask | Generated (n_rows=4)
  - Supports multi-GPU: each card runs its own samples
  - 2 samples per GPU per validation (configurable via `--validation_samples_per_gpu`)
  - Validation runs at step 0 and every 500 steps
  - Validation uses 81 frames at 720P resolution (960) with 8 inference steps
- Added new args:
  - `--validation_samples_per_gpu`: Number of validation samples per GPU (default 2)
  - `--validation_n_frames`: Number of frames for validation video (default 81)
- Changed `checkpointing_steps` from 50 to 500
- Changed `validation_steps` from 2000 to 500
- Removed dependency on `--validation_prompts` and `--validation_paths` for validation
- Fixed bugs:
  - `torch.randperm` requires CPU generator, not CUDA generator - created separate `cpu_generator`
  - `get_image_to_video_latent` expects lists of images, not single Image objects - wrapped frames in lists
  - Pipeline mask dimension mismatch for 5B model when latent spatial dims are odd - used `F.interpolate` instead of `::2` stride
  - **F variable shadowing**: Renamed `F` to `num_frames` in sanity check logging to avoid shadowing `torch.nn.functional` (caused AttributeError at line 2009)
  - **Tokenizers fork warning**: Set `TOKENIZERS_PARALLELISM=false` in shell script
  - **Aspect ratio mismatch**: GT and control videos may have different aspect ratios. Using GT aspect ratio for `target_w` caused latent shape mismatch (33 vs 32) in the denoising loop. Fixed by reading control video dimensions to compute `target_w`, ensuring both control_video and inpaint_video use the same spatial dimensions.
  - **Validation sampling count**: Only got 3 samples instead of 4 (2 GPUs × 2 samples). Fixed by replacing for-loop with while-loop and retry mechanism - ensures each GPU generates exactly `validation_samples_per_gpu` valid samples by retrying when file not found or frame read fails
- Added temporal reversal augmentation:
  - Training: 50% probability to reverse both GT and control videos along temporal axis
  - Validation: 50% probability to reverse, with correct first/last frame alignment (swap frames when reversed)
- Added logging for resolution and frame count during training data sampling
- **Mask-Aware Control Adapter** (v2.0):
  - Added automatic mask extraction from control video in dataset (detects black regions where RGB < 20)
  - Added `ControlMaskEncoder` module: small 3D conv (1ch → 64ch → latent_channels) that encodes mask and adds to control latents
  - Mask encoder is trained fully (not with LoRA) with 10x learning rate
  - Mask encoder parameters are saved with checkpoints (embedded in LoRA state dict with prefix `control_mask_encoder.`)
  - Validation now shows 4-panel output: GT | Control Video | Mask | Generated
  - This allows the model to learn to ignore invalid/black regions in 3D Gaussian rendered control videos
- **GT-Aware Mask Extraction** (v2.1):
  - Improved mask extraction to distinguish foreground black objects from background holes
  - Training: mask = (control black AND GT not black) = background hole only
  - If both control and GT are black at same position → foreground object (not masked)
  - Prevents masking naturally black foreground objects (black clothes, hair, etc.)
  - Validation: uses same GT-aware logic for mask visualization
  - At inference time: use control video black regions as mask (model learned to handle this)
  - Fixed DataLoader worker OOM by reducing `dataloader_num_workers` from 8 to 4 and using uint8 for mask storage

### 2026-07-29 (Session 1)
- Initial setup: configured remote dev workflow
- Created DEV_LOG.md
- First push to GitHub
- **Reset to upstream** `804a425` due to issues with previous local changes
- Force pushed to remote
- Stash preserved for horizontal_flip feature
