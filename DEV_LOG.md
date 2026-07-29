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

### 2026-07-29
- Initial setup: configured remote dev workflow
- Created DEV_LOG.md
- First push to GitHub
- **Reset to upstream** `804a425` due to issues with previous local changes
- Force pushed to remote
- Stash preserved for horizontal_flip feature
