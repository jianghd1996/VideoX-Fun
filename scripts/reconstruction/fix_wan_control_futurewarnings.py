#!/usr/bin/env python3
"""Fix deprecated VAE config access and CUDA autocast in the *local* Wan Control pipeline.

Does NOT download or replace VideoX-Fun: edits one specific file under
CodexRecon/VideoX-Fun, with a timestamped backup and Python syntax validation.

Usage:
  python fix_wan_control_futurewarnings.py --check
  python fix_wan_control_futurewarnings.py
"""
from __future__ import annotations

import argparse
import ast
import difflib
import re
from datetime import datetime
from pathlib import Path

DEFAULT_TARGET = (Path("/home/z00566689/dev/mnt/jiang_dev/CodexRecon") /
                  "VideoX-Fun/videox_fun/pipeline/pipeline_wan2_2_fun_control.py")
VAE_ATTRS = ("temporal_compression_ratio", "spatial_compression_ratio", "latent_channels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--check", action="store_true", help="Show proposed changes without modifying anything")
    args = parser.parse_args()
    target = args.target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    original = target.read_text(encoding="utf-8")
    updated = original
    changes = {}
    for attr in VAE_ATTRS:
        pattern = rf"\bself\.vae\.{attr}\b"
        updated, count = re.subn(pattern, f"self.vae.config.{attr}", updated)
        changes[attr] = count
    updated, autocast_count = re.subn(
        r"\btorch\.cuda\.amp\.autocast\(\s*dtype\s*=\s*weight_dtype\s*\)",
        'torch.amp.autocast("cuda", dtype=weight_dtype)', updated)
    changes["cuda_autocast"] = autocast_count

    # Syntax-check before applying changes.
    ast.parse(updated, filename=str(target))
    print(f"[WARNING FIX] target={target}")
    for key, count in changes.items():
        print(f"  {key}: {count} replacements")
    if original == updated:
        print("[WARNING FIX] Already up to date; no changes needed.")
        return
    print("".join(difflib.unified_diff(
        original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile=str(target), tofile=str(target) + " (fixed)")))
    if args.check:
        print("[WARNING FIX] Check only: nothing modified")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + f".bak_{stamp}")
    if backup.exists():
        raise FileExistsError(backup)
    backup.write_bytes(target.read_bytes())
    temp = target.with_name(target.name + ".warning_fix.tmp")
    try:
        temp.write_text(updated, encoding="utf-8")
        # Preserve original file permissions and replace atomically.
        temp.chmod(target.stat().st_mode)
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()
    print(f"[WARNING FIX] Updated {target}")
    print(f"[WARNING FIX] Backup {backup}")


if __name__ == "__main__":
    main()
