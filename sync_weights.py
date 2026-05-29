#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight sync daemon for VideoX-Fun training outputs.
Monitors checkpoint weights and validation samples, incrementally
copies new/changed files to a safe backup location.

Design:
- Poll + sleep, no inotify fd usage, near-zero CPU
- Compare size + mtime only, no hashing (save IO)
- File stability check before copy (waits for writes to finish)
- Supports nohup background execution with file logging

Usage:
    python sync_weights.py [--interval 60]
    python sync_weights.py --once
    nohup python sync_weights.py --interval 120 > /dev/null 2>&1 &

Config: modify the CONFIG dict below, or override via environment variables.
"""

import os
import sys
import time
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple

# ── CONFIG ────────────────────────────────────────────────────────────
CONFIG = {
    "src_root": os.environ.get(
        "SYNC_SRC",
        "/cache/01_code/VideoX-Fun/output_dir_wan2.2_5b_control_lora",
    ),
    "dst_root": os.environ.get(
        "SYNC_DST",
        "/data/jianghongda/03_output/0529_mask_lora",
    ),
    "log_dir": os.environ.get(
        "SYNC_LOG_DIR",
        "/cache/01_code/VideoX-Fun/logs",
    ),
    # Poll interval in seconds (weights save every many steps, 120s is plenty)
    "poll_interval": int(os.environ.get("SYNC_INTERVAL", "120")),
    # File stability: wait between samples + max retries
    "stable_wait": 2.0,
    "stable_retries": 3,
    # Subdirs to monitor (relative to src_root)
    "watch_dirs": [
        "",          # root-level weights / checkpoint dirs
        "sample",
        "samples",
        "validation",
    ],
    # File extensions to sync
    "sync_extensions": {
        ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx",
        ".json", ".yaml", ".yml", ".toml", ".txt", ".md",
        ".mp4", ".avi", ".mov", ".mkv", ".webm",
        ".gif", ".png", ".jpg", ".jpeg", ".webp",
    },
}

# ── LOGGING ───────────────────────────────────────────────────────────
def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, "sync_weights.log")
    logger = logging.getLogger("sync")
    logger.setLevel(logging.DEBUG)
    # File handler: full logs
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    # Console handler: INFO+ only
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ── FILE INDEXING ─────────────────────────────────────────────────────
def scan_tree(root: str, extensions: Set[str], logger: logging.Logger) -> Dict[str, Tuple[int, float]]:
    """
    Recursively scan a directory, return {relpath: (size, mtime)}.
    Only collects files matching the given extensions.
    """
    index: Dict[str, Tuple[int, float]] = {}
    if not os.path.isdir(root):
        return index
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in extensions:
                continue
            full = os.path.join(dirpath, fname)
            try:
                st = os.stat(full)
                rel = os.path.relpath(full, root)
                index[rel] = (st.st_size, st.st_mtime)
            except OSError as e:
                logger.warning("stat failed: %s (%s)", full, e)
    return index


def wait_stable(filepath: str, wait: float, retries: int, logger: logging.Logger) -> bool:
    """
    Wait for a file to stabilize (no longer being written).
    Returns True if size stays unchanged for `retries` consecutive checks,
    False if the file disappears or never stabilizes.
    """
    prev_size = -1
    for i in range(retries):
        time.sleep(wait)
        try:
            cur_size = os.path.getsize(filepath)
        except OSError:
            logger.debug("file gone while waiting: %s", filepath)
            return False
        if cur_size == prev_size and i > 0:
            return True
        prev_size = cur_size
    logger.debug("file never stabilized: %s (last_size=%d)", filepath, prev_size)
    return False

# ── SYNC LOGIC ────────────────────────────────────────────────────────
def sync_file(
    src_rel: str,
    src_base: str,
    dst_base: str,
    logger: logging.Logger,
) -> bool:
    """Copy a single file, auto-creating the destination directory tree."""
    src = os.path.join(src_base, src_rel)
    dst = os.path.join(dst_base, src_rel)
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        logger.info("COPIED %s", src_rel)
        return True
    except OSError as e:
        logger.error("copy failed: %s -> %s (%s)", src, dst, e)
        return False


def sync_loop(
    src_root: str,
    dst_root: str,
    watch_dirs: list,
    extensions: Set[str],
    interval: int,
    stable_wait: float,
    stable_retries: int,
    logger: logging.Logger,
):
    """
    Main loop: periodically scan source dirs, incrementally copy
    new/changed files to the destination.
    """
    prev_snapshots: Dict[str, Dict[str, Tuple[int, float]]] = {}

    logger.info("=" * 50)
    logger.info("Sync daemon started")
    logger.info("  Source:      %s", src_root)
    logger.info("  Destination: %s", dst_root)
    logger.info("  Interval:    %ds", interval)
    logger.info("  Watching:    %s", watch_dirs)
    logger.info("=" * 50)

    iteration = 0
    while True:
        iteration += 1
        loop_start = time.time()
        total_copied = 0

        for sub in watch_dirs:
            scan_path = os.path.join(src_root, sub) if sub else src_root
            if not os.path.isdir(scan_path):
                continue

            current = scan_tree(scan_path, extensions, logger)
            previous = prev_snapshots.get(sub, {})

            for rel, (size, mtime) in current.items():
                prev_info = previous.get(rel)

                if prev_info is None:
                    # New file -- wait for it to stabilize
                    logger.debug("new file detected: %s (size=%d)", rel, size)
                    full_src = os.path.join(scan_path, rel)
                    if wait_stable(full_src, stable_wait, stable_retries, logger):
                        if sync_file(rel, scan_path,
                                     dst_root if not sub else os.path.join(dst_root, sub),
                                     logger):
                            total_copied += 1
                    else:
                        logger.warning("skipped unstable file: %s", rel)

                elif size != prev_info[0] or mtime > prev_info[1] + 1:
                    # File changed (size differs or mtime updated by >1s)
                    logger.debug("changed file: %s (size %d->%d, mtime %.0f->%.0f)",
                                 rel, prev_info[0], size, prev_info[1], mtime)
                    full_src = os.path.join(scan_path, rel)
                    if wait_stable(full_src, stable_wait, stable_retries, logger):
                        if sync_file(rel, scan_path,
                                     dst_root if not sub else os.path.join(dst_root, sub),
                                     logger):
                            total_copied += 1
                    else:
                        logger.warning("skipped unstable file: %s", rel)

            prev_snapshots[sub] = current

        elapsed = time.time() - loop_start
        if total_copied > 0:
            logger.info("Iter #%d: %d file(s) synced (scan %.1fs)",
                        iteration, total_copied, elapsed)
        else:
            logger.debug("Iter #%d: no changes (scan %.1fs)", iteration, elapsed)

        sleep_time = max(1, interval - elapsed)
        time.sleep(sleep_time)

# ── ENTRY ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VideoX-Fun weight sync daemon")
    parser.add_argument("--interval", type=int, default=None,
                        help=f"Poll interval in seconds (default {CONFIG['poll_interval']})")
    parser.add_argument("--src", type=str, default=None,
                        help="Source directory (overrides config)")
    parser.add_argument("--dst", type=str, default=None,
                        help="Destination directory (overrides config)")
    parser.add_argument("--once", action="store_true",
                        help="Run one full sync then exit (no daemon loop)")
    args = parser.parse_args()

    src = args.src or CONFIG["src_root"]
    dst = args.dst or CONFIG["dst_root"]
    interval = args.interval or CONFIG["poll_interval"]
    log_dir = CONFIG["log_dir"]

    if not os.path.isdir(src):
        print(f"[FATAL] Source directory not found: {src}", file=sys.stderr)
        sys.exit(1)

    logger = setup_logging(log_dir)
    extensions = CONFIG["sync_extensions"]

    if args.once:
        logger.info("One-shot sync mode")
        for sub in CONFIG["watch_dirs"]:
            scan_path = os.path.join(src, sub) if sub else src
            if not os.path.isdir(scan_path):
                continue
            current = scan_tree(scan_path, extensions, logger)
            dst_sub = dst if not sub else os.path.join(dst, sub)
            count = 0
            for rel in current:
                full_dst = os.path.join(dst_sub, rel)
                if not os.path.exists(full_dst) or \
                   os.path.getsize(full_dst) != os.path.getsize(os.path.join(scan_path, rel)):
                    if sync_file(rel, scan_path, dst_sub, logger):
                        count += 1
            logger.info("Subdir '%s': %d file(s) synced", sub or ".", count)
        logger.info("One-shot sync complete")
    else:
        sync_loop(
            src_root=src,
            dst_root=dst,
            watch_dirs=CONFIG["watch_dirs"],
            extensions=extensions,
            interval=interval,
            stable_wait=CONFIG["stable_wait"],
            stable_retries=CONFIG["stable_retries"],
            logger=logger,
        )


if __name__ == "__main__":
    main()
