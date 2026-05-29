#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量同步脚本：监控 VideoX-Fun 训练输出目录，将新权重和 validation 样本
增量拷贝到安全存储路径。

设计原则：
- 轮询 + sleep，不占用 inotify fd，CPU 几乎为零
- 仅对比 size + mtime，不计算 hash（节省 IO）
- 检测文件是否还在写入中（两次采样 size 不变才算稳定）
- 支持 nohup 后台运行，所有输出走 log 文件

用法：
    python sync_weights.py [--interval 60]
    nohup python sync_weights.py --interval 120 > /dev/null 2>&1 &

配置：修改下方 CONFIG 字典即可（或通过环境变量覆盖）。
"""

import os
import sys
import time
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

# ── 配置 ──────────────────────────────────────────────────────────────
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
    # 轮询间隔（秒），权重通常几百步才存一次，120s 足够
    "poll_interval": int(os.environ.get("SYNC_INTERVAL", "120")),
    # 文件稳定检测：两次采样间隔 + 最大重试次数
    "stable_wait": 2.0,       # 等 2 秒再检查一次
    "stable_retries": 3,      # 连续 3 次 size 不变才算稳定
    # 需要监控的子目录（相对于 src_root）
    "watch_dirs": [
        "",                     # 根目录下的权重文件/checkpoint 目录
        "sample",
        "samples",
        "validation",
    ],
    # 需要同步的文件扩展名
    "sync_extensions": {
        ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx",
        ".json", ".yaml", ".yml", ".toml", ".txt", ".md",
        ".mp4", ".avi", ".mov", ".mkv", ".webm",
        ".gif", ".png", ".jpg", ".jpeg", ".webp",
    },
}

# ── 日志 ──────────────────────────────────────────────────────────────
def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, "sync_weights.log")
    logger = logging.getLogger("sync")
    logger.setLevel(logging.DEBUG)
    # 文件 handler：完整日志
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    # 控制台 handler：仅 INFO+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ── 文件索引 ──────────────────────────────────────────────────────────
def scan_tree(root: str, extensions: Set[str], logger: logging.Logger) -> Dict[str, Tuple[int, float]]:
    """
    递归扫描目录，返回 {相对路径: (size, mtime)} 的字典。
    只收集匹配扩展名的文件。
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
    等待文件写入完成：连续 retries 次采样 size 不变则返回 True。
    超时或文件消失返回 False。
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
            # 稳定
            return True
        prev_size = cur_size
    logger.debug("file never stabilized: %s (last_size=%d)", filepath, prev_size)
    return False

# ── 同步逻辑 ──────────────────────────────────────────────────────────
def sync_file(
    src_rel: str,
    src_base: str,
    dst_base: str,
    logger: logging.Logger,
) -> bool:
    """拷贝单个文件，自动创建目标目录。"""
    src = os.path.join(src_base, src_rel)
    dst = os.path.join(dst_base, src_rel)
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)  # copy2 保留 mtime
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
    主循环：定期扫描源目录，增量拷贝新/变化的文件到目标。
    """
    # 上次扫描的快照：{子目录 -> {相对路径: (size, mtime)}}
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
                    # 新文件 —— 等待写入稳定
                    logger.debug("new file detected: %s (size=%d)", rel, size)
                    full_src = os.path.join(scan_path, rel)
                    if wait_stable(full_src, stable_wait, stable_retries, logger):
                        if sync_file(rel, scan_path, dst_root if not sub else os.path.join(dst_root, sub), logger):
                            total_copied += 1
                    else:
                        logger.warning("skipped unstable file: %s", rel)

                elif size != prev_info[0] or mtime > prev_info[1] + 1:
                    # 文件变化（size 不同或 mtime 更新了 1s 以上）
                    logger.debug("changed file: %s (size %d->%d, mtime %.0f->%.0f)",
                                 rel, prev_info[0], size, prev_info[1], mtime)
                    full_src = os.path.join(scan_path, rel)
                    if wait_stable(full_src, stable_wait, stable_retries, logger):
                        if sync_file(rel, scan_path, dst_root if not sub else os.path.join(dst_root, sub), logger):
                            total_copied += 1
                    else:
                        logger.warning("skipped unstable file: %s", rel)

            # 更新快照
            prev_snapshots[sub] = current

        elapsed = time.time() - loop_start
        if total_copied > 0:
            logger.info("Iter #%d: %d file(s) synced (scan %.1fs)", iteration, total_copied, elapsed)
        else:
            logger.debug("Iter #%d: no changes (scan %.1fs)", iteration, elapsed)

        # 等待下次轮询
        sleep_time = max(1, interval - elapsed)
        time.sleep(sleep_time)

# ── 入口 ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VideoX-Fun 权重同步守护进程")
    parser.add_argument("--interval", type=int, default=None,
                        help=f"轮询间隔秒数（默认 {CONFIG['poll_interval']}）")
    parser.add_argument("--src", type=str, default=None,
                        help="源目录（覆盖配置）")
    parser.add_argument("--dst", type=str, default=None,
                        help="目标目录（覆盖配置）")
    parser.add_argument("--once", action="store_true",
                        help="只执行一次同步后退出（不走循环）")
    args = parser.parse_args()

    # 合并配置
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
                if not os.path.exists(full_dst) or os.path.getsize(full_dst) != os.path.getsize(os.path.join(scan_path, rel)):
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
