#!/usr/bin/env python3
"""
Generate dataset JSON for VideoX-Fun control training.

Directory structure:
    gt_root/
        subdir1/
            video1.mp4
            video2.mp4
        subdir2/
            video3.mp4
    
    control_root/
        subdir1/
            video1.mp4  # same name as gt_root/subdir1/video1.mp4
            video2.mp4
        subdir2/
            video3.mp4

Caption file (JSON):
    {
        "video1": "A dog running in the park",
        "video2": "A cat sleeping on the sofa",
        ...
    }
    Keys can be filename with or without extension.

Output JSON format (compatible with ImageVideoControlDataset):
    [
        {
            "file_path": "subdir1/video1.mp4",
            "control_file_path": "subdir1/video1.mp4",
            "text": "A dog running in the park",
            "type": "video"
        },
        ...
    ]

Usage:
    python scripts/generate_dataset_json.py \
        --gt_root /path/to/gt_videos \
        --control_root /path/to/control_signals \
        --caption_file /path/to/captions.json \
        --output dataset.json

    # Enable debug output:
    DEBUG=1 python scripts/generate_dataset_json.py \
        --gt_root ... --control_root ... --caption_file ... --output ...
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}

# Debug flag: set DEBUG=1 environment variable to enable verbose output
DEBUG = os.environ.get("DEBUG", "0") == "1"


def debug_log(msg: str):
    """Print debug message if DEBUG mode is enabled."""
    if DEBUG:
        print(f"[DEBUG] {msg}")


def validate_video_file(path: str) -> bool:
    """Check if video file exists and is non-empty."""
    if not os.path.exists(path):
        debug_log(f"File does not exist: {path}")
        return False
    size = os.path.getsize(path)
    if size == 0:
        debug_log(f"File is empty (0 bytes): {path}")
        return False
    debug_log(f"OK: {path} ({size} bytes)")
    return True


def find_videos(root_dir: str, label: str = "videos") -> Dict[str, str]:
    """
    Recursively find all videos under root_dir.
    
    Returns:
        Dict mapping relative path (without extension) to full path
        e.g. {"subdir1/video1": "/path/to/root/subdir1/video1.mp4"}
    """
    videos = {}
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"[ERROR] {label} directory not found: {root_dir}")
        sys.exit(1)
    
    print(f"Scanning {label} in {root_dir} ...")
    for ext in VIDEO_EXTENSIONS:
        for video_file in root_path.rglob(f"*{ext}"):
            rel_path = video_file.relative_to(root_path)
            rel_path_no_ext = rel_path.with_suffix('')
            key = str(rel_path_no_ext)
            videos[key] = str(video_file)
            debug_log(f"Found {label}: {key} -> {video_file}")
    
    print(f"  Found {len(videos)} {label}")
    debug_log(f"Total {label} found under {root_dir}: {len(videos)}")
    return videos


def load_captions(caption_file: str) -> Dict[str, str]:
    """
    Load captions from JSON file.
    
    Supports two formats:
    
    1. Dict format — keys can include path or be plain filename, with or without extension:
        {"subdir/video1": "..."} or {"video1.mp4": "..."}
    
    2. List format — each item is an object with caption text and a path key:
        [
            {
                "video_path": "/abs/path/to/video.mp4",
                "relative_path": "subdir/video.mp4",
                "video_name": "video.mp4",
                "caption": "A dog running..."
            },
            ...
        ]
    
    Keys are stored in two forms for flexible matching:
        - relative path without extension  (e.g. "subdir/video1")
        - filename stem only               (e.g. "video1")
    Both are added so match_videos can look up either way.
    """
    if not os.path.exists(caption_file):
        print(f"[ERROR] Caption file not found: {caption_file}")
        sys.exit(1)
    
    with open(caption_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    normalized: Dict[str, str] = {}
    
    if isinstance(raw, list):
        # List-of-objects format
        for item in raw:
            caption_text = item.get("caption", "")
            if not caption_text:
                continue
            
            # Build multiple lookup keys for flexible matching
            # 1) relative_path without extension (preferred)
            rel = item.get("relative_path", "")
            if rel:
                rel_key = str(Path(rel).with_suffix(''))
                normalized[rel_key] = caption_text
                # Also store the filename-stem-only form as fallback
                stem_key = Path(rel).stem
                if stem_key != rel_key:
                    normalized.setdefault(stem_key, caption_text)
            
            # 2) video_name (filename with ext) → stem
            vname = item.get("video_name", "")
            if vname:
                stem_key = Path(vname).stem
                normalized.setdefault(stem_key, caption_text)
            
            # 3) video_path basename as final fallback
            vpath = item.get("video_path", "")
            if vpath:
                stem_key = Path(vpath).stem
                normalized.setdefault(stem_key, caption_text)
            
            debug_log(f"Caption: {rel_key if rel else stem_key} -> {caption_text[:50]}{'...' if len(caption_text) > 50 else ''}")
        
        print(f"  Loaded {len(normalized)} caption entries (from {len(raw)} items)")
    
    elif isinstance(raw, dict):
        # Dict format
        for key, value in raw.items():
            # Store as-is without extension (preserves directory parts)
            key_no_ext = str(Path(key).with_suffix(''))
            normalized[key_no_ext] = value
            # Also store filename-stem-only as fallback
            stem_key = Path(key).stem
            if stem_key != key_no_ext:
                normalized.setdefault(stem_key, value)
            debug_log(f"Caption: {key_no_ext} -> {value[:50]}{'...' if len(value) > 50 else ''}")
        
        print(f"  Loaded {len(normalized)} caption entries (from {len(raw)} dict keys)")
    
    else:
        print(f"[ERROR] Unsupported caption format: {type(raw).__name__}. Expected list or dict.")
        sys.exit(1)
    
    return normalized


def match_videos(
    gt_videos: Dict[str, str],
    control_videos: Dict[str, str],
    captions: Dict[str, str],
    use_relative_paths: bool = True,
) -> List[Dict]:
    """
    Match GT videos with control videos and captions.
    
    Returns:
        List of dataset entries
    """
    dataset = []
    matched = 0
    skipped_no_control = []
    skipped_no_caption = []
    skipped_invalid = []
    
    for key, gt_path in gt_videos.items():
        # Check if control video exists
        if key not in control_videos:
            skipped_no_control.append(key)
            continue
        
        # Check if caption exists
        if key not in captions:
            skipped_no_caption.append(key)
            continue
        
        control_path = control_videos[key]
        
        # Validate files exist and are non-empty
        if not validate_video_file(gt_path):
            skipped_invalid.append(f"{key} (GT)")
            continue
        if not validate_video_file(control_path):
            skipped_invalid.append(f"{key} (control)")
            continue
        
        if use_relative_paths:
            file_path = f"{key}{Path(gt_path).suffix}"
            control_file_path = f"{key}{Path(control_path).suffix}"
        else:
            file_path = gt_path
            control_file_path = control_path
        
        entry = {
            "file_path": file_path,
            "control_file_path": control_file_path,
            "text": captions[key],
            "type": "video"
        }
        
        dataset.append(entry)
        matched += 1
        debug_log(f"Matched: {key}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Matching Summary:")
    print(f"  Total GT videos:      {len(gt_videos)}")
    print(f"  Total control videos: {len(control_videos)}")
    print(f"  Total captions:       {len(captions)}")
    print(f"  Matched:              {matched}")
    print(f"  Skipped (no control): {len(skipped_no_control)}")
    print(f"  Skipped (no caption): {len(skipped_no_caption)}")
    print(f"  Skipped (invalid file): {len(skipped_invalid)}")
    
    if skipped_no_control:
        print(f"\n  Missing control videos:")
        for key in skipped_no_control[:10]:
            print(f"    - {key}")
        if len(skipped_no_control) > 10:
            print(f"    ... and {len(skipped_no_control) - 10} more")
    
    if skipped_no_caption:
        print(f"\n  Missing captions:")
        for key in skipped_no_caption[:10]:
            print(f"    - {key}")
        if len(skipped_no_caption) > 10:
            print(f"    ... and {len(skipped_no_caption) - 10} more")
    
    if skipped_invalid:
        print(f"\n  Invalid files:")
        for key in skipped_invalid[:10]:
            print(f"    - {key}")
        if len(skipped_invalid) > 10:
            print(f"    ... and {len(skipped_invalid) - 10} more")
    
    print(f"{'='*60}\n")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset JSON for VideoX-Fun control training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--gt_root", type=str, required=True,
        help="Root directory containing GT videos (with subdirectories)"
    )
    parser.add_argument(
        "--control_root", type=str, required=True,
        help="Root directory containing control signal videos (with subdirectories)"
    )
    parser.add_argument(
        "--caption_file", type=str, required=True,
        help="JSON file mapping video names to captions"
    )
    parser.add_argument(
        "--output", type=str, default="dataset.json",
        help="Output JSON file path (default: dataset.json)"
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Data root prefix for relative paths (optional, passed to dataset)"
    )
    parser.add_argument(
        "--absolute_paths", action="store_true",
        help="Use absolute paths instead of relative paths"
    )
    
    args = parser.parse_args()
    
    print(f"GT root:       {args.gt_root}")
    print(f"Control root:  {args.control_root}")
    print(f"Caption file:  {args.caption_file}")
    print(f"Output:        {args.output}")
    print(f"Path mode:     {'absolute' if args.absolute_paths else 'relative'}")
    if DEBUG:
        print(f"Debug mode:    ON")
    
    # Step 1: Find videos
    print("\n[Step 1] Finding GT videos...")
    gt_videos = find_videos(args.gt_root, "GT videos")
    
    print("\n[Step 2] Finding control videos...")
    control_videos = find_videos(args.control_root, "control videos")
    
    # Step 2: Load captions
    print("\n[Step 3] Loading captions...")
    captions = load_captions(args.caption_file)
    
    # Step 3: Match and generate dataset
    print("\n[Step 4] Matching videos...")
    dataset = match_videos(
        gt_videos=gt_videos,
        control_videos=control_videos,
        captions=captions,
        use_relative_paths=not args.absolute_paths,
    )
    
    # Step 4: Validate
    if not dataset:
        print("\n[ERROR] No valid entries generated!")
        print("  Check the following:")
        print("    1. GT and control directories have matching subdirectory structure")
        print("    2. Video files have the same name in both GT and control directories")
        print("    3. Caption file keys match video filenames (with or without extension)")
        sys.exit(1)
    
    # Sanity check: print first 3 entries
    print("\n[Step 5] Sample entries (first 3):")
    for i, entry in enumerate(dataset[:3]):
        print(f"  [{i}] file_path:         {entry['file_path']}")
        print(f"      control_file_path: {entry['control_file_path']}")
        text_preview = entry['text'][:80]
        print(f"      text:              {text_preview}{'...' if len(entry['text']) > 80 else ''}")
    
    # Step 5: Save
    print(f"\n[Step 6] Saving to {args.output} ...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Done! Saved {len(dataset)} entries to {args.output}")
    print(f"{'='*60}")
    
    # Verification hint
    print(f"\nTo verify:")
    print(f'  python -c "import json; d=json.load(open(\'{args.output}\')); print(len(d), \'entries\'); import pprint; pprint.pprint(d[0])"')
    
    if DEBUG:
        print(f"\n[DEBUG] Full matched entries:")
        for entry in dataset:
            print(f"  {entry['file_path']}  <->  {entry['control_file_path']}")


if __name__ == "__main__":
    main()
