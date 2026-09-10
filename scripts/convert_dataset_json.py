#!/usr/bin/env python3
"""
Convert dataset json from old format to new format with mask_file_path.

Old format:
{
    "video_path": "/mnt/DataPart/.../static/xxx.mp4",
    "relative_path": "rendering_dataset_food_20260106/xxx.mp4",
    "video_name": "xxx.mp4",
    "caption": "...",
    "gpu_rank": 0
}

New format:
{
    "file_path": "static/xxx.mp4",
    "control_file_path": "static_gs_render_single_image/xxx.mp4",
    "mask_file_path": "static_gs_render_single_image_masks/xxx.mp4",
    "text": "...",
    "type": "video"
}
"""

import json
import argparse
from pathlib import Path


def convert_dataset(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted = []
    for item in data:
        relative_path = item.get('relative_path', '')
        caption = item.get('caption', '')
        
        # Mask files have '_mask' suffix before extension: xxx.mp4 -> xxx_mask.mp4
        p = Path(relative_path)
        mask_relative_path = str(p.parent / f"{p.stem}_mask{p.suffix}")
        
        new_item = {
            "file_path": f"static/{relative_path}",
            "control_file_path": f"static_gs_render_single_image/{relative_path}",
            "mask_file_path": f"static_gs_render_single_image_masks/{mask_relative_path}",
            "text": caption,
            "type": "video"
        }
        converted.append(new_item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(converted)} entries")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, 
                        default="/mnt/DataPart/jianghongda/dataset/livephoto/static_captions.json",
                        help="Input json path")
    parser.add_argument("--output", type=str,
                        default="/mnt/DataPart/jianghongda/dataset/livephoto/static_captions_new.json",
                        help="Output json path")
    args = parser.parse_args()
    
    convert_dataset(args.input, args.output)
