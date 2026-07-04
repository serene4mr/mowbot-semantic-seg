#!/usr/bin/env python3
"""
Convert COCO Segmentation dataset to Ultralytics YOLO Semantic Segmentation format
using ID-based overlap order (higher annotation IDs draw on top of lower ones).
"""

import os
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COCO dataset to Semantic Segmentation masks")
    parser.add_argument(
        "--src",
        type=str,
        default="data/japan-pear-orchard-260623-v1.0.coco-segmentation",
        help="Path to the source COCO segmentation dataset directory"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/japan-pear-orchard-semantic",
        help="Path to the destination semantic segmentation dataset directory"
    )
    parser.add_argument(
        "--bg-val",
        type=int,
        choices=[0, 255],
        default=255,
        help="Value for unannotated background pixels. If 255, they are ignored. If 0, they are treated as a valid 'background' class (other classes will shift by +1)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    bg_val = args.bg_val

    # 1. Verification of source paths
    if not src_root.exists():
        print(f"Error: Source directory {src_root} does not exist.")
        return

    print(f"Starting conversion from {src_root} to {dst_root} (bg_val={bg_val})...")

    # COCO Category ID to Grayscale Training ID mapping
    # If bg_val is 0, we insert 'background' at index 0 and shift other classes by +1
    if bg_val == 0:
        class_map = {
            1: 1, # cuttable
            2: 2, # foliage
            3: 3, # non-traversable
            4: 4  # traversable
        }
        yaml_names = {
            0: "background",
            1: "cuttable",
            2: "foliage",
            3: "non-traversable",
            4: "traversable"
        }
        pixel_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 255: 0}
    else:
        class_map = {
            1: 0, # cuttable
            2: 1, # foliage
            3: 2, # non-traversable
            4: 3  # traversable
        }
        yaml_names = {
            0: "cuttable",
            1: "foliage",
            2: "non-traversable",
            3: "traversable"
        }
        pixel_counts = {0: 0, 1: 0, 2: 0, 3: 0, 255: 0}

    # Split mapping: source folder name to target folder name
    split_map = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }

    # Stat counters
    stats = {}

    for src_split, dst_split in split_map.items():
        src_split_dir = src_root / src_split
        ann_file = src_split_dir / "_annotations.coco.json"

        if not ann_file.exists():
            print(f"Warning: Annotation file {ann_file} not found. Skipping split {src_split}.")
            continue

        print(f"Processing split: {src_split} -> {dst_split}...")
        
        # Create output directories
        dst_img_dir = dst_root / "images" / dst_split
        dst_mask_dir = dst_root / "masks" / dst_split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_mask_dir.mkdir(parents=True, exist_ok=True)

        # Load COCO annotations
        with open(ann_file, "r") as f:
            coco_data = json.load(f)

        # Build category map for debug/logs
        categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
        print(f"  Categories found in COCO: {categories}")

        # Group annotations by image_id
        annotations_by_img = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in annotations_by_img:
                annotations_by_img[img_id] = []
            annotations_by_img[img_id].append(ann)

        # Process each image
        img_count = 0
        current_pixel_counts = pixel_counts.copy()

        for img_entry in coco_data.get("images", []):
            img_id = img_entry["id"]
            file_name = img_entry["file_name"]
            width = img_entry["width"]
            height = img_entry["height"]

            src_img_path = src_split_dir / file_name
            if not src_img_path.exists():
                print(f"  Warning: Image file {src_img_path} not found. Skipping.")
                continue

            # Create blank mask with selected background value (0 or 255)
            mask = np.full((height, width), bg_val, dtype=np.uint8)

            # Get annotations for this image
            anns = annotations_by_img.get(img_id, [])

            # Sort strictly by annotation ID ascending:
            # higher IDs (appear later) will paint on top and override lower IDs (appear earlier)
            anns.sort(key=lambda x: x["id"])

            # Paint each annotation polygon
            for ann in anns:
                coco_cat_id = ann["category_id"]
                if coco_cat_id not in class_map:
                    continue
                
                train_class_id = class_map[coco_cat_id]
                segmentations = ann.get("segmentation", [])

                if isinstance(segmentations, list):
                    for seg in segmentations:
                        # Convert flat list of coords [x1, y1, x2, y2, ...] to poly points
                        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(mask, [poly], train_class_id)

            # Save the single-channel Grayscale PNG mask
            mask_file_name = Path(file_name).with_suffix(".png").name
            dst_mask_path = dst_mask_dir / mask_file_name
            
            # Save using PIL to guarantee proper Grayscale (L mode) encoding
            Image.fromarray(mask).save(dst_mask_path)

            # Copy original image to the destination images directory
            dst_img_path = dst_img_dir / file_name
            shutil.copy2(src_img_path, dst_img_path)

            # Update stats
            unique, counts = np.unique(mask, return_counts=True)
            for val, count in zip(unique, counts):
                if val in current_pixel_counts:
                    current_pixel_counts[val] += count

            img_count += 1

        print(f"  Processed {img_count} images for split {dst_split}.")
        stats[dst_split] = {
            "images": img_count,
            "pixels": current_pixel_counts
        }

    # 2. Write dataset.yaml configuration file
    yaml_names_str = "\n".join([f"  {k}: {v}" for k, v in yaml_names.items()])
    yaml_content = f"""# Ultralytics Semantic Segmentation Dataset config
# Converted from COCO segmentation format with ID-based overlap

path: {dst_root.resolve()}
train: images/train
val: images/val
test: images/test
masks_dir: masks

names:
{yaml_names_str}
"""
    yaml_path = dst_root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"\nWritten Ultralytics dataset config to {yaml_path}")

    # 3. Print Summary Stats
    print("\n================ CONVERSION SUMMARY ================")
    
    # Display labels naming mapping based on bg_val
    name_map = {}
    for k, v in yaml_names.items():
        name_map[k] = f"{v} ({k})"
    name_map[255] = "ignore (255)"

    for split, data in stats.items():
        print(f"\nSplit: {split}")
        print(f"  Images converted: {data['images']}")
        print("  Pixel distribution:")
        total_pixels = sum(data["pixels"].values())
        if total_pixels > 0:
            for val, count in data["pixels"].items():
                if count > 0 or val in yaml_names or val == 255:
                    name = name_map.get(val, f"unknown ({val})")
                    pct = (count / total_pixels) * 100
                    print(f"    - {name:<20}: {count:>10} pixels ({pct:>5.2f}%)")

if __name__ == "__main__":
    main()
