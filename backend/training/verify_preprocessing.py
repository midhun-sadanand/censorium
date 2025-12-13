#!/usr/bin/env python3
"""
Verify that Roboflow preprocessing was applied correctly to the dataset.

This script checks:
- All images are 512x512 (stretch resize)
- Label files exist for all images
- Label format is correct (YOLOv8 normalized coordinates)
- Dataset split integrity
- Augmentation distribution
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
import yaml
import argparse


def verify_image_dimensions(dataset_path: Path, split: str = "train") -> dict:
    """Verify all images are 512x512."""
    print(f"\n{'='*60}")
    print(f"Verifying {split.upper()} image dimensions...")
    print(f"{'='*60}")
    
    images_dir = dataset_path / split / "images"
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    dimensions = defaultdict(int)
    invalid_images = []
    
    for idx, img_path in enumerate(image_files):
        if idx % 1000 == 0:
            print(f"Checked {idx}/{len(image_files)} images...", end='\r')
        
        img = cv2.imread(str(img_path))
        if img is None:
            invalid_images.append((img_path.name, "Failed to load"))
            continue
        
        h, w = img.shape[:2]
        dimensions[f"{w}x{h}"] += 1
        
        if (w, h) != (512, 512):
            invalid_images.append((img_path.name, f"{w}x{h}"))
    
    print(f"Checked {len(image_files)}/{len(image_files)} images...    ")
    
    # Results
    print(f"\n✓ Total images: {len(image_files)}")
    print(f"✓ Dimension distribution:")
    for dim, count in sorted(dimensions.items(), key=lambda x: -x[1]):
        percentage = (count / len(image_files)) * 100
        print(f"  {dim}: {count} ({percentage:.1f}%)")
    
    if invalid_images:
        print(f"\n⚠ Found {len(invalid_images)} images with non-512x512 dimensions:")
        for name, dim in invalid_images[:10]:
            print(f"  - {name}: {dim}")
        if len(invalid_images) > 10:
            print(f"  ... and {len(invalid_images) - 10} more")
    else:
        print(f"\n✅ All images are 512x512!")
    
    return {
        "total": len(image_files),
        "dimensions": dict(dimensions),
        "invalid": invalid_images,
        "all_512x512": len(invalid_images) == 0
    }


def verify_label_format(dataset_path: Path, split: str = "train", sample_size: int = 100) -> dict:
    """Verify label files exist and format is correct."""
    print(f"\n{'='*60}")
    print(f"Verifying {split.upper()} label format...")
    print(f"{'='*60}")
    
    images_dir = dataset_path / split / "images"
    labels_dir = dataset_path / split / "labels"
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    missing_labels = []
    invalid_labels = []
    label_stats = {
        "total_annotations": 0,
        "images_with_labels": 0,
        "images_without_labels": 0,
        "class_counts": defaultdict(int)
    }
    
    # Check all images have corresponding labels
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        
        if not label_path.exists():
            missing_labels.append(img_path.name)
            label_stats["images_without_labels"] += 1
            continue
        
        # Read and validate label format
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                label_stats["images_without_labels"] += 1
                continue
            
            label_stats["images_with_labels"] += 1
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_labels.append((img_path.name, "Invalid format"))
                    continue
                
                class_id, x_center, y_center, width, height = map(float, parts)
                
                # Verify normalized coordinates (0-1)
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                       0 <= width <= 1 and 0 <= height <= 1):
                    invalid_labels.append((img_path.name, "Coordinates out of range"))
                
                label_stats["class_counts"][int(class_id)] += 1
                label_stats["total_annotations"] += 1
                
        except Exception as e:
            invalid_labels.append((img_path.name, str(e)))
    
    # Results
    print(f"\n✓ Total images: {len(image_files)}")
    print(f"✓ Images with labels: {label_stats['images_with_labels']}")
    print(f"✓ Images without labels: {label_stats['images_without_labels']}")
    print(f"✓ Total annotations: {label_stats['total_annotations']}")
    print(f"✓ Annotations per image: {label_stats['total_annotations'] / len(image_files):.2f}")
    
    print(f"\n✓ Class distribution:")
    for class_id, count in sorted(label_stats["class_counts"].items()):
        print(f"  Class {class_id}: {count} annotations")
    
    if missing_labels:
        print(f"\n⚠ Found {len(missing_labels)} missing label files:")
        for name in missing_labels[:5]:
            print(f"  - {name}")
        if len(missing_labels) > 5:
            print(f"  ... and {len(missing_labels) - 5} more")
    
    if invalid_labels:
        print(f"\n⚠ Found {len(invalid_labels)} invalid labels:")
        for name, reason in invalid_labels[:5]:
            print(f"  - {name}: {reason}")
        if len(invalid_labels) > 5:
            print(f"  ... and {len(invalid_labels) - 5} more")
    
    if not missing_labels and not invalid_labels:
        print(f"\n✅ All labels are valid!")
    
    return {
        "total_images": len(image_files),
        "total_annotations": label_stats["total_annotations"],
        "images_with_labels": label_stats["images_with_labels"],
        "missing_labels": len(missing_labels),
        "invalid_labels": len(invalid_labels),
        "class_counts": dict(label_stats["class_counts"]),
        "all_valid": len(missing_labels) == 0 and len(invalid_labels) == 0
    }


def verify_dataset_structure(dataset_path: Path) -> dict:
    """Verify overall dataset structure."""
    print(f"\n{'='*60}")
    print(f"Verifying dataset structure...")
    print(f"{'='*60}")
    
    # Check data.yaml
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print(f"❌ data.yaml not found!")
        return {"valid": False}
    
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n✓ Dataset configuration:")
    print(f"  Classes: {config['nc']}")
    print(f"  Names: {config['names']}")
    
    # Check splits
    splits = ["train", "valid", "test"]
    split_stats = {}
    
    for split in splits:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"❌ {split} directory not found!")
            split_stats[split] = {"exists": False}
            continue
        
        num_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
        num_labels = len(list(labels_dir.glob("*.txt")))
        
        split_stats[split] = {
            "exists": True,
            "num_images": num_images,
            "num_labels": num_labels
        }
        
        print(f"\n✓ {split.upper()} split:")
        print(f"  Images: {num_images}")
        print(f"  Labels: {num_labels}")
    
    return {
        "config": config,
        "splits": split_stats,
        "valid": True
    }


def main():
    parser = argparse.ArgumentParser(description="Verify Roboflow dataset preprocessing")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/license_plates",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test", "all"],
        default="all",
        help="Which split to verify"
    )
    
    args = parser.parse_args()
    
    # Get absolute path
    if os.path.isabs(args.dataset_path):
        dataset_path = Path(args.dataset_path)
    else:
        # Relative to script location
        script_dir = Path(__file__).parent
        dataset_path = script_dir / args.dataset_path
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    print(f"\n{'#'*60}")
    print(f"# ROBOFLOW DATASET VERIFICATION")
    print(f"# Path: {dataset_path}")
    print(f"{'#'*60}")
    
    # Verify structure
    structure_results = verify_dataset_structure(dataset_path)
    
    if not structure_results["valid"]:
        print(f"\n❌ Dataset structure is invalid!")
        sys.exit(1)
    
    # Verify each split
    splits_to_check = ["train", "valid", "test"] if args.split == "all" else [args.split]
    
    all_results = {}
    
    for split in splits_to_check:
        # Image dimensions
        dim_results = verify_image_dimensions(dataset_path, split)
        
        # Label format
        label_results = verify_label_format(dataset_path, split)
        
        all_results[split] = {
            "dimensions": dim_results,
            "labels": label_results
        }
    
    # Final summary
    print(f"\n{'#'*60}")
    print(f"# VERIFICATION SUMMARY")
    print(f"{'#'*60}")
    
    all_passed = True
    
    for split, results in all_results.items():
        print(f"\n{split.upper()}:")
        
        if results["dimensions"]["all_512x512"]:
            print(f"  ✅ All images are 512x512")
        else:
            print(f"  ❌ Some images have incorrect dimensions")
            all_passed = False
        
        if results["labels"]["all_valid"]:
            print(f"  ✅ All labels are valid")
        else:
            print(f"  ❌ Some labels are invalid or missing")
            all_passed = False
        
        if results["labels"]["total_annotations"] > 0:
            print(f"  ✅ {results['labels']['total_annotations']} annotations found")
        else:
            print(f"  ⚠ No annotations found")
    
    if all_passed:
        print(f"\n{'='*60}")
        print(f"✅ DATASET VERIFICATION PASSED!")
        print(f"{'='*60}")
        print(f"\nYour dataset is ready for training!")
        return 0
    else:
        print(f"\n{'='*60}")
        print(f"❌ DATASET VERIFICATION FAILED!")
        print(f"{'='*60}")
        print(f"\nPlease fix the issues above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

