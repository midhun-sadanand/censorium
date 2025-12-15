#!/usr/bin/env python3
"""
Roboflow Preprocessing Pipeline Replication

This script replicates the exact preprocessing and augmentation steps that Roboflow
applied to create the training dataset. It processes raw images and labels into
the final format used for training.

Preprocessing Steps (Applied to ALL images):
1. Auto-Orient: Correct EXIF rotation metadata
2. Resize: Stretch to 512×512 pixels (ignore aspect ratio)
3. Auto-Adjust Contrast: Histogram equalization for better visibility

Augmentation Steps (Applied to TRAINING set only, 3x multiplier):
- Brightness variation: -15% to +15%
- Creates 3 versions: original, darker (-15%), brighter (+15%)
- Results in 3x training images (e.g., 7,081 → 21,243)

Input Format:
- Raw images (any size, any format)
- YOLO format labels (.txt files with normalized coordinates)

Output Format:
- All images: 512×512 pixels
- Same label format (coordinates adjusted if needed)
- Directory structure: {split}/images/ and {split}/labels/
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import random


def auto_orient(image_path: Path) -> np.ndarray:
    """
    Apply auto-orientation based on EXIF data.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Properly oriented image as numpy array (BGR)
    """
    # Use PIL to handle EXIF orientation
    pil_image = Image.open(image_path)
    
    # Apply EXIF orientation if present
    try:
        from PIL import ImageOps
        pil_image = ImageOps.exif_transpose(pil_image)
    except Exception:
        pass  # No EXIF data or already oriented
    
    # Convert to numpy array (RGB)
    image_rgb = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    if len(image_rgb.shape) == 3:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    else:
        # Grayscale - convert to BGR
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
    
    return image_bgr


def resize_stretch(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Resize image to target size using stretch (ignore aspect ratio).
    
    This matches Roboflow's "Stretch" resize mode.
    
    Args:
        image: Input image (BGR)
        target_size: Target (width, height)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def auto_adjust_contrast(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization for contrast adjustment.
    
    This matches Roboflow's "Auto-Adjust Contrast" using histogram equalization.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Contrast-adjusted image
    """
    # Convert to YUV color space
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Apply histogram equalization to Y channel (luminance)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    
    # Convert back to BGR
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    return result


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust image brightness by a factor.
    
    Args:
        image: Input image (BGR)
        factor: Brightness factor (1.0 = no change, 0.85 = -15%, 1.15 = +15%)
        
    Returns:
        Brightness-adjusted image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust V channel (brightness)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    
    # Clip to valid range
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    # Convert back to BGR
    hsv = hsv.astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return result


def preprocess_image(image_path: Path, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Apply full preprocessing pipeline to an image.
    
    Steps:
    1. Auto-orient based on EXIF
    2. Resize to 512×512 (stretch)
    3. Auto-adjust contrast (histogram equalization)
    
    Args:
        image_path: Path to input image
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed image
    """
    # Step 1: Auto-orient
    image = auto_orient(image_path)
    
    # Step 2: Resize (stretch to target size)
    image = resize_stretch(image, target_size)
    
    # Step 3: Auto-adjust contrast
    image = auto_adjust_contrast(image)
    
    return image


def copy_label(label_path: Path, output_path: Path):
    """
    Copy label file to output location.
    
    Note: Since we're using stretch resize to 512×512, the normalized coordinates
    in YOLO format remain valid (they're already 0-1 normalized).
    
    Args:
        label_path: Source label file
        output_path: Destination label file
    """
    if label_path.exists():
        shutil.copy2(label_path, output_path)


def generate_augmentations(
    image: np.ndarray,
    brightness_range: Tuple[float, float] = (0.85, 1.15),
    num_augmentations: int = 2
) -> List[np.ndarray]:
    """
    Generate brightness augmentation variations.
    
    Roboflow applies -15% to +15% brightness with 3x multiplier:
    - Original image (100%)
    - Darker version (85%)
    - Brighter version (115%)
    
    Args:
        image: Preprocessed image
        brightness_range: (min_factor, max_factor) - default (0.85, 1.15) = -15% to +15%
        num_augmentations: Number of augmented versions to create (default 2)
        
    Returns:
        List of augmented images (not including original)
    """
    augmented = []
    
    min_factor, max_factor = brightness_range
    
    # Generate brightness factors
    if num_augmentations == 2:
        # Fixed: one darker, one brighter
        factors = [min_factor, max_factor]
    else:
        # Random factors in range
        factors = [random.uniform(min_factor, max_factor) for _ in range(num_augmentations)]
    
    for factor in factors:
        augmented_image = adjust_brightness(image, factor)
        augmented.append(augmented_image)
    
    return augmented


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    split: str = "train",
    apply_augmentation: bool = False,
    num_augmentations: int = 2,
    target_size: Tuple[int, int] = (512, 512)
):
    """
    Process a dataset split with preprocessing and optional augmentation.
    
    Args:
        input_dir: Input directory containing images/ and labels/
        output_dir: Output directory (will create images/ and labels/)
        split: Split name (train/valid/test)
        apply_augmentation: Whether to apply augmentation (only for train)
        num_augmentations: Number of augmented versions per image
        target_size: Target image size (width, height)
    """
    print(f"\n{'='*70}")
    print(f"Processing {split.upper()} split")
    print(f"{'='*70}")
    
    # Input paths
    input_images_dir = input_dir / "images"
    input_labels_dir = input_dir / "labels"
    
    # Output paths
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(input_images_dir.glob(ext)))
    
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images")
    print(f"Augmentation: {'Enabled' if apply_augmentation else 'Disabled'}")
    if apply_augmentation:
        print(f"  Creating {num_augmentations + 1}x images (original + {num_augmentations} augmented)")
    
    total_output = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc=f"Processing {split}"):
        # Get corresponding label path
        label_path = input_labels_dir / (img_path.stem + ".txt")
        
        # Preprocess image
        try:
            preprocessed_image = preprocess_image(img_path, target_size)
        except Exception as e:
            print(f"\nWarning: error processing {img_path.name}: {e}")
            continue
        
        # Save original preprocessed image
        output_img_path = output_images_dir / img_path.name
        cv2.imwrite(str(output_img_path), preprocessed_image)
        
        # Copy label
        output_label_path = output_labels_dir / (img_path.stem + ".txt")
        copy_label(label_path, output_label_path)
        
        total_output += 1
        
        # Apply augmentation if enabled
        if apply_augmentation:
            augmented_images = generate_augmentations(
                preprocessed_image,
                brightness_range=(0.85, 1.15),
                num_augmentations=num_augmentations
            )
            
            for idx, aug_img in enumerate(augmented_images):
                # Generate unique filename
                aug_img_name = f"{img_path.stem}_aug{idx+1}{img_path.suffix}"
                aug_img_path = output_images_dir / aug_img_name
                
                # Save augmented image
                cv2.imwrite(str(aug_img_path), aug_img)
                
                # Copy label with new name
                aug_label_path = output_labels_dir / f"{img_path.stem}_aug{idx+1}.txt"
                copy_label(label_path, aug_label_path)
                
                total_output += 1
    
    print(f"\n{split.upper()} complete:")
    print(f"   Input: {len(image_files)} images")
    print(f"   Output: {total_output} images")
    if apply_augmentation:
        print(f"   Multiplier: {total_output / len(image_files):.1f}x")


def create_data_yaml(output_dir: Path, class_names: List[str]):
    """
    Create data.yaml configuration file for YOLOv8.
    
    Args:
        output_dir: Output directory containing train/valid/test splits
        class_names: List of class names
    """
    data_yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images

nc: {len(class_names)}
names: {class_names}

roboflow:
  workspace: censorium
  project: license-plate-recognition-preprocessed
  version: 1
  license: CC BY 4.0
"""
    
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\nCreated data.yaml at {data_yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Replicate Roboflow preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing raw dataset (with train/valid/test subdirs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/license_plates_processed",
        help="Output directory for preprocessed dataset"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=512,
        help="Target image size (will be resized to SIZE×SIZE)"
    )
    parser.add_argument(
        "--augment-train",
        action="store_true",
        default=True,
        help="Apply augmentation to training set (3x multiplier)"
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=2,
        help="Number of augmented versions per training image (default 2 = 3x total)"
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs='+',
        default=["License_Plate"],
        help="Class names for data.yaml"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs='+',
        default=["train", "valid", "test"],
        help="Dataset splits to process"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_size = (args.target_size, args.target_size)
    
    print(f"\n{'#'*70}")
    print(f"# ROBOFLOW PREPROCESSING PIPELINE")
    print(f"{'#'*70}")
    print(f"\nInput Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Size: {target_size[0]}×{target_size[1]}")
    print(f"\nPreprocessing Steps:")
    print(f"  1. Auto-Orient (EXIF correction)")
    print(f"  2. Resize (Stretch to {target_size[0]}×{target_size[1]})")
    print(f"  3. Auto-Adjust Contrast (Histogram Equalization)")
    print(f"\nAugmentation:")
    print(f"  Training Set: {'Enabled' if args.augment_train else 'Disabled'}")
    if args.augment_train:
        print(f"  Brightness Range: -15% to +15%")
        print(f"  Multiplier: {args.num_augmentations + 1}x (original + {args.num_augmentations} augmented)")
    
    # Validate input directory
    if not input_dir.exists():
        print(f"\nError: input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in args.splits:
        input_split_dir = input_dir / split
        output_split_dir = output_dir / split
        
        if not input_split_dir.exists():
            print(f"\nWarning: {split} split not found, skipping")
            continue
        
        # Check if images and labels directories exist
        if not (input_split_dir / "images").exists():
            print(f"\nWarning: {split}/images not found, skipping")
            continue
        
        # Apply augmentation only to training set
        apply_augmentation = (split == "train" and args.augment_train)
        
        process_dataset(
            input_split_dir,
            output_split_dir,
            split=split,
            apply_augmentation=apply_augmentation,
            num_augmentations=args.num_augmentations,
            target_size=target_size
        )
    
    # Create data.yaml
    create_data_yaml(output_dir, args.class_names)
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE.")
    print(f"{'='*70}")
    print(f"\nOutput Location: {output_dir.absolute()}")
    print(f"\nDirectory Structure:")
    print(f"  {output_dir}/")
    print(f"    ├── data.yaml")
    for split in args.splits:
        if (output_dir / split).exists():
            num_images = len(list((output_dir / split / "images").glob("*.*")))
            num_labels = len(list((output_dir / split / "labels").glob("*.txt")))
            print(f"    ├── {split}/")
            print(f"    │   ├── images/ ({num_images} files)")
            print(f"    │   └── labels/ ({num_labels} files)")
    
    print(f"\nNext Steps:")
    print(f"  1. Verify dataset: python verify_preprocessing.py --dataset-path {output_dir}")
    print(f"  2. Train model: yolo detect train data={output_dir}/data.yaml model=yolov8n.pt")


if __name__ == "__main__":
    main()

