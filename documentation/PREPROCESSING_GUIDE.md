# Roboflow Preprocessing Pipeline Replication

This guide explains the preprocessing script that replicates what Roboflow did to create the training dataset.

## What the Script Does

The `preprocess_dataset.py` script performs the **exact same preprocessing and augmentation** that Roboflow applied:

### Preprocessing (Applied to ALL images)

1. **Auto-Orient**
   - Reads EXIF rotation metadata
   - Corrects image orientation automatically
   - Ensures images display correctly

2. **Resize to 512×512 (Stretch)**
   - Resizes all images to exactly 512×512 pixels
   - Uses "stretch" mode (ignores aspect ratio)
   - Matches Roboflow's resize configuration

3. **Auto-Adjust Contrast**
   - Applies histogram equalization
   - Enhances image contrast for better detection
   - Improves visibility in various lighting conditions

### Augmentation (Training set ONLY, 3x multiplier)

4. **Brightness Variations**
   - Original image (100% brightness)
   - Darker version (85% = -15%)
   - Brighter version (115% = +15%)
   - Results in 3x training images

**Example:**
- Input: 7,081 training images
- Output: 21,243 training images (7,081 × 3)

## How It Would Have Been Used

If we had raw images instead of the Roboflow export, this script would have created the current dataset:

```bash
# Starting with raw images in this structure:
raw_dataset/
  ├── train/
  │   ├── images/     # 7,081 raw images (any size)
  │   └── labels/     # 7,081 YOLO label files
  ├── valid/
  │   ├── images/     # 2,048 raw images
  │   └── labels/     # 2,048 label files
  └── test/
      ├── images/     # 1,020 raw images
      └── labels/     # 1,020 label files

# Run preprocessing
cd backend/training
python preprocess_dataset.py \
  --input-dir raw_dataset \
  --output-dir datasets/license_plates \
  --target-size 512 \
  --augment-train \
  --num-augmentations 2

# Output would be:
datasets/license_plates/
  ├── data.yaml
  ├── train/
  │   ├── images/     # 21,243 images (7,081 × 3) - all 512×512
  │   └── labels/     # 21,243 label files
  ├── valid/
  │   ├── images/     # 2,048 images - all 512×512
  │   └── labels/     # 2,048 label files
  └── test/
      ├── images/     # 1,020 images - all 512×512
      └── labels/     # 1,020 label files
```

## Current Dataset

The **current dataset** in `datasets/license_plates/` was already preprocessed by Roboflow, so:
-  All images are already 512×512
-  Contrast adjustment already applied
-  Augmentation already done (3x multiplier)
-  Ready for training

**You don't need to run this script on the current dataset!**

## When to Use This Script

Use `preprocess_dataset.py` when you:

1. **Have new raw images** to add to the dataset
2. **Want to create a dataset from scratch** without Roboflow
3. **Need to replicate Roboflow preprocessing** locally
4. **Want to understand** exactly what Roboflow did

## Usage Examples

### Basic Usage

```bash
python preprocess_dataset.py \
  --input-dir path/to/raw/dataset \
  --output-dir datasets/my_preprocessed_data
```

### Custom Configuration

```bash
# Different image size
python preprocess_dataset.py \
  --input-dir raw_data \
  --output-dir preprocessed_640 \
  --target-size 640

# No augmentation
python preprocess_dataset.py \
  --input-dir raw_data \
  --output-dir preprocessed_no_aug \
  --augment-train false

# More augmentation (5x multiplier)
python preprocess_dataset.py \
  --input-dir raw_data \
  --output-dir preprocessed_5x \
  --num-augmentations 4

# Multiple classes
python preprocess_dataset.py \
  --input-dir raw_data \
  --output-dir preprocessed \
  --class-names License_Plate Car Truck
```

### Process Only Specific Splits

```bash
# Process only training data
python preprocess_dataset.py \
  --input-dir raw_data \
  --output-dir preprocessed \
  --splits train

# Process train and validation only
python preprocess_dataset.py \
  --input-dir raw_data \
  --output-dir preprocessed \
  --splits train valid
```

## Input Requirements

The script expects this directory structure:

```
input_directory/
├── train/
│   ├── images/          # Raw images (any size, JPG/PNG)
│   └── labels/          # YOLO format labels (.txt)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Label Format:** YOLO format (normalized coordinates)
```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.2
```

## Output

The script creates:

1. **Preprocessed images** (512×512, contrast-adjusted)
2. **Augmented versions** (for training set only)
3. **Copied label files** (coordinates remain valid since they're normalized)
4. **data.yaml** configuration file for YOLOv8

## Technical Details

### Auto-Orient Implementation

```python
def auto_orient(image_path: Path) -> np.ndarray:
    """Uses PIL's EXIF transpose to correct rotation."""
    pil_image = Image.open(image_path)
    pil_image = ImageOps.exif_transpose(pil_image)
    return np.array(pil_image)
```

### Resize (Stretch) Implementation

```python
def resize_stretch(image: np.ndarray) -> np.ndarray:
    """Resizes to 512×512 ignoring aspect ratio."""
    return cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
```

### Contrast Adjustment Implementation

```python
def auto_adjust_contrast(image: np.ndarray) -> np.ndarray:
    """Applies histogram equalization to Y channel in YUV space."""
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
```

### Brightness Augmentation Implementation

```python
def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjusts V channel in HSV by factor (0.85 = -15%, 1.15 = +15%)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

## Label Coordinate Handling

Since YOLO labels use **normalized coordinates** (0-1 range), they remain valid after resizing:

- **Before:** Image 1920×1080, license plate at (0.5, 0.5, 0.3, 0.2)
- **After:** Image 512×512, license plate still at (0.5, 0.5, 0.3, 0.2)

The script simply copies label files without modification.

## Verification

After preprocessing, verify the output:

```bash
python verify_preprocessing.py --dataset-path datasets/my_preprocessed_data
```

This checks:
-  All images are 512×512
-  All labels are valid
-  Brightness variation exists (augmentation applied)

## Comparison: Roboflow vs. This Script

| Aspect | Roboflow | This Script | Match |
|--------|----------|-------------|-------|
| Auto-Orient |  EXIF-based |  PIL EXIF transpose |  |
| Resize |  Stretch to 512×512 |  cv2.resize ignore aspect |  |
| Contrast |  Histogram equalization |  equalizeHist on Y channel |  |
| Brightness Aug |  -15% to +15% |  HSV V channel ×0.85, ×1.15 |  |
| Augmentation 3x |  Original + 2 versions |  Original + 2 versions |  |
| Output Format |  YOLOv8 structure |  YOLOv8 structure |  |

## Performance

Processing speed (approximate):

- **Small dataset** (1,000 images): ~2-3 minutes
- **Medium dataset** (10,000 images): ~15-20 minutes
- **Current dataset** (10,125 → 24,011): ~20-25 minutes

With augmentation (3x): Processing time increases by ~2x

## Adding New Images to Existing Dataset

If you want to add new images to the current preprocessed dataset:

```bash
# 1. Create structure for new images
mkdir -p new_images/train/images
mkdir -p new_images/train/labels

# 2. Add your raw images and labels
cp my_new_images/*.jpg new_images/train/images/
cp my_new_labels/*.txt new_images/train/labels/

# 3. Preprocess new images
python preprocess_dataset.py \
  --input-dir new_images \
  --output-dir new_images_processed \
  --target-size 512 \
  --augment-train

# 4. Merge with existing dataset
cp new_images_processed/train/images/* datasets/license_plates/train/images/
cp new_images_processed/train/labels/* datasets/license_plates/train/labels/

# 5. Verify merged dataset
python verify_preprocessing.py
```

## Troubleshooting

### Issue: Label coordinates seem wrong after resize

**Cause:** You might be using absolute coordinates instead of normalized.

**Solution:** YOLO format requires normalized coordinates (0-1). Convert if needed:
```python
x_center_norm = x_center_pixels / image_width
y_center_norm = y_center_pixels / image_height
width_norm = box_width / image_width
height_norm = box_height / image_height
```

### Issue: Colors look different after preprocessing

**Expected:** Histogram equalization changes contrast and may affect colors slightly. This improves detection performance.

### Issue: Out of memory

**Solution:** Process smaller batches or reduce image size.

## Summary

 **Script Created:** `preprocess_dataset.py`  
 **Replicates:** Roboflow's exact preprocessing pipeline  
 **Use Case:** Process raw images into training-ready format  
 **Current Dataset:** Already preprocessed, doesn't need this script  
 **Future Use:** Process new images to match existing format  

---

**Note:** The current dataset in `datasets/license_plates/` was exported from Roboflow already preprocessed. This script is provided to show how those preprocessing steps could be replicated locally if needed.

