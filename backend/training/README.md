# YOLOv8 License Plate Detection Training Pipeline

Complete training pipeline for replicating the Roboflow license plate detection model.

## Quick Start

```bash
# Dataset is already extracted at: datasets/license_plates/
# Model weights downloaded: yolov8n.pt

# 1. Verify dataset
python verify_preprocessing.py

# 2. Train model (use ultralytics CLI for simplicity)
yolo detect train \
  data=datasets/license_plates/data.yaml \
  model=yolov8n.pt \
  epochs=85 \
  imgsz=512 \
  batch=16 \
  device=cpu \
  project=runs/detect \
  name=license_plate_train

# 3. Evaluate results
yolo detect val \
  model=runs/detect/license_plate_train/weights/best.pt \
  data=datasets/license_plates/data.yaml

# 4. Deploy if successful
cp runs/detect/license_plate_train/weights/best.pt ../models/license_plate.pt
```

## What's Been Done

### Completed Setup

1. **Dataset Extracted**: 24,011 images in YOLOv8 format
   - Train: 20,943 images (with 3x augmentation)
   - Valid: 2,048 images
   - Test: 1,020 images
   - All images: 512×512 (preprocessed)

2. **Model Downloaded**: yolov8n.pt (COCO pretrained)

3. **Verification Script**: `verify_preprocessing.py` validates dataset

4. **Directory Structure**: Proper training setup complete

### Dataset Information (from Roboflow)

**Original:**
- 10,125 images
- 10,633 license plate annotations
- 1 class: `License_Plate`

**Preprocessing Applied:**
- Auto-Orient
- Resize to 512×512 (stretch)
- Auto-Adjust Contrast (histogram equalization)

**Augmentation:**
- Brightness: -15% to +15%
- Result: 3x augmentation → 20,943 training images

**Target Metrics:**
- mAP@50: 97.7%
- Precision: 98.6%
- Recall: 95.4%

## Training Commands

###  Recommended: Use Ultralytics CLI

The simplest way to train matching Roboflow configuration:

```bash
# Standard training (85 epochs, matches Roboflow)
yolo detect train \
  data=datasets/license_plates/data.yaml \
  model=yolov8n.pt \
  epochs=85 \
  imgsz=512 \
  batch=16 \
  device=cpu \
  project=runs/detect \
  name=license_plate_train \
  patience=50 \
  save_period=10

# Quick test (1 epoch)
yolo detect train \
  data=datasets/license_plates/data.yaml \
  model=yolov8n.pt \
  epochs=1 \
  imgsz=512 \
  batch=4

# With GPU (if available)
yolo detect train \
  data=datasets/license_plates/data.yaml \
  model=yolov8n.pt \
  epochs=85 \
  imgsz=512 \
  batch=32 \
  device=0
```

## Training Time Estimates

- **CPU (M2 Mac):** 2-4 hours for 85 epochs
- **CPU (older):** 6-8 hours for 85 epochs  
- **GPU (RTX 3080):** 20-30 minutes
- **GPU (RTX 4090):** 15-20 minutes

## Evaluation

```bash
# Validate trained model
yolo detect val \
  model=runs/detect/license_plate_train/weights/best.pt \
  data=datasets/license_plates/data.yaml

# Test on images
yolo detect predict \
  model=runs/detect/license_plate_train/weights/best.pt \
  source=datasets/license_plates/test/images \
  save=True
```

## Directory Structure

```
training/
├── datasets/
│   └── license_plates/          # Extracted dataset
│       ├── data.yaml             # YOLOv8 config
│       ├── train/ (20,943)       # Training data
│       ├── valid/ (2,048)        # Validation data
│       └── test/ (1,020)         # Test data
├── runs/
│   └── detect/                   # Training outputs
│       └── license_plate_train/
│           ├── weights/
│           │   ├── best.pt       # Best model
│           │   └── last.pt       # Latest checkpoint
│           └── results.png       # Training curves
├── yolov8n.pt                    # COCO pretrained model
├── verify_preprocessing.py       # Dataset verification
└── README.md                     # This file
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
yolo detect train data=datasets/license_plates/data.yaml model=yolov8n.pt batch=4
```

### Resume Training
```bash
# Resume from last checkpoint
yolo detect train resume model=runs/detect/license_plate_train/weights/last.pt
```

### View Training Progress
```bash
# Training outputs are in runs/detect/license_plate_train/
# Open results.png to see training curves
open runs/detect/license_plate_train/results.png
```

## Face Detection Model

The face detection model uses **pretrained MTCNN** and does not require training:

- Model: MTCNN (Multi-task Cascaded CNN)
- Source: facenet-pytorch package
- Pretrained: Yes (WIDER FACE dataset)
- Location: `../models/face_detector.pt` (cached)
- Performance: >90% recall
- **No training needed** - downloads automatically

## Next Steps After Training

1. **Check metrics** in `runs/detect/license_plate_train/results.csv`
2. **View training curves** in `results.png`
3. **Validate on test set** using `yolo detect val`
4. **Compare to baseline** (Roboflow model: 97.7% mAP@50)
5. **Deploy if successful**: Copy `best.pt` to `../models/license_plate.pt`

## Training Configuration (Matches Roboflow)

The ultralytics CLI automatically uses optimal settings:

| Parameter | Value | Note |
|-----------|-------|------|
| Model | yolov8n.pt | COCO pretrained (Nano) |
| Epochs | 85 | Matches Roboflow training |
| Image Size | 512×512 | Matches preprocessing |
| Batch | 16 | Adjust for your hardware |
| Device | cpu/cuda | Auto-detect or specify |
| Augmentation | Auto | Includes brightness, flip, etc. |
| Optimizer | SGD | With momentum |
| Learning Rate | 0.01 → 0.0001 | Linear decay |

## Resources

- **Ultralytics Docs:** https://docs.ultralytics.com/
- **Roboflow Dataset:** https://universe.roboflow.com/censorium/license-plate-recognition-rxg4e-b3cku
- **Dataset Path:** `datasets/license_plates/`
- **Model Weights:** `yolov8n.pt` (COCO pretrained)

## Preprocessing Pipeline

### Current Dataset

The **current dataset is already preprocessed** by Roboflow and ready for training. You don't need to run preprocessing.

### Preprocessing Script (For New Data)

If you want to process new raw images to match the Roboflow preprocessing:

```bash
# Replicates Roboflow preprocessing on raw images
python preprocess_dataset.py \
  --input-dir path/to/raw/images \
  --output-dir datasets/my_processed_data \
  --target-size 512 \
  --augment-train \
  --num-augmentations 2
```

**What it does:**
1. Auto-Orient (EXIF correction)
2. Resize to 512×512 (stretch)
3. Auto-Adjust Contrast (histogram equalization)
4. Brightness augmentation -15% to +15% (training only, 3x multiplier)

See `PREPROCESSING_GUIDE.md` for details.

## Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `verify_preprocessing.py` | Verify dataset is correctly preprocessed | `python verify_preprocessing.py` |
| `preprocess_dataset.py` | Process raw images (Roboflow pipeline) | `python preprocess_dataset.py --input-dir raw_data` |

## Support

**Verification passed?** Run: `python verify_preprocessing.py`

**Training issues?** Check:
- Batch size (reduce if OOM)
- Device (cpu vs cuda)
- Dataset path in data.yaml

**Need to process new images?** See `PREPROCESSING_GUIDE.md`

**Need help?** Review training outputs in `runs/detect/license_plate_train/`

---

**Status:** Ready for Training  
**Dataset:** Verified (24,011 images)  
**Model:** Downloaded (yolov8n.pt)  
**Scripts:** Preprocessing + Verification available  
**Next:** Run training command above

