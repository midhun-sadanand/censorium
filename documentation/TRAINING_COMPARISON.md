# Custom Training Pipeline vs Roboflow - Comparison Guide

This document explains how to use the custom training pipeline and compare results with Roboflow-trained models.

## Overview

The custom training pipeline (`train_custom.py`) provides a programmatic way to train YOLOv8 models locally, allowing direct comparison with Roboflow's training results. This enables you to:

1. **Train models locally** using the same dataset and preprocessing
2. **Compare performance** between custom and Roboflow models
3. **Understand training differences** and choose the best model
4. **Reproduce results** independently of Roboflow

## Quick Start

### 1. Train Custom Model

```bash
cd backend/training

# Train with default settings (matches Roboflow: 85 epochs, 512x512, batch 16)
python train_custom.py --data datasets/license_plates/data.yaml

# Quick test (1 epoch)
python train_custom.py --data datasets/license_plates/data.yaml --epochs 1 --batch 4

# GPU training (faster)
python train_custom.py --data datasets/license_plates/data.yaml --device cuda --batch 32
```

### 2. Compare Models

```bash
# Compare custom vs Roboflow models
python compare_models.py \
  --custom runs/detect/custom_train/weights/best.pt \
  --roboflow ../models/license_plate.pt \
  --data datasets/license_plates/data.yaml
```

### 3. Use Best Model

After comparison, use the better-performing model in your application:

```bash
# If custom model is better, copy it
cp runs/detect/custom_train/weights/best.pt ../models/license_plate_custom.pt

# Update detector to use custom model (see Integration section)
```

## Training Configuration

### Default Settings (Matches Roboflow)

The custom training pipeline uses the same configuration as Roboflow:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | yolov8n.pt | COCO pretrained (Nano variant) |
| Epochs | 85 | Matches Roboflow training duration |
| Image Size | 512×512 | Matches preprocessing |
| Batch Size | 16 | Adjust for your hardware |
| Optimizer | SGD | With momentum (YOLOv8 default) |
| Learning Rate | Auto-scheduled | YOLOv8 default schedule |
| Augmentation | Auto | Includes brightness, flip, etc. |
| Patience | 50 | Early stopping patience |

### Customizing Training

You can adjust any parameter:

```bash
# Different model size
python train_custom.py --data datasets/license_plates/data.yaml --model yolov8s.pt

# More epochs
python train_custom.py --data datasets/license_plates/data.yaml --epochs 100

# Larger batch (if you have GPU memory)
python train_custom.py --data datasets/license_plates/data.yaml --batch 32

# Custom experiment name
python train_custom.py --data datasets/license_plates/data.yaml --name my_experiment_v2
```

## Model Comparison

### What Gets Compared

The comparison script evaluates both models on:

1. **Accuracy Metrics:**
   - mAP@50 (mean Average Precision at IoU=0.5)
   - mAP@50-95 (mean Average Precision at IoU=0.5:0.95)
   - Precision
   - Recall

2. **Speed Metrics:**
   - Mean inference time (milliseconds)
   - FPS (frames per second)
   - Standard deviation of inference time

3. **Differences:**
   - Direct comparison showing which model performs better

### Example Comparison Output

```
============================================================
COMPARISON SUMMARY
============================================================

 Accuracy Metrics:
Metric               Custom          Roboflow        Difference      
-----------------------------------------------------------------
mAP@50               0.9750          0.9770          -0.0020
Precision            0.9860          0.9860          0.0000
Recall               0.9540          0.9540          0.0000

Speed Metrics:
Metric                           Custom          Roboflow        
------------------------------------------------------------
Mean Inference (ms)             98.50           102.30
FPS                              10.15           9.78

Winner:
 Roboflow Model wins on mAP@50
```

### Interpreting Results

- **mAP@50**: Higher is better. Measures detection accuracy at IoU threshold 0.5
- **Precision**: Higher is better. Percentage of detections that are correct
- **Recall**: Higher is better. Percentage of actual objects detected
- **Inference Time**: Lower is better. Time to process one image
- **FPS**: Higher is better. Images processed per second

## Integration with Application

### Option 1: Use Custom Model (If Better)

If your custom model performs better, update the detector:

```python
# In app/main.py, update the model path:
plate_model_path = models_dir / "license_plate_custom.pt"  # Use custom model
```

### Option 2: Support Both Models

You can modify the detector to support model selection:

```python
# In app/detector_plate.py, add model selection:
def __init__(self, 
             model_path: Optional[str] = None,
             model_type: str = "roboflow",  # "roboflow" or "custom"
             confidence_threshold: float = 0.4):
    if model_type == "custom":
        default_path = models_dir / "license_plate_custom.pt"
    else:
        default_path = models_dir / "license_plate.pt"
    # ... rest of initialization
```

### Option 3: A/B Testing

For production, you can implement A/B testing to compare models on real data:

```python
# Run both models and compare results
custom_detector = LicensePlateDetector(model_path="license_plate_custom.pt")
roboflow_detector = LicensePlateDetector(model_path="license_plate.pt")

custom_results = custom_detector.detect(image)
roboflow_results = roboflow_detector.detect(image)

# Compare and log differences
```

## Training Workflow

### Complete Workflow

```
1. Prepare Dataset
   └─> datasets/license_plates/ (already done)

2. Verify Dataset
   └─> python verify_preprocessing.py

3. Train Custom Model
   └─> python train_custom.py --data datasets/license_plates/data.yaml

4. Evaluate Custom Model
   └─> python train_custom.py --validate-only --model-path runs/detect/custom_train/weights/best.pt

5. Compare Models
   └─> python compare_models.py --custom best.pt --roboflow ../models/license_plate.pt

6. Deploy Best Model
   └─> Copy best.pt to models/ directory
```

### Training Time Estimates

| Hardware | Time (85 epochs) |
|----------|------------------|
| M2 MacBook Pro (CPU) | 2-4 hours |
| Older CPU | 6-8 hours |
| RTX 3080 (GPU) | 20-30 minutes |
| RTX 4090 (GPU) | 15-20 minutes |

### Monitoring Training

Training progress is saved to:
- `runs/detect/custom_train/results.png` - Training curves
- `runs/detect/custom_train/results.csv` - Metrics per epoch
- `runs/detect/custom_train/weights/best.pt` - Best model
- `runs/detect/custom_train/weights/last.pt` - Latest checkpoint

## Expected Results

### Roboflow Baseline

From Roboflow training:
- **mAP@50**: 97.7%
- **Precision**: 98.6%
- **Recall**: 95.4%

### Custom Training Target

With identical configuration, custom training should achieve:
- **mAP@50**: ≥95% (within 3% of Roboflow)
- **Precision**: ≥96% (within 3% of 98.6%)
- **Recall**: ≥93% (within 3% of 95.4%)

### Why Differences Might Occur

Small differences (<3%) are expected due to:
- **Random initialization**: Different random seeds
- **Hardware differences**: CPU vs GPU numerical precision
- **Training environment**: PyTorch version, CUDA version
- **Augmentation randomness**: Different augmentation sequences

Larger differences (>5%) may indicate:
- **Configuration mismatch**: Different hyperparameters
- **Data preprocessing**: Different preprocessing steps
- **Training issues**: Learning rate, batch size, etc.

## Troubleshooting

### Training Fails

**Out of Memory:**
```bash
# Reduce batch size
python train_custom.py --data data.yaml --batch 4
```

**Slow Training:**
```bash
# Use GPU if available
python train_custom.py --data data.yaml --device cuda
```

**Model Not Improving:**
- Check learning rate (YOLOv8 auto-adjusts)
- Verify dataset quality: `python verify_preprocessing.py`
- Try more epochs: `--epochs 100`

### Comparison Fails

**Models Not Found:**
- Verify paths are correct
- Check that models exist: `ls -lh runs/detect/custom_train/weights/best.pt`

**Different Results:**
- Ensure same test set is used
- Check confidence threshold matches: `--confidence 0.4`
- Verify data.yaml points to same dataset

## Files Created

After training and comparison:

```
backend/training/
├── train_custom.py              # Custom training script
├── compare_models.py            # Model comparison script
├── runs/detect/
│   └── custom_train/
│       ├── weights/
│       │   ├── best.pt          # Best model
│       │   └── last.pt          # Latest checkpoint
│       ├── results.png          # Training curves
│       ├── results.csv          # Metrics per epoch
│       ├── training_summary.json
│       └── validation_metrics.json
└── comparison_results.json      # Comparison results
```

## Next Steps

1. **Train your custom model** using `train_custom.py`
2. **Compare with Roboflow** using `compare_models.py`
3. **Choose the best model** based on metrics
4. **Integrate into application** if custom model is better
5. **Document your findings** for the project report

## For Project Report

When writing your project report, you can state:

> "We implemented a custom YOLOv8 training pipeline using the Ultralytics API to train license plate detection models locally. This allowed us to compare our custom training results with Roboflow's trained models on the same test dataset. After evaluation, we found that [Roboflow/Custom] model achieved [X]% mAP@50, [Y]% precision, and [Z]% recall, and we chose to use [Roboflow/Custom] model for production based on [reason]."

## Support

- **Training issues**: Check `runs/detect/custom_train/` for logs
- **Comparison issues**: Verify model paths and data.yaml
- **Integration issues**: See `ROBOFLOW_INTEGRATION.md` for detector setup

---

**Status**:  Ready for Training and Comparison  
**Last Updated**: December 2025

