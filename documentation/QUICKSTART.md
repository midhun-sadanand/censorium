# Custom Training Pipeline - Quick Start Guide

This guide helps you quickly train a custom YOLOv8 model and compare it with Roboflow's model.

## Prerequisites

 Dataset extracted at `datasets/license_plates/`  
 Pretrained model `yolov8n.pt` downloaded  
 Python environment with ultralytics installed

## 5-Minute Quick Start

### Step 1: Verify Dataset (30 seconds)

```bash
cd backend/training
python verify_preprocessing.py
```

Expected output: ` DATASET VERIFICATION PASSED!`

### Step 2: Train Custom Model (2-4 hours on CPU, 20-30 min on GPU)

```bash
# Full training (85 epochs, matches Roboflow)
python train_custom.py --data datasets/license_plates/data.yaml

# OR quick test (1 epoch, ~10 minutes)
python train_custom.py --data datasets/license_plates/data.yaml --epochs 1 --batch 4
```

Training outputs:
- `runs/detect/custom_train/weights/best.pt` - Best model
- `runs/detect/custom_train/results.png` - Training curves
- `runs/detect/custom_train/training_summary.json` - Training metrics

### Step 3: Compare Models (5 minutes)

```bash
python compare_models.py \
  --custom runs/detect/custom_train/weights/best.pt \
  --roboflow ../models/license_plate.pt \
  --data datasets/license_plates/data.yaml
```

This generates:
- Comparison metrics (mAP, precision, recall)
- Speed benchmarks
- `comparison_results.json` - Full comparison report

### Step 4: Deploy Best Model (30 seconds)

If your custom model is better:

```bash
python deploy_model.py --model runs/detect/custom_train/weights/best.pt --name custom
```

This copies the model to `../models/license_plate_custom.pt` for use in the application.

## Understanding Results

### Training Metrics

Check `runs/detect/custom_train/results.png` for:
- **Loss curves**: Should decrease over epochs
- **mAP@50**: Should reach ≥95% (Roboflow baseline: 97.7%)
- **Precision/Recall**: Should be balanced

### Comparison Results

The comparison script shows:
- **mAP@50**: Higher is better (detection accuracy)
- **Precision**: Higher is better (fewer false positives)
- **Recall**: Higher is better (fewer missed detections)
- **Inference Time**: Lower is better (faster processing)

### Choosing the Best Model

Use the model with:
- **Higher mAP@50** (primary metric)
- **Balanced precision/recall** (not too many false positives or missed detections)
- **Acceptable speed** (meets <300ms requirement)

## Common Commands

```bash
# Resume training from checkpoint
python train_custom.py --data datasets/license_plates/data.yaml --resume runs/detect/custom_train/weights/last.pt

# Train on GPU
python train_custom.py --data datasets/license_plates/data.yaml --device cuda --batch 32

# Validate only (no training)
python train_custom.py --data datasets/license_plates/data.yaml --validate-only --model-path runs/detect/custom_train/weights/best.pt

# Compare with custom confidence threshold
python compare_models.py --custom best.pt --roboflow ../models/license_plate.pt --data data.yaml --confidence 0.5
```

## Troubleshooting

**Out of Memory:**
```bash
python train_custom.py --data datasets/license_plates/data.yaml --batch 4
```

**Training Too Slow:**
```bash
python train_custom.py --data datasets/license_plates/data.yaml --device cuda
```

**Model Not Improving:**
- Check dataset: `python verify_preprocessing.py`
- Try more epochs: `--epochs 100`
- Check learning rate (YOLOv8 auto-adjusts)

## Next Steps

1. **Review comparison results** in `comparison_results.json`
2. **Document findings** for your project report
3. **Deploy best model** if custom model is better
4. **Update application** to use the chosen model

## For Project Report

You can state:

> "We implemented a custom YOLOv8 training pipeline using the Ultralytics API to train license plate detection models locally. After training for 85 epochs with the same configuration as Roboflow (512×512 images, batch size 16, SGD optimizer), we compared our custom model with Roboflow's model on the test set. Our custom model achieved [X]% mAP@50, [Y]% precision, and [Z]% recall, compared to Roboflow's 97.7% mAP@50, 98.6% precision, and 95.4% recall. Based on this comparison, we chose to use [Roboflow/Custom] model for production because [reason]."

## Full Documentation

- **Training details**: See `TRAINING_COMPARISON.md`
- **Preprocessing**: See `PREPROCESSING_GUIDE.md`
- **Implementation**: See `IMPLEMENTATION_SUMMARY.md`

---

**Ready to train?** Run: `python train_custom.py --data datasets/license_plates/data.yaml`

