# Censorium

**Real-time face and license plate redaction for privacy protection**

Censorium is a complete image redaction system that automatically detects and obscures faces and license plates in images. Built with PyTorch, FastAPI, and Next.js, it provides both a web interface and CLI tool for privacy-preserving image processing.

## Quick Start

### Prerequisites

- **Python 3.10+** (check: `python3 --version`)
- **Node.js 18+** (check: `node --version`)
- **8GB+ RAM** (16GB recommended)
- **Git** (for cloning)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/censorium.git
cd censorium

# 2. Setup backend
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Setup frontend
cd ../frontend
npm install

# 4. Validate installation
cd ..
python validate_system.py
```

Expected output: `ALL CHECKS PASSED - System is ready!`

## Running the System

### Start Backend API

**Terminal 1:**
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for: `All models loaded successfully!` (takes ~30 seconds on first run)

**Alternative:** Use startup script:
```bash
./start_backend.sh
```

### Start Frontend UI

**Terminal 2:**
```bash
cd frontend
npm run dev
```

**Alternative:** Use startup script:
```bash
./start_frontend.sh
```

### Access Web Interface

Open `http://localhost:3000` in your browser. Drag and drop images to redact.

**API available at:** `http://localhost:8000`

## Usage

### Web Interface

1. Open `http://localhost:3000`
2. Drag and drop images or click to select
3. Adjust settings:
   - **Mode**: Blur or Pixelate
   - **Confidence**: Detection threshold (0.0-1.0)
4. View side-by-side comparison
5. Download redacted images

### CLI Tool

**Single image:**
```bash
cd backend
source venv/bin/activate
python run_redaction.py --input photo.jpg --output redacted.jpg
```

**Batch directory:**
```bash
python run_redaction.py \
  --input ./photos \
  --output ./redacted \
  --recursive \
  --mode blur \
  --confidence 0.5
```

**CLI Options:**
- `--input, -i`: Input file or directory (required)
- `--output, -o`: Output location (required)
- `--mode, -m`: `blur` or `pixelate` (default: `blur`)
- `--confidence, -c`: Threshold 0.0-1.0 (default: 0.5)
- `--padding, -p`: Bounding box padding (default: 0.1)
- `--recursive, -r`: Process subdirectories
- `--verbose, -v`: Detailed output
- `--save-stats`: Save statistics to JSON

### API Endpoints

**Health check:**
```bash
curl http://localhost:8000/health
```

**Redact image:**
```bash
curl -X POST http://localhost:8000/redact-image \
  -F "file=@image.jpg" \
  -F "mode=blur" \
  -F "confidence_threshold=0.5" \
  --output redacted.jpg
```

**Preview detections (draw boxes):**
```bash
curl -X POST http://localhost:8000/preview-detections \
  -F "file=@image.jpg" \
  --output preview.jpg
```

**Batch processing (returns ZIP):**
```bash
curl -X POST http://localhost:8000/redact-batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  --output redacted.zip
```

## Testing

### System Validation

```bash
python validate_system.py
```

Checks: Python version, Node.js, dependencies, file structure, imports.

### API Tests

```bash
cd backend
source venv/bin/activate

# Basic API tests
python test_api.py

# Test with specific image
python test_api.py /path/to/test/image.jpg
```

Tests: Health endpoint, stats endpoint, image redaction.

### Integration Verification

```bash
cd backend
source venv/bin/activate
python verify_integration.py
```

Verifies: Model loading, face detection, plate detection, unified pipeline.

### Roboflow Model Test

```bash
cd backend
source venv/bin/activate
python test_roboflow_model.py
```

Tests: Custom license plate model loading and inference.

## Training

### License Plate Detection Training

**Note:** Face detection uses pretrained MTCNN (no training needed).

#### Prerequisites

- Dataset in YOLOv8 format at `backend/training/datasets/license_plates/`
- Structure:
  ```
  datasets/license_plates/
  ├── data.yaml
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
  ```

#### Download Dataset from Roboflow

1. Export in **YOLOv8** format (not YOLOv5, YOLOv7, etc.)
2. Extract to `backend/training/datasets/license_plates/`
3. Verify structure matches above

#### Verify Dataset

```bash
cd backend/training
source ../../venv/bin/activate  # or activate backend venv
python verify_preprocessing.py
```

Expected: All images 512×512, labels valid, dataset structure correct.

#### Train Model (Ultralytics CLI)
```bash
cd backend/training
source ../../venv/bin/activate

# Full training
yolo detect train \
  data=datasets/license_plates/data.yaml \
  model=yolov8n.pt \
  epochs=85 \
  imgsz=512 \
  batch=16 \
  device=cpu \
  project=runs/detect \
  name=license_plate_train
```

**Training Time Estimates:**
- CPU (M2 Mac): 2-4 hours

#### Monitor Training

Training outputs are saved to `backend/training/runs/detect/license_plate_train/`:
- `weights/best.pt`: Best model checkpoint
- `weights/last.pt`: Latest checkpoint (for resuming)
- `results.png`: Training curves
- `results.csv`: Training metrics

View progress:
```bash
cd backend/training
open runs/detect/license_plate_train/results.png  # Mac
# Or navigate to the file in your file manager
```

#### Resume Training

```bash
cd backend/training
source ../../venv/bin/activate
yolo detect train \
  data=datasets/license_plates/data.yaml \
  model=runs/detect/license_plate_train/weights/last.pt \
  epochs=85 \
  imgsz=512 \
  batch=16 \
  device=cpu \
  project=runs/detect \
  name=license_plate_train \
  resume
```

#### Validate Trained Model

```bash
cd backend/training
source ../../venv/bin/activate
yolo detect val \
  model=runs/detect/license_plate_train/weights/best.pt \
  data=datasets/license_plates/data.yaml
```

#### Deploy Trained Model

After you are satisfied with the results, copy the best checkpoint into the backend models directory:

```bash
cd backend/training
cp runs/detect/license_plate_train/weights/best.pt ../models/license_plate.pt
```

### Training Troubleshooting

**Out of memory:**
```bash
# Reduce batch size
yolo detect train \
  data=datasets/license_plates/data.yaml \
  model=yolov8n.pt \
  epochs=85 \
  imgsz=512 \
  batch=4 \
  device=cpu \
  project=runs/detect \
  name=license_plate_train
```

**Training too slow:**
- Use GPU if available: `--device cuda`
- Reduce epochs for testing: `--epochs 10`
- Reduce image size: `--imgsz 416` (if dataset allows)

**Missing labels:**
- Verify dataset structure
- Check `data.yaml` paths are correct
- Run `verify_preprocessing.py` to diagnose

## Evaluation

### Face Detection Evaluation

```bash
cd backend
source venv/bin/activate

# Create sample dataset for testing
python evaluate/evaluate_face.py --create-sample

# Run evaluation on your dataset
python evaluate/evaluate_face.py \
  --dataset ./your_face_dataset \
  --annotations ./your_face_dataset/annotations.json \
  --confidence 0.9 \
  --output face_results.json \
  --plot face_results.png
```

**Output:** Precision, recall, F1 score, mAP metrics and plots.

### License Plate Evaluation

```bash
python evaluate/evaluate_plate.py \
  --dataset ./your_plate_dataset \
  --annotations ./your_plate_dataset/annotations.json \
  --confidence 0.4 \
  --output plate_results.json \
  --plot plate_results.png
```

**Output:** Detection metrics, confidence threshold analysis, comparison plots.

### Performance Benchmarks

```bash
python evaluate/benchmark.py \
  --image-dir ./test_images \
  --output benchmark_results.json \
  --num-runs 10
```

**Output:** Latency statistics, throughput, memory usage, inference speed distribution.

## Architecture

```
┌─────────────────┐
│   Next.js UI    │  (React, Tailwind, Drag-drop)
└────────┬────────┘
         │ HTTP/REST
         ↓
┌─────────────────┐
│  FastAPI Server │  (Python, async endpoints)
└────────┬────────┘
         │
    ┌────┴────┐
    ↓         ↓
┌─────────┐ ┌──────────────┐
│  MTCNN  │ │   YOLOv8     │  (Face & Plate Detection)
│  Face   │ │   Plates     │
└────┬────┘ └──────┬───────┘
     │             │
     └──────┬──────┘
            ↓
    ┌───────────────┐
    │   Redaction   │  (Blur/Pixelate)
    │    Engine     │
    └───────────────┘
```

## Models

### Face Detection

- **Model**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Framework**: facenet-pytorch
- **Pretrained**: Yes (WIDER FACE dataset)
- **Default Confidence**: 0.9
- **Performance**: ~50-100ms per 1080p image
- **No training needed** - downloads automatically

### License Plate Detection

- **Model**: YOLOv8n (Nano)
- **Framework**: Ultralytics
- **Default Confidence**: 0.4 (optimized for Roboflow models)
- **Performance**: ~30-80ms per 1080p image
- **Training**: See Training section above

### Redaction Modes

1. **Gaussian Blur**: Smooth blur (kernel size: 51, configurable)
2. **Pixelation**: Mosaic effect (block size: 15, configurable)

## Performance

**Tested on M2 MacBook Pro (16GB RAM):**

- Face Detection: >90% recall on WIDER FACE validation
- Plate Detection: >85% recall on test set
- End-to-end Latency: 100-250ms per 1080p image
- Throughput: 4-8 images/second
- Memory Usage: ~2GB baseline, ~3-4GB under load

## Project Structure

```
censorium/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── detector.py          # Unified detector interface
│   │   ├── detector_face.py     # Face detection (MTCNN)
│   │   ├── detector_plate.py    # License plate detection (YOLOv8)
│   │   ├── redaction.py         # Redaction engine
│   │   ├── schemas.py           # Pydantic models
│   │   └── utils.py             # Utility functions
│   ├── evaluate/
│   │   ├── evaluate_face.py     # Face detection evaluation
│   │   ├── evaluate_plate.py    # Plate detection evaluation
│   │   └── benchmark.py         # Performance benchmarks
│   ├── training/
│   │   ├── verify_preprocessing.py  # Dataset verification
│   │   ├── preprocess_dataset.py    # Preprocessing pipeline
│   │   ├── datasets/                # Training datasets
│   │   └── README.md                # Training documentation
│   ├── models/                  # Model weights (gitignored)
│   ├── run_redaction.py         # CLI tool
│   ├── test_api.py              # API tests
│   ├── test_roboflow_model.py   # Model tests
│   ├── verify_integration.py    # Integration tests
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Main page
│   │   └── layout.tsx           # App layout
│   ├── components/
│   │   └── RedactionViewer.tsx  # Image viewer component
│   ├── lib/
│   │   └── api.ts               # API client
│   └── package.json             # Node dependencies
├── examples/                    # Sample images
├── documentation/               # Additional docs (optional)
├── validate_system.py           # System validation script
├── start_backend.sh             # Backend startup script
├── start_frontend.sh            # Frontend startup script
└── README.md                    # This file
```

## Troubleshooting

### Models Not Loading

**Symptom:** "Models not loaded" error or slow startup

**Solutions:**
- Wait 30-60 seconds after starting backend (first-time model download)
- Check internet connection (MTCNN downloads on first use)
- Verify `backend/models/` directory exists and is writable
- Check backend logs for specific error messages

### Out of Memory

**Symptom:** Process killed or "CUDA out of memory"

**Solutions:**
- Close other applications
- Reduce batch size in training: `--batch 4`
- Process images sequentially (not in parallel)
- Use smaller images (resize before processing)
- Reduce confidence threshold (fewer detections = less memory)

### Slow Inference

**Symptom:** Processing takes >1 second per image

**Solutions:**
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce image resolution before processing
- Lower confidence thresholds (faster processing)
- Use GPU if available (modify detector initialization)
- Check system resources: `htop` or Activity Monitor

### Frontend Can't Connect to Backend

**Symptom:** "API offline" or connection errors

**Solutions:**
- Verify backend is running: `curl http://localhost:8000/health`
- Check backend is on port 8000 (default)
- Verify CORS settings in `backend/app/main.py`
- Check `frontend/.env.local` has: `NEXT_PUBLIC_API_URL=http://localhost:8000`
- Check firewall/antivirus blocking connections

### Port Already in Use

**Symptom:** "Address already in use" error

**Solutions:**
```bash
# Find process using port
lsof -i :8000  # Backend
lsof -i :3000  # Frontend

# Kill process or use different port
# Backend:
python -m uvicorn app.main:app --port 8001

# Frontend:
npm run dev -- -p 3001
```

### Training Errors

**Symptom:** Dataset not found or invalid format

**Solutions:**
- Verify dataset path in `data.yaml`
- Run `verify_preprocessing.py` to check dataset
- Ensure YOLOv8 format (not YOLOv5, COCO, etc.)
- Check all images are 512×512 (if required)
- Verify label files exist for all images

### Import Errors

**Symptom:** "ModuleNotFoundError" or import failures

**Solutions:**
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python3 --version` (need 3.10+)
- Verify you're in correct directory
- Run `validate_system.py` to diagnose

## Development

### Code Formatting

**Backend (Python):**
```bash
cd backend
source venv/bin/activate
pip install black ruff
black app/ evaluate/ training/
ruff check app/ evaluate/ training/
```

**Frontend (TypeScript):**
```bash
cd frontend
npm run lint
npm run format
```

### Adding New Features

1. Backend changes: Modify files in `backend/app/`
2. Frontend changes: Modify files in `frontend/app/` or `frontend/components/`
3. Test changes: Run `test_api.py` and `validate_system.py`
4. Restart services: Stop and restart backend/frontend

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **MTCNN**: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
- **YOLOv8**: Ultralytics YOLOv8 object detection
- **FastAPI**: Modern web framework for Python
- **Next.js**: React framework for production

## Citation

If you use Censorium in your research:

```bibtex
@software{censorium2025,
  title={Censorium: Real-time Image Redaction System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/censorium}
}
```

## Support

For issues or questions:
- Open a GitHub issue
- Check troubleshooting section above
- Review logs in backend terminal
- Run `validate_system.py` for diagnostics
