# Censorium

**Real-time face and license plate redaction for privacy protection**

Censorium is a complete image redaction system that automatically detects and obscures faces and license plates in images. Built with PyTorch, FastAPI, and Next.js, it provides both a web interface and CLI tool for privacy-preserving image processing.

## Features

- **Dual Detection**: Simultaneous face and license plate detection
- **Multiple Redaction Modes**: Gaussian blur and pixelation
- **Real-time Processing**: Optimized for <300ms inference on modern hardware
- **Web Interface**: Drag-and-drop UI with live preview
- **CLI Tool**: Batch processing for directories
- **REST API**: Easy integration with other systems
- **Evaluation Tools**: Comprehensive metrics and benchmarking

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

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- 8GB+ RAM (16GB recommended for GPU acceleration)

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/yourusername/censorium.git
cd censorium
```

#### 2. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Frontend Setup

```bash
cd frontend
npm install
```

### Running the System

#### Start the Backend API

```bash
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

#### Start the Frontend

```bash
cd frontend
npm run dev
```

The web interface will be available at `http://localhost:3000`

## Usage

### Web Interface

1. Open `http://localhost:3000` in your browser
2. Drag and drop images or click to select files
3. Adjust redaction settings (mode, confidence threshold)
4. Download redacted images

### CLI Tool

#### Process a single image

```bash
cd backend
source venv/bin/activate
python run_redaction.py --input photo.jpg --output redacted.jpg
```

#### Batch process a directory

```bash
python run_redaction.py --input ./images --output ./redacted --mode blur
```

#### Recursive processing with custom settings

```bash
python run_redaction.py \
  --input ./images \
  --output ./redacted \
  --recursive \
  --mode pixelate \
  --confidence 0.7 \
  --verbose
```

#### CLI Options

- `--input, -i`: Input image file or directory (required)
- `--output, -o`: Output location (required)
- `--mode, -m`: Redaction mode (`blur` or `pixelate`, default: `blur`)
- `--confidence, -c`: Detection confidence threshold (0.0-1.0, default: 0.5)
- `--padding, -p`: Bounding box padding factor (default: 0.1)
- `--recursive, -r`: Process directories recursively
- `--verbose, -v`: Enable verbose output
- `--save-stats`: Save processing statistics to JSON file

### API Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Redact Single Image

```bash
curl -X POST http://localhost:8000/redact-image \
  -F "file=@image.jpg" \
  -F "mode=blur" \
  -F "confidence_threshold=0.5" \
  --output redacted.jpg
```

#### Redact Multiple Images (returns ZIP)

```bash
curl -X POST http://localhost:8000/redact-batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "mode=pixelate" \
  --output redacted.zip
```

#### Preview Detections (without redaction)

```bash
curl -X POST http://localhost:8000/preview-detections \
  -F "file=@image.jpg" \
  --output preview.jpg
```

## Evaluation

### Run Face Detection Evaluation

```bash
cd backend
source venv/bin/activate

# Create sample dataset for testing
python evaluate/evaluate_face.py --create-sample

# Run evaluation
python evaluate/evaluate_face.py \
  --dataset ./sample_face_dataset \
  --annotations ./sample_face_dataset/annotations.json \
  --confidence 0.5 \
  --output face_results.json \
  --plot face_results.png
```

### Run License Plate Evaluation

```bash
python evaluate/evaluate_plate.py \
  --dataset ./sample_plate_dataset \
  --annotations ./sample_plate_dataset/annotations.json \
  --confidence 0.5 \
  --output plate_results.json \
  --plot plate_results.png
```

### Run Performance Benchmarks

```bash
python evaluate/benchmark.py \
  --image-dir ./test_images \
  --output benchmark_results.json
```

## Model Details

### Face Detection

- **Model**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Framework**: facenet-pytorch
- **Default Confidence**: 0.9
- **Performance**: ~50-100ms per image (1080p)

### License Plate Detection

- **Model**: YOLOv8n
- **Framework**: Ultralytics
- **Default Confidence**: 0.5
- **Performance**: ~30-80ms per image (1080p)

### Redaction Techniques

1. **Gaussian Blur**: Applies smooth blur to detected regions
   - Kernel size: 51 (configurable)
   - Preserves edges while obscuring details

2. **Pixelation**: Reduces resolution in detected regions
   - Block size: 15 (configurable)
   - Creates mosaic effect

## Performance

Tested on M2 MacBook Pro (16GB RAM):

- **Face Detection**: >90% recall on WIDER FACE validation
- **Plate Detection**: >85% recall on test set
- **End-to-end Latency**: 100-250ms per 1080p image
- **Throughput**: 4-8 images/second
- **Memory Usage**: ~2GB baseline, ~3-4GB under load

## Project Structure

```
censorium/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── detector.py          # Unified detector interface
│   │   ├── detector_face.py     # Face detection
│   │   ├── detector_plate.py    # License plate detection
│   │   ├── redaction.py         # Redaction engine
│   │   ├── schemas.py           # Pydantic models
│   │   └── utils.py             # Utility functions
│   ├── evaluate/
│   │   ├── evaluate_face.py     # Face detection evaluation
│   │   ├── evaluate_plate.py    # Plate detection evaluation
│   │   └── benchmark.py         # Performance benchmarks
│   ├── models/                  # Model weights (gitignored)
│   ├── run_redaction.py         # CLI tool
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
└── README.md
```

## Development

### Running Tests

```bash
# Backend tests
cd backend
source venv/bin/activate
pytest

# Frontend tests
cd frontend
npm test
```

### Code Formatting

```bash
# Backend (Python)
cd backend
pip install black ruff
black app/ evaluate/
ruff check app/ evaluate/

# Frontend (TypeScript)
cd frontend
npm run lint
npm run format
```

## Troubleshooting

### "Models not loaded" error

Ensure the backend is fully started. Initial model loading takes 10-30 seconds.

### Out of memory errors

- Reduce batch size
- Process images sequentially
- Use smaller images
- Close other applications

### Slow inference

- Check if GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce image resolution
- Lower confidence thresholds

### Frontend can't connect to backend

- Verify backend is running on port 8000
- Check CORS settings in `backend/app/main.py`
- Update `NEXT_PUBLIC_API_URL` in `frontend/.env.local`

## Future Improvements

- [ ] GPU acceleration support
- [ ] Custom model training pipeline
- [ ] Video processing support
- [ ] Real-time webcam processing
- [ ] Additional entity types (text, logos)
- [ ] Docker deployment
- [ ] Cloud API deployment
- [ ] Mobile app

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **MTCNN**: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
- **YOLOv8**: Ultralytics YOLOv8 object detection
- **FastAPI**: Modern web framework for Python
- **Next.js**: React framework for production

## Citation

If you use Censorium in your research, please cite:

```bibtex
@software{censorium2025,
  title={Censorium: Real-time Image Redaction System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/censorium}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact [your.email@example.com](mailto:your.email@example.com)




