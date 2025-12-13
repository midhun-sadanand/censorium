# Censorium Quick Start Guide

Get Censorium up and running in 5 minutes!

## Prerequisites

- Python 3.10+
- Node.js 18+
- 8GB RAM minimum

## Step 1: Start the Backend

Open a terminal and run:

```bash
./start_backend.sh
```

Or manually:

```bash
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for "All models loaded successfully!" message (takes ~30 seconds on first run).

## Step 2: Start the Frontend

Open a **new terminal** and run:

```bash
./start_frontend.sh
```

Or manually:

```bash
cd frontend
npm run dev
```

## Step 3: Use the Web Interface

1. Open http://localhost:3000 in your browser
2. Drag and drop an image
3. Wait for processing (usually <1 second)
4. Download the redacted result

## Step 4: Try the CLI (Optional)

```bash
cd backend
source venv/bin/activate

# Single image
python run_redaction.py --input photo.jpg --output redacted.jpg

# Batch process a directory
python run_redaction.py --input ./photos --output ./redacted --recursive
```

## Testing the API

```bash
cd backend
source venv/bin/activate
python test_api.py
```

With a test image:

```bash
python test_api.py /path/to/test/image.jpg
```

## Troubleshooting

### "Models not loaded"
Wait 30-60 seconds after starting backend. First model download takes time.

### Port already in use
Kill existing process or change port:
```bash
# Backend on different port
python -m uvicorn app.main:app --port 8001

# Frontend on different port
npm run dev -- -p 3001
```

### Out of memory
Close other applications or reduce image sizes.

### Can't connect frontend to backend
Check `frontend/.env.local` has correct API URL:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## API Examples

### cURL

```bash
# Redact image
curl -X POST http://localhost:8000/redact-image \
  -F "file=@image.jpg" \
  -F "mode=blur" \
  -o redacted.jpg

# Get metadata
curl -X POST http://localhost:8000/redact-image \
  -F "file=@image.jpg" \
  -F "return_metadata=true"

# Batch process
curl -X POST http://localhost:8000/redact-batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -o redacted.zip
```

### Python

```python
import requests

# Redact image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/redact-image',
        files={'file': f},
        data={'mode': 'blur', 'confidence_threshold': 0.5}
    )
    
with open('redacted.jpg', 'wb') as f:
    f.write(response.content)
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('mode', 'blur');

const response = await fetch('http://localhost:8000/redact-image', {
  method: 'POST',
  body: formData
});

const blob = await response.blob();
const url = URL.createObjectURL(blob);
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for architecture details
- Run evaluation scripts to test on your own datasets
- Customize detection thresholds and redaction settings

## Common Use Cases

### Journalism
Protect identities in photos before publication

### Social Media
Automatically redact faces in group photos

### Law Enforcement
Anonymize evidence photos and videos

### Research
De-identify datasets while preserving spatial information

### Real Estate
Blur faces and license plates in property photos

---

Need help? Check the troubleshooting section in README.md or open an issue on GitHub.


