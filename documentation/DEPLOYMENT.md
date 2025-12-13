# Censorium Deployment Guide

Guide for deploying Censorium in various environments.

## Development Deployment (Local)

Already covered in QUICKSTART.md - use `start_backend.sh` and `start_frontend.sh`.

## Production Deployment

### Docker Deployment (Recommended)

#### Backend Dockerfile

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Dockerfile

Create `frontend/Dockerfile`:

```dockerfile
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application
COPY . .

# Build
RUN npm run build

# Expose port
EXPOSE 3000

# Run application
CMD ["npm", "start"]
```

#### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
    restart: unless-stopped
```

Start with:
```bash
docker-compose up -d
```

### Cloud Deployment

#### AWS Deployment

**Option 1: EC2**

1. Launch EC2 instance (t3.medium or larger)
2. Install Docker and Docker Compose
3. Clone repository
4. Run `docker-compose up -d`
5. Configure security group (ports 80, 443, 8000, 3000)

**Option 2: ECS (Elastic Container Service)**

1. Build and push images to ECR
2. Create ECS task definitions
3. Create ECS service
4. Configure ALB for load balancing

**Option 3: Lambda + API Gateway**

For serverless deployment (requires adaptation):
- Package backend as Lambda function
- Use API Gateway for routing
- Frontend on S3 + CloudFront

#### Google Cloud Platform

**Cloud Run** (simplest):

```bash
# Backend
gcloud run deploy censorium-backend \
  --source ./backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Frontend
gcloud run deploy censorium-frontend \
  --source ./frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure

**Azure Container Instances**:

```bash
# Create resource group
az group create --name censorium --location eastus

# Deploy backend
az container create \
  --resource-group censorium \
  --name censorium-backend \
  --image your-registry/censorium-backend \
  --ports 8000

# Deploy frontend
az container create \
  --resource-group censorium \
  --name censorium-frontend \
  --image your-registry/censorium-frontend \
  --ports 3000
```

### Reverse Proxy (Nginx)

Create `nginx.conf`:

```nginx
upstream backend {
    server localhost:8000;
}

upstream frontend {
    server localhost:3000;
}

server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # API
    location /api/ {
        proxy_pass http://backend/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeouts for large images
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
        send_timeout 300;
    }
}
```

### SSL/TLS (Let's Encrypt)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

## Environment Variables

### Backend

Create `backend/.env`:

```bash
# Optional: Custom model paths
FACE_MODEL_PATH=/path/to/face_model.pt
PLATE_MODEL_PATH=/path/to/plate_model.pt

# Optional: Performance tuning
MAX_IMAGE_SIZE=4096
ENABLE_GPU=true

# Optional: Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/censorium/backend.log
```

### Frontend

Create `frontend/.env.production`:

```bash
NEXT_PUBLIC_API_URL=https://api.your-domain.com
```

## Performance Optimization

### Backend

1. **Enable GPU**:
   ```python
   # In detector initialization
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

2. **Model Optimization**:
   ```bash
   # Convert to TorchScript for faster inference
   python scripts/export_torchscript.py
   ```

3. **Caching**:
   - Use Redis for caching detection results
   - Cache model weights on disk

4. **Scaling**:
   - Run multiple Uvicorn workers:
     ```bash
     uvicorn app.main:app --workers 4
     ```

### Frontend

1. **Build Optimization**:
   ```bash
   npm run build
   npm run start  # Production mode
   ```

2. **CDN**: Host static assets on CDN

3. **Image Optimization**: Use Next.js Image component

## Monitoring

### Logs

**Backend**:
```bash
# View logs
tail -f backend.log

# Docker logs
docker logs -f censorium-backend
```

**Frontend**:
```bash
# Docker logs
docker logs -f censorium-frontend
```

### Metrics

Add Prometheus + Grafana for monitoring:

1. Install Prometheus exporter:
   ```bash
   pip install prometheus-fastapi-instrumentator
   ```

2. Add to FastAPI app:
   ```python
   from prometheus_fastapi_instrumentator import Instrumentator
   
   app = FastAPI()
   Instrumentator().instrument(app).expose(app)
   ```

3. Configure Prometheus to scrape `/metrics`

### Health Checks

Check endpoints:
```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000/api/health
```

## Security

### API Security

1. **Rate Limiting**:
   ```python
   from slowapi import Limiter
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   @app.post("/redact-image")
   @limiter.limit("10/minute")
   async def redact_image(...):
       ...
   ```

2. **Authentication** (optional):
   ```python
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   @app.post("/redact-image")
   async def redact_image(
       credentials: HTTPAuthorizationCredentials = Depends(security),
       ...
   ):
       # Verify token
       ...
   ```

3. **Input Validation**:
   - Already implemented via Pydantic
   - Add file size limits
   - Validate image formats

### Frontend Security

1. **Content Security Policy**:
   ```javascript
   // next.config.js
   module.exports = {
     async headers() {
       return [
         {
           source: '/:path*',
           headers: [
             {
               key: 'Content-Security-Policy',
               value: "default-src 'self'; img-src 'self' data: blob:;"
             }
           ]
         }
       ]
     }
   }
   ```

2. **CORS**: Already configured in backend

3. **HTTPS**: Always use HTTPS in production

## Backup and Recovery

### Model Weights

```bash
# Backup
tar -czf models_backup.tar.gz backend/models/

# Restore
tar -xzf models_backup.tar.gz -C backend/
```

### Database (if added)

```bash
# PostgreSQL backup
pg_dump censorium > backup.sql

# Restore
psql censorium < backup.sql
```

## Troubleshooting

### High Memory Usage

- Reduce batch size
- Add memory limits in Docker:
  ```yaml
  services:
    backend:
      mem_limit: 4g
  ```

### Slow Inference

- Check GPU availability
- Reduce image resolution
- Use model quantization

### API Timeouts

- Increase Nginx timeouts
- Increase Uvicorn timeout:
  ```bash
  uvicorn app.main:app --timeout-keep-alive 300
  ```

## Maintenance

### Updates

```bash
# Backend updates
cd backend
git pull
pip install -r requirements.txt
docker-compose restart backend

# Frontend updates
cd frontend
git pull
npm install
npm run build
docker-compose restart frontend
```

### Model Updates

1. Download new model weights
2. Place in `backend/models/`
3. Update configuration
4. Restart backend

---

For more details, see:
- README.md - General usage
- QUICKSTART.md - Local development
- TECHNICAL_REPORT.md - Architecture details


