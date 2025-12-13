"""
FastAPI main application for Censorium
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import logging
import time
import io
import zipfile
from pathlib import Path
import json

from .detector import EntityDetector
from .redaction import RedactionEngine
from .schemas import (
    RedactionMode, 
    RedactionResponse, 
    Detection, 
    HealthResponse
)
from .utils import bytes_to_cv2, cv2_to_bytes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Censorium API",
    description="Real-time face and license plate redaction API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (loaded on startup)
detector: Optional[EntityDetector] = None
redaction_engine: Optional[RedactionEngine] = None


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global detector, redaction_engine
    
    logger.info("Starting Censorium API...")
    
    try:
        logger.info("Loading detection models...")
        
        # Path to custom Roboflow license plate model
        import os
        models_dir = Path(__file__).parent.parent / "models"
        plate_model_path = models_dir / "license_plate.pt"
        
        if plate_model_path.exists():
            logger.info(f"Found Roboflow license plate model at {plate_model_path}")
        else:
            logger.warning(f"Roboflow model not found at {plate_model_path}, will use fallback")
            plate_model_path = None
        
        detector = EntityDetector(
            face_confidence=0.9,
            plate_confidence=0.4,  # Roboflow models work well at 0.4
            plate_model_path=str(plate_model_path) if plate_model_path else None
        )
        
        logger.info("Initializing redaction engine...")
        redaction_engine = RedactionEngine()
        
        logger.info("All models loaded successfully!")
        logger.info("Roboflow license plate detector ready!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="online",
        models_loaded=detector is not None and redaction_engine is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=detector is not None and redaction_engine is not None
    )


@app.post("/redact-image")
async def redact_image(
    file: UploadFile = File(...),
    mode: str = Form(default="blur"),
    confidence_threshold: float = Form(default=0.5),
    padding_factor: float = Form(default=0.1),
    blur_kernel_size: int = Form(default=51),
    pixelate_block_size: int = Form(default=15),
    return_metadata: bool = Form(default=False)
):
    """
    Redact faces and license plates in a single image
    
    Args:
        file: Image file to process
        mode: Redaction mode ('blur' or 'pixelate')
        confidence_threshold: Minimum detection confidence
        padding_factor: Bounding box padding factor
        blur_kernel_size: Gaussian blur kernel size
        pixelate_block_size: Pixelation block size
        return_metadata: Return JSON metadata instead of image
        
    Returns:
        Redacted image or JSON metadata
    """
    if detector is None or redaction_engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        start_time = time.time()
        
        # Read image file
        image_bytes = await file.read()
        image = bytes_to_cv2(image_bytes)
        
        height, width = image.shape[:2]
        
        # Detect entities
        detections = detector.detect_all(
            image,
            confidence_threshold=confidence_threshold,
            padding_factor=padding_factor
        )
        
        # Parse redaction mode
        try:
            redaction_mode = RedactionMode(mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {mode}. Use 'blur' or 'pixelate'"
            )
        
        # Apply redaction
        redacted_image = redaction_engine.redact(
            image,
            detections,
            mode=redaction_mode,
            blur_kernel_size=blur_kernel_size,
            pixelate_block_size=pixelate_block_size
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(
            f"Processed image: {width}x{height}, "
            f"{len(detections)} detections, "
            f"{processing_time:.2f}ms"
        )
        
        # Return metadata or image
        if return_metadata:
            response = RedactionResponse(
                detections=detections,
                processing_time_ms=processing_time,
                image_dimensions=(width, height)
            )
            return response
        else:
            # Convert to bytes and return
            image_bytes = cv2_to_bytes(redacted_image, format='.jpg', quality=95)
            return Response(content=image_bytes, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/redact-batch")
async def redact_batch(
    files: List[UploadFile] = File(...),
    mode: str = Form(default="blur"),
    confidence_threshold: float = Form(default=0.5),
    padding_factor: float = Form(default=0.1),
    blur_kernel_size: int = Form(default=51),
    pixelate_block_size: int = Form(default=15)
):
    """
    Redact multiple images and return as ZIP archive
    
    Args:
        files: List of image files
        mode: Redaction mode
        confidence_threshold: Minimum detection confidence
        padding_factor: Bounding box padding factor
        blur_kernel_size: Gaussian blur kernel size
        pixelate_block_size: Pixelation block size
        
    Returns:
        ZIP file containing redacted images
    """
    if detector is None or redaction_engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Parse redaction mode
        try:
            redaction_mode = RedactionMode(mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {mode}"
            )
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                try:
                    # Read and process image
                    image_bytes = await file.read()
                    image = bytes_to_cv2(image_bytes)
                    
                    # Detect and redact
                    detections = detector.detect_all(
                        image,
                        confidence_threshold=confidence_threshold,
                        padding_factor=padding_factor
                    )
                    
                    redacted_image = redaction_engine.redact(
                        image,
                        detections,
                        mode=redaction_mode,
                        blur_kernel_size=blur_kernel_size,
                        pixelate_block_size=pixelate_block_size
                    )
                    
                    # Convert to bytes
                    redacted_bytes = cv2_to_bytes(redacted_image, format='.jpg')
                    
                    # Add to ZIP with sanitized filename
                    filename = file.filename or f"image_{files.index(file)}.jpg"
                    zip_file.writestr(f"redacted_{filename}", redacted_bytes)
                    
                    logger.info(f"Processed {filename}: {len(detections)} detections")
                    
                except Exception as e:
                    logger.error(f"Failed to process {file.filename}: {e}")
                    continue
        
        # Return ZIP file
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=redacted_images.zip"}
        )
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preview-detections")
async def preview_detections(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.5),
    padding_factor: float = Form(default=0.1)
):
    """
    Preview detections without redaction (draws bounding boxes)
    
    Args:
        file: Image file
        confidence_threshold: Minimum detection confidence
        padding_factor: Bounding box padding factor
        
    Returns:
        Image with detection boxes drawn
    """
    if detector is None or redaction_engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read image
        image_bytes = await file.read()
        image = bytes_to_cv2(image_bytes)
        
        # Detect entities
        detections = detector.detect_all(
            image,
            confidence_threshold=confidence_threshold,
            padding_factor=padding_factor
        )
        
        # Draw preview
        preview_image = redaction_engine.preview_detections(image, detections)
        
        # Convert to bytes
        preview_bytes = cv2_to_bytes(preview_image, format='.jpg', quality=95)
        
        return Response(content=preview_bytes, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Error creating preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "status": "operational",
        "models_loaded": detector is not None and redaction_engine is not None,
        "supported_modes": ["blur", "pixelate"],
        "supported_entities": ["face", "license_plate"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



