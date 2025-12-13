"""
Data schemas for Censorium API
"""
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field
from enum import Enum


class RedactionMode(str, Enum):
    """Redaction mode options"""
    BLUR = "blur"
    PIXELATE = "pixelate"


class EntityType(str, Enum):
    """Types of entities that can be detected"""
    FACE = "face"
    LICENSE_PLATE = "license_plate"


class Detection(BaseModel):
    """Single detection result"""
    bbox: Tuple[int, int, int, int] = Field(
        ..., description="Bounding box as (x1, y1, x2, y2)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    entity_type: EntityType = Field(..., description="Type of detected entity")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bbox": [100, 100, 200, 200],
                "confidence": 0.95,
                "entity_type": "face"
            }
        }


class RedactionRequest(BaseModel):
    """Request parameters for redaction"""
    mode: RedactionMode = Field(default=RedactionMode.BLUR, description="Redaction mode")
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    padding_factor: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Padding factor for bounding boxes"
    )
    blur_kernel_size: int = Field(
        default=51, ge=3, le=201, description="Kernel size for Gaussian blur (odd number)"
    )
    pixelate_block_size: int = Field(
        default=15, ge=5, le=50, description="Block size for pixelation"
    )


class RedactionResponse(BaseModel):
    """Response for redaction operation"""
    detections: List[Detection] = Field(..., description="List of detected entities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    image_dimensions: Tuple[int, int] = Field(..., description="Image dimensions (width, height)")
    

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    version: str = "1.0.0"




