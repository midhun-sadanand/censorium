"""
Redaction engine for applying blur and pixelation to detected regions
"""
import numpy as np
import cv2
from typing import List, Tuple
import logging
from .schemas import Detection, RedactionMode
from .utils import clip_bbox

logger = logging.getLogger(__name__)


class RedactionEngine:
    """
    Engine for applying redaction techniques to images
    """
    
    def __init__(self):
        """Initialize redaction engine"""
        logger.info("RedactionEngine initialized")
    
    def redact(self,
               image: np.ndarray,
               detections: List[Detection],
               mode: RedactionMode = RedactionMode.BLUR,
               blur_kernel_size: int = 51,
               pixelate_block_size: int = 15) -> np.ndarray:
        """
        Apply redaction to detected regions in image
        
        Args:
            image: Image as numpy array (BGR format)
            detections: List of Detection objects
            mode: Redaction mode (blur or pixelate)
            blur_kernel_size: Kernel size for Gaussian blur (must be odd)
            pixelate_block_size: Block size for pixelation
            
        Returns:
            Redacted image as numpy array
        """
        # Work on a copy to avoid modifying original
        redacted_image = image.copy()
        
        height, width = image.shape[:2]
        image_shape = (height, width)
        
        for detection in detections:
            bbox = detection.bbox
            
            # Clip bbox to image boundaries
            x1, y1, x2, y2 = clip_bbox(bbox, image_shape)
            
            # Ensure valid region
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bbox after clipping: {bbox}")
                continue
            
            # Extract region
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                logger.warning(f"Empty region for bbox: {bbox}")
                continue
            
            # Apply redaction
            try:
                if mode == RedactionMode.BLUR:
                    redacted_region = self._apply_gaussian_blur(region, blur_kernel_size)
                elif mode == RedactionMode.PIXELATE:
                    redacted_region = self._apply_pixelation(region, pixelate_block_size)
                else:
                    logger.error(f"Unknown redaction mode: {mode}")
                    continue
                
                # Replace region in image
                redacted_image[y1:y2, x1:x2] = redacted_region
                
            except Exception as e:
                logger.error(f"Failed to redact region {bbox}: {e}")
                continue
        
        return redacted_image
    
    def _apply_gaussian_blur(self, 
                            region: np.ndarray,
                            kernel_size: int) -> np.ndarray:
        """
        Apply Gaussian blur to region
        
        Args:
            region: Image region as numpy array
            kernel_size: Kernel size (must be odd and positive)
            
        Returns:
            Blurred region
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Ensure minimum kernel size
        kernel_size = max(3, kernel_size)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
        
        return blurred
    
    def _apply_pixelation(self,
                         region: np.ndarray,
                         block_size: int) -> np.ndarray:
        """
        Apply pixelation to region
        
        Args:
            region: Image region as numpy array
            block_size: Size of pixelation blocks
            
        Returns:
            Pixelated region
        """
        height, width = region.shape[:2]
        
        # Ensure valid block size
        block_size = max(5, min(block_size, min(height, width)))
        
        # Calculate dimensions for downsampled image
        temp_height = max(1, height // block_size)
        temp_width = max(1, width // block_size)
        
        # Downsample
        temp = cv2.resize(region, (temp_width, temp_height), interpolation=cv2.INTER_LINEAR)
        
        # Upsample back to original size (creates pixelated effect)
        pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def redact_with_rectangle(self,
                             image: np.ndarray,
                             detections: List[Detection],
                             color: Tuple[int, int, int] = (0, 0, 0),
                             thickness: int = -1) -> np.ndarray:
        """
        Redact by drawing rectangles (solid fill or outline)
        
        Args:
            image: Image as numpy array
            detections: List of Detection objects
            color: Rectangle color in BGR
            thickness: Thickness of rectangle (-1 for filled)
            
        Returns:
            Redacted image
        """
        redacted_image = image.copy()
        
        height, width = image.shape[:2]
        image_shape = (height, width)
        
        for detection in detections:
            bbox = detection.bbox
            x1, y1, x2, y2 = clip_bbox(bbox, image_shape)
            
            cv2.rectangle(redacted_image, (x1, y1), (x2, y2), color, thickness)
        
        return redacted_image
    
    def redact_selective(self,
                        image: np.ndarray,
                        detections: List[Detection],
                        mode: RedactionMode = RedactionMode.BLUR,
                        entity_types: List[str] = None,
                        blur_kernel_size: int = 51,
                        pixelate_block_size: int = 15) -> np.ndarray:
        """
        Apply redaction only to specific entity types
        
        Args:
            image: Image as numpy array
            detections: List of Detection objects
            mode: Redaction mode
            entity_types: List of entity types to redact (None = all)
            blur_kernel_size: Kernel size for blur
            pixelate_block_size: Block size for pixelation
            
        Returns:
            Redacted image
        """
        if entity_types is not None:
            # Filter detections by entity type
            filtered_detections = [
                d for d in detections 
                if d.entity_type.value in entity_types
            ]
        else:
            filtered_detections = detections
        
        return self.redact(
            image,
            filtered_detections,
            mode,
            blur_kernel_size,
            pixelate_block_size
        )
    
    def preview_detections(self,
                          image: np.ndarray,
                          detections: List[Detection],
                          show_confidence: bool = True,
                          color_map: dict = None) -> np.ndarray:
        """
        Draw detection boxes on image for preview (no redaction)
        
        Args:
            image: Image as numpy array
            detections: List of Detection objects
            show_confidence: Whether to show confidence scores
            color_map: Custom color map for entity types
            
        Returns:
            Image with detection boxes drawn
        """
        preview_image = image.copy()
        
        if color_map is None:
            color_map = {
                'face': (0, 255, 0),  # Green
                'license_plate': (255, 0, 0)  # Blue
            }
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            entity_type = detection.entity_type.value
            confidence = detection.confidence
            
            color = color_map.get(entity_type, (0, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(preview_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_confidence:
                label = f"{entity_type}: {confidence:.2f}"
            else:
                label = entity_type
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                preview_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                preview_image,
                label,
                (x1, y1 - baseline - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return preview_image




