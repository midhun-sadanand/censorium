"""
Image processing utilities for Censorium
"""
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format (BGR)
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        numpy array in BGR format
    """
    # Convert PIL to RGB numpy array
    rgb_image = np.array(pil_image.convert('RGB'))
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV format (BGR) to PIL Image
    
    Args:
        cv2_image: numpy array in BGR format
        
    Returns:
        PIL Image object
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL
    return Image.fromarray(rgb_image)


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        numpy array in BGR format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image


def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> None:
    """
    Save image to file
    
    Args:
        image: numpy array in BGR format
        output_path: Path to save image
        quality: JPEG quality (0-100)
    """
    # Set compression parameters
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif output_path.lower().endswith('.png'):
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    else:
        params = []
    
    success = cv2.imwrite(output_path, image, params)
    if not success:
        raise ValueError(f"Failed to save image to {output_path}")


def clip_bbox(bbox: Tuple[int, int, int, int], 
              image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Clip bounding box to image boundaries
    
    Args:
        bbox: (x1, y1, x2, y2)
        image_shape: (height, width)
        
    Returns:
        Clipped (x1, y1, x2, y2)
    """
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    # Ensure x2 > x1 and y2 > y1
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    
    return (x1, y1, x2, y2)


def expand_bbox(bbox: Tuple[int, int, int, int], 
                padding_factor: float,
                image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Expand bounding box by padding factor
    
    Args:
        bbox: (x1, y1, x2, y2)
        padding_factor: Factor to expand bbox (e.g., 0.1 = 10% padding)
        image_shape: (height, width)
        
    Returns:
        Expanded and clipped (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    
    pad_w = int(width * padding_factor)
    pad_h = int(height * padding_factor)
    
    x1_new = x1 - pad_w
    y1_new = y1 - pad_h
    x2_new = x2 + pad_w
    y2_new = y2 + pad_h
    
    return clip_bbox((x1_new, y1_new, x2_new, y2_new), image_shape)


def calculate_iou(bbox1: Tuple[int, int, int, int], 
                  bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        bbox1: (x1, y1, x2, y2)
        bbox2: (x1, y1, x2, y2)
        
    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def non_max_suppression(bboxes: List[Tuple[int, int, int, int]], 
                        scores: List[float],
                        iou_threshold: float = 0.5) -> List[int]:
    """
    Apply Non-Maximum Suppression to remove duplicate detections
    
    Args:
        bboxes: List of bounding boxes
        scores: List of confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if len(bboxes) == 0:
        return []
    
    # Sort by score (descending)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    keep = []
    while sorted_indices:
        # Take the highest scoring box
        current = sorted_indices[0]
        keep.append(current)
        sorted_indices = sorted_indices[1:]
        
        # Remove boxes with high IoU with current box
        filtered_indices = []
        for idx in sorted_indices:
            iou = calculate_iou(bboxes[current], bboxes[idx])
            if iou < iou_threshold:
                filtered_indices.append(idx)
        sorted_indices = filtered_indices
    
    return keep


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to OpenCV format
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        numpy array in BGR format
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes")
    return image


def cv2_to_bytes(image: np.ndarray, format: str = '.jpg', quality: int = 95) -> bytes:
    """
    Convert OpenCV image to bytes
    
    Args:
        image: numpy array in BGR format
        format: Image format ('.jpg' or '.png')
        quality: JPEG quality (0-100)
        
    Returns:
        Image as bytes
    """
    if format.lower() in ['.jpg', '.jpeg']:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif format.lower() == '.png':
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    else:
        encode_param = []
    
    success, buffer = cv2.imencode(format, image, encode_param)
    if not success:
        raise ValueError(f"Failed to encode image to {format}")
    
    return buffer.tobytes()




