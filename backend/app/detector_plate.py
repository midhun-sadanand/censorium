"""
License plate detection using YOLOv8
"""
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import logging
import os

logger = logging.getLogger(__name__)


class LicensePlateDetector:
    """
    License plate detector using YOLOv8
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.4):
        """
        Initialize license plate detector
        
        Args:
            model_path: Path to custom YOLO model weights (from Roboflow)
            confidence_threshold: Minimum confidence for detection (default 0.4 for Roboflow models)
        """
        logger.info("Initializing LicensePlateDetector")
        
        self.confidence_threshold = confidence_threshold
        
        # Default to using the custom Roboflow-trained model
        if model_path is None:
            # Look for license plate model in models directory
            default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'license_plate.pt')
            if os.path.exists(default_path):
                model_path = default_path
                logger.info(f"Using default Roboflow license plate model: {default_path}")
            else:
                logger.warning("No custom model found, falling back to YOLOv8n base model")
                model_path = 'yolov8n.pt'
        
        if not os.path.exists(model_path) and model_path != 'yolov8n.pt':
            raise FileNotFoundError(f"License plate model not found: {model_path}")
        
        logger.info(f"Loading license plate model from {model_path}")
        self.model = YOLO(model_path)
        
        # Check if this is a custom trained model
        self.is_custom_model = 'license_plate' in str(model_path).lower()
        
        logger.info(f"LicensePlateDetector initialized successfully (custom model: {self.is_custom_model})")
    
    def detect(self, 
               image: np.ndarray,
               confidence_threshold: Optional[float] = None) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        Detect license plates in image
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            confidence_threshold: Override default confidence threshold
            
        Returns:
            Tuple of (bboxes, confidences)
            - bboxes: List of (x1, y1, x2, y2) tuples
            - confidences: List of confidence scores
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Run YOLOv8 detection
        results = self.model(image, verbose=False, conf=confidence_threshold)
        
        bboxes = []
        confidences = []
        
        # Process results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Ensure valid bounding box
                    if x2 > x1 and y2 > y1:
                        # If using custom Roboflow model, trust all detections above threshold
                        if self.is_custom_model:
                            bboxes.append((x1, y1, x2, y2))
                            confidences.append(float(conf))
                        else:
                            # For base model, filter by aspect ratio
                            width = x2 - x1
                            height = y2 - y1
                            aspect_ratio = width / height if height > 0 else 0
                            
                            # License plates typically have aspect ratio between 2:1 and 6:1
                            if 1.5 <= aspect_ratio <= 8.0:
                                bboxes.append((x1, y1, x2, y2))
                                confidences.append(float(conf))
        
        return bboxes, confidences
    
    def detect_batch(self,
                     images: List[np.ndarray],
                     confidence_threshold: Optional[float] = None) -> List[Tuple[List[Tuple[int, int, int, int]], List[float]]]:
        """
        Detect license plates in multiple images (batch processing)
        
        Args:
            images: List of images as numpy arrays
            confidence_threshold: Override default confidence threshold
            
        Returns:
            List of (bboxes, confidences) tuples for each image
        """
        results = []
        for image in images:
            result = self.detect(image, confidence_threshold)
            results.append(result)
        return results
    
    def load_custom_model(self, model_path: str):
        """
        Load a custom trained model for license plate detection
        
        Args:
            model_path: Path to custom YOLOv8 weights
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading custom model from {model_path}")
        self.model = YOLO(model_path)
        logger.info("Custom model loaded successfully")



