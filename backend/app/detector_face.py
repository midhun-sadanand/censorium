"""
Face detection using RetinaFace (via facenet-pytorch)
"""
import numpy as np
import torch
from typing import List, Tuple, Optional
from facenet_pytorch import MTCNN
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector using MTCNN (Multi-task Cascaded Convolutional Networks)
    from facenet-pytorch, which includes RetinaFace-style detection
    """
    
    def __init__(self, device: Optional[str] = None, confidence_threshold: float = 0.9):
        """
        Initialize face detector
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detection
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing FaceDetector on device: {self.device}")
        
        # Initialize MTCNN detector
        self.detector = MTCNN(
            keep_all=True,
            device=self.device,
            thresholds=[0.6, 0.7, confidence_threshold],  # Three stage thresholds
            post_process=False,
            select_largest=False
        )
        
        self.confidence_threshold = confidence_threshold
        logger.info("FaceDetector initialized successfully")
    
    def detect(self, 
               image: np.ndarray, 
               confidence_threshold: Optional[float] = None) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        Detect faces in image
        
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
        
        # Convert BGR to RGB
        if len(image.shape) == 2:
            # Grayscale - convert to RGB
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image[:, :, ::-1].copy()  # BGR to RGB
        
        # Detect faces
        # MTCNN expects RGB PIL Image or numpy array
        boxes, probs = self.detector.detect(image_rgb)
        
        bboxes = []
        confidences = []
        
        if boxes is not None and probs is not None:
            for box, prob in zip(boxes, probs):
                if prob >= confidence_threshold:
                    # Convert to integer coordinates
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Ensure valid bounding box
                    if x2 > x1 and y2 > y1:
                        bboxes.append((x1, y1, x2, y2))
                        confidences.append(float(prob))
        
        return bboxes, confidences
    
    def detect_batch(self, 
                     images: List[np.ndarray],
                     confidence_threshold: Optional[float] = None) -> List[Tuple[List[Tuple[int, int, int, int]], List[float]]]:
        """
        Detect faces in multiple images (batch processing)
        
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
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'detector'):
            del self.detector
            if self.device == 'cuda':
                torch.cuda.empty_cache()




