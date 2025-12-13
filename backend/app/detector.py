"""
Unified detector interface that orchestrates face and license plate detection
"""
import numpy as np
from typing import List, Optional, Tuple
import logging
from .detector_face import FaceDetector
from .detector_plate import LicensePlateDetector
from .schemas import Detection, EntityType
from .utils import expand_bbox, non_max_suppression

logger = logging.getLogger(__name__)


class EntityDetector:
    """
    Unified detector that orchestrates multiple detection models
    """
    
    def __init__(self,
                 face_confidence: float = 0.9,
                 plate_confidence: float = 0.5,
                 device: Optional[str] = None,
                 plate_model_path: Optional[str] = None):
        """
        Initialize entity detector with face and license plate detectors
        
        Args:
            face_confidence: Confidence threshold for face detection
            plate_confidence: Confidence threshold for plate detection
            device: Device to run models on
            plate_model_path: Path to custom license plate model
        """
        logger.info("Initializing EntityDetector")
        
        self.face_detector = FaceDetector(
            device=device,
            confidence_threshold=face_confidence
        )
        
        self.plate_detector = LicensePlateDetector(
            model_path=plate_model_path,
            confidence_threshold=plate_confidence
        )
        
        logger.info("EntityDetector initialized successfully")
    
    def detect_all(self,
                   image: np.ndarray,
                   confidence_threshold: float = 0.5,
                   padding_factor: float = 0.1,
                   apply_nms: bool = True,
                   nms_iou_threshold: float = 0.5) -> List[Detection]:
        """
        Detect all entities (faces and license plates) in image
        
        Args:
            image: Image as numpy array (BGR format)
            confidence_threshold: Minimum confidence threshold
            padding_factor: Factor to expand bounding boxes
            apply_nms: Whether to apply non-maximum suppression
            nms_iou_threshold: IoU threshold for NMS
            
        Returns:
            List of Detection objects sorted by confidence
        """
        height, width = image.shape[:2]
        image_shape = (height, width)
        
        detections = []
        
        # Detect faces
        try:
            face_bboxes, face_confs = self.face_detector.detect(
                image, 
                confidence_threshold=confidence_threshold
            )
            
            for bbox, conf in zip(face_bboxes, face_confs):
                # Expand bbox with padding
                expanded_bbox = expand_bbox(bbox, padding_factor, image_shape)
                
                detection = Detection(
                    bbox=expanded_bbox,
                    confidence=conf,
                    entity_type=EntityType.FACE
                )
                detections.append(detection)
            
            logger.debug(f"Detected {len(face_bboxes)} faces")
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
        
        # Detect license plates
        try:
            plate_bboxes, plate_confs = self.plate_detector.detect(
                image,
                confidence_threshold=confidence_threshold
            )
            
            for bbox, conf in zip(plate_bboxes, plate_confs):
                # Expand bbox with padding
                expanded_bbox = expand_bbox(bbox, padding_factor, image_shape)
                
                detection = Detection(
                    bbox=expanded_bbox,
                    confidence=conf,
                    entity_type=EntityType.LICENSE_PLATE
                )
                detections.append(detection)
            
            logger.debug(f"Detected {len(plate_bboxes)} license plates")
        except Exception as e:
            logger.error(f"License plate detection failed: {e}")
        
        # Apply NMS to remove duplicate detections
        if apply_nms and len(detections) > 1:
            detections = self._apply_nms(detections, nms_iou_threshold)
        
        # Sort by confidence (descending)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        return detections
    
    def _apply_nms(self, 
                   detections: List[Detection],
                   iou_threshold: float) -> List[Detection]:
        """
        Apply non-maximum suppression to detections
        
        Args:
            detections: List of Detection objects
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of Detection objects
        """
        if len(detections) <= 1:
            return detections
        
        bboxes = [d.bbox for d in detections]
        scores = [d.confidence for d in detections]
        
        keep_indices = non_max_suppression(bboxes, scores, iou_threshold)
        
        filtered_detections = [detections[i] for i in keep_indices]
        
        logger.debug(f"NMS: {len(detections)} -> {len(filtered_detections)} detections")
        
        return filtered_detections
    
    def detect_faces_only(self,
                          image: np.ndarray,
                          confidence_threshold: float = 0.9,
                          padding_factor: float = 0.1) -> List[Detection]:
        """
        Detect only faces in image
        
        Args:
            image: Image as numpy array
            confidence_threshold: Minimum confidence threshold
            padding_factor: Factor to expand bounding boxes
            
        Returns:
            List of face Detection objects
        """
        height, width = image.shape[:2]
        image_shape = (height, width)
        
        detections = []
        
        face_bboxes, face_confs = self.face_detector.detect(
            image,
            confidence_threshold=confidence_threshold
        )
        
        for bbox, conf in zip(face_bboxes, face_confs):
            expanded_bbox = expand_bbox(bbox, padding_factor, image_shape)
            
            detection = Detection(
                bbox=expanded_bbox,
                confidence=conf,
                entity_type=EntityType.FACE
            )
            detections.append(detection)
        
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
    
    def detect_plates_only(self,
                           image: np.ndarray,
                           confidence_threshold: float = 0.5,
                           padding_factor: float = 0.1) -> List[Detection]:
        """
        Detect only license plates in image
        
        Args:
            image: Image as numpy array
            confidence_threshold: Minimum confidence threshold
            padding_factor: Factor to expand bounding boxes
            
        Returns:
            List of license plate Detection objects
        """
        height, width = image.shape[:2]
        image_shape = (height, width)
        
        detections = []
        
        plate_bboxes, plate_confs = self.plate_detector.detect(
            image,
            confidence_threshold=confidence_threshold
        )
        
        for bbox, conf in zip(plate_bboxes, plate_confs):
            expanded_bbox = expand_bbox(bbox, padding_factor, image_shape)
            
            detection = Detection(
                bbox=expanded_bbox,
                confidence=conf,
                entity_type=EntityType.LICENSE_PLATE
            )
            detections.append(detection)
        
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections




