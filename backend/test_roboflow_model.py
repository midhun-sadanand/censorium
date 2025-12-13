"""
Quick test script to verify Roboflow license plate model is working
"""
import sys
import os
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.detector_plate import LicensePlateDetector
import cv2
import numpy as np

def test_model_loading():
    """Test that the Roboflow model loads correctly"""
    print("=" * 60)
    print("Testing Roboflow License Plate Model")
    print("=" * 60)
    
    try:
        print("\n1. Loading model...")
        detector = LicensePlateDetector()
        print("   ✅ Model loaded successfully!")
        print(f"   - Using custom Roboflow model: {detector.is_custom_model}")
        print(f"   - Confidence threshold: {detector.confidence_threshold}")
        
        print("\n2. Creating test image...")
        # Create a dummy test image (blank image for initialization test)
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        print("   ✅ Test image created")
        
        print("\n3. Running inference test...")
        bboxes, confidences = detector.detect(test_image)
        print(f"   ✅ Inference successful!")
        print(f"   - Detections: {len(bboxes)}")
        
        if len(bboxes) > 0:
            print("\n   Detected bounding boxes:")
            for i, (bbox, conf) in enumerate(zip(bboxes, confidences)):
                print(f"   - Detection {i+1}: bbox={bbox}, confidence={conf:.3f}")
        else:
            print("   - No license plates detected (expected for blank image)")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour Roboflow model is ready to use!")
        print("Start the API with: uvicorn app.main:app --reload")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Model file not found")
        print(f"   {e}")
        print("\nMake sure license_plate.pt is in backend/models/")
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)


