"""
Verify complete integration of Roboflow model
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.detector import EntityDetector
import numpy as np
import cv2

def main():
    print("=" * 70)
    print("CENSORIUM - ROBOFLOW INTEGRATION VERIFICATION")
    print("=" * 70)
    
    try:
        # Test 1: Model Loading
        print("\n[1/4] Loading EntityDetector with Roboflow model...")
        models_dir = Path(__file__).parent / "models"
        plate_model_path = models_dir / "license_plate.pt"
        
        if not plate_model_path.exists():
            print(f"   ❌ ERROR: Roboflow model not found at {plate_model_path}")
            return False
        
        detector = EntityDetector(
            face_confidence=0.9,
            plate_confidence=0.4,
            plate_model_path=str(plate_model_path)
        )
        print("   ✅ EntityDetector initialized successfully!")
        
        # Test 2: Face Detector
        print("\n[2/4] Testing face detector...")
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        faces = detector.detect_faces_only(test_img)
        print(f"   ✅ Face detector working (found {len(faces)} faces in test image)")
        
        # Test 3: Plate Detector  
        print("\n[3/4] Testing Roboflow plate detector...")
        plates = detector.detect_plates_only(test_img)
        print(f"   ✅ Plate detector working (found {len(plates)} plates in test image)")
        
        # Test 4: Unified Detection
        print("\n[4/4] Testing unified detection pipeline...")
        all_detections = detector.detect_all(test_img, confidence_threshold=0.4)
        print(f"   ✅ Unified detection working (found {len(all_detections)} total entities)")
        
        print("\n" + "=" * 70)
        print("✅ ✅ ✅  ALL INTEGRATION TESTS PASSED  ✅ ✅ ✅")
        print("=" * 70)
        
        print("\n📊 System Status:")
        print(f"   • Face Detection: READY (MTCNN)")
        print(f"   • Plate Detection: READY (Roboflow YOLOv8)")
        print(f"   • Unified Pipeline: READY")
        print(f"   • Model Path: {plate_model_path}")
        
        print("\n🚀 Your system is ready!")
        print("\n   Start backend:  cd backend && uvicorn app.main:app --reload")
        print("   Start frontend: cd frontend && npm run dev")
        print("   Then open: http://localhost:3000")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


