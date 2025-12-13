"""
Simple test script to verify API is working
"""
import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("Health check passed")
            print(f"  Status: {data['status']}")
            print(f"  Models loaded: {data['models_loaded']}")
            return True
        else:
            print(f"ERROR: Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API. Is it running?")
        print(f"  Try: cd backend && python -m uvicorn app.main:app")
        return False
    except Exception as e:
        print(f"ERROR: Health check error: {e}")
        return False


def test_redact_image(image_path: str):
    """Test image redaction endpoint"""
    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'mode': 'blur',
                'confidence_threshold': 0.5,
                'return_metadata': 'true'
            }
            
            response = requests.post(
                f"{API_URL}/redact-image",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print("Redaction test passed")
            print(f"  Detections: {len(result['detections'])}")
            print(f"  Processing time: {result['processing_time_ms']:.0f}ms")
            
            # Print detection breakdown
            faces = sum(1 for d in result['detections'] if d['entity_type'] == 'face')
            plates = sum(1 for d in result['detections'] if d['entity_type'] == 'license_plate')
            print(f"  Faces: {faces}, Plates: {plates}")
            return True
        else:
            print(f"ERROR: Redaction test failed: {response.status_code}")
            print(f"  {response.text}")
            return False
    except Exception as e:
        print(f"ERROR: Redaction test error: {e}")
        return False


def test_stats():
    """Test stats endpoint"""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("Stats endpoint passed")
            print(f"  Status: {data['status']}")
            print(f"  Supported modes: {', '.join(data['supported_modes'])}")
            print(f"  Supported entities: {', '.join(data['supported_entities'])}")
            return True
        else:
            print(f"ERROR: Stats endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: Stats endpoint error: {e}")
        return False


def main():
    print("="*60)
    print("CENSORIUM API TESTS")
    print("="*60)
    print()
    
    # Test 1: Health check
    print("Test 1: Health Check")
    health_ok = test_health()
    print()
    
    if not health_ok:
        print("Cannot proceed with other tests - API not responding")
        sys.exit(1)
    
    # Test 2: Stats endpoint
    print("Test 2: Stats Endpoint")
    test_stats()
    print()
    
    # Test 3: Image redaction (if test image provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Test 3: Image Redaction ({image_path})")
        test_redact_image(image_path)
    else:
        print("Test 3: Image Redaction (skipped - no image provided)")
        print("  Usage: python test_api.py <path_to_test_image>")
    
    print()
    print("="*60)
    print("TESTS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()




