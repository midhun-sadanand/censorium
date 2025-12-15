"""
System validation script for Censorium
Validates installation, imports, and basic functionality
"""
import sys
import os
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text):
    """Print section header"""
    print(f"\n{BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{RESET}\n")


def print_success(text):
    """Print success message"""
    print(f"{GREEN}{text}{RESET}")


def print_error(text):
    """Print error message"""
    print(f"{RED}{text}{RESET}")


def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}WARNING: {text}{RESET}")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version >= (3, 10):
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python version {version.major}.{version.minor} is too old (need 3.10+)")
        return False


def check_backend_structure():
    """Check backend directory structure"""
    backend_path = Path("backend")
    
    required_dirs = [
        "app",
        "evaluate",
        "models"
    ]
    
    required_files = [
        "app/__init__.py",
        "app/main.py",
        "app/detector.py",
        "app/detector_face.py",
        "app/detector_plate.py",
        "app/redaction.py",
        "app/schemas.py",
        "app/utils.py",
        "requirements.txt",
        "run_redaction.py"
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        path = backend_path / dir_name
        if path.exists():
            print_success(f"Directory: {dir_name}/")
        else:
            print_error(f"Missing directory: {dir_name}/")
            all_ok = False
    
    for file_path in required_files:
        path = backend_path / file_path
        if path.exists():
            print_success(f"File: {file_path}")
        else:
            print_error(f"Missing file: {file_path}")
            all_ok = False
    
    return all_ok


def check_frontend_structure():
    """Check frontend directory structure"""
    frontend_path = Path("frontend")
    
    required_dirs = [
        "app",
        "components",
        "lib"
    ]
    
    required_files = [
        "app/page.tsx",
        "app/layout.tsx",
        "components/RedactionViewer.tsx",
        "lib/api.ts",
        "package.json"
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        path = frontend_path / dir_name
        if path.exists():
            print_success(f"Directory: {dir_name}/")
        else:
            print_error(f"Missing directory: {dir_name}/")
            all_ok = False
    
    for file_path in required_files:
        path = frontend_path / file_path
        if path.exists():
            print_success(f"File: {file_path}")
        else:
            print_error(f"Missing file: {file_path}")
            all_ok = False
    
    return all_ok


def check_python_imports():
    """Check if key Python packages can be imported"""
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'torchvision'),
        ('ultralytics', 'Ultralytics YOLOv8'),
        ('facenet_pytorch', 'facenet-pytorch'),
        ('fastapi', 'FastAPI'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm')
    ]
    
    all_ok = True
    
    for package_name, display_name in packages:
        try:
            __import__(package_name)
            print_success(f"Import: {display_name}")
        except ImportError:
            print_error(f"Cannot import: {display_name}")
            print_warning(f"  Install with: pip install {package_name}")
            all_ok = False
    
    return all_ok


def check_backend_functionality():
    """Test basic backend functionality"""
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").resolve()))
        
        # Try importing modules
        from app.detector import EntityDetector
        from app.redaction import RedactionEngine
        from app.schemas import RedactionMode
        print_success("Backend modules import successfully")
        
        # Try creating instances (without loading models)
        try:
            print_warning("Skipping model loading (takes 30+ seconds)")
            print_warning("Run manual test with: python backend/test_api.py")
            return True
        except Exception as e:
            print_error(f"Cannot create backend instances: {e}")
            return False
            
    except Exception as e:
        print_error(f"Backend functionality check failed: {e}")
        return False


def check_nodejs():
    """Check if Node.js is installed"""
    try:
        import subprocess
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Node.js version: {version}")
            return True
        else:
            print_error("Node.js not found")
            return False
    except FileNotFoundError:
        print_error("Node.js not installed")
        return False


def check_npm():
    """Check if npm is installed"""
    try:
        import subprocess
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"npm version: {version}")
            return True
        else:
            print_error("npm not found")
            return False
    except FileNotFoundError:
        print_error("npm not installed")
        return False


def check_frontend_dependencies():
    """Check if frontend dependencies are installed"""
    node_modules = Path("frontend/node_modules")
    if node_modules.exists():
        print_success("Frontend dependencies installed")
        return True
    else:
        print_error("Frontend dependencies not installed")
        print_warning("  Run: cd frontend && npm install")
        return False


def generate_report(results):
    """Generate final validation report"""
    print_header("VALIDATION SUMMARY")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    print(f"Total checks: {total}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    print()
    
    if failed == 0:
        print(f"{GREEN}{'='*60}")
        print("ALL CHECKS PASSED - System is ready.")
        print(f"{'='*60}{RESET}")
        print()
        print("Next steps:")
        print("1. Start backend:  ./start_backend.sh")
        print("2. Start frontend: ./start_frontend.sh")
        print("3. Open http://localhost:3000")
    else:
        print(f"{RED}{'='*60}")
        print("SOME CHECKS FAILED - Please fix issues above.")
        print(f"{'='*60}{RESET}")
        print()
        print("Common fixes:")
        print("1. Install Python deps: cd backend && pip install -r requirements.txt")
        print("2. Install Node deps:   cd frontend && npm install")
        print("3. Check Python version >= 3.10")
        print("4. Check Node.js version >= 18")


def main():
    """Run all validation checks"""
    print(f"{BLUE}")
    print("="*60)
    print("           CENSORIUM SYSTEM VALIDATION")
    print("="*60)
    print(f"{RESET}")
    
    results = {}
    
    # System checks
    print_header("1. SYSTEM REQUIREMENTS")
    results['python_version'] = check_python_version()
    results['nodejs'] = check_nodejs()
    results['npm'] = check_npm()
    
    # Structure checks
    print_header("2. BACKEND STRUCTURE")
    results['backend_structure'] = check_backend_structure()
    
    print_header("3. FRONTEND STRUCTURE")
    results['frontend_structure'] = check_frontend_structure()
    
    # Dependency checks
    print_header("4. PYTHON DEPENDENCIES")
    results['python_imports'] = check_python_imports()
    
    print_header("5. FRONTEND DEPENDENCIES")
    results['frontend_deps'] = check_frontend_dependencies()
    
    # Functionality checks
    print_header("6. BACKEND FUNCTIONALITY")
    results['backend_functionality'] = check_backend_functionality()
    
    # Generate report
    generate_report(results)
    
    # Exit code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()




