"""
CLI tool for batch image redaction
"""
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time
from typing import List
import json

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.detector import EntityDetector
from app.redaction import RedactionEngine
from app.schemas import RedactionMode
from app.utils import load_image, save_image


class BatchRedactor:
    """
    Batch image redaction processor
    """
    
    def __init__(self,
                 face_confidence: float = 0.9,
                 plate_confidence: float = 0.5,
                 verbose: bool = False):
        """
        Initialize batch redactor
        
        Args:
            face_confidence: Face detection confidence threshold
            plate_confidence: License plate detection confidence threshold
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        
        if self.verbose:
            print("Initializing detection models...")
        
        self.detector = EntityDetector(
            face_confidence=face_confidence,
            plate_confidence=plate_confidence
        )
        
        self.redaction_engine = RedactionEngine()
        
        if self.verbose:
            print("Models loaded successfully!")
    
    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         mode: str = 'blur',
                         confidence_threshold: float = 0.5,
                         padding_factor: float = 0.1,
                         blur_kernel_size: int = 51,
                         pixelate_block_size: int = 15,
                         formats: List[str] = None,
                         recursive: bool = False) -> dict:
        """
        Process all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            mode: Redaction mode ('blur' or 'pixelate')
            confidence_threshold: Minimum detection confidence
            padding_factor: Bounding box padding factor
            blur_kernel_size: Gaussian blur kernel size
            pixelate_block_size: Pixelation block size
            formats: List of file extensions to process
            recursive: Process subdirectories recursively
            
        Returns:
            Dictionary with processing statistics
        """
        if formats is None:
            formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        input_path = Path(input_dir)
        image_files = []
        
        if recursive:
            for ext in formats:
                image_files.extend(input_path.rglob(f'*{ext}'))
                image_files.extend(input_path.rglob(f'*{ext.upper()}'))
        else:
            for ext in formats:
                image_files.extend(input_path.glob(f'*{ext}'))
                image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return {}
        
        print(f"Found {len(image_files)} images to process")
        
        # Parse redaction mode
        try:
            redaction_mode = RedactionMode(mode.lower())
        except ValueError:
            print(f"Invalid mode: {mode}. Using 'blur'")
            redaction_mode = RedactionMode.BLUR
        
        # Process images
        stats = {
            'total': len(image_files),
            'successful': 0,
            'failed': 0,
            'total_detections': 0,
            'total_faces': 0,
            'total_plates': 0,
            'total_time': 0,
            'avg_time_per_image': 0,
            'errors': []
        }
        
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                start_time = time.time()
                
                # Load image
                image = load_image(str(image_file))
                
                # Detect entities
                detections = self.detector.detect_all(
                    image,
                    confidence_threshold=confidence_threshold,
                    padding_factor=padding_factor
                )
                
                # Apply redaction
                redacted_image = self.redaction_engine.redact(
                    image,
                    detections,
                    mode=redaction_mode,
                    blur_kernel_size=blur_kernel_size,
                    pixelate_block_size=pixelate_block_size
                )
                
                # Save output
                relative_path = image_file.relative_to(input_path)
                output_path = Path(output_dir) / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                save_image(redacted_image, str(output_path))
                
                # Update stats
                elapsed = time.time() - start_time
                stats['successful'] += 1
                stats['total_detections'] += len(detections)
                stats['total_faces'] += sum(1 for d in detections if d.entity_type.value == 'face')
                stats['total_plates'] += sum(1 for d in detections if d.entity_type.value == 'license_plate')
                stats['total_time'] += elapsed
                
                if self.verbose:
                    print(f"\n{image_file.name}: {len(detections)} detections, {elapsed*1000:.0f}ms")
                
            except Exception as e:
                stats['failed'] += 1
                error_msg = f"{image_file.name}: {str(e)}"
                stats['errors'].append(error_msg)
                
                if self.verbose:
                    print(f"\nError processing {image_file.name}: {e}")
        
        # Calculate average time
        if stats['successful'] > 0:
            stats['avg_time_per_image'] = stats['total_time'] / stats['successful']
        
        return stats
    
    def process_single_image(self,
                            input_path: str,
                            output_path: str,
                            mode: str = 'blur',
                            confidence_threshold: float = 0.5,
                            padding_factor: float = 0.1,
                            blur_kernel_size: int = 51,
                            pixelate_block_size: int = 15) -> dict:
        """
        Process a single image
        
        Args:
            input_path: Input image path
            output_path: Output image path
            mode: Redaction mode
            confidence_threshold: Minimum detection confidence
            padding_factor: Bounding box padding factor
            blur_kernel_size: Gaussian blur kernel size
            pixelate_block_size: Pixelation block size
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        # Load image
        image = load_image(input_path)
        
        # Detect entities
        detections = self.detector.detect_all(
            image,
            confidence_threshold=confidence_threshold,
            padding_factor=padding_factor
        )
        
        # Parse mode
        try:
            redaction_mode = RedactionMode(mode.lower())
        except ValueError:
            redaction_mode = RedactionMode.BLUR
        
        # Apply redaction
        redacted_image = self.redaction_engine.redact(
            image,
            detections,
            mode=redaction_mode,
            blur_kernel_size=blur_kernel_size,
            pixelate_block_size=pixelate_block_size
        )
        
        # Save output
        save_image(redacted_image, output_path)
        
        elapsed = time.time() - start_time
        
        return {
            'num_detections': len(detections),
            'num_faces': sum(1 for d in detections if d.entity_type.value == 'face'),
            'num_plates': sum(1 for d in detections if d.entity_type.value == 'license_plate'),
            'processing_time_ms': elapsed * 1000,
            'output_path': output_path
        }


def print_stats(stats: dict):
    """Print processing statistics"""
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Total images: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nTotal detections: {stats['total_detections']}")
    print(f"  Faces: {stats['total_faces']}")
    print(f"  License plates: {stats['total_plates']}")
    print(f"\nProcessing time:")
    print(f"  Total: {stats['total_time']:.2f}s")
    print(f"  Average per image: {stats['avg_time_per_image']*1000:.2f}ms")
    
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Batch image redaction tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in a directory
  python run_redaction.py --input ./images --output ./redacted
  
  # Use pixelation instead of blur
  python run_redaction.py --input ./images --output ./redacted --mode pixelate
  
  # Process single image
  python run_redaction.py --input photo.jpg --output redacted.jpg
  
  # Recursive processing with custom confidence
  python run_redaction.py --input ./images --output ./redacted --recursive --confidence 0.7
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input image file or directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output image file or directory')
    parser.add_argument('--mode', '-m', type=str, default='blur',
                       choices=['blur', 'pixelate'],
                       help='Redaction mode (default: blur)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--padding', '-p', type=float, default=0.1,
                       help='Bounding box padding factor (default: 0.1)')
    parser.add_argument('--blur-kernel', type=int, default=51,
                       help='Gaussian blur kernel size (default: 51)')
    parser.add_argument('--pixelate-block', type=int, default=15,
                       help='Pixelation block size (default: 15)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Process directories recursively')
    parser.add_argument('--format', type=str, nargs='+',
                       default=['.jpg', '.jpeg', '.png', '.bmp'],
                       help='Image formats to process (default: jpg jpeg png bmp)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--save-stats', type=str,
                       help='Save statistics to JSON file')
    
    args = parser.parse_args()
    
    # Initialize redactor
    redactor = BatchRedactor(verbose=args.verbose)
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        print(f"Processing single image: {args.input}")
        result = redactor.process_single_image(
            args.input,
            args.output,
            mode=args.mode,
            confidence_threshold=args.confidence,
            padding_factor=args.padding,
            blur_kernel_size=args.blur_kernel,
            pixelate_block_size=args.pixelate_block
        )
        
        print(f"\nProcessing complete!")
        print(f"Detections: {result['num_detections']} "
              f"({result['num_faces']} faces, {result['num_plates']} plates)")
        print(f"Processing time: {result['processing_time_ms']:.0f}ms")
        print(f"Output saved to: {result['output_path']}")
        
    elif input_path.is_dir():
        # Process directory
        stats = redactor.process_directory(
            args.input,
            args.output,
            mode=args.mode,
            confidence_threshold=args.confidence,
            padding_factor=args.padding,
            blur_kernel_size=args.blur_kernel,
            pixelate_block_size=args.pixelate_block,
            formats=args.format,
            recursive=args.recursive
        )
        
        print_stats(stats)
        
        # Save stats if requested
        if args.save_stats:
            with open(args.save_stats, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nStatistics saved to {args.save_stats}")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()




