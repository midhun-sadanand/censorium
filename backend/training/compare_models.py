#!/usr/bin/env python3
"""
Model Comparison Script

Compares custom-trained YOLOv8 models with Roboflow-trained models on the same test set.
This allows quantitative comparison of training approaches and helps determine which
model performs better for the license plate detection task.

Usage:
    python compare_models.py --custom runs/detect/custom_train/weights/best.pt --roboflow ../models/license_plate.pt --data datasets/license_plates/data.yaml
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
from datetime import datetime

import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare two YOLOv8 models on the same test dataset
    """
    
    def __init__(
        self,
        custom_model_path: str,
        roboflow_model_path: str,
        data_yaml: str,
        test_images_dir: Optional[str] = None,
        confidence_threshold: float = 0.4
    ):
        """
        Initialize model comparator
        
        Args:
            custom_model_path: Path to custom-trained model
            roboflow_model_path: Path to Roboflow-trained model
            data_yaml: Path to data.yaml configuration
            test_images_dir: Optional path to test images (if not in data.yaml)
            confidence_threshold: Confidence threshold for detection
        """
        self.custom_model_path = Path(custom_model_path)
        self.roboflow_model_path = Path(roboflow_model_path)
        self.data_yaml = Path(data_yaml)
        self.confidence_threshold = confidence_threshold
        
        # Validate paths
        if not self.custom_model_path.exists():
            raise FileNotFoundError(f"Custom model not found: {self.custom_model_path}")
        if not self.roboflow_model_path.exists():
            raise FileNotFoundError(f"Roboflow model not found: {self.roboflow_model_path}")
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Data config not found: {self.data_yaml}")
        
        # Load models
        logger.info("Loading models...")
        logger.info(f"  Custom model: {self.custom_model_path}")
        logger.info(f"  Roboflow model: {self.roboflow_model_path}")
        
        self.custom_model = YOLO(str(self.custom_model_path))
        self.roboflow_model = YOLO(str(self.roboflow_model_path))
        
        # Get test images directory
        if test_images_dir:
            self.test_images_dir = Path(test_images_dir)
        else:
            # Try to extract from data.yaml
            try:
                import yaml
            except ImportError:
                # Fallback: parse YAML manually for simple cases
                with open(self.data_yaml, 'r') as f:
                    content = f.read()
                    # Simple extraction of test path
                    for line in content.split('\n'):
                        if line.strip().startswith('test:'):
                            test_path = line.split(':', 1)[1].strip()
                            break
                    else:
                        test_path = ''
            else:
                with open(self.data_yaml, 'r') as f:
                    data_config = yaml.safe_load(f)
                test_path = data_config.get('test', '')
            
            if test_path:
                # Handle relative paths
                if not Path(test_path).is_absolute():
                    self.test_images_dir = self.data_yaml.parent / test_path
                else:
                    self.test_images_dir = Path(test_path)
            else:
                # Default to standard location
                self.test_images_dir = self.data_yaml.parent / "test" / "images"
                if not self.test_images_dir.exists():
                    raise ValueError("Could not determine test images directory. Provide --test-images-dir")
        
        if not self.test_images_dir.exists():
            raise FileNotFoundError(f"Test images directory not found: {self.test_images_dir}")
        
        logger.info(f"Test images directory: {self.test_images_dir}")
    
    def evaluate_model(self, model: YOLO, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a model on the test set
        
        Args:
            model: YOLO model instance
            model_name: Name for logging
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        # Run validation
        start_time = time.time()
        results = model.val(
            data=str(self.data_yaml),
            split='test',
            conf=self.confidence_threshold,
            verbose=True
        )
        eval_time = time.time() - start_time
        
        # Extract metrics
        metrics = {
            'model_name': model_name,
            'evaluation_time_seconds': eval_time,
        }
        
        # Try to extract box metrics
        if hasattr(results, 'box'):
            box_metrics = results.box
            metrics.update({
                'mAP50': float(box_metrics.mAP50) if hasattr(box_metrics, 'mAP50') else None,
                'mAP50-95': float(box_metrics.mAP50_95) if hasattr(box_metrics, 'mAP50_95') else None,
                'precision': float(box_metrics.mp) if hasattr(box_metrics, 'mp') else None,
                'recall': float(box_metrics.mr) if hasattr(box_metrics, 'mr') else None,
            })
        elif hasattr(results, 'results_dict'):
            # Alternative format
            results_dict = results.results_dict
            metrics.update({
                'mAP50': results_dict.get('metrics/mAP50(B)', None),
                'mAP50-95': results_dict.get('metrics/mAP50-95(B)', None),
                'precision': results_dict.get('metrics/precision(B)', None),
                'recall': results_dict.get('metrics/recall(B)', None),
            })
        
        return metrics
    
    def benchmark_inference_speed(self, model: YOLO, model_name: str, num_images: int = 50) -> Dict[str, float]:
        """
        Benchmark inference speed on test images
        
        Args:
            model: YOLO model instance
            model_name: Name for logging
            num_images: Number of images to test
            
        Returns:
            Dictionary with speed metrics
        """
        logger.info(f"\nBenchmarking inference speed for {model_name}...")
        
        # Get test images
        test_images = list(self.test_images_dir.glob("*.jpg")) + list(self.test_images_dir.glob("*.png"))
        test_images = test_images[:num_images]
        
        if not test_images:
            logger.warning(f"No test images found in {self.test_images_dir}")
            return {}
        
        times = []
        for img_path in tqdm(test_images, desc=f"Benchmarking {model_name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            start = time.time()
            _ = model(img, conf=self.confidence_threshold, verbose=False)
            times.append(time.time() - start)
        
        if not times:
            return {}
        
        times = np.array(times) * 1000  # Convert to milliseconds
        
        return {
            'mean_inference_time_ms': float(np.mean(times)),
            'std_inference_time_ms': float(np.std(times)),
            'min_inference_time_ms': float(np.min(times)),
            'max_inference_time_ms': float(np.max(times)),
            'fps': float(1000.0 / np.mean(times))
        }
    
    def compare(self, benchmark_speed: bool = True) -> Dict[str, Any]:
        """
        Compare both models
        
        Args:
            benchmark_speed: Whether to benchmark inference speed
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info("=" * 60)
        logger.info("Model Comparison: Custom vs Roboflow")
        logger.info("=" * 60)
        
        # Evaluate both models
        custom_metrics = self.evaluate_model(self.custom_model, "Custom Model")
        roboflow_metrics = self.evaluate_model(self.roboflow_model, "Roboflow Model")
        
        # Benchmark speed if requested
        if benchmark_speed:
            custom_speed = self.benchmark_inference_speed(self.custom_model, "Custom Model")
            roboflow_speed = self.benchmark_inference_speed(self.roboflow_model, "Roboflow Model")
            
            custom_metrics.update(custom_speed)
            roboflow_metrics.update(roboflow_speed)
        
        # Create comparison
        comparison = {
            'comparison_date': datetime.now().isoformat(),
            'custom_model_path': str(self.custom_model_path),
            'roboflow_model_path': str(self.roboflow_model_path),
            'test_images_dir': str(self.test_images_dir),
            'confidence_threshold': self.confidence_threshold,
            'custom_model': custom_metrics,
            'roboflow_model': roboflow_metrics,
        }
        
        # Calculate differences
        if custom_metrics.get('mAP50') and roboflow_metrics.get('mAP50'):
            comparison['differences'] = {
                'mAP50_diff': float(custom_metrics['mAP50'] - roboflow_metrics['mAP50']),
                'precision_diff': float(custom_metrics.get('precision', 0) - roboflow_metrics.get('precision', 0)) if custom_metrics.get('precision') and roboflow_metrics.get('precision') else None,
                'recall_diff': float(custom_metrics.get('recall', 0) - roboflow_metrics.get('recall', 0)) if custom_metrics.get('recall') and roboflow_metrics.get('recall') else None,
            }
        
        # Print summary
        self.print_comparison_summary(comparison)
        
        return comparison
    
    def print_comparison_summary(self, comparison: Dict[str, Any]):
        """Print a formatted comparison summary"""
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 60)
        
        custom = comparison['custom_model']
        roboflow = comparison['roboflow_model']
        
        logger.info("\nAccuracy Metrics:")
        logger.info(f"{'Metric':<20} {'Custom':<15} {'Roboflow':<15} {'Difference':<15}")
        logger.info("-" * 65)
        
        if custom.get('mAP50') and roboflow.get('mAP50'):
            diff = custom['mAP50'] - roboflow['mAP50']
            logger.info(f"{'mAP@50':<20} {custom['mAP50']:<15.4f} {roboflow['mAP50']:<15.4f} {diff:+.4f}")
        
        if custom.get('precision') and roboflow.get('precision'):
            diff = custom['precision'] - roboflow['precision']
            logger.info(f"{'Precision':<20} {custom['precision']:<15.4f} {roboflow['precision']:<15.4f} {diff:+.4f}")
        
        if custom.get('recall') and roboflow.get('recall'):
            diff = custom['recall'] - roboflow['recall']
            logger.info(f"{'Recall':<20} {custom['recall']:<15.4f} {roboflow['recall']:<15.4f} {diff:+.4f}")
        
        if custom.get('mean_inference_time_ms') and roboflow.get('mean_inference_time_ms'):
            logger.info("\nSpeed Metrics:")
            logger.info(f"{'Metric':<30} {'Custom':<15} {'Roboflow':<15}")
            logger.info("-" * 60)
            logger.info(f"{'Mean Inference (ms)':<30} {custom['mean_inference_time_ms']:<15.2f} {roboflow['mean_inference_time_ms']:<15.2f}")
            logger.info(f"{'FPS':<30} {custom.get('fps', 0):<15.2f} {roboflow.get('fps', 0):<15.2f}")
        
        # Determine winner
        logger.info("\nWinner:")
        if custom.get('mAP50') and roboflow.get('mAP50'):
            if custom['mAP50'] > roboflow['mAP50']:
                logger.info("  Custom Model wins on mAP@50")
            elif roboflow['mAP50'] > custom['mAP50']:
                logger.info("  Roboflow Model wins on mAP@50")
            else:
                logger.info("  Tie on mAP@50")
        
        logger.info("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compare Custom vs Roboflow YOLOv8 Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python compare_models.py --custom runs/detect/custom_train/weights/best.pt --roboflow ../models/license_plate.pt --data datasets/license_plates/data.yaml
  
  # With custom test directory
  python compare_models.py --custom best.pt --roboflow ../models/license_plate.pt --data data.yaml --test-images-dir datasets/license_plates/test/images
        """
    )
    
    parser.add_argument(
        '--custom',
        type=str,
        required=True,
        help='Path to custom-trained model weights'
    )
    
    parser.add_argument(
        '--roboflow',
        type=str,
        required=True,
        help='Path to Roboflow-trained model weights'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data.yaml configuration file'
    )
    
    parser.add_argument(
        '--test-images-dir',
        type=str,
        default=None,
        help='Path to test images directory (optional, extracted from data.yaml if not provided)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.4,
        help='Confidence threshold for evaluation (default: 0.4)'
    )
    
    parser.add_argument(
        '--no-speed-benchmark',
        action='store_true',
        help='Skip inference speed benchmarking'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save comparison results JSON (default: comparison_results.json)'
    )
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = ModelComparator(
        custom_model_path=args.custom,
        roboflow_model_path=args.roboflow,
        data_yaml=args.data,
        test_images_dir=args.test_images_dir,
        confidence_threshold=args.confidence
    )
    
    # Run comparison
    results = comparator.compare(benchmark_speed=not args.no_speed_benchmark)
    
    # Save results
    output_path = args.output or "comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nComparison results saved to: {output_path}")


if __name__ == "__main__":
    main()

