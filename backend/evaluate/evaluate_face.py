"""
Face detection evaluation using WIDER FACE dataset
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.detector_face import FaceDetector
from app.utils import load_image, calculate_iou


class FaceEvaluator:
    """
    Evaluator for face detection performance
    """
    
    def __init__(self, detector: FaceDetector, iou_threshold: float = 0.5):
        """
        Initialize evaluator
        
        Args:
            detector: Face detector instance
            iou_threshold: IoU threshold for considering a detection as correct
        """
        self.detector = detector
        self.iou_threshold = iou_threshold
    
    def evaluate_image(self,
                       image_path: str,
                       ground_truth_bboxes: List[Tuple[int, int, int, int]],
                       confidence_threshold: float = 0.5) -> Dict:
        """
        Evaluate detection on a single image
        
        Args:
            image_path: Path to image
            ground_truth_bboxes: List of ground truth bounding boxes
            confidence_threshold: Detection confidence threshold
            
        Returns:
            Dictionary with evaluation metrics for this image
        """
        # Load image
        image = load_image(image_path)
        
        # Measure inference time
        start_time = time.time()
        detected_bboxes, confidences = self.detector.detect(
            image, 
            confidence_threshold=confidence_threshold
        )
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Calculate metrics
        true_positives = 0
        false_positives = 0
        matched_gt = set()
        
        for det_bbox in detected_bboxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_bbox in enumerate(ground_truth_bboxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(det_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= self.iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1
        
        false_negatives = len(ground_truth_bboxes) - true_positives
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'num_detections': len(detected_bboxes),
            'num_ground_truth': len(ground_truth_bboxes),
            'inference_time_ms': inference_time
        }
    
    def evaluate_dataset(self,
                        dataset_path: str,
                        annotations_file: str,
                        confidence_threshold: float = 0.5,
                        max_images: int = None) -> Dict:
        """
        Evaluate on entire dataset
        
        Args:
            dataset_path: Path to dataset root
            annotations_file: Path to annotations file (JSON format)
            confidence_threshold: Detection confidence threshold
            max_images: Maximum number of images to evaluate (None = all)
            
        Returns:
            Dictionary with aggregated metrics
        """
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        inference_times = []
        
        # Process images
        images_to_process = list(annotations.items())
        if max_images:
            images_to_process = images_to_process[:max_images]
        
        for image_name, gt_data in tqdm(images_to_process, desc="Evaluating"):
            image_path = os.path.join(dataset_path, image_name)
            
            if not os.path.exists(image_path):
                continue
            
            gt_bboxes = gt_data.get('bboxes', [])
            
            try:
                result = self.evaluate_image(
                    image_path,
                    gt_bboxes,
                    confidence_threshold
                )
                
                total_tp += result['true_positives']
                total_fp += result['false_positives']
                total_fn += result['false_negatives']
                inference_times.append(result['inference_time_ms'])
                
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'mean_inference_time_ms': np.mean(inference_times),
            'median_inference_time_ms': np.median(inference_times),
            'p95_inference_time_ms': np.percentile(inference_times, 95),
            'total_images': len(images_to_process)
        }


def create_sample_dataset(output_dir: str, num_images: int = 50):
    """
    Create a sample dataset with synthetic annotations for testing
    
    Args:
        output_dir: Directory to save sample data
        num_images: Number of sample images to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample annotations
    annotations = {}
    
    for i in range(num_images):
        image_name = f"sample_{i:04d}.jpg"
        
        # Generate random bounding boxes
        num_faces = np.random.randint(0, 5)
        bboxes = []
        
        for _ in range(num_faces):
            x1 = np.random.randint(0, 800)
            y1 = np.random.randint(0, 600)
            w = np.random.randint(50, 200)
            h = np.random.randint(50, 200)
            bboxes.append([x1, y1, x1 + w, y1 + h])
        
        annotations[image_name] = {'bboxes': bboxes}
    
    # Save annotations
    annotations_path = os.path.join(output_dir, 'annotations.json')
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Sample dataset created at {output_dir}")
    print(f"Annotations saved to {annotations_path}")


def plot_results(results: Dict, output_path: str):
    """
    Plot evaluation results
    
    Args:
        results: Results dictionary from evaluate_dataset
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Metrics bar chart
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [results['precision'], results['recall'], results['f1_score']]
    
    axes[0].bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c'])
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Score')
    axes[0].set_title('Detection Metrics')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Latency stats
    latency_metrics = ['Mean', 'Median', 'P95']
    latency_values = [
        results['mean_inference_time_ms'],
        results['median_inference_time_ms'],
        results['p95_inference_time_ms']
    ]
    
    axes[1].bar(latency_metrics, latency_values, color=['#9b59b6', '#f39c12', '#e67e22'])
    axes[1].set_ylabel('Time (ms)')
    axes[1].set_title('Inference Latency')
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(latency_values):
        axes[1].text(i, v + 5, f'{v:.1f}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate face detection performance')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--annotations', type=str, help='Path to annotations JSON file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to evaluate')
    parser.add_argument('--output', type=str, default='face_eval_results.json', help='Output JSON file')
    parser.add_argument('--plot', type=str, default='face_eval_plot.png', help='Output plot file')
    parser.add_argument('--create-sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--sample-dir', type=str, default='sample_face_dataset', help='Sample dataset directory')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset(args.sample_dir)
        return
    
    if not args.dataset or not args.annotations:
        print("Error: --dataset and --annotations are required (or use --create-sample)")
        return
    
    # Initialize detector
    print("Initializing face detector...")
    detector = FaceDetector(confidence_threshold=args.confidence)
    
    # Initialize evaluator
    evaluator = FaceEvaluator(detector, iou_threshold=0.5)
    
    # Run evaluation
    print(f"Evaluating on {args.dataset}...")
    results = evaluator.evaluate_dataset(
        args.dataset,
        args.annotations,
        confidence_threshold=args.confidence,
        max_images=args.max_images
    )
    
    # Print results
    print("\n" + "="*50)
    print("FACE DETECTION EVALUATION RESULTS")
    print("="*50)
    print(f"Total Images: {results['total_images']}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1 Score: {results['f1_score']:.3f}")
    print(f"True Positives: {results['true_positives']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"False Negatives: {results['false_negatives']}")
    print(f"\nInference Time:")
    print(f"  Mean: {results['mean_inference_time_ms']:.2f} ms")
    print(f"  Median: {results['median_inference_time_ms']:.2f} ms")
    print(f"  P95: {results['p95_inference_time_ms']:.2f} ms")
    print("="*50)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Plot results
    plot_results(results, args.plot)


if __name__ == '__main__':
    main()




