"""
Benchmarking tools for Censorium
"""
import argparse
import time
import numpy as np
import psutil
import json
from pathlib import Path
from typing import Dict, List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.detector import EntityDetector
from app.redaction import RedactionEngine
from app.schemas import RedactionMode
from app.utils import load_image


class Benchmark:
    """
    Comprehensive benchmarking suite
    """
    
    def __init__(self):
        """Initialize benchmark"""
        self.detector = EntityDetector()
        self.redaction_engine = RedactionEngine()
    
    def benchmark_throughput(self, 
                            image_paths: List[str],
                            num_iterations: int = 100) -> Dict:
        """
        Measure throughput (images per second)
        
        Args:
            image_paths: List of image paths to test
            num_iterations: Number of iterations
            
        Returns:
            Dictionary with throughput metrics
        """
        print(f"Running throughput benchmark ({num_iterations} iterations)...")
        
        # Load images
        images = [load_image(path) for path in image_paths[:10]]  # Limit to 10 images
        
        start_time = time.time()
        
        for _ in range(num_iterations):
            for image in images:
                detections = self.detector.detect_all(image, confidence_threshold=0.5)
                _ = self.redaction_engine.redact(image, detections, RedactionMode.BLUR)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        total_images = num_iterations * len(images)
        throughput = total_images / elapsed
        
        return {
            'total_images': total_images,
            'elapsed_time_seconds': elapsed,
            'throughput_images_per_second': throughput,
            'avg_time_per_image_ms': (elapsed / total_images) * 1000
        }
    
    def benchmark_memory(self, image_paths: List[str]) -> Dict:
        """
        Measure memory usage
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Memory usage metrics
        """
        print("Running memory benchmark...")
        
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process images
        for path in image_paths[:50]:  # Limit to 50 images
            try:
                image = load_image(path)
                detections = self.detector.detect_all(image)
                _ = self.redaction_engine.redact(image, detections)
            except:
                continue
        
        # Peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': peak_memory - baseline_memory
        }
    
    def benchmark_latency_distribution(self, 
                                      image_paths: List[str],
                                      num_samples: int = 200) -> Dict:
        """
        Measure latency distribution across different image sizes
        
        Args:
            image_paths: List of image paths
            num_samples: Number of samples to measure
            
        Returns:
            Latency distribution metrics
        """
        print(f"Running latency distribution benchmark ({num_samples} samples)...")
        
        latencies = []
        image_sizes = []
        
        for path in image_paths[:num_samples]:
            try:
                image = load_image(path)
                height, width = image.shape[:2]
                image_sizes.append((width, height))
                
                start_time = time.time()
                detections = self.detector.detect_all(image)
                _ = self.redaction_engine.redact(image, detections)
                latency = (time.time() - start_time) * 1000
                
                latencies.append(latency)
            except:
                continue
        
        latencies = np.array(latencies)
        
        return {
            'num_samples': len(latencies),
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p90_ms': float(np.percentile(latencies, 90)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99))
        }
    
    def run_full_benchmark(self, image_dir: str) -> Dict:
        """
        Run complete benchmark suite
        
        Args:
            image_dir: Directory containing test images
            
        Returns:
            Complete benchmark results
        """
        # Find all images
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(list(Path(image_dir).glob(ext)))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(image_paths)} images for benchmarking")
        
        results = {
            'num_test_images': len(image_paths),
            'throughput': self.benchmark_throughput(image_paths, num_iterations=10),
            'memory': self.benchmark_memory(image_paths),
            'latency': self.benchmark_latency_distribution(image_paths, num_samples=min(200, len(image_paths)))
        }
        
        return results


def print_results(results: Dict):
    """Pretty print benchmark results"""
    print("\n" + "="*60)
    print("CENSORIUM BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\nTest Images: {results['num_test_images']}")
    
    print("\n--- THROUGHPUT ---")
    t = results['throughput']
    print(f"Images Processed: {t['total_images']}")
    print(f"Total Time: {t['elapsed_time_seconds']:.2f}s")
    print(f"Throughput: {t['throughput_images_per_second']:.2f} images/sec")
    print(f"Avg Time per Image: {t['avg_time_per_image_ms']:.2f}ms")
    
    print("\n--- MEMORY ---")
    m = results['memory']
    print(f"Baseline: {m['baseline_memory_mb']:.2f} MB")
    print(f"Peak: {m['peak_memory_mb']:.2f} MB")
    print(f"Increase: {m['memory_increase_mb']:.2f} MB")
    
    print("\n--- LATENCY DISTRIBUTION ---")
    l = results['latency']
    print(f"Samples: {l['num_samples']}")
    print(f"Mean: {l['mean_latency_ms']:.2f}ms")
    print(f"Median: {l['median_latency_ms']:.2f}ms")
    print(f"Std Dev: {l['std_latency_ms']:.2f}ms")
    print(f"Min: {l['min_latency_ms']:.2f}ms")
    print(f"Max: {l['max_latency_ms']:.2f}ms")
    print(f"P90: {l['p90_ms']:.2f}ms")
    print(f"P95: {l['p95_ms']:.2f}ms")
    print(f"P99: {l['p99_ms']:.2f}ms")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run Censorium benchmarks')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory with test images')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = Benchmark()
    results = benchmark.run_full_benchmark(args.image_dir)
    
    # Print results
    print_results(results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()




