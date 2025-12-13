#!/usr/bin/env python3
"""
Custom YOLOv8 Training Pipeline

This script provides a programmatic training pipeline for YOLOv8 license plate detection,
allowing direct comparison with Roboflow-trained models. It uses the Ultralytics YOLO API
programmatically rather than CLI commands, enabling better integration and customization.

Usage:
    python train_custom.py --data datasets/license_plates/data.yaml --epochs 85
    python train_custom.py --data datasets/license_plates/data.yaml --epochs 85 --resume runs/detect/custom_train/weights/last.pt
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomTrainer:
    """
    Custom YOLOv8 trainer that replicates Roboflow training configuration
    """
    
    def __init__(
        self,
        data_yaml: str,
        model: str = "yolov8n.pt",
        epochs: int = 85,
        imgsz: int = 512,
        batch: int = 16,
        device: Optional[str] = None,
        project: str = "runs/detect",
        name: str = "custom_train",
        patience: int = 50,
        save_period: int = 10,
        resume: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize custom trainer
        
        Args:
            data_yaml: Path to data.yaml configuration file
            model: Path to pretrained model or model name (yolov8n.pt, yolov8s.pt, etc.)
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            device: Device to use ('cpu', 'cuda', '0', '1', etc.) or None for auto-detect
            project: Project directory for outputs
            name: Experiment name
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            resume: Path to checkpoint to resume from
            **kwargs: Additional training arguments
        """
        self.data_yaml = Path(data_yaml)
        self.model_path = model
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.project = project
        self.name = name
        self.patience = patience
        self.save_period = save_period
        self.resume = resume
        self.kwargs = kwargs
        
        # Validate data.yaml exists
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Data configuration not found: {self.data_yaml}")
        
        # Validate model exists (if it's a file path)
        if self.model_path.endswith('.pt') and not Path(self.model_path).exists():
            # Check if it's in the training directory
            training_dir = Path(__file__).parent
            potential_path = training_dir / self.model_path
            if potential_path.exists():
                self.model_path = str(potential_path)
            else:
                logger.warning(f"Model file not found: {self.model_path}, will download if needed")
        
        logger.info(f"Initializing CustomTrainer")
        logger.info(f"  Data: {self.data_yaml}")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Image size: {self.imgsz}")
        logger.info(f"  Batch size: {self.batch}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Project: {self.project}")
        logger.info(f"  Name: {self.name}")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model
        
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Custom YOLOv8 Training")
        logger.info("=" * 60)
        
        # Load model
        logger.info(f"Loading model: {self.model_path}")
        model = YOLO(self.model_path)
        
        # Prepare training arguments
        train_args = {
            'data': str(self.data_yaml),
            'epochs': self.epochs,
            'imgsz': self.imgsz,
            'batch': self.batch,
            'device': self.device,
            'project': self.project,
            'name': self.name,
            'patience': self.patience,
            'save_period': self.save_period,
            'verbose': True,
            **self.kwargs
        }
        
        # Add resume if specified
        if self.resume:
            train_args['resume'] = self.resume
            logger.info(f"Resuming training from: {self.resume}")
        
        logger.info("Training configuration:")
        for key, value in train_args.items():
            logger.info(f"  {key}: {value}")
        
        # Start training
        logger.info("\n" + "=" * 60)
        logger.info("Training started at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("=" * 60 + "\n")
        
        try:
            # Train the model
            results = model.train(**train_args)
            
            # Get training results
            results_path = Path(self.project) / self.name
            best_model_path = results_path / "weights" / "best.pt"
            last_model_path = results_path / "weights" / "last.pt"
            
            # Extract metrics from results
            metrics = {
                'best_model': str(best_model_path) if best_model_path.exists() else None,
                'last_model': str(last_model_path) if last_model_path.exists() else None,
                'results_dir': str(results_path),
                'training_complete': True
            }
            
            # Try to extract final metrics from results object
            if hasattr(results, 'results_dict'):
                metrics.update(results.results_dict)
            
            logger.info("\n" + "=" * 60)
            logger.info("Training completed successfully!")
            logger.info("=" * 60)
            logger.info(f"Best model: {best_model_path}")
            logger.info(f"Last checkpoint: {last_model_path}")
            logger.info(f"Results directory: {results_path}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate the trained model
        
        Args:
            model_path: Path to model weights (defaults to best.pt from training)
            
        Returns:
            Dictionary containing validation metrics
        """
        if model_path is None:
            model_path = Path(self.project) / self.name / "weights" / "best.pt"
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Validating model: {model_path}")
        
        model = YOLO(str(model_path))
        results = model.val(data=str(self.data_yaml))
        
        # Extract metrics
        metrics = {
            'model_path': str(model_path),
            'mAP50': getattr(results, 'box', {}).get('mAP50', None) if hasattr(results, 'box') else None,
            'mAP50-95': getattr(results, 'box', {}).get('mAP50-95', None) if hasattr(results, 'box') else None,
            'precision': getattr(results, 'box', {}).get('precision', None) if hasattr(results, 'box') else None,
            'recall': getattr(results, 'box', {}).get('recall', None) if hasattr(results, 'box') else None,
        }
        
        logger.info("Validation metrics:")
        for key, value in metrics.items():
            if value is not None:
                logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        return metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Custom YOLOv8 Training Pipeline for License Plate Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (85 epochs, matches Roboflow)
  python train_custom.py --data datasets/license_plates/data.yaml
  
  # Quick test (1 epoch)
  python train_custom.py --data datasets/license_plates/data.yaml --epochs 1 --batch 4
  
  # Resume training
  python train_custom.py --data datasets/license_plates/data.yaml --resume runs/detect/custom_train/weights/last.pt
  
  # GPU training
  python train_custom.py --data datasets/license_plates/data.yaml --device cuda --batch 32
  
  # Custom experiment name
  python train_custom.py --data datasets/license_plates/data.yaml --name my_experiment
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data.yaml configuration file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Path to pretrained model or model name (default: yolov8n.pt)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=85,
        help='Number of training epochs (default: 85, matches Roboflow)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=512,
        help='Image size for training (default: 512, matches preprocessing)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16, adjust for your hardware)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu, cuda, 0, 1, etc.) or None for auto-detect'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory for outputs (default: runs/detect)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='custom_train',
        help='Experiment name (default: custom_train)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (default: 50)'
    )
    
    parser.add_argument(
        '--save-period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs (default: 10)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation (requires --model-path)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model for validation (used with --validate-only)'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CustomTrainer(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
        resume=args.resume
    )
    
    if args.validate_only:
        # Only validate
        if args.model_path:
            metrics = trainer.validate(model_path=args.model_path)
        else:
            metrics = trainer.validate()
        
        # Save metrics to JSON
        metrics_file = Path(args.project) / args.name / "validation_metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Validation metrics saved to: {metrics_file}")
    else:
        # Train
        metrics = trainer.train()
        
        # Save training summary
        summary_file = Path(args.project) / args.name / "training_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Training summary saved to: {summary_file}")
        
        # Run validation
        logger.info("\n" + "=" * 60)
        logger.info("Running validation on best model...")
        logger.info("=" * 60)
        val_metrics = trainer.validate()
        
        # Save validation metrics
        val_file = Path(args.project) / args.name / "validation_metrics.json"
        with open(val_file, 'w') as f:
            json.dump(val_metrics, f, indent=2)
        logger.info(f"Validation metrics saved to: {val_file}")


if __name__ == "__main__":
    main()

