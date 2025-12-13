#!/usr/bin/env python3
"""
Model Deployment Helper

Helper script to deploy trained models to the models directory for use in the application.
This makes it easy to switch between custom-trained and Roboflow models.

Usage:
    python deploy_model.py --model runs/detect/custom_train/weights/best.pt --name custom
    python deploy_model.py --model ../models/license_plate.pt --name roboflow
"""

import argparse
import sys
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def deploy_model(model_path: str, name: str = "custom", backup: bool = True):
    """
    Deploy a trained model to the models directory
    
    Args:
        model_path: Path to the model weights file
        name: Model name identifier ("custom" or "roboflow")
        backup: Whether to backup existing model if it exists
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine target path
    backend_dir = Path(__file__).parent.parent
    models_dir = backend_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    if name == "custom":
        target_path = models_dir / "license_plate_custom.pt"
    elif name == "roboflow":
        target_path = models_dir / "license_plate.pt"
    else:
        target_path = models_dir / f"license_plate_{name}.pt"
    
    # Backup existing model if it exists
    if target_path.exists() and backup:
        backup_path = models_dir / f"{target_path.stem}_backup.pt"
        logger.info(f"Backing up existing model to {backup_path}")
        shutil.copy2(target_path, backup_path)
    
    # Copy model
    logger.info(f"Deploying model from {model_path} to {target_path}")
    shutil.copy2(model_path, target_path)
    
    logger.info(f"Model deployed successfully!")
    logger.info(f"   Location: {target_path}")
    logger.info(f"   Size: {target_path.stat().st_size / (1024*1024):.2f} MB")
    
    return target_path


def main():
    parser = argparse.ArgumentParser(
        description="Deploy trained model to models directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy custom-trained model
  python deploy_model.py --model runs/detect/custom_train/weights/best.pt --name custom
  
  # Deploy Roboflow model
  python deploy_model.py --model ../models/license_plate.pt --name roboflow
  
  # Deploy without backup
  python deploy_model.py --model best.pt --name custom --no-backup
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model weights file (.pt)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='custom',
        choices=['custom', 'roboflow'],
        help='Model name identifier (default: custom)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not backup existing model'
    )
    
    args = parser.parse_args()
    
    try:
        deploy_model(
            model_path=args.model,
            name=args.name,
            backup=not args.no_backup
        )
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

