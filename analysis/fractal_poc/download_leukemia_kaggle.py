#!/usr/bin/env python3
"""
Download Leukemia (ALL) Dataset from Kaggle
============================================

Downloads the Acute Lymphoblastic Leukemia dataset:
- Benign (normal hematogones)
- Malignant (ALL subtypes: Early Pre-B, Pre-B, Pro-B)

Source: https://www.kaggle.com/datasets/mehradaria/leukemia
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

KAGGLE_DATASET = "mehradaria/leukemia"


def check_kaggle_cli():
    """Check if Kaggle CLI is installed and configured."""
    try:
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    return False


def download_via_kaggle():
    """Download dataset using Kaggle CLI."""
    leukemia_dir = DATA_DIR / "leukemia_ALL"
    leukemia_dir.mkdir(exist_ok=True)
    
    if not check_kaggle_cli():
        return False
    
    logger.info(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    
    try:
        # Download
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET, '-p', str(leukemia_dir)],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Download failed: {result.stderr}")
            return False
        
        # Extract
        zip_files = list(leukemia_dir.glob("*.zip"))
        for zf in zip_files:
            logger.info(f"Extracting {zf.name}...")
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(leukemia_dir)
            zf.unlink()
        
        logger.info("Download and extraction complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def show_manual_instructions():
    """Show manual download instructions."""
    leukemia_dir = DATA_DIR / "leukemia_ALL"
    
    print("""
================================================================================
MANUAL DOWNLOAD INSTRUCTIONS - Leukemia (ALL) Dataset
================================================================================

Option 1 - Install Kaggle CLI:
    pip install kaggle
    # Create API token at: https://www.kaggle.com/settings
    # Save kaggle.json to ~/.kaggle/kaggle.json
    # Run this script again

Option 2 - Manual Download:
    1. Go to: https://www.kaggle.com/datasets/mehradaria/leukemia
    2. Click "Download" (requires Kaggle login)
    3. Extract the zip file to: {leukemia_dir}
    
Expected structure after extraction:
    {leukemia_dir}/
    ├── Original/
    │   ├── Benign/          <- Normal cells
    │   └── Malignant/       <- Leukemia cells
    │       ├── Early Pre-B/
    │       ├── Pre-B/
    │       └── Pro-B/
    └── Segmented/
        └── ...

Dataset Info:
    - 3,256 images from 89 patients
    - Classes: Benign (normal) vs Malignant (ALL subtypes)
    - 100x magnification microscopy
    - JPG format
================================================================================
""".format(leukemia_dir=leukemia_dir))


def check_dataset():
    """Check if dataset is already downloaded."""
    leukemia_dir = DATA_DIR / "leukemia_ALL"
    
    # Check various possible structures
    possible_paths = [
        leukemia_dir / "Original",
        leukemia_dir / "original", 
        leukemia_dir / "Benign",
        leukemia_dir / "benign",
    ]
    
    for path in possible_paths:
        if path.exists():
            # Count images
            n_images = len(list(path.rglob("*.jpg"))) + len(list(path.rglob("*.bmp")))
            if n_images > 0:
                logger.info(f"Dataset found at {path.parent}")
                logger.info(f"  Total images: {n_images}")
                return True
    
    return False


def main():
    DATA_DIR.mkdir(exist_ok=True)
    
    # Check if already downloaded
    if check_dataset():
        logger.info("Leukemia dataset already available!")
        return True
    
    # Try Kaggle download
    if check_kaggle_cli():
        if download_via_kaggle():
            return True
    
    # Show manual instructions
    show_manual_instructions()
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

