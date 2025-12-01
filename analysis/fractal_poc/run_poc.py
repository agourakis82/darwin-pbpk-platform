#!/usr/bin/env python3
"""
Fractal PoC - Blood Cell Image Analysis
========================================

This script downloads the BCCD dataset and calculates fractal dimensions
for blood cell images to validate the hypothesis that blood microstructure
heterogeneity can be quantified using fractal analysis.

Usage:
    python run_poc.py --download   # Download dataset
    python run_poc.py --analyze    # Run analysis
    python run_poc.py --all        # Do both
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
BCCD_URL = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip"


def download_dataset():
    """Download BCCD dataset from GitHub."""
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "bccd.zip"
    
    if (DATA_DIR / "BCCD_Dataset-master").exists():
        logger.info("Dataset already downloaded")
        return True
    
    logger.info(f"Downloading BCCD dataset from {BCCD_URL}")
    
    try:
        import urllib.request
        urllib.request.urlretrieve(BCCD_URL, zip_path)
        logger.info("Download complete. Extracting...")
        
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        zip_path.unlink()  # Remove zip file
        logger.info("Dataset extracted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def find_images(data_dir: Path) -> List[Path]:
    """Find all JPEG images in the dataset."""
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(data_dir.rglob(ext))
    return sorted(images)


def run_analysis(max_images: int = 50):
    """Run fractal dimension analysis on blood cell images."""
    from fractal_dimension import analyze_image, FractalResult
    import numpy as np
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Find images
    bccd_dir = DATA_DIR / "BCCD_Dataset-master" / "BCCD" / "JPEGImages"
    if not bccd_dir.exists():
        logger.error(f"Dataset not found at {bccd_dir}. Run with --download first.")
        return None
    
    images = find_images(bccd_dir)
    logger.info(f"Found {len(images)} images")
    
    if max_images:
        images = images[:max_images]
        logger.info(f"Analyzing first {max_images} images")
    
    results = []
    fractal_dims = []
    
    for i, img_path in enumerate(images):
        try:
            result = analyze_image(str(img_path), threshold=0.5)
            results.append(result)
            fractal_dims.append(result.fractal_dimension)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(images)} images")
                
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
    
    # Calculate statistics
    fractal_dims = np.array(fractal_dims)
    stats = {
        "n_images": len(fractal_dims),
        "mean_df": float(np.mean(fractal_dims)),
        "std_df": float(np.std(fractal_dims)),
        "min_df": float(np.min(fractal_dims)),
        "max_df": float(np.max(fractal_dims)),
        "median_df": float(np.median(fractal_dims)),
        "cv_percent": float(np.std(fractal_dims) / np.mean(fractal_dims) * 100)
    }
    
    # Save results
    results_data = {
        "statistics": stats,
        "individual_results": [
            {"image": r.image_path, "df": r.fractal_dimension, "r2": r.r_squared}
            for r in results
        ]
    }
    
    results_file = RESULTS_DIR / "fractal_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("FRACTAL DIMENSION ANALYSIS RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Images analyzed: {stats['n_images']}")
    logger.info(f"Mean df: {stats['mean_df']:.4f} ± {stats['std_df']:.4f}")
    logger.info(f"Range: [{stats['min_df']:.4f}, {stats['max_df']:.4f}]")
    logger.info(f"Coefficient of Variation: {stats['cv_percent']:.2f}%")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {results_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Fractal PoC for Blood Cell Images")
    parser.add_argument('--download', action='store_true', help='Download BCCD dataset')
    parser.add_argument('--analyze', action='store_true', help='Run fractal analysis')
    parser.add_argument('--all', action='store_true', help='Download and analyze')
    parser.add_argument('--max-images', type=int, default=50, help='Max images to analyze')
    
    args = parser.parse_args()
    
    if args.all or args.download:
        if not download_dataset():
            sys.exit(1)
    
    if args.all or args.analyze:
        stats = run_analysis(max_images=args.max_images)
        if stats is None:
            sys.exit(1)
    
    if not any([args.download, args.analyze, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()

