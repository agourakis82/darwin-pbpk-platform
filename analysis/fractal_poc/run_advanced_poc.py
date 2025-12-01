#!/usr/bin/env python3
"""
Advanced Fractal PoC - Cell Distribution Analysis
==================================================

Analyzes cell spatial distribution patterns using fractal dimension
and clustering metrics.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"


def main():
    from advanced_fractal import analyze_cell_distribution, CellDistributionResult
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Find images
    bccd_dir = DATA_DIR / "BCCD_Dataset-master" / "BCCD" / "JPEGImages"
    if not bccd_dir.exists():
        logger.error(f"Dataset not found. Run: python run_poc.py --download")
        sys.exit(1)
    
    images = sorted(bccd_dir.glob("*.jpg"))[:50]  # First 50 images
    logger.info(f"Analyzing {len(images)} images for cell distribution patterns")
    
    results = []
    
    for i, img_path in enumerate(images):
        try:
            result = analyze_cell_distribution(str(img_path))
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(images)} images")
                
        except Exception as e:
            logger.warning(f"Failed: {img_path.name}: {e}")
    
    # Calculate statistics
    valid_results = [r for r in results if r.n_cells_detected >= 3]
    
    df_boundaries = [r.df_boundaries for r in valid_results if not np.isnan(r.df_boundaries)]
    df_distribution = [r.df_distribution for r in valid_results if not np.isnan(r.df_distribution)]
    clustering = [r.clustering_index for r in valid_results if not np.isnan(r.clustering_index)]
    n_cells = [r.n_cells_detected for r in valid_results]
    
    stats = {
        "n_images_total": len(results),
        "n_images_valid": len(valid_results),
        "cell_detection": {
            "mean_cells_per_image": float(np.mean(n_cells)),
            "std_cells": float(np.std(n_cells)),
            "range": [int(np.min(n_cells)), int(np.max(n_cells))]
        },
        "df_boundaries": {
            "mean": float(np.mean(df_boundaries)),
            "std": float(np.std(df_boundaries)),
            "range": [float(np.min(df_boundaries)), float(np.max(df_boundaries))],
            "cv_percent": float(np.std(df_boundaries) / np.mean(df_boundaries) * 100)
        },
        "df_distribution": {
            "mean": float(np.mean(df_distribution)),
            "std": float(np.std(df_distribution)),
            "range": [float(np.min(df_distribution)), float(np.max(df_distribution))],
            "cv_percent": float(np.std(df_distribution) / np.mean(df_distribution) * 100)
        },
        "clustering_index": {
            "mean": float(np.mean(clustering)),
            "std": float(np.std(clustering)),
            "interpretation": "R<1=clustered, R=1=random, R>1=dispersed"
        }
    }
    
    # Individual results
    individual = []
    for r in valid_results:
        individual.append({
            "image": Path(r.image_path).name,
            "n_cells": r.n_cells_detected,
            "df_boundaries": round(r.df_boundaries, 4) if not np.isnan(r.df_boundaries) else None,
            "df_distribution": round(r.df_distribution, 4) if not np.isnan(r.df_distribution) else None,
            "clustering_R": round(r.clustering_index, 4) if not np.isnan(r.clustering_index) else None,
            "cell_density": round(r.cell_density, 6)
        })
    
    # Save results
    output = {"statistics": stats, "individual_results": individual}
    results_file = RESULTS_DIR / "advanced_fractal_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("ADVANCED FRACTAL ANALYSIS - CELL DISTRIBUTION PATTERNS")
    print("="*70)
    print(f"\nImages analyzed: {stats['n_images_valid']} (valid) / {stats['n_images_total']} (total)")
    print(f"\nCell Detection:")
    print(f"  Mean cells/image: {stats['cell_detection']['mean_cells_per_image']:.1f} ± {stats['cell_detection']['std_cells']:.1f}")
    print(f"  Range: {stats['cell_detection']['range']}")
    print(f"\nFractal Dimension - Cell Boundaries:")
    print(f"  df = {stats['df_boundaries']['mean']:.4f} ± {stats['df_boundaries']['std']:.4f}")
    print(f"  CV = {stats['df_boundaries']['cv_percent']:.2f}%")
    print(f"\nFractal Dimension - Cell Distribution (Point Pattern):")
    print(f"  df = {stats['df_distribution']['mean']:.4f} ± {stats['df_distribution']['std']:.4f}")
    print(f"  CV = {stats['df_distribution']['cv_percent']:.2f}%")
    print(f"\nClustering Index (Clark-Evans R):")
    print(f"  R = {stats['clustering_index']['mean']:.4f} ± {stats['clustering_index']['std']:.4f}")
    print(f"  Interpretation: {stats['clustering_index']['interpretation']}")
    print("="*70)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

