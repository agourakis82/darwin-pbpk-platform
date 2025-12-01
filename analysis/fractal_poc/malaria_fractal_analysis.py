#!/usr/bin/env python3
"""
Malaria Fractal Analysis - Detailed Comparison
===============================================

Compares fractal dimension between:
- Uninfected (normal) red blood cells
- Parasitized (malaria-infected) red blood cells

Both from the same NIH dataset with identical acquisition methodology.
This is a proper controlled comparison.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"


@dataclass
class CellMetrics:
    """Metrics for a single cell image."""
    image_name: str
    condition: str  # 'Parasitized' or 'Uninfected'
    df_image: float  # Fractal dim of whole image
    df_edge: float   # Fractal dim of edge-detected image
    intensity_mean: float
    intensity_std: float
    area_ratio: float  # ratio of cell area to background


def analyze_cell_image(image_path: Path, condition: str) -> CellMetrics:
    """Analyze a single malaria cell image."""
    from PIL import Image
    from scipy import ndimage
    from fractal_dimension import box_counting, calculate_fractal_dimension
    
    # Load image
    img = Image.open(image_path).convert('L')  # Grayscale
    arr = np.array(img) / 255.0
    
    # 1. Fractal dimension of binarized image
    threshold = np.mean(arr)
    binary = (arr < threshold).astype(int)
    
    try:
        box_sizes, box_counts = box_counting(binary, min_box_size=2, max_box_size=min(arr.shape)//4)
        df_image, _ = calculate_fractal_dimension(box_sizes, box_counts)
    except:
        df_image = np.nan
    
    # 2. Fractal dimension of edges (Sobel)
    from scipy.ndimage import sobel
    edge_x = sobel(arr, axis=0)
    edge_y = sobel(arr, axis=1)
    edges = np.hypot(edge_x, edge_y)
    edge_binary = (edges > np.percentile(edges, 70)).astype(int)
    
    try:
        box_sizes_e, box_counts_e = box_counting(edge_binary, min_box_size=2, max_box_size=min(arr.shape)//4)
        df_edge, _ = calculate_fractal_dimension(box_sizes_e, box_counts_e)
    except:
        df_edge = np.nan
    
    # 3. Intensity statistics
    intensity_mean = np.mean(arr)
    intensity_std = np.std(arr)
    
    # 4. Area ratio (cell vs background)
    area_ratio = np.sum(binary) / binary.size
    
    return CellMetrics(
        image_name=image_path.name,
        condition=condition,
        df_image=df_image,
        df_edge=df_edge,
        intensity_mean=intensity_mean,
        intensity_std=intensity_std,
        area_ratio=area_ratio
    )


def run_analysis(n_samples: int = 100):
    """Run fractal analysis on malaria dataset."""
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Paths
    parasitized_dir = DATA_DIR / "malaria_cells" / "Parasitized"
    uninfected_dir = DATA_DIR / "malaria_cells" / "Uninfected"
    
    if not parasitized_dir.exists() or not uninfected_dir.exists():
        logger.error("Malaria dataset not found. Run download_malaria_nih.py first.")
        return None
    
    # Get random samples from each condition
    np.random.seed(42)  # Reproducibility
    
    para_images = list(parasitized_dir.glob("*.png"))
    uninf_images = list(uninfected_dir.glob("*.png"))
    
    para_sample = np.random.choice(para_images, min(n_samples, len(para_images)), replace=False)
    uninf_sample = np.random.choice(uninf_images, min(n_samples, len(uninf_images)), replace=False)
    
    logger.info(f"Analyzing {len(para_sample)} Parasitized + {len(uninf_sample)} Uninfected cells")
    
    # Analyze
    results = []
    
    for i, img_path in enumerate(para_sample):
        try:
            result = analyze_cell_image(Path(img_path), 'Parasitized')
            results.append(result)
        except Exception as e:
            pass
        if (i + 1) % 20 == 0:
            logger.info(f"  Parasitized: {i+1}/{len(para_sample)}")
    
    for i, img_path in enumerate(uninf_sample):
        try:
            result = analyze_cell_image(Path(img_path), 'Uninfected')
            results.append(result)
        except Exception as e:
            pass
        if (i + 1) % 20 == 0:
            logger.info(f"  Uninfected: {i+1}/{len(uninf_sample)}")
    
    # Separate by condition
    para_results = [r for r in results if r.condition == 'Parasitized']
    uninf_results = [r for r in results if r.condition == 'Uninfected']
    
    # Calculate statistics
    def get_stats(data: List[CellMetrics], metric: str) -> Dict:
        values = [getattr(r, metric) for r in data if not np.isnan(getattr(r, metric))]
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'n': len(values)
        }
    
    metrics = ['df_image', 'df_edge', 'intensity_mean', 'intensity_std', 'area_ratio']
    
    comparison = {}
    for metric in metrics:
        para_stats = get_stats(para_results, metric)
        uninf_stats = get_stats(uninf_results, metric)
        
        # Statistical test (t-test)
        para_vals = [getattr(r, metric) for r in para_results if not np.isnan(getattr(r, metric))]
        uninf_vals = [getattr(r, metric) for r in uninf_results if not np.isnan(getattr(r, metric))]
        
        t_stat, p_value = stats.ttest_ind(para_vals, uninf_vals)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(para_vals)**2 + np.std(uninf_vals)**2) / 2)
        cohens_d = (np.mean(para_vals) - np.mean(uninf_vals)) / pooled_std if pooled_std > 0 else 0
        
        comparison[metric] = {
            'parasitized': para_stats,
            'uninfected': uninf_stats,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05)
        }
    
    # Save results
    output = {
        'n_parasitized': len(para_results),
        'n_uninfected': len(uninf_results),
        'comparison': comparison,
        'individual_results': [asdict(r) for r in results]
    }
    
    results_file = RESULTS_DIR / "malaria_fractal_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print results
    print_results(comparison, len(para_results), len(uninf_results))
    
    return comparison


def print_results(comparison: Dict, n_para: int, n_uninf: int):
    """Print formatted results."""
    print("\n" + "="*80)
    print("MALARIA FRACTAL ANALYSIS - PARASITIZED vs UNINFECTED")
    print("="*80)
    print(f"\nSample size: {n_para} Parasitized, {n_uninf} Uninfected")
    print("\n" + "-"*80)
    print(f"{'Metric':<20} {'Parasitized':<18} {'Uninfected':<18} {'Cohen d':<10} {'p-value':<10} {'Sig?'}")
    print("-"*80)
    
    for metric, data in comparison.items():
        para = f"{data['parasitized']['mean']:.4f}±{data['parasitized']['std']:.4f}"
        uninf = f"{data['uninfected']['mean']:.4f}±{data['uninfected']['std']:.4f}"
        sig = "***" if data['p_value'] < 0.001 else "**" if data['p_value'] < 0.01 else "*" if data['p_value'] < 0.05 else ""
        print(f"{metric:<20} {para:<18} {uninf:<18} {data['cohens_d']:+.3f}     {data['p_value']:.2e}   {sig}")
    
    print("-"*80)
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    print("Effect size: |d|<0.2=small, 0.2-0.8=medium, >0.8=large")
    print("="*80)


if __name__ == "__main__":
    run_analysis(n_samples=200)

