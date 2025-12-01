#!/usr/bin/env python3
"""
Compare Fractal Dimensions Across Datasets
===========================================

Compares fractal dimension and cell distribution patterns between:
- Normal blood cells (BCCD)
- Pathological samples (when available)
- Synthetic pathological (for pipeline testing)

This is the key experiment to validate the hypothesis.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"


@dataclass
class DatasetStats:
    """Statistics for a dataset."""
    name: str
    n_images: int
    df_boundaries_mean: float
    df_boundaries_std: float
    df_distribution_mean: float
    df_distribution_std: float
    clustering_mean: float
    clustering_std: float


def analyze_dataset(name: str, image_dir: Path, max_images: int = 30) -> DatasetStats:
    """Analyze a single dataset and return statistics."""
    from advanced_fractal import analyze_cell_distribution
    
    # Find images
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        images.extend(image_dir.rglob(ext))
    images = sorted(images)[:max_images]
    
    if not images:
        logger.warning(f"No images found in {image_dir}")
        return None
    
    logger.info(f"Analyzing {name}: {len(images)} images")
    
    results = []
    for img_path in images:
        try:
            result = analyze_cell_distribution(str(img_path))
            if result.n_cells_detected >= 3:
                results.append(result)
        except Exception as e:
            pass  # Skip failed images
    
    if len(results) < 5:
        logger.warning(f"Only {len(results)} valid results for {name}")
        return None
    
    # Extract metrics
    df_b = [r.df_boundaries for r in results if not np.isnan(r.df_boundaries)]
    df_d = [r.df_distribution for r in results if not np.isnan(r.df_distribution)]
    clust = [r.clustering_index for r in results if not np.isnan(r.clustering_index)]
    
    return DatasetStats(
        name=name,
        n_images=len(results),
        df_boundaries_mean=np.mean(df_b),
        df_boundaries_std=np.std(df_b),
        df_distribution_mean=np.mean(df_d),
        df_distribution_std=np.std(df_d),
        clustering_mean=np.mean(clust),
        clustering_std=np.std(clust)
    )


def statistical_test(stats1: DatasetStats, stats2: DatasetStats) -> Dict:
    """Perform statistical comparison between two datasets."""
    from scipy import stats as scipy_stats
    
    # Effect size (Cohen's d)
    def cohens_d(m1, s1, m2, s2):
        pooled_std = np.sqrt((s1**2 + s2**2) / 2)
        return (m1 - m2) / pooled_std if pooled_std > 0 else 0
    
    d_boundaries = cohens_d(
        stats1.df_boundaries_mean, stats1.df_boundaries_std,
        stats2.df_boundaries_mean, stats2.df_boundaries_std
    )
    
    d_distribution = cohens_d(
        stats1.df_distribution_mean, stats1.df_distribution_std,
        stats2.df_distribution_mean, stats2.df_distribution_std
    )
    
    d_clustering = cohens_d(
        stats1.clustering_mean, stats1.clustering_std,
        stats2.clustering_mean, stats2.clustering_std
    )
    
    return {
        "comparison": f"{stats1.name} vs {stats2.name}",
        "effect_sizes": {
            "df_boundaries_d": round(d_boundaries, 3),
            "df_distribution_d": round(d_distribution, 3),
            "clustering_d": round(d_clustering, 3)
        },
        "interpretation": {
            "boundaries": "large" if abs(d_boundaries) > 0.8 else "medium" if abs(d_boundaries) > 0.5 else "small",
            "distribution": "large" if abs(d_distribution) > 0.8 else "medium" if abs(d_distribution) > 0.5 else "small",
            "clustering": "large" if abs(d_clustering) > 0.8 else "medium" if abs(d_clustering) > 0.5 else "small"
        }
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Define datasets to analyze
    datasets = {
        "BCCD_Normal": DATA_DIR / "BCCD_Dataset-master" / "BCCD" / "JPEGImages",
    }

    # Add malaria dataset if available (comparing infected vs uninfected)
    malaria_para = DATA_DIR / "malaria_cells" / "Parasitized"
    malaria_uninf = DATA_DIR / "malaria_cells" / "Uninfected"

    if malaria_para.exists():
        datasets["Malaria_Infected"] = malaria_para
    if malaria_uninf.exists():
        datasets["Malaria_Normal"] = malaria_uninf

    # Add synthetic for comparison
    if (DATA_DIR / "synthetic_pathological").exists():
        datasets["Synthetic_Patho"] = DATA_DIR / "synthetic_pathological"
    
    # Analyze each dataset
    all_stats = {}
    for name, path in datasets.items():
        if path.exists():
            stats = analyze_dataset(name, path)
            if stats:
                all_stats[name] = stats
    
    if len(all_stats) < 2:
        logger.error("Need at least 2 datasets for comparison")
        return
    
    # Print comparison
    print("\n" + "="*80)
    print("FRACTAL DIMENSION COMPARISON ACROSS DATASETS")
    print("="*80)
    print(f"\n{'Dataset':<20} {'N':<6} {'df_bound':<15} {'df_dist':<15} {'Clustering R':<15}")
    print("-"*80)
    
    for name, s in all_stats.items():
        print(f"{name:<20} {s.n_images:<6} "
              f"{s.df_boundaries_mean:.4f}±{s.df_boundaries_std:.4f}  "
              f"{s.df_distribution_mean:.4f}±{s.df_distribution_std:.4f}  "
              f"{s.clustering_mean:.4f}±{s.clustering_std:.4f}")
    
    # Statistical comparisons
    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS (Effect Size - Cohen's d)")
    print("="*80)
    
    stats_list = list(all_stats.values())
    comparisons = []
    for i in range(len(stats_list)):
        for j in range(i+1, len(stats_list)):
            comp = statistical_test(stats_list[i], stats_list[j])
            comparisons.append(comp)
            print(f"\n{comp['comparison']}:")
            for metric, d in comp['effect_sizes'].items():
                # Extract base metric name
                base_metric = metric.replace('_d', '').replace('df_', '')
                if base_metric in comp['interpretation']:
                    interp = comp['interpretation'][base_metric]
                else:
                    interp = "?"
                direction = "↑" if d > 0 else "↓" if d < 0 else "="
                print(f"  {metric}: d={d:+.3f} ({interp}) {direction}")
    
    # Save results
    output = {
        "datasets": {name: vars(s) for name, s in all_stats.items()},
        "comparisons": comparisons
    }
    
    with open(RESULTS_DIR / "dataset_comparison.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Results saved to: {RESULTS_DIR / 'dataset_comparison.json'}")


if __name__ == "__main__":
    main()

