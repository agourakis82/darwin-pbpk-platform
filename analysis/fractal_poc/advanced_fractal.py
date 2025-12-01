#!/usr/bin/env python3
"""
Advanced Fractal Analysis - Cell Distribution Pattern
======================================================

This module analyzes the SPATIAL DISTRIBUTION of cells, not just the image.
The hypothesis is that cell distribution patterns (clustering, spacing)
may correlate with blood heterogeneity and pharmacokinetic behavior.

Key insight: We need to analyze:
1. Cell centroids as point patterns
2. Cell boundaries as fractal objects
3. Inter-cell distances and their distribution
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CellDistributionResult:
    """Results from cell distribution fractal analysis."""
    image_path: str
    n_cells_detected: int
    df_boundaries: float       # Fractal dim of cell boundaries
    df_distribution: float     # Fractal dim of cell positions (point pattern)
    mean_cell_area: float
    std_cell_area: float
    cell_density: float        # cells per 1000 pixels²
    clustering_index: float    # Measure of cell clustering


def segment_cells(image_array: np.ndarray, 
                  method: str = "threshold") -> Tuple[np.ndarray, List[dict]]:
    """
    Segment blood cells from image.
    
    Returns binary mask and list of cell properties.
    """
    from scipy import ndimage
    
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2)
    else:
        gray = image_array.copy()
    
    # Normalize
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    
    # Blood cells are typically darker (RBCs) or have distinct staining
    # Use Otsu-like thresholding
    threshold = np.mean(gray) - 0.5 * np.std(gray)
    binary = gray < threshold
    
    # Clean up with morphological operations
    binary = ndimage.binary_opening(binary, iterations=2)
    binary = ndimage.binary_closing(binary, iterations=2)
    
    # Label connected components
    labeled, n_features = ndimage.label(binary)
    
    # Get cell properties
    cells = []
    for i in range(1, n_features + 1):
        cell_mask = labeled == i
        area = np.sum(cell_mask)
        
        # Filter by size (remove noise and artifacts)
        if 100 < area < 10000:  # Reasonable cell size range
            y_coords, x_coords = np.where(cell_mask)
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            cells.append({
                'id': i,
                'area': area,
                'centroid': (centroid_x, centroid_y),
                'mask': cell_mask
            })
    
    return binary, cells


def calculate_point_pattern_df(centroids: List[Tuple[float, float]], 
                               image_shape: Tuple[int, int]) -> float:
    """
    Calculate fractal dimension of cell centroid distribution.
    
    Uses box-counting on the point pattern.
    """
    if len(centroids) < 5:
        return np.nan
    
    # Create binary image of centroids
    centroid_image = np.zeros(image_shape, dtype=int)
    for x, y in centroids:
        xi, yi = int(x), int(y)
        if 0 <= xi < image_shape[1] and 0 <= yi < image_shape[0]:
            # Create small marker for each centroid
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nxi, nyi = xi + dx, yi + dy
                    if 0 <= nxi < image_shape[1] and 0 <= nyi < image_shape[0]:
                        centroid_image[nyi, nxi] = 1
    
    # Box counting
    from fractal_dimension import box_counting, calculate_fractal_dimension
    box_sizes, box_counts = box_counting(centroid_image)
    df, r2 = calculate_fractal_dimension(box_sizes, box_counts)
    
    return df


def calculate_boundary_df(cells: List[dict]) -> float:
    """Calculate fractal dimension of cell boundaries."""
    if len(cells) < 1:
        return np.nan
    
    from scipy import ndimage
    from fractal_dimension import box_counting, calculate_fractal_dimension
    
    # Combine all cell masks
    combined = None
    for cell in cells:
        if combined is None:
            combined = cell['mask'].astype(int)
        else:
            combined = np.logical_or(combined, cell['mask']).astype(int)
    
    # Extract boundaries using erosion
    eroded = ndimage.binary_erosion(combined)
    boundaries = combined.astype(int) - eroded.astype(int)
    
    # Box counting on boundaries
    box_sizes, box_counts = box_counting(boundaries)
    df, r2 = calculate_fractal_dimension(box_sizes, box_counts)
    
    return df


def calculate_clustering_index(centroids: List[Tuple[float, float]], 
                               image_shape: Tuple[int, int]) -> float:
    """
    Calculate clustering index using nearest neighbor distances.
    
    Clark-Evans R: R < 1 means clustered, R > 1 means dispersed
    """
    if len(centroids) < 2:
        return np.nan
    
    from scipy.spatial.distance import cdist
    
    points = np.array(centroids)
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    
    # Nearest neighbor distances
    nn_distances = np.min(distances, axis=1)
    mean_nn = np.mean(nn_distances)
    
    # Expected mean NN distance for random distribution
    area = image_shape[0] * image_shape[1]
    density = len(centroids) / area
    expected_nn = 0.5 / np.sqrt(density)
    
    # Clark-Evans R
    R = mean_nn / expected_nn
    
    return R


def analyze_cell_distribution(image_path: str) -> CellDistributionResult:
    """Perform complete cell distribution analysis."""
    from PIL import Image
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Segment cells
    binary, cells = segment_cells(img_array)
    
    if len(cells) < 3:
        logger.warning(f"Only {len(cells)} cells detected in {image_path}")
    
    # Extract centroids
    centroids = [c['centroid'] for c in cells]
    
    # Calculate metrics
    df_distribution = calculate_point_pattern_df(centroids, binary.shape)
    df_boundaries = calculate_boundary_df(cells)
    
    areas = [c['area'] for c in cells]
    mean_area = np.mean(areas) if areas else 0
    std_area = np.std(areas) if areas else 0
    
    cell_density = len(cells) / (binary.shape[0] * binary.shape[1]) * 1000
    clustering = calculate_clustering_index(centroids, binary.shape)
    
    return CellDistributionResult(
        image_path=str(image_path),
        n_cells_detected=len(cells),
        df_boundaries=df_boundaries,
        df_distribution=df_distribution,
        mean_cell_area=mean_area,
        std_cell_area=std_area,
        cell_density=cell_density,
        clustering_index=clustering
    )

