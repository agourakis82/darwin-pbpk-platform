#!/usr/bin/env python3
"""
Fractal Dimension Analysis of Blood Cell Images
================================================

This module implements box-counting algorithm for calculating fractal dimension
of blood cell images. The goal is to explore the relationship between blood
microstructure heterogeneity and pharmacokinetic parameters.

Theory:
    For a fractal object, the number of boxes N(ε) of size ε needed to cover
    the object scales as: N(ε) ∝ ε^(-D)
    
    Where D is the fractal dimension.
    
    Taking logarithms: log(N(ε)) = -D * log(ε) + C
    
    The slope of log(N) vs log(1/ε) gives the fractal dimension.

References:
    - Kopelman R. (1986) J. Stat. Phys. 42:185-200
    - Jung et al. (2023) Pharmaceutics 15:304
"""

import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FractalResult:
    """Result of fractal dimension analysis."""
    image_path: str
    fractal_dimension: float
    r_squared: float
    box_sizes: List[int]
    box_counts: List[int]
    method: str = "box_counting"
    

def box_counting(binary_image: np.ndarray, 
                 min_box_size: int = 2,
                 max_box_size: Optional[int] = None,
                 n_sizes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform box-counting on a binary image.
    
    Parameters
    ----------
    binary_image : np.ndarray
        Binary image (0s and 1s)
    min_box_size : int
        Minimum box size in pixels
    max_box_size : int, optional
        Maximum box size (defaults to min dimension / 4)
    n_sizes : int
        Number of box sizes to use
        
    Returns
    -------
    box_sizes : np.ndarray
        Array of box sizes used
    box_counts : np.ndarray
        Number of boxes containing pixels for each size
    """
    if max_box_size is None:
        max_box_size = min(binary_image.shape) // 4
    
    # Generate logarithmically spaced box sizes
    box_sizes = np.unique(np.logspace(
        np.log10(min_box_size),
        np.log10(max_box_size),
        n_sizes
    ).astype(int))
    
    box_counts = []
    
    for box_size in box_sizes:
        # Count boxes that contain at least one pixel
        count = 0
        for i in range(0, binary_image.shape[0], box_size):
            for j in range(0, binary_image.shape[1], box_size):
                box = binary_image[i:i+box_size, j:j+box_size]
                if np.any(box):
                    count += 1
        box_counts.append(count)
    
    return box_sizes, np.array(box_counts)


def calculate_fractal_dimension(box_sizes: np.ndarray, 
                                box_counts: np.ndarray) -> Tuple[float, float]:
    """
    Calculate fractal dimension from box-counting data.
    
    Uses linear regression on log-log plot.
    
    Returns
    -------
    fractal_dim : float
        Estimated fractal dimension
    r_squared : float
        R² of the linear fit
    """
    # Take logarithms
    log_sizes = np.log(1.0 / box_sizes)
    log_counts = np.log(box_counts)
    
    # Linear regression
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dim = coeffs[0]
    
    # Calculate R²
    y_pred = np.polyval(coeffs, log_sizes)
    ss_res = np.sum((log_counts - y_pred) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return fractal_dim, r_squared


def analyze_image(image_path: str, 
                  threshold: float = 0.5,
                  invert: bool = False) -> FractalResult:
    """
    Analyze a single image for fractal dimension.
    
    Parameters
    ----------
    image_path : str
        Path to image file
    threshold : float
        Threshold for binarization (0-1)
    invert : bool
        If True, invert the binary image
        
    Returns
    -------
    FractalResult
        Analysis result with fractal dimension
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL required. Install with: pip install Pillow")
    
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img) / 255.0
    
    # Binarize
    binary = (img_array > threshold).astype(int)
    if invert:
        binary = 1 - binary
    
    # Box counting
    box_sizes, box_counts = box_counting(binary)
    
    # Calculate fractal dimension
    fractal_dim, r_squared = calculate_fractal_dimension(box_sizes, box_counts)
    
    return FractalResult(
        image_path=str(image_path),
        fractal_dimension=fractal_dim,
        r_squared=r_squared,
        box_sizes=box_sizes.tolist(),
        box_counts=box_counts.tolist()
    )

