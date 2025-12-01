#!/usr/bin/env python3
"""
Fractal PK Model: Image → Heterogeneity → Pharmacokinetics
===========================================================

Theoretical model connecting blood cell image fractal dimension
to pharmacokinetic parameters via Kopelman's fractal kinetics.

THEORETICAL MODEL - NOT YET VALIDATED
This model requires empirical validation with paired image+PK data.

References:
    - Kopelman R. (1986) J. Stat. Phys. 42:185-200
    - Jung et al. (2023) Pharmaceutics 15:304
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class FractalMetrics:
    """Fractal metrics extracted from blood cell image."""
    df_edge: float          # Fractal dim of cell boundaries [1, 2]
    df_distribution: float  # Fractal dim of cell positions [0, 2]
    clustering_R: float     # Clark-Evans clustering index [0, ∞)
    
    def validate(self) -> bool:
        """Validate metrics are in expected ranges."""
        return (1.0 <= self.df_edge <= 2.0 and
                0.0 <= self.df_distribution <= 2.0 and
                self.clustering_R >= 0)


@dataclass
class DrugProperties:
    """Drug physicochemical properties."""
    name: str
    logP: float             # Lipophilicity
    MW: float               # Molecular weight (Da)
    fu_reference: float     # Reference fraction unbound [0, 1]
    pKa: Optional[float] = None
    
    
@dataclass 
class FractalPKParameters:
    """Pharmacokinetic parameters predicted by fractal model."""
    h: float                    # Heterogeneity exponent [0, 1]
    k_el_modifier: float        # Elimination rate modifier
    CL_modifier: float          # Clearance modifier
    fu_predicted: float         # Predicted fraction unbound
    confidence: str             # "theoretical" until validated


class FractalPKModel:
    """
    Model connecting blood image fractal dimension to PK parameters.
    
    Theory:
        In heterogeneous media, rate constants become time-dependent:
        k(t) = k₀ × t^(-h)
        
        where h is the heterogeneity exponent related to fractal dimension.
        
    Proposed relationship:
        h ≈ α × (2 - df_edge) + β × (2 - df_dist) + γ × |1 - R|
        
    Simplified (first approximation):
        h ≈ 2 - df_edge
    """
    
    # Model coefficients (to be empirically determined)
    ALPHA = 0.6    # Weight for df_edge contribution
    BETA = 0.3     # Weight for df_distribution contribution  
    GAMMA = 0.1    # Weight for clustering contribution
    
    # PK adjustment factors (theoretical)
    BINDING_FACTOR = 0.3  # How much h affects protein binding
    
    def __init__(self, 
                 alpha: float = None,
                 beta: float = None,
                 gamma: float = None):
        """Initialize model with coefficients."""
        self.alpha = alpha or self.ALPHA
        self.beta = beta or self.BETA
        self.gamma = gamma or self.GAMMA
        
    def calculate_h(self, metrics: FractalMetrics) -> float:
        """
        Calculate heterogeneity exponent h from fractal metrics.
        
        Returns h ∈ [0, 1] where:
            h = 0: homogeneous (classical kinetics)
            h = 1: maximally heterogeneous
        """
        # Contributions from each metric
        edge_contribution = self.alpha * (2.0 - metrics.df_edge)
        dist_contribution = self.beta * (2.0 - metrics.df_distribution)
        clust_contribution = self.gamma * abs(1.0 - metrics.clustering_R)
        
        # Combined h
        h = edge_contribution + dist_contribution + clust_contribution
        
        # Clamp to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        
        return h
    
    def calculate_h_simple(self, df_edge: float) -> float:
        """
        Simplified model: h ≈ 2 - df_edge
        
        For quick estimation when only edge fractal dimension is available.
        """
        return np.clip(2.0 - df_edge, 0.0, 1.0)
    
    def predict_pk_modifiers(self, 
                             metrics: FractalMetrics,
                             drug: DrugProperties,
                             time_point: float = 1.0) -> FractalPKParameters:
        """
        Predict PK parameter modifiers based on fractal analysis.
        
        Parameters
        ----------
        metrics : FractalMetrics
            Fractal metrics from blood cell image
        drug : DrugProperties
            Drug physicochemical properties
        time_point : float
            Time point for time-dependent rate calculation (hours)
            
        Returns
        -------
        FractalPKParameters
            Predicted PK parameter modifiers
        """
        # Calculate heterogeneity exponent
        h = self.calculate_h(metrics)
        
        # Time-dependent rate modifier: k(t)/k₀ = t^(-h)
        # At t=1h, modifier = 1
        # At t>1h, modifier < 1 (slower rate)
        k_el_modifier = time_point ** (-h) if time_point > 0 else 1.0
        
        # Clearance modifier (integrated effect)
        # CL_eff = CL_0 × (1 + h × tissue_factor)
        # For now, assume tissue_factor ≈ 0.5
        CL_modifier = 1.0 + h * 0.5
        
        # Fraction unbound prediction
        # More heterogeneous → more variable binding → effective fu changes
        fu_ref = drug.fu_reference
        
        # Correction factor based on heterogeneous binding theory
        # fu_eff = fu_ref × (1 - h × (1-fu_ref) × binding_factor)
        correction = 1.0 - h * (1.0 - fu_ref) * self.BINDING_FACTOR
        fu_predicted = fu_ref * correction
        
        # Clamp fu to valid range
        fu_predicted = np.clip(fu_predicted, 0.01, 1.0)
        
        return FractalPKParameters(
            h=h,
            k_el_modifier=k_el_modifier,
            CL_modifier=CL_modifier,
            fu_predicted=fu_predicted,
            confidence="theoretical"
        )
    
    def __repr__(self):
        return f"FractalPKModel(α={self.alpha}, β={self.beta}, γ={self.gamma})"


# Convenience function for quick predictions
def predict_pk_from_image_metrics(df_edge: float,
                                  df_distribution: float,
                                  clustering_R: float,
                                  fu_reference: float = 0.5,
                                  drug_name: str = "generic") -> dict:
    """
    Quick prediction of PK modifiers from image metrics.
    
    Example:
        >>> result = predict_pk_from_image_metrics(
        ...     df_edge=1.69, 
        ...     df_distribution=0.74,
        ...     clustering_R=2.74,
        ...     fu_reference=0.3
        ... )
        >>> print(f"h = {result['h']:.3f}")
    """
    metrics = FractalMetrics(df_edge, df_distribution, clustering_R)
    drug = DrugProperties(drug_name, logP=2.0, MW=400, fu_reference=fu_reference)
    
    model = FractalPKModel()
    result = model.predict_pk_modifiers(metrics, drug)
    
    return {
        'h': result.h,
        'k_el_modifier': result.k_el_modifier,
        'CL_modifier': result.CL_modifier,
        'fu_predicted': result.fu_predicted,
        'confidence': result.confidence
    }

