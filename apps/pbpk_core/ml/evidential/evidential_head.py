"""
Evidential Head for uncertainty quantification.

Predicts parameters of a Normal-Inverse-Gamma (NIG) distribution:
- gamma: predicted mean
- nu: virtual observation count (inverse epistemic uncertainty)
- alpha: shape parameter (controls aleatoric uncertainty)
- beta: scale parameter (controls aleatoric uncertainty)

Reference:
- Amini et al. "Deep Evidential Regression" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """
    Evidential regression head that outputs NIG parameters.
    
    Args:
        input_dim: Dimension of input features
        min_val: Minimum value for nu, alpha, beta (prevents numerical issues)
    """
    
    def __init__(self, input_dim: int, min_val: float = 1e-4):
        super().__init__()
        self.min_val = min_val
        
        # Predict 4 parameters for NIG distribution
        self.linear = nn.Linear(input_dim, 4)
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with:
                - gamma: predicted mean [batch_size]
                - nu: virtual observations [batch_size]
                - alpha: shape parameter [batch_size]
                - beta: scale parameter [batch_size]
        """
        out = self.linear(x)  # [batch_size, 4]
        
        # Split into 4 parameters
        gamma = out[:, 0]  # Mean prediction (unbounded)
        
        # nu, alpha, beta must be positive
        # Use softplus to ensure positivity: softplus(x) = log(1 + exp(x))
        nu = F.softplus(out[:, 1]) + self.min_val
        alpha = F.softplus(out[:, 2]) + self.min_val + 1  # alpha > 1 for valid variance
        beta = F.softplus(out[:, 3]) + self.min_val
        
        return {
            'gamma': gamma,
            'nu': nu,
            'alpha': alpha,
            'beta': beta
        }
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> dict:
        """
        Predict mean and uncertainty estimates.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with:
                - mean: predicted mean (gamma)
                - epistemic: epistemic uncertainty
                - aleatoric: aleatoric uncertainty
                - total: total uncertainty
        """
        params = self.forward(x)
        
        gamma = params['gamma']
        nu = params['nu']
        alpha = params['alpha']
        beta = params['beta']
        
        # Mean prediction is gamma
        mean = gamma
        
        # Epistemic uncertainty (from nu)
        # Higher nu = more "virtual observations" = less epistemic uncertainty
        epistemic = beta / (nu * (alpha - 1))
        
        # Aleatoric uncertainty (from alpha, beta)
        # Variance of the predicted distribution
        aleatoric = beta / (alpha - 1)
        
        # Total uncertainty
        total = epistemic + aleatoric
        
        return {
            'mean': mean,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total
        }

