"""
Evidential Deep Learning Head
==============================

Implements evidential regression head that outputs parameters for
a Normal-Inverse-Gamma (NIG) distribution, enabling uncertainty
quantification with epistemic and aleatoric separation.

Reference: Amini et al. "Deep Evidential Regression", NeurIPS 2020

Author: Dr. Agourakis
Date: October 26, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EvidentialHead(nn.Module):
    """
    Evidential regression head for uncertainty quantification.
    
    Outputs four parameters for Normal-Inverse-Gamma distribution:
        - γ (gamma): Predicted mean
        - ν (nu): Precision of the Normal distribution (> 0)
        - α (alpha): Shape parameter of Inverse-Gamma (> 1)
        - β (beta): Rate parameter of Inverse-Gamma (> 0)
    
    Epistemic uncertainty: β / (α - 1)
    Aleatoric uncertainty: β / (ν * (α - 1))
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize evidential head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super(EvidentialHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer (4 parameters)
        self.fc2 = nn.Linear(hidden_dim, 4)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through evidential head.
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Tuple of (gamma, nu, alpha, beta):
                - gamma: [batch_size] - predicted mean
                - nu: [batch_size] - precision (> 0)
                - alpha: [batch_size] - shape (> 1)
                - beta: [batch_size] - rate (> 0)
        """
        # Hidden layer with ReLU
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # Output layer
        out = self.fc2(x)
        
        # Split into 4 parameters
        gamma = out[:, 0]  # Mean (unconstrained)
        nu_raw = out[:, 1]
        alpha_raw = out[:, 2]
        beta_raw = out[:, 3]
        
        # Apply constraints using softplus
        # softplus(x) = log(1 + exp(x)) ensures positivity
        nu = F.softplus(nu_raw)           # ν > 0
        alpha = F.softplus(alpha_raw) + 1.0  # α > 1
        beta = F.softplus(beta_raw)       # β > 0
        
        return gamma, nu, alpha, beta
    
    def compute_uncertainties(
        self,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute uncertainties from evidential parameters.
        
        Args:
            gamma, nu, alpha, beta: Evidential parameters
        
        Returns:
            Tuple of (epistemic_var, aleatoric_var, total_var)
        """
        # Epistemic uncertainty (reducible with more data)
        epistemic_var = beta / (alpha - 1)
        
        # Aleatoric uncertainty (irreducible noise)
        aleatoric_var = beta / (nu * (alpha - 1))
        
        # Total uncertainty
        total_var = epistemic_var + aleatoric_var
        
        return epistemic_var, aleatoric_var, total_var
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> dict:
        """
        Predict with full uncertainty quantification.
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Dictionary with:
                - mean: Predicted value
                - epistemic_std: Epistemic uncertainty (standard deviation)
                - aleatoric_std: Aleatoric uncertainty
                - total_std: Total uncertainty
        """
        gamma, nu, alpha, beta = self.forward(x)
        epistemic_var, aleatoric_var, total_var = self.compute_uncertainties(
            gamma, nu, alpha, beta
        )
        
        return {
            'mean': gamma,
            'epistemic_std': torch.sqrt(epistemic_var),
            'aleatoric_std': torch.sqrt(aleatoric_var),
            'total_std': torch.sqrt(total_var)
        }
    
    def __repr__(self):
        return (
            f"EvidentialHead(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  dropout={self.dropout}\n"
            f")"
        )

