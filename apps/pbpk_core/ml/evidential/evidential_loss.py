"""
Evidential Loss for training evidential regression models.

The loss consists of:
1. Negative log-likelihood (NLL) of the Normal-Inverse-Gamma distribution
2. Regularization term to prevent over-confidence

Reference:
- Amini et al. "Deep Evidential Regression" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import numpy as np


class EvidentialLoss(nn.Module):
    """
    Evidential regression loss.
    
    Args:
        coeff: Coefficient for regularization term (lambda in paper)
               Higher values = more regularization = less confidence
               Typical range: 0.01 - 1.0
    """
    
    def __init__(self, coeff: float = 0.1):
        super().__init__()
        self.coeff = coeff
        
    def forward(self, gamma, nu, alpha, beta, target, reduce=True):
        """
        Compute evidential loss.
        
        Args:
            gamma: Predicted mean [batch_size]
            nu: Virtual observations [batch_size]
            alpha: Shape parameter [batch_size]
            beta: Scale parameter [batch_size]
            target: Ground truth values [batch_size]
            reduce: If True, return mean loss; if False, return per-sample loss
            
        Returns:
            Loss value (scalar if reduce=True, [batch_size] if reduce=False)
        """
        # Compute error
        error = (target - gamma).abs()
        
        # NLL term (negative log-likelihood)
        # Based on Normal-Inverse-Gamma distribution
        nll = (
            0.5 * torch.log(np.pi / nu)
            - alpha * torch.log(2 * beta * (1 + nu))
            + (alpha + 0.5) * torch.log(nu * error ** 2 + 2 * beta * (1 + nu))
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        
        # Regularization term
        # Penalizes over-confidence (low epistemic uncertainty) for incorrect predictions
        # error_reg = |target - gamma| * (2 * nu + alpha)
        error_reg = error * (2 * nu + alpha)
        
        # Total loss
        loss = nll + self.coeff * error_reg
        
        if reduce:
            return loss.mean()
        else:
            return loss
    
    def nll_only(self, gamma, nu, alpha, beta, target, reduce=True):
        """
        Compute only the NLL term (for analysis).
        
        Args:
            gamma: Predicted mean [batch_size]
            nu: Virtual observations [batch_size]
            alpha: Shape parameter [batch_size]
            beta: Scale parameter [batch_size]
            target: Ground truth values [batch_size]
            reduce: If True, return mean; if False, return per-sample
            
        Returns:
            NLL value
        """
        error = (target - gamma).abs()
        
        nll = (
            0.5 * torch.log(np.pi / nu)
            - alpha * torch.log(2 * beta * (1 + nu))
            + (alpha + 0.5) * torch.log(nu * error ** 2 + 2 * beta * (1 + nu))
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        
        if reduce:
            return nll.mean()
        else:
            return nll


def expected_calibration_error(predictions, targets, uncertainties, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well the predicted uncertainties match actual errors.
    Lower ECE = better calibrated uncertainties.
    
    Args:
        predictions: Predicted values [N]
        targets: Ground truth values [N]
        uncertainties: Predicted uncertainties (total) [N]
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value (scalar)
    """
    predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    targets = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    uncertainties = uncertainties.cpu().numpy() if torch.is_tensor(uncertainties) else uncertainties
    
    # Compute actual errors
    errors = np.abs(predictions - targets)
    
    # Sort by uncertainty
    sorted_indices = np.argsort(uncertainties)
    sorted_errors = errors[sorted_indices]
    sorted_uncertainties = uncertainties[sorted_indices]
    
    # Bin samples
    n_samples = len(predictions)
    bin_size = n_samples // n_bins
    
    ece = 0.0
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n_samples
        
        if end_idx <= start_idx:
            continue
        
        bin_errors = sorted_errors[start_idx:end_idx]
        bin_uncertainties = sorted_uncertainties[start_idx:end_idx]
        
        # Average error in bin
        avg_error = np.mean(bin_errors)
        
        # Average predicted uncertainty in bin
        avg_uncertainty = np.mean(bin_uncertainties)
        
        # Calibration error for this bin
        bin_weight = (end_idx - start_idx) / n_samples
        ece += bin_weight * np.abs(avg_error - avg_uncertainty)
    
    return ece

