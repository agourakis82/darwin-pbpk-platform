"""
KEC-PINN Loss Functions
========================

Implements evidential regression loss and physics-informed constraints
for PBPK parameter prediction.

Loss components:
1. Evidential Loss: NLL + regularization term
2. Physics Loss: PBPK constraints (bounds, mass balance, consistency)
3. Total Loss: α * L_data + β * L_physics

Author: Dr. Agourakis
Date: October 26, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def evidential_loss(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    lambda_reg: float = 0.01
) -> torch.Tensor:
    """
    Evidential regression loss (Amini et al., NeurIPS 2020).
    
    Args:
        gamma: Predicted mean [N]
        nu: Precision parameter [N]
        alpha: Shape parameter [N]
        beta: Rate parameter [N]
        y_true: Ground truth targets [N]
        mask: Boolean mask for observed values [N]
        lambda_reg: Regularization weight
    
    Returns:
        Scalar loss value
    """
    # Filter to observed samples only
    gamma_obs = gamma[mask]
    nu_obs = nu[mask]
    alpha_obs = alpha[mask]
    beta_obs = beta[mask]
    y_obs = y_true[mask]
    
    if gamma_obs.numel() == 0:
        # No observed samples, return zero loss
        return torch.tensor(0.0, device=gamma.device)
    
    # Negative log-likelihood (NLL) term
    # log p(y | γ, ν, α, β)
    pi = torch.tensor(3.14159265359, device=gamma.device)
    
    nll = (
        0.5 * torch.log(pi / nu_obs)
        - alpha_obs * torch.log(2 * beta_obs)
        + (alpha_obs + 0.5) * torch.log(
            nu_obs * (y_obs - gamma_obs)**2 + 2 * beta_obs
        )
        + torch.lgamma(alpha_obs)
        - torch.lgamma(alpha_obs + 0.5)
    )
    
    # Regularization term: penalize large errors with small evidence
    # Encourages model to have high uncertainty when uncertain
    error = torch.abs(y_obs - gamma_obs)
    evidence = 2 * nu_obs + alpha_obs
    reg = error * evidence
    
    # Total evidential loss
    loss = (nll + lambda_reg * reg).mean()
    
    return loss


def pbpk_physics_loss(
    predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    masks: Dict[str, torch.Tensor],
    fu_bounds: Tuple[float, float] = (0.0, 1.0),
    vd_min: float = 0.01,
    cl_max: float = 1500.0,
    cl_vd_ratio_max: float = 10.0
) -> torch.Tensor:
    """
    Physics-informed loss for PBPK constraints.
    
    Constraints:
        1. fu bounds: 0 ≤ fu ≤ 1
        2. Vd positivity: Vd > vd_min
        3. CL upper bound: CL ≤ cl_max (hepatic blood flow)
        4. CL/Vd ratio consistency: CL/Vd < cl_vd_ratio_max
    
    Args:
        predictions: Dictionary with keys 'fu', 'vd', 'cl'
                    Each contains (gamma, nu, alpha, beta) tensors
        masks: Dictionary with boolean masks for each parameter
        fu_bounds: (min, max) bounds for fu
        vd_min: Minimum Vd (L/kg)
        cl_max: Maximum CL (mL/min/kg)
        cl_vd_ratio_max: Maximum CL/Vd ratio
    
    Returns:
        Scalar physics loss value
    """
    # Extract predicted means
    fu_gamma = predictions['fu'][0]
    vd_gamma = predictions['vd'][0]
    cl_gamma = predictions['cl'][0]
    
    # Find samples with all three parameters observed
    complete_mask = masks['fu'] & masks['vd'] & masks['cl']
    
    if complete_mask.sum() == 0:
        # No complete samples, return zero physics loss
        return torch.tensor(0.0, device=fu_gamma.device)
    
    # Filter to complete samples
    fu_c = fu_gamma[complete_mask]
    vd_c = torch.exp(vd_gamma[complete_mask])  # Vd is log-scale
    cl_c = torch.exp(cl_gamma[complete_mask])  # CL is log-scale
    
    # Constraint 1: fu bounds [0, 1]
    loss_fu_lower = F.relu(-fu_c).mean()  # Penalize fu < 0
    loss_fu_upper = F.relu(fu_c - fu_bounds[1]).mean()  # Penalize fu > 1
    
    # Constraint 2: Vd positivity
    loss_vd_pos = F.relu(vd_min - vd_c).mean()  # Penalize Vd < vd_min
    
    # Constraint 3: CL upper bound (hepatic blood flow)
    loss_cl_upper = F.relu(cl_c - cl_max).mean()  # Penalize CL > cl_max
    
    # Constraint 4: CL/Vd ratio consistency
    cl_vd_ratio = cl_c / vd_c
    loss_ratio = F.relu(cl_vd_ratio - cl_vd_ratio_max).mean()
    
    # Total physics loss
    physics_loss = (
        loss_fu_lower + loss_fu_upper +
        loss_vd_pos + loss_cl_upper +
        loss_ratio
    )
    
    return physics_loss


def total_loss(
    predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    y_true: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    alpha: float = 1.0,
    beta: float = 0.01,
    lambda_reg: float = 0.01,
    fu_bounds: Tuple[float, float] = (0.0, 1.0),
    vd_min: float = 0.01,
    cl_max: float = 1500.0,
    cl_vd_ratio_max: float = 10.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Total KEC-PINN loss: α * L_data + β * L_physics
    
    Args:
        predictions: Dictionary with evidential parameters for fu, vd, cl
        y_true: Dictionary with ground truth targets
        masks: Dictionary with observation masks
        alpha: Weight for data loss (default 1.0)
        beta: Weight for physics loss (default 0.01)
        lambda_reg: Evidential regularization weight
        fu_bounds, vd_min, cl_max, cl_vd_ratio_max: Physics constraints
    
    Returns:
        Tuple of (total_loss, loss_components_dict)
    """
    # Data loss (evidential) for each parameter
    loss_data_fu = evidential_loss(
        *predictions['fu'], y_true['fu'], masks['fu'], lambda_reg
    )
    
    loss_data_vd = evidential_loss(
        *predictions['vd'], y_true['vd'], masks['vd'], lambda_reg
    )
    
    loss_data_cl = evidential_loss(
        *predictions['cl'], y_true['cl'], masks['cl'], lambda_reg
    )
    
    loss_data_total = loss_data_fu + loss_data_vd + loss_data_cl
    
    # Physics loss
    loss_physics = pbpk_physics_loss(
        predictions, masks,
        fu_bounds, vd_min, cl_max, cl_vd_ratio_max
    )
    
    # Combined loss
    loss_total = alpha * loss_data_total + beta * loss_physics
    
    # Return total loss and components for logging
    loss_components = {
        'total': loss_total,
        'data': loss_data_total,
        'data_fu': loss_data_fu,
        'data_vd': loss_data_vd,
        'data_cl': loss_data_cl,
        'physics': loss_physics,
        'alpha': alpha,
        'beta': beta
    }
    
    return loss_total, loss_components

