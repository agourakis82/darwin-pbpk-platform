"""
PBPK (Physiologically-Based Pharmacokinetics) Physics Constraints.

Implements various physics-informed constraints:
1. Physiological bounds (fu ∈ [0,1], Vd > 0, CL > 0)
2. Mass balance relationships
3. Clearance-Volume relationships

Reference:
- Rowland & Tozer, "Clinical Pharmacokinetics and Pharmacodynamics" (2011)
"""

import torch
import torch.nn as nn
import numpy as np


class PBPKConstraints:
    """
    PBPK physics constraints for pharmacokinetic parameters.
    
    Constraints:
    1. Bounds: fu ∈ [0, 1], Vd > 0, CL > 0
    2. Mass balance: Simplified PBPK relationships
    3. Physiological plausibility
    """
    
    @staticmethod
    def bounds_penalty(fu, vd, cl):
        """
        Penalize violations of physiological bounds.
        
        Args:
            fu: Fraction unbound (logit space)
            vd: Volume of distribution (log1p space)
            cl: Clearance (log1p space)
            
        Returns:
            Penalty value (0 if all constraints satisfied)
        """
        penalty = 0.0
        
        # Fu should be in [0, 1] after sigmoid
        # In logit space, extreme values indicate violations
        fu_sigmoid = torch.sigmoid(fu)
        # Penalize if too close to boundaries (numerical issues)
        penalty += torch.mean(torch.relu(fu_sigmoid - 0.99))  # Too high
        penalty += torch.mean(torch.relu(0.01 - fu_sigmoid))  # Too low
        
        # Vd and CL should be positive (in log space, already enforced)
        # But penalize if too extreme
        penalty += torch.mean(torch.relu(vd - 10.0))  # log1p(Vd) > 10 → Vd > 22000 L (unrealistic!)
        penalty += torch.mean(torch.relu(cl - 10.0))  # log1p(CL) > 10 → CL > 22000 L/h (unrealistic!)
        
        return penalty
    
    @staticmethod
    def mass_balance_penalty(fu, vd, cl):
        """
        Simplified mass balance constraint.
        
        In PBPK, clearance relates to volume and fraction unbound:
        CL_total ≈ fu * CL_intrinsic
        
        This is a soft relationship, not strict.
        
        Args:
            fu: Fraction unbound (logit space)
            vd: Volume of distribution (log1p space)
            cl: Clearance (log1p space)
            
        Returns:
            Mass balance penalty
        """
        # Convert to original space
        fu_real = torch.sigmoid(fu)
        vd_real = torch.expm1(vd)  # inverse of log1p
        cl_real = torch.expm1(cl)
        
        # Simplified relationship: CL shouldn't be too large relative to Vd * fu
        # This is a very soft constraint
        ratio = cl_real / (vd_real * fu_real + 1e-6)
        
        # Penalize extreme ratios (both too high and too low)
        penalty = torch.mean(torch.relu(ratio - 100.0))  # Too high
        penalty += torch.mean(torch.relu(0.01 - ratio))  # Too low
        
        return penalty
    
    @staticmethod
    def all_constraints(fu, vd, cl, weights={'bounds': 1.0, 'mass_balance': 0.1}):
        """
        Compute all physics constraints.
        
        Args:
            fu, vd, cl: Predicted parameters (in transformed space)
            weights: Dictionary of weights for each constraint type
            
        Returns:
            Total physics penalty
        """
        penalty = 0.0
        
        if weights.get('bounds', 0) > 0:
            penalty += weights['bounds'] * PBPKConstraints.bounds_penalty(fu, vd, cl)
        
        if weights.get('mass_balance', 0) > 0:
            penalty += weights['mass_balance'] * PBPKConstraints.mass_balance_penalty(fu, vd, cl)
        
        return penalty


class PhysicsLoss(nn.Module):
    """
    Combined loss: Data fit + Physics constraints.
    
    Variants:
    1. Soft: Simple weighted sum
    2. Curriculum: Gradually increase physics weight
    3. Selective: Apply only to high-confidence predictions
    4. Bayesian: Sample physics parameters
    5. Validation-only: Physics for early stopping, not backprop
    """
    
    def __init__(self, variant='soft', beta=0.01, **kwargs):
        """
        Args:
            variant: 'soft', 'curriculum', 'selective', 'bayesian', 'validation_only'
            beta: Physics weight (for soft variant)
            **kwargs: Additional variant-specific parameters
        """
        super().__init__()
        self.variant = variant
        self.beta = beta
        self.kwargs = kwargs
        
        # For curriculum
        self.current_epoch = 0
        self.warmup_epochs = kwargs.get('warmup_epochs', 20)
        
    def forward(self, predictions, targets, masks, epoch=None):
        """
        Compute combined loss.
        
        Args:
            predictions: Dict with 'fu', 'vd', 'cl' predictions
            targets: Dict with 'fu', 'vd', 'cl' targets
            masks: Dict with 'fu', 'vd', 'cl' masks (which samples have data)
            epoch: Current epoch (for curriculum)
            
        Returns:
            Total loss (data + physics)
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        # Data loss (MSE on observed values)
        data_loss = 0.0
        loss_fn = nn.MSELoss()
        
        for target_name in ['fu', 'vd', 'cl']:
            mask = masks[target_name]
            if mask.sum() > 0:
                data_loss += loss_fn(
                    predictions[target_name][mask],
                    targets[target_name][mask]
                )
        
        # Physics loss
        physics_loss = PBPKConstraints.all_constraints(
            predictions['fu'],
            predictions['vd'],
            predictions['cl']
        )
        
        # Combine based on variant
        if self.variant == 'soft':
            total_loss = data_loss + self.beta * physics_loss
            
        elif self.variant == 'curriculum':
            # Gradually increase physics weight
            if self.current_epoch < self.warmup_epochs:
                beta_t = 0.0  # No physics during warmup
            else:
                progress = (self.current_epoch - self.warmup_epochs) / self.warmup_epochs
                beta_t = self.beta * min(1.0, progress)
            
            total_loss = data_loss + beta_t * physics_loss
            
        elif self.variant == 'selective':
            # Only apply physics to samples with all targets observed
            # (high confidence about physics relationships)
            all_observed = masks['fu'] & masks['vd'] & masks['cl']
            
            if all_observed.sum() > 0:
                selective_physics = PBPKConstraints.all_constraints(
                    predictions['fu'][all_observed],
                    predictions['vd'][all_observed],
                    predictions['cl'][all_observed]
                )
                total_loss = data_loss + self.beta * selective_physics
            else:
                total_loss = data_loss
                
        elif self.variant == 'bayesian':
            # Sample beta from a distribution (simple version)
            # In practice, would use proper Bayesian inference
            beta_sample = self.beta * (1.0 + 0.2 * torch.randn(1).item())  # ±20% noise
            beta_sample = max(0.0, beta_sample)
            
            total_loss = data_loss + beta_sample * physics_loss
            
        elif self.variant == 'validation_only':
            # Physics not in training loss (only for monitoring)
            total_loss = data_loss
            
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        
        return total_loss, data_loss, physics_loss


def compute_physics_metrics(predictions, targets, masks):
    """
    Compute physics constraint violations for monitoring.
    
    Args:
        predictions: Dict with 'fu', 'vd', 'cl'
        targets: Dict with 'fu', 'vd', 'cl'
        masks: Dict with 'fu', 'vd', 'cl'
        
    Returns:
        Dictionary with physics metrics
    """
    with torch.no_grad():
        # Bounds violations
        bounds_penalty = PBPKConstraints.bounds_penalty(
            predictions['fu'],
            predictions['vd'],
            predictions['cl']
        ).item()
        
        # Mass balance violations
        mass_balance_penalty = PBPKConstraints.mass_balance_penalty(
            predictions['fu'],
            predictions['vd'],
            predictions['cl']
        ).item()
        
        # Total physics violation
        total_physics = PBPKConstraints.all_constraints(
            predictions['fu'],
            predictions['vd'],
            predictions['cl']
        ).item()
        
        # Percentage of samples violating bounds
        fu_sigmoid = torch.sigmoid(predictions['fu'])
        fu_violations = ((fu_sigmoid < 0.01) | (fu_sigmoid > 0.99)).float().mean().item()
        
        vd_violations = (predictions['vd'] > 10.0).float().mean().item()
        cl_violations = (predictions['cl'] > 10.0).float().mean().item()
        
    return {
        'bounds_penalty': bounds_penalty,
        'mass_balance_penalty': mass_balance_penalty,
        'total_physics': total_physics,
        'fu_violations_%': fu_violations * 100,
        'vd_violations_%': vd_violations * 100,
        'cl_violations_%': cl_violations * 100
    }

