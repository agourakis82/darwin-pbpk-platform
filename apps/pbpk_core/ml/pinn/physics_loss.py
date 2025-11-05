"""
Physics Loss - PBPK ODE Constraints
===================================

Week 2 - MOONSHOT Implementation
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class PhysicsLoss:
    """
    Physics-based loss functions for PINN.
    
    Combines:
    - Data loss (MSE or Huber on known parameters)
    - Physics loss (ODE residuals)
    - Boundary conditions (physiological ranges)
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        beta: float = 0.1, 
        gamma: float = 0.05,
        loss_fn: str = 'mse',
        huber_delta: float = 1.0
    ):
        """
        Args:
            alpha: Weight for data loss
            beta: Weight for physics loss
            gamma: Weight for boundary loss
            loss_fn: Data loss function ('mse' or 'huber')
            huber_delta: Delta parameter for Huber loss (if used)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_fn = loss_fn
        self.huber_delta = huber_delta
        
        if loss_fn == 'mse':
            self.mse = nn.MSELoss()
        elif loss_fn == 'huber':
            self.mse = nn.HuberLoss(delta=huber_delta)
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}. Use 'mse' or 'huber'")
    
    def data_loss(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Data loss (MSE or Huber) on known parameters"""
        loss = torch.tensor(0.0, device=list(pred.values())[0].device)
        count = 0
        
        for key in ['fu', 'vd', 'cl']:
            if key in pred and key in target:
                loss += self.mse(pred[key], target[key])
                count += 1
        
        return loss / max(count, 1)
    
    def physics_loss(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Physics-based constraints (simplified for Week 2).
        Full PBPK ODE implementation in Week 3.
        """
        loss = torch.tensor(0.0, device=list(params.values())[0].device)
        
        # Example: Penalize extreme parameter values
        if 'vd' in params and 'cl' in params:
            # Elimination rate constant should be reasonable
            k_elim = params['cl'] / (params['vd'] + 1e-6)
            # Penalize if k_elim is too extreme
            loss += torch.mean(torch.relu(k_elim - 10.0))  # k > 10 h^-1 is unlikely
            loss += torch.mean(torch.relu(0.001 - k_elim))  # k < 0.001 h^-1 is unlikely
        
        return loss
    
    def boundary_loss(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enforce physiological parameter ranges"""
        loss = torch.tensor(0.0, device=list(params.values())[0].device)
        
        # fu should be in (0, 1) - already enforced by sigmoid, but add soft penalty
        if 'fu' in params:
            # Penalize values too close to boundaries
            loss += torch.mean(torch.exp(-100 * params['fu']))  # Too close to 0
            loss += torch.mean(torch.exp(-100 * (1 - params['fu'])))  # Too close to 1
        
        # Vd should be in reasonable range (0.1 - 1000 L)
        if 'vd' in params:
            loss += torch.mean(torch.relu(0.1 - params['vd']))
            loss += torch.mean(torch.relu(params['vd'] - 1000.0))
        
        # CL should be in reasonable range (0.01 - 100 L/h)
        if 'cl' in params:
            loss += torch.mean(torch.relu(0.01 - params['cl']))
            loss += torch.mean(torch.relu(params['cl'] - 100.0))
        
        return loss
    
    def __call__(
        self,
        params_pred: Dict[str, torch.Tensor],
        params_target: Optional[Dict[str, torch.Tensor]] = None,
        compute_physics: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Returns:
            Dict with individual and total losses
        """
        losses = {}
        
        # Data loss
        if params_target is not None:
            losses['data'] = self.alpha * self.data_loss(params_pred, params_target)
        else:
            losses['data'] = torch.tensor(0.0, device=list(params_pred.values())[0].device)
        
        # Physics loss
        if compute_physics:
            losses['physics'] = self.beta * self.physics_loss(params_pred)
        else:
            losses['physics'] = torch.tensor(0.0, device=list(params_pred.values())[0].device)
        
        # Boundary loss
        losses['boundary'] = self.gamma * self.boundary_loss(params_pred)
        
        # Total loss
        losses['total'] = losses['data'] + losses['physics'] + losses['boundary']
        
        return losses

