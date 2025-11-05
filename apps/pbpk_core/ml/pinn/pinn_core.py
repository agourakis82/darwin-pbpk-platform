"""
PINN Core - Physics-Informed Neural Network for PBPK
====================================================

Week 2 - MOONSHOT Implementation
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PINNConfig:
    """Configuration for PINN model"""
    input_dim: int = 981
    hidden_dims: List[int] = None
    predict_fu: bool = True
    predict_vd: bool = True
    predict_cl: bool = True
    dropout: float = 0.2
    loss_fn: str = 'mse'  # 'mse' or 'huber'
    huber_delta: float = 1.0  # Delta for Huber loss
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


class PBPKPhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for PBPK modeling.
    
    Learns PK parameters (fu, Vd, CL) from multimodal embeddings
    while respecting PBPK ODEs.
    """
    
    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder
        layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Task-specific heads
        if config.predict_fu:
            self.fu_head = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # fu ∈ [0,1]
            )
        
        if config.predict_vd:
            self.vd_head = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()  # Vd > 0
            )
        
        if config.predict_cl:
            self.cl_head = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()  # CL > 0
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PINN.
        
        Args:
            x: Multimodal embeddings [batch_size, 981]
            
        Returns:
            Dict with predicted parameters
        """
        features = self.encoder(x)
        
        output = {}
        if self.config.predict_fu:
            output['fu'] = self.fu_head(features)
        if self.config.predict_vd:
            output['vd'] = self.vd_head(features)
        if self.config.predict_cl:
            output['cl'] = self.cl_head(features)
        
        return output
    
    def predict(self, x: torch.Tensor) -> Dict[str, float]:
        """Make prediction and return as dict"""
        self.eval()
        with torch.no_grad():
            output = self(x)
            return {k: v.item() for k, v in output.items()}

