"""
KEC-PINN: Complete Model Architecture
======================================

Knowledge-Enhanced Evidential Physics-Informed Neural Network for
PBPK parameter prediction with uncertainty quantification.

Architecture:
    ChemBERTa (768d) + GIN (256d) + KEC (10d) + Mol Props (4d)
    → Concat (1038d)
    → Transformer Attention (→ 519d)
    → Shared Encoder (→ 128d)
    → 3 Evidential Heads (fu, Vd, CL)

Author: Dr. Agourakis
Date: October 26, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .evidential_head import EvidentialHead


class TransformerAttention(nn.Module):
    """
    Transformer attention module with self-attention and feed-forward.
    """
    
    def __init__(
        self,
        d_model: int = 1038,
        nhead: int = 6,  # Must divide d_model (1038 / 6 = 173)
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize transformer attention module.
        
        Args:
            d_model: Input dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feed-forward network
            dropout: Dropout probability
        """
        super(TransformerAttention, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model // 2)  # Reduce to 519
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model // 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer attention.
        
        Args:
            x: Input features [batch_size, d_model]
        
        Returns:
            Output features [batch_size, d_model // 2]
        """
        # Add sequence dimension for attention [batch, 1, d_model]
        x_seq = x.unsqueeze(1)
        
        # Self-attention with residual
        attn_out, _ = self.self_attn(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(1)  # Remove sequence dim
        x = self.norm1(x + attn_out)
        
        # Feed-forward with dimension reduction
        ff_out = self.linear1(x)
        ff_out = F.relu(ff_out)
        ff_out = self.dropout1(ff_out)
        ff_out = self.linear2(ff_out)
        ff_out = self.dropout2(ff_out)
        
        # Residual connection (need to match dimensions)
        # Use first half of x for residual
        x_reduced = x[:, :self.d_model // 2]
        out = self.norm2(x_reduced + ff_out)
        
        return out


class KECPINN(nn.Module):
    """
    Complete KEC-PINN model for PBPK parameter prediction.
    
    Inputs:
        - x_chemberta: ChemBERTa embeddings [batch, 768]
        - x_gin: GIN graph embeddings [batch, 256]
        - x_kec: KEC topological features [batch, 10]
        - x_props: Additional molecular properties [batch, 4]
    
    Outputs:
        - Dictionary with evidential parameters for fu, Vd, CL
    """
    
    def __init__(
        self,
        chemberta_dim: int = 768,
        gin_dim: int = 256,
        kec_dim: int = 10,
        props_dim: int = 4,
        attention_nhead: int = 6,  # Must divide total input_dim (1038)
        attention_dim_ff: int = 2048,
        shared_hidden_dims: list = [256, 128],
        evidential_hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize KEC-PINN model.
        
        Args:
            chemberta_dim: ChemBERTa embedding dimension
            gin_dim: GIN embedding dimension
            kec_dim: KEC feature dimension
            props_dim: Additional molecular properties dimension
            attention_nhead: Number of attention heads
            attention_dim_ff: Dimension of attention feed-forward
            shared_hidden_dims: Hidden dimensions for shared encoder
            evidential_hidden_dim: Hidden dimension for evidential heads
            dropout: Dropout probability
        """
        super(KECPINN, self).__init__()
        
        self.chemberta_dim = chemberta_dim
        self.gin_dim = gin_dim
        self.kec_dim = kec_dim
        self.props_dim = props_dim
        
        # Total input dimension
        self.input_dim = chemberta_dim + gin_dim + kec_dim + props_dim
        
        # Transformer attention module
        self.attention = TransformerAttention(
            d_model=self.input_dim,
            nhead=attention_nhead,
            dim_feedforward=attention_dim_ff,
            dropout=dropout
        )
        
        # Shared encoder
        self.shared_encoder = nn.ModuleList()
        prev_dim = self.input_dim // 2  # Output from attention
        
        for hidden_dim in shared_hidden_dims:
            self.shared_encoder.append(nn.Linear(prev_dim, hidden_dim))
            self.shared_encoder.append(nn.ReLU())
            self.shared_encoder.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Evidential heads for each parameter
        self.head_fu = EvidentialHead(
            input_dim=shared_hidden_dims[-1],
            hidden_dim=evidential_hidden_dim,
            dropout=dropout
        )
        
        self.head_vd = EvidentialHead(
            input_dim=shared_hidden_dims[-1],
            hidden_dim=evidential_hidden_dim,
            dropout=dropout
        )
        
        self.head_cl = EvidentialHead(
            input_dim=shared_hidden_dims[-1],
            hidden_dim=evidential_hidden_dim,
            dropout=dropout
        )
        
    def forward(
        self,
        x_chemberta: torch.Tensor,
        x_gin: torch.Tensor,
        x_kec: torch.Tensor,
        x_props: torch.Tensor = None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through KEC-PINN.
        
        Args:
            x_chemberta: ChemBERTa embeddings [batch, 768]
            x_gin: GIN embeddings [batch, 256]
            x_kec: KEC features [batch, 10]
            x_props: Additional properties [batch, 4] (optional)
        
        Returns:
            Dictionary with keys 'fu', 'vd', 'cl', each containing
            tuple of (gamma, nu, alpha, beta)
        """
        # Concatenate all features
        if x_props is not None:
            x_hybrid = torch.cat([x_chemberta, x_gin, x_kec, x_props], dim=-1)
        else:
            # Use zeros for props if not provided
            batch_size = x_chemberta.shape[0]
            x_props_dummy = torch.zeros(batch_size, self.props_dim, device=x_chemberta.device)
            x_hybrid = torch.cat([x_chemberta, x_gin, x_kec, x_props_dummy], dim=-1)
        
        # Attention module
        x_attn = self.attention(x_hybrid)
        
        # Shared encoder
        x_shared = x_attn
        for layer in self.shared_encoder:
            x_shared = layer(x_shared)
        
        # Evidential heads
        fu_params = self.head_fu(x_shared)
        vd_params = self.head_vd(x_shared)
        cl_params = self.head_cl(x_shared)
        
        return {
            'fu': fu_params,
            'vd': vd_params,
            'cl': cl_params
        }
    
    def predict_with_uncertainty(
        self,
        x_chemberta: torch.Tensor,
        x_gin: torch.Tensor,
        x_kec: torch.Tensor,
        x_props: torch.Tensor = None
    ) -> Dict[str, dict]:
        """
        Predict with full uncertainty quantification.
        
        Args:
            Same as forward()
        
        Returns:
            Dictionary with keys 'fu', 'vd', 'cl', each containing:
                - mean: Predicted value
                - epistemic_std: Epistemic uncertainty
                - aleatoric_std: Aleatoric uncertainty
                - total_std: Total uncertainty
        """
        # Get evidential parameters
        params_dict = self.forward(x_chemberta, x_gin, x_kec, x_props)
        
        # Compute uncertainties for each parameter
        results = {}
        for param_name, params in params_dict.items():
            gamma, nu, alpha, beta = params
            
            epistemic_var = beta / (alpha - 1)
            aleatoric_var = beta / (nu * (alpha - 1))
            total_var = epistemic_var + aleatoric_var
            
            results[param_name] = {
                'mean': gamma,
                'epistemic_std': torch.sqrt(epistemic_var),
                'aleatoric_std': torch.sqrt(aleatoric_var),
                'total_std': torch.sqrt(total_var)
            }
        
        return results
    
    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return (
            f"KECPINN(\n"
            f"  input_dims: ChemBERTa={self.chemberta_dim}, GIN={self.gin_dim}, "
            f"KEC={self.kec_dim}, Props={self.props_dim}\n"
            f"  total_input_dim: {self.input_dim}\n"
            f"  total_parameters: {total_params:,}\n"
            f"  trainable_parameters: {trainable_params:,}\n"
            f")"
        )

