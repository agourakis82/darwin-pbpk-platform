"""
KEC-PINN: Knowledge-Enhanced Evidential Physics-Informed Neural Network
========================================================================

Main module for PBPK parameter prediction with uncertainty quantification.
"""

from .evidential_head import EvidentialHead
from .kec_pinn_model import KECPINN, TransformerAttention
from .kec_loss import evidential_loss, pbpk_physics_loss, total_loss

__all__ = [
    'EvidentialHead',
    'KECPINN',
    'TransformerAttention',
    'evidential_loss',
    'pbpk_physics_loss',
    'total_loss'
]

