"""
Evidential Deep Learning module for uncertainty quantification.
"""

from .evidential_head import EvidentialHead
from .evidential_loss import EvidentialLoss, expected_calibration_error

__all__ = ['EvidentialHead', 'EvidentialLoss', 'expected_calibration_error']

