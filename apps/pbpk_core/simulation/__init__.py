"""
PBPK Simulation Module

Módulos:
- dynamic_gnn_pbpk: Dynamic Graph Neural Network para simulação PBPK (SOTA)
- ode_pbpk_solver: ODE solver tradicional (ground truth para treinamento)
"""

from .dynamic_gnn_pbpk import (
    DynamicPBPKGNN,
    DynamicPBPKSimulator,
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
    NUM_ORGANS,
    DEFAULT_DYNAMIC_GNN_CHECKPOINT,
)

from .ode_pbpk_solver import (
    ODEPBPKSolver,
    ODEState
)

__all__ = [
    "DynamicPBPKGNN",
    "DynamicPBPKSimulator",
    "PBPKPhysiologicalParams",
    "PBPK_ORGANS",
    "NUM_ORGANS",
    "DEFAULT_DYNAMIC_GNN_CHECKPOINT",
    "ODEPBPKSolver",
    "ODEState"
]

