"""
Multimodal Molecular Encoder - REAL IMPLEMENTATION
===================================================

Integra 5 encoders moleculares ortogonais:

1. ChemBERTa (768 dim) - Semantic/latent features via Transformers
2. GNN (128 dim) - Topological features via graph convolutions
3. KEC (15 dim) - Fractal-entropic descriptors (NOVEL - Master's thesis)
4. 3D Conformer (50 dim) - Spatial/geometric descriptors
5. QM (15 dim) - Quantum/electronic descriptors

TOTAL: 976 dimensions - 100% REAL, SEM MOCKS!

Autor: Dr. Demetrios Chiuratto Agourakis
Data: Outubro 2025
"""

from .multimodal_encoder import MultimodalMolecularEncoder

__all__ = ['MultimodalMolecularEncoder']

