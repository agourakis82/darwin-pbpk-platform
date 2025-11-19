"""
Serviços da API

- EmbeddingService: Converte SMILES em embeddings
- ModelService: Carrega modelos treinados e faz predições
"""

from .embedding_service import EmbeddingService, get_embedding_service
from .model_service import ModelService, get_model_service, compute_rdkit_descriptors

__all__ = [
    'EmbeddingService',
    'get_embedding_service',
    'ModelService',
    'get_model_service',
    'compute_rdkit_descriptors'
]

