"""
Serviço de Embeddings para API

Converte SMILES em embeddings multimodais para predições PBPK.

Autor: Dr. Demetrios Chiuratto Agourakis
Criado: 2025-11-08
"""

import torch
import numpy as np
from typing import Optional, List
import logging
from pathlib import Path
import sys

# Adicionar path do projeto
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.ml.multimodal import MultimodalMolecularEncoder

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Serviço singleton para gerar embeddings moleculares.

    Usa MultimodalMolecularEncoder para converter SMILES em embeddings
    de 976 dimensões (ChemBERTa + GNN + KEC + 3D + QM).
    """

    _instance: Optional['EmbeddingService'] = None
    _encoder: Optional[MultimodalMolecularEncoder] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._encoder is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🔧 Inicializando EmbeddingService (device: {device})...")
            self._encoder = MultimodalMolecularEncoder(
                device=device,
                parallel=True,
                verbose=False  # Não mostrar logs detalhados na API
            )
            logger.info("✅ EmbeddingService inicializado")

    def encode(self, smiles: str) -> np.ndarray:
        """
        Codifica um SMILES em embedding multimodal.

        Args:
            smiles: SMILES string

        Returns:
            numpy array de shape (976,) com embedding multimodal
        """
        try:
            embedding = self._encoder.encode(smiles)
            return embedding
        except Exception as e:
            logger.error(f"Erro ao codificar SMILES '{smiles}': {e}")
            # Retornar embedding zero em caso de erro
            return np.zeros(976, dtype=np.float32)

    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Codifica múltiplos SMILES em batch.

        Args:
            smiles_list: Lista de SMILES strings

        Returns:
            numpy array de shape (len(smiles_list), 976)
        """
        embeddings = []
        for smiles in smiles_list:
            embedding = self.encode(smiles)
            embeddings.append(embedding)
        return np.array(embeddings)

    def encode_chemberta(self, smiles: str) -> np.ndarray:
        """
        Codifica um SMILES apenas com ChemBERTa (768d).

        Útil para modelos que esperam embeddings de 788d (768 ChemBERTa + 20 RDKit).

        Args:
            smiles: SMILES string

        Returns:
            numpy array de shape (768,) com embedding ChemBERTa
        """
        try:
            # Usar apenas encoder ChemBERTa do multimodal
            chemberta_encoder = self._encoder._encoders['chemberta']
            embedding = chemberta_encoder.encode(smiles)
            return embedding
        except Exception as e:
            logger.error(f"Erro ao codificar SMILES com ChemBERTa '{smiles}': {e}")
            return np.zeros(768, dtype=np.float32)


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Obtém instância singleton do EmbeddingService"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

