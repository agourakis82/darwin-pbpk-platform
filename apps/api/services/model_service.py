"""
Serviço de Modelos para API

Carrega modelos treinados e faz predições de parâmetros PK.

Autor: Dr. Demetrios Chiuratto Agourakis
Criado: 2025-11-08
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List
import logging
from pathlib import Path
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors

# Adicionar path do projeto
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class FlexiblePKModel(nn.Module):
    """
    Modelo Trial 84 para predição de parâmetros PK.

    Arquitetura:
    - Input: 788d (768d ChemBERTa + 20d RDKit)
    - Hidden: [384, 1024, 640]
    - Dropout: 0.492
    - Activation: GELU
    - Output: 3 (Fu, Vd, Clearance)
    """

    def __init__(
        self,
        input_dim=788,
        hidden_dims=[384, 1024, 640],
        output_dim=3,
        dropout=0.492,
        use_batch_norm=False,
        activation='gelu'
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'elu':
                layers.append(nn.ELU())

            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def compute_rdkit_descriptors(mol: Chem.Mol) -> np.ndarray:
    """
    Computa 20 descritores RDKit padrão.

    Returns:
        numpy array de shape (20,)
    """
    if mol is None:
        return np.zeros(20, dtype=np.float32)

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.FractionCsp3(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.BertzCT(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol),
        Descriptors.Kappa3(mol),
        Descriptors.Chi0(mol),
        Descriptors.Chi1(mol),
        Descriptors.Chi0n(mol),
        Descriptors.Chi1n(mol),
    ]

    # Normalizar e tratar NaN
    descriptors = np.array(descriptors, dtype=np.float32)
    descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=1000.0, neginf=-1000.0)

    return descriptors


def inverse_transform_fu(pred: float) -> float:
    """Inverse logit transform para Fu"""
    pred = np.clip(pred, -10, 10)
    return float(1 / (1 + np.exp(-pred)))


def inverse_transform_vd(pred: float) -> float:
    """Inverse log1p transform para Vd"""
    pred = np.clip(pred, -10, 10)
    return float(np.expm1(pred))


def inverse_transform_clearance(pred: float) -> float:
    """Inverse log1p transform para Clearance"""
    pred = np.clip(pred, -10, 10)
    return float(np.expm1(pred))


class ModelService:
    """
    Serviço singleton para carregar e usar modelos treinados.
    """

    _instance: Optional['ModelService'] = None
    _models: Dict[str, nn.Module] = {}
    _device: str = "cpu"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🔧 Inicializando ModelService (device: {self._device})...")
            self._initialized = True

    def load_model(self, model_name: str, model_path: Path) -> bool:
        """
        Carrega um modelo treinado.

        Args:
            model_name: Nome do modelo (ex: 'flexible_pk')
            model_path: Caminho para o checkpoint (.pt)

        Returns:
            True se carregado com sucesso
        """
        try:
            if not model_path.exists():
                logger.warning(f"Modelo não encontrado: {model_path}")
                return False

            checkpoint = torch.load(model_path, map_location=self._device)

            # Criar modelo
            if model_name == "flexible_pk":
                model = FlexiblePKModel(
                    input_dim=788,
                    hidden_dims=[384, 1024, 640],
                    output_dim=3,
                    dropout=0.492,
                    activation='gelu'
                )
            else:
                logger.error(f"Tipo de modelo desconhecido: {model_name}")
                return False

            # Carregar weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(self._device)
            model.eval()

            self._models[model_name] = model
            logger.info(f"✅ Modelo '{model_name}' carregado de {model_path}")
            return True

        except Exception as e:
            logger.error(f"Erro ao carregar modelo '{model_name}': {e}", exc_info=True)
            return False

    def predict_pk_parameters(
        self,
        chemberta_embedding: np.ndarray,
        rdkit_descriptors: np.ndarray
    ) -> Dict[str, float]:
        """
        Prediz parâmetros PK usando modelo FlexiblePK.

        Args:
            chemberta_embedding: Embedding ChemBERTa (768d)
            rdkit_descriptors: Descritores RDKit (20d)

        Returns:
            Dict com fu_plasma, vd, clearance
        """
        if "flexible_pk" not in self._models:
            logger.warning("Modelo 'flexible_pk' não carregado, usando valores padrão")
            return {
                "fu_plasma": 0.1,
                "vd": 1.0,
                "clearance": 1.0
            }

        try:
            # Concatenar embeddings (788d total)
            if chemberta_embedding.shape[0] != 768:
                # Se embedding multimodal (976d), usar apenas primeiros 768d
                chemberta_embedding = chemberta_embedding[:768]

            features = np.concatenate([chemberta_embedding, rdkit_descriptors])
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self._device)

            # Predição
            model = self._models["flexible_pk"]
            with torch.no_grad():
                pred = model(features)
                pred = pred.cpu().numpy()[0]

            # Inverse transforms
            fu = inverse_transform_fu(pred[0])
            vd = inverse_transform_vd(pred[1])
            clearance = inverse_transform_clearance(pred[2])

            return {
                "fu_plasma": fu,
                "vd": vd,
                "clearance": clearance
            }

        except Exception as e:
            logger.error(f"Erro na predição: {e}", exc_info=True)
            return {
                "fu_plasma": 0.1,
                "vd": 1.0,
                "clearance": 1.0
            }

    def get_loaded_models(self) -> List[str]:
        """Retorna lista de modelos carregados"""
        return list(self._models.keys())


# Singleton instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Obtém instância singleton do ModelService"""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service

