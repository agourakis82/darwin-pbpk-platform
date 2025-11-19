"""
Router para gerenciamento de modelos

Endpoints:
- GET /models - Lista modelos disponíveis
- GET /models/{model_name} - Informações sobre um modelo específico

Autor: Dr. Demetrios Chiuratto Agourakis
Criado: 2025-11-08
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys
import logging
import torch

# Adicionar path do projeto
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.api.models import ModelsListResponse, ModelInfo

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    Lista todos os modelos disponíveis.
    """
    try:
        models_dir = project_root / "models"
        models = []

        if not models_dir.exists():
            return ModelsListResponse(models=[], total=0)

        # Procurar por modelos treinados
        model_patterns = [
            ("dynamic_gnn", "Dynamic GNN PBPK", "dynamic_gnn*"),
            ("gnn_multitask", "GNN Multi-task (Fu, Vd, CL)", "gnn_multitask*"),
            ("mlp_baseline", "MLP Baseline", "mlp_baseline*"),
        ]

        for model_type, model_name, pattern in model_patterns:
            # Procurar diretórios de modelos
            for model_path in models_dir.glob(pattern):
                if model_path.is_dir():
                    # Verificar se tem best_model.pt
                    best_model = model_path / "best_model.pt"
                    if best_model.exists():
                        # Tentar carregar para contar parâmetros
                        try:
                            checkpoint = torch.load(best_model, map_location="cpu")
                            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                                state_dict = checkpoint["model_state_dict"]
                                num_params = sum(p.numel() for p in state_dict.values())
                            else:
                                num_params = None
                        except Exception:
                            num_params = None

                        models.append(ModelInfo(
                            name=model_name,
                            type=model_type,
                            path=str(model_path.relative_to(project_root)),
                            loaded=False,  # TODO: verificar se está carregado
                            parameters=num_params,
                            device="cpu"  # TODO: detectar device real
                        ))

        return ModelsListResponse(
            models=models,
            total=len(models)
        )

    except Exception as e:
        logger.error(f"Erro ao listar modelos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    Informações sobre um modelo específico.
    """
    try:
        models_dir = project_root / "models"
        model_path = models_dir / model_name

        if not model_path.exists() or not model_path.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"Modelo {model_name} não encontrado"
            )

        best_model = model_path / "best_model.pt"
        if not best_model.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint do modelo {model_name} não encontrado"
            )

        # Carregar informações do modelo
        try:
            checkpoint = torch.load(best_model, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                num_params = sum(p.numel() for p in state_dict.values())
            else:
                num_params = None
        except Exception:
            num_params = None

        return ModelInfo(
            name=model_name,
            type=model_name.split("_")[0] if "_" in model_name else model_name,
            path=str(model_path.relative_to(project_root)),
            loaded=False,
            parameters=num_params,
            device="cpu"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

