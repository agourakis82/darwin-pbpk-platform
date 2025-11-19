"""
Dependencies para a API

Autor: Dr. Demetrios Chiuratto Agourakis
Criado: 2025-11-08
"""

from typing import Optional
import torch
from pathlib import Path
import sys

# Adicionar path do projeto
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def get_model(model_type: str, model_path: Optional[str] = None):
    """
    Dependency para carregar modelos sob demanda.

    TODO: Implementar cache de modelos
    """
    # Por enquanto, retornar None (lazy loading será implementado)
    return None

