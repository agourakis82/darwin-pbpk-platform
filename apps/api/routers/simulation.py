"""
Router para simulações PBPK

Endpoints:
- POST /simulate/dynamic-gnn - Simulação usando Dynamic GNN
- POST /simulate/ode - Simulação usando ODE solver

Autor: Dr. Demetrios Chiuratto Agourakis
Criado: 2025-11-08
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List
import torch
import numpy as np
from pathlib import Path
import sys
import logging

# Adicionar path do projeto
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.api.models import (
    PBPKSimulationRequest,
    PBPKResponse,
    PhysiologicalParams
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/simulate/dynamic-gnn", response_model=PBPKResponse)
async def simulate_dynamic_gnn(request: PBPKSimulationRequest):
    """
    Simulação PBPK usando Dynamic GNN (SOTA).

    Baseado em: arXiv 2024 (R² 0.9342)
    """
    try:
        logger.info(f"Simulação Dynamic GNN: SMILES={request.smiles[:50]}..., dose={request.dose}mg")

        from apps.pbpk_core.simulation import DynamicPBPKSimulator

        # Criar simulator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        simulator = DynamicPBPKSimulator(device=device)

        # Extrair parâmetros fisiológicos
        clearance_hepatic = 10.0  # Default
        clearance_renal = 5.0  # Default
        partition_coeffs = None

        if request.physiological_params:
            clearance_hepatic = request.physiological_params.clearance_hepatic or clearance_hepatic
            clearance_renal = request.physiological_params.clearance_renal or clearance_renal
            partition_coeffs = request.physiological_params.partition_coeffs

        # Time points
        if request.time_points is None:
            time_points = np.linspace(0, 24, 100).tolist()
        else:
            time_points = request.time_points

        # Simular
        result = simulator.simulate(
            dose=request.dose,
            clearance_hepatic=clearance_hepatic,
            clearance_renal=clearance_renal,
            partition_coeffs=partition_coeffs,
            time_points=np.array(time_points)
        )

        # Converter resultado
        concentrations = {}
        time_points_result = result.get("time", time_points)

        # Resultado já vem como dict com nomes dos órgãos
        for organ, conc_array in result.items():
            if organ != "time":
                concentrations[organ] = conc_array.tolist()

        # Garantir que todos os órgãos estão presentes
        organs = ["blood", "liver", "kidney", "brain", "heart", "lung",
                 "muscle", "adipose", "gut", "skin", "bone", "spleen",
                 "pancreas", "other"]
        for organ in organs:
            if organ not in concentrations:
                concentrations[organ] = [0.0] * len(time_points_result)

        # Resumo
        summary = {}
        for organ in organs:
            concs = concentrations.get(organ, [0.0])
            summary[f"{organ}_cmax"] = max(concs) if concs else 0.0
            summary[f"{organ}_tmax"] = time_points_result[np.argmax(concs)] if concs else 0.0
            if len(concs) > 1:
                auc = np.trapz(concs, time_points_result)
                summary[f"{organ}_auc"] = float(auc)
            else:
                summary[f"{organ}_auc"] = 0.0

        return PBPKResponse(
            smiles=request.smiles,
            dose=request.dose,
            route=request.route,
            model_type="dynamic_gnn",
            time_points=time_points_result.tolist() if hasattr(time_points_result, 'tolist') else time_points_result,
            concentrations=concentrations,
            summary=summary
        )

    except Exception as e:
        logger.error(f"Erro na simulação Dynamic GNN: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate/ode", response_model=PBPKResponse)
async def simulate_ode(request: PBPKSimulationRequest):
    """
    Simulação PBPK usando ODE solver tradicional.

    Método clássico para comparação com Dynamic GNN.
    """
    try:
        logger.info(f"Simulação ODE: SMILES={request.smiles[:50]}..., dose={request.dose}mg")

        from apps.pbpk_core.simulation import ODEPBPKSolver

        # TODO: Implementar simulação ODE completa
        raise HTTPException(
            status_code=501,
            detail="Simulação ODE ainda não implementada na API"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na simulação ODE: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

