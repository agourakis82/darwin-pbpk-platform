"""
Router para predições PBPK

Endpoints:
- POST /predict/pbpk - Predição PBPK completa
- POST /predict/parameters - Predição de parâmetros PK

Autor: Dr. Demetrios Chiuratto Agourakis
Criado: 2025-11-08
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import torch
import numpy as np
from pathlib import Path
import sys
import logging

# Adicionar path do projeto
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.api.services import (
    get_embedding_service,
    get_model_service,
    compute_rdkit_descriptors
)
from rdkit import Chem

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict/pbpk", response_model=PBPKResponse)
async def predict_pbpk(request: PBPKRequest):
    """
    Predição PBPK completa usando Dynamic GNN ou ODE solver.

    Retorna concentrações ao longo do tempo para todos os órgãos.
    """
    try:
        logger.info(f"Predição PBPK: SMILES={request.smiles[:50]}..., dose={request.dose}mg, route={request.route}")

        # Por enquanto, usar Dynamic GNN como padrão
        if request.model_type == ModelType.DYNAMIC_GNN:
            from apps.pbpk_core.simulation import DynamicPBPKSimulator

            # Criar simulator
            device = "cuda" if torch.cuda.is_available() else "cpu"
            simulator = DynamicPBPKSimulator(device=device)

            # Time points padrão se não fornecido
            if request.time_points is None:
                time_points = np.linspace(0, 24, 100).tolist()  # 0-24h, 100 pontos
            else:
                time_points = request.time_points

            # Converter SMILES para embeddings e predizer parâmetros PK
            embedding_service = get_embedding_service()
            model_service = get_model_service()

            # Obter embedding ChemBERTa e descritores RDKit
            mol = Chem.MolFromSmiles(request.smiles)
            if mol is None:
                raise HTTPException(status_code=400, detail=f"SMILES inválido: {request.smiles}")

            chemberta_embedding = embedding_service.encode_chemberta(request.smiles)
            rdkit_descriptors = compute_rdkit_descriptors(mol)

            # Predizer parâmetros PK
            pk_params = model_service.predict_pk_parameters(chemberta_embedding, rdkit_descriptors)

            # Estimar clearance e partition coeffs a partir dos parâmetros PK
            # Clearance total ≈ clearance predito * peso corporal (70kg padrão)
            clearance_total = pk_params["clearance"] * 70.0  # L/h
            clearance_hepatic = clearance_total * 0.7  # ~70% hepático
            clearance_renal = clearance_total * 0.3  # ~30% renal

            # Estimar partition coefficients básicos (simplificado)
            # Kp aproximado baseado em Vd e propriedades moleculares
            partition_coeffs = {
                "liver": 1.5,  # Típico para muitos fármacos
                "kidney": 1.2,
                "brain": 0.5 if pk_params["fu_plasma"] > 0.1 else 0.1,  # BBB
                "adipose": 2.0 if pk_params["vd"] > 1.0 else 1.0,  # Lipofílico
            }

            # Simular
            result = simulator.simulate(
                dose=request.dose,
                clearance_hepatic=clearance_hepatic,
                clearance_renal=clearance_renal,
                partition_coeffs=partition_coeffs,
                time_points=np.array(time_points)
            )

            # Converter resultado para formato de resposta
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

            # Calcular resumo (Cmax, Tmax, AUC)
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
                model_type=request.model_type.value,
                time_points=time_points_result.tolist() if hasattr(time_points_result, 'tolist') else time_points_result,
                concentrations=concentrations,
                summary=summary
            )

        elif request.model_type == ModelType.ODE_SOLVER:
            from apps.pbpk_core.simulation import ODEPBPKSolver

            # TODO: Implementar ODE solver
            raise HTTPException(
                status_code=501,
                detail="ODE solver ainda não implementado na API"
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Modelo {request.model_type} não suportado"
            )

    except Exception as e:
        logger.error(f"Erro na predição PBPK: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/parameters", response_model=PKParametersResponse)
async def predict_pk_parameters(request: PKParametersRequest):
    """
    Predição de parâmetros PK (Fu, Vd, CL) usando modelos ML.

    Retorna fraction unbound, volume of distribution e clearance.
    """
    try:
        logger.info(f"Predição parâmetros PK: SMILES={request.smiles[:50]}...")

        # Converter SMILES para embeddings
        embedding_service = get_embedding_service()
        model_service = get_model_service()

        mol = Chem.MolFromSmiles(request.smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail=f"SMILES inválido: {request.smiles}")

        # Obter embedding ChemBERTa e descritores RDKit
        chemberta_embedding = embedding_service.encode_chemberta(request.smiles)
        rdkit_descriptors = compute_rdkit_descriptors(mol)

        # Predizer parâmetros PK
        pk_params = model_service.predict_pk_parameters(chemberta_embedding, rdkit_descriptors)

        # Calcular half-life: t1/2 = ln(2) * Vd / CL
        # Vd em L/kg, CL em L/h/kg
        vd_per_kg = pk_params["vd"]
        clearance_per_kg = pk_params["clearance"]

        if clearance_per_kg > 0:
            half_life = 0.693 * vd_per_kg / clearance_per_kg  # horas
        else:
            half_life = None

        return PKParametersResponse(
            smiles=request.smiles,
            fu_plasma=pk_params["fu_plasma"],
            vd=pk_params["vd"],
            clearance=pk_params["clearance"],
            half_life=half_life,
            model_type=request.model_type.value
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na predição de parâmetros PK: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

