"""
Pydantic models para validação de dados da API

Autor: Dr. Demetrios Chiuratto Agourakis
Criado: 2025-11-08
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Literal
from enum import Enum


class ModelType(str, Enum):
    """Tipos de modelos disponíveis"""
    DYNAMIC_GNN = "dynamic_gnn"
    ODE_SOLVER = "ode_solver"
    MLP_BASELINE = "mlp_baseline"
    GNN_MULTITASK = "gnn_multitask"


class PBPKRequest(BaseModel):
    """Request para predição PBPK"""
    smiles: str = Field(..., description="SMILES string do composto")
    dose: float = Field(..., gt=0, description="Dose administrada (mg)")
    route: Literal["iv", "oral", "im", "sc"] = Field(
        default="iv",
        description="Via de administração"
    )
    time_points: Optional[List[float]] = Field(
        default=None,
        description="Pontos temporais para simulação (horas). Se None, usa padrão [0-24h]"
    )
    model_type: ModelType = Field(
        default=ModelType.DYNAMIC_GNN,
        description="Tipo de modelo a usar"
    )

    @validator('smiles')
    def validate_smiles(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("SMILES não pode ser vazio")
        return v.strip()


class PhysiologicalParams(BaseModel):
    """Parâmetros fisiológicos customizados"""
    volumes: Optional[Dict[str, float]] = Field(
        default=None,
        description="Volumes dos compartimentos (L)"
    )
    blood_flows: Optional[Dict[str, float]] = Field(
        default=None,
        description="Fluxos sanguíneos (L/h)"
    )
    clearance_hepatic: Optional[float] = Field(
        default=None,
        ge=0,
        description="Clearance hepático (L/h)"
    )
    clearance_renal: Optional[float] = Field(
        default=None,
        ge=0,
        description="Clearance renal (L/h)"
    )
    partition_coeffs: Optional[Dict[str, float]] = Field(
        default=None,
        description="Coeficientes de partição (Kp) por órgão"
    )


class PBPKSimulationRequest(BaseModel):
    """Request para simulação PBPK completa"""
    smiles: str = Field(..., description="SMILES string do composto")
    dose: float = Field(..., gt=0, description="Dose administrada (mg)")
    route: Literal["iv", "oral", "im", "sc"] = Field(
        default="iv",
        description="Via de administração"
    )
    time_points: Optional[List[float]] = Field(
        default=None,
        description="Pontos temporais (horas)"
    )
    physiological_params: Optional[PhysiologicalParams] = Field(
        default=None,
        description="Parâmetros fisiológicos customizados"
    )
    model_type: ModelType = Field(
        default=ModelType.DYNAMIC_GNN,
        description="Tipo de modelo"
    )


class PKParametersRequest(BaseModel):
    """Request para predição de parâmetros PK"""
    smiles: str = Field(..., description="SMILES string do composto")
    model_type: ModelType = Field(
        default=ModelType.GNN_MULTITASK,
        description="Tipo de modelo"
    )


class OrganConcentration(BaseModel):
    """Concentração em um órgão específico"""
    organ: str
    concentrations: List[float] = Field(..., description="Concentrações ao longo do tempo")
    time_points: List[float] = Field(..., description="Pontos temporais (horas)")


class PBPKResponse(BaseModel):
    """Response de predição PBPK"""
    smiles: str
    dose: float
    route: str
    model_type: str
    time_points: List[float]
    concentrations: Dict[str, List[float]] = Field(
        ...,
        description="Concentrações por órgão: {organ: [concentrations]}"
    )
    summary: Dict[str, float] = Field(
        ...,
        description="Resumo: Cmax, Tmax, AUC por órgão"
    )


class PKParametersResponse(BaseModel):
    """Response de parâmetros PK"""
    smiles: str
    fu_plasma: float = Field(..., description="Fraction unbound in plasma", ge=0, le=1)
    vd: float = Field(..., description="Volume of distribution (L/kg)", gt=0)
    clearance: float = Field(..., description="Clearance (L/h/kg)", ge=0)
    half_life: Optional[float] = Field(None, description="Half-life (h)", gt=0)
    model_type: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    cuda_available: bool
    device_count: int
    models_loaded: int


class ModelInfo(BaseModel):
    """Informações sobre um modelo"""
    name: str
    type: str
    path: Optional[str]
    loaded: bool
    parameters: Optional[int]
    device: Optional[str]


class ModelsListResponse(BaseModel):
    """Lista de modelos disponíveis"""
    models: List[ModelInfo]
    total: int

