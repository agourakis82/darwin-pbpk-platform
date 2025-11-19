"""
Darwin PBPK Platform - REST API

API REST para predições PBPK usando Dynamic GNN e modelos tradicionais.

Endpoints:
- POST /api/v1/predict/pbpk - Predição PBPK completa
- POST /api/v1/predict/parameters - Predição de parâmetros PK (Fu, Vd, CL)
- POST /api/v1/simulate/dynamic-gnn - Simulação usando Dynamic GNN
- POST /api/v1/simulate/ode - Simulação usando ODE solver
- GET /api/v1/models - Lista modelos disponíveis
- GET /api/v1/health - Health check
- GET /api/v1/docs - Documentação Swagger

Autor: Dr. Demetrios Chiuratto Agourakis
Criado: 2025-11-08
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, List
import sys

# Adicionar path do projeto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.api.routers import pbpk, simulation, models
from apps.api.models import HealthResponse
from apps.api.services import get_model_service
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model cache
_model_cache: Dict[str, torch.nn.Module] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager para carregar modelos na inicialização"""
    logger.info("🚀 Iniciando Darwin PBPK Platform API...")

    # Verificar CUDA
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA disponível: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("⚠️  CUDA não disponível, usando CPU")

    # Carregar modelos (lazy loading será feito nos routers)
    logger.info("📦 Modelos serão carregados sob demanda")

    # Tentar carregar modelo FlexiblePK se existir
    project_root = Path(__file__).parent.parent.parent
    model_paths = [
        project_root / "models" / "expanded" / "best_model_expanded.pt",
        project_root / "models" / "trial84" / "best_model.pt",
    ]

    model_service = get_model_service()
    for model_path in model_paths:
        if model_path.exists():
            logger.info(f"📦 Carregando modelo: {model_path.name}")
            if model_service.load_model("flexible_pk", model_path):
                logger.info("✅ Modelo carregado com sucesso!")
                break

    yield

    # Cleanup
    logger.info("🛑 Encerrando API...")
    _model_cache.clear()


# Criar aplicação FastAPI
app = FastAPI(
    title="Darwin PBPK Platform API",
    description="AI-Powered Pharmacokinetic Prediction Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar origens
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers
app.include_router(pbpk.router, prefix="/api/v1", tags=["PBPK Predictions"])
app.include_router(simulation.router, prefix="/api/v1", tags=["Simulations"])
app.include_router(models.router, prefix="/api/v1", tags=["Models"])


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "name": "Darwin PBPK Platform API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/v1/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    return HealthResponse(
        status="healthy",
        cuda_available=cuda_available,
        device_count=device_count,
        models_loaded=len(_model_cache)
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Erro não tratado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

