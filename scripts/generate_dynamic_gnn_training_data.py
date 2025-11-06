"""
Gera Dataset de Treinamento para Dynamic GNN PBPK

Usa ODE solver como ground truth para gerar dados de treinamento.

Autor: Dr. Demetrios Chiuratto Agourakis
Data: Novembro 2025
"""

import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import argparse
import sys
from tqdm import tqdm

# Adicionar raiz do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.ode_pbpk_solver import ODEPBPKSolver
from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
    NUM_ORGANS
)


def generate_random_params(
    seed: Optional[int] = None
) -> PBPKPhysiologicalParams:
    """
    Gera parâmetros fisiológicos aleatórios para diversidade.
    
    Args:
        seed: Seed para reprodutibilidade
    
    Returns:
        Parâmetros fisiológicos aleatórios
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Clearance (L/h) - log-normal distribution
    clearance_hepatic = np.random.lognormal(mean=2.0, sigma=1.0)  # ~7-50 L/h
    clearance_renal = np.random.lognormal(mean=1.5, sigma=0.8)   # ~3-20 L/h
    
    # Partition coefficients - variar por órgão
    partition_coeffs = {}
    for organ in PBPK_ORGANS:
        if organ == "blood":
            partition_coeffs[organ] = 1.0
        elif organ == "brain":
            # BBB - baixo Kp
            partition_coeffs[organ] = np.random.lognormal(mean=-0.5, sigma=0.5)  # ~0.3-1.5
        elif organ == "adipose":
            # Tecido adiposo - alto Kp para lipofílicos
            partition_coeffs[organ] = np.random.lognormal(mean=1.0, sigma=0.8)  # ~1-10
        elif organ == "liver":
            # Fígado - moderado a alto
            partition_coeffs[organ] = np.random.lognormal(mean=0.5, sigma=0.6)  # ~0.8-4
        else:
            # Outros órgãos - moderado
            partition_coeffs[organ] = np.random.lognormal(mean=0.0, sigma=0.5)  # ~0.5-2
    
    return PBPKPhysiologicalParams(
        clearance_hepatic=clearance_hepatic,
        clearance_renal=clearance_renal,
        partition_coeffs=partition_coeffs
    )


def generate_training_sample(
    dose_range: Tuple[float, float] = (10.0, 1000.0),
    t_max: float = 24.0,
    num_time_points: int = 100,
    seed: Optional[int] = None
) -> Dict:
    """
    Gera uma amostra de treinamento.
    
    Args:
        dose_range: Range de doses (mg)
        t_max: Tempo máximo (horas)
        num_time_points: Número de pontos temporais
        seed: Seed para reprodutibilidade
    
    Returns:
        Dict com:
        - dose: Dose (mg)
        - params: Parâmetros fisiológicos
        - concentrations: [NUM_ORGANS, num_time_points]
        - time_points: [num_time_points]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Gerar parâmetros aleatórios
    params = generate_random_params()
    
    # Gerar dose aleatória
    dose = np.random.uniform(dose_range[0], dose_range[1])
    
    # Resolver ODE
    solver = ODEPBPKSolver(params)
    time_points = np.linspace(0, t_max, num_time_points)
    results = solver.solve(dose, time_points)
    
    # Organizar concentrações
    concentrations = np.zeros((NUM_ORGANS, num_time_points))
    for i, organ in enumerate(PBPK_ORGANS):
        concentrations[i] = results[organ]
    
    return {
        "dose": float(dose),
        "clearance_hepatic": float(params.clearance_hepatic),
        "clearance_renal": float(params.clearance_renal),
        "partition_coeffs": {k: float(v) for k, v in params.partition_coeffs.items()},
        "concentrations": concentrations,
        "time_points": time_points
    }


def generate_dataset(
    num_samples: int = 1000,
    output_dir: Path = Path("data/dynamic_gnn_training"),
    dose_range: Tuple[float, float] = (10.0, 1000.0),
    t_max: float = 24.0,
    num_time_points: int = 100,
    seed: int = 42
) -> None:
    """
    Gera dataset completo de treinamento.
    
    Args:
        num_samples: Número de amostras
        output_dir: Diretório de saída
        dose_range: Range de doses
        t_max: Tempo máximo
        num_time_points: Número de pontos temporais
        seed: Seed para reprodutibilidade
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GERANDO DATASET DE TREINAMENTO - Dynamic GNN PBPK")
    print("=" * 80)
    print(f"\nConfiguração:")
    print(f"   Número de amostras: {num_samples:,}")
    print(f"   Dose range: {dose_range[0]:.1f} - {dose_range[1]:.1f} mg")
    print(f"   Tempo máximo: {t_max:.1f} horas")
    print(f"   Pontos temporais: {num_time_points}")
    print(f"   Output: {output_dir}")
    print()
    
    # Gerar amostras
    samples = []
    for i in tqdm(range(num_samples), desc="Gerando amostras"):
        sample = generate_training_sample(
            dose_range=dose_range,
            t_max=t_max,
            num_time_points=num_time_points,
            seed=seed + i if seed is not None else None
        )
        samples.append(sample)
    
    # Salvar como numpy arrays (eficiente)
    print("\n💾 Salvando dataset...")
    
    # Extrair arrays
    doses = np.array([s["dose"] for s in samples])
    clearances_hepatic = np.array([s["clearance_hepatic"] for s in samples])
    clearances_renal = np.array([s["clearance_renal"] for s in samples])
    concentrations = np.stack([s["concentrations"] for s in samples])  # [num_samples, NUM_ORGANS, num_time_points]
    time_points = samples[0]["time_points"]  # Mesmo para todos
    
    # Partition coefficients (matriz)
    partition_coeffs = np.zeros((num_samples, NUM_ORGANS))
    for i, sample in enumerate(samples):
        for j, organ in enumerate(PBPK_ORGANS):
            partition_coeffs[i, j] = sample["partition_coeffs"][organ]
    
    # Salvar
    np.savez_compressed(
        output_dir / "training_data.npz",
        doses=doses,
        clearances_hepatic=clearances_hepatic,
        clearances_renal=clearances_renal,
        partition_coeffs=partition_coeffs,
        concentrations=concentrations,
        time_points=time_points
    )
    
    # Salvar metadados
    metadata = {
        "num_samples": num_samples,
        "dose_range": dose_range,
        "t_max": t_max,
        "num_time_points": num_time_points,
        "num_organs": NUM_ORGANS,
        "organs": PBPK_ORGANS,
        "seed": seed
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Estatísticas
    print("\n📊 Estatísticas do dataset:")
    print(f"   Doses: {doses.min():.2f} - {doses.max():.2f} mg (média: {doses.mean():.2f})")
    print(f"   Clearance hepático: {clearances_hepatic.min():.2f} - {clearances_hepatic.max():.2f} L/h")
    print(f"   Clearance renal: {clearances_renal.min():.2f} - {clearances_renal.max():.2f} L/h")
    print(f"   Concentrações shape: {concentrations.shape}")
    print(f"   Tamanho do arquivo: {(output_dir / 'training_data.npz').stat().st_size / (1024**2):.2f} MB")
    
    print("\n✅ Dataset gerado com sucesso!")
    print(f"   Arquivo: {output_dir / 'training_data.npz'}")
    print(f"   Metadados: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset de treinamento para Dynamic GNN PBPK")
    parser.add_argument("--num-samples", type=int, default=1000, help="Número de amostras")
    parser.add_argument("--output-dir", type=str, default="data/dynamic_gnn_training", help="Diretório de saída")
    parser.add_argument("--dose-min", type=float, default=10.0, help="Dose mínima (mg)")
    parser.add_argument("--dose-max", type=float, default=1000.0, help="Dose máxima (mg)")
    parser.add_argument("--t-max", type=float, default=24.0, help="Tempo máximo (horas)")
    parser.add_argument("--num-time-points", type=int, default=100, help="Número de pontos temporais")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    
    args = parser.parse_args()
    
    generate_dataset(
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
        dose_range=(args.dose_min, args.dose_max),
        t_max=args.t_max,
        num_time_points=args.num_time_points,
        seed=args.seed
    )

