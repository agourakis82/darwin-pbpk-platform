#!/usr/bin/env python3
"""
Verifica normalização no modelo comparando com ODE solver tradicional
Criado: 2025-11-17
Autor: AI Assistant + Dr. Agourakis
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    DynamicPBPKGNN,
    DynamicPBPKSimulator,
    PBPKPhysiologicalParams,
)
from apps.pbpk_core.simulation.ode_pbpk_solver import ODEPBPKSolver


def compare_gnn_vs_ode(
    dose: float = 100.0,
    clearance_hepatic: float = 10.0,
    clearance_renal: float = 5.0,
    checkpoint_path: Optional[Path] = None,
    output_dir: Path = Path("models/dynamic_gnn_v4_compound/normalization_check"),
) -> Dict:
    """
    Compara previsões do GNN com ODE solver tradicional
    """
    print("🔍 VERIFICAÇÃO DE NORMALIZAÇÃO (GNN vs ODE)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_points = np.linspace(0, 24, 100)

    # 1. ODE Solver (ground truth)
    print("\n1️⃣  Executando ODE solver...")
    from apps.pbpk_core.simulation.ode_pbpk_solver import PBPKPhysiologicalParams as ODEParams
    ode_params = ODEParams(
        clearance_hepatic=clearance_hepatic,
        clearance_renal=clearance_renal,
    )
    ode_solver = ODEPBPKSolver(physiological_params=ode_params)

    ode_result = ode_solver.solve(dose, time_points)
    ode_blood = ode_result['blood']

    print(f"   Cmax (ODE): {ode_blood.max():.4f} mg/L")
    print(f"   AUC (ODE): {np.trapz(ode_blood, time_points):.4f} mg·h/L")

    # 2. GNN Model
    print("\n2️⃣  Executando GNN model...")
    model = DynamicPBPKGNN(
        node_dim=16,
        edge_dim=4,
        hidden_dim=128,
        num_gnn_layers=4,
        num_temporal_steps=len(time_points) - 1,
        dt=float(time_points[1] - time_points[0]),
        use_attention=True,
    )

    simulator = DynamicPBPKSimulator(
        model=model,
        device=device,
        checkpoint_path=str(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else None,
        map_location=device,
        strict=False,
    )

    partition_coeffs = {organ: 1.0 for organ in ode_solver.params.volumes.keys()}
    gnn_result = simulator.simulate(
        dose=dose,
        clearance_hepatic=clearance_hepatic,
        clearance_renal=clearance_renal,
        partition_coeffs=partition_coeffs,
        time_points=time_points,
    )
    gnn_blood = gnn_result['blood']

    print(f"   Cmax (GNN): {gnn_blood.max():.4f} mg/L")
    print(f"   AUC (GNN): {np.trapz(gnn_blood, time_points):.4f} mg·h/L")

    # 3. Comparação
    print("\n3️⃣  Comparação:")
    cmax_ratio = gnn_blood.max() / ode_blood.max()
    auc_ratio = np.trapz(gnn_blood, time_points) / np.trapz(ode_blood, time_points)

    print(f"   Razão Cmax (GNN/ODE): {cmax_ratio:.4f}")
    print(f"   Razão AUC (GNN/ODE): {auc_ratio:.4f}")

    # Verificar concentração inicial
    print("\n4️⃣  Verificação de normalização:")
    blood_volume = ode_solver.params.volumes['blood']
    expected_initial_conc = dose / blood_volume
    ode_initial = ode_blood[0]
    gnn_initial = gnn_blood[0]

    print(f"   Concentração inicial esperada: {expected_initial_conc:.4f} mg/L")
    print(f"   ODE inicial: {ode_initial:.4f} mg/L (razão: {ode_initial/expected_initial_conc:.4f})")
    print(f"   GNN inicial: {gnn_initial:.4f} mg/L (razão: {gnn_initial/expected_initial_conc:.4f})")

    # 5. Visualização
    print("\n5️⃣  Gerando visualização...")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Curvas de concentração
    ax1 = axes[0]
    ax1.plot(time_points, ode_blood, 'b-', label='ODE Solver', linewidth=2)
    ax1.plot(time_points, gnn_blood, 'r--', label='GNN Model', linewidth=2)
    ax1.set_xlabel('Tempo (h)')
    ax1.set_ylabel('Concentração (mg/L)')
    ax1.set_title('Comparação GNN vs ODE Solver')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Diferença relativa
    ax2 = axes[1]
    relative_diff = (gnn_blood - ode_blood) / (ode_blood + 1e-6) * 100
    ax2.plot(time_points, relative_diff, 'g-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Tempo (h)')
    ax2.set_ylabel('Diferença Relativa (%)')
    ax2.set_title('Diferença Relativa: (GNN - ODE) / ODE × 100%')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "normalization_comparison.png", dpi=160)
    plt.close()

    # Salvar resultados
    results = {
        'dose': float(dose),
        'clearance_hepatic': float(clearance_hepatic),
        'clearance_renal': float(clearance_renal),
        'ode_cmax': float(ode_blood.max()),
        'gnn_cmax': float(gnn_blood.max()),
        'cmax_ratio': float(cmax_ratio),
        'ode_auc': float(np.trapz(ode_blood, time_points)),
        'gnn_auc': float(np.trapz(gnn_blood, time_points)),
        'auc_ratio': float(auc_ratio),
        'expected_initial_conc': float(expected_initial_conc),
        'ode_initial': float(ode_initial),
        'gnn_initial': float(gnn_initial),
        'normalization_ok': bool(abs(cmax_ratio - 1.0) < 0.2 and abs(auc_ratio - 1.0) < 0.2),
    }

    import json
    with open(output_dir / "normalization_check.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Resultados salvos em: {output_dir}")
    print(f"   - normalization_comparison.png")
    print(f"   - normalization_check.json")

    if results['normalization_ok']:
        print("\n✅ Normalização OK (diferença < 20%)")
    else:
        print("\n⚠️  Normalização com problemas (diferença > 20%)")

    return results


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Verifica normalização (GNN vs ODE)")
    ap.add_argument("--checkpoint", type=str, help="Caminho para checkpoint do GNN")
    ap.add_argument("--dose", type=float, default=100.0, help="Dose (mg)")
    ap.add_argument("--cl-hepatic", type=float, default=10.0, help="Clearance hepático (L/h)")
    ap.add_argument("--cl-renal", type=float, default=5.0, help="Clearance renal (L/h)")
    ap.add_argument("--output-dir", type=str, default="models/dynamic_gnn_v4_compound/normalization_check", help="Diretório de saída")
    args = ap.parse_args()

    compare_gnn_vs_ode(
        dose=args.dose,
        clearance_hepatic=args.cl_hepatic,
        clearance_renal=args.cl_renal,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    from typing import Dict, Optional
    main()

