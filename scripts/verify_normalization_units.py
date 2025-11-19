#!/usr/bin/env python3
"""
Verifica normalização e unidades do modelo
- Compara com ODE solver (ground truth)
- Verifica volume de distribuição
- Verifica conversões de unidades
- Analisa normalização por dose

Criado: 2025-11-18
Autor: AI Assistant + Dr. Agourakis
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    DynamicPBPKGNN,
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
)
from apps.pbpk_core.simulation.ode_pbpk_solver import ODEPBPKSolver


def calculate_cmax_auc(concentrations: np.ndarray, time_points: np.ndarray, organ_idx: int = 0) -> tuple:
    """Calcula Cmax e AUC"""
    if len(concentrations.shape) == 2:
        conc = concentrations[organ_idx, :]
    else:
        conc = concentrations
    cmax = float(np.max(conc))
    auc = float(np.trapz(conc, time_points))
    return cmax, auc


def verify_normalization_units(
    checkpoint_path: Path,
    experimental_data_path: Path,
    metadata_path: Path,
    output_dir: Path,
    device: str = "cuda",
):
    """Verifica normalização e unidades"""
    print("🔍 VERIFICAÇÃO DE NORMALIZAÇÃO E UNIDADES")
    print("=" * 70)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Carregar dados
    data = np.load(experimental_data_path, allow_pickle=True)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    doses = data['doses']
    clearances_hepatic = data['clearances_hepatic']
    clearances_renal = data['clearances_renal']
    partition_coeffs = data['partition_coeffs']

    # Carregar modelo GNN
    print("\n1️⃣  Carregando modelo GNN...")
    model_gnn = DynamicPBPKGNN(
        node_dim=16,
        edge_dim=4,
        hidden_dim=128,
        num_gnn_layers=4,
        num_temporal_steps=120,
        dt=0.1,
        use_attention=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_gnn.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model_gnn.load_state_dict(checkpoint, strict=False)
    model_gnn = model_gnn.to(device)
    model_gnn.eval()

    # Criar ODE solver
    print("\n2️⃣  Criando ODE solver...")
    ode_params = PBPKPhysiologicalParams()  # Usar parâmetros padrão
    ode_solver = ODEPBPKSolver(physiological_params=ode_params)

    # Comparar previsões
    print("\n3️⃣  Comparando previsões GNN vs ODE...")
    results = []

    # Testar com algumas doses diferentes
    test_doses = [10.0, 50.0, 100.0, 200.0, 500.0]

    for test_dose in test_doses:
        # Usar parâmetros médios do dataset
        avg_cl_hepatic = float(np.mean(clearances_hepatic))
        avg_cl_renal = float(np.mean(clearances_renal))
        avg_kp = partition_coeffs.mean(axis=0)

        partition_dict = {organ: float(avg_kp[i]) for i, organ in enumerate(PBPK_ORGANS)}
        params = PBPKPhysiologicalParams(
            clearance_hepatic=avg_cl_hepatic,
            clearance_renal=avg_cl_renal,
            partition_coeffs=partition_dict,
        )

        # GNN
        with torch.no_grad():
            result_gnn = model_gnn(test_dose, params)
            conc_gnn = result_gnn["concentrations"].cpu().numpy()
            time_points = result_gnn["time_points"].cpu().numpy()

        cmax_gnn, auc_gnn = calculate_cmax_auc(conc_gnn, time_points, organ_idx=0)

        # ODE - criar solver com parâmetros específicos
        ode_params_specific = PBPKPhysiologicalParams(
            clearance_hepatic=avg_cl_hepatic,
            clearance_renal=avg_cl_renal,
            partition_coeffs=partition_dict,
        )
        ode_solver_specific = ODEPBPKSolver(physiological_params=ode_params_specific)

        # Simular
        result_ode = ode_solver_specific.solve(
            dose=test_dose,
            time_points=time_points,
        )
        conc_ode = np.array([result_ode.get(organ, np.zeros_like(time_points)) for organ in PBPK_ORGANS])
        cmax_ode, auc_ode = calculate_cmax_auc(conc_ode, time_points, organ_idx=0)

        # Concentração inicial esperada
        # C0 = Dose / V_blood (assumindo V_blood ≈ 5 L)
        v_blood = 5.0  # L
        c0_expected = test_dose / v_blood  # mg/L

        # Concentração inicial real (t=0)
        c0_gnn = float(conc_gnn[0, 0])  # blood = índice 0
        c0_ode = float(conc_ode[0, 0])

        results.append({
            'dose': test_dose,
            'c0_expected': c0_expected,
            'c0_gnn': c0_gnn,
            'c0_ode': c0_ode,
            'c0_ratio_gnn': c0_gnn / c0_expected if c0_expected > 0 else np.nan,
            'c0_ratio_ode': c0_ode / c0_expected if c0_expected > 0 else np.nan,
            'cmax_gnn': cmax_gnn,
            'cmax_ode': cmax_ode,
            'cmax_ratio': cmax_gnn / cmax_ode if cmax_ode > 0 else np.nan,
            'auc_gnn': auc_gnn,
            'auc_ode': auc_ode,
            'auc_ratio': auc_gnn / auc_ode if auc_ode > 0 else np.nan,
        })

    df = pd.DataFrame(results)

    print("\n📊 RESULTADOS DA COMPARAÇÃO:")
    print("=" * 70)
    print(f"{'Dose':<8} {'C0 Exp':<10} {'C0 GNN':<10} {'C0 ODE':<10} {'C0 Ratio GNN':<15} {'C0 Ratio ODE':<15}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['dose']:<8.1f} {row['c0_expected']:<10.2f} {row['c0_gnn']:<10.2f} {row['c0_ode']:<10.2f} {row['c0_ratio_gnn']:<15.2f} {row['c0_ratio_ode']:<15.2f}")

    print(f"\n{'Dose':<8} {'Cmax GNN':<12} {'Cmax ODE':<12} {'Cmax Ratio':<15} {'AUC GNN':<12} {'AUC ODE':<12} {'AUC Ratio':<15}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['dose']:<8.1f} {row['cmax_gnn']:<12.2f} {row['cmax_ode']:<12.2f} {row['cmax_ratio']:<15.2f} {row['auc_gnn']:<12.2f} {row['auc_ode']:<12.2f} {row['auc_ratio']:<15.2f}")

    # Análise de normalização
    print("\n4️⃣  Análise de normalização:")
    print(f"   C0 Ratio GNN médio: {df['c0_ratio_gnn'].mean():.2f} (deveria ser ~1.0)")
    print(f"   C0 Ratio ODE médio: {df['c0_ratio_ode'].mean():.2f} (deveria ser ~1.0)")
    print(f"   Cmax Ratio médio: {df['cmax_ratio'].mean():.2f} (deveria ser ~1.0)")
    print(f"   AUC Ratio médio: {df['auc_ratio'].mean():.2f} (deveria ser ~1.0)")

    # Verificar se há problema de escala por dose
    print("\n5️⃣  Verificando escala por dose:")
    for _, row in df.iterrows():
        if row['c0_ratio_gnn'] != 1.0:
            print(f"   ⚠️  Dose {row['dose']:.1f} mg: C0 ratio = {row['c0_ratio_gnn']:.2f} (deveria ser 1.0)")

    # Visualizações
    print("\n6️⃣  Gerando visualizações...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # C0 vs Dose
    ax = axes[0, 0]
    ax.scatter(df['dose'], df['c0_gnn'], label='GNN', alpha=0.7, s=100)
    ax.scatter(df['dose'], df['c0_ode'], label='ODE', alpha=0.7, s=100, marker='x')
    ax.plot(df['dose'], df['c0_expected'], 'r--', label='Esperado (Dose/5L)', linewidth=2)
    ax.set_xlabel('Dose (mg)')
    ax.set_ylabel('Concentração Inicial (mg/L)')
    ax.set_title('Concentração Inicial vs Dose')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cmax vs Dose
    ax = axes[0, 1]
    ax.scatter(df['dose'], df['cmax_gnn'], label='GNN', alpha=0.7, s=100)
    ax.scatter(df['dose'], df['cmax_ode'], label='ODE', alpha=0.7, s=100, marker='x')
    ax.set_xlabel('Dose (mg)')
    ax.set_ylabel('Cmax (mg/L)')
    ax.set_title('Cmax vs Dose')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC vs Dose
    ax = axes[1, 0]
    ax.scatter(df['dose'], df['auc_gnn'], label='GNN', alpha=0.7, s=100)
    ax.scatter(df['dose'], df['auc_ode'], label='ODE', alpha=0.7, s=100, marker='x')
    ax.set_xlabel('Dose (mg)')
    ax.set_ylabel('AUC (mg·h/L)')
    ax.set_title('AUC vs Dose')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ratios
    ax = axes[1, 1]
    ax.plot(df['dose'], df['c0_ratio_gnn'], 'o-', label='C0 Ratio (GNN)', linewidth=2)
    ax.plot(df['dose'], df['cmax_ratio'], 's-', label='Cmax Ratio', linewidth=2)
    ax.plot(df['dose'], df['auc_ratio'], '^-', label='AUC Ratio', linewidth=2)
    ax.axhline(1.0, color='r', linestyle='--', label='Ideal', linewidth=2)
    ax.set_xlabel('Dose (mg)')
    ax.set_ylabel('Ratio (GNN/ODE)')
    ax.set_title('Ratios GNN/ODE vs Dose')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'normalization_units_verification.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Salvar resultados
    summary = {
        'c0_ratio_gnn_mean': float(df['c0_ratio_gnn'].mean()),
        'c0_ratio_ode_mean': float(df['c0_ratio_ode'].mean()),
        'cmax_ratio_mean': float(df['cmax_ratio'].mean()),
        'auc_ratio_mean': float(df['auc_ratio'].mean()),
        'normalization_ok': bool(abs(df['c0_ratio_gnn'].mean() - 1.0) < 0.1),
        'results': df.to_dict('records'),
    }

    with open(output_dir / 'normalization_units_verification.json', 'w') as f:
        json.dump(summary, f, indent=2)

    df.to_csv(output_dir / 'normalization_units_verification.csv', index=False)

    print(f"\n✅ Verificação concluída!")
    print(f"   Resultados salvos em: {output_dir}")

    if not summary['normalization_ok']:
        print(f"\n⚠️  PROBLEMA DE NORMALIZAÇÃO DETECTADO!")
        print(f"   C0 Ratio GNN médio = {summary['c0_ratio_gnn_mean']:.2f} (deveria ser ~1.0)")
        print(f"   Isso pode explicar o problema de escala do Cmax!")


def main():
    parser = argparse.ArgumentParser(description="Verifica normalização e unidades")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint do modelo")
    parser.add_argument("--experimental-data", required=True, help="Dados experimentais (.npz)")
    parser.add_argument("--experimental-metadata", required=True, help="Metadata experimental (.json)")
    parser.add_argument("--output-dir", required=True, help="Diretório de saída")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verify_normalization_units(
        Path(args.checkpoint),
        Path(args.experimental_data),
        Path(args.experimental_metadata),
        output_dir,
        args.device,
    )


if __name__ == "__main__":
    main()

