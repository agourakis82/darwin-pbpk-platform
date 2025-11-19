#!/usr/bin/env python3
"""
Investiga problema de escala do Cmax em detalhes
- Compara distribuições de Cmax previsto vs observado
- Verifica unidades e normalização
- Analisa por composto e por faixa de dose
- Identifica padrões sistemáticos

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
from scipy import stats
from scipy.stats import pearsonr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    DynamicPBPKGNN,
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
)


def calculate_cmax_auc(concentrations: np.ndarray, time_points: np.ndarray, organ_idx: int = 0) -> tuple:
    """Calcula Cmax e AUC"""
    if len(concentrations.shape) == 2:
        conc = concentrations[organ_idx, :]
    else:
        conc = concentrations
    cmax = float(np.max(conc))
    auc = float(np.trapz(conc, time_points))
    return cmax, auc


def investigate_cmax_scale(
    checkpoint_path: Path,
    experimental_data_path: Path,
    metadata_path: Path,
    output_dir: Path,
    device: str = "cuda",
):
    """Investiga problema de escala do Cmax"""
    import torch

    print("🔍 INVESTIGAÇÃO DO PROBLEMA DE ESCALA DO CMAX")
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

    # Carregar modelo
    print("\n1️⃣  Carregando modelo...")
    model = DynamicPBPKGNN(
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
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()

    # Coletar dados
    print("\n2️⃣  Coletando previsões...")
    results = []

    with torch.no_grad():
        for i in range(len(doses)):
            partition_dict = {organ: float(partition_coeffs[i, j]) for j, organ in enumerate(PBPK_ORGANS)}
            params = PBPKPhysiologicalParams(
                clearance_hepatic=float(clearances_hepatic[i]),
                clearance_renal=float(clearances_renal[i]),
                partition_coeffs=partition_dict,
            )
            result = model(float(doses[i]), params)
            pred_conc = result["concentrations"].cpu().numpy()
            time_points = result["time_points"].cpu().numpy()

            pred_cmax, pred_auc = calculate_cmax_auc(pred_conc, time_points, organ_idx=0)

            meta = metadata[i] if i < len(metadata) else {}

            # Cmax observado
            obs_cmax = None
            if 'cmax_obs_mg_l' in meta and meta['cmax_obs_mg_l'] is not None:
                try:
                    obs_cmax = float(meta['cmax_obs_mg_l'])
                except (ValueError, TypeError):
                    pass

            if obs_cmax is None:
                cmax_raw = meta.get('cmax_obs', None)
                if cmax_raw is not None and not isinstance(cmax_raw, dict):
                    try:
                        obs_cmax = float(cmax_raw) / 1000.0
                    except (ValueError, TypeError):
                        pass

            if obs_cmax is not None and obs_cmax > 0:
                results.append({
                    'compound_id': meta.get('drug_name', f'compound_{i}'),
                    'dose': float(doses[i]),
                    'cl_hepatic': float(clearances_hepatic[i]),
                    'cl_renal': float(clearances_renal[i]),
                    'pred_cmax': pred_cmax,
                    'obs_cmax': obs_cmax,
                    'ratio': pred_cmax / obs_cmax if obs_cmax > 0 else np.nan,
                    'log10_ratio': np.log10(pred_cmax / obs_cmax) if obs_cmax > 0 and pred_cmax > 0 else np.nan,
                })

    df = pd.DataFrame(results)

    print(f"\n   Total de compostos com Cmax observado: {len(df)}")
    print(f"   Cmax previsto: min={df['pred_cmax'].min():.4f}, max={df['pred_cmax'].max():.4f}, mean={df['pred_cmax'].mean():.4f}, median={df['pred_cmax'].median():.4f} mg/L")
    print(f"   Cmax observado: min={df['obs_cmax'].min():.4f}, max={df['obs_cmax'].max():.4f}, mean={df['obs_cmax'].mean():.4f}, median={df['obs_cmax'].median():.4f} mg/L")
    print(f"   Razão média (pred/obs): {df['ratio'].mean():.2f}×")
    print(f"   Razão mediana (pred/obs): {df['ratio'].median():.2f}×")

    # Análise por faixa de dose
    print("\n3️⃣  Análise por faixa de dose...")
    df['dose_category'] = pd.cut(df['dose'], bins=[0, 50, 100, 200, 500, np.inf], labels=['<50', '50-100', '100-200', '200-500', '>500'])
    dose_analysis = df.groupby('dose_category').agg({
        'ratio': ['mean', 'median', 'std', 'count'],
        'pred_cmax': ['mean', 'median'],
        'obs_cmax': ['mean', 'median'],
    }).round(2)
    print(dose_analysis)

    # Análise por faixa de clearance
    print("\n4️⃣  Análise por faixa de clearance hepático...")
    df['cl_category'] = pd.cut(df['cl_hepatic'], bins=[0, 5, 10, 20, 50, np.inf], labels=['<5', '5-10', '10-20', '20-50', '>50'])
    cl_analysis = df.groupby('cl_category').agg({
        'ratio': ['mean', 'median', 'std', 'count'],
        'pred_cmax': ['mean', 'median'],
        'obs_cmax': ['mean', 'median'],
    }).round(2)
    print(cl_analysis)

    # Análise por faixa de Cmax observado
    print("\n5️⃣  Análise por faixa de Cmax observado...")
    df['cmax_category'] = pd.cut(df['obs_cmax'], bins=[0, 0.01, 0.1, 1.0, 10.0, np.inf], labels=['<0.01', '0.01-0.1', '0.1-1.0', '1.0-10', '>10'])
    cmax_analysis = df.groupby('cmax_category').agg({
        'ratio': ['mean', 'median', 'std', 'count'],
        'pred_cmax': ['mean', 'median'],
        'obs_cmax': ['mean', 'median'],
    }).round(2)
    print(cmax_analysis)

    # Visualizações
    print("\n6️⃣  Gerando visualizações...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Distribuição de Cmax previsto vs observado
    ax = axes[0, 0]
    ax.scatter(df['obs_cmax'], df['pred_cmax'], alpha=0.6, s=50)
    ax.plot([df['obs_cmax'].min(), df['obs_cmax'].max()],
            [df['obs_cmax'].min(), df['obs_cmax'].max()], 'r--', label='Ideal')
    ax.set_xlabel('Cmax Observado (mg/L)')
    ax.set_ylabel('Cmax Previsto (mg/L)')
    ax.set_title('Cmax: Previsto vs Observado')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Distribuição de razões
    ax = axes[0, 1]
    ratios = df['ratio'].dropna()
    ax.hist(ratios, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(ratios.median(), color='r', linestyle='--', label=f'Mediana: {ratios.median():.2f}×')
    ax.axvline(1.0, color='g', linestyle='--', label='Ideal: 1.0×')
    ax.set_xlabel('Razão (Pred/Obs)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição de Razões Cmax')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Razão vs Dose
    ax = axes[0, 2]
    ax.scatter(df['dose'], df['ratio'], alpha=0.6, s=50)
    ax.axhline(1.0, color='g', linestyle='--', label='Ideal: 1.0×')
    ax.set_xlabel('Dose (mg)')
    ax.set_ylabel('Razão (Pred/Obs)')
    ax.set_title('Razão Cmax vs Dose')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Razão vs Clearance Hepático
    ax = axes[1, 0]
    ax.scatter(df['cl_hepatic'], df['ratio'], alpha=0.6, s=50)
    ax.axhline(1.0, color='g', linestyle='--', label='Ideal: 1.0×')
    ax.set_xlabel('Clearance Hepático (L/h)')
    ax.set_ylabel('Razão (Pred/Obs)')
    ax.set_title('Razão Cmax vs CL Hepático')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Razão vs Cmax Observado
    ax = axes[1, 1]
    ax.scatter(df['obs_cmax'], df['ratio'], alpha=0.6, s=50)
    ax.axhline(1.0, color='g', linestyle='--', label='Ideal: 1.0×')
    ax.set_xlabel('Cmax Observado (mg/L)')
    ax.set_ylabel('Razão (Pred/Obs)')
    ax.set_title('Razão vs Cmax Observado')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Boxplot por categoria de dose
    ax = axes[1, 2]
    df_box = df[df['dose_category'].notna()].copy()
    categories = df_box['dose_category'].cat.categories
    data_to_plot = [df_box[df_box['dose_category'] == cat]['ratio'].dropna().values for cat in categories]
    ax.boxplot(data_to_plot, labels=categories)
    ax.axhline(1.0, color='g', linestyle='--', label='Ideal: 1.0×')
    ax.set_ylabel('Razão (Pred/Obs)')
    ax.set_title('Razão Cmax por Faixa de Dose')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'cmax_scale_investigation.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Salvar resultados
    summary = {
        'n_compounds': len(df),
        'pred_cmax_stats': {
            'min': float(df['pred_cmax'].min()),
            'max': float(df['pred_cmax'].max()),
            'mean': float(df['pred_cmax'].mean()),
            'median': float(df['pred_cmax'].median()),
            'std': float(df['pred_cmax'].std()),
        },
        'obs_cmax_stats': {
            'min': float(df['obs_cmax'].min()),
            'max': float(df['obs_cmax'].max()),
            'mean': float(df['obs_cmax'].mean()),
            'median': float(df['obs_cmax'].median()),
            'std': float(df['obs_cmax'].std()),
        },
        'ratio_stats': {
            'mean': float(df['ratio'].mean()),
            'median': float(df['ratio'].median()),
            'std': float(df['ratio'].std()),
            'min': float(df['ratio'].min()),
            'max': float(df['ratio'].max()),
        },
        'dose_analysis': {str(k): {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating)) else str(v2) for k2, v2 in v.items()} for k, v in dose_analysis.to_dict().items()},
        'cl_analysis': {str(k): {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating)) else str(v2) for k2, v2 in v.items()} for k, v in cl_analysis.to_dict().items()},
        'cmax_analysis': {str(k): {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating)) else str(v2) for k2, v2 in v.items()} for k, v in cmax_analysis.to_dict().items()},
    }

    with open(output_dir / 'cmax_scale_investigation.json', 'w') as f:
        json.dump(summary, f, indent=2)

    df.to_csv(output_dir / 'cmax_scale_investigation.csv', index=False)

    print(f"\n✅ Investigação concluída!")
    print(f"   Resultados salvos em: {output_dir}")
    print(f"\n📊 RESUMO:")
    print(f"   Razão média (pred/obs): {df['ratio'].mean():.2f}×")
    print(f"   Razão mediana (pred/obs): {df['ratio'].median():.2f}×")
    print(f"   Cmax previsto médio: {df['pred_cmax'].mean():.4f} mg/L")
    print(f"   Cmax observado médio: {df['obs_cmax'].mean():.4f} mg/L")

    # Identificar compostos problemáticos
    print(f"\n🔍 COMPOSTOS MAIS PROBLEMÁTICOS (maior razão):")
    top_problematic = df.nlargest(5, 'ratio')[['compound_id', 'dose', 'pred_cmax', 'obs_cmax', 'ratio']]
    print(top_problematic.to_string(index=False))

    print(f"\n🔍 COMPOSTOS COM MELHOR AJUSTE (razão mais próxima de 1.0):")
    best_fit = df.loc[(df['ratio'] - 1.0).abs().nsmallest(5).index][['compound_id', 'dose', 'pred_cmax', 'obs_cmax', 'ratio']]
    print(best_fit.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Investiga problema de escala do Cmax")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint do modelo")
    parser.add_argument("--experimental-data", required=True, help="Dados experimentais (.npz)")
    parser.add_argument("--experimental-metadata", required=True, help="Metadata experimental (.json)")
    parser.add_argument("--output-dir", required=True, help="Diretório de saída")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    investigate_cmax_scale(
        Path(args.checkpoint),
        Path(args.experimental_data),
        Path(args.experimental_metadata),
        output_dir,
        args.device,
    )


if __name__ == "__main__":
    main()

