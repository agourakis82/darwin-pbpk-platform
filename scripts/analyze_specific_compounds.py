#!/usr/bin/env python3
"""
Análise detalhada por composto específico
- Identifica compostos problemáticos
- Analisa parâmetros (dose, CL, Kp) de cada composto
- Compara com valores esperados da literatura
- Gera relatório por composto

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


def calculate_cmax_auc(concentrations: np.ndarray, time_points: np.ndarray, organ_idx: int = 0) -> tuple:
    """Calcula Cmax e AUC"""
    if len(concentrations.shape) == 2:
        conc = concentrations[organ_idx, :]
    else:
        conc = concentrations
    cmax = float(np.max(conc))
    auc = float(np.trapz(conc, time_points))
    return cmax, auc


def analyze_specific_compounds(
    checkpoint_path: Path,
    experimental_data_path: Path,
    metadata_path: Path,
    output_dir: Path,
    device: str = "cuda",
):
    """Análise detalhada por composto"""
    print("🔍 ANÁLISE DETALHADA POR COMPOSTO")
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

    # Analisar cada composto
    print("\n2️⃣  Analisando compostos...")
    results = []

    with torch.no_grad():
        for i in range(len(doses)):
            meta = metadata[i] if i < len(metadata) else {}
            compound_name = meta.get('drug_name', f'compound_{i}')

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

            # AUC observado
            obs_auc = None
            if 'auc_obs_mg_h_l' in meta and meta['auc_obs_mg_h_l'] is not None:
                try:
                    obs_auc = float(meta['auc_obs_mg_h_l'])
                except (ValueError, TypeError):
                    pass

            if obs_auc is None:
                auc_raw = meta.get('auc_obs', None)
                if auc_raw is not None and not isinstance(auc_raw, dict):
                    try:
                        obs_auc = float(auc_raw) / 1000.0
                    except (ValueError, TypeError):
                        pass

            # Calcular concentração inicial esperada
            v_blood = 5.0  # L
            c0_expected = doses[i] / v_blood
            c0_pred = float(pred_conc[0, 0])

            # Calcular razões
            cmax_ratio = pred_cmax / obs_cmax if obs_cmax is not None and obs_cmax > 0 else np.nan
            auc_ratio = pred_auc / obs_auc if obs_auc is not None and obs_auc > 0 else np.nan
            c0_ratio = c0_pred / c0_expected if c0_expected > 0 else np.nan

            # Kp médio
            kp_mean = float(np.mean(partition_coeffs[i]))
            kp_max = float(np.max(partition_coeffs[i]))
            kp_min = float(np.min(partition_coeffs[i]))

            # CL total
            cl_total = clearances_hepatic[i] + clearances_renal[i]

            results.append({
                'compound': compound_name,
                'dose': float(doses[i]),
                'cl_hepatic': float(clearances_hepatic[i]),
                'cl_renal': float(clearances_renal[i]),
                'cl_total': float(cl_total),
                'kp_mean': kp_mean,
                'kp_min': kp_min,
                'kp_max': kp_max,
                'c0_expected': c0_expected,
                'c0_pred': c0_pred,
                'c0_ratio': c0_ratio,
                'pred_cmax': pred_cmax,
                'obs_cmax': obs_cmax if obs_cmax is not None else np.nan,
                'cmax_ratio': cmax_ratio,
                'pred_auc': pred_auc,
                'obs_auc': obs_auc if obs_auc is not None else np.nan,
                'auc_ratio': auc_ratio,
            })

    df = pd.DataFrame(results)

    # Filtrar compostos com dados observados
    df_with_obs = df[df['obs_cmax'].notna() & (df['obs_cmax'] > 0)].copy()

    print(f"\n   Total de compostos: {len(df)}")
    print(f"   Compostos com Cmax observado: {len(df_with_obs)}")

    # Identificar compostos problemáticos
    print("\n3️⃣  Compostos mais problemáticos (maior razão Cmax):")
    top_problematic = df_with_obs.nlargest(10, 'cmax_ratio')[['compound', 'dose', 'cl_total', 'kp_mean', 'pred_cmax', 'obs_cmax', 'cmax_ratio']]
    print(top_problematic.to_string(index=False))

    # Análise de padrões
    print("\n4️⃣  Análise de padrões:")

    # Por dose
    print("\n   Por faixa de dose:")
    df_with_obs['dose_category'] = pd.cut(df_with_obs['dose'], bins=[0, 10, 50, 100, 200, np.inf], labels=['<10', '10-50', '50-100', '100-200', '>200'])
    dose_stats = df_with_obs.groupby('dose_category').agg({
        'cmax_ratio': ['mean', 'median', 'count'],
        'dose': 'mean',
    }).round(2)
    print(dose_stats)

    # Por CL total
    print("\n   Por faixa de CL total:")
    df_with_obs['cl_category'] = pd.cut(df_with_obs['cl_total'], bins=[0, 5, 10, 20, 50, np.inf], labels=['<5', '5-10', '10-20', '20-50', '>50'])
    cl_stats = df_with_obs.groupby('cl_category').agg({
        'cmax_ratio': ['mean', 'median', 'count'],
        'cl_total': 'mean',
    }).round(2)
    print(cl_stats)

    # Por Kp médio
    print("\n   Por faixa de Kp médio:")
    df_with_obs['kp_category'] = pd.cut(df_with_obs['kp_mean'], bins=[0, 0.5, 1.0, 2.0, 5.0, np.inf], labels=['<0.5', '0.5-1.0', '1.0-2.0', '2.0-5.0', '>5.0'])
    kp_stats = df_with_obs.groupby('kp_category').agg({
        'cmax_ratio': ['mean', 'median', 'count'],
        'kp_mean': 'mean',
    }).round(2)
    print(kp_stats)

    # Visualizações
    print("\n5️⃣  Gerando visualizações...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Cmax ratio vs Dose
    ax = axes[0, 0]
    ax.scatter(df_with_obs['dose'], df_with_obs['cmax_ratio'], alpha=0.6, s=100)
    ax.axhline(1.0, color='g', linestyle='--', label='Ideal', linewidth=2)
    ax.set_xlabel('Dose (mg)')
    ax.set_ylabel('Razão Cmax (Pred/Obs)')
    ax.set_title('Razão Cmax vs Dose')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cmax ratio vs CL total
    ax = axes[0, 1]
    ax.scatter(df_with_obs['cl_total'], df_with_obs['cmax_ratio'], alpha=0.6, s=100)
    ax.axhline(1.0, color='g', linestyle='--', label='Ideal', linewidth=2)
    ax.set_xlabel('CL Total (L/h)')
    ax.set_ylabel('Razão Cmax (Pred/Obs)')
    ax.set_title('Razão Cmax vs CL Total')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cmax ratio vs Kp médio
    ax = axes[0, 2]
    ax.scatter(df_with_obs['kp_mean'], df_with_obs['cmax_ratio'], alpha=0.6, s=100)
    ax.axhline(1.0, color='g', linestyle='--', label='Ideal', linewidth=2)
    ax.set_xlabel('Kp Médio')
    ax.set_ylabel('Razão Cmax (Pred/Obs)')
    ax.set_title('Razão Cmax vs Kp Médio')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # C0 ratio vs Dose
    ax = axes[1, 0]
    ax.scatter(df['dose'], df['c0_ratio'], alpha=0.6, s=100)
    ax.axhline(1.0, color='g', linestyle='--', label='Ideal', linewidth=2)
    ax.set_xlabel('Dose (mg)')
    ax.set_ylabel('Razão C0 (Pred/Esperado)')
    ax.set_title('Normalização C0 vs Dose')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top 10 compostos problemáticos
    ax = axes[1, 1]
    top_10 = df_with_obs.nlargest(10, 'cmax_ratio')
    ax.barh(range(len(top_10)), top_10['cmax_ratio'], alpha=0.7)
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10['compound'], fontsize=8)
    ax.set_xlabel('Razão Cmax (Pred/Obs)')
    ax.set_title('Top 10 Compostos Problemáticos')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')

    # Distribuição de razões
    ax = axes[1, 2]
    ratios = df_with_obs['cmax_ratio'].dropna()
    ax.hist(ratios, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(ratios.median(), color='r', linestyle='--', label=f'Mediana: {ratios.median():.2f}×')
    ax.axvline(1.0, color='g', linestyle='--', label='Ideal: 1.0×')
    ax.set_xlabel('Razão Cmax (Pred/Obs)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição de Razões Cmax')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'compound_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Salvar resultados
    df.to_csv(output_dir / 'compound_analysis.csv', index=False)

    # Resumo
    summary = {
        'n_compounds': len(df),
        'n_with_obs': len(df_with_obs),
        'cmax_ratio_stats': {
            'mean': float(df_with_obs['cmax_ratio'].mean()),
            'median': float(df_with_obs['cmax_ratio'].median()),
            'std': float(df_with_obs['cmax_ratio'].std()),
            'min': float(df_with_obs['cmax_ratio'].min()),
            'max': float(df_with_obs['cmax_ratio'].max()),
        },
        'c0_ratio_stats': {
            'mean': float(df['c0_ratio'].mean()),
            'median': float(df['c0_ratio'].median()),
            'std': float(df['c0_ratio'].std()),
        },
        'top_problematic': top_problematic.to_dict('records'),
    }

    with open(output_dir / 'compound_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Análise por composto concluída!")
    print(f"   Resultados salvos em: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Análise detalhada por composto")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint do modelo")
    parser.add_argument("--experimental-data", required=True, help="Dados experimentais (.npz)")
    parser.add_argument("--experimental-metadata", required=True, help="Metadata experimental (.json)")
    parser.add_argument("--output-dir", required=True, help="Diretório de saída")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze_specific_compounds(
        Path(args.checkpoint),
        Path(args.experimental_data),
        Path(args.experimental_metadata),
        output_dir,
        args.device,
    )


if __name__ == "__main__":
    main()

