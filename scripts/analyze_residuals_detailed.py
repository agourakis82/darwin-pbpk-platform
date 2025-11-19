#!/usr/bin/env python3
"""
Análise detalhada de resíduos
- Resíduos vs predito
- Resíduos vs observado
- Resíduos vs dose, clearance, etc.
- Identifica padrões sistemáticos (heterocedasticidade, viés)
- Testes estatísticos

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
from scipy.stats import shapiro, normaltest

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


def analyze_residuals(
    checkpoint_path: Path,
    experimental_data_path: Path,
    metadata_path: Path,
    output_dir: Path,
    device: str = "cuda",
):
    """Análise detalhada de resíduos"""
    import torch

    print("📊 ANÁLISE DETALHADA DE RESÍDUOS")
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
    print("\n2️⃣  Coletando previsões e calculando resíduos...")
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

            if obs_cmax is not None and obs_cmax > 0:
                residual_cmax = pred_cmax - obs_cmax
                residual_cmax_pct = (residual_cmax / obs_cmax) * 100 if obs_cmax > 0 else np.nan
                residual_cmax_log = np.log10(pred_cmax) - np.log10(obs_cmax) if obs_cmax > 0 and pred_cmax > 0 else np.nan
            else:
                residual_cmax = np.nan
                residual_cmax_pct = np.nan
                residual_cmax_log = np.nan

            if obs_auc is not None and obs_auc > 0:
                residual_auc = pred_auc - obs_auc
                residual_auc_pct = (residual_auc / obs_auc) * 100 if obs_auc > 0 else np.nan
                residual_auc_log = np.log10(pred_auc) - np.log10(obs_auc) if obs_auc > 0 and pred_auc > 0 else np.nan
            else:
                residual_auc = np.nan
                residual_auc_pct = np.nan
                residual_auc_log = np.nan

            results.append({
                'compound_id': meta.get('drug_name', f'compound_{i}'),
                'dose': float(doses[i]),
                'cl_hepatic': float(clearances_hepatic[i]),
                'cl_renal': float(clearances_renal[i]),
                'pred_cmax': pred_cmax if obs_cmax is not None else np.nan,
                'obs_cmax': obs_cmax if obs_cmax is not None else np.nan,
                'residual_cmax': residual_cmax if obs_cmax is not None else np.nan,
                'residual_cmax_pct': residual_cmax_pct if obs_cmax is not None else np.nan,
                'residual_cmax_log': residual_cmax_log if obs_cmax is not None else np.nan,
                'pred_auc': pred_auc if obs_auc is not None else np.nan,
                'obs_auc': obs_auc if obs_auc is not None else np.nan,
                'residual_auc': residual_auc if obs_auc is not None else np.nan,
                'residual_auc_pct': residual_auc_pct if obs_auc is not None else np.nan,
                'residual_auc_log': residual_auc_log if obs_auc is not None else np.nan,
            })

    df = pd.DataFrame(results)

    # Filtrar apenas valores válidos
    df_cmax = df[df['obs_cmax'].notna() & (df['obs_cmax'] > 0)].copy()
    df_auc = df[df['obs_auc'].notna() & (df['obs_auc'] > 0)].copy()

    print(f"\n   Compostos com Cmax: {len(df_cmax)}")
    print(f"   Compostos com AUC: {len(df_auc)}")

    # Estatísticas de resíduos
    print("\n3️⃣  Estatísticas de resíduos (Cmax):")
    print(f"   Resíduo médio: {df_cmax['residual_cmax'].mean():.4f} mg/L")
    print(f"   Resíduo mediano: {df_cmax['residual_cmax'].median():.4f} mg/L")
    print(f"   Desvio padrão: {df_cmax['residual_cmax'].std():.4f} mg/L")
    print(f"   Resíduo % médio: {df_cmax['residual_cmax_pct'].mean():.2f}%")
    print(f"   Resíduo % mediano: {df_cmax['residual_cmax_pct'].median():.2f}%")

    print("\n   Estatísticas de resíduos (AUC):")
    print(f"   Resíduo médio: {df_auc['residual_auc'].mean():.4f} mg·h/L")
    print(f"   Resíduo mediano: {df_auc['residual_auc'].median():.4f} mg·h/L")
    print(f"   Desvio padrão: {df_auc['residual_auc'].std():.4f} mg·h/L")
    print(f"   Resíduo % médio: {df_auc['residual_auc_pct'].mean():.2f}%")
    print(f"   Resíduo % mediano: {df_auc['residual_auc_pct'].median():.2f}%")

    # Testes de normalidade
    print("\n4️⃣  Testes de normalidade dos resíduos:")
    if len(df_cmax) > 3:
        shapiro_cmax = shapiro(df_cmax['residual_cmax'].dropna())
        print(f"   Cmax - Shapiro-Wilk: W={shapiro_cmax.statistic:.4f}, p={shapiro_cmax.pvalue:.4f}")
        if shapiro_cmax.pvalue < 0.05:
            print("      ⚠️  Resíduos NÃO são normais (p < 0.05)")
        else:
            print("      ✅ Resíduos são normais (p >= 0.05)")

    if len(df_auc) > 3:
        shapiro_auc = shapiro(df_auc['residual_auc'].dropna())
        print(f"   AUC - Shapiro-Wilk: W={shapiro_auc.statistic:.4f}, p={shapiro_auc.pvalue:.4f}")
        if shapiro_auc.pvalue < 0.05:
            print("      ⚠️  Resíduos NÃO são normais (p < 0.05)")
        else:
            print("      ✅ Resíduos são normais (p >= 0.05)")

    # Teste de viés (t-test se resíduo médio é diferente de zero)
    print("\n5️⃣  Teste de viés (resíduo médio = 0?):")
    if len(df_cmax) > 1:
        ttest_cmax = stats.ttest_1samp(df_cmax['residual_cmax'].dropna(), 0)
        print(f"   Cmax - t-test: t={ttest_cmax.statistic:.4f}, p={ttest_cmax.pvalue:.4f}")
        if ttest_cmax.pvalue < 0.05:
            print(f"      ⚠️  Viés significativo: resíduo médio = {df_cmax['residual_cmax'].mean():.4f} mg/L")
        else:
            print("      ✅ Sem viés significativo")

    if len(df_auc) > 1:
        ttest_auc = stats.ttest_1samp(df_auc['residual_auc'].dropna(), 0)
        print(f"   AUC - t-test: t={ttest_auc.statistic:.4f}, p={ttest_auc.pvalue:.4f}")
        if ttest_auc.pvalue < 0.05:
            print(f"      ⚠️  Viés significativo: resíduo médio = {df_auc['residual_auc'].mean():.4f} mg·h/L")
        else:
            print("      ✅ Sem viés significativo")

    # Visualizações
    print("\n6️⃣  Gerando visualizações...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    # Cmax: Resíduos vs Predito
    ax = axes[0, 0]
    ax.scatter(df_cmax['pred_cmax'], df_cmax['residual_cmax'], alpha=0.6, s=50)
    ax.axhline(0, color='r', linestyle='--', label='Ideal')
    ax.set_xlabel('Cmax Previsto (mg/L)')
    ax.set_ylabel('Resíduo (Pred - Obs) (mg/L)')
    ax.set_title('Cmax: Resíduos vs Previsto')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cmax: Resíduos vs Observado
    ax = axes[0, 1]
    ax.scatter(df_cmax['obs_cmax'], df_cmax['residual_cmax'], alpha=0.6, s=50)
    ax.axhline(0, color='r', linestyle='--', label='Ideal')
    ax.set_xlabel('Cmax Observado (mg/L)')
    ax.set_ylabel('Resíduo (Pred - Obs) (mg/L)')
    ax.set_title('Cmax: Resíduos vs Observado')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC: Resíduos vs Predito
    ax = axes[1, 0]
    ax.scatter(df_auc['pred_auc'], df_auc['residual_auc'], alpha=0.6, s=50)
    ax.axhline(0, color='r', linestyle='--', label='Ideal')
    ax.set_xlabel('AUC Previsto (mg·h/L)')
    ax.set_ylabel('Resíduo (Pred - Obs) (mg·h/L)')
    ax.set_title('AUC: Resíduos vs Previsto')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC: Resíduos vs Observado
    ax = axes[1, 1]
    ax.scatter(df_auc['obs_auc'], df_auc['residual_auc'], alpha=0.6, s=50)
    ax.axhline(0, color='r', linestyle='--', label='Ideal')
    ax.set_xlabel('AUC Observado (mg·h/L)')
    ax.set_ylabel('Resíduo (Pred - Obs) (mg·h/L)')
    ax.set_title('AUC: Resíduos vs Observado')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Distribuição de resíduos (Cmax)
    ax = axes[2, 0]
    ax.hist(df_cmax['residual_cmax'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='Ideal')
    ax.axvline(df_cmax['residual_cmax'].mean(), color='g', linestyle='--', label=f'Média: {df_cmax["residual_cmax"].mean():.4f}')
    ax.set_xlabel('Resíduo (mg/L)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição de Resíduos (Cmax)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Distribuição de resíduos (AUC)
    ax = axes[2, 1]
    ax.hist(df_auc['residual_auc'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='Ideal')
    ax.axvline(df_auc['residual_auc'].mean(), color='g', linestyle='--', label=f'Média: {df_auc["residual_auc"].mean():.4f}')
    ax.set_xlabel('Resíduo (mg·h/L)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição de Resíduos (AUC)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'residuals_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Salvar resultados
    summary = {
        'cmax': {
            'n': len(df_cmax),
            'residual_mean': float(df_cmax['residual_cmax'].mean()),
            'residual_median': float(df_cmax['residual_cmax'].median()),
            'residual_std': float(df_cmax['residual_cmax'].std()),
            'residual_pct_mean': float(df_cmax['residual_cmax_pct'].mean()),
            'shapiro_w': float(shapiro_cmax.statistic) if len(df_cmax) > 3 else None,
            'shapiro_p': float(shapiro_cmax.pvalue) if len(df_cmax) > 3 else None,
            'ttest_statistic': float(ttest_cmax.statistic) if len(df_cmax) > 1 else None,
            'ttest_p': float(ttest_cmax.pvalue) if len(df_cmax) > 1 else None,
        },
        'auc': {
            'n': len(df_auc),
            'residual_mean': float(df_auc['residual_auc'].mean()),
            'residual_median': float(df_auc['residual_auc'].median()),
            'residual_std': float(df_auc['residual_auc'].std()),
            'residual_pct_mean': float(df_auc['residual_auc_pct'].mean()),
            'shapiro_w': float(shapiro_auc.statistic) if len(df_auc) > 3 else None,
            'shapiro_p': float(shapiro_auc.pvalue) if len(df_auc) > 3 else None,
            'ttest_statistic': float(ttest_auc.statistic) if len(df_auc) > 1 else None,
            'ttest_p': float(ttest_auc.pvalue) if len(df_auc) > 1 else None,
        },
    }

    with open(output_dir / 'residuals_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)

    df.to_csv(output_dir / 'residuals_analysis.csv', index=False)

    print(f"\n✅ Análise de resíduos concluída!")
    print(f"   Resultados salvos em: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Análise detalhada de resíduos")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint do modelo")
    parser.add_argument("--experimental-data", required=True, help="Dados experimentais (.npz)")
    parser.add_argument("--experimental-metadata", required=True, help="Metadata experimental (.json)")
    parser.add_argument("--output-dir", required=True, help="Diretório de saída")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze_residuals(
        Path(args.checkpoint),
        Path(args.experimental_data),
        Path(args.experimental_metadata),
        output_dir,
        args.device,
    )


if __name__ == "__main__":
    main()

