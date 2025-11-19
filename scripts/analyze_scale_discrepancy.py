#!/usr/bin/env python3
"""
Analisa discrepância de escala entre previsões e observados
Criado: 2025-11-17
Autor: AI Assistant + Dr. Agourakis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_scale_discrepancy(
    validation_results_path: Path,
    metadata_path: Path,
    output_dir: Path,
):
    """Analisa discrepância de escala"""
    df = pd.read_csv(validation_results_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Filtrar apenas com dados observados
    df_valid = df[df['obs_cmax'].notna() & df['obs_auc'].notna()].copy()

    # Calcular razões
    df_valid['cmax_ratio'] = df_valid['pred_cmax'] / df_valid['obs_cmax']
    df_valid['auc_ratio'] = df_valid['pred_auc'] / df_valid['obs_auc']

    # Análise por dose
    print("📊 Análise de Discrepância de Escala")
    print("=" * 70)
    print(f"\nTotal de compostos com dados observados: {len(df_valid)}")
    print(f"\n📈 Cmax:")
    print(f"  Razão média (pred/obs): {df_valid['cmax_ratio'].mean():.2f}×")
    print(f"  Razão mediana: {df_valid['cmax_ratio'].median():.2f}×")
    print(f"  Razão min/max: {df_valid['cmax_ratio'].min():.2f}× / {df_valid['cmax_ratio'].max():.2f}×")

    print(f"\n📈 AUC:")
    print(f"  Razão média (pred/obs): {df_valid['auc_ratio'].mean():.2f}×")
    print(f"  Razão mediana: {df_valid['auc_ratio'].median():.2f}×")
    print(f"  Razão min/max: {df_valid['auc_ratio'].min():.2f}× / {df_valid['auc_ratio'].max():.2f}×")

    # Análise por dose
    print(f"\n📊 Análise por Dose:")
    dose_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    dose_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']
    df_valid['dose_bin'] = pd.cut(df_valid['dose'], bins=dose_bins, labels=dose_labels)

    for bin_label in dose_labels:
        bin_data = df_valid[df_valid['dose_bin'] == bin_label]
        if len(bin_data) > 0:
            print(f"\n  Dose {bin_label} mg ({len(bin_data)} compostos):")
            print(f"    Cmax ratio: {bin_data['cmax_ratio'].mean():.2f}×")
            print(f"    AUC ratio: {bin_data['auc_ratio'].mean():.2f}×")

    # Análise por clearance
    print(f"\n📊 Análise por Clearance:")
    # Adicionar clearances do metadata
    clearances_hepatic = []
    clearances_renal = []
    for i, row in df_valid.iterrows():
        meta_idx = int(i) if i < len(metadata) else -1
        if meta_idx >= 0 and meta_idx < len(metadata):
            meta = metadata[meta_idx]
            cl_lit = meta.get('CL_lit', None)
            if cl_lit:
                clearances_hepatic.append(cl_lit * 0.7)
                clearances_renal.append(cl_lit * 0.3)
            else:
                clearances_hepatic.append(None)
                clearances_renal.append(None)
        else:
            clearances_hepatic.append(None)
            clearances_renal.append(None)

    df_valid['cl_hepatic'] = clearances_hepatic
    df_valid['cl_renal'] = clearances_renal
    df_valid['cl_total'] = df_valid['cl_hepatic'] + df_valid['cl_renal']

    if df_valid['cl_total'].notna().sum() > 0:
        cl_bins = [0, 10, 25, 50, 100, float('inf')]
        cl_labels = ['0-10', '10-25', '25-50', '50-100', '100+']
        df_valid['cl_bin'] = pd.cut(df_valid['cl_total'], bins=cl_bins, labels=cl_labels)

        for bin_label in cl_labels:
            bin_data = df_valid[df_valid['cl_bin'] == bin_label]
            if len(bin_data) > 0:
                print(f"\n  CL {bin_label} L/h ({len(bin_data)} compostos):")
                print(f"    Cmax ratio: {bin_data['cmax_ratio'].mean():.2f}×")
                print(f"    AUC ratio: {bin_data['auc_ratio'].mean():.2f}×")

    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cmax ratio vs dose
    ax1 = axes[0, 0]
    ax1.scatter(df_valid['dose'], df_valid['cmax_ratio'], alpha=0.6)
    ax1.axhline(1.0, color='r', linestyle='--', label='Ideal (1.0×)')
    ax1.axhline(2.0, color='orange', linestyle='--', label='Aceitável (2.0×)')
    ax1.set_xlabel('Dose (mg)')
    ax1.set_ylabel('Razão Cmax (pred/obs)')
    ax1.set_title('Cmax: Razão Previsto/Observado vs. Dose')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # AUC ratio vs dose
    ax2 = axes[0, 1]
    ax2.scatter(df_valid['dose'], df_valid['auc_ratio'], alpha=0.6)
    ax2.axhline(1.0, color='r', linestyle='--', label='Ideal (1.0×)')
    ax2.axhline(2.0, color='orange', linestyle='--', label='Aceitável (2.0×)')
    ax2.set_xlabel('Dose (mg)')
    ax2.set_ylabel('Razão AUC (pred/obs)')
    ax2.set_title('AUC: Razão Previsto/Observado vs. Dose')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Cmax ratio vs clearance
    if df_valid['cl_total'].notna().sum() > 0:
        ax3 = axes[1, 0]
        valid_cl = df_valid[df_valid['cl_total'].notna()]
        ax3.scatter(valid_cl['cl_total'], valid_cl['cmax_ratio'], alpha=0.6)
        ax3.axhline(1.0, color='r', linestyle='--', label='Ideal (1.0×)')
        ax3.axhline(2.0, color='orange', linestyle='--', label='Aceitável (2.0×)')
        ax3.set_xlabel('Clearance Total (L/h)')
        ax3.set_ylabel('Razão Cmax (pred/obs)')
        ax3.set_title('Cmax: Razão vs. Clearance')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # AUC ratio vs clearance
    if df_valid['cl_total'].notna().sum() > 0:
        ax4 = axes[1, 1]
        valid_cl = df_valid[df_valid['cl_total'].notna()]
        ax4.scatter(valid_cl['cl_total'], valid_cl['auc_ratio'], alpha=0.6)
        ax4.axhline(1.0, color='r', linestyle='--', label='Ideal (1.0×)')
        ax4.axhline(2.0, color='orange', linestyle='--', label='Aceitável (2.0×)')
        ax4.set_xlabel('Clearance Total (L/h)')
        ax4.set_ylabel('Razão AUC (pred/obs)')
        ax4.set_title('AUC: Razão vs. Clearance')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "scale_discrepancy_analysis.png", dpi=160)
    plt.close()

    # Salvar análise detalhada
    analysis = {
        'cmax_ratio_mean': float(df_valid['cmax_ratio'].mean()),
        'cmax_ratio_median': float(df_valid['cmax_ratio'].median()),
        'cmax_ratio_std': float(df_valid['cmax_ratio'].std()),
        'auc_ratio_mean': float(df_valid['auc_ratio'].mean()),
        'auc_ratio_median': float(df_valid['auc_ratio'].median()),
        'auc_ratio_std': float(df_valid['auc_ratio'].std()),
        'num_compounds': len(df_valid),
    }

    with open(output_dir / "scale_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    # Salvar CSV com análise
    df_valid.to_csv(output_dir / "validation_results_with_ratios.csv", index=False)

    print(f"\n✅ Análise salva em: {output_dir}")
    print(f"   - scale_discrepancy_analysis.png")
    print(f"   - scale_analysis.json")
    print(f"   - validation_results_with_ratios.csv")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Analisa discrepância de escala")
    ap.add_argument("--validation-results", required=True, help="CSV de resultados de validação")
    ap.add_argument("--metadata", required=True, help="JSON de metadata")
    ap.add_argument("--output-dir", required=True, help="Diretório de saída")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze_scale_discrepancy(
        Path(args.validation_results),
        Path(args.metadata),
        output_dir,
    )


if __name__ == "__main__":
    main()


