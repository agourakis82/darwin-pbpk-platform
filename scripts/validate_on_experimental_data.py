#!/usr/bin/env python3
"""
Valida DynamicPBPKGNN em dados experimentais reais
Criado: 2025-11-17
Autor: AI Assistant + Dr. Agourakis

Compara previsões do modelo com dados experimentais observados (Cmax, AUC).
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
import sys
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    DynamicPBPKGNN,
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
)
from scripts.train_dynamic_gnn_pbpk import create_physiological_params


def fold_error(predicted: float, observed: float) -> float:
    """Calcula Fold Error"""
    if observed == 0:
        return float('inf')
    ratio = predicted / observed
    return max(ratio, 1.0 / ratio)


def geometric_mean_fold_error(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Calcula Geometric Mean Fold Error"""
    mask = observed > 0
    if not np.any(mask):
        return float('nan')
    ratio = predicted[mask] / observed[mask]
    log_ratio = np.log10(np.maximum(ratio, 1e-10))
    return float(10 ** np.mean(np.abs(log_ratio)))


def calculate_cmax_auc(concentrations: np.ndarray, time_points: np.ndarray, organ_idx: int = 0) -> tuple:
    """
    Calcula Cmax e AUC a partir de curvas de concentração

    Args:
        concentrations: [NUM_ORGANS, T] ou [T]
        time_points: [T]
        organ_idx: índice do órgão (0 = blood)

    Returns:
        (cmax, auc)
    """
    if len(concentrations.shape) == 2:
        conc = concentrations[organ_idx, :]  # Blood concentration
    else:
        conc = concentrations

    cmax = np.max(conc)
    cmax_idx = np.argmax(conc)

    # AUC usando trapézio
    auc = np.trapz(conc, time_points)

    return float(cmax), float(auc)


def validate_model_on_experimental(
    model: DynamicPBPKGNN,
    experimental_data_path: Path,
    metadata_path: Path,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """Valida modelo em dados experimentais"""
    # Carregar dados experimentais
    data = np.load(experimental_data_path, allow_pickle=True)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    doses = data['doses']
    clearances_hepatic = data['clearances_hepatic']
    clearances_renal = data['clearances_renal']
    partition_coeffs = data['partition_coeffs']
    compound_ids = data.get('compound_ids', np.array([f'compound_{i}' for i in range(len(doses))]))

    model.eval()
    results = []

    print(f"🔬 Validando modelo em {len(doses)} compostos experimentais...")

    with torch.no_grad():
        for i in range(len(doses)):
            # Parâmetros
            dose = float(doses[i])
            cl_hepatic = float(clearances_hepatic[i])
            cl_renal = float(clearances_renal[i])
            kp = partition_coeffs[i]

            # Criar parâmetros fisiológicos
            partition_dict = {organ: float(kp[j]) for j, organ in enumerate(PBPK_ORGANS)}
            params = PBPKPhysiologicalParams(
                clearance_hepatic=cl_hepatic,
                clearance_renal=cl_renal,
                partition_coeffs=partition_dict,
            )

            # Predizer
            result = model(dose, params)
            pred_conc = result["concentrations"]  # [NUM_ORGANS, T] - tensor
            time_points = result["time_points"]  # [T] - tensor

            # Converter para numpy
            pred_conc_np = pred_conc.cpu().numpy() if isinstance(pred_conc, torch.Tensor) else pred_conc
            time_points_np = time_points.cpu().numpy() if isinstance(time_points, torch.Tensor) else time_points

            # Calcular Cmax e AUC previstos (blood = índice 0)
            pred_cmax, pred_auc = calculate_cmax_auc(pred_conc_np, time_points_np, organ_idx=0)

            # Dados observados (se disponíveis)
            meta = metadata[i] if i < len(metadata) else {}

            # Priorizar valores convertidos (mg/L, mg·h/L) se disponíveis
            obs_cmax = None
            obs_auc = None

            # Cmax: tentar valor convertido primeiro, depois original
            if 'cmax_obs_mg_l' in meta and meta['cmax_obs_mg_l'] is not None:
                try:
                    obs_cmax = float(meta['cmax_obs_mg_l'])
                except (ValueError, TypeError):
                    pass

            if obs_cmax is None:
                obs_cmax_raw = meta.get('cmax_obs', None)
                if obs_cmax_raw is not None:
                    try:
                        obs_cmax_ng_ml = float(obs_cmax_raw)
                        # Converter: ng/mL → mg/L = ng/mL / 1000 (conversão direta de massa)
                        obs_cmax = obs_cmax_ng_ml / 1000.0
                    except (ValueError, TypeError):
                        obs_cmax = None

            # AUC: tentar valor convertido primeiro, depois original
            if 'auc_obs_mg_h_l' in meta and meta['auc_obs_mg_h_l'] is not None:
                try:
                    obs_auc = float(meta['auc_obs_mg_h_l'])
                except (ValueError, TypeError):
                    pass

            if obs_auc is None:
                obs_auc_raw = meta.get('auc_obs', None)
                if obs_auc_raw is not None:
                    try:
                        obs_auc_ng_h_ml = float(obs_auc_raw)
                        # Converter: ng·h/mL → mg·h/L = ng·h/mL / 1000 (conversão direta de massa)
                        obs_auc = obs_auc_ng_h_ml / 1000.0
                    except (ValueError, TypeError):
                        obs_auc = None

            # Compound ID
            if i < len(compound_ids):
                comp_id = str(compound_ids[i])
            else:
                comp_id = meta.get('drug_name', f'compound_{i}')

            result_item = {
                'compound_id': comp_id,
                'drug_name': meta.get('drug_name', comp_id),
                'dose': dose,
                'pred_cmax': pred_cmax,
                'pred_auc': pred_auc,
                'obs_cmax': obs_cmax,
                'obs_auc': obs_auc,
                'cmax_fe': fold_error(pred_cmax, obs_cmax) if obs_cmax is not None and obs_cmax > 0 else None,
                'auc_fe': fold_error(pred_auc, obs_auc) if obs_auc is not None and obs_auc > 0 else None,
            }
            results.append(result_item)

            if (i + 1) % 10 == 0:
                print(f"   Processados {i + 1}/{len(doses)} compostos...")

    # Calcular métricas agregadas
    cmax_fe_list = [r['cmax_fe'] for r in results if r['cmax_fe'] is not None]
    auc_fe_list = [r['auc_fe'] for r in results if r['auc_fe'] is not None]

    metrics = {
        'num_compounds': len(results),
        'num_with_cmax': len(cmax_fe_list),
        'num_with_auc': len(auc_fe_list),
        'cmax_fe_mean': float(np.mean(cmax_fe_list)) if cmax_fe_list else None,
        'cmax_fe_median': float(np.median(cmax_fe_list)) if cmax_fe_list else None,
        'cmax_fe_p67': float(np.percentile(cmax_fe_list, 67)) if cmax_fe_list else None,
        'cmax_gmfe': geometric_mean_fold_error(
            np.array([r['pred_cmax'] for r in results if r['obs_cmax']]),
            np.array([r['obs_cmax'] for r in results if r['obs_cmax']])
        ) if cmax_fe_list else None,
        'auc_fe_mean': float(np.mean(auc_fe_list)) if auc_fe_list else None,
        'auc_fe_median': float(np.median(auc_fe_list)) if auc_fe_list else None,
        'auc_fe_p67': float(np.percentile(auc_fe_list, 67)) if auc_fe_list else None,
        'auc_gmfe': geometric_mean_fold_error(
            np.array([r['pred_auc'] for r in results if r['obs_auc']]),
            np.array([r['obs_auc'] for r in results if r['obs_auc']])
        ) if auc_fe_list else None,
        'cmax_percent_within_2x': float(np.mean(np.array(cmax_fe_list) <= 2.0) * 100) if cmax_fe_list else None,
        'auc_percent_within_2x': float(np.mean(np.array(auc_fe_list) <= 2.0) * 100) if auc_fe_list else None,
    }

    # Salvar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "validation_results.csv", index=False)

    with open(output_dir / "validation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Gerar plots
    if cmax_fe_list:
        plt.figure(figsize=(10, 6))
        plt.hist(cmax_fe_list, bins=30, edgecolor='black')
        plt.axvline(2.0, color='r', linestyle='--', label='FE = 2.0 (aceitável)')
        plt.axvline(1.5, color='orange', linestyle='--', label='FE = 1.5 (excelente)')
        plt.xlabel('Fold Error (Cmax)')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Fold Error - Cmax')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "cmax_fe_distribution.png", dpi=160)
        plt.close()

    if auc_fe_list:
        plt.figure(figsize=(10, 6))
        plt.hist(auc_fe_list, bins=30, edgecolor='black')
        plt.axvline(2.0, color='r', linestyle='--', label='FE = 2.0 (aceitável)')
        plt.axvline(1.5, color='orange', linestyle='--', label='FE = 1.5 (excelente)')
        plt.xlabel('Fold Error (AUC)')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Fold Error - AUC')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "auc_fe_distribution.png", dpi=160)
        plt.close()

    # Scatter plots
    if cmax_fe_list:
        obs_cmax = [r['obs_cmax'] for r in results if r['obs_cmax']]
        pred_cmax = [r['pred_cmax'] for r in results if r['obs_cmax']]
        plt.figure(figsize=(10, 8))
        plt.scatter(obs_cmax, pred_cmax, alpha=0.6)
        min_val = min(min(obs_cmax), min(pred_cmax))
        max_val = max(max(obs_cmax), max(pred_cmax))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        plt.plot([min_val, max_val], [min_val * 2, max_val * 2], 'g--', label='2×')
        plt.plot([min_val, max_val], [min_val / 2, max_val / 2], 'g--')
        plt.xlabel('Cmax Observado')
        plt.ylabel('Cmax Previsto')
        plt.title('Cmax: Previsto vs. Observado')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(output_dir / "cmax_scatter.png", dpi=160)
        plt.close()

    if auc_fe_list:
        obs_auc = [r['obs_auc'] for r in results if r['obs_auc']]
        pred_auc = [r['pred_auc'] for r in results if r['obs_auc']]
        plt.figure(figsize=(10, 8))
        plt.scatter(obs_auc, pred_auc, alpha=0.6)
        min_val = min(min(obs_auc), min(pred_auc))
        max_val = max(max(obs_auc), max(pred_auc))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        plt.plot([min_val, max_val], [min_val * 2, max_val * 2], 'g--', label='2×')
        plt.plot([min_val, max_val], [min_val / 2, max_val / 2], 'g--')
        plt.xlabel('AUC Observado')
        plt.ylabel('AUC Previsto')
        plt.title('AUC: Previsto vs. Observado')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(output_dir / "auc_scatter.png", dpi=160)
        plt.close()

    return metrics


def main():
    ap = argparse.ArgumentParser(description="Valida DynamicPBPKGNN em dados experimentais")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint do modelo")
    ap.add_argument("--experimental-data", required=True, help="Dataset experimental (.npz)")
    ap.add_argument("--metadata", required=True, help="Metadata experimental (.json)")
    ap.add_argument("--output-dir", required=True, help="Diretório de saída")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--node-dim", type=int, default=16)
    ap.add_argument("--edge-dim", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-gnn-layers", type=int, default=3)
    ap.add_argument("--num-temporal-steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.1)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))

    # Carregar modelo
    model = DynamicPBPKGNN(
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_temporal_steps=args.num_temporal_steps,
        dt=args.dt,
    ).to(device)
    state = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(state["model_state_dict"])

    # Validar
    metrics = validate_model_on_experimental(
        model,
        Path(args.experimental_data),
        Path(args.metadata),
        device,
        output_dir,
    )

    print("\n✅ Validação concluída!")
    print(f"\n📊 Métricas de Validação Externa:")
    if metrics['num_with_cmax'] > 0:
        print(f"\n📈 Cmax:")
        print(f"  FE médio: {metrics['cmax_fe_mean']:.3f}")
        print(f"  FE mediano: {metrics['cmax_fe_median']:.3f}")
        print(f"  FE p67: {metrics['cmax_fe_p67']:.3f}")
        print(f"  GMFE: {metrics['cmax_gmfe']:.3f}")
        print(f"  % dentro de 2.0×: {metrics['cmax_percent_within_2x']:.1f}%")
        print(f"  Critério FE ≤ 2.0: {'✅ PASSOU' if metrics['cmax_fe_p67'] <= 2.0 else '❌ FALHOU'}")

    if metrics['num_with_auc'] > 0:
        print(f"\n📈 AUC:")
        print(f"  FE médio: {metrics['auc_fe_mean']:.3f}")
        print(f"  FE mediano: {metrics['auc_fe_median']:.3f}")
        print(f"  FE p67: {metrics['auc_fe_p67']:.3f}")
        print(f"  GMFE: {metrics['auc_gmfe']:.3f}")
        print(f"  % dentro de 2.0×: {metrics['auc_percent_within_2x']:.1f}%")
        print(f"  Critério FE ≤ 2.0: {'✅ PASSOU' if metrics['auc_fe_p67'] <= 2.0 else '❌ FALHOU'}")

    print(f"\n📁 Resultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()

