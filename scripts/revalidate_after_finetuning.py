#!/usr/bin/env python3
"""
Revalida modelo após fine-tuning e aplica fator de calibração
Compara: modelo original, fine-tuned, e fine-tuned + calibrado

Criado: 2025-11-17
Autor: AI Assistant + Dr. Agourakis
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    DynamicPBPKGNN,
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
)


def fold_error(predicted: float, observed: float) -> float:
    """Calcula Fold Error"""
    if observed == 0 or predicted <= 0:
        return float('inf')
    ratio = predicted / observed
    return max(ratio, 1.0 / ratio)


def geometric_mean_fold_error(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Calcula Geometric Mean Fold Error"""
    mask = (observed > 0) & (predicted > 0)
    if not np.any(mask):
        return float('nan')
    ratio = predicted[mask] / observed[mask]
    log_ratio = np.log10(np.maximum(ratio, 1e-10))
    return float(10 ** np.mean(np.abs(log_ratio)))


def percent_within_fold(predicted: np.ndarray, observed: np.ndarray, fold: float) -> float:
    """Percentual de previsões dentro de X-fold"""
    mask = (observed > 0) & (predicted > 0)
    if not np.any(mask):
        return 0.0
    fe = np.array([fold_error(p, o) for p, o in zip(predicted[mask], observed[mask])])
    return float(100.0 * np.sum(fe <= fold) / len(fe))


def calculate_cmax_auc(concentrations: np.ndarray, time_points: np.ndarray, organ_idx: int = 0) -> Tuple[float, float]:
    """Calcula Cmax e AUC"""
    if len(concentrations.shape) == 2:
        conc = concentrations[organ_idx, :]
    else:
        conc = concentrations
    cmax = float(np.max(conc))
    auc = float(np.trapz(conc, time_points))
    return cmax, auc


def validate_model(
    model: DynamicPBPKGNN,
    experimental_data_path: Path,
    metadata_path: Path,
    device: torch.device,
    scale_factor: Optional[float] = None,
) -> Dict:
    """Valida modelo em dados experimentais"""
    data = np.load(experimental_data_path, allow_pickle=True)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    doses = data['doses']
    clearances_hepatic = data['clearances_hepatic']
    clearances_renal = data['clearances_renal']
    partition_coeffs = data['partition_coeffs']

    model.eval()
    pred_cmax_list = []
    pred_auc_list = []
    obs_cmax_list = []
    obs_auc_list = []

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

            # Aplicar fator de calibração se fornecido
            if scale_factor is not None:
                pred_cmax *= scale_factor
                pred_auc *= scale_factor

            meta = metadata[i] if i < len(metadata) else {}
            obs_cmax = None
            obs_auc = None

            if 'cmax_obs_mg_l' in meta and meta['cmax_obs_mg_l'] is not None:
                try:
                    obs_cmax = float(meta['cmax_obs_mg_l'])
                except (ValueError, TypeError):
                    pass

            if obs_cmax is None:
                obs_cmax_raw = meta.get('cmax_obs', None)
                if obs_cmax_raw is not None and not isinstance(obs_cmax_raw, dict):
                    try:
                        obs_cmax_ng_ml = float(obs_cmax_raw)
                        obs_cmax = obs_cmax_ng_ml / 1000.0
                    except (ValueError, TypeError):
                        pass

            if 'auc_obs_mg_h_l' in meta and meta['auc_obs_mg_h_l'] is not None:
                try:
                    obs_auc = float(meta['auc_obs_mg_h_l'])
                except (ValueError, TypeError):
                    pass

            if obs_auc is None:
                obs_auc_raw = meta.get('auc_obs', None)
                if obs_auc_raw is not None and not isinstance(obs_auc_raw, dict):
                    try:
                        obs_auc_ng_h_ml = float(obs_auc_raw)
                        obs_auc = obs_auc_ng_h_ml / 1000.0
                    except (ValueError, TypeError):
                        pass

            if obs_cmax is not None and obs_cmax > 0:
                pred_cmax_list.append(pred_cmax)
                obs_cmax_list.append(obs_cmax)

            if obs_auc is not None and obs_auc > 0:
                pred_auc_list.append(pred_auc)
                obs_auc_list.append(obs_auc)

    pred_cmax_arr = np.array(pred_cmax_list)
    pred_auc_arr = np.array(pred_auc_list)
    obs_cmax_arr = np.array(obs_cmax_list)
    obs_auc_arr = np.array(obs_auc_list)

    # Métricas Cmax
    cmax_fe = np.array([fold_error(p, o) for p, o in zip(pred_cmax_arr, obs_cmax_arr)])
    cmax_fe = cmax_fe[np.isfinite(cmax_fe)]
    cmax_gmfe = geometric_mean_fold_error(pred_cmax_arr, obs_cmax_arr)
    cmax_pct_1_25 = percent_within_fold(pred_cmax_arr, obs_cmax_arr, 1.25)
    cmax_pct_1_5 = percent_within_fold(pred_cmax_arr, obs_cmax_arr, 1.5)
    cmax_pct_2_0 = percent_within_fold(pred_cmax_arr, obs_cmax_arr, 2.0)

    # Métricas AUC
    auc_fe = np.array([fold_error(p, o) for p, o in zip(pred_auc_arr, obs_auc_arr)])
    auc_fe = auc_fe[np.isfinite(auc_fe)]
    auc_gmfe = geometric_mean_fold_error(pred_auc_arr, obs_auc_arr)
    auc_pct_1_25 = percent_within_fold(pred_auc_arr, obs_auc_arr, 1.25)
    auc_pct_1_5 = percent_within_fold(pred_auc_arr, obs_auc_arr, 1.5)
    auc_pct_2_0 = percent_within_fold(pred_auc_arr, obs_auc_arr, 2.0)

    # Correlação
    cmax_r, cmax_p = pearsonr(pred_cmax_arr, obs_cmax_arr) if len(pred_cmax_arr) > 1 else (0.0, 1.0)
    auc_r, auc_p = pearsonr(pred_auc_arr, obs_auc_arr) if len(pred_auc_arr) > 1 else (0.0, 1.0)

    return {
        'cmax': {
            'n': len(pred_cmax_arr),
            'fe_mean': float(np.mean(cmax_fe)) if len(cmax_fe) > 0 else float('nan'),
            'fe_median': float(np.median(cmax_fe)) if len(cmax_fe) > 0 else float('nan'),
            'gmfe': cmax_gmfe,
            'pct_1.25x': cmax_pct_1_25,
            'pct_1.5x': cmax_pct_1_5,
            'pct_2.0x': cmax_pct_2_0,
            'r': float(cmax_r),
            'r2': float(cmax_r ** 2),
        },
        'auc': {
            'n': len(pred_auc_arr),
            'fe_mean': float(np.mean(auc_fe)) if len(auc_fe) > 0 else float('nan'),
            'fe_median': float(np.median(auc_fe)) if len(auc_fe) > 0 else float('nan'),
            'gmfe': auc_gmfe,
            'pct_1.25x': auc_pct_1_25,
            'pct_1.5x': auc_pct_1_5,
            'pct_2.0x': auc_pct_2_0,
            'r': float(auc_r),
            'r2': float(auc_r ** 2),
        },
        'predictions': {
            'cmax_pred': pred_cmax_arr.tolist(),
            'cmax_obs': obs_cmax_arr.tolist(),
            'auc_pred': pred_auc_arr.tolist(),
            'auc_obs': obs_auc_arr.tolist(),
        }
    }


def plot_comparison(metrics_dict: Dict[str, Dict], output_dir: Path):
    """Plota comparação entre modelos"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Cmax scatter
    ax = axes[0, 0]
    for name, metrics in metrics_dict.items():
        pred = np.array(metrics['predictions']['cmax_pred'])
        obs = np.array(metrics['predictions']['cmax_obs'])
        ax.scatter(obs, pred, alpha=0.6, label=name, s=30)
    ax.plot([0, obs.max()], [0, obs.max()], 'k--', label='Ideal', linewidth=1)
    ax.set_xlabel('Cmax Observado (mg/L)')
    ax.set_ylabel('Cmax Previsto (mg/L)')
    ax.set_title('Cmax: Previsto vs Observado')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC scatter
    ax = axes[0, 1]
    for name, metrics in metrics_dict.items():
        pred = np.array(metrics['predictions']['auc_pred'])
        obs = np.array(metrics['predictions']['auc_obs'])
        ax.scatter(obs, pred, alpha=0.6, label=name, s=30)
    ax.plot([0, obs.max()], [0, obs.max()], 'k--', label='Ideal', linewidth=1)
    ax.set_xlabel('AUC Observado (mg·h/L)')
    ax.set_ylabel('AUC Previsto (mg·h/L)')
    ax.set_title('AUC: Previsto vs Observado')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cmax FE distribution
    ax = axes[1, 0]
    for name, metrics in metrics_dict.items():
        fe_mean = metrics['cmax']['fe_mean']
        if not np.isnan(fe_mean):
            ax.bar(name, fe_mean, alpha=0.7, label=name)
    ax.set_ylabel('Fold Error (média)')
    ax.set_title('Cmax: Fold Error Médio')
    ax.grid(True, alpha=0.3, axis='y')

    # AUC FE distribution
    ax = axes[1, 1]
    for name, metrics in metrics_dict.items():
        fe_mean = metrics['auc']['fe_mean']
        if not np.isnan(fe_mean):
            ax.bar(name, fe_mean, alpha=0.7, label=name)
    ax.set_ylabel('Fold Error (média)')
    ax.set_title('AUC: Fold Error Médio')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Revalida modelo após fine-tuning")
    parser.add_argument("--original-checkpoint", required=True, help="Checkpoint do modelo original")
    parser.add_argument("--finetuned-checkpoint", required=True, help="Checkpoint do modelo fine-tuned")
    parser.add_argument("--calibration-results", required=True, help="JSON com fator de calibração")
    parser.add_argument("--experimental-data", required=True, help="Dados experimentais (.npz)")
    parser.add_argument("--experimental-metadata", required=True, help="Metadata experimental (.json)")
    parser.add_argument("--output-dir", required=True, help="Diretório de saída")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Carregar fator de calibração
    with open(args.calibration_results, 'r') as f:
        calib_data = json.load(f)
    scale_factor = calib_data.get('optimal_scale_factor', 1.0)

    print("🔬 REVALIDAÇÃO APÓS FINE-TUNING")
    print("=" * 70)

    # Criar modelo
    model_config = {
        'node_dim': 16,
        'edge_dim': 4,
        'hidden_dim': 128,
        'num_gnn_layers': 4,
        'num_temporal_steps': 120,
        'dt': 0.1,
        'use_attention': True,
    }

    # 1. Modelo original
    print("\n1️⃣  Validando modelo original...")
    model_original = DynamicPBPKGNN(**model_config)
    checkpoint_orig = torch.load(args.original_checkpoint, map_location=device)
    if isinstance(checkpoint_orig, dict) and 'model_state_dict' in checkpoint_orig:
        model_original.load_state_dict(checkpoint_orig['model_state_dict'], strict=False)
    else:
        model_original.load_state_dict(checkpoint_orig, strict=False)
    model_original = model_original.to(device)
    metrics_original = validate_model(
        model_original, Path(args.experimental_data), Path(args.experimental_metadata), device
    )

    # 2. Modelo fine-tuned
    print("\n2️⃣  Validando modelo fine-tuned...")
    model_finetuned = DynamicPBPKGNN(**model_config)
    checkpoint_ft = torch.load(args.finetuned_checkpoint, map_location=device)
    if isinstance(checkpoint_ft, dict) and 'model_state_dict' in checkpoint_ft:
        model_finetuned.load_state_dict(checkpoint_ft['model_state_dict'], strict=False)
    else:
        model_finetuned.load_state_dict(checkpoint_ft, strict=False)
    model_finetuned = model_finetuned.to(device)
    metrics_finetuned = validate_model(
        model_finetuned, Path(args.experimental_data), Path(args.experimental_metadata), device
    )

    # 3. Modelo fine-tuned + calibrado
    print("\n3️⃣  Validando modelo fine-tuned + calibrado...")
    metrics_finetuned_calib = validate_model(
        model_finetuned, Path(args.experimental_data), Path(args.experimental_metadata), device, scale_factor=scale_factor
    )

    # Comparação
    print("\n📊 COMPARAÇÃO DE RESULTADOS:")
    print("=" * 70)

    metrics_dict = {
        'Original': metrics_original,
        'Fine-tuned': metrics_finetuned,
        'Fine-tuned + Calibrado': metrics_finetuned_calib,
    }

    # Tabela Cmax
    print("\n📈 Cmax:")
    print(f"{'Modelo':<25} {'FE médio':<12} {'GMFE':<10} {'% 2.0x':<10} {'R²':<10}")
    print("-" * 70)
    for name, metrics in metrics_dict.items():
        cmax = metrics['cmax']
        print(f"{name:<25} {cmax['fe_mean']:<12.2f} {cmax['gmfe']:<10.2f} {cmax['pct_2.0x']:<10.1f} {cmax['r2']:<10.3f}")

    # Tabela AUC
    print("\n📈 AUC:")
    print(f"{'Modelo':<25} {'FE médio':<12} {'GMFE':<10} {'% 2.0x':<10} {'R²':<10}")
    print("-" * 70)
    for name, metrics in metrics_dict.items():
        auc = metrics['auc']
        print(f"{name:<25} {auc['fe_mean']:<12.2f} {auc['gmfe']:<10.2f} {auc['pct_2.0x']:<10.1f} {auc['r2']:<10.3f}")

    # Salvar resultados
    with open(output_dir / 'revalidation_results.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    # Plot
    plot_comparison(metrics_dict, output_dir)

    print(f"\n✅ Revalidação concluída!")
    print(f"   Resultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()

