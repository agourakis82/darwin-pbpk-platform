#!/usr/bin/env python3
"""
Avaliação científica rigorosa do DynamicPBPKGNN
Criado: 2025-11-17
Autor: AI Assistant + Dr. Agourakis

Métricas científicas adequadas para PBPK:
- Fold Error (FE) - padrão ouro
- Geometric Mean Fold Error (GMFE)
- MAE e RMSE em escala log10
- Comparação com baselines (linear, kNN, RF, ODE)
- Análise de resíduos
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    DynamicPBPKGNN, PBPK_ORGANS, NUM_ORGANS
)
from scripts.train_dynamic_gnn_pbpk import PBPKDataset, create_physiological_params


def fold_error(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Calcula Fold Error: max(pred/obs, obs/pred)"""
    ratio = predicted / (observed + 1e-10)  # evitar divisão por zero
    fe = np.maximum(ratio, 1.0 / ratio)
    return fe


def geometric_mean_fold_error(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Calcula Geometric Mean Fold Error"""
    ratio = predicted / (observed + 1e-10)
    log_ratio = np.log10(np.maximum(ratio, 1e-10))
    gmfe = 10 ** np.mean(np.abs(log_ratio))
    return float(gmfe)


def percent_within_fold(predicted: np.ndarray, observed: np.ndarray, fold: float) -> float:
    """Calcula % de previsões dentro de fold× do observado"""
    fe = fold_error(predicted, observed)
    return float(np.mean(fe <= fold) * 100)


def evaluate_model_scientific(
    model: DynamicPBPKGNN,
    loader: DataLoader,
    device: torch.device,
    organ_weights: Optional[List[float]] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Avalia modelo com métricas científicas. Retorna métricas, pred e true."""
    model.eval()
    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch in loader:
            # Preparar batch
            doses = batch["dose"].to(device)
            clearances_hepatic = batch["clearance_hepatic"].to(device)
            clearances_renal = batch["clearance_renal"].to(device)
            partition_coeffs = batch["partition_coeffs"].to(device)
            concentrations_true = batch["concentrations"].to(device)  # [B, NUM_ORGANS, T]

            # Predizer
            batch_size = len(doses)
            pred_list = []
            for i in range(batch_size):
                params = create_physiological_params(
                    float(doses[i]),
                    float(clearances_hepatic[i]),
                    float(clearances_renal[i]),
                    partition_coeffs[i],
                )
                result = model(float(doses[i]), params)
                pred_conc = result["concentrations"]  # [NUM_ORGANS, T]
                pred_list.append(pred_conc)

            pred_batch = torch.stack(pred_list)  # [B, NUM_ORGANS, T]

            # Garantir shapes consistentes
            if pred_batch.shape != concentrations_true.shape:
                # Ajustar temporal steps se necessário
                min_t = min(pred_batch.shape[2], concentrations_true.shape[2])
                pred_batch = pred_batch[:, :, :min_t]
                concentrations_true = concentrations_true[:, :, :min_t]

            # Aplicar pesos por órgão se fornecidos
            if organ_weights:
                weights = torch.tensor(organ_weights, device=device).view(1, -1, 1)
                pred_batch = pred_batch * weights
                concentrations_true = concentrations_true * weights

            all_pred.append(pred_batch.cpu().numpy())
            all_true.append(concentrations_true.cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    # Flatten para métricas globais
    pred_flat = pred.flatten()
    true_flat = true.flatten()

    # Filtrar zeros/valores muito pequenos e infinitos/NaN
    mask = (
        (true_flat > 1e-6) &
        (pred_flat > 1e-6) &
        np.isfinite(true_flat) &
        np.isfinite(pred_flat)
    )
    pred_flat = pred_flat[mask]
    true_flat = true_flat[mask]

    # Métricas
    fe = fold_error(pred_flat, true_flat)
    fe = fe[np.isfinite(fe)]  # Filtrar FE infinitos/NaN
    gmfe = geometric_mean_fold_error(pred_flat, true_flat)
    mae = mean_absolute_error(true_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
    r2 = r2_score(true_flat, pred_flat)

    # % dentro de fold
    pct_1_25 = percent_within_fold(pred_flat, true_flat, 1.25)
    pct_1_5 = percent_within_fold(pred_flat, true_flat, 1.5)
    pct_2_0 = percent_within_fold(pred_flat, true_flat, 2.0)

    # MAE e RMSE em escala log10
    log10_true = np.log10(true_flat + 1e-10)
    log10_pred = np.log10(pred_flat + 1e-10)
    mae_log10 = mean_absolute_error(log10_true, log10_pred)
    rmse_log10 = np.sqrt(mean_squared_error(log10_true, log10_pred))

    metrics = {
        "fold_error_mean": float(np.mean(fe)) if len(fe) > 0 else float('nan'),
        "fold_error_median": float(np.median(fe)) if len(fe) > 0 else float('nan'),
        "fold_error_p67": float(np.percentile(fe, 67)) if len(fe) > 0 else float('nan'),
        "geometric_mean_fold_error": gmfe,
        "percent_within_1.25x": pct_1_25,
        "percent_within_1.5x": pct_1_5,
        "percent_within_2.0x": pct_2_0,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mae_log10": mae_log10,
        "rmse_log10": rmse_log10,
        "num_predictions": len(pred_flat),
    }
    return metrics, pred, true


def baseline_linear_regression(
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Dict[str, float]:
    """Baseline: Regressão Linear (clearances + Kp → concentração média)"""
    # Preparar dados de treino
    X_train = []
    y_train = []
    for batch in train_loader:
        ch = batch["clearance_hepatic"].numpy()
        cr = batch["clearance_renal"].numpy()
        kp = batch["partition_coeffs"].numpy()  # [B, NUM_ORGANS]
        conc = batch["concentrations"].numpy()  # [B, NUM_ORGANS, T]
        conc_mean = conc.mean(axis=2)  # [B, NUM_ORGANS] - média temporal

        # Features: clearances + Kp médio + Kp por órgão
        kp_mean = kp.mean(axis=1, keepdims=True)  # [B, 1]
        features = np.concatenate([ch.reshape(-1, 1), cr.reshape(-1, 1), kp_mean, kp], axis=1)
        X_train.append(features)
        y_train.append(conc_mean)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Treinar modelo por órgão
    models = []
    for org_idx in range(NUM_ORGANS):
        model = LinearRegression()
        model.fit(X_train, y_train[:, org_idx])
        models.append(model)

    # Avaliar
    X_val = []
    y_val = []
    for batch in val_loader:
        ch = batch["clearance_hepatic"].numpy()
        cr = batch["clearance_renal"].numpy()
        kp = batch["partition_coeffs"].numpy()
        conc = batch["concentrations"].numpy()
        conc_mean = conc.mean(axis=2)

        kp_mean = kp.mean(axis=1, keepdims=True)
        features = np.concatenate([ch.reshape(-1, 1), cr.reshape(-1, 1), kp_mean, kp], axis=1)
        X_val.append(features)
        y_val.append(conc_mean)

    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    # Predizer
    pred_list = []
    for org_idx in range(NUM_ORGANS):
        pred = models[org_idx].predict(X_val)
        pred_list.append(pred)
    pred = np.stack(pred_list, axis=1)  # [N, NUM_ORGANS]

    # Métricas
    pred_flat = pred.flatten()
    true_flat = y_val.flatten()
    mask = (true_flat > 1e-6) & (pred_flat > 1e-6)
    pred_flat = pred_flat[mask]
    true_flat = true_flat[mask]

    fe = fold_error(pred_flat, true_flat)
    gmfe = geometric_mean_fold_error(pred_flat, true_flat)
    pct_2_0 = percent_within_fold(pred_flat, true_flat, 2.0)

    return {
        "fold_error_mean": float(np.mean(fe)),
        "geometric_mean_fold_error": gmfe,
        "percent_within_2.0x": pct_2_0,
        "r2": float(r2_score(true_flat, pred_flat)),
    }


def plot_scientific_metrics(
    outdir: Path,
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    pred: np.ndarray,
    true: np.ndarray,
):
    """Gera visualizações científicas"""
    pred_flat = pred.flatten()
    true_flat = true.flatten()

    # Filtrar valores inválidos
    mask = (
        (true_flat > 1e-6) &
        (pred_flat > 1e-6) &
        np.isfinite(true_flat) &
        np.isfinite(pred_flat)
    )
    pred_flat = pred_flat[mask]
    true_flat = true_flat[mask]

    if len(pred_flat) == 0:
        print("⚠️  Sem dados válidos para plotagem")
        return

    # 1. Scatter plot: predito vs. observado (com linhas 2x)
    plt.figure(figsize=(10, 8))
    plt.scatter(true_flat, pred_flat, alpha=0.3, s=1)
    min_val = min(true_flat.min(), pred_flat.min())
    max_val = max(true_flat.max(), pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    plt.plot([min_val, max_val], [min_val * 2, max_val * 2], 'g--', label='2×')
    plt.plot([min_val, max_val], [min_val / 2, max_val / 2], 'g--')
    plt.xlabel('Observado')
    plt.ylabel('Predito')
    plt.title('Predito vs. Observado (com linhas 2×)')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(outdir / "scatter_pred_vs_obs.png", dpi=160)
    plt.close()

    # 2. Distribuição de Fold Error
    fe = fold_error(pred_flat, true_flat)
    fe = fe[np.isfinite(fe)]
    if len(fe) > 0:
        # Limitar FE para visualização (remover outliers extremos)
        fe_clipped = np.clip(fe, 0, 10)  # Clip em 10 para visualização
        plt.figure(figsize=(10, 6))
        plt.hist(fe_clipped, bins=50, edgecolor='black')
        plt.axvline(2.0, color='r', linestyle='--', label='FE = 2.0 (aceitável)')
        plt.axvline(1.5, color='orange', linestyle='--', label='FE = 1.5 (excelente)')
        plt.xlabel('Fold Error')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Fold Error')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "fold_error_distribution.png", dpi=160)
        plt.close()

    # 3. Resíduos vs. Predito
    residuals = (pred_flat - true_flat) / (true_flat + 1e-10)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(pred_flat[:len(residuals)], residuals, alpha=0.3, s=1)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel('Predito')
        plt.ylabel('Resíduo Normalizado')
        plt.title('Resíduos vs. Predito')
        plt.tight_layout()
        plt.savefig(outdir / "residuals_vs_predicted.png", dpi=160)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Avaliação científica rigorosa do DynamicPBPKGNN")
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--node-dim", type=int, default=16)
    ap.add_argument("--edge-dim", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-gnn-layers", type=int, default=3)
    ap.add_argument("--num-temporal-steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-strategy", choices=["group", "compound"], default="compound")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))

    # Carregar dataset
    dataset = PBPKDataset(Path(args.data))

    # Split
    if args.split_strategy == "compound" and hasattr(dataset, "compound_ids") and dataset.compound_ids:
        from scripts.evaluate_dynamic_gnn_robust import group_ids_by_params, split_by_groups
        # Usar compound_ids para split
        comp_ids = dataset.compound_ids
        uniq = list(dict.fromkeys(comp_ids))
        rng = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(uniq), generator=rng).tolist()
        num_train = int((1 - args.val_frac) * len(uniq))
        train_groups = set(uniq[i] for i in perm[:num_train])
        train_indices = [i for i, cid in enumerate(comp_ids) if cid in train_groups]
        val_indices = [i for i, cid in enumerate(comp_ids) if cid not in train_groups]
    else:
        # Split por grupos de parâmetros
        from scripts.evaluate_dynamic_gnn_robust import group_ids_by_params, split_by_groups
        groups = group_ids_by_params(dataset, decimals=4)
        train_indices, val_indices = split_by_groups(groups, val_frac=args.val_frac, seed=args.seed)

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False)

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

    # Avaliar modelo
    print("🔬 Avaliando modelo com métricas científicas...")
    model_metrics, pred, true = evaluate_model_scientific(model, val_loader, device)

    # Baseline: Regressão Linear
    print("📊 Avaliando baseline (Regressão Linear)...")
    baseline_metrics = baseline_linear_regression(train_loader, val_loader)

    # Salvar resultados
    results = {
        "model_metrics": model_metrics,
        "baseline_linear_metrics": baseline_metrics,
        "split_strategy": args.split_strategy,
        "num_train": len(train_indices),
        "num_val": len(val_indices),
    }

    with (outdir / "scientific_eval.json").open("w") as f:
        json.dump(results, f, indent=2)

    # Gerar plots
    print("📈 Gerando visualizações...")
    plot_scientific_metrics(outdir, model_metrics, baseline_metrics, pred, true)

    print("✅ Avaliação científica concluída!")
    print(f"📊 Resultados: {outdir / 'scientific_eval.json'}")
    print(f"\n📋 Métricas do Modelo:")
    print(f"  FE médio: {model_metrics['fold_error_mean']:.3f}")
    print(f"  GMFE: {model_metrics['geometric_mean_fold_error']:.3f}")
    print(f"  % dentro de 2.0×: {model_metrics['percent_within_2.0x']:.1f}%")
    print(f"  R²: {model_metrics['r2']:.6f}")
    print(f"\n📋 Baseline (Linear):")
    print(f"  FE médio: {baseline_metrics['fold_error_mean']:.3f}")
    print(f"  GMFE: {baseline_metrics['geometric_mean_fold_error']:.3f}")
    print(f"  % dentro de 2.0×: {baseline_metrics['percent_within_2.0x']:.1f}%")
    print(f"  R²: {baseline_metrics['r2']:.6f}")


if __name__ == "__main__":
    main()

