#!/usr/bin/env python3
"""
Avaliação ampla de checkpoint do DynamicPBPKGNN
Criado: 2025-11-15 22:10 -03
Autor: AI Assistant + Dr. Agourakis

Funções:
- Carrega dataset .npz (mesmo formato do treino)
- Constrói split determinístico (train/val) para avaliação
- Carrega checkpoint e roda inferência batelada (forward_batch)
- Calcula MSE, MAE e R² por órgão e globais
- Salva métricas em CSV/JSON e gráficos comparativos
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Path do projeto para import local
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (  # noqa: E402
    DynamicPBPKGNN,
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
    NUM_ORGANS,
)
from scripts.train_dynamic_gnn_pbpk import (  # noqa: E402
    PBPKDataset,
    create_physiological_params,
)


@torch.no_grad()
def evaluate_loader(
    model: DynamicPBPKGNN,
    dataloader: DataLoader,
    device: torch.device,
    exclude_t0: bool = False,
    max_time_points: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Roda avaliação no dataloader e retorna métricas por órgão e globais:
    - MSE por órgão
    - MAE por órgão
    - R² por órgão
    Além disso, calcula métricas globais agregando todos os órgãos/tempos.
    """
    model.eval()
    # Acumular listas para métricas globais
    y_true_all = []
    y_pred_all = []

    # Acumular por órgão
    per_organ_true = {organ: [] for organ in PBPK_ORGANS}
    per_organ_pred = {organ: [] for organ in PBPK_ORGANS}

    for batch in dataloader:
        dose = batch["dose"].to(device)
        clearance_hepatic = batch["clearance_hepatic"].to(device)
        clearance_renal = batch["clearance_renal"].to(device)
        partition_coeffs = batch["partition_coeffs"].to(device)
        true_conc = batch["concentrations"].to(device)  # [B, Org, T] ou [Org, T]
        time_points = batch["time_points"].to(device)

        if time_points.dim() > 1:
            time_points = time_points[0]

        params_batch = [
            create_physiological_params(
                dose=0.0,
                clearance_hepatic=clearance_hepatic[i].item(),
                clearance_renal=clearance_renal[i].item(),
                partition_coeffs=partition_coeffs[i].detach().cpu(),
            )
            for i in range(len(dose))
        ]

        results = model.forward_batch(dose, params_batch, time_points)
        pred_conc = results["concentrations"]  # [B, Org, T]

        # Ajustar comprimento temporal se necessário
        if pred_conc.shape[-1] != true_conc.shape[-1]:
            min_t = min(pred_conc.shape[-1], true_conc.shape[-1])
            pred_conc = pred_conc[..., :min_t]
            true_conc = true_conc[..., :min_t]

        # Filtrar tempo: excluir t=0 e/ou limitar janela
        if exclude_t0 or (max_time_points is not None):
            t_start = 1 if exclude_t0 else 0
            t_end = max_time_points if max_time_points is not None else pred_conc.shape[-1]
            pred_conc = pred_conc[..., t_start:t_end]
            true_conc = true_conc[..., t_start:t_end]

        # Empilhar para globais
        y_true_all.append(true_conc.detach().cpu().numpy())
        y_pred_all.append(pred_conc.detach().cpu().numpy())

        # Por órgão
        for organ_idx, organ in enumerate(PBPK_ORGANS):
            per_organ_true[organ].append(
                true_conc[:, organ_idx, :].detach().cpu().numpy()
            )
            per_organ_pred[organ].append(
                pred_conc[:, organ_idx, :].detach().cpu().numpy()
            )

    # Concatenar
    y_true_all = np.concatenate(y_true_all, axis=0)  # [N, Org, T]
    y_pred_all = np.concatenate(y_pred_all, axis=0)  # [N, Org, T]

    # Globais (flatten Org × T)
    y_true_flat = y_true_all.reshape(-1)
    y_pred_flat = y_pred_all.reshape(-1)

    # Métricas globais (implementação sem sklearn)
    diff = y_true_flat - y_pred_flat
    global_mse = float(np.mean(diff ** 2))
    global_mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    mu = float(np.mean(y_true_flat))
    ss_tot = float(np.sum((y_true_flat - mu) ** 2))
    global_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    mse_per_organ: Dict[str, float] = {}
    mae_per_organ: Dict[str, float] = {}
    r2_per_organ: Dict[str, float] = {}

    for organ in PBPK_ORGANS:
        yt = np.concatenate(per_organ_true[organ], axis=0).reshape(-1)
        yp = np.concatenate(per_organ_pred[organ], axis=0).reshape(-1)
        d = yt - yp
        mse_per_organ[organ] = float(np.mean(d ** 2))
        mae_per_organ[organ] = float(np.mean(np.abs(d)))
        ss_res_o = float(np.sum(d ** 2))
        mu_o = float(np.mean(yt))
        ss_tot_o = float(np.sum((yt - mu_o) ** 2))
        r2_per_organ[organ] = float(1.0 - ss_res_o / ss_tot_o) if ss_tot_o > 0 else float("nan")

    global_metrics = {"mse": global_mse, "mae": global_mae, "r2": global_r2}
    return mse_per_organ, mae_per_organ, r2_per_organ | {"__global__": global_metrics}


def plot_residuals_per_organ(
    outdir: Path,
    per_organ_true: Dict[str, np.ndarray],
    per_organ_pred: Dict[str, np.ndarray],
) -> None:
    """Reservado para expansão futura (não usado nesta versão)."""
    # Placeholder proposital para futuras versões com residuals por órgão/tempo
    return


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação de checkpoint DynamicPBPKGNN")
    parser.add_argument("--data", type=str, required=True, help="Caminho para .npz")
    parser.add_argument("--checkpoint", type=str, required=True, help="best_model.pt")
    parser.add_argument("--output-dir", type=str, required=True, help="dir de saída")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")  # cuda/cpu/auto
    parser.add_argument("--node-dim", type=int, default=16)
    parser.add_argument("--edge-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-gnn-layers", type=int, default=3)
    parser.add_argument("--num-temporal-steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-t0", action="store_true", help="Excluir t=0 das métricas")
    parser.add_argument("--max-time-points", type=int, default=None, help="Limitar número de pontos de tempo usados nas métricas")
    args = parser.parse_args()

    data_path = Path(args.data)
    ckpt_path = Path(args.checkpoint)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Dataset e split determinístico (mesma fração usada no treino, seed fixa)
    full_dataset = PBPKDataset(data_path)
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Modelo
    model = DynamicPBPKGNN(
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_temporal_steps=args.num_temporal_steps,
        dt=args.dt,
    ).to(device)

    # Carregar checkpoint (somente pesos do modelo)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    # Avaliar
    mse_org, mae_org, r2_org = evaluate_loader(
        model,
        val_loader,
        device,
        exclude_t0=args.exclude_t0,
        max_time_points=args.max_time_points,
    )

    # Salvar métricas
    per_organ_table = []
    for organ in PBPK_ORGANS:
        per_organ_table.append(
            {
                "organ": organ,
                "mse": mse_org[organ],
                "mae": mae_org[organ],
                "r2": r2_org.get(organ, float("nan")),
            }
        )
    # Global
    per_organ_table.append(
        {
            "organ": "__global__",
            "mse": r2_org["__global__"]["mse"],
            "mae": r2_org["__global__"]["mae"],
            "r2": r2_org["__global__"]["r2"],
        }
    )

    import pandas as pd

    df = pd.DataFrame(per_organ_table)
    df.to_csv(outdir / "evaluation_metrics_per_organ.csv", index=False)
    with (outdir / "evaluation_metrics.json").open("w") as fh:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "data": str(data_path),
                "device": str(device),
                "metrics": {
                    "per_organ": {row["organ"]: {k: float(row[k]) for k in ["mse", "mae", "r2"]} for _, row in df.iterrows() if row["organ"] != "__global__"},
                    "global": r2_org["__global__"],
                },
            },
            fh,
            indent=2,
        )

    # Plot simples: barras de R² por órgão
    plt.figure(figsize=(10, 4))
    organs = [row["organ"] for row in per_organ_table if row["organ"] != "__global__"]
    r2_vals = [row["r2"] for row in per_organ_table if row["organ"] != "__global__"]
    plt.bar(organs, r2_vals, color="#2ca02c")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("R² (val)")
    plt.title("R² por órgão (validação)")
    plt.tight_layout()
    plt.savefig(outdir / "r2_per_organ.png", dpi=160)
    plt.close()

    print("✅ Avaliação concluída.")
    print(f" - CSV: {outdir / 'evaluation_metrics_per_organ.csv'}")
    print(f" - JSON: {outdir / 'evaluation_metrics.json'}")
    print(f" - Plot: {outdir / 'r2_per_organ.png'}")


if __name__ == "__main__":
    main()


