#!/usr/bin/env python3
"""
Avaliação robusta do DynamicPBPKGNN
Criado: 2025-11-15 22:35 -03
Autor: AI Assistant + Dr. Agourakis

Recursos:
- Split por grupos de "composto" (proxy por parâmetros redondados)
- Métricas por janela temporal (subfaixas de tempo)
- Métricas em escala linear e log1p
- Baselines:
    (a) baseline_mean: média no treino por órgão×tempo
    (b) baseline_zero: todos zeros (referência inferior)

Saídas:
- JSON consolidado com métricas por janela (modelo vs baselines)
- CSV por janela e por órgão
- Gráficos de barras de R² por janela (modelo vs baselines)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (  # noqa: E402
    DynamicPBPKGNN, PBPK_ORGANS
)
from scripts.train_dynamic_gnn_pbpk import PBPKDataset, create_physiological_params  # noqa: E402


def to_device(x, device):  # pequenos auxiliares
    return x.to(device) if hasattr(x, "to") else x


def group_ids_by_params(dataset: PBPKDataset, decimals: int = 4) -> Dict[str, List[int]]:
    """
    Agrupa índices por um proxy de identidade de composto:
    (clearances + partition_coeffs) arredondados.
    """
    groups: Dict[str, List[int]] = {}
    for i in range(len(dataset)):
        ch = float(round(dataset.clearances_hepatic[i].item(), decimals))
        cr = float(round(dataset.clearances_renal[i].item(), decimals))
        pc = np.round(dataset.partition_coeffs[i].numpy(), decimals)
        key = f"{ch}|{cr}|" + ",".join(map(lambda v: f"{v:.{decimals}f}", pc.tolist()))
        groups.setdefault(key, []).append(i)
    return groups


def split_by_groups(groups: Dict[str, List[int]], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:n_val])
    train_idx, val_idx = [], []
    for k, idxs in groups.items():
        (val_idx if k in val_keys else train_idx).extend(idxs)
    return train_idx, val_idx


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    mu = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mu) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2}


def evaluate_model_on_loader(
    model: DynamicPBPKGNN,
    loader: DataLoader,
    device: torch.device,
    windows: List[Tuple[int, int]],
    use_log1p: bool,
    conc_threshold: float = 0.0,
    organ_weights: List[float] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Avalia o modelo por janelas temporais (val) e retorna métricas globais por janela.
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            dose = to_device(batch["dose"], device)
            ch = to_device(batch["clearance_hepatic"], device)
            cr = to_device(batch["clearance_renal"], device)
            pc = to_device(batch["partition_coeffs"], device)
            conc = to_device(batch["concentrations"], device)  # [B, Org, T]
            t = to_device(batch["time_points"], device)
            if t.dim() > 1:
                t = t[0]
            params_batch = [
                create_physiological_params(
                    dose=0.0,
                    clearance_hepatic=ch[i].item(),
                    clearance_renal=cr[i].item(),
                    partition_coeffs=pc[i].detach().cpu(),
                )
                for i in range(len(dose))
            ]
            out = model.forward_batch(dose, params_batch, t)
            yhat = out["concentrations"]
            T = min(yhat.shape[-1], conc.shape[-1])
            preds.append(yhat[..., :T].detach().cpu().numpy())
            trues.append(conc[..., :T].detach().cpu().numpy())
    YP = np.concatenate(preds, axis=0)  # [N, Org, T]
    YT = np.concatenate(trues, axis=0)
    results: Dict[str, Dict[str, float]] = {}
    for (s, e) in windows:
        s2 = max(s, 1)  # excluir t=0 implicitamente
        e2 = min(e, YP.shape[-1])
        yp_win = YP[..., s2:e2]
        yt_win = YT[..., s2:e2]
        # threshold por concentração: mascarar pontos com yt < threshold
        if conc_threshold > 0.0:
            mask = (yt_win >= conc_threshold)
            # Para evitar viés de broadcasting, filtramos flatten com a mesma máscara
            yp_f = yp_win[mask]
            yt_f = yt_win[mask]
        else:
            yp_f = yp_win.reshape(-1)
            yt_f = yt_win.reshape(-1)
        if use_log1p:
            yp_f = np.log1p(np.maximum(yp_f, 0.0))
            yt_f = np.log1p(np.maximum(yt_f, 0.0))
        # pesos por órgão (opcional)
        if organ_weights is not None and conc_threshold <= 0.0:
            # Quando sem threshold: aplicar pesos por órgão em nível [Org,T]
            ow = np.asarray(organ_weights, dtype=float)  # [Org]
            w = np.repeat(ow[:, None], repeats=yt_win.shape[-1], axis=1)  # [Org,T]
            w = w.reshape(-1)
            yt_flat = yt_win.reshape(-1)
            yp_flat = yp_win.reshape(-1)
            # cálculo ponderado
            diff = yt_flat - yp_flat
            mse = float(np.average(diff ** 2, weights=w))
            mae = float(np.average(np.abs(diff), weights=w))
            mu = float(np.average(yt_flat, weights=w))
            ss_res = float(np.average((yt_flat - yp_flat) ** 2, weights=w) * w.size)
            ss_tot = float(np.average((yt_flat - mu) ** 2, weights=w) * w.size)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            results[f"{s2}:{e2}"] = {"mse": mse, "mae": mae, "r2": r2}
        else:
            results[f"{s2}:{e2}"] = compute_metrics(yt_f, yp_f)
    return results


def evaluate_baselines(
    train_loader: DataLoader,
    val_loader: DataLoader,
    windows: List[Tuple[int, int]],
    use_log1p: bool,
    conc_threshold: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """
    Baselines:
      - mean: média no treino por órgão×tempo
      - zero: tudo zero
    """
    # Agregar média no treino
    train_conc = []
    with torch.no_grad():
        for batch in train_loader:
            c = batch["concentrations"].numpy()
            train_conc.append(c)
    TR = np.concatenate(train_conc, axis=0)  # [Ntr, Org, T]
    mean_train = TR.mean(axis=0)  # [Org, T]

    # Coletar val truths
    val_truths = []
    with torch.no_grad():
        for batch in val_loader:
            val_truths.append(batch["concentrations"].numpy())
    VT = np.concatenate(val_truths, axis=0)  # [Nv, Org, T]

    results: Dict[str, Dict[str, float]] = {}
    for (s, e) in windows:
        s2 = max(s, 1)
        e2 = min(e, VT.shape[-1])
        # baseline mean
        yb_mean = np.broadcast_to(mean_train[:, s2:e2], VT[:, :, s2:e2].shape)
        # baseline zero
        yb_zero = np.zeros_like(VT[:, :, s2:e2])
        yt = VT[:, :, s2:e2]
        if use_log1p:
            yb_mean = np.log1p(np.maximum(yb_mean, 0.0))
            yb_zero = np.log1p(np.maximum(yb_zero, 0.0))
            yt = np.log1p(np.maximum(yt, 0.0))
        if conc_threshold > 0.0:
            mask = (yt >= conc_threshold)
            res_mean = compute_metrics(yt[mask], yb_mean[mask])
            res_zero = compute_metrics(yt[mask], yb_zero[mask])
        else:
            res_mean = compute_metrics(yt.reshape(-1), yb_mean.reshape(-1))
            res_zero = compute_metrics(yt.reshape(-1), yb_zero.reshape(-1))
        results[f"mean:{s2}:{e2}"] = res_mean
        results[f"zero:{s2}:{e2}"] = res_zero
    return results


def plot_window_r2(outdir: Path, model_lin: Dict[str, Dict[str, float]], model_log: Dict[str, Dict[str, float]], base_lin: Dict[str, Dict[str, float]], base_log: Dict[str, Dict[str, float]]) -> None:
    def extract(vals: Dict[str, Dict[str, float]], keys: List[str]) -> List[float]:
        return [vals[k]["r2"] for k in keys]
    win_keys = [k for k in model_lin.keys()]
    plt.figure(figsize=(10, 4.5))
    x = np.arange(len(win_keys))
    width = 0.18
    plt.bar(x - 1.5*width, extract(model_lin, win_keys), width, label="Model R² (lin)")
    plt.bar(x - 0.5*width, extract(model_log, win_keys), width, label="Model R² (log1p)")
    plt.bar(x + 0.5*width, [base_lin[f"mean:{k}"]["r2"] for k in win_keys], width, label="Baseline mean (lin)")
    plt.bar(x + 1.5*width, [base_log[f"mean:{k}"]["r2"] for k in win_keys], width, label="Baseline mean (log1p)")
    plt.xticks(x, win_keys, rotation=0)
    plt.ylabel("R²")
    plt.title("R² por janela temporal — modelo vs baseline mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "r2_by_window.png", dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação robusta do DynamicPBPKGNN")
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
    ap.add_argument("--round-decimals", type=int, default=4)
    ap.add_argument("--windows", type=str, default="1-12,12-24,24-48,48-100")
    ap.add_argument("--conc-threshold", type=float, default=0.0, help="Ignorar pontos com concentração < threshold")
    ap.add_argument("--organ-weights", type=str, default="", help="Pesos por órgão (lista de 14 floats separadas por vírgula)")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))

    dataset = PBPKDataset(Path(args.data))
    groups = group_ids_by_params(dataset, decimals=args.round_decimals)
    train_idx, val_idx = split_by_groups(groups, val_frac=args.val_frac, seed=args.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

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

    # Preparar janelas
    wins: List[Tuple[int, int]] = []
    for span in args.windows.split(","):
        a, b = span.split("-")
        wins.append((int(a), int(b)))

    # Modelo: linear e log1p
    ow = None
    if args.organ_weights:
        ow = [float(x) for x in args.organ_weights.split(",")]
    model_lin = evaluate_model_on_loader(model, val_loader, device, wins, use_log1p=False, conc_threshold=args.conc_threshold, organ_weights=ow)
    model_log = evaluate_model_on_loader(model, val_loader, device, wins, use_log1p=True, conc_threshold=args.conc_threshold, organ_weights=ow)
    # Baselines (usam dados do treino para estimar média)
    base_lin = evaluate_baselines(train_loader, val_loader, wins, use_log1p=False, conc_threshold=args.conc_threshold)
    base_log = evaluate_baselines(train_loader, val_loader, wins, use_log1p=True, conc_threshold=args.conc_threshold)

    # Salvar JSON consolidado
    with (outdir / "robust_eval.json").open("w") as fh:
        json.dump(
            {
                "windows": wins,
                "model_linear": model_lin,
                "model_log1p": model_log,
                "baseline_mean_linear": {k.split("mean:")[1]: v for k, v in base_lin.items() if k.startswith("mean:")},
                "baseline_mean_log1p": {k.split("mean:")[1]: v for k, v in base_log.items() if k.startswith("mean:")},
                "baseline_zero_linear": {k.split("zero:")[1]: v for k, v in base_lin.items() if k.startswith("zero:")},
                "baseline_zero_log1p": {k.split("zero:")[1]: v for k, v in base_log.items() if k.startswith("zero:")},
                "group_split": {"num_groups": len(groups), "train_idx": len(train_idx), "val_idx": len(val_idx)},
            },
            fh,
            indent=2,
        )

    # Plot R² por janela
    plot_window_r2(outdir, model_lin, model_log, base_lin, base_log)

    print("✅ Avaliação robusta concluída.")
    print(f" - JSON: {outdir / 'robust_eval.json'}")
    print(f" - Plot: {outdir / 'r2_by_window.png'}")


if __name__ == "__main__":
    main()


