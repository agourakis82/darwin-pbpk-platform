#!/usr/bin/env python3
"""
Baseline kNN (memorization) para DynamicPBPKGNN
Criado: 2025-11-15 23:05 -03
Autor: AI Assistant + Dr. Agourakis

Descrição:
- Usa features simples (clearances + partition_coeffs) para encontrar k vizinhos
  no conjunto de treino (split por grupos) e prever a série de concentração como
  média das séries dos vizinhos (por órgão×tempo).
- Retorna métricas globais por janelas temporais (lin e log1p).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from scripts.train_dynamic_gnn_pbpk import PBPKDataset  # noqa: E402


def build_features(dataset: PBPKDataset) -> np.ndarray:
    ch = dataset.clearances_hepatic.numpy().reshape(-1, 1)
    cr = dataset.clearances_renal.numpy().reshape(-1, 1)
    pc = dataset.partition_coeffs.numpy()  # [N, Org]
    X = np.concatenate([ch, cr, pc], axis=1)
    return X


def group_split_hash(dataset: PBPKDataset, train_frac: float = 0.8, seed: int = 42) -> Tuple[List[int], List[int]]:
    def make_hash(i: int) -> str:
        ch = float(dataset.clearances_hepatic[i])
        cr = float(dataset.clearances_renal[i])
        pc = dataset.partition_coeffs[i].numpy()
        return f"{round(ch,6)}|{round(cr,6)}|" + ",".join(f"{float(x):.6f}" for x in pc.tolist())
    hashes = [make_hash(i) for i in range(len(dataset))]
    uniq = list(dict.fromkeys(hashes))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    ntr = int(train_frac * len(uniq))
    tr_keys = set(uniq[:ntr])
    train_idx = [i for i, h in enumerate(hashes) if h in tr_keys]
    val_idx = [i for i, h in enumerate(hashes) if h not in tr_keys]
    return train_idx, val_idx


def knn_predict(X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, k: int = 3) -> np.ndarray:
    # Distância euclidiana simples
    # Retorna média das Y_train dos k vizinhos
    preds = []
    for xv in X_val:
        d = np.linalg.norm(X_train - xv[None, :], axis=1)
        idx = np.argsort(d)[:k]
        preds.append(Y_train[idx].mean(axis=0))
    return np.stack(preds, axis=0)  # [Nv, Org, T]


def compute_metrics(yt: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
    diff = yt - yp
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    mu = float(np.mean(yt))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((yt - mu) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2}


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline kNN para DynamicPBPKGNN")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--windows", type=str, default="1-12,12-24,24-48,48-100")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = PBPKDataset(Path(args.data))
    tr_idx, va_idx = group_split_hash(ds, train_frac=args.train_frac, seed=args.seed)
    ds_tr = Subset(ds, tr_idx)
    ds_va = Subset(ds, va_idx)

    # Features e targets
    X_tr = build_features(ds)
    Y = ds.concentrations.numpy()  # [N, Org, T]
    X_train = X_tr[tr_idx]
    Y_train = Y[tr_idx]
    X_val = X_tr[va_idx]
    Y_val = Y[va_idx]

    # kNN
    Y_pred = knn_predict(X_train, Y_train, X_val, k=args.k)

    # Janelas
    wins = []
    for span in args.windows.split(","):
        a, b = span.split("-")
        wins.append((int(a), int(b)))

    import json
    results = {}
    for (s, e) in wins:
        s2 = max(s, 1)
        yp = Y_pred[:, :, s2:e]
        yt = Y_val[:, :, s2:e]
        # linear
        lin = compute_metrics(yt.reshape(-1), yp.reshape(-1))
        # log1p
        yp_l = np.log1p(np.maximum(yp, 0.0))
        yt_l = np.log1p(np.maximum(yt, 0.0))
        logm = compute_metrics(yt_l.reshape(-1), yp_l.reshape(-1))
        results[f"{s2}:{e}"] = {"linear": lin, "log1p": logm}

    with (out / "knn_baseline.json").open("w") as fh:
        json.dump(results, fh, indent=2)
    print("✅ kNN baseline concluído.")
    print(f" - JSON: {out / 'knn_baseline.json'}")


if __name__ == "__main__":
    main()



