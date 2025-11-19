#!/usr/bin/env python3
"""
Auditoria do dataset Dynamic GNN PBPK
Criado: 2025-11-15 22:55 -03
Autor: AI Assistant + Dr. Agourakis

Checks:
- Dimensões e estatísticas básicas
- Duplicatas exatas nas entradas (dose, clearances, partition_coeffs)
- Hash de parâmetros: cardinalidade e colisões
- Sobreposição entre splits (random vs group) de acordo com hash de parâmetros
- Variância temporal por órgão (detecção de séries quase-constantes)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_npz(p: Path) -> Dict[str, np.ndarray]:
    data = np.load(p)
    return {k: data[k] for k in data.files}


def make_param_hash(clear_h: float, clear_r: float, part: np.ndarray, dose: float, decimals: int = 6) -> str:
    ch = round(float(clear_h), decimals)
    cr = round(float(clear_r), decimals)
    d = round(float(dose), decimals)
    pc = np.round(part.astype(float), decimals)
    return f"{d}|{ch}|{cr}|" + ",".join(f"{v:.{decimals}f}" for v in pc.tolist())


def random_split(n: int, val_frac: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = int(n * val_frac)
    return idx[val_n:], idx[:val_n]


def group_split(hashes: List[str], val_frac: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = np.array(list(dict.fromkeys(hashes)))  # unique preserving order
    rng.shuffle(keys)
    val_n = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:val_n])
    train_idx, val_idx = [], []
    for i, h in enumerate(hashes):
        (val_idx if h in val_keys else train_idx).append(i)
    return np.array(train_idx), np.array(val_idx)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auditoria dataset DynamicPBPKGNN")
    ap.add_argument("--data", required=True)
    ap.add_argument("--decimals", type=int, default=6)
    args = ap.parse_args()

    data = load_npz(Path(args.data))
    doses = data["doses"]           # [N]
    ch = data["clearances_hepatic"] # [N]
    cr = data["clearances_renal"]   # [N]
    pc = data["partition_coeffs"]   # [N, Org]
    conc = data["concentrations"]   # [N, Org, T] ou [N, T, Org]
    t = data["time_points"]         # [T]

    # Garantir shape [N, Org, T]
    if conc.ndim == 3 and conc.shape[1] != pc.shape[1]:
        conc = np.transpose(conc, (0, 2, 1))

    N, Org, T = conc.shape
    print(f"N={N}, Org={Org}, T={T}")
    print(f"doses: min={doses.min():.6g}, max={doses.max():.6g}")
    print(f"clearances_hepatic: min={ch.min():.6g}, max={ch.max():.6g}")
    print(f"clearances_renal:   min={cr.min():.6g}, max={cr.max():.6g}")

    # Hashes de parâmetros
    hashes = [make_param_hash(ch[i], cr[i], pc[i], doses[i], decimals=args.decimals) for i in range(N)]
    unique_hashes, counts = np.unique(hashes, return_counts=True)
    dup_total = int((counts > 1).sum())
    print(f"Parâmetros únicos: {len(unique_hashes)} / {N}  (duplicatas de parâmetro: {dup_total})")

    # Duplicatas exatas (inputs + outputs)
    # Amostra: verificar k pares aleatórios de iguais parâmetros
    dup_pairs = [(i, j) for i in range(N) for j in range(i+1, N) if hashes[i] == hashes[j]]
    exact_out_dups = 0
    for i, j in dup_pairs[:2000]:  # limitar inspeção
        if np.allclose(conc[i], conc[j], atol=1e-12, rtol=0.0):
            exact_out_dups += 1
    print(f"Duplicatas de parâmetro inspecionadas: {len(dup_pairs[:2000])}, saídas idênticas: {exact_out_dups}")

    # Variância temporal por órgão
    var_time = conc.var(axis=2).mean(axis=0)  # média da variância temporal por órgão
    df_var = pd.DataFrame({"organ_idx": np.arange(Org), "var_time_mean": var_time})
    print("Variância temporal média por órgão (top-5 baixos):")
    print(df_var.sort_values("var_time_mean").head(5).to_string(index=False))

    # Sobreposição entre splits
    tr_r, va_r = random_split(N)
    tr_g, va_g = group_split(hashes, val_frac=0.2)
    # Interseção por hash: se qualquer hash aparece nos dois, há potencial leak de parâmetro
    set_tr_r = set([hashes[i] for i in tr_r])
    set_va_r = set([hashes[i] for i in va_r])
    set_tr_g = set([hashes[i] for i in tr_g])
    set_va_g = set([hashes[i] for i in va_g])
    leak_r = len(set_tr_r & set_va_r)
    leak_g = len(set_tr_g & set_va_g)
    print(f"Random split: hashes em comum train/val = {leak_r}")
    print(f"Group split:  hashes em comum train/val = {leak_g}")


if __name__ == "__main__":
    main()



