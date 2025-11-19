#!/usr/bin/env python3
"""
Deduplicação do dataset Dynamic GNN PBPK por hash de parâmetros
Criado: 2025-11-15 23:20 -03
Autor: AI Assistant + Dr. Agourakis

Regra:
- Hash = (dose, clearance_hepatic, clearance_renal, partition_coeffs arredondados)
- Mantém a primeira ocorrência de cada hash
- Salva .npz com arrays filtrados preservando shapes [N, Org, T]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def make_param_hash(dose: float, ch: float, cr: float, pc: np.ndarray, decimals: int = 6) -> str:
    d = round(float(dose), decimals)
    ch = round(float(ch), decimals)
    cr = round(float(cr), decimals)
    pc = np.round(pc.astype(float), decimals)
    return f"{d}|{ch}|{cr}|" + ",".join(f"{v:.{decimals}f}" for v in pc.tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description="Deduplicar dataset DynamicPBPKGNN por hash de parâmetros")
    ap.add_argument("--input", required=True, help="Caminho para .npz original")
    ap.add_argument("--output", required=True, help="Caminho de saída .npz deduplicado")
    ap.add_argument("--decimals", type=int, default=6, help="Arredondamento no hash")
    args = ap.parse_args()

    data = np.load(args.input)
    doses = data["doses"]
    ch = data["clearances_hepatic"]
    cr = data["clearances_renal"]
    pc = data["partition_coeffs"]
    conc = data["concentrations"]
    time_points = data["time_points"]

    # Garantir [N, Org, T]
    if conc.ndim == 3 and conc.shape[1] != pc.shape[1]:
        conc = np.transpose(conc, (0, 2, 1))

    N = len(doses)
    seen = set()
    keep_idx = []
    for i in range(N):
        h = make_param_hash(doses[i], ch[i], cr[i], pc[i], decimals=args.decimals)
        if h in seen:
            continue
        seen.add(h)
        keep_idx.append(i)

    keep_idx = np.array(keep_idx, dtype=int)
    print(f"Original N={N} -> Deduplicado N={len(keep_idx)} (removidos {N-len(keep_idx)})")

    np.savez_compressed(
        args.output,
        doses=doses[keep_idx],
        clearances_hepatic=ch[keep_idx],
        clearances_renal=cr[keep_idx],
        partition_coeffs=pc[keep_idx],
        concentrations=conc[keep_idx],
        time_points=time_points,
    )
    print(f"✅ Dataset deduplicado salvo em: {args.output}")


if __name__ == "__main__":
    main()



