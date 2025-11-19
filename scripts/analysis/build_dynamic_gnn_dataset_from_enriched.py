#!/usr/bin/env python3
"""Gera dataset NPZ para o DynamicPBPKGNN a partir de parâmetros reais.

Estratégia: usar o simulador pré-treinado para gerar curvas alvo (distilação) com
os parâmetros observados em `pbpk_parameters_wide_enriched_v3.csv`.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

import torch

import sys
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PARAMS_PATH = BASE_DIR / "analysis" / "pbpk_parameters_wide_enriched_v3.csv"
CHECKPOINT = BASE_DIR / "models" / "dynamic_gnn_full" / "best_model.pt"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "pbpk_enriched"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "dynamic_gnn_dataset_enriched_v4.npz"

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    DynamicPBPKGNN,
    DynamicPBPKSimulator,
    PBPK_ORGANS,
    NUM_ORGANS,
)

TIME_HOURS = 24.0
DT_HOURS = 0.5
TIME_POINTS = np.arange(0.0, TIME_HOURS + DT_HOURS, DT_HOURS)
DEFAULT_DOSE = 100.0
RENAL_RATIO = 0.25


def resolve_clearances(row: pd.Series) -> tuple[float, float]:
    hepatic = row.get("clearance_hepatic_l_h")
    total = row.get("clearance_l_h")
    microsome_hepatic = row.get("microsome_hepatic_l_h")

    if not np.isnan(hepatic):
        hepatic_cl = float(hepatic)
    elif not np.isnan(total):
        hepatic_cl = float(total) * (1.0 - RENAL_RATIO)
    elif not np.isnan(microsome_hepatic):
        hepatic_cl = float(microsome_hepatic)
    else:
        hepatic_cl = 10.0

    renal_cl = float(total - hepatic_cl) if not np.isnan(total) else hepatic_cl * RENAL_RATIO
    renal_cl = max(renal_cl, 0.1)
    return hepatic_cl, renal_cl


def resolve_partition_coeffs(row: pd.Series) -> np.ndarray:
    vd = row.get("vd_l_kg")
    base = np.ones(NUM_ORGANS, dtype=np.float32)
    if not np.isnan(vd) and vd > 0:
        scale = min(max(vd / 40.0, 0.5), 3.0)
        base *= scale
    return base


def main() -> None:
    parser = ArgumentParser(description="Gera dataset sintético para o DynamicPBPKGNN")
    parser.add_argument('--max-samples', type=int, default=None, help='Limite de compostos a simular (default: todos)')
    parser.add_argument('--dose-min', type=float, default=50.0, help='Dose mínima (mg) para variação')
    parser.add_argument('--dose-max', type=float, default=200.0, help='Dose máxima (mg) para variação')
    parser.add_argument('--noise-kp-std', type=float, default=0.15, help='Desvio padrão do ruído lognormal nos Kp por órgão')
    parser.add_argument('--noise-clear-frac', type=float, default=0.10, help='Ruído relativo (gaussiano) nos clearances (± frac)')
    parser.add_argument('--output', type=str, default=str(OUTPUT_PATH), help='Caminho de saída do NPZ')
    args = parser.parse_args()

    params_df = pd.read_csv(PARAMS_PATH)
    if args.max_samples is not None:
        params_df = params_df.head(args.max_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DynamicPBPKGNN(
        node_dim=16,
        edge_dim=4,
        hidden_dim=64,
        num_gnn_layers=3,
        num_temporal_steps=len(TIME_POINTS) - 1,
        dt=float(DT_HOURS),
        use_attention=True,
    )
    simulator = DynamicPBPKSimulator(
        model=model,
        device=device,
        checkpoint_path=str(CHECKPOINT) if CHECKPOINT.exists() else None,
        map_location=device,
        strict=False,
    )

    doses = []
    cl_hepatic = []
    cl_renal = []
    partition_coeffs = []
    concentrations = []
    compound_ids = []

    rng = np.random.default_rng(42)
    for _, row in tqdm(params_df.iterrows(), total=len(params_df), desc="Simulating"):
        hepatic_cl, renal_cl = resolve_clearances(row)
        kp = resolve_partition_coeffs(row)
        # ruído em Kp (lognormal, média 1.0)
        if args.noise_kp_std > 0:
            kp_noise = rng.lognormal(mean=0.0, sigma=args.noise_kp_std, size=kp.shape).astype(np.float32)
            kp = (kp * kp_noise).astype(np.float32)
        # ruído em clearances (gaussiano relativo)
        if args.noise_clear_frac > 0:
            hepatic_cl = float(max(0.01, hepatic_cl * (1.0 + rng.normal(0.0, args.noise_clear_frac))))
            renal_cl = float(max(0.01, renal_cl * (1.0 + rng.normal(0.0, args.noise_clear_frac))))
        # dose variável
        dose_val = float(rng.uniform(args.dose_min, args.dose_max))

        partition_dict = {organ: float(kp[i]) for i, organ in enumerate(PBPK_ORGANS)}
        result = simulator.simulate(
            dose=dose_val,
            clearance_hepatic=hepatic_cl,
            clearance_renal=renal_cl,
            partition_coeffs=partition_dict,
            time_points=TIME_POINTS,
        )

        doses.append(dose_val)
        cl_hepatic.append(hepatic_cl)
        cl_renal.append(renal_cl)
        partition_coeffs.append(kp)
        # compound id (preferir chembl_id; fallback para índice)
        cid = row.get("chembl_id")
        if pd.isna(cid):
            cid = row.get("canonical_smiles")
        if pd.isna(cid):
            cid = str(_)  # index
        compound_ids.append(str(cid))

        organ_arrays = []
        for organ in PBPK_ORGANS:
            conc = result.get(organ)
            if conc is None:
                raise ValueError(f"Órgão {organ} ausente na simulação")
            organ_arrays.append(np.asarray(conc, dtype=np.float32))
        conc_matrix = np.stack(organ_arrays, axis=0)
        concentrations.append(conc_matrix)

    np.savez_compressed(
        args.output,
        doses=np.array(doses, dtype=np.float32),
        clearances_hepatic=np.array(cl_hepatic, dtype=np.float32),
        clearances_renal=np.array(cl_renal, dtype=np.float32),
        partition_coeffs=np.stack(partition_coeffs).astype(np.float32),
        concentrations=np.stack(concentrations).astype(np.float32),
        time_points=TIME_POINTS.astype(np.float32),
        compound_ids=np.array(compound_ids, dtype=np.str_),
    )

    print(f"Dataset salvo em {args.output} (n={len(doses)})")


if __name__ == "__main__":
    main()
