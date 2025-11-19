#!/usr/bin/env python3
"""Exploração de dados de clearance hepático e simulação PBPK.

- Entrada: data/clearance_hepatocyte_az.tab (µL/min/10^6 cells)
- Saídas: analysis/literature_clearance_stats.json, analysis/literature_simulation_summary.csv
- CLI parametrizável: ajusta fu, top-N e checkpoint.

Autor: AI Assistant (Darwin Workspace)
Data: 2025-11-11
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import DynamicPBPKGNN, DynamicPBPKSimulator

DATA_PATH = BASE_DIR / "data" / "clearance_hepatocyte_az.tab"
OUTPUT_DIR = BASE_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEPATO_PER_GRAM = 1.2e8
LIVER_WEIGHT_G = 1500.0
HEPATIC_BLOOD_FLOW_L_H = 90.0
DEFAULT_FU = 0.5
RENAL_FRACTION = 0.25
DEFAULT_DOSE_MG = 100.0
CHECKPOINT = BASE_DIR / "models" / "dynamic_gnn_full" / "best_model.pt"


@dataclass
class CompoundResult:
    compound_id: str
    smiles: str
    clearance_cell: float
    fu: float
    clint_l_h: float
    hepatic_cl_l_h: float
    renal_cl_l_h: float
    cmax: float
    t_cmax: float
    final_blood: float


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    df = pd.read_csv(path, sep="	")
    df = df.rename(columns={"ID": "compound_id", "X": "smiles", "Y": "clearance"})
    df["clearance"] = pd.to_numeric(df["clearance"], errors="coerce")
    df = df.dropna(subset=["clearance", "smiles"])
    return df


def dataset_summary(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "count": int(df["clearance"].count()),
        "mean": float(df["clearance"].mean()),
        "median": float(df["clearance"].median()),
        "std": float(df["clearance"].std()),
        "min": float(df["clearance"].min()),
        "max": float(df["clearance"].max()),
        "quantiles": df["clearance"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
    }


def clearance_cell_to_clint(value: float) -> float:
    cl_uL_min_per_cell = value
    cl_l_min = cl_uL_min_per_cell * 1e-6 * HEPATO_PER_GRAM * LIVER_WEIGHT_G
    return cl_l_min * 60.0


def well_stirred_model(clint: float, fu: float = DEFAULT_FU, q_h: float = HEPATIC_BLOOD_FLOW_L_H) -> float:
    return (q_h * fu * clint) / (q_h + fu * clint)


def select_extremes(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    extremes = pd.concat([df.nlargest(top_n, "clearance"), df.nsmallest(top_n, "clearance")])
    return extremes.drop_duplicates(subset=["compound_id"]).reset_index(drop=True)


def run_simulations(df: pd.DataFrame, top_n: int, fuse: float, checkpoint: Optional[Path]) -> List[CompoundResult]:
    selection = select_extremes(df, top_n)
    model = DynamicPBPKGNN(num_temporal_steps=100, dt=0.1)
    checkpoint_path = str(checkpoint) if checkpoint and checkpoint.exists() else None
    simulator = DynamicPBPKSimulator(
        model=model,
        device="cpu",
        checkpoint_path=checkpoint_path,
        strict=True,
    )

    results: List[CompoundResult] = []
    for _, row in selection.iterrows():
        cl_cell = float(row["clearance"])
        clint = clearance_cell_to_clint(cl_cell)
        hepatic_cl = well_stirred_model(clint, fuse)
        renal_cl = hepatic_cl * RENAL_FRACTION

        sim_output = simulator.simulate(
            dose=DEFAULT_DOSE_MG,
            clearance_hepatic=hepatic_cl,
            clearance_renal=renal_cl,
            partition_coeffs=None,
        )
        time = sim_output["time"]
        blood = sim_output["blood"]

        results.append(
            CompoundResult(
                compound_id=row["compound_id"],
                smiles=row["smiles"],
                clearance_cell=cl_cell,
                fu=fuse,
                clint_l_h=clint,
                hepatic_cl_l_h=hepatic_cl,
                renal_cl_l_h=renal_cl,
                cmax=float(np.max(blood)),
                t_cmax=float(time[np.argmax(blood)]),
                final_blood=float(blood[-1]),
            )
        )
    return results


def export_results(prefix: str, stats: Dict[str, float], results: List[CompoundResult]) -> None:
    stats_path = OUTPUT_DIR / f"{prefix}_clearance_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if results:
        df = pd.DataFrame([r.__dict__ for r in results])
        df = df.sort_values(by="clearance_cell", ascending=False)
        df.to_csv(OUTPUT_DIR / f"{prefix}_simulation_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explora dados experimentais de clearance e roda simulações PBPK")
    parser.add_argument("--dataset", type=Path, default=DATA_PATH, help="Arquivo tab com dados experimentais")
    parser.add_argument("--top", type=int, default=3, help="Quantidade de compostos extremos na análise")
    parser.add_argument("--fu", type=float, default=DEFAULT_FU, help="Fração não ligada (fu)")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT, help="Checkpoint do modelo dinâmico")
    parser.add_argument("--output-prefix", type=str, default="literature", help="Prefixo para arquivos de saída")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)
    stats = dataset_summary(df)
    results = run_simulations(df, args.top, args.fu, args.checkpoint)
    export_results(args.output_prefix, stats, results)

    print("Resumo salvo em:")
    print(f"  - {OUTPUT_DIR / f'{args.output_prefix}_clearance_stats.json'}")
    if results:
        print(f"  - {OUTPUT_DIR / f'{args.output_prefix}_simulation_summary.csv'}")
    else:
        print("  - Nenhum resultado (lista vazia)")


if __name__ == "__main__":
    main()
