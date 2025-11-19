#!/usr/bin/env python3
"""Normaliza o dataset TDC ADME para formato PBPK consolidado."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

PARAMETER_MAPPING: Dict[str, str] = {
    "clearance_hepatocyte": "clearance_hepatocyte_ul_min_per_1e6cells",
    "clearance_microsome": "clearance_microsome_ul_min_mg",
    "vdss_lombardo": "volume_distribution_l_kg",
    "half_life_obach": "half_life_hours",
    "bioavailability_ma": "bioavailability_fraction",
    "ppbr_az": "fraction_unbound",
    "caco2_wang": "caco2_permeability_log",
}

UNITS_HINT: Dict[str, str] = {
    "clearance_hepatocyte_ul_min_per_1e6cells": "uL/min/10^6 cells",
    "clearance_microsome_ul_min_mg": "uL/min/mg protein",
    "volume_distribution_l_kg": "L/kg",
    "half_life_hours": "hours",
    "bioavailability_fraction": "fraction (0-1)",
    "fraction_unbound": "fraction unbound (0-1)",
    "caco2_permeability_log": "log10(cm/s)",
}


def normalize_fraction(value: float) -> float:
    if value is None:
        return None
    if value > 1.0:
        return value / 100.0
    return value


def scale_values(parameter: str, series: pd.Series) -> pd.Series:
    if parameter in {"bioavailability_fraction", "fraction_unbound"}:
        return series.apply(normalize_fraction)
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description="Normaliza dataset TDC ADME para PBPK")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external_datasets/tdc_adme_complete.csv"),
        help="Arquivo consolidado do TDC",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external_datasets/tdc_pbpk_parameters.parquet"),
        help="Arquivo parquet de saída",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis/tdc_parameter_summary.csv"),
        help="Resumo estatístico por parâmetro",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    df = pd.read_csv(args.input)
    value_col = "Y" if "Y" in df.columns else "value"
    records = []

    for source, group in df.groupby("source"):
        if source not in PARAMETER_MAPPING:
            continue
        parameter = PARAMETER_MAPPING[source]
        series = pd.to_numeric(group[value_col], errors="coerce")
        series = scale_values(parameter, series)
        valid = series.notna()
        if not valid.any():
            continue
        sub = group.loc[valid, ["Drug_ID", "Drug" if "Drug" in group.columns else "Drug_ID"]].copy()
        sub["parameter"] = parameter
        sub["value"] = series.loc[valid]
        sub.rename(columns={"Drug": "drug_name"}, inplace=True)
        records.append(sub)

    if not records:
        raise RuntimeError("Nenhum dado válido encontrado no dataset TDC.")

    long_df = pd.concat(records, ignore_index=True)

    summary_df = (
        long_df.groupby("parameter")
        .agg(
            count=("value", "count"),
            mean=("value", "mean"),
            median=("value", "median"),
            std=("value", "std"),
            min=("value", "min"),
            max=("value", "max"),
        )
        .reset_index()
    )
    summary_df["unit"] = summary_df["parameter"].map(UNITS_HINT)
    summary_df.to_csv(args.summary, index=False)

    pivot_df = (
        long_df.pivot_table(
            index=["Drug_ID", "drug_name"],
            columns="parameter",
            values="value",
            aggfunc="median",
        )
        .reset_index()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pivot_df.to_parquet(args.output, index=False)

    print(f"✅ Dataset PBPK salvo em {args.output} ({len(pivot_df)} compostos)")
    print(f"📈 Resumo estatístico atualizado: {args.summary}")


if __name__ == "__main__":
    main()
