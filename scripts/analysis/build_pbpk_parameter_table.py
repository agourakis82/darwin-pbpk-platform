#!/usr/bin/env python3
"""Consolida parâmetros PBPK a partir dos datasets TDC e ChEMBL.

Saídas principais:
- analysis/pbpk_parameters_long.csv  (formato longo, cada parâmetro por linha)
- analysis/pbpk_parameters_wide.csv  (pivot consolidado por composto)

Conversões implementadas:
- clearance hepatocitário (µL/min/10^6 células) → L/h via modelo well-stirred
- fração não ligada (fu) derivada de ppbr (percentual ligado)
- normalização de unidades ChEMBL para L/h, L/kg, frações, etc.

Autor: AI Assistant (Darwin Workspace)
Data: 2025-11-12
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
TDC_PATH = BASE_DIR / "data" / "external_datasets" / "tdc_adme_complete.csv"
CHEMBL_PATH = BASE_DIR / "data" / "external_datasets" / "chembl_adme_complete.csv"
OUTPUT_DIR = BASE_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEPATO_PER_GRAM = 1.2e8  # células por grama de fígado
LIVER_WEIGHT_G = 1500.0  # g
HEPATIC_BLOOD_FLOW_L_H = 90.0  # L/h
MICROSOMAL_PROTEIN_MG_PER_G_LIVER = 40.0  # mg de proteína microsomal por g de fígado
DEFAULT_FU = 0.5


# ---------------------------------------------------------------------------
# Conversões auxiliares
# ---------------------------------------------------------------------------

def clearance_cell_to_clint_lph(value: float) -> float:
    """µL/min/10^6 células → L/h."""
    cl_l_min = value * 1e-6 * HEPATO_PER_GRAM * LIVER_WEIGHT_G
    return cl_l_min * 60.0


def well_stirred_clearance(clint_lph: float, fu: float = DEFAULT_FU) -> float:
    return (HEPATIC_BLOOD_FLOW_L_H * fu * clint_lph) / (
        HEPATIC_BLOOD_FLOW_L_H + fu * clint_lph
    )


def microsome_clearance_to_clint_lph(value: float) -> float:
    """µL/min/mg proteína → L/h (assume 40 mg/g e fígado 1.5 kg)."""
    cl_l_min = value * 1e-6 * MICROSOMAL_PROTEIN_MG_PER_G_LIVER * LIVER_WEIGHT_G
    return cl_l_min * 60.0


def convert_clearance_units(value: float, units: str) -> Optional[float]:
    u = str(units).lower() if isinstance(units, str) else str(units or "").lower()
    if "l/h" in u or "l/hr" in u:
        return value
    if "ml/min" in u:
        return (value / 1000.0) * 60.0
    if "ul/min" in u:
        return (value / 1e6) * 60.0
    if "l/min" in u:
        return value * 60.0
    return None


def convert_volume_units(value: float, units: str) -> Optional[float]:
    u = str(units).lower() if isinstance(units, str) else str(units or "").lower()
    if "l/kg" in u:
        return value
    if "ml/kg" in u:
        return value / 1000.0
    if "l" == u.strip():
        return value
    return None


def convert_half_life_units(value: float, units: str) -> Optional[float]:
    u = str(units).lower() if isinstance(units, str) else str(units or "").lower()
    if "h" in u:
        return value
    if "min" in u:
        return value / 60.0
    if "s" in u:
        return value / 3600.0
    return None


def convert_fraction(value: float, units: str) -> Optional[float]:
    if value is None or math.isnan(value):
        return None
    u = str(units).lower() if isinstance(units, str) else str(units or "").lower()
    if "%" in u and value > 1:
        return value / 100.0
    if value <= 1.0:
        return value
    return None


def convert_ppb_to_fu(percent_bound: float) -> Optional[float]:
    if percent_bound is None or math.isnan(percent_bound):
        return None
    if percent_bound > 1.0:
        return max(min(1.0 - percent_bound / 100.0, 1.0), 0.0)
    return percent_bound


# ---------------------------------------------------------------------------
# Carregamento e transformação TDC
# ---------------------------------------------------------------------------

TDC_PARAM_MAP: Dict[str, List[Tuple[str, str]]] = {
    "clearance_hepatocyte": [
        ("clearance_cell_uL_min_1e6", "µL/min/10^6 cells"),
        ("clearance_hepatic_l_h", "L/h"),
    ],
    "clearance_microsome": [
        ("microsome_clearance_uL_min_mg", "µL/min/mg protein"),
        ("microsome_clint_l_h", "L/h"),
        ("microsome_hepatic_l_h", "L/h"),
    ],
    "vdss_lombardo": [
        ("vd_l_kg", "L/kg"),
    ],
    "half_life_obach": [
        ("half_life_h", "hours"),
    ],
    "bioavailability_ma": [
        ("bioavailability_frac", "fraction"),
    ],
    "ppbr_az": [
        ("fu_frac", "fraction"),
    ],
    "caco2_wang": [
        ("caco2_log_permeability", "log10(cm/s)"),
    ],
}


def load_tdc_records() -> List[Dict[str, object]]:
    if not TDC_PATH.exists():
        return []

    df = pd.read_csv(TDC_PATH)
    records: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        source = row["source"]
        if source not in TDC_PARAM_MAP:
            continue
        value = pd.to_numeric(row.get("Y"), errors="coerce")
        if pd.isna(value):
            continue

        chembl_id = row.get("Drug_ID") if isinstance(row.get("Drug_ID"), str) else None
        drug_name = row.get("Drug")

        for param, unit in TDC_PARAM_MAP[source]:
            if param == "clearance_cell_uL_min_1e6":
                final_value = value
            elif param == "clearance_hepatic_l_h":
                clint = clearance_cell_to_clint_lph(value)
                final_value = well_stirred_clearance(clint)
            elif param == "microsome_clint_l_h":
                final_value = microsome_clearance_to_clint_lph(value)
            elif param == "microsome_hepatic_l_h":
                clint = microsome_clearance_to_clint_lph(value)
                final_value = well_stirred_clearance(clint)
            elif param == "bioavailability_frac":
                final_value = value if value <= 1.0 else value / 100.0
            elif param == "fu_frac":
                final_value = convert_ppb_to_fu(value)
            else:
                final_value = value

            if final_value is None or (isinstance(final_value, float) and math.isnan(final_value)):
                continue

            records.append(
                {
                    "chembl_id": chembl_id,
                    "drug_name": drug_name,
                    "parameter": param,
                    "value": float(final_value),
                    "unit": unit,
                    "dataset": "TDC",
                }
            )

    return records


# ---------------------------------------------------------------------------
# Carregamento e transformação ChEMBL
# ---------------------------------------------------------------------------

CHEMBL_PARAM_MAP: Dict[str, str] = {
    "clearance": "clearance_l_h",
    "volume_distribution": "vd_l_kg",
    "half_life": "half_life_h",
    "bioavailability": "bioavailability_frac",
    "plasma_protein_binding": "fu_frac",  # convertido para fu quando possível
    "caco2_permeability": "caco2_permeability_cm_s",
}


def load_chembl_records() -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    if not CHEMBL_PATH.exists():
        return [], pd.DataFrame()

    df = pd.read_csv(CHEMBL_PATH)
    records: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        label = row.get("query_label")
        if label not in CHEMBL_PARAM_MAP:
            continue

        value = pd.to_numeric(row.get("standard_value"), errors="coerce")
        if pd.isna(value):
            continue

        units = row.get("standard_units") or ""
        param = CHEMBL_PARAM_MAP[label]
        converted: Optional[float] = None

        if param == "clearance_l_h":
            converted = convert_clearance_units(value, units)
        elif param == "vd_l_kg":
            converted = convert_volume_units(value, units)
        elif param == "half_life_h":
            converted = convert_half_life_units(value, units)
        elif param == "bioavailability_frac":
            converted = convert_fraction(value, units)
        elif param == "fu_frac":
            converted = convert_ppb_to_fu(value if "%" in units else 1 - value)
        elif param == "caco2_permeability_cm_s":
            converted = value

        if converted is None:
            continue

        records.append(
            {
                "chembl_id": row.get("molecule_chembl_id"),
                "drug_name": None,
                "parameter": param,
                "value": float(converted),
                "unit": "normalized",
                "dataset": "ChEMBL",
            }
        )

    smiles_lookup = (
        df[["molecule_chembl_id", "canonical_smiles"]]
        .drop_duplicates()
        .rename(columns={"molecule_chembl_id": "chembl_id", "canonical_smiles": "smiles"})
    )

    return records, smiles_lookup


# ---------------------------------------------------------------------------
# Consolidação
# ---------------------------------------------------------------------------

def build_long_table() -> pd.DataFrame:
    records = load_tdc_records()
    chembl_records, smiles_lookup = load_chembl_records()
    records.extend(chembl_records)

    long_df = pd.DataFrame(records)
    if not long_df.empty:
        long_df = long_df.dropna(subset=["value"]).reset_index(drop=True)

    long_path = OUTPUT_DIR / "pbpk_parameters_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Long table -> {long_path} ({len(long_df)} linhas)")

    return long_df, smiles_lookup


def build_wide_table(long_df: pd.DataFrame, smiles_lookup: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        raise RuntimeError("Tabela longa vazia; nada para consolidar.")

    # Agrupar por composto e parâmetro (média das medições)
    pivot_df = (
        long_df.groupby(["chembl_id", "parameter"]) ["value"]
        .mean()
        .unstack()
        .reset_index()
    )

    # Incorporar SMILES quando disponíveis
    if not smiles_lookup.empty:
        pivot_df = pivot_df.merge(smiles_lookup, on="chembl_id", how="left")

    wide_path = OUTPUT_DIR / "pbpk_parameters_wide.csv"
    pivot_df.to_csv(wide_path, index=False)
    print(f"Wide table -> {wide_path} ({len(pivot_df)} compostos)")

    return pivot_df


def main() -> None:
    long_df, smiles_lookup = build_long_table()
    if long_df.empty:
        print("⚠️ Nenhum registro consolidado.")
        return

    build_wide_table(long_df, smiles_lookup)
    print("✅ Consolidação concluída.")


if __name__ == "__main__":
    main()
