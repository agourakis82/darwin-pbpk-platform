#!/usr/bin/env python3
"""Complementa SMILES ausentes consultando PubChem (PubChemPy).

Entrada: analysis/pbpk_parameters_wide_enriched.csv (após ChEMBL).
Saída: analysis/pbpk_parameters_wide_enriched_v2.csv, com log.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import pubchempy as pcp
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pubchempy não instalado. Rode `pip install pubchempy`." ) from exc

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "analysis" / "pbpk_parameters_wide_enriched.csv"
OUTPUT_PATH = BASE_DIR / "analysis" / "pbpk_parameters_wide_enriched_v2.csv"
LOG_PATH = BASE_DIR / "analysis" / "pbpk_enrichment_pubchem_log.json"


CACHE_DIR = BASE_DIR / "analysis" / "cache_pubchem"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cached_smiles(query: str) -> Optional[str]:
    cache_file = CACHE_DIR / f"{query.replace('/', '_').replace(' ', '_')[:80]}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data.get("smiles")
    smiles = fetch_smiles(query)
    cache_file.write_text(json.dumps({"smiles": smiles}, ensure_ascii=False), encoding="utf-8")
    return smiles


def fetch_smiles(query: str) -> Optional[str]:
    try:
        compounds = pcp.get_compounds(query, "name")
        smiles = {c.canonical_smiles for c in compounds if c.canonical_smiles}
        if len(smiles) == 1:
            return smiles.pop()
    except Exception:
        return None
    return None


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    missing_mask = df["smiles"].isna() | df["smiles"].astype(str).str.strip().isin(["", "nan"])
    missing = df[missing_mask].copy()

    stats = defaultdict(int)
    filled = {}

    for idx, row in missing.iterrows():
        query = str(row.get("chembl_id", "")).strip()
        if not query:
            continue
        # não tente novamente se já é um CHEMBL (provável sem estrutura)
        if query.upper().startswith("CHEMBL"):
            stats["chembl_without_structure"] += 1
            continue
        smiles = cached_smiles(query)
        if smiles:
            filled[idx] = smiles
            stats["filled_pubchem"] += 1
        else:
            stats["unresolved"] += 1

    for idx, smiles in filled.items():
        df.at[idx, "smiles"] = smiles

    df.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "total_missing_initial": int(missing_mask.sum()),
        "filled_pubchem": stats.get("filled_pubchem", 0),
        "chembl_without_structure": stats.get("chembl_without_structure", 0),
        "unresolved": stats.get("unresolved", 0),
    }
    LOG_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Enriquecimento PubChem concluído:")
    print(json.dumps(summary, indent=2))
    print(f"Arquivo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
