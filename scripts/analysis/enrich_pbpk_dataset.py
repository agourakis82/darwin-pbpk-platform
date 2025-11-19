#!/usr/bin/env python3
"""Enriquece `analysis/pbpk_parameters_wide.csv` com SMILES e nomes oficiais.

- Tenta preencher SMILES ausentes consultando a API do ChEMBL.
- Para compostos sem CHEMBL ID, usa busca por nome (limitada a resultados únicos).
- Gera arquivo `analysis/pbpk_parameters_wide_enriched.csv` e log resumido.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    from chembl_webresource_client.new_client import new_client
except ImportError as exc:  # pragma: no cover
    raise SystemExit("chembl_webresource_client não instalado. Rodar `pip install chembl_webresource_client`." ) from exc

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "analysis" / "pbpk_parameters_wide.csv"
OUTPUT_PATH = BASE_DIR / "analysis" / "pbpk_parameters_wide_enriched.csv"
LOG_PATH = BASE_DIR / "analysis" / "pbpk_enrichment_log.json"

MAX_QUERY = 2000


molecule = new_client.molecule


def fetch_smiles_for_id(chembl_id: str) -> Optional[str]:
    try:
        record = molecule.get(chembl_id)
        if record and record.get("molecule_structures"):
            return record["molecule_structures"].get("canonical_smiles")
    except Exception:
        return None
    return None


def search_smiles_by_name(name: str) -> Optional[str]:
    try:
        res = molecule.search(name)
        hits = list(res[:5])
        smiles = {hit.get("molecule_structures", {}).get("canonical_smiles") for hit in hits if hit.get("molecule_structures")}
        smiles = {s for s in smiles if s}
        if len(smiles) == 1:
            return smiles.pop()
    except Exception:
        return None
    return None


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    missing_mask = df["smiles"].isna() | df["smiles"].astype(str).str.strip().isin(["", "nan"])
    missing = df[missing_mask].copy()

    enriched = {}
    stats: Dict[str, int] = defaultdict(int)

    for _, row in missing.iterrows():
        chembl_id = row.get("chembl_id")
        drug_name = row.get("drug_name") if "drug_name" in row else None
        smiles = None

        if isinstance(chembl_id, str) and chembl_id.upper().startswith("CHEMBL"):
            smiles = fetch_smiles_for_id(chembl_id)
            if smiles:
                stats["filled_by_id"] += 1
        if not smiles and isinstance(drug_name, str) and drug_name:
            smiles = search_smiles_by_name(drug_name)
            if smiles:
                stats["filled_by_name"] += 1

        if smiles:
            enriched[(row.name)] = smiles
        else:
            stats["unresolved"] += 1

    for idx, smiles in enriched.items():
        df.at[idx, "smiles"] = smiles

    df.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "total_missing_initial": int(missing_mask.sum()),
        "filled_by_id": stats.get("filled_by_id", 0),
        "filled_by_name": stats.get("filled_by_name", 0),
        "unresolved": stats.get("unresolved", 0),
    }
    LOG_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Enriquecimento concluído:")
    print(json.dumps(summary, indent=2))
    print(f"Arquivo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
