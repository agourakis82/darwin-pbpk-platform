#!/usr/bin/env python3
"""Constrói dataset NPZ para treinos PBPK com dados enriquecidos."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = BASE_DIR / "analysis" / "pbpk_parameters_wide_enriched_v3.csv"
EMB_PATH = BASE_DIR / "analysis" / "pbpk_chemberta_embeddings_enriched_v3.npz"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "pbpk_enriched"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "pbpk_enriched_v3.npz"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

PRIMARY_TARGETS = [
    "clearance_hepatic_l_h",
    "clearance_l_h",
    "microsome_hepatic_l_h",
]

AUX_TARGETS = [
    "microsome_clint_l_h",
    "fu_frac",
    "vd_l_kg",
    "bioavailability_frac",
]


if __name__ == "__main__":
    params_df = pd.read_csv(PARAMS_PATH)
    emb_data = np.load(EMB_PATH, allow_pickle=True)
    smiles_list = emb_data["smiles"].astype(str)
    embeddings = emb_data["embeddings"]
    emb_map = {smiles: vec for smiles, vec in zip(smiles_list, embeddings)}

    params_df["embedding"] = params_df["smiles"].map(emb_map)
    params_df = params_df.dropna(subset=["embedding"])

    # Selecionar alvo primário (primeiro não nulo)
    def resolve_target(row):
        for col in PRIMARY_TARGETS:
            val = row.get(col)
            if pd.notna(val):
                return float(val)
        return np.nan

    params_df["target_clearance"] = params_df.apply(resolve_target, axis=1)
    params_df = params_df.dropna(subset=["target_clearance"])

    embedding_array = np.stack(params_df["embedding"].to_numpy())
    target_array = params_df["target_clearance"].to_numpy(dtype=np.float32)

    aux_arrays = {}
    for col in AUX_TARGETS:
        if col in params_df.columns:
            aux_arrays[col] = params_df[col].to_numpy(dtype=np.float32)

    np.savez_compressed(
        OUTPUT_PATH,
        embeddings=embedding_array.astype(np.float32),
        target_clearance=target_array,
        smiles=params_df["smiles"].to_numpy(dtype=object),
        chembl_ids=params_df["chembl_id"].to_numpy(dtype=object),
        **aux_arrays,
    )

    manifest = {
        "source_parameters": str(PARAMS_PATH),
        "source_embeddings": str(EMB_PATH),
        "num_samples": int(len(params_df)),
        "embedding_dim": int(embedding_array.shape[1]),
        "primary_target": "target_clearance",
        "primary_target_cols": PRIMARY_TARGETS,
        "auxiliary_targets": [col for col in AUX_TARGETS if col in params_df.columns],
        "output": str(OUTPUT_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
