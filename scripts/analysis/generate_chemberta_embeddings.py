#!/usr/bin/env python3
"""Gera embeddings ChemBERTa para conjuntos PBPK consolidados."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from apps.pbpk_core.ml.multimodal.chemberta_encoder import ChemBERTaEncoder

OUTPUT_DIR = BASE_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_smiles(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise ValueError(f"Coluna 'smiles' não encontrada em {path}")
    smiles = df["smiles"].dropna().astype(str).str.strip()
    smiles = [s for s in smiles if s and s.lower() != "nan" and not s.startswith("(")]
    return sorted(set(smiles))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera embeddings ChemBERTa para PBPK")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "analysis" / "pbpk_parameters_wide.csv")
    parser.add_argument("--output-prefix", type=str, default="pbpk_chemberta_embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    smiles = load_smiles(args.input)
    if not smiles:
        raise RuntimeError(f"Nenhum SMILES válido encontrado em {args.input}")

    print(f"Encontrados {len(smiles)} SMILES únicos. Gerando embeddings...")
    encoder = ChemBERTaEncoder()
    start = time.time()
    embeddings = encoder.encode_batch(smiles, batch_size=args.batch_size)
    elapsed = time.time() - start

    np.savez_compressed(
        OUTPUT_DIR / f"{args.output_prefix}.npz",
        smiles=np.array(smiles, dtype=object),
        embeddings=embeddings,
    )

    metadata = {
        "model": encoder.model_name,
        "embedding_dim": int(encoder.get_embedding_dim()),
        "device": encoder.device,
        "num_smiles": len(smiles),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "input_file": str(args.input),
        "output_prefix": args.output_prefix,
        "batch_size": args.batch_size,
    }
    (OUTPUT_DIR / f"{args.output_prefix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Embeddings salvos em {OUTPUT_DIR / f'{args.output_prefix}.npz'}")
    print("Metadata:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
