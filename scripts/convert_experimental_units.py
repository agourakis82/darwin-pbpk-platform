#!/usr/bin/env python3
"""
Corrige conversão de unidades em dados experimentais usando massa molar
Criado: 2025-11-17
Autor: AI Assistant + Dr. Agourakis

Converte:
- ng/mL → mg/L usando massa molar
- ng·h/mL → mg·h/L usando massa molar
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rdkit import Chem
from rdkit.Chem import Descriptors


def get_molecular_weight(smiles: Optional[str], drug_name: Optional[str] = None) -> Optional[float]:
    """
    Obtém massa molar a partir de SMILES ou nome do fármaco

    Returns:
        Massa molar em g/mol, ou None se não conseguir calcular
    """
    # Tentar SMILES primeiro
    if smiles and isinstance(smiles, str) and smiles.strip():
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            return float(mw)

    # Fallback: buscar SMILES por nome (simplificado - poderia usar PubChem API)
    # Por enquanto, retornar None e usar valores padrão
    return None


def convert_ng_per_ml_to_mg_per_l(value_ng_ml: float, molecular_weight: float = None) -> float:
    """
    Converte ng/mL para mg/L

    Fórmula: mg/L = (ng/mL) / 1000
    (Conversão direta de massa, não depende de MW)

    Exemplo:
    - Concentração = 100 ng/mL
    - mg/L = 100 / 1000 = 0.1 mg/L

    Nota: MW não é necessário para esta conversão (ng e mg são unidades de massa)
    """
    return value_ng_ml / 1000.0


def convert_ng_h_per_ml_to_mg_h_per_l(value_ng_h_ml: float, molecular_weight: float = None) -> float:
    """
    Converte ng·h/mL para mg·h/L

    Fórmula: mg·h/L = (ng·h/mL) / 1000
    (Conversão direta de massa, não depende de MW)
    """
    return value_ng_h_ml / 1000.0


def process_metadata_with_unit_conversion(
    metadata_path: Path,
    output_path: Path,
    smiles_column: Optional[str] = None,
) -> Dict:
    """
    Processa metadata e converte unidades usando massa molar
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    converted_metadata = []
    molecular_weights = []
    conversion_stats = {
        'total': len(metadata),
        'with_smiles': 0,
        'mw_calculated': 0,
        'mw_fallback': 0,
        'cmax_converted': 0,
        'auc_converted': 0,
    }

    for i, meta in enumerate(metadata):
        drug_name = meta.get('drug_name', f'compound_{i}')
        smiles = meta.get('smiles', meta.get('SMILES', None))

        # Obter massa molar
        mw = get_molecular_weight(smiles, drug_name)
        if mw:
            conversion_stats['mw_calculated'] += 1
            if smiles:
                conversion_stats['with_smiles'] += 1
        else:
            # Fallback: usar MW médio de fármacos (~400 g/mol)
            mw = 400.0
            conversion_stats['mw_fallback'] += 1

        molecular_weights.append(mw)

        # Converter Cmax se disponível
        cmax_obs = meta.get('cmax_obs', None)
        if cmax_obs is not None:
            try:
                cmax_ng_ml = float(cmax_obs)
                cmax_mg_l = convert_ng_per_ml_to_mg_per_l(cmax_ng_ml, mw)
                meta['cmax_obs_mg_l'] = cmax_mg_l
                meta['cmax_obs_original'] = cmax_ng_ml
                meta['cmax_unit_original'] = 'ng/mL'
                conversion_stats['cmax_converted'] += 1
            except (ValueError, TypeError):
                pass

        # Converter AUC se disponível
        auc_obs = meta.get('auc_obs', None)
        if auc_obs is not None:
            try:
                auc_ng_h_ml = float(auc_obs)
                auc_mg_h_l = convert_ng_h_per_ml_to_mg_h_per_l(auc_ng_h_ml, mw)
                meta['auc_obs_mg_h_l'] = auc_mg_h_l
                meta['auc_obs_original'] = auc_ng_h_ml
                meta['auc_unit_original'] = 'ng·h/mL'
                conversion_stats['auc_converted'] += 1
            except (ValueError, TypeError):
                pass

        # Adicionar massa molar ao metadata
        meta['molecular_weight'] = mw
        meta['smiles'] = smiles  # Garantir que SMILES está presente

        converted_metadata.append(meta)

    # Salvar metadata convertido
    with open(output_path, 'w') as f:
        json.dump(converted_metadata, f, indent=2)

    print(f"✅ Conversão de unidades concluída:")
    print(f"   Total de compostos: {conversion_stats['total']}")
    print(f"   Com SMILES: {conversion_stats['with_smiles']}")
    print(f"   MW calculado: {conversion_stats['mw_calculated']}")
    print(f"   MW fallback (400 g/mol): {conversion_stats['mw_fallback']}")
    print(f"   Cmax convertidos: {conversion_stats['cmax_converted']}")
    print(f"   AUC convertidos: {conversion_stats['auc_converted']}")
    print(f"   MW médio: {np.mean(molecular_weights):.1f} g/mol")
    print(f"   MW min/max: {np.min(molecular_weights):.1f} / {np.max(molecular_weights):.1f} g/mol")

    return {
        'metadata': converted_metadata,
        'molecular_weights': molecular_weights,
        'stats': conversion_stats,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Converte unidades experimentais usando massa molar")
    ap.add_argument("--metadata", required=True, help="Arquivo metadata original (.json)")
    ap.add_argument("--output", required=True, help="Arquivo metadata convertido (.json)")
    args = ap.parse_args()

    metadata_path = Path(args.metadata)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        print(f"❌ Arquivo não encontrado: {metadata_path}")
        return

    process_metadata_with_unit_conversion(metadata_path, output_path)


if __name__ == "__main__":
    main()

