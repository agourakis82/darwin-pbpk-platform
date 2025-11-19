#!/usr/bin/env python3
"""
Carrega e converte dados experimentais reais para validação do DynamicPBPKGNN
Criado: 2025-11-17
Autor: AI Assistant + Dr. Agourakis

Fontes de dados:
- /mnt/f/DARWIN_VALIDATION/datasets/real_clinical_pk_data.json
- /mnt/f/DARWIN_VALIDATION/datasets/ULTIMATE_DATASET_v1_normalized_with_smiles.json
- /mnt/f/DARWIN_VALIDATION/datasets/pkdb_extracted/
"""
from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import PBPK_ORGANS, NUM_ORGANS


def load_real_clinical_pk_data(json_path: Path) -> pd.DataFrame:
    """Carrega dados clínicos reais de PK"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    records = []
    for item in data:
        record = {
            'drug_name': item.get('drug_name', ''),
            'dose': item.get('dose', None),
            'dose_unit': item.get('dose_unit', 'mg'),
            'cmax_obs': item.get('cmax_obs', None),
            'auc_obs': item.get('auc_obs', None),
            'tmax_obs': item.get('tmax_obs', None),
            'CL_lit': item.get('CL_lit', None),
            'Vd_lit': item.get('Vd_lit', None),
            'half_life': item.get('half_life', None),
            'bioavailability': item.get('bioavailability', None),
            'fu': item.get('fu', None),
            'protein_binding': item.get('protein_binding', None),
            'source': item.get('source', ''),
            'reference': item.get('reference', ''),
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df


def load_ultimate_dataset(json_path: Path) -> pd.DataFrame:
    """Carrega dataset consolidado"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Assumir que é uma lista de dicionários ou um dicionário com lista
    if isinstance(data, dict):
        if 'drugs' in data:
            items = data['drugs']
        elif 'data' in data:
            items = data['data']
        else:
            items = list(data.values())[0] if data else []
    else:
        items = data

    records = []
    for item in items:
        if isinstance(item, dict):
            records.append(item)

    df = pd.DataFrame(records)
    return df


def load_pkdb_data(pkdb_dir: Path) -> pd.DataFrame:
    """Carrega dados do PKDB"""
    individuals_path = pkdb_dir / "individuals.csv"
    studies_path = pkdb_dir / "studies.csv"
    interventions_path = pkdb_dir / "interventions.csv"

    if not individuals_path.exists():
        return pd.DataFrame()

    individuals = pd.read_csv(individuals_path)
    studies = pd.read_csv(studies_path) if studies_path.exists() else pd.DataFrame()
    interventions = pd.read_csv(interventions_path) if interventions_path.exists() else pd.DataFrame()

    # Merge se necessário
    if not studies.empty and 'study_id' in individuals.columns:
        individuals = individuals.merge(studies, on='study_id', how='left', suffixes=('', '_study'))

    return individuals


def convert_to_pbpk_format(
    df: pd.DataFrame,
    source: str = "unknown"
) -> Dict[str, np.ndarray]:
    """
    Converte dados experimentais para formato PBPK

    Retorna dicionário com:
    - doses: array de doses
    - clearances_hepatic: array de clearances hepáticos (estimados)
    - clearances_renal: array de clearances renais (estimados)
    - partition_coeffs: array de Kp por órgão
    - compound_ids: array de IDs de compostos
    - metadata: dicionário com informações adicionais
    """
    # Filtrar registros com dados mínimos necessários
    required_cols = ['dose']
    available_cols = [col for col in required_cols if col in df.columns]

    if not available_cols:
        print(f"⚠️  Dataset {source} não tem colunas necessárias")
        return {}

    # Converter dose para mg (assumir que está em mg se não especificado)
    doses = []
    clearances_hepatic = []
    clearances_renal = []
    partition_coeffs = []
    compound_ids = []
    metadata = []

    for idx, row in df.iterrows():
        # Dose
        dose = row.get('dose', None)
        if pd.isna(dose) or dose is None:
            continue

        # Converter unidade se necessário
        dose_unit = row.get('dose_unit', 'mg')
        if dose_unit.lower() in ['g', 'gram', 'grams']:
            dose = dose * 1000  # converter para mg

        # Clearance total (CL_lit)
        cl_total = row.get('CL_lit', None)
        if pd.isna(cl_total) or cl_total is None:
            # Estimar a partir de half_life e Vd se disponível
            half_life = row.get('half_life', None)
            vd = row.get('Vd_lit', None)
            if not pd.isna(half_life) and not pd.isna(vd) and half_life > 0:
                cl_total = (0.693 * vd) / half_life  # CL = (ln(2) * Vd) / t1/2
            else:
                continue  # Pular se não temos clearance

        # Separar clearance hepático e renal (estimativa simples)
        # Assumir 70% hepático, 30% renal (aproximação comum)
        cl_hepatic = cl_total * 0.7
        cl_renal = cl_total * 0.3

        # Partition coefficients (Kp)
        # Estimar a partir de Vd se disponível
        vd = row.get('Vd_lit', None)
        if pd.isna(vd) or vd is None:
            # Usar valores padrão se Vd não disponível
            kp_values = np.ones(NUM_ORGANS) * 1.0
        else:
            # Estimar Kp médio a partir de Vd
            # Vd ≈ Vp + Vt * Kp_avg, onde Vp ≈ 3L (plasma), Vt ≈ 40L (tecido)
            vp = 3.0  # L (volume de plasma)
            vt = 40.0  # L (volume de tecido total)
            if vt > 0:
                kp_avg = (vd - vp) / vt
                kp_avg = max(0.1, min(10.0, kp_avg))  # Limitar entre 0.1 e 10
            else:
                kp_avg = 1.0

            # Distribuir Kp por órgão (valores relativos)
            # Liver e kidney têm maior perfusão, brain tem BBB
            kp_organs = {
                'blood': 1.0,
                'liver': kp_avg * 1.5,
                'kidney': kp_avg * 1.2,
                'brain': kp_avg * 0.3,  # BBB
                'heart': kp_avg * 1.0,
                'lung': kp_avg * 1.0,
                'muscle': kp_avg * 0.8,
                'adipose': kp_avg * 2.0,  # Lipofílico
                'gut': kp_avg * 1.0,
                'skin': kp_avg * 0.8,
                'bone': kp_avg * 0.5,
                'spleen': kp_avg * 1.0,
                'pancreas': kp_avg * 1.0,
                'other': kp_avg * 1.0,
            }

            kp_values = np.array([kp_organs.get(organ, kp_avg) for organ in PBPK_ORGANS])

        # Compound ID
        drug_name = row.get('drug_name', f'compound_{idx}')
        compound_id = str(drug_name).lower().replace(' ', '_')

        doses.append(dose)
        clearances_hepatic.append(cl_hepatic)
        clearances_renal.append(cl_renal)
        partition_coeffs.append(kp_values)
        compound_ids.append(compound_id)

        # Metadata
        metadata.append({
            'drug_name': drug_name,
            'source': source,
            'cmax_obs': row.get('cmax_obs', None),
            'auc_obs': row.get('auc_obs', None),
            'tmax_obs': row.get('tmax_obs', None),
            'half_life': row.get('half_life', None),
            'Vd_lit': row.get('Vd_lit', None),
            'CL_lit': row.get('CL_lit', None),
        })

    if len(doses) == 0:
        return {}

    return {
        'doses': np.array(doses, dtype=np.float32),
        'clearances_hepatic': np.array(clearances_hepatic, dtype=np.float32),
        'clearances_renal': np.array(clearances_renal, dtype=np.float32),
        'partition_coeffs': np.array(partition_coeffs, dtype=np.float32),
        'compound_ids': np.array(compound_ids, dtype=object),
        'metadata': metadata,
    }


def main():
    ap = argparse.ArgumentParser(description="Carrega dados experimentais para validação")
    ap.add_argument("--output", required=True, help="Caminho de saída (.npz)")
    ap.add_argument("--source", choices=["real_clinical", "ultimate", "pkdb", "all"], default="all",
                    help="Fonte de dados a carregar")
    args = ap.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_data = {
        'doses': [],
        'clearances_hepatic': [],
        'clearances_renal': [],
        'partition_coeffs': [],
        'compound_ids': [],
        'metadata': [],
    }

    # Carregar real_clinical_pk_data.json
    if args.source in ["real_clinical", "all"]:
        clinical_path = Path("/mnt/f/DARWIN_VALIDATION/datasets/real_clinical_pk_data.json")
        if clinical_path.exists():
            print(f"📊 Carregando {clinical_path}...")
            df = load_real_clinical_pk_data(clinical_path)
            print(f"   {len(df)} registros carregados")
            data = convert_to_pbpk_format(df, source="real_clinical")
            if data:
                for key in all_data.keys():
                    if key in data:
                        if isinstance(data[key], list):
                            all_data[key].extend(data[key])
                        else:
                            all_data[key].append(data[key])
                print(f"   ✅ {len(data['doses'])} compostos convertidos")

    # Carregar ULTIMATE_DATASET
    if args.source in ["ultimate", "all"]:
        ultimate_path = Path("/mnt/f/DARWIN_VALIDATION/datasets/ULTIMATE_DATASET_v1_normalized_with_smiles.json")
        if ultimate_path.exists():
            print(f"📊 Carregando {ultimate_path}...")
            df = load_ultimate_dataset(ultimate_path)
            print(f"   {len(df)} registros carregados")
            data = convert_to_pbpk_format(df, source="ultimate")
            if data:
                for key in all_data.keys():
                    if key in data:
                        if isinstance(data[key], list):
                            all_data[key].extend(data[key])
                        else:
                            all_data[key].append(data[key])
                print(f"   ✅ {len(data['doses'])} compostos convertidos")

    # Carregar PKDB
    if args.source in ["pkdb", "all"]:
        pkdb_path = Path("/mnt/f/DARWIN_VALIDATION/datasets/pkdb_extracted")
        if pkdb_path.exists():
            print(f"📊 Carregando {pkdb_path}...")
            df = load_pkdb_data(pkdb_path)
            if not df.empty:
                print(f"   {len(df)} registros carregados")
                data = convert_to_pbpk_format(df, source="pkdb")
                if data:
                    for key in all_data.keys():
                        if key in data:
                            if isinstance(data[key], list):
                                all_data[key].extend(data[key])
                            else:
                                all_data[key].append(data[key])
                    print(f"   ✅ {len(data['doses'])} compostos convertidos")

    # Consolidar dados
    if len(all_data['doses']) == 0:
        print("❌ Nenhum dado carregado!")
        return

    # Converter listas para arrays
    final_data = {}
    for key in ['doses', 'clearances_hepatic', 'clearances_renal', 'partition_coeffs']:
        if all_data[key]:
            if isinstance(all_data[key][0], np.ndarray):
                final_data[key] = np.concatenate(all_data[key])
            else:
                final_data[key] = np.array(all_data[key])

    # Compound IDs
    if all_data['compound_ids']:
        final_data['compound_ids'] = np.array(all_data['compound_ids'], dtype=object)

    # Metadata (salvar separadamente)
    metadata_path = output_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(all_data['metadata'], f, indent=2)

    # Salvar NPZ
    np.savez_compressed(output_path, **final_data)

    print(f"\n✅ Dataset experimental salvo:")
    print(f"   {output_path}")
    print(f"   {metadata_path}")
    print(f"\n📊 Estatísticas:")
    print(f"   Total de compostos: {len(final_data['doses'])}")
    print(f"   Doses: min={final_data['doses'].min():.2f}, max={final_data['doses'].max():.2f}, mean={final_data['doses'].mean():.2f} mg")
    print(f"   CL hepático: min={final_data['clearances_hepatic'].min():.2f}, max={final_data['clearances_hepatic'].max():.2f}, mean={final_data['clearances_hepatic'].mean():.2f} L/h")
    print(f"   CL renal: min={final_data['clearances_renal'].min():.2f}, max={final_data['clearances_renal'].max():.2f}, mean={final_data['clearances_renal'].mean():.2f} L/h")


if __name__ == "__main__":
    main()


