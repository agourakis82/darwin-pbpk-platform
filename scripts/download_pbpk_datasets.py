#!/usr/bin/env python3
"""
📥 DOWNLOAD E CONSOLIDAÇÃO DE DATASETS PBPK
============================================

Baixa e consolida múltiplos datasets públicos de PBPK:
1. TDC (Therapeutics Data Commons) - 17k moléculas
2. ChEMBL - ADME/PBPK properties
3. DrugBank - 100+ drugs com PK data
4. PK-DB - Concentration-time curves
5. HTTK - High-throughput toxicokinetics

Objetivo: Expandir de 478 → 5000+ moléculas

Author: Dr. Demetrios Chiuratto Agourakis
Date: October 28, 2025
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import json
import requests
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = BASE_DIR / 'data' / 'external_datasets'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("📥 DOWNLOAD DE DATASETS PBPK")
print("="*80)


# ==================== TDC (THERAPEUTICS DATA COMMONS) ====================

def download_tdc_adme():
    """Download TDC ADME datasets"""
    print("\n1️⃣ TDC (Therapeutics Data Commons)...")

    try:
        from tdc.single_pred import ADME

        datasets_to_download = [
            'caco2_wang',           # Permeability
            'bioavailability_ma',   # Bioavailability
            'vdss_lombardo',        # Volume of distribution
            'clearance_microsome',  # Clearance
            'clearance_hepatocyte', # Hepatic clearance
            'half_life_obach',      # Half-life
            'ppbr_az',              # Plasma protein binding
        ]

        all_data = []

        for dataset_name in datasets_to_download:
            try:
                print(f"\n  Downloading {dataset_name}...")
                data = ADME(name=dataset_name)
                df = data.get_data()

                # Add dataset source
                df['source'] = dataset_name
                df['dataset'] = 'TDC'

                print(f"    ✓ {len(df)} compounds")
                all_data.append(df)

            except Exception as e:
                print(f"    ⚠️  Failed: {e}")
                continue

        if len(all_data) > 0:
            # Combine all TDC datasets
            tdc_df = pd.concat(all_data, ignore_index=True)

            # Save
            output_path = OUTPUT_DIR / 'tdc_adme_complete.csv'
            tdc_df.to_csv(output_path, index=False)

            print(f"\n  ✅ TDC saved: {len(tdc_df)} total entries")
            print(f"     Unique compounds: {tdc_df['Drug_ID'].nunique() if 'Drug_ID' in tdc_df.columns else 'N/A'}")
            print(f"     File: {output_path}")

            return tdc_df
        else:
            print("  ⚠️  No TDC data downloaded")
            return None

    except ImportError:
        print("  ⚠️  PyTDC not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyTDC", "-q"])
        print("  ✓ PyTDC installed. Please re-run the script.")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


# ==================== CHEMBL ====================

def download_chembl_adme():
    """Download ChEMBL ADME data via API"""
    print("\n2️⃣ ChEMBL ADME Data...")

    try:
        from chembl_webresource_client.new_client import new_client

        activity = new_client.activity
        query_specs = [
            {
                "label": "clearance",
                "standard_types": ["Clearance", "Intrinsic Clearance"],
                "units_hint": "(L/h|l/min|ml/min|uL/min)",
            },
            {
                "label": "volume_distribution",
                "standard_types": ["Volume of distribution", "VDss"],
                "units_hint": "(L/kg|L)",
            },
            {
                "label": "half_life",
                "standard_types": ["Half-life", "Half life"],
                "units_hint": "(h|hr|hours|min)",
            },
            {
                "label": "bioavailability",
                "standard_types": ["Bioavailability"],
                "units_hint": "(%)",
            },
            {
                "label": "plasma_protein_binding",
                "standard_types": ["Plasma protein binding"],
                "units_hint": "(%)",
            },
            {
                "label": "caco2_permeability",
                "standard_types": ["Permeability", "Caco-2 permeability"],
                "units_hint": "(cm/s)",
            },
        ]

        frames = []

        for spec in query_specs:
            print(f"\n  Querying {spec['label']} ...")
            collected = []
            for stype in spec["standard_types"]:
                try:
                    results = activity.filter(
                        standard_type__iexact=stype,
                        standard_value__isnull=False,
                                                molecule_chembl_id__isnull=False,
                    ).only([
                        'molecule_chembl_id',
                        'canonical_smiles',
                        'standard_type',
                        'standard_value',
                        'standard_units',
                        'assay_type'
                    ])

                    rows = list(results[:2000])
                    if rows:
                        df = pd.DataFrame(rows)
                        df["query_label"] = spec["label"]
                        df["query_standard_type"] = stype
                        df["units_hint"] = spec.get("units_hint")
                        df["dataset"] = "ChEMBL"
                        collected.append(df)
                        print(f"    ✓ {len(df)} entries for standard_type={stype!r}")
                    else:
                        print(f"    ⚠️  No entries for standard_type={stype!r}")
                except Exception as inner_exc:
                    print(f"    ⚠️  Failed standard_type={stype!r}: {inner_exc}")
                    continue

            if collected:
                frames.append(pd.concat(collected, ignore_index=True))
            else:
                print(f"  ⚠️  No data gathered for label={spec['label']}")

        if frames:
            chembl_df = pd.concat(frames, ignore_index=True)
            output_path = OUTPUT_DIR / 'chembl_adme_complete.csv'
            chembl_df.to_csv(output_path, index=False)
            print(f"\n  ✅ ChEMBL saved: {len(chembl_df)} total entries")
            print(f"     Unique compounds: {chembl_df['molecule_chembl_id'].nunique()}")
            print(f"     File: {output_path}")
            return chembl_df

        print("  ⚠️  No ChEMBL data downloaded")
        return None

    except ImportError:
        print("  ⚠️  ChEMBL client not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chembl_webresource_client", "-q"])
        print("  ✓ ChEMBL client installed. Please re-run the script.")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


# ==================== DRUGBANK (LOCAL) ====================

def load_drugbank_local():
    """Load DrugBank data from local storage"""
    print("\n3️⃣ DrugBank (Local)...")

    drugbank_paths = [
        Path('/mnt/f/datasets/pbpk/drugbank'),
        BASE_DIR / 'data' / 'external' / 'drugbank'
    ]

    for path in drugbank_paths:
        if path.exists():
            print(f"  Found: {path}")

            # Look for CSV/JSON files
            files = list(path.glob('*.csv')) + list(path.glob('*.json'))

            if len(files) > 0:
                print(f"  Files found: {len(files)}")

                all_data = []
                for file in files:
                    try:
                        if file.suffix == '.csv':
                            df = pd.read_csv(file)
                        elif file.suffix == '.json':
                            df = pd.read_json(file)
                        else:
                            continue

                        df['source_file'] = file.name
                        df['dataset'] = 'DrugBank'
                        all_data.append(df)
                        print(f"    ✓ {file.name}: {len(df)} entries")

                    except Exception as e:
                        print(f"    ⚠️  {file.name}: {e}")
                        continue

                if len(all_data) > 0:
                    drugbank_df = pd.concat(all_data, ignore_index=True)

                    # Save
                    output_path = OUTPUT_DIR / 'drugbank_complete.csv'
                    drugbank_df.to_csv(output_path, index=False)

                    print(f"\n  ✅ DrugBank saved: {len(drugbank_df)} entries")
                    print(f"     File: {output_path}")

                    return drugbank_df

    print("  ⚠️  DrugBank data not found")
    return None


# ==================== VALIDATION SET (LOCAL) ====================

def load_validation_100():
    """Load validation_drugs_100.json"""
    print("\n4️⃣ Validation Set (100 drugs)...")

    val_paths = [
        Path('/mnt/f/datasets/pbpk/validation_drugs_100.json'),
        BASE_DIR / 'data' / 'external' / 'validation_drugs_100.json'
    ]

    for path in val_paths:
        if path.exists():
            print(f"  Found: {path}")

            try:
                with open(path, 'r') as f:
                    drugs = json.load(f)

                # Convert to DataFrame
                data_list = []
                for drug in drugs:
                    if 'pk_parameters' in drug and 'smiles' in drug:
                        pk = drug['pk_parameters']
                        data_list.append({
                            'drug_name': drug.get('drug_name', 'Unknown'),
                            'smiles': drug['smiles'],
                            'fu': pk.get('fu'),
                            'vd': pk.get('vd'),
                            'clearance': pk.get('clearance'),
                            'half_life': pk.get('half_life'),
                            'cmax': pk.get('cmax'),
                            'tmax': pk.get('tmax'),
                            'auc': pk.get('auc'),
                            'dataset': 'Validation_100'
                        })

                if len(data_list) > 0:
                    val_df = pd.DataFrame(data_list)

                    # Save
                    output_path = OUTPUT_DIR / 'validation_100_complete.csv'
                    val_df.to_csv(output_path, index=False)

                    print(f"  ✅ Validation set saved: {len(val_df)} drugs")
                    print(f"     Fu: {val_df['fu'].notna().sum()}")
                    print(f"     Vd: {val_df['vd'].notna().sum()}")
                    print(f"     CL: {val_df['clearance'].notna().sum()}")
                    print(f"     File: {output_path}")

                    return val_df

            except Exception as e:
                print(f"  ❌ Error: {e}")

    print("  ⚠️  Validation set not found")
    return None


# ==================== PUBCHEM ====================

def download_pubchem_sample():
    """Download sample from PubChem"""
    print("\n5️⃣ PubChem Sample (1000 FDA drugs)...")

    try:
        import pubchempy as pcp

        print("  Querying PubChem for FDA-approved drugs...")

        # Search for FDA-approved drugs
        compounds = pcp.get_compounds('fda approved', 'name', listkey_count=1000)

        data_list = []
        for compound in compounds[:1000]:
            try:
                data_list.append({
                    'cid': compound.cid,
                    'smiles': compound.canonical_smiles,
                    'iupac_name': compound.iupac_name,
                    'molecular_formula': compound.molecular_formula,
                    'molecular_weight': compound.molecular_weight,
                    'dataset': 'PubChem_FDA'
                })
            except:
                continue

        if len(data_list) > 0:
            pubchem_df = pd.DataFrame(data_list)

            # Save
            output_path = OUTPUT_DIR / 'pubchem_fda_sample.csv'
            pubchem_df.to_csv(output_path, index=False)

            print(f"  ✅ PubChem saved: {len(pubchem_df)} compounds")
            print(f"     File: {output_path}")

            return pubchem_df

    except ImportError:
        print("  ⚠️  PubChemPy not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pubchempy", "-q"])
        print("  ✓ PubChemPy installed. Please re-run the script.")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


# ==================== CONSOLIDATION ====================

def consolidate_datasets():
    """Consolidate all downloaded datasets"""
    print("\n" + "="*80)
    print("📊 CONSOLIDATING DATASETS")
    print("="*80)

    # Load all available datasets
    datasets = []
    dataset_names = []

    files = list(OUTPUT_DIR.glob('*.csv'))

    if len(files) == 0:
        print("⚠️  No datasets downloaded yet")
        return None

    for file in files:
        try:
            df = pd.read_csv(file)
            datasets.append(df)
            dataset_names.append(file.stem)
            print(f"  ✓ Loaded {file.name}: {len(df)} entries")
        except Exception as e:
            print(f"  ⚠️  Failed to load {file.name}: {e}")

    if len(datasets) == 0:
        print("❌ No datasets loaded")
        return None

    # Combine all datasets
    print("\n📦 Combining datasets...")
    combined_df = pd.concat(datasets, ignore_index=True)

    print(f"  Total entries: {len(combined_df)}")

    # Standardize column names
    column_mapping = {
        'Drug': 'smiles',
        'SMILES': 'smiles',
        'Smiles': 'smiles',
        'canonical_smiles': 'smiles',
        'Y': 'value',
        'value': 'value',
        'standard_value': 'value',
    }

    combined_df = combined_df.rename(columns=column_mapping)

    # Remove duplicates by SMILES
    if 'smiles' in combined_df.columns:
        initial_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['smiles'], keep='first')
        print(f"  Removed {initial_len - len(combined_df)} duplicates")
        print(f"  Unique compounds: {len(combined_df)}")

    # Save consolidated dataset
    output_path = OUTPUT_DIR / 'consolidated_pbpk_dataset.csv'
    combined_df.to_csv(output_path, index=False)

    print(f"\n✅ Consolidated dataset saved: {output_path}")
    print(f"   Total compounds: {len(combined_df)}")

    # Summary statistics
    print("\n📊 Dataset Summary:")
    if 'dataset' in combined_df.columns:
        dataset_counts = combined_df['dataset'].value_counts()
        for dataset, count in dataset_counts.items():
            print(f"  {dataset:20s}: {count:6d} compounds")

    # Check PBPK parameters
    pbpk_params = ['fu', 'vd', 'clearance', 'half_life', 'bioavailability']
    available_params = [p for p in pbpk_params if p in combined_df.columns]

    if len(available_params) > 0:
        print("\n🎯 PBPK Parameters Available:")
        for param in available_params:
            n_available = combined_df[param].notna().sum()
            pct = (n_available / len(combined_df)) * 100
            print(f"  {param:20s}: {n_available:6d} ({pct:5.1f}%)")

    return combined_df


# ==================== MAIN ====================

def main():
    """Main execution"""

    print("\n🚀 Starting dataset download...")
    print("="*80)

    # Download each dataset
    tdc_df = download_tdc_adme()
    chembl_df = download_chembl_adme()
    drugbank_df = load_drugbank_local()
    val_df = load_validation_100()
    pubchem_df = download_pubchem_sample()

    # Consolidate
    consolidated_df = consolidate_datasets()

    # Final summary
    print("\n" + "="*80)
    print("🎉 DOWNLOAD COMPLETE!")
    print("="*80)

    successful_downloads = sum([
        tdc_df is not None,
        chembl_df is not None,
        drugbank_df is not None,
        val_df is not None,
        pubchem_df is not None
    ])

    print(f"\n✅ Successfully downloaded: {successful_downloads}/5 datasets")

    if consolidated_df is not None:
        print(f"✅ Total unique compounds: {len(consolidated_df)}")
        print(f"✅ Output directory: {OUTPUT_DIR}")

        print("\n🎯 Next Steps:")
        print("1. Review consolidated dataset")
        print("2. Generate embeddings for new compounds")
        print("3. Merge with existing KEC dataset")
        print("4. Re-train models with expanded data")
    else:
        print("\n⚠️  No datasets consolidated. Check errors above.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

