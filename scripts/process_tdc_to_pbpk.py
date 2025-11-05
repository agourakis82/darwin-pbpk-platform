#!/usr/bin/env python3
"""
🔄 PROCESSAR TDC → PBPK FORMAT
===============================

Converte dados TDC ADME para formato PBPK compatível com KEC:
- PPB → Fu (fraction unbound)
- Vdss → Vd (volume of distribution)
- Clearance (microsome/hepatocyte) → CL (in vivo)

Author: Dr. Demetrios Chiuratto Agourakis
Date: October 28, 2025
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Paths
TDC_DATA = BASE_DIR / 'data' / 'external_datasets' / 'tdc_adme_complete.csv'
OUTPUT_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🔄 PROCESSANDO TDC → PBPK")
print("="*80)


# ==================== CONVERSION FUNCTIONS ====================

def ppbr_to_fu(ppbr_log):
    """
    Convert PPBR (Plasma Protein Binding Ratio) to Fu
    
    TDC ppbr_az: log10(% bound)
    Fu = fraction unbound = (100 - % bound) / 100
    """
    # Clip log value to avoid overflow
    ppbr_log = np.clip(ppbr_log, -2, 2)  # 1% to 100%
    ppbr_pct = 10 ** ppbr_log  # % bound
    ppbr_pct = np.clip(ppbr_pct, 0, 100)  # Clip to valid range
    fu = (100 - ppbr_pct) / 100
    fu = np.clip(fu, 0.001, 0.999)  # Avoid extreme values
    return fu


def vdss_to_vd(vdss_log):
    """
    Convert Vdss (log scale) to Vd
    
    TDC vdss_lombardo: log10(L/kg)
    Vd = 10^Y
    """
    # Clip log value to avoid overflow
    vdss_log = np.clip(vdss_log, -2, 2)  # 0.01 to 100 L/kg
    vd = 10 ** vdss_log
    vd = np.clip(vd, 0.01, 100)  # Physiological range
    return vd


def clearance_microsome_to_invivo(cl_micro_log):
    """
    Convert microsomal clearance to in vivo hepatic clearance
    
    TDC clearance_microsome: log10(μL/min/mg protein)
    
    Scaling (simplified IVIVE):
    1. CL_int = 10^Y (μL/min/mg)
    2. Scale to whole liver: CL_int * 45 mg protein/g liver * 1800 g liver
    3. Convert to mL/min
    4. Apply hepatic extraction: CL_h = Q_h * E_h
    """
    # Clip log value to avoid overflow
    cl_micro_log = np.clip(cl_micro_log, -3, 3)  # 0.001 to 1000
    cl_micro = 10 ** cl_micro_log  # μL/min/mg protein
    
    # Scaling factors (literature values)
    mg_protein_per_g_liver = 45  # mg/g
    liver_weight = 1.8  # kg (1800 g)
    liver_blood_flow = 1500  # mL/min (Q_h)
    
    # Scale to whole liver (mL/min)
    cl_int_liver = cl_micro * mg_protein_per_g_liver * liver_weight * 1000 / 1000
    
    # Well-stirred model: CL_h = Q_h * (CL_int / (Q_h + CL_int))
    cl_hepatic = liver_blood_flow * (cl_int_liver / (liver_blood_flow + cl_int_liver))
    
    # Convert to L/min
    cl_hepatic_l = cl_hepatic / 1000
    
    return np.clip(cl_hepatic_l, 0.001, 2.0)  # Physiological range


def clearance_hepatocyte_to_invivo(cl_hepa_log):
    """
    Convert hepatocyte clearance to in vivo hepatic clearance
    
    TDC clearance_hepatocyte: log10(μL/min/10^6 cells)
    
    Scaling:
    1. CL_int = 10^Y (μL/min/10^6 cells)
    2. Scale to whole liver: CL_int * 120*10^6 cells/g * 1800 g
    3. Apply hepatic extraction
    """
    # Clip log value to avoid overflow
    cl_hepa_log = np.clip(cl_hepa_log, -3, 3)  # 0.001 to 1000
    cl_hepa = 10 ** cl_hepa_log  # μL/min/10^6 cells
    
    # Scaling factors
    cells_per_g_liver = 120e6  # cells/g
    liver_weight = 1.8  # kg
    liver_blood_flow = 1500  # mL/min
    
    # Scale to whole liver (mL/min)
    cl_int_liver = cl_hepa * (cells_per_g_liver / 1e6) * liver_weight * 1000 / 1000
    
    # Well-stirred model
    cl_hepatic = liver_blood_flow * (cl_int_liver / (liver_blood_flow + cl_int_liver))
    
    # Convert to L/min
    cl_hepatic_l = cl_hepatic / 1000
    
    return np.clip(cl_hepatic_l, 0.001, 2.0)


# ==================== PROCESSING ====================

def process_tdc_data():
    """Process TDC data to PBPK format"""
    print("\n📁 Loading TDC data...")
    
    if not TDC_DATA.exists():
        print(f"❌ File not found: {TDC_DATA}")
        return None
    
    df = pd.read_csv(TDC_DATA)
    print(f"  ✓ Loaded {len(df)} entries")
    print(f"  ✓ Unique compounds: {df['Drug_ID'].nunique()}")
    
    # Create output dataframe
    print("\n🔄 Converting to PBPK format...")
    
    pbpk_data = []
    
    # Group by compound
    for drug_id, group in df.groupby('Drug_ID'):
        entry = {
            'drug_id': drug_id,
            'smiles': group['Drug'].iloc[0],
            'fu': np.nan,
            'vd': np.nan,
            'clearance': np.nan,
            'half_life': np.nan,
            'bioavailability': np.nan,
            'sources': []
        }
        
        for _, row in group.iterrows():
            source = row['source']
            value = row['Y']
            
            # Convert based on source
            if source == 'ppbr_az':
                entry['fu'] = ppbr_to_fu(value)
                entry['sources'].append('ppbr_az')
                
            elif source == 'vdss_lombardo':
                entry['vd'] = vdss_to_vd(value)
                entry['sources'].append('vdss_lombardo')
                
            elif source == 'clearance_microsome':
                if np.isnan(entry['clearance']):  # Only if not already set
                    entry['clearance'] = clearance_microsome_to_invivo(value)
                    entry['sources'].append('clearance_microsome')
                
            elif source == 'clearance_hepatocyte':
                # Prefer hepatocyte over microsome if both available
                entry['clearance'] = clearance_hepatocyte_to_invivo(value)
                entry['sources'].append('clearance_hepatocyte')
                
            elif source == 'half_life_obach':
                # Clip to avoid overflow
                value_clipped = np.clip(value, -2, 3)  # 0.01 to 1000 hours
                entry['half_life'] = 10 ** value_clipped  # hours
                entry['sources'].append('half_life_obach')
                
            elif source == 'bioavailability_ma':
                # Clip to avoid overflow
                value_clipped = np.clip(value, -2, 2)  # 1% to 100%
                entry['bioavailability'] = 10 ** value_clipped  # %
                entry['sources'].append('bioavailability_ma')
        
        entry['sources'] = ','.join(entry['sources'])
        pbpk_data.append(entry)
    
    # Create DataFrame
    pbpk_df = pd.DataFrame(pbpk_data)
    
    print(f"\n✅ Processed {len(pbpk_df)} compounds")
    
    # Statistics
    print("\n📊 Data availability:")
    print(f"  Fu:             {pbpk_df['fu'].notna().sum():5d} ({pbpk_df['fu'].notna().sum()/len(pbpk_df)*100:.1f}%)")
    print(f"  Vd:             {pbpk_df['vd'].notna().sum():5d} ({pbpk_df['vd'].notna().sum()/len(pbpk_df)*100:.1f}%)")
    print(f"  Clearance:      {pbpk_df['clearance'].notna().sum():5d} ({pbpk_df['clearance'].notna().sum()/len(pbpk_df)*100:.1f}%)")
    print(f"  Half-life:      {pbpk_df['half_life'].notna().sum():5d} ({pbpk_df['half_life'].notna().sum()/len(pbpk_df)*100:.1f}%)")
    print(f"  Bioavailability:{pbpk_df['bioavailability'].notna().sum():5d} ({pbpk_df['bioavailability'].notna().sum()/len(pbpk_df)*100:.1f}%)")
    
    # Value ranges
    print("\n📈 Value ranges:")
    for col in ['fu', 'vd', 'clearance']:
        if pbpk_df[col].notna().sum() > 0:
            vals = pbpk_df[col].dropna()
            print(f"  {col:12s}: [{vals.min():.4f}, {vals.max():.4f}], median={vals.median():.4f}")
    
    return pbpk_df


def save_processed_data(pbpk_df):
    """Save processed data"""
    print("\n💾 Saving processed data...")
    
    # Save as CSV
    csv_path = OUTPUT_DIR / 'tdc_pbpk_processed.csv'
    pbpk_df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV saved: {csv_path}")
    
    # Save as pickle (for easy loading)
    pkl_path = OUTPUT_DIR / 'tdc_pbpk_processed.pkl'
    pbpk_df.to_pickle(pkl_path)
    print(f"  ✓ Pickle saved: {pkl_path}")
    
    # Save statistics
    stats = {
        'total_compounds': len(pbpk_df),
        'fu_available': int(pbpk_df['fu'].notna().sum()),
        'vd_available': int(pbpk_df['vd'].notna().sum()),
        'clearance_available': int(pbpk_df['clearance'].notna().sum()),
        'value_ranges': {
            'fu': {
                'min': float(pbpk_df['fu'].min()),
                'max': float(pbpk_df['fu'].max()),
                'median': float(pbpk_df['fu'].median())
            } if pbpk_df['fu'].notna().sum() > 0 else None,
            'vd': {
                'min': float(pbpk_df['vd'].min()),
                'max': float(pbpk_df['vd'].max()),
                'median': float(pbpk_df['vd'].median())
            } if pbpk_df['vd'].notna().sum() > 0 else None,
            'clearance': {
                'min': float(pbpk_df['clearance'].min()),
                'max': float(pbpk_df['clearance'].max()),
                'median': float(pbpk_df['clearance'].median())
            } if pbpk_df['clearance'].notna().sum() > 0 else None
        }
    }
    
    import json
    stats_path = OUTPUT_DIR / 'tdc_pbpk_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Stats saved: {stats_path}")
    
    return csv_path, pkl_path


# ==================== MAIN ====================

def main():
    """Main execution"""
    
    # Process TDC data
    pbpk_df = process_tdc_data()
    
    if pbpk_df is None:
        print("\n❌ Processing failed")
        return
    
    # Save
    csv_path, pkl_path = save_processed_data(pbpk_df)
    
    # Summary
    print("\n" + "="*80)
    print("🎉 PROCESSING COMPLETE!")
    print("="*80)
    
    print(f"\n✅ Processed {len(pbpk_df)} compounds from TDC")
    print(f"✅ Output files:")
    print(f"   - {csv_path}")
    print(f"   - {pkl_path}")
    
    # Next steps
    print("\n🎯 Next steps:")
    print("1. Merge with KEC dataset")
    print("   python scripts/merge_kec_tdc.py")
    print("\n2. Generate embeddings for new compounds")
    print("   python scripts/generate_embeddings_tdc.py")
    print("\n3. Re-train model with expanded data")
    print("   python scripts/train_pbpk_expanded.py")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

