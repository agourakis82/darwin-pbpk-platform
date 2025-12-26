#!/usr/bin/env python3
"""
Análise de Overfitting - Versão Python (mais robusta)

Autor: Dr. Sounio Agourakis + AI Assistant
Data: 2025-11-18
"""

import json
import os
import sys

def main():
    synthetic_path = 'models/dynamic_gnn_v4_compound/evaluation_scientific/scientific_eval.json'
    experimental_path = 'models/dynamic_gnn_v4_compound/revalidation/revalidation_results.json'
    
    if not os.path.exists(synthetic_path) or not os.path.exists(experimental_path):
        print("⚠️  Arquivos não encontrados")
        return
    
    synthetic = json.load(open(synthetic_path))
    experimental = json.load(open(experimental_path))
    
    # Extrair GMFE
    synth_gmfe = synthetic['model_metrics']['geometric_mean_fold_error']
    
    # Experimental - Fine-tuned
    exp_cmax = experimental['Fine-tuned']['cmax']['gmfe']
    exp_auc = experimental['Fine-tuned']['auc']['gmfe']
    
    # Análise
    gap_ratio_cmax = exp_cmax / synth_gmfe
    gap_ratio_auc = exp_auc / synth_gmfe
    
    print("=" * 80)
    print("ANÁLISE DE OVERFITTING - RESULTADOS")
    print("=" * 80)
    print()
    print(f"Cmax:")
    print(f"  - GMFE Sintético: {synth_gmfe:.6f}")
    print(f"  - GMFE Experimental: {exp_cmax:.2f}")
    print(f"  - Gap Ratio: {gap_ratio_cmax:.2f}x")
    print(f"  - Overfitting: {'🚨 DETECTADO' if gap_ratio_cmax > 10.0 else '✅ Não detectado'}")
    print()
    print(f"AUC:")
    print(f"  - GMFE Sintético: {synth_gmfe:.6f}")
    print(f"  - GMFE Experimental: {exp_auc:.2f}")
    print(f"  - Gap Ratio: {gap_ratio_auc:.2f}x")
    print(f"  - Overfitting: {'🚨 DETECTADO' if gap_ratio_auc > 10.0 else '✅ Não detectado'}")
    print()
    
    # Salvar análise
    analysis = {
        'cmax': {
            'synthetic_gmfe': synth_gmfe,
            'experimental_gmfe': exp_cmax,
            'gap_ratio': gap_ratio_cmax,
            'overfitting_detected': gap_ratio_cmax > 10.0
        },
        'auc': {
            'synthetic_gmfe': synth_gmfe,
            'experimental_gmfe': exp_auc,
            'gap_ratio': gap_ratio_auc,
            'overfitting_detected': gap_ratio_auc > 10.0
        }
    }
    
    output_dir = 'julia-migration/logs/overfitting_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/overfitting_analysis_final.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✅ Análise salva em: {output_dir}/")

if __name__ == '__main__':
    main()
