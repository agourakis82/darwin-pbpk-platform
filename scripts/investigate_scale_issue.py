#!/usr/bin/env python3
"""
Investiga e corrige problema de escala nas previsões
Criado: 2025-11-17
Autor: AI Assistant + Dr. Agourakis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def investigate_scale_issue():
    """Investiga problema de escala"""

    # Carregar dados de treino
    print("📊 Carregando dados de treino...")
    train_data = np.load('data/processed/pbpk_enriched/dynamic_gnn_dataset_enriched_v4.npz', allow_pickle=True)

    train_doses = train_data['doses']
    train_concentrations = train_data['concentrations']  # [N, NUM_ORGANS, T]
    train_cl_hepatic = train_data['clearances_hepatic']

    # Carregar dados experimentais
    print("📊 Carregando dados experimentais...")
    exp_data = np.load('data/processed/pbpk_enriched/experimental_validation_data.npz', allow_pickle=True)
    with open('data/processed/pbpk_enriched/experimental_validation_data_converted.metadata.json', 'r') as f:
        exp_metadata = json.load(f)

    exp_doses = exp_data['doses']
    exp_cl_hepatic = exp_data['clearances_hepatic']

    # Carregar resultados de validação
    print("📊 Carregando resultados de validação...")
    val_results = pd.read_csv('models/dynamic_gnn_v4_compound/validation_experimental_final/validation_results.csv')

    print("\n" + "=" * 70)
    print("🔍 ANÁLISE DE ESCALA")
    print("=" * 70)

    # 1. Comparar doses
    print("\n1️⃣  COMPARAÇÃO DE DOSES:")
    print(f"   Treino: min={train_doses.min():.2f}, max={train_doses.max():.2f}, mean={train_doses.mean():.2f} mg")
    print(f"   Experimental: min={exp_doses.min():.2f}, max={exp_doses.max():.2f}, mean={exp_doses.mean():.2f} mg")

    # 2. Comparar clearances
    print("\n2️⃣  COMPARAÇÃO DE CLEARANCES:")
    print(f"   Treino CL hepático: min={train_cl_hepatic.min():.2f}, max={train_cl_hepatic.max():.2f}, mean={train_cl_hepatic.mean():.2f} L/h")
    print(f"   Experimental CL hepático: min={exp_cl_hepatic.min():.2f}, max={exp_cl_hepatic.max():.2f}, mean={exp_cl_hepatic.mean():.2f} L/h")

    # 3. Analisar concentrações do treino
    print("\n3️⃣  CONCENTRAÇÕES DO DATASET DE TREINO:")
    # Blood = índice 0
    train_blood_conc = train_concentrations[:, 0, :]  # [N, T]
    train_cmax = train_blood_conc.max(axis=1)  # [N]
    train_auc = np.trapz(train_blood_conc, axis=1)  # [N] - aproximação simples

    print(f"   Cmax (blood): min={train_cmax.min():.6f}, max={train_cmax.max():.6f}, mean={train_cmax.mean():.6f} mg/L")
    print(f"   AUC (blood): min={train_auc.min():.6f}, max={train_auc.max():.6f}, mean={train_auc.mean():.6f} mg·h/L")

    # 4. Analisar previsões vs observados
    print("\n4️⃣  PREVISÕES VS OBSERVADOS:")
    val_with_obs = val_results[val_results['obs_cmax'].notna() & val_results['obs_auc'].notna()]
    print(f"   Compostos com dados observados: {len(val_with_obs)}")
    print(f"   Cmax previsto: min={val_with_obs['pred_cmax'].min():.4f}, max={val_with_obs['pred_cmax'].max():.4f}, mean={val_with_obs['pred_cmax'].mean():.4f} mg/L")
    print(f"   Cmax observado: min={val_with_obs['obs_cmax'].min():.4f}, max={val_with_obs['obs_cmax'].max():.4f}, mean={val_with_obs['obs_cmax'].mean():.4f} mg/L")
    print(f"   Razão média (pred/obs): {val_with_obs['pred_cmax'].mean() / val_with_obs['obs_cmax'].mean():.2f}×")

    # 5. Análise de escala esperada
    print("\n5️⃣  ANÁLISE DE ESCALA ESPERADA:")
    # Para uma dose de 100 mg e volume de sangue de 5L:
    # Concentração inicial = dose / volume = 100 mg / 5 L = 20 mg/L
    # Mas isso é apenas a concentração inicial (t=0)
    # A concentração ao longo do tempo diminui devido ao clearance

    # Calcular concentração inicial esperada para doses experimentais
    blood_volume = 5.0  # L (padrão)
    exp_initial_conc = exp_doses / blood_volume
    print(f"   Concentração inicial esperada (dose/volume_sangue):")
    print(f"     Min: {exp_initial_conc.min():.4f} mg/L")
    print(f"     Max: {exp_initial_conc.max():.4f} mg/L")
    print(f"     Mean: {exp_initial_conc.mean():.4f} mg/L")

    # Comparar com Cmax observado
    obs_cmax_values = val_with_obs['obs_cmax'].values
    print(f"\n   Cmax observado (média): {obs_cmax_values.mean():.4f} mg/L")
    print(f"   Razão (Cmax_obs / Conc_inicial_esperada): {obs_cmax_values.mean() / exp_initial_conc.mean():.4f}")
    print(f"   (Esperado: < 1.0, pois Cmax < concentração inicial devido a distribuição)")

    # 6. Hipóteses sobre a discrepância
    print("\n6️⃣  HIPÓTESES SOBRE A DISCREPÂNCIA:")
    print("   a) Modelo pode estar prevendo concentrações muito altas")
    print("   b) Parâmetros experimentais (CL, Kp) podem estar incorretos")
    print("   c) Pode haver problema de normalização no modelo")
    print("   d) Doses experimentais podem estar em unidades diferentes")

    # 7. Verificar se há correlação entre dose e razão
    print("\n7️⃣  CORRELAÇÃO DOSE vs RAZÃO:")
    val_with_obs['cmax_ratio'] = val_with_obs['pred_cmax'] / val_with_obs['obs_cmax']
    correlation = val_with_obs['dose'].corr(val_with_obs['cmax_ratio'])
    print(f"   Correlação dose vs razão Cmax: {correlation:.4f}")
    if abs(correlation) > 0.5:
        print("   ⚠️  Correlação forte - sugere problema de escala dependente da dose")
    else:
        print("   ✅ Correlação fraca - problema pode ser sistêmico")

    # 8. Verificar se há correlação entre clearance e razão
    print("\n8️⃣  CORRELAÇÃO CLEARANCE vs RAZÃO:")
    # Tentar extrair clearances do metadata
    clearances = []
    for i in range(len(val_with_obs)):
        idx = val_with_obs.index[i]
        if idx < len(exp_metadata):
            cl_lit = exp_metadata[idx].get('CL_lit', None)
            if cl_lit:
                clearances.append(float(cl_lit))
            else:
                clearances.append(None)
        else:
            clearances.append(None)

    val_with_obs['cl_total'] = clearances
    val_with_cl = val_with_obs[val_with_obs['cl_total'].notna()]
    if len(val_with_cl) > 0:
        correlation_cl = val_with_cl['cl_total'].corr(val_with_cl['cmax_ratio'])
        print(f"   Correlação clearance vs razão Cmax: {correlation_cl:.4f}")
        if abs(correlation_cl) > 0.5:
            print("   ⚠️  Correlação forte - sugere problema com estimativa de clearance")

    # 9. Recomendações
    print("\n" + "=" * 70)
    print("💡 RECOMENDAÇÕES:")
    print("=" * 70)
    print("1. Verificar se doses experimentais estão corretas (unidades)")
    print("2. Refinar estimativas de clearance (usar dados experimentais quando disponíveis)")
    print("3. Verificar se há normalização incorreta no modelo")
    print("4. Considerar fine-tuning do modelo em dados experimentais")
    print("5. Implementar calibração de escala baseada em dados experimentais")

    # Salvar análise
    analysis = {
        'train_doses': {
            'min': float(train_doses.min()),
            'max': float(train_doses.max()),
            'mean': float(train_doses.mean()),
        },
        'exp_doses': {
            'min': float(exp_doses.min()),
            'max': float(exp_doses.max()),
            'mean': float(exp_doses.mean()),
        },
        'train_cl_hepatic': {
            'min': float(train_cl_hepatic.min()),
            'max': float(train_cl_hepatic.max()),
            'mean': float(train_cl_hepatic.mean()),
        },
        'exp_cl_hepatic': {
            'min': float(exp_cl_hepatic.min()),
            'max': float(exp_cl_hepatic.max()),
            'mean': float(exp_cl_hepatic.mean()),
        },
        'train_cmax': {
            'min': float(train_cmax.min()),
            'max': float(train_cmax.max()),
            'mean': float(train_cmax.mean()),
        },
        'pred_cmax_mean': float(val_with_obs['pred_cmax'].mean()),
        'obs_cmax_mean': float(val_with_obs['obs_cmax'].mean()),
        'cmax_ratio_mean': float(val_with_obs['cmax_ratio'].mean()),
        'dose_correlation': float(correlation) if not np.isnan(correlation) else None,
    }

    output_dir = Path('models/dynamic_gnn_v4_compound/scale_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'scale_investigation.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✅ Análise salva em: {output_dir / 'scale_investigation.json'}")


if __name__ == "__main__":
    investigate_scale_issue()


