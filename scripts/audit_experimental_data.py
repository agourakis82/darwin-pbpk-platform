#!/usr/bin/env python3
"""
Audita dados experimentais usando métodos estatísticos robustos (SOTA)
- Teste de Grubbs para outliers
- Método de Tukey (IQR)
- Verificação de unidades
- Filtragem de dados inválidos

Criado: 2025-11-17
Autor: AI Assistant + Dr. Agourakis
Baseado em: FDA/EMA guidelines e métodos estatísticos robustos
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def grubbs_test(data: np.ndarray, alpha: float = 0.05) -> List[int]:
    """
    Teste de Grubbs para detecção de outliers

    Args:
        data: Array de dados
        alpha: Nível de significância

    Returns:
        Lista de índices de outliers
    """
    outliers = []
    data_copy = data.copy()

    while len(data_copy) > 2:
        n = len(data_copy)
        mean = np.mean(data_copy)
        std = np.std(data_copy, ddof=1)

        if std == 0:
            break

        # Calcular estatística G para cada ponto
        g_values = np.abs(data_copy - mean) / std
        max_idx = np.argmax(g_values)
        g_max = g_values[max_idx]

        # Valor crítico (t-student)
        t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(
            t_critical ** 2 / (n - 2 + t_critical ** 2)
        )

        if g_max > g_critical:
            # Encontrar índice original
            original_idx = np.where(data == data_copy[max_idx])[0]
            if len(original_idx) > 0:
                outliers.append(int(original_idx[0]))
            data_copy = np.delete(data_copy, max_idx)
        else:
            break

    return outliers


def tukey_outliers(data: np.ndarray, k: float = 1.5) -> List[int]:
    """
    Método de Tukey (IQR) para detecção de outliers

    Args:
        data: Array de dados
        k: Multiplicador do IQR (padrão: 1.5)

    Returns:
        Lista de índices de outliers
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outliers.tolist()


def audit_experimental_data(
    data_path: Path,
    metadata_path: Path,
    output_dir: Path,
    dose_range: Tuple[float, float] = (0.1, 2000.0),
    clearance_range: Tuple[float, float] = (0.01, 500.0),
) -> Dict:
    """
    Audita dados experimentais usando métodos SOTA

    Returns:
        Dicionário com resultados da auditoria
    """
    print("🔍 AUDITORIA DE DADOS EXPERIMENTAIS (SOTA)")
    print("=" * 70)

    # Carregar dados
    data = np.load(data_path, allow_pickle=True)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    doses = data['doses']
    clearances_hepatic = data['clearances_hepatic']
    clearances_renal = data['clearances_renal']
    compound_ids = data.get('compound_ids', np.array([f'compound_{i}' for i in range(len(doses))]))

    n_original = len(doses)
    print(f"\n📊 Dados originais: {n_original} compostos")

    # 1. Filtragem por faixas razoáveis
    print("\n1️⃣  FILTRAGEM POR FAIXAS RAZOÁVEIS:")
    valid_dose = (doses >= dose_range[0]) & (doses <= dose_range[1])
    valid_cl_hepatic = (clearances_hepatic >= clearance_range[0]) & (clearances_hepatic <= clearance_range[1])
    valid_cl_renal = (clearances_renal >= clearance_range[0]) & (clearances_renal <= clearance_range[1])
    valid_all = valid_dose & valid_cl_hepatic & valid_cl_renal

    print(f"   Doses válidas ({dose_range[0]}-{dose_range[1]} mg): {valid_dose.sum()}/{n_original}")
    print(f"   CL hepático válido ({clearance_range[0]}-{clearance_range[1]} L/h): {valid_cl_hepatic.sum()}/{n_original}")
    print(f"   CL renal válido ({clearance_range[0]}-{clearance_range[1]} L/h): {valid_cl_renal.sum()}/{n_original}")
    print(f"   Total válido: {valid_all.sum()}/{n_original}")

    # 2. Detecção de outliers (Grubbs)
    print("\n2️⃣  DETECÇÃO DE OUTLIERS (Teste de Grubbs):")
    dose_outliers_grubbs = grubbs_test(doses[valid_all]) if valid_all.sum() > 2 else []
    cl_outliers_grubbs = grubbs_test(clearances_hepatic[valid_all]) if valid_all.sum() > 2 else []
    print(f"   Outliers em doses: {len(dose_outliers_grubbs)}")
    print(f"   Outliers em CL hepático: {len(cl_outliers_grubbs)}")

    # 3. Detecção de outliers (Tukey)
    print("\n3️⃣  DETECÇÃO DE OUTLIERS (Método de Tukey - IQR):")
    dose_outliers_tukey = tukey_outliers(doses[valid_all])
    cl_outliers_tukey = tukey_outliers(clearances_hepatic[valid_all])
    print(f"   Outliers em doses: {len(dose_outliers_tukey)}")
    print(f"   Outliers em CL hepático: {len(cl_outliers_tukey)}")

    # 4. Combinar filtros
    print("\n4️⃣  FILTRO FINAL:")
    # Índices válidos após filtragem de faixas
    valid_indices = np.where(valid_all)[0]

    # Remover outliers detectados por ambos os métodos
    all_outliers = set(dose_outliers_grubbs + dose_outliers_tukey + cl_outliers_grubbs + cl_outliers_tukey)
    final_valid_indices = [i for i, idx in enumerate(valid_indices) if i not in all_outliers]
    final_valid_original = valid_indices[final_valid_indices]

    n_final = len(final_valid_original)
    print(f"   Compostos após auditoria: {n_final}/{n_original} ({100*n_final/n_original:.1f}%)")

    # 5. Estatísticas dos dados filtrados
    print("\n5️⃣  ESTATÍSTICAS DOS DADOS FILTRADOS:")
    filtered_doses = doses[final_valid_original]
    filtered_cl_hepatic = clearances_hepatic[final_valid_original]
    filtered_cl_renal = clearances_renal[final_valid_original]

    print(f"   Doses: min={filtered_doses.min():.2f}, max={filtered_doses.max():.2f}, mean={filtered_doses.mean():.2f} mg")
    print(f"   CL hepático: min={filtered_cl_hepatic.min():.2f}, max={filtered_cl_hepatic.max():.2f}, mean={filtered_cl_hepatic.mean():.2f} L/h")
    print(f"   CL renal: min={filtered_cl_renal.min():.2f}, max={filtered_cl_renal.max():.2f}, mean={filtered_cl_renal.mean():.2f} L/h")

    # 6. Salvar dados filtrados
    print("\n6️⃣  SALVANDO DADOS FILTRADOS:")
    # Tratar compound_ids corretamente
    if len(compound_ids) > 0 and len(compound_ids) == len(doses):
        filtered_compound_ids = compound_ids[final_valid_original]
    else:
        filtered_compound_ids = np.array([f'compound_{i}' for i in range(n_final)], dtype=object)

    filtered_data = {
        'doses': filtered_doses.astype(np.float32),
        'clearances_hepatic': filtered_cl_hepatic.astype(np.float32),
        'clearances_renal': filtered_cl_renal.astype(np.float32),
        'partition_coeffs': data['partition_coeffs'][final_valid_original].astype(np.float32),
        'compound_ids': filtered_compound_ids,
    }

    filtered_metadata = [metadata[i] for i in final_valid_original if i < len(metadata)]

    output_data_path = output_dir / "experimental_validation_data_audited.npz"
    output_metadata_path = output_dir / "experimental_validation_data_audited.metadata.json"

    np.savez_compressed(output_data_path, **filtered_data)
    with open(output_metadata_path, 'w') as f:
        json.dump(filtered_metadata, f, indent=2)

    print(f"   ✅ Dados salvos em: {output_data_path}")
    print(f"   ✅ Metadata salva em: {output_metadata_path}")

    # 7. Relatório de auditoria
    audit_report = {
        'n_original': int(n_original),
        'n_after_range_filter': int(valid_all.sum()),
        'n_after_outlier_removal': int(n_final),
        'dose_outliers_grubbs': len(dose_outliers_grubbs),
        'dose_outliers_tukey': len(dose_outliers_tukey),
        'cl_outliers_grubbs': len(cl_outliers_grubbs),
        'cl_outliers_tukey': len(cl_outliers_tukey),
        'filtered_stats': {
            'doses': {
                'min': float(filtered_doses.min()),
                'max': float(filtered_doses.max()),
                'mean': float(filtered_doses.mean()),
                'median': float(np.median(filtered_doses)),
            },
            'clearances_hepatic': {
                'min': float(filtered_cl_hepatic.min()),
                'max': float(filtered_cl_hepatic.max()),
                'mean': float(filtered_cl_hepatic.mean()),
                'median': float(np.median(filtered_cl_hepatic)),
            },
        },
    }

    report_path = output_dir / "audit_report.json"
    with open(report_path, 'w') as f:
        json.dump(audit_report, f, indent=2)

    print(f"   ✅ Relatório salvo em: {report_path}")

    return audit_report


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Audita dados experimentais (SOTA)")
    ap.add_argument("--data", required=True, help="Caminho para dados experimentais (.npz)")
    ap.add_argument("--metadata", required=True, help="Caminho para metadata (.json)")
    ap.add_argument("--output-dir", required=True, help="Diretório de saída")
    ap.add_argument("--dose-min", type=float, default=0.1, help="Dose mínima aceitável (mg)")
    ap.add_argument("--dose-max", type=float, default=2000.0, help="Dose máxima aceitável (mg)")
    ap.add_argument("--clearance-min", type=float, default=0.01, help="Clearance mínimo aceitável (L/h)")
    ap.add_argument("--clearance-max", type=float, default=500.0, help="Clearance máximo aceitável (L/h)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_experimental_data(
        Path(args.data),
        Path(args.metadata),
        output_dir,
        dose_range=(args.dose_min, args.dose_max),
        clearance_range=(args.clearance_min, args.clearance_max),
    )


if __name__ == "__main__":
    main()

