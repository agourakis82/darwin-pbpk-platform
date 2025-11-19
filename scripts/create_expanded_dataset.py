#!/usr/bin/env python3
"""
Cria dataset expandido com doses baixas e Kp extremos
- Adiciona exemplos com doses < 10 mg
- Adiciona exemplos com Kp < 0.5 e Kp > 5.0
- Balanceia dataset por dose e Kp
- Usa ODE solver como ground truth

Criado: 2025-11-18
Autor: AI Assistant + Dr. Agourakis
Método: Expansão sistemática do dataset com foco em casos problemáticos
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.ode_pbpk_solver import ODEPBPKSolver
from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
    NUM_ORGANS,
)

TIME_POINTS = np.linspace(0, 12.0, 120).astype(np.float32)


def create_expanded_dataset(
    base_dataset_path: Path,
    output_path: Path,
    n_low_dose: int = 200,
    n_extreme_kp_low: int = 100,
    n_extreme_kp_high: int = 100,
    seed: int = 42,
):
    """Cria dataset expandido"""
    print("📊 CRIAÇÃO DE DATASET EXPANDIDO")
    print("=" * 70)

    rng = np.random.default_rng(seed)

    # Carregar dataset base
    print("\n1️⃣  Carregando dataset base...")
    base_data = np.load(base_dataset_path, allow_pickle=True)
    base_doses = base_data['doses']
    base_cl_hepatic = base_data['clearances_hepatic']
    base_cl_renal = base_data['clearances_renal']
    base_kp = base_data['partition_coeffs']
    base_concentrations = base_data['concentrations']

    print(f"   Dataset base: {len(base_doses)} amostras")

    # Criar ODE solver
    print("\n2️⃣  Criando ODE solver...")
    ode_params = PBPKPhysiologicalParams()
    ode_solver = ODEPBPKSolver(physiological_params=ode_params)

    # Coletar dados existentes
    all_doses = list(base_doses)
    all_cl_hepatic = list(base_cl_hepatic)
    all_cl_renal = list(base_cl_renal)
    all_kp = list(base_kp)
    all_concentrations = list(base_concentrations)

    # Adicionar doses baixas (< 10 mg)
    print(f"\n3️⃣  Adicionando {n_low_dose} exemplos com doses baixas (< 10 mg)...")
    dose_range_low = (0.1, 10.0)
    cl_range = (base_cl_hepatic.min(), base_cl_hepatic.max())
    cl_renal_range = (base_cl_renal.min(), base_cl_renal.max())
    kp_range = (base_kp.min(), base_kp.max())

    for i in tqdm(range(n_low_dose), desc="Doses baixas"):
        dose = float(rng.uniform(*dose_range_low))
        cl_hepatic = float(rng.uniform(*cl_range))
        cl_renal = float(rng.uniform(*cl_renal_range))

        # Kp aleatório mas razoável
        kp = rng.uniform(0.1, 10.0, size=NUM_ORGANS).astype(np.float32)

        partition_dict = {organ: float(kp[j]) for j, organ in enumerate(PBPK_ORGANS)}
        params = PBPKPhysiologicalParams(
            clearance_hepatic=cl_hepatic,
            clearance_renal=cl_renal,
            partition_coeffs=partition_dict,
        )

        # Simular com ODE
        solver = ODEPBPKSolver(physiological_params=params)
        result = solver.solve(dose, TIME_POINTS)

        # Organizar concentrações
        conc_matrix = np.zeros((NUM_ORGANS, len(TIME_POINTS)), dtype=np.float32)
        for j, organ in enumerate(PBPK_ORGANS):
            conc_matrix[j, :] = result[organ].astype(np.float32)

        all_doses.append(dose)
        all_cl_hepatic.append(cl_hepatic)
        all_cl_renal.append(cl_renal)
        all_kp.append(kp)
        all_concentrations.append(conc_matrix)

    # Adicionar Kp muito baixo (< 0.5)
    print(f"\n4️⃣  Adicionando {n_extreme_kp_low} exemplos com Kp muito baixo (< 0.5)...")
    dose_range = (10.0, 500.0)

    for i in tqdm(range(n_extreme_kp_low), desc="Kp baixo"):
        dose = float(rng.uniform(*dose_range))
        cl_hepatic = float(rng.uniform(*cl_range))
        cl_renal = float(rng.uniform(*cl_renal_range))

        # Kp muito baixo (< 0.5)
        kp = rng.uniform(0.01, 0.5, size=NUM_ORGANS).astype(np.float32)

        partition_dict = {organ: float(kp[j]) for j, organ in enumerate(PBPK_ORGANS)}
        params = PBPKPhysiologicalParams(
            clearance_hepatic=cl_hepatic,
            clearance_renal=cl_renal,
            partition_coeffs=partition_dict,
        )

        # Simular com ODE
        solver = ODEPBPKSolver(physiological_params=params)
        result = solver.solve(dose, TIME_POINTS)

        # Organizar concentrações
        conc_matrix = np.zeros((NUM_ORGANS, len(TIME_POINTS)), dtype=np.float32)
        for j, organ in enumerate(PBPK_ORGANS):
            conc_matrix[j, :] = result[organ].astype(np.float32)

        all_doses.append(dose)
        all_cl_hepatic.append(cl_hepatic)
        all_cl_renal.append(cl_renal)
        all_kp.append(kp)
        all_concentrations.append(conc_matrix)

    # Adicionar Kp muito alto (> 5.0)
    print(f"\n5️⃣  Adicionando {n_extreme_kp_high} exemplos com Kp muito alto (> 5.0)...")

    for i in tqdm(range(n_extreme_kp_high), desc="Kp alto"):
        dose = float(rng.uniform(*dose_range))
        cl_hepatic = float(rng.uniform(*cl_range))
        cl_renal = float(rng.uniform(*cl_renal_range))

        # Kp muito alto (> 5.0)
        kp = rng.uniform(5.0, 20.0, size=NUM_ORGANS).astype(np.float32)

        partition_dict = {organ: float(kp[j]) for j, organ in enumerate(PBPK_ORGANS)}
        params = PBPKPhysiologicalParams(
            clearance_hepatic=cl_hepatic,
            clearance_renal=cl_renal,
            partition_coeffs=partition_dict,
        )

        # Simular com ODE
        solver = ODEPBPKSolver(physiological_params=params)
        result = solver.solve(dose, TIME_POINTS)

        # Organizar concentrações
        conc_matrix = np.zeros((NUM_ORGANS, len(TIME_POINTS)), dtype=np.float32)
        for j, organ in enumerate(PBPK_ORGANS):
            conc_matrix[j, :] = result[organ].astype(np.float32)

        all_doses.append(dose)
        all_cl_hepatic.append(cl_hepatic)
        all_cl_renal.append(cl_renal)
        all_kp.append(kp)
        all_concentrations.append(conc_matrix)

    # Converter para arrays
    print("\n6️⃣  Consolidando dataset...")
    # Garantir que todas as concentrações tenham o mesmo shape
    target_shape = (NUM_ORGANS, len(TIME_POINTS))
    for i, conc in enumerate(all_concentrations):
        if conc.shape != target_shape:
            # Interpolar ou truncar se necessário
            if conc.shape[1] != len(TIME_POINTS):
                from scipy.interpolate import interp1d
                old_time = np.linspace(0, 12.0, conc.shape[1])
                new_conc = np.zeros(target_shape, dtype=np.float32)
                for j in range(NUM_ORGANS):
                    f = interp1d(old_time, conc[j, :], kind='linear', fill_value='extrapolate')
                    new_conc[j, :] = f(TIME_POINTS)
                all_concentrations[i] = new_conc
            elif conc.shape[0] != NUM_ORGANS:
                # Padding ou truncamento
                new_conc = np.zeros(target_shape, dtype=np.float32)
                min_organs = min(conc.shape[0], NUM_ORGANS)
                new_conc[:min_organs, :] = conc[:min_organs, :]
                all_concentrations[i] = new_conc

    final_doses = np.array(all_doses, dtype=np.float32)
    final_cl_hepatic = np.array(all_cl_hepatic, dtype=np.float32)
    final_cl_renal = np.array(all_cl_renal, dtype=np.float32)
    final_kp = np.stack(all_kp).astype(np.float32)
    final_concentrations = np.stack(all_concentrations).astype(np.float32)

    # Estatísticas
    print(f"\n📊 ESTATÍSTICAS DO DATASET EXPANDIDO:")
    print(f"   Total de amostras: {len(final_doses)}")
    print(f"   Doses: min={final_doses.min():.2f}, max={final_doses.max():.2f}, mean={final_doses.mean():.2f} mg")
    print(f"   CL hepático: min={final_cl_hepatic.min():.2f}, max={final_cl_hepatic.max():.2f}, mean={final_cl_hepatic.mean():.2f} L/h")
    print(f"   CL renal: min={final_cl_renal.min():.2f}, max={final_cl_renal.max():.2f}, mean={final_cl_renal.mean():.2f} L/h")
    print(f"   Kp médio: min={final_kp.mean(axis=1).min():.2f}, max={final_kp.mean(axis=1).max():.2f}, mean={final_kp.mean(axis=1).mean():.2f}")

    # Distribuição por dose
    n_low = np.sum(final_doses < 10.0)
    n_medium = np.sum((final_doses >= 10.0) & (final_doses < 100.0))
    n_high = np.sum(final_doses >= 100.0)
    print(f"\n   Distribuição por dose:")
    print(f"      < 10 mg: {n_low} ({100*n_low/len(final_doses):.1f}%)")
    print(f"      10-100 mg: {n_medium} ({100*n_medium/len(final_doses):.1f}%)")
    print(f"      > 100 mg: {n_high} ({100*n_high/len(final_doses):.1f}%)")

    # Distribuição por Kp
    kp_mean_all = final_kp.mean(axis=1)
    n_kp_low = np.sum(kp_mean_all < 0.5)
    n_kp_normal = np.sum((kp_mean_all >= 0.5) & (kp_mean_all <= 5.0))
    n_kp_high = np.sum(kp_mean_all > 5.0)
    print(f"\n   Distribuição por Kp médio:")
    print(f"      < 0.5: {n_kp_low} ({100*n_kp_low/len(final_doses):.1f}%)")
    print(f"      0.5-5.0: {n_kp_normal} ({100*n_kp_normal/len(final_doses):.1f}%)")
    print(f"      > 5.0: {n_kp_high} ({100*n_kp_high/len(final_doses):.1f}%)")

    # Salvar dataset
    print(f"\n7️⃣  Salvando dataset expandido...")
    np.savez_compressed(
        output_path,
        doses=final_doses,
        clearances_hepatic=final_cl_hepatic,
        clearances_renal=final_cl_renal,
        partition_coeffs=final_kp,
        concentrations=final_concentrations,
        time_points=TIME_POINTS,
    )

    print(f"\n✅ Dataset expandido criado!")
    print(f"   Salvo em: {output_path}")
    print(f"   Tamanho: {len(final_doses)} amostras (original: {len(base_doses)})")


def main():
    parser = argparse.ArgumentParser(description="Cria dataset expandido com doses baixas e Kp extremos")
    parser.add_argument("--base-dataset", required=True, help="Dataset base (.npz)")
    parser.add_argument("--output", required=True, help="Caminho de saída (.npz)")
    parser.add_argument("--n-low-dose", type=int, default=200, help="Número de exemplos com doses baixas")
    parser.add_argument("--n-extreme-kp-low", type=int, default=100, help="Número de exemplos com Kp muito baixo")
    parser.add_argument("--n-extreme-kp-high", type=int, default=100, help="Número de exemplos com Kp muito alto")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    args = parser.parse_args()

    create_expanded_dataset(
        Path(args.base_dataset),
        Path(args.output),
        n_low_dose=args.n_low_dose,
        n_extreme_kp_low=args.n_extreme_kp_low,
        n_extreme_kp_high=args.n_extreme_kp_high,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

