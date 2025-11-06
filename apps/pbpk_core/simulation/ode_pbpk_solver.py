"""
ODE Solver PBPK - Ground Truth para Treinamento

Solver ODE tradicional para PBPK (14 compartimentos).
Usado para gerar dados de treinamento para o Dynamic GNN.

Autor: Dr. Demetrios Chiuratto Agourakis
Data: Novembro 2025
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
    NUM_ORGANS
)


@dataclass
class ODEState:
    """Estado do sistema ODE (concentrações em cada compartimento)."""
    concentrations: np.ndarray  # [NUM_ORGANS]


class ODEPBPKSolver:
    """
    Solver ODE para modelo PBPK de 14 compartimentos.
    
    Baseado em modelo de fluxo sanguíneo padrão:
    - Blood (central) conecta todos os órgãos
    - Clearance hepático e renal
    - Partition coefficients (Kp)
    """
    
    def __init__(self, physiological_params: PBPKPhysiologicalParams):
        self.params = physiological_params
        self.blood_idx = PBPK_ORGANS.index("blood")
        self.liver_idx = PBPK_ORGANS.index("liver")
        self.kidney_idx = PBPK_ORGANS.index("kidney")
    
    def _ode_system(
        self,
        y: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Sistema de EDOs para PBPK.
        
        Args:
            y: Concentrações [NUM_ORGANS]
            t: Tempo (horas)
        
        Returns:
            dydt: Derivadas temporais [NUM_ORGANS]
        """
        dydt = np.zeros(NUM_ORGANS)
        
        C_blood = y[self.blood_idx]
        
        # Para cada órgão (exceto blood)
        for i, organ in enumerate(PBPK_ORGANS):
            if i == self.blood_idx:
                continue
            
            # Parâmetros do órgão
            V_organ = self.params.volumes.get(organ, 1.0)
            Q_organ = self.params.blood_flows.get(organ, 0.0)
            Kp_organ = self.params.partition_coeffs.get(organ, 1.0)
            
            # Concentração no órgão
            C_organ = y[i]
            
            # Fluxo de entrada (blood -> organ)
            # Taxa = Q * (C_blood - C_organ/Kp)
            dydt[i] = (Q_organ / V_organ) * (C_blood - C_organ / Kp_organ)
            
            # Fluxo de saída (organ -> blood)
            dydt[self.blood_idx] -= (Q_organ / self.params.volumes["blood"]) * (C_blood - C_organ / Kp_organ)
        
        # Clearance hepático
        if self.params.clearance_hepatic > 0:
            C_liver = y[self.liver_idx]
            clearance_rate = self.params.clearance_hepatic / self.params.volumes["blood"]
            dydt[self.blood_idx] -= clearance_rate * C_blood
        
        # Clearance renal
        if self.params.clearance_renal > 0:
            C_kidney = y[self.kidney_idx]
            clearance_rate = self.params.clearance_renal / self.params.volumes["blood"]
            dydt[self.blood_idx] -= clearance_rate * C_blood
        
        return dydt
    
    def solve(
        self,
        dose: float,
        time_points: np.ndarray,
        initial_conditions: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Resolve o sistema ODE.
        
        Args:
            dose: Dose administrada (mg)
            time_points: Pontos de tempo (horas)
            initial_conditions: Condições iniciais (opcional)
        
        Returns:
            Dict com concentrações por órgão ao longo do tempo
        """
        # Condições iniciais
        if initial_conditions is None:
            y0 = np.zeros(NUM_ORGANS)
            # Dose inicial no blood
            blood_volume = self.params.volumes["blood"]
            y0[self.blood_idx] = dose / blood_volume  # mg/L
        else:
            y0 = initial_conditions.copy()
        
        # Resolver ODE
        solution = odeint(self._ode_system, y0, time_points)
        
        # Organizar resultados
        results = {}
        for i, organ in enumerate(PBPK_ORGANS):
            results[organ] = solution[:, i]
        
        results["time"] = time_points
        
        return results
    
    def simulate(
        self,
        dose: float,
        t_max: float = 24.0,
        num_points: int = 100,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Simula PBPK com parâmetros padrão.
        
        Args:
            dose: Dose (mg)
            t_max: Tempo máximo (horas)
            num_points: Número de pontos temporais
        
        Returns:
            Dict com concentrações por órgão
        """
        time_points = np.linspace(0, t_max, num_points)
        return self.solve(dose, time_points, **kwargs)


if __name__ == "__main__":
    # Teste básico
    print("=" * 80)
    print("TESTE: ODE Solver PBPK")
    print("=" * 80)
    
    # Parâmetros
    params = PBPKPhysiologicalParams(
        clearance_hepatic=10.0,  # L/h
        clearance_renal=5.0,      # L/h
        partition_coeffs={
            "liver": 2.0,
            "kidney": 1.5,
            "brain": 0.5,  # BBB
            "adipose": 3.0  # Lipofílico
        }
    )
    
    # Criar solver
    solver = ODEPBPKSolver(params)
    
    # Simular
    dose = 100.0  # mg
    results = solver.simulate(dose, t_max=24.0, num_points=100)
    
    print(f"\n✅ Simulação completa:")
    print(f"   Dose: {dose} mg")
    print(f"   Time points: {len(results['time'])}")
    
    # Mostrar concentrações finais
    print(f"\n📊 Concentrações finais (t={results['time'][-1]:.2f}h):")
    for organ in PBPK_ORGANS:
        conc = results[organ][-1]
        print(f"   {organ:12s}: {conc:8.4f} mg/L")
    
    print("\n" + "=" * 80)
    print("✅ TESTE CONCLUÍDO!")
    print("=" * 80)

