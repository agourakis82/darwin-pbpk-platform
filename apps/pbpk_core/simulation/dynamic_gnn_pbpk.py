"""
Dynamic Graph Neural Network for PBPK Simulation

Baseado em: arXiv 2024 - Dynamic GNN for PBPK (R² 0.9342)

Arquitetura:
- Graph: 14 órgãos (nodes)
- Edges: Fluxos sanguíneos, clearance, partition coefficients
- Temporal: Evolution via GNN layers
- Attention: Critical organs (liver, kidney, brain)

Vantagens sobre ODE:
- R² 0.93+ vs 0.85-0.90 (ODE tradicional)
- Menos dependência de parâmetros fisiológicos
- Aprende interações não-lineares dos dados
- Mais rápido (forward pass vs ODE solver)

Autor: Dr. Demetrios Chiuratto Agourakis
Data: Novembro 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

# 14 compartimentos PBPK padrão
PBPK_ORGANS = [
    "blood",      # 0 - Plasma/sangue
    "liver",      # 1 - Fígado (metabolismo)
    "kidney",     # 2 - Rim (excreção)
    "brain",      # 3 - Cérebro (BBB)
    "heart",      # 4 - Coração
    "lung",       # 5 - Pulmão
    "muscle",     # 6 - Músculo
    "adipose",    # 7 - Tecido adiposo
    "gut",        # 8 - Intestino (absorção)
    "skin",       # 9 - Pele
    "bone",       # 10 - Osso
    "spleen",     # 11 - Baço
    "pancreas",   # 12 - Pâncreas
    "other"       # 13 - Resto do corpo
]

NUM_ORGANS = len(PBPK_ORGANS)


@dataclass
class PBPKPhysiologicalParams:
    """Parâmetros fisiológicos padrão (70kg adulto)"""
    # Volumes (L)
    volumes: Dict[str, float] = None
    
    # Fluxos sanguíneos (L/h)
    blood_flows: Dict[str, float] = None
    
    # Clearance (L/h)
    clearance_hepatic: float = 0.0
    clearance_renal: float = 0.0
    
    # Partition coefficients (Kp)
    partition_coeffs: Dict[str, float] = None
    
    def __post_init__(self):
        if self.volumes is None:
            self.volumes = {
                "blood": 5.0,
                "liver": 1.8,
                "kidney": 0.31,
                "brain": 1.4,
                "heart": 0.33,
                "lung": 0.5,
                "muscle": 30.0,
                "adipose": 15.0,
                "gut": 1.1,
                "skin": 3.3,
                "bone": 10.0,
                "spleen": 0.18,
                "pancreas": 0.1,
                "other": 5.0
            }
        
        if self.blood_flows is None:
            self.blood_flows = {
                "blood": 0.0,  # Não aplicável
                "liver": 90.0,
                "kidney": 60.0,
                "brain": 50.0,
                "heart": 20.0,
                "lung": 300.0,  # Cardiac output
                "muscle": 75.0,
                "adipose": 12.0,
                "gut": 45.0,
                "skin": 10.0,
                "bone": 5.0,
                "spleen": 15.0,
                "pancreas": 5.0,
                "other": 20.0
            }
        
        if self.partition_coeffs is None:
            # Default Kp = 1.0 (sem dados específicos)
            self.partition_coeffs = {organ: 1.0 for organ in PBPK_ORGANS}


class OrganMessagePassing(MessagePassing):
    """
    Message passing layer para interações entre órgãos.
    
    Captura:
    - Fluxos sanguíneos (edges)
    - Clearance (node features)
    - Partition coefficients (edge features)
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='add', flow='target_to_source')
        
        self.node_feature_dim = node_dim  # Renomeado para evitar conflito
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Attention weights (para órgãos críticos)
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            size: Optional size tuple
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Computa mensagens entre órgãos.
        
        Args:
            x_i: Source node features [num_edges, node_dim]
            x_j: Target node features [num_edges, node_dim]
            edge_attr: Edge features [num_edges, edge_dim]
        """
        # Concatenar features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Computar mensagem
        msg = self.message_mlp(msg_input)
        
        # Attention (peso da mensagem)
        attn = torch.sigmoid(self.attention(msg))
        
        return attn * msg
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Atualiza node features após agregação de mensagens.
        
        Args:
            aggr_out: Agregated messages [num_nodes, hidden_dim]
            x: Original node features [num_nodes, node_dim]
        """
        # Concatenar com features originais
        update_input = torch.cat([x, aggr_out], dim=-1)
        
        # Atualizar
        updated = self.update_mlp(update_input)
        
        # Residual connection
        return x + updated


class DynamicPBPKGNN(nn.Module):
    """
    Dynamic Graph Neural Network para simulação PBPK.
    
    Baseado em: arXiv 2024 (R² 0.9342)
    
    Arquitetura:
    1. Graph Construction: 14 órgãos (nodes) + fluxos (edges)
    2. Temporal Evolution: GNN layers para evolução temporal
    3. Attention: Órgãos críticos (liver, kidney, brain)
    4. Prediction: Concentrações ao longo do tempo
    """
    
    def __init__(
        self,
        node_dim: int = 16,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        num_temporal_steps: int = 100,
        dt: float = 0.1,  # Timestep (horas)
        use_attention: bool = True
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_temporal_steps = num_temporal_steps
        self.dt = dt
        self.use_attention = use_attention
        
        # Node feature encoder (concentração + propriedades do órgão)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge feature encoder (fluxo + Kp)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # GNN layers para evolução temporal
        self.gnn_layers = nn.ModuleList([
            OrganMessagePassing(
                node_dim=hidden_dim,
                edge_dim=hidden_dim // 2,
                hidden_dim=hidden_dim
            )
            for _ in range(num_gnn_layers)
        ])
        
        # Attention para órgãos críticos
        if use_attention:
            self.organ_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True
            )
        
        # Temporal evolution (RNN-like)
        self.temporal_evolution = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Output head (concentração)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Concentração
            nn.ReLU()  # Concentração >= 0
        )
        
        # Critical organs indices (para attention)
        self.critical_organs_idx = [
            PBPK_ORGANS.index("liver"),
            PBPK_ORGANS.index("kidney"),
            PBPK_ORGANS.index("brain")
        ]
    
    def build_pbpk_graph(
        self,
        physiological_params: PBPKPhysiologicalParams,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Constrói grafo PBPK com 14 órgãos.
        
        Args:
            physiological_params: Parâmetros fisiológicos
            device: Device (CPU/GPU)
        
        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim]
            node_features: [num_nodes, node_dim]
        """
        # Construir edges (conexões entre órgãos via sangue)
        # Estrutura: Blood (central) conecta todos os órgãos
        blood_idx = PBPK_ORGANS.index("blood")
        
        edges = []
        edge_attrs = []
        
        for i, organ in enumerate(PBPK_ORGANS):
            if i == blood_idx:
                continue
            
            # Blood -> Organ (entrada)
            edges.append([blood_idx, i])
            flow = physiological_params.blood_flows.get(organ, 0.0)
            kp = physiological_params.partition_coeffs.get(organ, 1.0)
            edge_attrs.append([flow, kp, 1.0, 0.0])  # [flow, Kp, direction, clearance]
            
            # Organ -> Blood (saída)
            edges.append([i, blood_idx])
            edge_attrs.append([flow, 1.0/kp, -1.0, 0.0])  # Fluxo reverso
        
        # Adicionar clearance edges (liver -> blood, kidney -> blood)
        liver_idx = PBPK_ORGANS.index("liver")
        kidney_idx = PBPK_ORGANS.index("kidney")
        
        # Liver clearance
        if physiological_params.clearance_hepatic > 0:
            edges.append([liver_idx, blood_idx])
            edge_attrs.append([
                0.0,  # Não é fluxo sanguíneo
                1.0,
                0.0,
                physiological_params.clearance_hepatic
            ])
        
        # Kidney clearance
        if physiological_params.clearance_renal > 0:
            edges.append([kidney_idx, blood_idx])
            edge_attrs.append([
                0.0,
                1.0,
                0.0,
                physiological_params.clearance_renal
            ])
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32, device=device)
        
        # Node features: [volume, clearance_local, initial_concentration, ...]
        node_features = []
        for organ in PBPK_ORGANS:
            volume = physiological_params.volumes.get(organ, 1.0)
            clearance = 0.0
            if organ == "liver":
                clearance = physiological_params.clearance_hepatic
            elif organ == "kidney":
                clearance = physiological_params.clearance_renal
            
            # Node features: [volume, clearance, initial_conc, ...]
            # Expandir para node_dim features
            features = [volume, clearance, 0.0] + [0.0] * (self.node_dim - 3)
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float32, device=device)
        
        return edge_index, edge_attr, node_features
    
    def forward(
        self,
        dose: float,
        physiological_params: PBPKPhysiologicalParams,
        time_points: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Simula PBPK usando Dynamic GNN.
        
        Args:
            dose: Dose administrada (mg)
            physiological_params: Parâmetros fisiológicos
            time_points: Pontos de tempo (horas). Se None, usa num_temporal_steps
        
        Returns:
            Dict com:
            - concentrations: [num_organs, num_time_points]
            - time_points: [num_time_points]
        """
        device = next(self.parameters()).device
        
        # Construir grafo
        edge_index, edge_attr, node_features = self.build_pbpk_graph(
            physiological_params, device
        )
        
        # Inicializar concentração (dose no blood)
        blood_idx = PBPK_ORGANS.index("blood")
        blood_volume = physiological_params.volumes["blood"]
        initial_concentration = dose / blood_volume  # mg/L
        node_features[blood_idx, 2] = initial_concentration  # Initial conc
        
        # Encodar features
        node_emb = self.node_encoder(node_features)  # [num_organs, hidden_dim]
        edge_emb = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim//2]
        
        # Time points
        if time_points is None:
            time_points = torch.linspace(
                0, self.num_temporal_steps * self.dt,
                self.num_temporal_steps + 1,
                device=device
            )
        else:
            time_points = time_points.to(device)
        
        num_time_points = len(time_points)
        
        # Ajustar num_temporal_steps para match time_points fornecido
        if time_points is not None and len(time_points) > 1:
            # Usar número de pontos fornecido (menos 1 porque começamos do t=0)
            actual_steps = len(time_points) - 1
        else:
            actual_steps = self.num_temporal_steps
        
        # Evolução temporal
        concentrations = []
        current_node_state = node_emb  # [num_organs, hidden_dim]
        
        # Estado oculto do GRU (2 layers)
        h_0 = torch.zeros(2, 1, self.hidden_dim, device=device)  # [num_layers, batch, hidden]
        
        # Número de steps de evolução (num_time_points - 1 porque já temos o inicial)
        num_evolution_steps = num_time_points - 1
        
        # Debug: verificar se está correto
        if num_evolution_steps < 1:
            raise ValueError(f"num_evolution_steps deve ser >= 1, mas é {num_evolution_steps} (num_time_points={num_time_points})")
        
        for t_idx in range(num_evolution_steps):
            # Aplicar GNN layers
            x = current_node_state  # [num_organs, hidden_dim]
            
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x, edge_index, edge_emb)
            
            # Attention em órgãos críticos (opcional)
            if self.use_attention:
                # Extrair órgãos críticos
                critical_nodes = x[self.critical_organs_idx].unsqueeze(0)  # [1, 3, hidden_dim]
                all_nodes = x.unsqueeze(0)  # [1, num_organs, hidden_dim]
                
                # Self-attention
                attended, _ = self.organ_attention(all_nodes, critical_nodes, critical_nodes)
                x = attended.squeeze(0)
            
            # Evolução temporal (GRU) - processar cada órgão separadamente
            # Ou usar média global para simplificar
            x_mean = x.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, hidden_dim]
            x_evolved, h_0 = self.temporal_evolution(x_mean, h_0)
            
            # Expandir de volta para todos os órgãos
            x_evolved_expanded = x_evolved.expand(NUM_ORGANS, -1, -1).squeeze(1)  # [num_organs, hidden_dim]
            current_node_state = x_evolved_expanded
            
            # Predizer concentração
            conc = self.output_head(x)  # [num_organs, 1]
            concentrations.append(conc.squeeze(-1))  # [num_organs]
        
        # Adicionar concentração inicial
        initial_conc = torch.zeros(NUM_ORGANS, device=device)
        initial_conc[blood_idx] = initial_concentration
        
        # Stack: [initial] + [predicted] = [num_time_points, num_organs]
        concentrations = torch.stack([initial_conc] + concentrations, dim=0)  # [num_time_points, num_organs]
        
        # Transpor para [num_organs, num_time_points]
        concentrations = concentrations.t()
        
        # Se time_points foi fornecido externamente, garantir que shapes batem
        if time_points is not None and len(time_points) != concentrations.shape[1]:
            # Truncar ou interpolar para match
            target_len = len(time_points)
            if concentrations.shape[1] > target_len:
                concentrations = concentrations[:, :target_len]
            elif concentrations.shape[1] < target_len:
                # Interpolar (simples: repetir último valor)
                last_conc = concentrations[:, -1:].expand(-1, target_len - concentrations.shape[1])
                concentrations = torch.cat([concentrations, last_conc], dim=1)
        
        return {
            "concentrations": concentrations,  # [num_organs, num_time_points]
            "time_points": time_points,
            "organ_names": PBPK_ORGANS
        }


class DynamicPBPKSimulator:
    """
    Wrapper para DynamicPBPKGNN com interface similar ao ODE solver.
    """
    
    def __init__(
        self,
        model: Optional[DynamicPBPKGNN] = None,
        device: str = "cpu"
    ):
        if model is None:
            model = DynamicPBPKGNN()
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def simulate(
        self,
        dose: float,
        clearance_hepatic: float = 0.0,
        clearance_renal: float = 0.0,
        partition_coeffs: Optional[Dict[str, float]] = None,
        time_points: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Simula PBPK.
        
        Args:
            dose: Dose (mg)
            clearance_hepatic: Clearance hepático (L/h)
            clearance_renal: Clearance renal (L/h)
            partition_coeffs: Partition coefficients por órgão
            time_points: Pontos de tempo (horas)
        
        Returns:
            Dict com concentrações por órgão ao longo do tempo
        """
        # Parâmetros fisiológicos
        params = PBPKPhysiologicalParams(
            clearance_hepatic=clearance_hepatic,
            clearance_renal=clearance_renal,
            partition_coeffs=partition_coeffs or {}
        )
        
        # Converter time_points para tensor se necessário
        if time_points is not None:
            time_points = torch.tensor(time_points, dtype=torch.float32)
        
        # Simular
        with torch.no_grad():
            results = self.model(dose, params, time_points)
        
        # Converter para numpy
        output = {}
        for i, organ in enumerate(results["organ_names"]):
            output[organ] = results["concentrations"][i].cpu().numpy()
        
        output["time"] = results["time_points"].cpu().numpy()
        
        return output


if __name__ == "__main__":
    # Teste básico
    print("=" * 80)
    print("TESTE: Dynamic GNN para PBPK")
    print("=" * 80)
    
    # Criar modelo
    model = DynamicPBPKGNN(
        node_dim=16,
        edge_dim=4,
        hidden_dim=64,
        num_gnn_layers=3,
        num_temporal_steps=100,
        dt=0.1
    )
    
    print(f"\n✅ Modelo criado:")
    print(f"   Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Órgãos: {NUM_ORGANS}")
    
    # Parâmetros de teste
    params = PBPKPhysiologicalParams(
        clearance_hepatic=10.0,  # L/h
        clearance_renal=5.0,  # L/h
        partition_coeffs={
            "liver": 2.0,
            "kidney": 1.5,
            "brain": 0.5,  # BBB
            "adipose": 3.0  # Lipofílico
        }
    )
    
    # Simular
    dose = 100.0  # mg
    results = model(dose, params)
    
    print(f"\n✅ Simulação completa:")
    print(f"   Dose: {dose} mg")
    print(f"   Time points: {len(results['time_points'])}")
    print(f"   Concentrações shape: {results['concentrations'].shape}")
    
    # Mostrar concentrações finais
    print(f"\n📊 Concentrações finais (t={results['time_points'][-1]:.2f}h):")
    for i, organ in enumerate(results["organ_names"]):
        conc = results["concentrations"][i, -1].item()
        print(f"   {organ:12s}: {conc:8.4f} mg/L")
    
    print("\n" + "=" * 80)
    print("✅ TESTE CONCLUÍDO!")
    print("=" * 80)

