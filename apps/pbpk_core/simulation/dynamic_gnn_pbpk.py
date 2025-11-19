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
from torch_geometric.nn import MessagePassing
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass

# 14 compartimentos PBPK padrão
PBPK_ORGANS = [
    "blood",  # 0 - Plasma/sangue
    "liver",  # 1 - Fígado (metabolismo)
    "kidney",  # 2 - Rim (excreção)
    "brain",  # 3 - Cérebro (BBB)
    "heart",  # 4 - Coração
    "lung",  # 5 - Pulmão
    "muscle",  # 6 - Músculo
    "adipose",  # 7 - Tecido adiposo
    "gut",  # 8 - Intestino (absorção)
    "skin",  # 9 - Pele
    "bone",  # 10 - Osso
    "spleen",  # 11 - Baço
    "pancreas",  # 12 - Pâncreas
    "other",  # 13 - Resto do corpo
]

NUM_ORGANS = len(PBPK_ORGANS)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DYNAMIC_GNN_CHECKPOINT = (
    PROJECT_ROOT / "models" / "dynamic_gnn_enriched_v3" / "best_model.pt"
)


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
                "other": 5.0,
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
                "other": 20.0,
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
        super().__init__(aggr="add", flow="target_to_source")

        self.node_feature_dim = node_dim  # Renomeado para evitar conflito
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Attention weights (para órgãos críticos)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
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
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
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
        use_attention: bool = True,
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
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge feature encoder (fluxo + Kp)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # GNN layers para evolução temporal
        self.gnn_layers = nn.ModuleList(
            [
                OrganMessagePassing(
                    node_dim=hidden_dim,
                    edge_dim=hidden_dim // 2,
                    hidden_dim=hidden_dim,
                )
                for _ in range(num_gnn_layers)
            ]
        )

        # Attention para órgãos críticos
        if use_attention:
            self.organ_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, batch_first=True
            )

        # Temporal evolution (RNN-like)
        self.temporal_evolution = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Output head (concentração)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Concentração
            nn.ReLU(),  # Concentração >= 0
        )

        # Critical organs indices (para attention)
        self.critical_organs_idx = [
            PBPK_ORGANS.index("liver"),
            PBPK_ORGANS.index("kidney"),
            PBPK_ORGANS.index("brain"),
        ]

    def build_pbpk_graph(
        self,
        physiological_params: PBPKPhysiologicalParams,
        device: torch.device,
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
            edge_attrs.append([flow, 1.0 / kp, -1.0, 0.0])  # Fluxo reverso

        # Adicionar clearance edges (liver -> blood, kidney -> blood)
        liver_idx = PBPK_ORGANS.index("liver")
        kidney_idx = PBPK_ORGANS.index("kidney")

        # Liver clearance
        if physiological_params.clearance_hepatic > 0:
            edges.append([liver_idx, blood_idx])
            edge_attrs.append(
                [
                    0.0,  # Não é fluxo sanguíneo
                    1.0,
                    0.0,
                    physiological_params.clearance_hepatic,
                ]
            )

        # Kidney clearance
        if physiological_params.clearance_renal > 0:
            edges.append([kidney_idx, blood_idx])
            edge_attrs.append([0.0, 1.0, 0.0, physiological_params.clearance_renal])

        edge_index = (
            torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        )
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
        time_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Simula PBPK para uma única amostra."""
        device = next(self.parameters()).device
        batch_result = self.forward_batch(
            doses=torch.tensor([dose], dtype=torch.float32, device=device),
            physiological_params_batch=[physiological_params],
            time_points=time_points,
        )
        return {
            "concentrations": batch_result["concentrations"][0],
            "time_points": batch_result["time_points"],
            "organ_names": batch_result["organ_names"],
        }

    def forward_batch(
        self,
        doses: torch.Tensor,
        physiological_params_batch: List[PBPKPhysiologicalParams],
        time_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Simula PBPK para um batch de amostras.

        Args:
            doses: tensor [B]
            physiological_params_batch: lista com B instâncias
            time_points: tensor [T] compartilhado (opcional)
        """
        device = next(self.parameters()).device
        if not isinstance(doses, torch.Tensor):
            doses = torch.tensor(doses, dtype=torch.float32, device=device)
        else:
            doses = doses.to(device)

        batch_size = len(physiological_params_batch)
        if batch_size == 0:
            raise ValueError("physiological_params_batch não pode ser vazio.")

        blood_idx = PBPK_ORGANS.index("blood")
        edge_indices = []
        edge_attrs = []
        node_features = []
        initial_concs = torch.zeros(batch_size, NUM_ORGANS, device=device)

        for idx, params in enumerate(physiological_params_batch):
            edge_index_i, edge_attr_i, node_features_i = self.build_pbpk_graph(
                params, device
            )
            node_features_i = node_features_i.clone()

            blood_volume = params.volumes["blood"]
            init_conc = doses[idx].item() / blood_volume
            node_features_i[blood_idx, 2] = init_conc
            initial_concs[idx, blood_idx] = init_conc

            edge_indices.append(edge_index_i + idx * NUM_ORGANS)
            edge_attrs.append(edge_attr_i)
            node_features.append(node_features_i)

        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
        node_features = torch.cat(node_features, dim=0)

        node_emb = self.node_encoder(node_features)
        edge_emb = self.edge_encoder(edge_attr)

        if time_points is None:
            time_points = torch.linspace(
                0,
                self.num_temporal_steps * self.dt,
                self.num_temporal_steps + 1,
                device=device,
            )
        else:
            time_points = time_points.to(device)

        num_time_points = len(time_points)
        num_evolution_steps = num_time_points - 1
        if num_evolution_steps < 1:
            raise ValueError(
                f"num_evolution_steps deve ser >= 1, mas é {num_evolution_steps}"
            )

        current_node_state = node_emb
        h_0 = torch.zeros(2, batch_size, self.hidden_dim, device=device)
        concentrations = []

        for _ in range(num_evolution_steps):
            x = current_node_state
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x, edge_index, edge_emb)

            x = x.view(batch_size, NUM_ORGANS, self.hidden_dim)

            if self.use_attention:
                critical_nodes = x[:, self.critical_organs_idx, :]
                x, _ = self.organ_attention(x, critical_nodes, critical_nodes)

            x_mean = x.mean(dim=1, keepdim=True)
            x_evolved, h_0 = self.temporal_evolution(x_mean, h_0)
            x_evolved_expanded = x_evolved.repeat(1, NUM_ORGANS, 1)
            current_node_state = x_evolved_expanded.view(-1, self.hidden_dim)

            conc = self.output_head(x.reshape(-1, self.hidden_dim))
            conc = conc.view(batch_size, NUM_ORGANS)
            concentrations.append(conc)

        concentrations = torch.stack(concentrations, dim=1)
        concentrations = torch.cat([initial_concs.unsqueeze(1), concentrations], dim=1)
        concentrations = concentrations.permute(0, 2, 1)

        if len(time_points) != concentrations.shape[-1]:
            target_len = len(time_points)
            current_len = concentrations.shape[-1]
            if current_len > target_len:
                concentrations = concentrations[..., :target_len]
            else:
                pad = concentrations[..., -1:].expand(-1, -1, target_len - current_len)
                concentrations = torch.cat([concentrations, pad], dim=-1)

        return {
            "concentrations": concentrations,
            "time_points": time_points,
            "organ_names": PBPK_ORGANS,
        }


class DynamicPBPKSimulator:
    """
    Wrapper para DynamicPBPKGNN com interface similar ao ODE solver.
    """

    def __init__(
        self,
        model: Optional[DynamicPBPKGNN] = None,
        device: str = "cpu",
        checkpoint_path: Optional[Union[str, Path]] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ):
        if model is None:
            model = DynamicPBPKGNN()
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        if checkpoint_path is not None:
            self.load_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=map_location or device,
                strict=strict,
            )

    def simulate(
        self,
        dose: float,
        clearance_hepatic: float = 0.0,
        clearance_renal: float = 0.0,
        partition_coeffs: Optional[Dict[str, float]] = None,
        time_points: Optional[np.ndarray] = None,
        **kwargs,
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
            partition_coeffs=partition_coeffs or {},
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

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> None:
        """
        Carrega pesos treinados do DynamicPBPKGNN.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

        map_location = map_location or self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
            if not isinstance(state_dict, dict):
                raise ValueError(
                    f"Formato de checkpoint inválido em {checkpoint_path}."
                )

        self.model.load_state_dict(state_dict, strict=strict)
        self.model.to(self.device)
        self.model.eval()


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Executa simulação PBPK com Dynamic GNN"
    )
    parser.add_argument(
        "--dose", type=float, default=100.0, help="Dose administrada (mg)"
    )
    parser.add_argument(
        "--clearance-hepatic",
        type=float,
        default=10.0,
        help="Clearance hepático (L/h)",
    )
    parser.add_argument(
        "--clearance-renal",
        type=float,
        default=5.0,
        help="Clearance renal (L/h)",
    )
    parser.add_argument(
        "--partition",
        action="append",
        default=[],
        help="Coeficientes de partição no formato organ=valor (pode repetir)",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=100,
        help="Número de passos temporais",
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Delta t em horas")
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Dimensão interna (hidden_dim)"
    )
    parser.add_argument(
        "--num-gnn-layers", type=int, default=3, help="Número de camadas GNN"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_DYNAMIC_GNN_CHECKPOINT),
        help="Caminho para checkpoint treinado (default: models/dynamic_gnn_enriched_v3/best_model.pt)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument(
        "--log-file", type=str, help="Arquivo Markdown para salvar resumo"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Número de órgãos nos picos exibidos",
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Carregar checkpoint com strict=False",
    )

    args = parser.parse_args()

    partition_coeffs: Dict[str, float] = {}
    for item in args.partition:
        if "=" not in item:
            raise ValueError(
                f"Formato inválido para partition: '{item}'. Use organ=valor"
            )
        organ, value = item.split("=", 1)
        partition_coeffs[organ.strip()] = float(value)

    model = DynamicPBPKGNN(
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_temporal_steps=args.time_steps,
        dt=args.dt,
    )
    simulator = DynamicPBPKSimulator(
        model=model,
        device=args.device,
        checkpoint_path=args.checkpoint,
        strict=not args.relaxed,
    )

    results = simulator.simulate(
        dose=args.dose,
        clearance_hepatic=args.clearance_hepatic,
        clearance_renal=args.clearance_renal,
        partition_coeffs=partition_coeffs if partition_coeffs else None,
    )

    time_points = results["time"]
    blood = results["blood"]
    cmax = float(blood.max())
    t_cmax = float(time_points[blood.argmax()])
    final_blood = float(blood[-1])

    peaks = []
    for organ, series in results.items():
        if organ == "time":
            continue
        series = np.asarray(series)
        peaks.append((organ, float(series.max()), float(series[-1])))
    peaks.sort(key=lambda item: item[1], reverse=True)

    print("=" * 80)
    print("Dynamic GNN PBPK Simulation")
    print("=" * 80)
    print(f"Dose: {args.dose} mg")
    print(f"Clearance hepático: {args.clearance_hepatic} L/h")
    print(f"Clearance renal:   {args.clearance_renal} L/h")
    if partition_coeffs:
        print(f"Partition coeffs:  {partition_coeffs}")
    if args.checkpoint:
        print(f"Checkpoint:        {args.checkpoint}")
    print(f"Time steps:        {args.time_steps}")
    print(f"Δt:                {args.dt} h")
    print("-" * 80)
    print(f"Cmax (blood): {cmax:.4f} mg/L @ t={t_cmax:.2f} h")
    print(f"Final blood: {final_blood:.4f} mg/L")
    print("-" * 80)
    print("Top órgãos (pico)")
    for organ, peak, final in peaks[: args.top]:
        print(f"- {organ:12s}: peak {peak:.4f} mg/L | final {final:.4f} mg/L")
    print("=" * 80)

    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Dynamic GNN PBPK Simulation\n",
            f"Generated: {datetime.now().isoformat()}\n",
            "\n## Settings\n",
            f"- Dose: {args.dose} mg\n",
            f"- Hepatic clearance: {args.clearance_hepatic} L/h\n",
            f"- Renal clearance: {args.clearance_renal} L/h\n",
            f"- Temporal steps: {args.time_steps} (dt={args.dt} h)\n",
            f"- Checkpoint: {args.checkpoint or 'N/A'}\n",
            "\n## Peak Concentrations\n",
        ]
        for organ, peak, final in peaks[: args.top]:
            lines.append(f"- {organ}: peak {peak:.4f} mg/L, final {final:.4f} mg/L\n")
        lines.append("\n## Plasma (blood) kinetics\n")
        lines.append(f"- Cmax: {cmax:.4f} mg/L at t = {t_cmax:.2f} h\n")
        lines.append(f"- Final concentration: {final_blood:.4f} mg/L\n")
        log_path.write_text("".join(lines), encoding="utf-8")
        print(f"Log salvo em {log_path}")
