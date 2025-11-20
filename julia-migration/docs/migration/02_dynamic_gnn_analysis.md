# Análise Linha por Linha: Dynamic GNN

**Arquivo Python:** `apps/pbpk_core/simulation/dynamic_gnn_pbpk.py` (760 linhas)
**Data:** 2025-11-18
**Autor:** AI Assistant + Dr. Demetrios Agourakis

---

## 1. Visão Geral

**Total de linhas:** ~760 linhas
**Função principal:** Dynamic Graph Neural Network para simulação PBPK
**Base científica:** arXiv 2024 (R² 0.9342 vs 0.85-0.90 ODE tradicional)

---

## 2. Análise Linha por Linha

### LINHA 205-285: `DynamicPBPKGNN.__init__()`

```python
def __init__(
    self,
    node_dim: int = 16,
    edge_dim: int = 4,
    hidden_dim: int = 64,
    num_gnn_layers: int = 3,
    num_temporal_steps: int = 100,
    dt: float = 0.1,
    use_attention: bool = True,
):
    super().__init__()

    # Node feature encoder
    self.node_encoder = nn.Sequential(
        nn.Linear(node_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )

    # Edge feature encoder
    self.edge_encoder = nn.Sequential(
        nn.Linear(edge_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, hidden_dim // 2),
    )

    # GNN layers
    self.gnn_layers = nn.ModuleList([
        OrganMessagePassing(...) for _ in range(num_gnn_layers)
    ])

    # Attention
    if use_attention:
        self.organ_attention = nn.MultiheadAttention(...)

    # Temporal evolution
    self.temporal_evolution = nn.GRU(...)

    # Output head
    self.output_head = nn.Sequential(...)
```

**Análise:**
- **PyTorch layers:** `nn.Sequential`, `nn.Linear`, `nn.GRU`
- **Oportunidade Julia:** Flux.jl (`Chain`, `Dense`, `GRU`)

**Refatoração Julia:**
```julia
struct DynamicPBPKGNN
    node_encoder::Chain
    edge_encoder::Chain
    gnn_layers::Vector{GraphNeuralNetworks.GNNLayer}
    temporal_evolution::GRU
    output_head::Chain
    use_attention::Bool
    organ_attention::Union{MultiheadAttention, Nothing}

    function DynamicPBPKGNN(;
        node_dim::Int = 16,
        edge_dim::Int = 4,
        hidden_dim::Int = 64,
        num_gnn_layers::Int = 3,
        num_temporal_steps::Int = 100,
        dt::Float64 = 0.1,
        use_attention::Bool = true,
    )
        # Node encoder
        node_encoder = Chain(
            Dense(node_dim, hidden_dim, relu),
            Dense(hidden_dim, hidden_dim),
        )

        # Edge encoder
        edge_encoder = Chain(
            Dense(edge_dim, hidden_dim ÷ 2, relu),
            Dense(hidden_dim ÷ 2, hidden_dim ÷ 2),
        )

        # GNN layers
        gnn_layers = [OrganMessagePassing(...) for _ in 1:num_gnn_layers]

        # Attention
        organ_attention = use_attention ? MultiheadAttention(...) : nothing

        # Temporal evolution
        temporal_evolution = GRU(hidden_dim, hidden_dim, num_layers=2)

        # Output head
        output_head = Chain(
            Dense(hidden_dim, hidden_dim ÷ 2, relu),
            Dense(hidden_dim ÷ 2, 1),
            x -> relu.(x),  # Concentração >= 0
        )

        new(node_encoder, edge_encoder, gnn_layers, temporal_evolution,
            output_head, use_attention, organ_attention)
    end
end
```

**Inovações:**
1. **Type-safe struct:** Immutable, type-stable
2. **Flux.jl:** Framework ML nativo
3. **GPU-ready:** CUDA.jl integration

---

### LINHA 293-374: `build_pbpk_graph()`

```python
def build_pbpk_graph(
    self,
    physiological_params: PBPKPhysiologicalParams,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Construir edges (conexões entre órgãos via sangue)
    blood_idx = PBPK_ORGANS.index("blood")

    edges = []
    edge_attrs = []

    for i, organ in enumerate(PBPK_ORGANS):
        if i == blood_idx:
            continue

        # Blood -> Organ
        edges.append([blood_idx, i])
        flow = physiological_params.blood_flows.get(organ, 0.0)
        kp = physiological_params.partition_coeffs.get(organ, 1.0)
        edge_attrs.append([flow, kp, 1.0, 0.0])

        # Organ -> Blood
        edges.append([i, blood_idx])
        edge_attrs.append([flow, 1.0 / kp, -1.0, 0.0])

    # Clearance edges
    # ...

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32, device=device)

    # Node features
    node_features = []
    for organ in PBPK_ORGANS:
        # ...
        node_features.append(features)

    node_features = torch.tensor(node_features, dtype=torch.float32, device=device)

    return edge_index, edge_attr, node_features
```

**Análise:**
- **Graph construction:** Manual (listas Python)
- **Oportunidade:** GraphNeuralNetworks.jl (type-safe)

**Refatoração Julia:**
```julia
function build_pbpk_graph(
    p::PBPKParams,
    device::AbstractDevice = cpu,
)::Tuple{GNNGraph, SVector{14, Float64}}
    # Construir edges (type-safe)
    edges = Vector{Tuple{Int, Int}}()
    edge_attrs = Vector{SVector{4, Float64}}()

    for i in 1:NUM_ORGANS
        if i == BLOOD_IDX
            continue
        end

        # Blood -> Organ
        push!(edges, (BLOOD_IDX, i))
        flow = p.blood_flows[i]
        kp = p.partition_coeffs[i]
        push!(edge_attrs, SVector(flow, kp, 1.0, 0.0))

        # Organ -> Blood
        push!(edges, (i, BLOOD_IDX))
        push!(edge_attrs, SVector(flow, 1.0 / kp, -1.0, 0.0))
    end

    # Clearance edges
    # ...

    # Criar grafo (GraphNeuralNetworks.jl)
    g = GNNGraph(edges, edge_attr=edge_attrs)

    # Node features (stack-allocated)
    node_features = SVector{14, Float64}([
        p.volumes[i],
        (i == LIVER_IDX ? p.clearance_hepatic : (i == KIDNEY_IDX ? p.clearance_renal : 0.0)),
        0.0,  # initial concentration
        # ... outros features
    ] for i in 1:NUM_ORGANS)

    return g, node_features
end
```

**Inovações:**
1. **GraphNeuralNetworks.jl:** Type-safe graph construction
2. **Stack allocation:** SVector para node features
3. **Type-stable:** Zero overhead

---

### LINHA 395-505: `forward_batch()`

```python
def forward_batch(
    self,
    doses: torch.Tensor,
    physiological_params_batch: List[PBPKPhysiologicalParams],
    time_points: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    batch_size = len(physiological_params_batch)

    # Construir grafos para batch
    edge_indices = []
    edge_attrs = []
    node_features = []

    for idx, params in enumerate(physiological_params_batch):
        edge_index_i, edge_attr_i, node_features_i = self.build_pbpk_graph(params, device)
        # ...
        edge_indices.append(edge_index_i + idx * NUM_ORGANS)
        # ...

    edge_index = torch.cat(edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs, dim=0)
    node_features = torch.cat(node_features, dim=0)

    # Forward pass
    node_emb = self.node_encoder(node_features)
    edge_emb = self.edge_encoder(edge_attr)

    # Temporal evolution
    for _ in range(num_evolution_steps):
        # Message passing
        x = current_node_state
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_emb)

        # Attention
        if self.use_attention:
            x, _ = self.organ_attention(x, critical_nodes, critical_nodes)

        # GRU
        x_evolved, h_0 = self.temporal_evolution(x_mean, h_0)

        # Output
        conc = self.output_head(x)
        concentrations.append(conc)

    return {
        "concentrations": torch.stack(concentrations, dim=1),
        "time_points": time_points,
        "organ_names": PBPK_ORGANS,
    }
```

**Análise:**
- **Bottleneck:** Graph construction para batch
- **Bottleneck:** Message passing loop
- **Oportunidade:** GraphNeuralNetworks.jl batching, GPU acceleration

**Refatoração Julia:**
```julia
function forward_batch(
    model::DynamicPBPKGNN,
    doses::Vector{Float64},
    params_batch::Vector{PBPKParams},
    time_points::Vector{Float64} = collect(TIME_POINTS),
    device::AbstractDevice = cpu,
)
    batch_size = length(params_batch)

    # Construir batch graph (GraphNeuralNetworks.jl)
    graphs = [build_pbpk_graph(p, device) for p in params_batch]
    batch_graph = batch(graphs)  # Type-safe batching

    # Node/edge encoders
    node_emb = model.node_encoder(batch_graph.node_features)
    edge_emb = model.edge_encoder(batch_graph.edge_attr)

    # Temporal evolution
    concentrations = Vector{Matrix{Float64}}()
    h_0 = zeros(2, batch_size, hidden_dim)

    for t_idx in 1:(length(time_points) - 1)
        # Message passing
        x = node_emb
        for gnn_layer in model.gnn_layers
            x = gnn_layer(batch_graph, x, edge_emb)
        end

        # Attention
        if model.use_attention
            x = model.organ_attention(x, critical_nodes, critical_nodes)
        end

        # GRU
        x_evolved, h_0 = model.temporal_evolution(x_mean, h_0)

        # Output
        conc = model.output_head(x)
        push!(concentrations, conc)
    end

    return Dict(
        "concentrations" => stack(concentrations, dims=2),
        "time_points" => time_points,
        "organ_names" => PBPK_ORGANS,
    )
end
```

**Inovações:**
1. **GraphNeuralNetworks.jl batching:** Type-safe, eficiente
2. **GPU acceleration:** CUDA.jl (melhor que PyTorch)
3. **Type-stable:** Zero overhead
4. **Automatic differentiation:** Zygote.jl nativo

---

## 3. Ganhos de Performance Esperados

### GNN Forward Pass:
- **Python (PyTorch):** ~5-10ms por batch
- **Julia (Flux.jl + CUDA.jl):** Similar ou melhor
- **Ganho:** GPU acceleration nativo, type stability

### Training:
- **Python (PyTorch):** ~100-200ms por epoch (batch 8)
- **Julia (Flux.jl):** Similar ou melhor
- **Ganho:** Automatic differentiation nativo, type stability

---

## 4. Próximos Passos

1. ✅ Análise linha por linha - **CONCLUÍDA**
2. ⏳ Implementação Julia - **PRÓXIMO**
3. ⏳ Testes unitários - **PENDENTE**
4. ⏳ Benchmark vs PyTorch - **PENDENTE**

---

**Última atualização:** 2025-11-18

