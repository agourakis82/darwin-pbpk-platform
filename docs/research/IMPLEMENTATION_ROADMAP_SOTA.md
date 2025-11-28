# 🚀 Roadmap de Implementação SOTA - Darwin PBPK Platform

**Data:** 2025-11-18
**Autor:** Dr. Demetrios Agourakis + AI Assistant
**Baseado em:** Literatura SOTA Q1 2025

---

## 📋 Visão Geral

Este documento detalha o roadmap para atualizar os modelos do Darwin PBPK Platform para o estado da arte (SOTA) Q1 2025, baseado na análise comparativa com a literatura recente.

---

## 🎯 Fase 1: Prioridades ALTAS (2 semanas)

### 1.1 Implementar ChemBERTa Real

**Status Atual:** Placeholder
**SOTA Q1 2025:** ChemBERTa v2 (2024) ou modelos mais recentes

**Implementação:**
```julia
# julia-migration/src/DarwinPBPK/ml/multimodal_encoder.jl

using Transformers
using Transformers.TextEncoders

struct ChemBERTaEncoder
    model::Any  # Transformers.jl model
    tokenizer::Any
    device

    function ChemBERTaEncoder(device = cpu)
        # Carregar modelo ChemBERTa
        # Opções:
        # 1. seyonec/ChemBERTa-zinc-base-v1 (768d)
        # 2. DeepChem/ChemBERTa-77M-MLM (mais recente)
        model, tokenizer = load_model("seyonec/ChemBERTa-zinc-base-v1")
        new(model, tokenizer, device)
    end
end

function (encoder::ChemBERTaEncoder)(smiles::String)::Vector{Float64}
    # Tokenizar
    tokens = encode(encoder.tokenizer, smiles)

    # Encoder
    embedding = encoder.model(tokens)

    # Pooling (mean ou [CLS] token)
    pooled = mean(embedding, dims=1)  # [1, 768]

    return vec(pooled)  # [768]
end
```

**Dependências:**
- `Transformers.jl` (já no Project.toml)
- Verificar compatibilidade com HuggingFace models

**Testes:**
- Testar tokenization de SMILES
- Validar dimensões de embedding (768d)
- Benchmark de performance

**Tempo Estimado:** 2-3 dias

---

### 1.2 Implementar GNN Encoder Real

**Status Atual:** Placeholder
**SOTA Q1 2025:** GraphMVP, 3D-Mol, ou D-MPNN

**Implementação:**
```julia
# julia-migration/src/DarwinPBPK/ml/multimodal_encoder.jl

using GraphNeuralNetworks
using Flux

struct GNNEncoder
    gnn_layers::Vector{Any}  # GATConv ou GCNConv
    pooling::Any  # Global pooling
    device

    function GNNEncoder(device = cpu)
        # Opção 1: GAT (Graph Attention Network)
        gnn_layers = [
            GATConv(20 => 128, num_heads=4),  # node features: 20
            GATConv(128 => 256, num_heads=4),
            GATConv(256 => 256, num_heads=1),  # Final layer
        ]

        # Opção 2: D-MPNN (Directed Message Passing Neural Network)
        # Implementação customizada necessária

        # Global pooling (mean ou attention)
        pooling = GlobalAttentionPool(Dense(256, 1))

        new(gnn_layers, pooling, device)
    end
end

function (encoder::GNNEncoder)(graph::GNNGraph)::Vector{Float64}
    x = graph.x  # Node features [num_nodes, node_dim]

    # Message passing
    for layer in encoder.gnn_layers
        x = layer(graph, x)
    end

    # Global pooling
    graph_pooled = encoder.pooling(graph, x)  # [256]

    return vec(graph_pooled)
end
```

**Dependências:**
- `GraphNeuralNetworks.jl` (já no Project.toml)
- Verificar suporte para GATConv

**Testes:**
- Testar com grafos moleculares (RDKit → GNNGraph)
- Validar dimensões de embedding (256d)
- Benchmark de performance

**Tempo Estimado:** 2-3 dias

---

### 1.3 Implementar GRU para Temporal Evolution

**Status Atual:** Chain simples (Dense)
**SOTA Q1 2025:** GRU/LSTM ou Neural ODEs

**Implementação:**
```julia
# julia-migration/src/DarwinPBPK/dynamic_gnn.jl

# ATUAL (simplificado):
# temporal_evolution = Chain(
#     Dense(hidden_dim, hidden_dim, tanh),
#     Dense(hidden_dim, hidden_dim)
# )

# SOTA (GRU):
temporal_evolution = Chain(
    Flux.Recur(Flux.GRUCell(hidden_dim, hidden_dim)),
    Flux.Recur(Flux.GRUCell(hidden_dim, hidden_dim))
)

# OU (alternativa mais simples):
temporal_evolution = Flux.Recur(
    Flux.GRUCell(hidden_dim, hidden_dim)
)
```

**Modificações em `forward_batch`:**
```julia
# ATUAL:
x_evolved = model.temporal_evolution(x_mean_flat)

# SOTA (com estado):
state = Flux.Zeros(hidden_dim)  # Estado inicial
for t in 1:num_evolution_steps
    x_evolved, state = model.temporal_evolution(x_mean_flat, state)
    # ...
end
```

**Testes:**
- Comparar performance vs. Chain simples
- Validar estabilidade numérica
- Benchmark de velocidade

**Tempo Estimado:** 1-2 dias

---

## 🎯 Fase 2: Prioridades MÉDIAS (3 semanas)

### 2.1 Implementar Cross-Attention Fusion

**Status Atual:** Concatenação simples
**SOTA Q1 2025:** Cross-Attention, Hierarchical Fusion

**Implementação:**
```julia
# julia-migration/src/DarwinPBPK/ml/multimodal_encoder.jl

struct CrossAttentionFusion
    q_proj::Dense  # Query projection
    k_proj::Dense  # Key projection
    v_proj::Dense  # Value projection
    output_proj::Dense
    num_heads::Int
    head_dim::Int

    function CrossAttentionFusion(
        input_dims::Vector{Int},
        output_dim::Int = FUSION_DIM,
        num_heads::Int = 8,
    )
        # Assumir primeira modalidade como query, outras como key/value
        query_dim = input_dims[1]
        key_dim = sum(input_dims[2:end])

        head_dim = output_dim ÷ num_heads

        q_proj = Dense(query_dim, output_dim)
        k_proj = Dense(key_dim, output_dim)
        v_proj = Dense(key_dim, output_dim)
        output_proj = Dense(output_dim, output_dim)

        new(q_proj, k_proj, v_proj, output_proj, num_heads, head_dim)
    end
end

function (fusion::CrossAttentionFusion)(embeddings::Vector{Vector{Float64}})::Vector{Float64}
    # Separar query e key/value
    query = embeddings[1]  # ChemBERTa
    keys_values = vcat(embeddings[2:end]...)  # GNN, SchNet, etc.

    # Projections
    Q = fusion.q_proj(query)  # [output_dim]
    K = fusion.k_proj(keys_values)  # [output_dim]
    V = fusion.v_proj(keys_values)  # [output_dim]

    # Multi-head attention (simplificado)
    # Q, K, V: [output_dim]
    # Reshape para [num_heads, head_dim]
    Q_reshaped = reshape(Q, fusion.num_heads, fusion.head_dim)
    K_reshaped = reshape(K, fusion.num_heads, fusion.head_dim)
    V_reshaped = reshape(V, fusion.num_heads, fusion.head_dim)

    # Attention scores
    scores = Q_reshaped * K_reshaped' ./ sqrt(fusion.head_dim)  # [num_heads, num_heads]
    attn_weights = softmax(scores, dims=2)

    # Weighted sum
    attn_output = attn_weights * V_reshaped  # [num_heads, head_dim]
    attn_output_flat = vec(attn_output)  # [output_dim]

    # Output projection
    output = fusion.output_proj(attn_output_flat)

    return output
end
```

**Testes:**
- Comparar com concatenação simples
- Validar atenção entre modalidades
- Benchmark de performance

**Tempo Estimado:** 3-4 dias

---

### 2.2 Implementar Loss Evidential Completo

**Status Atual:** Loss simplificado
**SOTA Q1 2025:** Evidential Loss v2 com Distributions.jl

**Implementação:**
```julia
# julia-migration/src/DarwinPBPK/ml/evidential.jl

using Distributions

function evidential_loss(
    α::AbstractMatrix,
    β::AbstractMatrix,
    γ::AbstractMatrix,
    ν::AbstractMatrix,
    y_true::AbstractMatrix,
    λ::Float64 = 0.1,
)
    # Normal-Inverse-Gamma (NIG) distribution
    # p(y | μ, σ²) = N(y | μ, σ²)
    # p(μ, σ² | α, β, γ, ν) = NIG(μ, σ² | α, β, γ, ν)

    # Parâmetros da NIG
    μ = γ  # Mean
    σ² = β ./ ((α .- 1.0) .* ν)  # Variance

    # Negative log-likelihood (NLL)
    # NLL = -log p(y | α, β, γ, ν)
    # Para NIG: NLL = log(Γ(ν)) - log(Γ(α)) + (α - 1) * log(β)
    #           - ν * log(ν * β + (y - γ)²)

    nll = sum(
        loggamma.(ν) .- loggamma.(α) .+
        (α .- 1.0) .* log.(β) .-
        ν .* log.(ν .* β .+ (y_true .- γ).^2)
    )

    # Regularização (encorajar alta confiança)
    # Penalizar baixa confiança (alta incerteza)
    reg = λ * sum(1.0 ./ (α .+ β))

    return nll + reg
end
```

**Testes:**
- Comparar com loss simplificado
- Validar calibração de incerteza
- Testar com diferentes valores de λ

**Tempo Estimado:** 2-3 dias

---

### 2.3 Adicionar Modalidades Adicionais

**Status Atual:** Apenas ChemBERTa e GNN (placeholders)
**SOTA Q1 2025:** 5-7 modalidades

#### a) SchNet (3D Encoder)

```julia
# TODO: Implementar SchNet encoder
# SchNet usa geometria 3D molecular (coordenadas atômicas)
# Requer: RDKit para obter coordenadas 3D
```

#### b) 3D Conformers

```julia
# TODO: Implementar encoder de conformers 3D
# Usar RDKit para gerar conformers
# Encoder: GraphNeuralNetworks.jl com edge features 3D
```

#### c) QM Features

```julia
# TODO: Implementar descritores quânticos
# Usar RDKit ou pacotes QM (DFT, MP2)
# Features: HOMO, LUMO, gap, dipole, etc.
```

**Tempo Estimado:** 5-7 dias (total)

---

## 🎯 Fase 3: Prioridades BAIXAS (4 semanas)

### 3.1 Implementar Neural ODEs

**Status Atual:** Apenas ODEs físicas
**SOTA Q1 2025:** Neural ODEs para PBPK

**Implementação:**
```julia
# julia-migration/src/DarwinPBPK/neural_ode.jl

using DifferentialEquations
using Flux

struct NeuralODEPBPK
    neural_network::Chain
    ode_solver::Any
end

function neural_ode!(du, u, p, t)
    # Neural network prediz du/dt
    # u: estado atual (concentrações)
    # p: parâmetros (dose, clearance, etc.)
    # t: tempo
    du .= neural_network(vcat(u, p, [t]))
end

function solve_neural_ode(
    model::NeuralODEPBPK,
    u0::Vector{Float64},
    tspan::Tuple{Float64, Float64},
    p::Vector{Float64},
)
    prob = ODEProblem(neural_ode!, u0, tspan, p)
    sol = solve(prob, model.ode_solver)
    return sol
end
```

**Tempo Estimado:** 7-10 dias

---

### 3.2 Edge-Enhanced Message Passing

**Status Atual:** Edge features básicos (4 dims)
**SOTA Q1 2025:** Edge features enriquecidos (8-16 dims)

**Implementação:**
```julia
# Enriquecer edge features:
# - Flow direction (entrada/saída)
# - Clearance type (hepatic/renal/none)
# - Organ hierarchy (nível na hierarquia)
# - Temporal dynamics (taxa de mudança)
# - Partition coefficient (Kp)
# - Volume ratio
```

**Tempo Estimado:** 5-7 dias

---

## 📊 Métricas de Sucesso

### Performance Esperada (Após Implementações)

| Métrica | Atual | Meta (SOTA) | Status |
|---------|-------|-------------|--------|
| **GMFE (Cmax)** | 1.036 | < 1.05 | ✅ |
| **% Within 2× (Cmax)** | 100% | > 95% | ✅ |
| **R² (Cmax)** | - | > 0.90 | ⏳ |
| **R² (AUC)** | - | > 0.95 | ⏳ |
| **Uncertainty Calibration (ECE)** | - | < 0.05 | ⏳ |

---

## 🧪 Plano de Testes

### Testes Unitários

1. **ChemBERTa Encoder:**
   - Testar tokenization de SMILES
   - Validar dimensões (768d)
   - Benchmark de velocidade

2. **GNN Encoder:**
   - Testar com grafos moleculares
   - Validar dimensões (256d)
   - Benchmark de velocidade

3. **GRU Temporal Evolution:**
   - Comparar com Chain simples
   - Validar estabilidade numérica
   - Testar com diferentes hidden_dims

4. **Cross-Attention Fusion:**
   - Testar com múltiplas modalidades
   - Validar atenção entre modalidades
   - Benchmark de performance

5. **Evidential Loss:**
   - Comparar com loss simplificado
   - Validar calibração de incerteza
   - Testar com diferentes λ

### Testes de Integração

1. **Pipeline Completo:**
   - Multimodal Encoder → Dynamic GNN → Evidential Head
   - Validar end-to-end
   - Benchmark de performance

2. **Validação Científica:**
   - Comparar com ODE solver
   - Validar métricas científicas (GMFE, R², etc.)
   - Testar com dataset de validação

---

## 📅 Timeline

### Semana 1-2: Fase 1 (Prioridades ALTAS)
- [ ] Implementar ChemBERTa Real (2-3 dias)
- [ ] Implementar GNN Encoder Real (2-3 dias)
- [ ] Implementar GRU para Temporal Evolution (1-2 dias)
- [ ] Testes e validação (2-3 dias)

### Semana 3-5: Fase 2 (Prioridades MÉDIAS)
- [ ] Implementar Cross-Attention Fusion (3-4 dias)
- [ ] Implementar Loss Evidential Completo (2-3 dias)
- [ ] Adicionar Modalidades Adicionais (5-7 dias)
- [ ] Testes e validação (3-4 dias)

### Semana 6-9: Fase 3 (Prioridades BAIXAS)
- [ ] Implementar Neural ODEs (7-10 dias)
- [ ] Edge-Enhanced Message Passing (5-7 dias)
- [ ] Testes e validação (3-4 dias)

### Semana 10: Validação Final
- [ ] Comparação com literatura SOTA
- [ ] Relatório final
- [ ] Preparação para publicação

---

## 📚 Referências Críticas

### Artigos-Chave (2024-2025)

1. **Graph Neural Networks:**
   - "Graph Transformer Networks" (2024)
   - "Temporal Graph Networks" (2024)
   - "Neural ODEs for Graphs" (2024)

2. **Multimodal Encoders:**
   - "ChemBERTa v2" (2024)
   - "GraphMVP" (2024)
   - "Cross-Modal Attention" (2024)

3. **Evidential Learning:**
   - "Evidential Deep Learning v2" (2024)
   - "Calibrated Uncertainty" (2024)

4. **Neural ODEs:**
   - "Physics-Informed Neural Networks" (2024)
   - "Neural SDEs" (2024)

---

## ✅ Checklist de Implementação

### Fase 1 (Prioridades ALTAS)
- [ ] ChemBERTa Real implementado
- [ ] GNN Encoder Real implementado
- [ ] GRU para Temporal Evolution implementado
- [ ] Testes unitários passando
- [ ] Benchmark de performance

### Fase 2 (Prioridades MÉDIAS)
- [ ] Cross-Attention Fusion implementado
- [ ] Loss Evidential Completo implementado
- [ ] Modalidades Adicionais implementadas
- [ ] Testes de integração passando

### Fase 3 (Prioridades BAIXAS)
- [ ] Neural ODEs implementado
- [ ] Edge-Enhanced Message Passing implementado
- [ ] Validação científica completa

---

**Status:** Roadmap criado. Próximo passo: Iniciar Fase 1 (Prioridades ALTAS).

