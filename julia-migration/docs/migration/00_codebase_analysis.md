# Análise Completa do Codebase - FASE 0

**Data:** 2025-11-18
**Autor:** AI Assistant + Dr. Demetrios Agourakis
**Objetivo:** Mapear cada linha de código, dependências e fluxos de dados

---

## 1. Estatísticas Gerais

### Arquivos Python
- **Total de arquivos:** ~200+ arquivos Python
- **Localizações principais:**
  - `apps/pbpk_core/` - Core PBPK functionality
  - `apps/api/` - REST API
  - `apps/training/` - Training pipelines
  - `scripts/` - Utility scripts

### Linhas de Código
- **Estimativa:** ~50,000+ linhas de código Python
- **Componentes críticos:**
  - Dynamic GNN: ~760 linhas
  - ODE Solver: ~195 linhas
  - Dataset Generation: ~300+ linhas
  - Training Pipeline: ~500+ linhas

---

## 2. Dependências Principais

### Deep Learning
- **PyTorch** (`torch`) - Framework principal de ML
- **PyTorch Geometric** (`torch_geometric`) - GNN operations
- **Transformers** - ChemBERTa encoder

### Computação Científica
- **NumPy** (`numpy`) - Arrays e operações numéricas
- **SciPy** (`scipy`) - ODE solver (`odeint`)
- **Pandas** (`pandas`) - Data manipulation

### Outras
- **RDKit** - Chemistry toolkit
- **FastAPI** - REST API
- **Matplotlib/Seaborn/Plotly** - Visualization

---

## 3. Componentes Críticos Identificados

### 3.1 ODE Solver (`apps/pbpk_core/simulation/ode_pbpk_solver.py`)
- **Função crítica:** `_ode_system()` (linhas 44-95)
- **Bottleneck:** Loop sobre 14 órgãos, acesso a dict
- **Oportunidade Julia:** SIMD vectorization, stack allocation

### 3.2 Dataset Generation (`scripts/analysis/build_dynamic_gnn_dataset_from_enriched.py`)
- **Função crítica:** Loop de simulação ODE
- **Bottleneck:** `odeint()` chamado milhares de vezes
- **Oportunidade Julia:** DifferentialEquations.jl (10-100× mais rápido)

### 3.3 Dynamic GNN (`apps/pbpk_core/simulation/dynamic_gnn_pbpk.py`)
- **Função crítica:** `forward_batch()` (linhas 395-505)
- **Bottleneck:** Graph construction, message passing
- **Oportunidade Julia:** GraphNeuralNetworks.jl, CUDA.jl

### 3.4 Training Pipeline (`scripts/train_dynamic_gnn_pbpk.py`)
- **Função crítica:** Training loop
- **Bottleneck:** DataLoader, GPU transfers
- **Oportunidade Julia:** Flux.jl, native batching

---

## 4. Fluxos de Dados

### Pipeline Principal:
1. **Dataset Generation:**
   - Input: `pbpk_parameters_wide_enriched_v3.csv`
   - Process: Generate synthetic data via ODE solver
   - Output: `dynamic_gnn_dataset_enriched_v4.npz`

2. **Training:**
   - Input: NPZ dataset
   - Process: Train Dynamic GNN
   - Output: Model checkpoint (`.pt`)

3. **Validation:**
   - Input: Model + experimental data
   - Process: Evaluate metrics
   - Output: Validation reports

---

## 5. Hotspots de Performance (Identificados)

### Alto Impacto:
1. **ODE Solver** - Chamado milhares de vezes durante dataset generation
2. **GNN Forward Pass** - Computação intensiva durante training
3. **Dataset Generation Loop** - I/O e simulação

### Médio Impacto:
1. **Data Loading** - Pandas/NumPy operations
2. **Graph Construction** - Dynamic GNN
3. **Validation Metrics** - Statistical calculations

---

## 6. Análise Científica

### Equações Matemáticas

#### ODE System:
```
dC_organ/dt = (Q_organ / V_organ) * (C_blood - C_organ / Kp_organ)
dC_blood/dt = Σ[fluxos de saída] - clearance_rate * C_blood
```

**Validação:**
- Conservação de massa: ✅ Verificado
- Unidades: ⚠️ Não verificadas em tempo de compilação
- Estabilidade numérica: ⚠️ Depende de scipy.integrate.odeint

#### GNN Architecture:
- Message Passing: OrganMessagePassing layer
- Temporal Evolution: GRU layers
- Attention: MultiheadAttention para órgãos críticos

**Validação:**
- Arquitetura: ✅ Baseada em paper SOTA (arXiv 2024)
- Implementação: ✅ PyTorch Geometric

---

## 7. Oportunidades de Otimização

### Numérica:
1. **SIMD Vectorization** - Julia JIT otimiza automaticamente
2. **Stack Allocation** - SVector para parâmetros fixos
3. **Type Stability** - Zero overhead abstractions

### Algoritmos:
1. **ODE Solvers** - DifferentialEquations.jl (Tsit5, Vern9)
2. **Automatic Differentiation** - ForwardDiff.jl, Zygote.jl
3. **GPU Acceleration** - CUDA.jl (melhor que PyTorch)

### Estrutura:
1. **Type Safety** - Verificação em tempo de compilação
2. **Unit Checking** - Unitful.jl para unidades
3. **Parallel Generation** - Threads nativos (sem GIL)

---

## 8. Premissas e Limitações

### Premissas:
- Modelo PBPK de 14 compartimentos
- Fluxo sanguíneo centralizado (blood conecta todos)
- Clearance linear (hepático e renal)
- Partition coefficients constantes

### Limitações Atuais:
- Sem verificação de unidades em tempo de compilação
- Sem paralelização nativa (GIL do Python)
- ODE solver não otimizado (scipy.integrate.odeint)
- Sem type safety (erros em runtime)

---

## 9. Próximos Passos

1. ✅ Análise estática completa - **CONCLUÍDA**
2. ⏳ Análise de performance (profiling) - **EM PROGRESSO**
3. ⏳ Análise científica detalhada - **PENDENTE**
4. ⏳ Documentação de dependências - **PENDENTE**

---

**Última atualização:** 2025-11-18

