# Guia de Execução - Migração Julia

**Data:** 2025-11-18
**Status:** Pronto para execução

---

## 🎯 Pré-requisitos

### 1. Instalar Julia 1.9+

```bash
# Linux (via juliaup)
curl -fsSL https://install.julialang.org | sh

# Ou baixar de: https://julialang.org/downloads/
```

### 2. Verificar Instalação

```bash
julia --version
# Deve mostrar: julia version 1.9.x ou superior
```

---

## 🚀 Setup Inicial

### 1. Ativar Ambiente Julia

```bash
cd /home/agourakis82/workspace/darwin-pbpk-platform/julia-migration
julia
```

### 2. Instalar Dependências

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Isso irá instalar todas as dependências listadas em `Project.toml`:
- DifferentialEquations.jl
- Flux.jl
- GraphNeuralNetworks.jl
- CUDA.jl
- Unitful.jl
- HTTP.jl
- E mais...

**Tempo estimado:** 5-15 minutos (dependendo da conexão)

---

## ✅ Validação Inicial

### 1. Testar Importação do Módulo

```julia
using DarwinPBPK
```

Se não houver erros, o módulo foi carregado com sucesso.

### 2. Executar Testes Unitários

```julia
using Pkg
Pkg.test("DarwinPBPK")
```

**Testes incluídos:**
- `test/test_ode_solver.jl` - Validação do ODE solver
- `test/test_complete.jl` - Testes completos do sistema

**Tempo estimado:** 1-5 minutos

---

## 📊 Benchmarks de Performance

### 1. Benchmark do ODE Solver

```julia
include("benchmarks/benchmark_ode_solver.jl")
```

**O que mede:**
- Tempo de execução do ODE solver
- Comparação com Python (se disponível)
- Validação de conservação de massa

**Ganho esperado:** 50-500× mais rápido que Python

### 2. Benchmark Completo

```julia
include("benchmarks/benchmark_complete.jl")
```

**O que mede:**
- Performance end-to-end
- Memory usage
- GPU acceleration (se disponível)

---

## 🔬 Validação Científica

### 1. Validação Numérica vs Python

```julia
using DarwinPBPK

# Carregar modelo Python (se disponível)
# Comparar resultados ODE solver
# Validar erro relativo < 1e-6
```

### 2. Validação em Dados Experimentais

```julia
using DarwinPBPK.Validation

# Carregar dados experimentais
# Executar predições
# Calcular métricas regulatórias:
#   - Fold Error (FE)
#   - Geometric Mean Fold Error (GMFE)
#   - % within 1.25x, 1.5x, 2.0x
#   - R², MAE, RMSE (log10 scale)
```

---

## 🎓 Exemplos de Uso

### 1. ODE Solver

```julia
using DarwinPBPK.ODESolver

# Criar parâmetros fisiológicos
params = PBPKPhysiologicalParams(
    clearance_hepatic=10.0,  # L/h
    clearance_renal=5.0,     # L/h
    partition_coeffs=Dict(
        "liver" => 2.0,
        "kidney" => 1.5,
        "brain" => 0.5,
        # ... outros órgãos
    )
)

# Simular PBPK
time_points = 0.0:0.1:24.0
result = solve_ode(100.0, params, collect(time_points))

# Acessar concentrações
concentrations = result.u  # Array de concentrações por órgão
```

### 2. Dataset Generation

```julia
using DarwinPBPK.DatasetGeneration
using DataFrames, CSV

# Carregar parâmetros
params_df = CSV.read("path/to/pbpk_parameters_wide_enriched_v3.csv", DataFrame)

# Gerar dataset
generate_dataset(
    params_df,
    "output_dataset.npz";
    max_samples=1000,
    dose_min=50.0,
    dose_max=200.0,
    noise_kp_std=0.15,
    noise_clear_frac=0.10
)
```

### 3. Dynamic GNN

```julia
using DarwinPBPK.DynamicGNN
using CUDA

# Criar modelo
model = DynamicPBPKGNN(
    node_dim=16,
    edge_dim=4,
    hidden_dim=64,
    num_gnn_layers=3,
    num_temporal_steps=100,
    dt=0.1,
    use_attention=true
)

# Mover para GPU (se disponível)
if CUDA.functional()
    model = model |> gpu
end

# Fazer predição
dose = 100.0
params = PBPKPhysiologicalParams(...)
time_points = collect(0.0:0.1:24.0)

result = model(dose, params, time_points)
concentrations = result.concentrations
```

### 4. Training

```julia
using DarwinPBPK.Training
using Flux

# Carregar dataset
dataset = PBPKDataset("path/to/dataset.npz")

# Criar modelo
model = DynamicPBPKGNN(...)

# Configurar otimizador
optimizer = Adam(0.001)

# Treinar
train_model(
    model,
    dataset,
    optimizer;
    epochs=100,
    batch_size=32,
    device="cuda"  # ou "cpu"
)
```

### 5. Validation

```julia
using DarwinPBPK.Validation

# Carregar modelo treinado
model = load_model("path/to/checkpoint.jl")

# Carregar dados de validação
val_dataset = PBPKDataset("path/to/validation.npz")

# Avaliar
metrics, pred, true = evaluate_model_scientific(
    model,
    val_dataset,
    "cuda"
)

# Métricas disponíveis:
println("GMFE: ", metrics["geometric_mean_fold_error"])
println("% within 2.0x: ", metrics["percent_within_2.0x"])
println("R²: ", metrics["r2"])
```

---

## 🐛 Troubleshooting

### Erro: "Package not found"

```julia
# Atualizar registro de pacotes
using Pkg
Pkg.update()
Pkg.instantiate()
```

### Erro: "CUDA not available"

```julia
# Verificar se CUDA está instalado
using CUDA
CUDA.functional()  # Deve retornar true

# Se false, instalar CUDA toolkit:
# https://developer.nvidia.com/cuda-downloads
```

### Erro: "Out of memory"

```julia
# Reduzir batch size
batch_size = 16  # ao invés de 32

# Ou usar CPU
device = "cpu"
```

### Performance lenta na primeira execução

**Normal!** Julia usa JIT compilation. A primeira execução compila o código. Execuções subsequentes serão muito mais rápidas.

---

## 📈 Próximos Passos

1. ✅ Executar testes unitários
2. ✅ Executar benchmarks
3. ✅ Validação numérica vs Python
4. ✅ Validação científica completa
5. ⏳ Otimização final de hotspots (FASE 6)

---

## 📚 Documentação Adicional

- `README.md` - Visão geral do projeto
- `EXECUTIVE_SUMMARY.md` - Resumo executivo
- `docs/migration/` - Análises detalhadas linha por linha
- `docs/SCIENTIFIC_VALIDATION_REPORT.md` - Validação científica
- `docs/NATURE_TIER_DOCUMENTATION.md` - Documentação Nature-tier

---

**Última atualização:** 2025-11-18

