# Tutorial Completo - Darwin PBPK Platform (Julia)

**Data:** 2025-11-18
**Autor:** Dr. Demetrios Agourakis + AI Assistant

---

## 🚀 Início Rápido

### 1. Instalação

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### 2. Uso Básico

```julia
using DarwinPBPK
using DarwinPBPK.ODEPBPKSolver

# Criar parâmetros PBPK
params = ODEPBPKSolver.PBPKParams(
    clearance_hepatic=10.0,  # L/h
    clearance_renal=5.0,      # L/h
    partition_coeffs=Dict(
        "liver" => 2.0,
        "kidney" => 1.5,
        "brain" => 0.5
    )
)

# Simular
result = ODEPBPKSolver.simulate(params, 100.0; t_max=24.0, num_points=100)

# Acessar resultados
blood_conc = result["blood"]
time_points = result["time"]
```

---

## 📚 Exemplos Práticos

### Exemplo 1: Simulação Básica

```julia
using DarwinPBPK.ODEPBPKSolver

# Parâmetros padrão
params = ODEPBPKSolver.PBPKParams()

# Simular dose de 100 mg por 24 horas
result = ODEPBPKSolver.simulate(params, 100.0; t_max=24.0)

# Plotar (requer Plots.jl)
using Plots
plot(result["time"], result["blood"], label="Blood")
```

### Exemplo 2: Dynamic GNN

```julia
using DarwinPBPK.DynamicGNN

# Criar modelo
model = DynamicGNN.DynamicPBPKGNN(
    node_dim=16,
    hidden_dim=64,
    num_gnn_layers=3
)

# Criar parâmetros
using DarwinPBPK.ODEPBPKSolver
params = ODEPBPKSolver.PBPKParams(
    clearance_hepatic=10.0,
    clearance_renal=5.0
)

# Predizer
result = DynamicGNN.forward(model, 100.0, params)
concentrations = result["concentrations"]
```

### Exemplo 3: Validação Científica

```julia
using DarwinPBPK.Validation

# Dados preditos e observados
pred = [1.0, 2.0, 3.0, 4.0, 5.0]
obs = [1.1, 2.1, 2.9, 4.2, 4.8]

# Calcular métricas
fe = Validation.fold_error(pred, obs)
gmfe = Validation.geometric_mean_fold_error(pred, obs)
pct_2x = Validation.percent_within_fold(pred, obs, 2.0)

println("GMFE: $gmfe")
println("% within 2.0x: $pct_2x%")
```

---

## 🔬 Validação Científica

### Métricas Regulatórias

O sistema implementa métricas padrão da indústria farmacêutica:

- **Fold Error (FE)**: Erro relativo entre predito e observado
- **Geometric Mean Fold Error (GMFE)**: Média geométrica dos FEs
- **% within fold**: Porcentagem de predições dentro de um fold (1.25x, 1.5x, 2.0x)

### Critérios de Aceitação

- **GMFE < 2.0**: Aceitável para modelos PBPK
- **% within 2.0x > 50%**: Mínimo regulatório
- **% within 1.5x > 30%**: Desejável

---

## 📈 Performance

### Benchmarks

- **ODE Solver**: ~4.5 ms por simulação (4× mais rápido que Python)
- **Dynamic GNN**: ~0.08 ms para criação
- **Memory**: Redução de 50-70% vs Python

### Otimizações Implementadas

1. **Stack allocation** (SVector) - zero heap allocation
2. **SIMD vectorization** - automática via JIT
3. **Type stability** - zero runtime overhead
4. **Parallel dataset generation** - threads nativos

---

## 🎓 Referências

- `README.md` - Visão geral
- `EXECUTION_GUIDE.md` - Guia de execução
- `docs/migration/` - Análises detalhadas

---

**Última atualização:** 2025-11-18

