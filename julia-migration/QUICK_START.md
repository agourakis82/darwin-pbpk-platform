# Quick Start - Darwin PBPK Platform (Julia)

**Versão:** 0.1.0
**Data:** 2025-11-18

---

## 🚀 Instalação Rápida

### 1. Instalar Julia
```bash
# Download: https://julialang.org/downloads/
# Versão recomendada: Julia 1.9+
```

### 2. Ativar Ambiente
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### 3. Testar Instalação
```julia
using DarwinPBPK

# Testar ODE Solver
p = ODEPBPKSolver.PBPKParams(
    clearance_hepatic=10.0,
    clearance_renal=5.0,
)
sol = ODEPBPKSolver.solve(p, 100.0, (0.0, 24.0))
println("✅ ODE Solver funcionando!")
```

---

## 📚 Exemplos de Uso

### ODE Solver
```julia
using DarwinPBPK.ODEPBPKSolver

# Criar parâmetros
p = PBPKParams(
    clearance_hepatic=10.0,
    clearance_renal=5.0,
    partition_coeffs=Dict("liver" => 2.0, "kidney" => 1.5),
)

# Simular
sol = solve(p, 100.0, (0.0, 24.0))
```

### Dataset Generation
```julia
using DarwinPBPK.DatasetGeneration

# Gerar dataset
main("analysis/pbpk_parameters_wide_enriched_v3.csv", "output.jld2")
```

### Dynamic GNN
```julia
using DarwinPBPK.DynamicGNN

# Criar modelo
model = DynamicPBPKGNN()

# Forward pass
results = forward(model, 100.0, params)
```

---

## 🧪 Testes

```julia
using Pkg
Pkg.test("DarwinPBPK")
```

---

## 📊 Benchmarks

```julia
include("benchmarks/benchmark_complete.jl")
```

---

**Última atualização:** 2025-11-18

