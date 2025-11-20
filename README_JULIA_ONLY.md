# Darwin PBPK Platform - Julia Only

**Status:** ✅ **100% Julia - 0% Python**

---

## 🎯 Repositório 100% Julia

Este repositório foi **completamente migrado para Julia**. Não há mais código Python.

---

## 🚀 Quick Start

### 1. Instalar Julia 1.9+

```bash
# Linux (via juliaup)
curl -fsSL https://install.julialang.org | sh

# Ou baixar de: https://julialang.org/downloads/
```

### 2. Setup do Projeto

```bash
cd julia-migration
julia
```

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### 3. Executar

```julia
using DarwinPBPK

# Exemplo: ODE Solver
using DarwinPBPK.ODEPBPKSolver
params = ODEPBPKSolver.PBPKParams()
result = ODEPBPKSolver.solve_ode_problem(params, 100.0, (0.0, 24.0))
```

---

## 📚 Documentação

- **Guia de Execução:** `julia-migration/EXECUTION_GUIDE.md`
- **Tutorial:** `julia-migration/docs/TUTORIAL.md`
- **Migração Completa:** `docs/COMPLETE_PYTHON_TO_JULIA_MIGRATION.md`

---

## ✅ Componentes Implementados

- ✅ ODE Solver (4× mais rápido que Python)
- ✅ Dataset Generation
- ✅ Dynamic GNN
- ✅ Training Pipeline
- ✅ Validation (GMFE 1.036, 100% within folds)
- ✅ REST API

---

## 📊 Performance

- **ODE Solver:** 4.5ms (4× vs Python)
- **Validação Científica:** GMFE 1.036
- **Testes:** 6/6 passando

---

**Última atualização:** 2025-11-18

