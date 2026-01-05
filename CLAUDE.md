# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Darwin PBPK Platform** is an AI-powered physiologically-based pharmacokinetic (PBPK) prediction platform using multi-modal molecular representations. The primary implementation is in Julia (`julia-migration/`) with supporting components in Rust (MedLang/Sounio compilers) and Python (legacy API).

**Core Focus**: Q1 scientific rigor for computational drug discovery with SOTA deep learning for PK parameter prediction.

## Build & Test Commands

### Julia (Primary)

```bash
cd julia-migration

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific test file
julia --project=. test/test_medlang.jl

# Start REPL with project
julia --project=.

# Use the module
julia --project=. -e 'using DarwinPBPK; # your code'
```

### MedLang Compiler (Rust submodule)

```bash
cd Darwin-medlang/compiler

cargo build --release
cargo test
cargo test --test golden_tests       # Golden file tests
cargo test --test end_to_end         # E2E tests

# Update golden files
UPDATE_GOLDEN=true cargo test --test golden_tests

# Compile MedLang to backends
./target/release/mlc compile model.medlang --backend julia
./target/release/mlc compile model.medlang --backend sounio
./target/release/mlc compile model.medlang --backend stan

cargo clippy && cargo fmt
```

### Sounio Compiler (Rust submodule)

```bash
cd Darwin-sounio/compiler

cargo build
cargo test
cargo clippy
```

### Python API (Legacy)

```bash
python scripts/run_api.py --reload
# API docs: http://localhost:8000/api/v1/docs
```

## Architecture

### Julia Module Structure (`julia-migration/src/DarwinPBPK/`)

The main module (`DarwinPBPK.jl`) integrates ~40+ submodules:

**Core PBPK**:
- `ode_solver.jl` - DifferentialEquations.jl ODE solver (50-500x faster than Python)
- `compartment_models.jl` - Physiological compartment models
- `patient_profile.jl` - Patient demographics & scaling
- `fractal_blood.jl` - CTRW dynamics, multi-phase blood modeling
- `compartments/` - Specialized models (GI, coagulation, RBC dynamics, etc.)

**ML/GNN**:
- `dynamic_gnn.jl` - GAT + TransformerConv with GraphNeuralNetworks.jl
- `ml/bayesian_uq.jl` - Bayesian UQ with Turing.jl
- `ml/evidential.jl` - Evidential deep learning
- `ml/neural_ode.jl` - Neural ODE PBPK models

**DSL Integration**:
- `medlang/MedLang.jl` - MedLang DSL parser and compiler
- `sounio/SounioIntegration.jl` - Julia↔Sounio FFI bridge

**Semantic/FAIR**:
- `semantic/SemanticCore.jl` - JSON-LD, DOID, QUDT, PROV-O integration

### MedLang Compilation Pipeline

```
.medlang → Lexer → Parser → TypeChecker → IR → CodeGen → .stan/.jl/.sio
```

Supports three backends: Stan (Bayesian), Julia (native ODE), Sounio (epistemic computing)

### Key Dependencies

**Julia**: DifferentialEquations.jl, Flux.jl, GraphNeuralNetworks.jl, Turing.jl, Unitful.jl, CUDA.jl
**Rust**: logos (lexer), nom (parser), clap (CLI), miette (diagnostics)

## Development Guidelines

### Package Organization

- PBPK core logic → `julia-migration/src/DarwinPBPK/`
- MedLang compiler → `Darwin-medlang/compiler/src/`
- API endpoints → `apps/api/routers/`
- New compartment models → `julia-migration/src/DarwinPBPK/compartments/`
- ML components → `julia-migration/src/DarwinPBPK/ml/`

### Code Quality

1. **Type safety**: Use Julia's type system and Unitful.jl for dimensional analysis
2. **Dimensional analysis**: Verify unit consistency (Clearance = L³/T, Volume = L³, etc.)
3. **Tests alongside code**: All new modules need corresponding tests in `julia-migration/test/`

### Commit Message Format

```
feat(component): brief description
fix(component): brief description
```

Components: `julia`, `medlang`, `sounio`, `api`, `docs`

## Important Files

- `julia-migration/Project.toml` - Julia dependencies
- `julia-migration/src/DarwinPBPK.jl` - Main module with all exports
- `Darwin-medlang/compiler/tests/golden_tests.rs` - Golden file tests (UPDATE_GOLDEN=true to update)
- `julia-migration/test/test_medlang.jl` - MedLang integration tests

## Submodules

After cloning:
```bash
git submodule update --init --recursive
```

- `Darwin-medlang/` - MedLang DSL compiler
- `Darwin-sounio/` - Sounio language compiler

## Performance Notes

- Julia ODE solver: ~0.04-0.36ms per simulation (vs ~18ms Python)
- Parallel dataset generation via Julia threads (no GIL)
- GPU acceleration via CUDA.jl
- Stiff solvers (Rodas4/FBDF): 10-100x speedup for coagulation/fibrinolysis

---

## Sounio v0.97.0 Integration

**Location**: `julia-migration/src/DarwinPBPK/sounio/`

### MedLang → Sounio Pipeline

```julia
using DarwinPBPK

# Compile and run MedLang source via Sounio
result = compile_and_run_medlang("""
    model SimplePK
    compartment plasma: 5.0 L
    parameter ka: 1.2 1/h
    parameter ke: 0.3 1/h
    state A_gut: 100.0 mg
    state A_plasma: 0.0 mg
end
"""; dose_mg=100.0, duration_h=24.0)
```

### Epistemic UQ (Knowledge{T})

```julia
# Convert Bayesian posterior to Sounio Knowledge type
posterior = sample_posterior(model, data)
clearance_knowledge = bayesian_to_knowledge(posterior, :CL)

# Run simulation with confidence gating
result = run_sounio_pbpk_gated(model, request; min_confidence=0.80)
```

### Key Files

- `SounioIntegration.jl` - FFI bridge, v0.97.0 API
- `EpistemicTypes.jl` - Knowledge{T}, Provenance, ConfidenceLevel
- `UQBridge.jl` - bayesian_to_knowledge, mcdropout_to_knowledge

---

## Stiff ODE Solvers (Coagulation/Fibrinolysis)

**Location**: `julia-migration/src/DarwinPBPK/compartments/`

### Coagulation (25-state, stiffness ~1000:1)

```julia
using DarwinPBPK

system = create_coagulation_system(tissue_factor=0.005)
config = CoagulationSolverConfig(algorithm=:rodas4, use_sparse=true)
times, results = simulate_coagulation_stiff!(system, (0.0, 3600.0); config)
```

### Fibrinolysis (16-state, stiffness ~360,000:1)

```julia
system = create_fibrinolytic_system(fibrin_amount=1000.0)
config = FibrinolysisSolverConfig(algorithm=:fbdf, abstol=1e-9)
times, results = simulate_fibrinolysis_stiff!(system, (0.0, 3600.0); config)
```

### Performance vs Euler

| Model | Euler Steps (1000s) | Stiff Steps | Speedup |
|-------|---------------------|-------------|---------|
| Coagulation | ~1M | ~1-10K | 100x |
| Fibrinolysis | ~1M | ~100-1K | 1000x |

---

## Semantic Web Layer (FAIR Data)

**Location**: `julia-migration/src/DarwinPBPK/semantic/`

### Ontologies Integrated

- **DOID** - Human Disease Ontology (21MB JSON parser)
- **QUDT/UO** - Unit mappings (40+ PK units)
- **PROV-O** - W3C Provenance
- **Schema.org** - Drug, MedicalStudy types

### Usage

```julia
using DarwinPBPK

# Load DOID
doid = load_doid("julia-migration/data/ontologies/doid.json")
results = search_doid(doid, "diabetes")

# Create semantic drug
drug = create_semantic_drug("Metformin"; chebi_id="CHEBI:6801")
json_str = export_to_jsonld(drug)
```

### REST API Endpoints

```
GET  /api/semantic/contexts
GET  /api/semantic/ontologies
POST /api/semantic/drug
POST /api/semantic/simulation
```

Download ontologies: `cd julia-migration/data/ontologies && ./download_ontologies.sh`
