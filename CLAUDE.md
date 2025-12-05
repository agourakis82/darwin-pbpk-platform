# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Darwin PBPK Platform** is an AI-powered physiologically-based pharmacokinetic (PBPK) prediction platform using multi-modal molecular representations. The project is primarily implemented in Julia (migrated from Python) with supporting components in Rust and Python.

**Core Focus**: Q1 scientific rigor for computational drug discovery with SOTA deep learning for pharmacokinetic parameter prediction.

## Repository Structure

```
darwin-pbpk-platform/
├── julia-migration/           # PRIMARY: Julia PBPK implementation (v2.5.0)
│   ├── src/DarwinPBPK/       # Main module with PBPK components
│   │   ├── ode_solver.jl     # DifferentialEquations.jl ODE solver
│   │   ├── dynamic_gnn.jl    # Graph Neural Networks (GAT + TransformerConv)
│   │   ├── fractal_blood.jl  # CTRW dynamics, multi-phase blood modeling
│   │   ├── medlang/          # MedLang DSL integration
│   │   └── ml/               # ML components (evidential, bayesian_uq)
│   └── test/                 # Julia tests
├── Darwin-medlang/           # Submodule: MedLang compiler (Rust)
├── Darwin-demetrios/         # Submodule: Demetrios language compiler (Rust)
├── apps/api/                 # FastAPI REST endpoints
├── data/                     # Datasets and embeddings
├── models/                   # Trained model checkpoints
└── analysis/                 # Research and experimental code
```

## Build & Test Commands

### Julia (Primary)

```bash
cd julia-migration

# Activate and install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a specific test file
julia --project=. test/test_medlang.jl

# Start REPL with project
julia --project=.

# Run ODE solver
julia --project=. -e 'using DarwinPBPK; # your code'
```

### MedLang Compiler (Rust submodule)

```bash
cd Darwin-medlang/compiler

# Build
cargo build --release

# Run tests (103 tests)
cargo test

# Specific test suites
cargo test --test golden_tests
cargo test --test end_to_end

# Compile MedLang to Stan/Julia
./target/release/mlc compile model.medlang --backend julia

# Lint
cargo clippy
cargo fmt
```

### Demetrios Compiler (Rust submodule)

```bash
cd Darwin-demetrios/compiler

cargo build
cargo test
cargo clippy
```

### Python API (Legacy/Support)

```bash
# Run API server
python scripts/run_api.py --reload

# API docs at http://localhost:8000/api/v1/docs
```

## Architecture

### Julia Module Structure (DarwinPBPK)

The main Julia module (`julia-migration/src/DarwinPBPK.jl`) integrates:

1. **Patient Modeling**: `patient_profile.jl`, `compartment_models.jl`
2. **ODE Solving**: `ode_solver.jl` using DifferentialEquations.jl (50-500× faster than Python)
3. **ML/GNN**: `dynamic_gnn.jl` with GraphNeuralNetworks.jl, Flux.jl
4. **Fractal Dynamics**: `fractal_blood.jl` for CTRW and multi-phase modeling
5. **Bayesian UQ**: `ml/bayesian_uq.jl` with Turing.jl
6. **MedLang DSL**: `medlang/MedLang.jl` for domain-specific modeling
7. **Validation**: `validation.jl` with regulatory metrics (FE, GMFE, R²)

### Key Dependencies

**Julia**: DifferentialEquations.jl, Flux.jl, GraphNeuralNetworks.jl, Turing.jl, Unitful.jl, CUDA.jl
**Rust**: logos (lexer), nom (parser), clap (CLI), miette (diagnostics)

### MedLang Compilation Pipeline

```
.medlang → Lexer → Parser → TypeChecker → IR → CodeGen → .stan/.jl
```

The type system uses M·L·T dimensional analysis for compile-time unit verification.

## Development Guidelines

### Package Organization

When adding features, identify the correct location:
- PBPK core logic → `julia-migration/src/DarwinPBPK/`
- MedLang compiler → `Darwin-medlang/compiler/src/`
- API endpoints → `apps/api/routers/`

### Code Quality Requirements

1. **Tests are mandatory**: Create tests alongside new code
2. **Type safety**: Use Julia's type system and Unitful.jl for units
3. **Dimensional analysis**: Verify unit consistency (Clearance = L³/T, etc.)
4. **No mixing packages**: Import between modules, don't copy code

### Commit Message Format

```
feat(component): brief description
fix(component): brief description
```

Components: `julia`, `medlang`, `demetrios`, `api`, `docs`

## Important Files

- `julia-migration/Project.toml` - Julia dependencies
- `Darwin-medlang/docs/examples/one_comp_oral_pk.medlang` - Canonical MedLang example
- `Darwin-medlang/compiler/tests/golden_tests.rs` - Golden file tests (set UPDATE_GOLDEN=true to update)

## Submodules

This repo contains git submodules:
- `Darwin-medlang/` - MedLang DSL compiler
- `Darwin-demetrios/` - Demetrios language compiler

After cloning: `git submodule update --init --recursive`

## Performance Notes

- Julia ODE solver: ~0.04-0.36ms per simulation (vs ~18ms Python)
- Parallel dataset generation via Julia threads (no GIL)
- GPU acceleration available via CUDA.jl

---

## Semantic Web Layer (FAIR Data) - December 2025

### Overview

The platform implements a comprehensive **Semantic Web layer** for FAIR (Findable, Accessible, Interoperable, Reusable) data compliance, enabling linked data integration with biomedical ontologies.

**Location**: `julia-migration/src/DarwinPBPK/semantic/`

### Modules

| Module | Description |
|--------|-------------|
| `SemanticCore.jl` | Main module re-exporting all semantic functions |
| `contexts.jl` | JSON-LD 1.1 contexts with OBO Foundry prefixes |
| `qudt_units.jl` | QUDT/UO unit mappings (40+ PK units) |
| `jsonld_serializer.jl` | Entity serialization to JSON-LD |
| `provenance.jl` | PROV-O provenance annotations |
| `schema_org.jl` | Schema.org Drug, MedicalStudy, PropertyValue |
| `doid_ontology.jl` | Disease Ontology (DOID) full integration |

### Ontologies Integrated

**Fully Integrated (with parsers)**:
- **DOID** - Human Disease Ontology v2025-11-25 (27MB OWL, 21MB JSON)
- **QUDT** - Quantities, Units, Dimensions, Types
- **UO** - Units Ontology
- **PROV-O** - W3C Provenance Ontology
- **Schema.org** - Drug, MedicalStudy, MedicalCode types

**Prefix Support (JSON-LD contexts)**:
- ChEBI, DrOn, DINTO, DIDEO, OBI, BFO, PATO, IAO, STATO, RO, GO, UBERON, HP

### REST API Semantic Endpoints

```
GET  /api/semantic/contexts      # JSON-LD 1.1 contexts
GET  /api/semantic/ontologies    # List supported ontologies
GET  /api/semantic/units         # QUDT units with UO mappings
POST /api/semantic/drug          # Serialize drug to JSON-LD
POST /api/semantic/ddi           # Serialize DDI (DINTO/DIDEO)
POST /api/semantic/parameter     # Serialize PK parameter (QUDT)
POST /api/semantic/simulation    # Serialize simulation with PROV-O
GET  /api/semantic/turtle/{type}/{id}  # RDF Turtle format
GET  /api/semantic/schema.org/drug/{id}  # Schema.org Drug
```

### DOID Usage Example

```julia
using DarwinPBPK

# Load DOID (one-time, ~21MB JSON)
doid = load_doid("julia-migration/data/ontologies/doid.json")

# Search diseases
results = search_doid(doid, "diabetes")  # returns Vector{DOIDTerm}

# Get by ID
dm = get_disease_by_id(doid, "DOID:9352")  # diabetes mellitus
println(dm.name)        # "diabetes mellitus"
println(dm.definition)  # Full definition

# Cross-references
icd10 = get_disease_xrefs(dm, "ICD10CM")  # ["E08", "E09", "E10", ...]
mesh = get_disease_xrefs(dm, "MESH")      # ["D003920"]
omim = get_disease_xrefs(dm, "OMIM")

# Hierarchy navigation
hierarchy = get_disease_hierarchy(doid, "DOID:9352", depth=2)
# ancestors, descendants

# Drug class → diseases
diseases = get_diseases_for_drug_class(doid, "antidiabetic")
```

### JSON-LD Serialization Example

```julia
using DarwinPBPK

# Create semantic drug entity
drug = create_semantic_drug(
    "Metformin";
    chebi_id = "CHEBI:6801",
    drugbank_id = "DB00331",
    molecular_weight = 129.16,
    mechanism_of_action = "AMPK activation"
)

# Add provenance
drug = annotate_with_provenance(drug, :pbpk_model)

# Export
json_str = export_to_jsonld(drug)
turtle_str = export_to_turtle(drug)
```

### Ontology Download Script

```bash
cd julia-migration/data/ontologies
./download_ontologies.sh  # Downloads DOID, UO, OBI
```

### Files Reference

```
julia-migration/
├── src/DarwinPBPK/
│   ├── semantic/
│   │   ├── SemanticCore.jl        # Main module
│   │   ├── contexts.jl            # JSON-LD contexts
│   │   ├── qudt_units.jl          # QUDT unit mappings
│   │   ├── jsonld_serializer.jl   # Serialization
│   │   ├── provenance.jl          # PROV-O
│   │   ├── schema_org.jl          # Schema.org types
│   │   └── doid_ontology.jl       # DOID parser/index
│   └── api/
│       └── rest_api.jl            # Includes semantic endpoints
├── data/ontologies/
│   ├── doid.owl                   # 27MB - Full DOID OWL
│   ├── doid.json                  # 21MB - DOID JSON (for Julia)
│   ├── doid.obo                   # 6.7MB - DOID OBO format
│   └── download_ontologies.sh     # Download script
└── test/
    └── test_semantic.jl           # Semantic layer tests
```
