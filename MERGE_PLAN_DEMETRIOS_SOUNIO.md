# Demetrios → Sounio Merge Plan

**Status**: PLANNING
**Date**: December 26, 2025
**Scope**: Massive architectural merge of two epistemic programming languages

---

## Executive Summary

This document outlines the strategy for merging **Demetrios** (v0.83.0) into **Sounio** (github.com/sounio-lang/sounio). Both are Rust-based epistemic programming languages created for scientific computing with uncertainty propagation.

### Merge Direction

```
Demetrios ──────────────────────────────────▶ Sounio
(PBPK-focused,    MERGE INTO    (Broader scientific,
 ~150KB stdlib)                  151k+ lines stdlib)
```

**Rationale**: Sounio has a larger, more comprehensive stdlib covering multiple domains (fMRI, causal inference, GPU, signal processing). Demetrios has a mature PBPK stdlib that should be integrated.

---

## Project Comparison

### Core Statistics

| Attribute | Demetrios | Sounio |
|-----------|-----------|--------|
| Version | v0.83.0 | - |
| Language | Rust | Rust 1.70+ |
| License | MIT/Apache 2.0 | MIT |
| Primary Domain | PBPK/Pharmacology | Multi-domain Scientific |
| Stdlib Size | ~150KB | 151,000+ lines |
| Repository | github.com/chiuratto-AI/demetrios | github.com/sounio-lang/sounio |

### Feature Comparison

| Feature | Demetrios | Sounio | Merge Strategy |
|---------|-----------|--------|----------------|
| Knowledge<T> type | ✅ | ✅ | Unify APIs |
| Unit type system | ✅ Phantom types | ✅ | Port Demetrios M·L·T |
| Algebraic effects | ✅ effect[IO,Mut,Alloc] | ? | Port to Sounio |
| Refinement types | ✅ SMT-verified | ? | Port to Sounio |
| Linear types | ✅ | ? | Port to Sounio |
| Provenance tracking | ✅ | ✅ | Unify APIs |
| GUM compliance | ✅ | ✅ | Already aligned |
| ISO 17025 | Partial | ✅ | Use Sounio's |
| PBPK stdlib | ✅ 150KB | ✅ 9,800 lines | Merge both |
| ODE solver | ✅ Tsit5 native | ? | Port Demetrios |
| fMRI processing | ❌ | ✅ 5,073 lines | Keep Sounio |
| Causal inference | ❌ | ✅ 3,773 lines | Keep Sounio |
| GPU acceleration | ? | ✅ 2,487 lines | Keep Sounio |

---

## Merge Work Packages

### Package 1: Repository Setup & Infrastructure

**Estimated Effort**: 2-3 days
**Dependencies**: None

#### Tasks:

1. **Fork Sounio repository**
   ```bash
   git clone https://github.com/sounio-lang/sounio
   cd sounio
   git checkout -b merge/demetrios-v0.83.0
   ```

2. **Add Demetrios as submodule for reference**
   ```bash
   git submodule add https://github.com/chiuratto-AI/demetrios demetrios-reference
   ```

3. **Set up CI/CD for merge branch**
   - Configure GitHub Actions
   - Add cross-project test suite
   - Set up benchmark comparisons

4. **Create feature flags for incremental merge**
   ```toml
   # Cargo.toml
   [features]
   demetrios-effects = []
   demetrios-refinement = []
   demetrios-pbpk = []
   ```

---

### Package 2: Compiler Core Alignment

**Estimated Effort**: 2-3 weeks
**Dependencies**: Package 1

#### 2.1 Lexer/Parser Comparison

Compare token sets and grammar:

| Component | Demetrios | Sounio | Action |
|-----------|-----------|--------|--------|
| Lexer lib | logos | ? | Align to common |
| Parser lib | nom | ? | Align to common |
| Diagnostics | miette | ? | Align to common |

#### 2.2 Type System Unification

**Critical**: Both have Knowledge<T> but potentially different implementations.

```rust
// Demetrios Knowledge type
struct EpistemicValue<T> {
    value: T,
    uncertainty: f64,
    confidence: f64,
    source: Source,
}

// Sounio Knowledge type (from docs)
struct Knowledge<T> {
    value: T,
    uncertainty: UncertaintyBounds,
    provenance: ProvenanceChain,
    confidence: ConfidenceLevel,
}
```

**Action**: Create unified `Knowledge<T>` that supports both APIs.

#### 2.3 Unit Type System

Port Demetrios' M·L·T dimensional analysis:

```d
// Demetrios phantom types
type Mass<U>
type Volume<U>
type Time<U>

// Derived
type Clearance = Volume<L> / Time<h>
type Concentration = Mass<mg> / Volume<L>
```

**Files to port**:
- `Darwin-demetrios/compiler/src/types/units.rs`
- `Darwin-demetrios/compiler/src/types/dimensional.rs`

#### 2.4 Algebraic Effects System

Port Demetrios' effect tracking:

```d
fn simulate() -> effect[IO, Mut, Alloc, GPU] Result
```

**Files to port**:
- `Darwin-demetrios/compiler/src/effects/mod.rs`
- `Darwin-demetrios/compiler/src/effects/io.rs`
- `Darwin-demetrios/compiler/src/effects/mut.rs`

#### 2.5 Refinement Types

Port SMT-verified constraints:

```d
type Fraction = Real where { 0.0 <= self <= 1.0 }
type Age = Real where { 0.0 <= self <= 120.0 }
```

**Integration**: May require Z3 or similar SMT solver binding.

---

### Package 3: Standard Library Migration

**Estimated Effort**: 3-4 weeks
**Dependencies**: Package 2 (type system)

#### 3.1 Sounio Stdlib Structure (Target)

```
sounio/stdlib/
├── epistemic/      (7,780 lines) - Keep as-is
├── medlang/        (9,800 lines) - Enhance with Demetrios
├── fmri/           (5,073 lines) - Keep as-is
├── causal/         (3,773 lines) - Keep as-is
├── connectivity/   (3,792 lines) - Keep as-is
├── signal/         (3,068 lines) - Keep as-is
├── gpu/            (2,487 lines) - Keep as-is
└── [NEW] darwin_pbpk/  - Port from Demetrios
```

#### 3.2 Demetrios PBPK Stdlib to Port

```
Darwin-demetrios/stdlib/darwin_pbpk/
├── simulation.d        (23KB, 680 lines)  → stdlib/darwin_pbpk/simulation.sou
├── tsit5_pbpk14.d      (23.9KB)           → stdlib/darwin_pbpk/solvers/tsit5.sou
├── compartments/
│   ├── brain.d         (25.1KB)           → stdlib/darwin_pbpk/organs/brain.sou
│   ├── liver.d         (16.7KB)           → stdlib/darwin_pbpk/organs/liver.sou
│   └── kidney.d        (23.7KB)           → stdlib/darwin_pbpk/organs/kidney.sou
├── ddi/
│   └── mechanistic_ddi.d (17KB)           → stdlib/darwin_pbpk/ddi/mechanistic.sou
└── pbpk/
    ├── types.d         (7.8KB)            → stdlib/darwin_pbpk/types.sou
    ├── covariate.d     (13.2KB)           → stdlib/darwin_pbpk/patient/covariate.sou
    ├── population.d    (15.6KB)           → stdlib/darwin_pbpk/population/mod.sou
    ├── error_models.d  (15.6KB)           → stdlib/darwin_pbpk/statistics/error_models.sou
    └── regulatory.d    (35.4KB)           → stdlib/darwin_pbpk/validation/regulatory.sou
```

#### 3.3 Stdlib Reconciliation Strategy

For `medlang/` (both have implementations):

| Sounio (9,800 lines) | Demetrios (~150KB) | Resolution |
|---------------------|-------------------|------------|
| pk_models.sou | simulation.d | Merge: Demetrios has Tsit5 |
| pd_models.sou | - | Keep Sounio |
| pbpk.sou | tsit5_pbpk14.d | Replace with Demetrios |
| - | compartments/*.d | Add from Demetrios |
| - | ddi/mechanistic_ddi.d | Add from Demetrios |
| - | regulatory.d | Add from Demetrios |

---

### Package 4: ODE Solver Migration

**Estimated Effort**: 1-2 weeks
**Dependencies**: Package 3 (types)

#### 4.1 Demetrios Tsit5 Implementation

Native adaptive ODE solver (0.02-0.20ms per step):

```d
// tsit5_pbpk14.d structure
struct Tsit5Solver {
    butcher_tableau: ButcherCoeffs,
    error_control: AdaptiveStepControl,
    state: SolverState,
}

fn tsit5_step(
    f: fn(&State, f64) -> StateDerivatives,
    state: &State,
    t: f64,
    dt: f64
) -> (State, f64)  // (new_state, error_estimate)
```

#### 4.2 Port Strategy

1. Extract Tsit5 as standalone module
2. Adapt to Sounio's type conventions
3. Add GPU acceleration hooks (Sounio has GPU stdlib)
4. Benchmark against existing Sounio ODE solvers

---

### Package 5: MedLang Integration

**Estimated Effort**: 1 week
**Dependencies**: Packages 2, 3

#### 5.1 Current State

Darwin-medlang can compile to:
- Stan (.stan)
- Julia (.jl)
- Julia PINN (.jl)
- **Demetrios (.d)** ← Needs updating for Sounio

#### 5.2 Migration Tasks

1. **Create Sounio backend** (`codegen/sounio.rs`)
   - Fork from `codegen/demetrios.rs`
   - Update syntax for Sounio conventions
   - Update stdlib imports

2. **Update CLI**
   ```bash
   mlc compile model.medlang --backend sounio -o model.sou
   ```

3. **Golden tests**
   - Port Demetrios golden tests
   - Add Sounio-specific test cases

---

### Package 6: Julia FFI Bridge

**Estimated Effort**: 1-2 weeks
**Dependencies**: Package 4

#### 6.1 Current Julia Integration

```julia
# DemetriosIntegration.jl (679 lines)
struct DemetriosCompiler ... end
struct DemetriosModel ... end
compile_demetrios(compiler, source_file; target)
run_demetrios_pbpk(model, request)
```

#### 6.2 Migration Tasks

1. **Create SounioIntegration.jl**
   - Port from DemetriosIntegration.jl
   - Update binary names (`dc` → `souc`)
   - Update output formats

2. **Update medlang_demetrios_compiler.jl**
   - Rename to `medlang_sounio_compiler.jl`
   - Update UNIT_MAPPING for Sounio syntax
   - Update code generation templates

---

### Package 7: Testing & Validation

**Estimated Effort**: 2-3 weeks
**Dependencies**: All previous packages

#### 7.1 Test Suite Migration

| Test Type | Demetrios Location | Sounio Location | Status |
|-----------|-------------------|-----------------|--------|
| Unit tests | compiler/tests/ | compiler/tests/ | Merge |
| Golden tests | compiler/tests/golden/ | compiler/tests/golden/ | Port |
| End-to-end | test/*.d | tests/*.sou | Create |
| Fuzzing | - | compiler/fuzz/ | Extend |
| Benchmarks | - | compiler/benches/ | Add |

#### 7.2 Validation Criteria

1. **Compiler correctness**
   - All Demetrios programs compile with Sounio
   - Equivalent semantics for shared features

2. **Performance**
   - ODE solver: ≤0.25ms/step (match Demetrios)
   - Compilation: <5 seconds for PBPK models
   - Memory: No regression from Demetrios

3. **Regulatory compliance**
   - GMFE <2.0 for PBPK predictions
   - 90% within 2-fold of observed
   - Maintain ISO 17025 compliance

---

## Timeline Overview

```
Week 1-2:   Package 1 - Repository Setup
Week 3-5:   Package 2 - Compiler Core Alignment
Week 6-9:   Package 3 - Standard Library Migration
Week 10-11: Package 4 - ODE Solver Migration
Week 12:    Package 5 - MedLang Integration
Week 13-14: Package 6 - Julia FFI Bridge
Week 15-17: Package 7 - Testing & Validation
Week 18:    Final integration and release

Total: ~4.5 months
```

---

## Risk Analysis

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Type system incompatibility | Blocks all | Start with type unification |
| Effect system conflicts | Major refactoring | Feature-flag incremental merge |
| Performance regression | User adoption | Continuous benchmarking |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Stdlib API breaks | User migration | Provide compatibility layer |
| Build system changes | CI/CD issues | Parallel build pipelines |
| Documentation gaps | User confusion | Document as we merge |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| License compatibility | Legal | Both MIT-compatible |
| Test coverage gaps | Quality | Comprehensive test plan |

---

## Open Questions for Clarification

1. **Type System**
   - Does Sounio have phantom types for units?
   - How does Sounio's Knowledge<T> differ from Demetrios' EpistemicValue<T>?

2. **Effects**
   - Does Sounio have an algebraic effect system?
   - If not, is there appetite to add one?

3. **Refinement Types**
   - Does Sounio support SMT-verified constraints?
   - What solver would be used?

4. **ODE Solvers**
   - What ODE solvers does Sounio currently have?
   - Is there existing GPU-accelerated ODE support?

5. **Governance**
   - Who maintains Sounio?
   - What's the PR/review process?
   - Release cadence?

---

## Immediate Next Steps

1. **Clone Sounio repository** and perform deep analysis
2. **Create feature comparison matrix** at compiler level
3. **Identify breaking changes** required
4. **Draft API compatibility layer** for gradual migration
5. **Set up development environment** with both codebases
6. **Begin Package 1** (Repository Setup)

---

## Files in darwin-pbpk-platform to Update

After merge, these files need updating:

| File | Changes Required |
|------|-----------------|
| `CLAUDE.md` | Update Demetrios → Sounio references |
| `DEMETRIOS_INTEGRATION.md` | Deprecate, redirect to Sounio |
| `DEMETRIOS_STDLIB_DISCOVERED.md` | Merge into Sounio docs |
| `julia-migration/src/DarwinPBPK/demetrios/` | Rename to `sounio/` |
| `julia-migration/src/DarwinPBPK/medlang/medlang_demetrios_compiler.jl` | Port to Sounio |
| `Darwin-demetrios/` | Remove submodule after merge |
| `Darwin-medlang/compiler/src/codegen/demetrios.rs` | Rename to `sounio.rs` |
| `*.d` example files | Convert to `.sou` |

---

## Appendix A: Sounio Stdlib Domains

| Domain | Lines | Description |
|--------|-------|-------------|
| epistemic/ | 7,780 | Core types & provenance |
| medlang/ | 9,800 | PK/PD models + PBPK |
| fmri/ | 5,073 | Neuroimaging pipelines |
| causal/ | 3,773 | Causal discovery |
| connectivity/ | 3,792 | Graph metrics |
| signal/ | 3,068 | DSP & spectral analysis |
| gpu/ | 2,487 | CUDA kernels |

## Appendix B: Demetrios PBPK Stdlib Modules

| Module | Size | Key Functions |
|--------|------|---------------|
| simulation.d | 23KB | run_pbpk_simulation() |
| tsit5_pbpk14.d | 23.9KB | Adaptive ODE solver |
| compartments/brain.d | 25.1KB | BBB transport |
| compartments/liver.d | 16.7KB | CYP metabolism |
| compartments/kidney.d | 23.7KB | Renal elimination |
| ddi/mechanistic_ddi.d | 17KB | DDI prediction |
| pbpk/regulatory.d | 35.4KB | FDA/EMA metrics |

---

**Document Version**: 1.0
**Authors**: Claude Code
**Repository**: darwin-pbpk-platform
**Branch**: claude/plan-demetrios-sounio-merge-UTnPQ
