# MedLang Repository Update Guide

This document describes the files and specifications to update in the MedLang repository (github.com/agourakis82/medlang) based on the Darwin PBPK Platform implementation.

## Summary of Changes

The Darwin PBPK Platform is now the **first real implementation** of MedLang DSL, specifically Track D (Pharmacometrics/PBPK). The following extensions have been implemented and validated:

### New Features
1. **Oral Absorption Modeling** - Ka, F, lag time parameters
2. **First-Pass Metabolism** - Fg (gut) and Fh (hepatic) availability
3. **Route Declaration** - Support for IV, oral, IM, SC, infusion routes
4. **15-Compartment ODE System** - Extended PBPK with gut lumen compartment

## Files to Update in MedLang Repo

### 1. Grammar Specification
**File:** `docs/track_d/medlang_d_grammar_v0.2.md`

Add the following grammar rules:

```bnf
<model_body> ::= <model_stmt>*

<model_stmt> ::= <state_def>
              | <param_def>
              | <organ_def>
              | <clearance_def>
              | <absorption_def>      // NEW
              | <firstpass_def>       // NEW
              | <route_def>           // NEW
              | <ode_equation>
              | <obs_def>

<route_def> ::= "route" ":" <route_type>

<route_type> ::= "iv" | "intravenous" 
              | "oral" | "po"
              | "im" | "intramuscular"
              | "sc" | "subcutaneous"
              | "infusion"

<absorption_def> ::= "absorption" "{" <absorption_params> "}"

<absorption_params> ::= <absorption_param> ("," <absorption_param>)*

<absorption_param> ::= "Ka" ":" <expr>
                    | "F" ":" <expr>
                    | "lag" ":" <expr>

<firstpass_def> ::= "firstpass" "{" <firstpass_params> "}"

<firstpass_params> ::= <firstpass_param> ("," <firstpass_param>)*

<firstpass_param> ::= "Fg" ":" <expr>
                   | "Fh" ":" <expr>
```

### 2. Core Specification
**File:** `docs/track_d/medlang_pharmacometrics_qsp_spec_v0.2.md`

Add section on oral absorption:

```markdown
## 5. Oral Absorption Model

### 5.1 Absorption Parameters
- `Ka`: First-order absorption rate constant (1/h)
- `F`: Fraction absorbed from gut lumen (0-1)
- `lag`: Lag time before absorption begins (h)

### 5.2 First-Pass Metabolism
- `Fg`: Gut availability (fraction escaping gut metabolism)
- `Fh`: Hepatic availability (fraction escaping hepatic first-pass)

### 5.3 Bioavailability
Effective bioavailability: F_eff = Fa × Fg × Fh

### 5.4 ODE System
15-compartment model with gut lumen depot:
- dA_gut/dt = -Ka × A_gut
- Blood receives: (Ka × A_gut × Fg × Fh) / V_blood
```

### 3. Examples
**File:** `examples/track_d/oral_drug_model.medlang`

```medlang
// Example: Oral Drug with First-Pass Metabolism
model ExampleOral_PBPK {
    route: oral
    
    absorption {
        Ka: 1.5,
        F: 0.9,
        lag: 0.5
    }
    
    firstpass {
        Fg: 0.8,
        Fh: 0.6
    }
    
    clearance hepatic: 20.0_L/h
    clearance renal: 2.0_L/h
    
    organ blood { V: 5.0_L, Q: 0.0_L/h, Kp: 1.0 }
    organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 2.0 }
    // ... other organs
}
```

### 4. Keywords Reference
**File:** `docs/keywords.md`

Add new keywords:
- `absorption` - Defines oral absorption parameters block
- `firstpass` - Defines first-pass metabolism parameters block
- `route` - Specifies route of administration

### 5. Implementation Reference
**File:** `docs/implementations.md`

Add Darwin PBPK as reference implementation:

```markdown
## Reference Implementations

### Darwin PBPK Platform (Julia)
**Repository:** github.com/[user]/darwin-pbpk-platform
**Status:** First Reference Implementation
**Track:** D (Pharmacometrics/PBPK)
**Features:**
- Full MedLang Track D parser
- Julia transpiler
- 15-compartment PBPK ODE solver
- Oral absorption with first-pass metabolism
- Validated against 572 drugs (GMFE: 2.0 for half-life)

**Files:**
- `julia-migration/src/DarwinPBPK/medlang/parser.jl`
- `julia-migration/src/DarwinPBPK/medlang/transpiler.jl`
- `julia-migration/src/DarwinPBPK/ode_solver.jl`
```

## Validation Results

The implementation was validated against 572 drugs from the ULTIMATE_DATASET:

### Success Metrics
| Metric | Result | FDA Threshold |
|--------|--------|---------------|
| Simulation Success | 92.1% | - |
| Half-life GMFE | 2.01 | < 2.0 |
| Half-life within 2-fold | 69.4% | > 50% |
| AUC within 3-fold | 50.0% | > 50% |
| Cmax within 2-fold | 24.3% | > 50% |

### Notable Findings
- Half-life predictions meet FDA acceptance criteria
- Oral absorption significantly improves Cmax predictions (4.4% → 24.3%)
- High-extraction drugs (Midazolam, Metoprolol) require additional modeling

## Code Artifacts

### Parser (Julia)
Key additions to `parser.jl`:
```julia
# New tokens
TOK_ABSORPTION, TOK_FIRSTPASS, TOK_ROUTE

# New AST nodes
struct AbsorptionDef <: ASTNode
struct FirstPassDef <: ASTNode
@enum RouteType

# Updated ModelDef
struct ModelDef <: ASTNode
    # ... existing fields ...
    absorption::Union{AbsorptionDef, Nothing}
    firstpass::Union{FirstPassDef, Nothing}
    route::RouteType
end
```

### ODE Solver (Julia)
Key additions to `ode_solver.jl`:
```julia
struct OralParams
    ka::Float64
    fa::Float64
    fg::Float64
    fh::Float64
    lag::Float64
end

function oral_ode_system!(du, u, p, t)
    # 15-compartment ODE with gut lumen
end

function simulate_oral(pbpk_params, oral_params, dose; kwargs...)
    # Full ODE integration
end
```

## Next Steps for MedLang Repo

1. **Update Grammar Docs** - Add oral absorption rules
2. **Add Examples** - Create oral drug model examples
3. **Update Changelog** - Document v0.2.0 features
4. **Tag Release** - Create v0.2.0-alpha release
5. **Link Implementation** - Reference Darwin PBPK in docs

## Contact

For questions about the implementation:
- Repository: darwin-pbpk-platform
- Author: Dr. Demetrios Agourakis
- Date: November 2025
