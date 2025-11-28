# MedLang DSL Integration for Darwin PBPK

**First Real Implementation of MedLang DSL**

This module provides the first real-world implementation of the [MedLang DSL](https://github.com/agourakis82/medlang) for PBPK (Physiologically-Based Pharmacokinetic) modeling.

## Overview

MedLang is a medical-native, GPU/HPC-accelerated programming language designed to unify:
- Quantum pharmacology (HF/DFT/QM/MM)
- Clinical reasoning and protocols
- AI models (MLP, GNN, PINN)
- Probabilistic measures
- Fractal signal analysis

This implementation focuses on **Track D (Pharmacometrics & QSP)**, specifically targeting PBPK model definitions with unit-safe typing.

## Features

- **MedLang Track D Parser**: Full grammar support for PBPK model definitions
- **Unit-Safe Type System**: Compile-time dimensional analysis (mg, L, L/h, etc.)
- **Julia Transpiler**: Generate optimized Julia code from MedLang source
- **PBPKParams Integration**: Direct compilation to Darwin PBPK structs
- **Population Modeling**: NLME random effects support
- **Timeline/Dosing**: Declarative dosing schedules

## Quick Start

```julia
using DarwinPBPK
using DarwinPBPK.MedLang

# Define a PBPK model using MedLang syntax
source = """
model MyDrug {
    // Organ definitions
    organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 2.5 }
    organ kidney { V: 0.31_L, Q: 60.0_L/h, Kp: 1.8 }
    organ brain { V: 1.4_L, Q: 50.0_L/h, Kp: 0.3 }

    // Clearance mechanisms
    clearance hepatic: 15.0_L/h
    clearance renal: 3.0_L/h
}
"""

# Compile to PBPKParams
params = compile_model(source)

# Simulate directly from MedLang
results = simulate_medlang(source, 100.0; t_max=24.0)

# Generate Julia code
julia_code = generate_julia_module(source)
```

## MedLang Syntax Reference

### Model Definition

```medlang
model ModelName {
    // States (drug mass in compartments)
    state A_central : DoseMass = 0_mg
    state A_peripheral : DoseMass = 0_mg

    // Parameters
    param CL : Clearance = 10.0_L/h
    param V : Volume = 50.0_L
    param Ka : RateConst = 1.0_1/h

    // Organ definitions
    organ liver {
        V: 1.8_L,          // Volume
        Q: 90.0_L/h,       // Blood flow
        Kp: 2.5            // Partition coefficient
    }

    // Clearance mechanisms
    clearance hepatic: 15.0_L/h
    clearance renal: 3.0_L/h

    // Differential equations
    d/dt A_central = Ka * A_gut - (CL / V) * A_central

    // Observables
    obs C_plasma : ConcMass = A_central / V
}
```

### Population Model (NLME)

```medlang
population DrugPop {
    model DrugModel

    // Fixed effects
    param theta_CL : Clearance = 10.0_L/h
    param theta_V : Volume = 50.0_L

    // Random effects
    rand eta_CL : f64 ~ Normal(0, 0.35)
    rand eta_V : f64 ~ Normal(0, 0.30)

    // Covariates
    input WT : Mass

    // Parameter binding with allometric scaling
    bind_params(individual) {
        let CL = theta_CL * (WT / 70_kg)^0.75 * exp(eta_CL)
        let V = theta_V * (WT / 70_kg) * exp(eta_V)
    }
}
```

### Dosing Timeline

```medlang
timeline MultiDose_QD {
    at 0_h: dose { amount = 100_mg, to = A_blood }
    at 24_h: dose { amount = 100_mg, to = A_blood }
    at 48_h: dose { amount = 100_mg, to = A_blood }

    // Observation times
    at 0.5_h: observe C_plasma
    at 1_h: observe C_plasma
    at 4_h: observe C_plasma
    at 24_h: observe C_plasma
}
```

## Supported Units

### Mass
- `mg`, `g`, `kg`, `ug`, `ng`, `pg`

### Volume
- `L`, `mL`, `uL`, `dL`

### Time
- `h`, `min`, `s`, `d`

### Derived
- `L/h`, `mL/min` (Clearance)
- `mg/L`, `ug/L` (Concentration)
- `1/h`, `1/min` (Rate constant)

## API Reference

### Core Functions

```julia
# Parse MedLang source to AST
ast = parse_medlang(source::String)

# Compile to PBPKParams
params = compile_model(source::String; model_name=nothing)

# Compile from file
params = compile_file("model.medlang"; model_name=nothing)

# Load and parse file
ast = load_medlang("model.medlang")

# Generate Julia module
julia_code = generate_julia_module(source::String)

# Simulate directly
results = simulate_medlang(source, dose; t_max=24.0, num_points=100)

# Validate model
issues = validate_model(source::String)

# Get model description
description = describe_model(source::String)
```

### AST Types

- `MedLangAST`: Root AST containing models, populations, timelines
- `ModelDef`: Model definition with states, params, organs, clearances, ODEs
- `PopulationDef`: Population model with random effects
- `TimelineDef`: Dosing timeline with events
- `StateDef`, `ParamDef`, `ObsDef`: Declaration types
- `OrganDef`, `ClearanceDef`: PBPK-specific types
- `DoseEvent`, `ObserveEvent`: Timeline events

## File Structure

```
medlang/
├── MedLang.jl           # Main module (public API)
├── parser.jl            # Lexer and parser
├── transpiler.jl        # Julia code generator
├── README.md            # This documentation
└── examples/
    ├── standard_pbpk.medlang     # 14-compartment PBPK
    └── drug_specific.medlang     # Drug-specific models
```

## Examples

See `examples/` directory for complete model definitions:

- **standard_pbpk.medlang**: Standard 14-compartment PBPK model
- **drug_specific.medlang**: Drug-specific models (Metformin, Midazolam, Warfarin, etc.)

## References

- [MedLang Repository](https://github.com/agourakis82/medlang)
- [MedLang Core Spec v0.1](https://github.com/agourakis82/medlang/blob/main/docs/medlang_core_spec_v0.1.md)
- [MedLang-D Grammar](https://github.com/agourakis82/medlang/blob/main/docs/medlang_d_minimal_grammar_v0.md)
- [Track D Pharmacometrics Spec](https://github.com/agourakis82/medlang/blob/main/docs/medlang_pharmacometrics_qsp_spec_v0.1.md)

## Author

Dr. Demetrios Agourakis

## Version

- Implementation: v0.1.0
- MedLang Spec: Track D v0.1

## License

Same as Darwin PBPK Platform
