# MedLang Specification v1.0

## A Domain-Specific Language for Physiologically-Based Pharmacokinetic Modeling

**Authors:** Dr. Sounio Agourakis  
**Date:** November 2025  
**Status:** Draft Specification

---

## 1. Introduction

MedLang is a domain-specific language (DSL) designed for expressing pharmacokinetic and pharmacodynamic (PK/PD) models with:

1. **Domain expressiveness** - Syntax mirrors how pharmacologists think
2. **Type safety** - Compile-time unit checking via Sounio backend
3. **Multi-target compilation** - Julia (dev), Sounio (production), PharmML (regulatory)
4. **Formal semantics** - Unambiguous interpretation for regulatory submission

### 1.1 Design Philosophy

```
"Write models like you write protocols - 
 the compiler handles the engineering"
```

MedLang separates **what** (the model) from **how** (the implementation):

| Concern | MedLang | Backends |
|---------|---------|----------|
| Model structure | `compartment Liver { ... }` | ODE system generation |
| Units | `CL: 25 L/h` | Type checking, conversion |
| DDI | `inhibits CYP3A4 with Ki = 0.015 µM` | R-model equations |
| Population | `varies with CV = 30%` | Monte Carlo sampling |
| Simulation | `simulate 0..24 h` | ODE solver selection |

---

## 2. Lexical Structure

### 2.1 Keywords

```
// Model structure
model, drug, patient, population
compartment, flow, volume, concentration

// Pharmacokinetics
absorption, distribution, metabolism, elimination
clearance, bioavailability, half_life

// Enzymes and transporters
enzyme, transporter, substrate, inhibitor, inducer
CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2, CYP2C8
OATP1B1, OATP1B3, BCRP, PGP, MRP2

// DDI mechanisms
inhibits, induces, competitive, noncompetitive, uncompetitive
mechanism_based, time_dependent
Ki, kinact, KI, Emax, EC50, IC50

// Parameters
param, state, observable, derived
varies, fixed, estimated

// Simulation
simulate, steady_state, multiple_dose
dose, infusion, oral, iv, sc, im

// Control flow
if, then, else, when, during
for, in, while

// Types
Real, Fraction, Concentration, Amount, Volume, Time, Clearance
```

### 2.2 Units (First-Class Citizens)

```
// Mass
mg, g, kg, µg, ng, pg
mol, mmol, µmol, nmol, pmol

// Volume
L, mL, µL, dL

// Time
h, min, s, day, week

// Concentration
M, mM, µM, nM, pM          // molar
mg/L, µg/mL, ng/mL, pg/mL  // mass/volume

// Clearance
L/h, mL/min, mL/min/kg, µL/min/mg

// Rate constants
1/h, 1/min, h⁻¹, min⁻¹

// Compound units (automatically derived)
mg/L/h, µM·h, L/h/kg
```

### 2.3 Literals

```
// Numbers with units
dose: 100 mg
CL: 25.5 L/h
Ki: 0.015 µM
fm: 0.94          // dimensionless fraction

// Ranges
time: 0..24 h
dose_range: 10..100 mg

// Distributions (for population models)
CL ~ LogNormal(mean: 25 L/h, CV: 30%)
Vd ~ Normal(mean: 100 L, SD: 20 L)

// Enumerations
route: oral | iv | sc | im
sex: male | female
phenotype: PM | IM | NM | UM
```

### 2.4 Comments

```
// Single line comment

/* Multi-line
   comment */

/// Documentation comment (exported to reports)
/// @param dose The administered dose
/// @returns Plasma concentration at time t

//! Warning comment (generates compiler warning)
//! TODO: Validate against clinical data
```

---

## 3. Grammar (EBNF)

```ebnf
(* Top-level *)
program         = { model_def | drug_def | patient_def | import_stmt } ;

(* Model definition *)
model_def       = "model" IDENTIFIER "{" model_body "}" ;
model_body      = { drug_block | patient_block | compartment_block 
                  | parameter_block | equation_block | ddi_block
                  | simulation_block } ;

(* Drug definition *)
drug_def        = "drug" IDENTIFIER "{" drug_body "}" ;
drug_body       = { property_stmt | metabolism_block | transport_block } ;

property_stmt   = IDENTIFIER ":" value_expr unit? ;
value_expr      = NUMBER | IDENTIFIER | expr ;

(* Metabolism block *)
metabolism_block = "metabolism" "{" { enzyme_stmt } "}" ;
enzyme_stmt      = enzyme_name ":" fraction ["," enzyme_params] ;
enzyme_name      = "CYP3A4" | "CYP2D6" | "CYP2C9" | "CYP2C19" 
                 | "CYP1A2" | "CYP2C8" | "CYP2E1" | "UGT" | "other" ;
enzyme_params    = "Km" "=" value_expr unit "," "Vmax" "=" value_expr unit ;

(* Transport block *)
transport_block  = "transport" "{" { transporter_stmt } "}" ;
transporter_stmt = transporter_name ":" role ["," transport_params] ;
transporter_name = "OATP1B1" | "OATP1B3" | "PGP" | "BCRP" | "MRP2" | "OCT1" ;
role             = "substrate" | "inhibitor" | "inducer" ;

(* Compartment definition *)
compartment_block = "compartment" IDENTIFIER "{" compartment_body "}" ;
compartment_body  = { volume_stmt | flow_stmt | state_stmt | equation_stmt } ;

volume_stmt      = "volume" ":" value_expr unit ;
flow_stmt        = "flow" ["from" IDENTIFIER] ["to" IDENTIFIER] ":" value_expr unit ;
state_stmt       = "state" IDENTIFIER ":" type_expr ["=" value_expr unit] ;

(* Parameters *)
parameter_block  = "parameters" "{" { param_stmt } "}" ;
param_stmt       = "param" IDENTIFIER ":" type_expr "=" value_expr unit
                   [variability_clause] [bounds_clause] ;
variability_clause = "varies" distribution ;
bounds_clause    = "bounds" "(" value_expr ".." value_expr ")" ;
distribution     = "Normal" "(" dist_params ")"
                 | "LogNormal" "(" dist_params ")"
                 | "Uniform" "(" dist_params ")" ;

(* Equations *)
equation_block   = "equations" "{" { equation_stmt } "}" ;
equation_stmt    = ode_equation | algebraic_equation ;
ode_equation     = "d" IDENTIFIER "/dt" "=" expr ;
algebraic_equation = IDENTIFIER "=" expr ;

(* DDI Block *)
ddi_block        = "ddi" "{" { ddi_stmt } "}" ;
ddi_stmt         = "with" IDENTIFIER "{" ddi_body "}" ;
ddi_body         = mechanism_stmt { "," ddi_param_stmt } ;
mechanism_stmt   = "mechanism" ":" ddi_mechanism ;
ddi_mechanism    = "competitive" | "noncompetitive" | "uncompetitive"
                 | "mechanism_based" | "induction" | "mixed" ;
ddi_param_stmt   = ("Ki" | "kinact" | "KI" | "Emax" | "EC50") ":" value_expr unit ;

(* Simulation *)
simulation_block = "simulate" "{" sim_body "}" ;
sim_body         = { dosing_stmt | time_stmt | output_stmt | setting_stmt } ;
dosing_stmt      = "dose" value_expr unit route ["at" time_expr]
                   ["every" value_expr unit "for" INTEGER "doses"] ;
time_stmt        = "time" ":" value_expr ".." value_expr unit ;
output_stmt      = "observe" IDENTIFIER ["as" STRING] ;
setting_stmt     = "setting" IDENTIFIER ":" value_expr ;

(* Expressions *)
expr             = term { ("+" | "-") term } ;
term             = factor { ("*" | "/") factor } ;
factor           = primary { "^" primary } ;
primary          = NUMBER [unit]
                 | IDENTIFIER ["(" [expr {"," expr}] ")"]
                 | "(" expr ")"
                 | unary_op primary ;
unary_op         = "-" | "+" | "log" | "exp" | "sqrt" ;

(* Types *)
type_expr        = base_type [unit] | compound_type ;
base_type        = "Real" | "Fraction" | "Concentration" | "Amount" 
                 | "Volume" | "Time" | "Clearance" | "Rate" ;
compound_type    = type_expr "/" type_expr 
                 | type_expr "*" type_expr ;

(* Units *)
unit             = simple_unit | compound_unit ;
simple_unit      = UNIT_SYMBOL ;
compound_unit    = unit "/" unit | unit "*" unit | unit "^" INTEGER ;
```

---

## 4. Type System

### 4.1 Dimensional Analysis

MedLang enforces dimensional correctness at compile time:

```medlang
// VALID: Units match
CL: Clearance = 25 L/h
Vd: Volume = 100 L
ke: Rate = CL / Vd  // L/h / L = 1/h ✓

// INVALID: Unit mismatch (compile error!)
ke: Rate = CL + Vd  // Error: Cannot add L/h and L
```

### 4.2 Type Hierarchy

```
Any
├── Numeric
│   ├── Real                    // Dimensionless
│   ├── Fraction               // 0 ≤ x ≤ 1 (refinement type)
│   └── Dimensional<D>         // With dimension D
│       ├── Mass<U>            // mg, g, kg, ...
│       ├── Volume<U>          // L, mL, ...
│       ├── Time<U>            // h, min, s, ...
│       ├── Amount<U>          // mol, mmol, ...
│       ├── Concentration<U>   // M, mg/L, ...
│       ├── Clearance<U>       // L/h, mL/min, ...
│       └── Rate<U>            // 1/h, 1/min, ...
├── Categorical
│   ├── Route                  // oral, iv, sc, im
│   ├── Sex                    // male, female
│   └── Phenotype              // PM, IM, NM, UM
└── Distribution
    ├── Normal<T>
    ├── LogNormal<T>
    └── Uniform<T>
```

### 4.3 Refinement Types

```medlang
// Fraction is a refinement of Real
type Fraction = Real where { 0.0 <= self <= 1.0 }

// Positive real
type Positive = Real where { self > 0.0 }

// Bounded parameter
type ValidKi = Concentration where { 0.001 µM <= self <= 1000 µM }
```

### 4.4 Unit Inference

```medlang
// Explicit units
CL: 25 L/h

// Inferred from expression
ke = CL / Vd  // Compiler infers: 1/h

// Inferred from context
param ka: Rate = 1.5  // Context: Rate → infers 1/h
```

---

## 5. Semantic Model

### 5.1 Compartmental Structure

```medlang
model TwoCompartment {
    compartment Central {
        volume: Vc
        state C_central: Concentration = 0 µM
    }
    
    compartment Peripheral {
        volume: Vp
        state C_periph: Concentration = 0 µM
    }
    
    flow Central <-> Peripheral: Q  // Bidirectional
}
```

Compiles to ODE system:
```
dA_central/dt = -CL*C_central - Q*(C_central - C_periph) + input(t)
dA_periph/dt  = Q*(C_central - C_periph)
```

### 5.2 DDI Semantics

```medlang
drug Midazolam {
    metabolism {
        CYP3A4: 94%, Km = 4 µM, Vmax = 500 pmol/min/mg
    }
    
    ddi {
        with Ketoconazole {
            mechanism: competitive
            Ki: 0.015 µM
            affects: [gut, liver]  // Both Fg and Fh
        }
    }
}
```

Compiles to:
```
// Gut-wall DDI
Ig = Dose_perpetrator / 250 mL  // Gut lumen concentration
Fg_ddi = 1 - (1 - Fg_baseline) / (1 + Ig/Ki)

// Hepatic DDI  
Ih = Cu_plasma + portal_contribution
CLint_ddi = CLint_baseline / (1 + Ih/Ki)
```

### 5.3 Population Semantics

```medlang
population Healthy_Adults {
    n: 1000
    
    // Fixed effects
    CL_pop: 25 L/h
    Vd_pop: 100 L
    
    // Random effects
    CL ~ LogNormal(mean: CL_pop, CV: 30%)
    Vd ~ LogNormal(mean: Vd_pop, CV: 25%)
    
    // Covariates
    CL = CL_pop * (weight/70)^0.75 * (age > 65 ? 0.8 : 1.0)
    
    // Correlations
    correlation(CL, Vd) = 0.3
}
```

---

## 6. Compilation Targets

### 6.1 Sounio Backend (Primary)

MedLang → Sounio provides:
- **Compile-time unit checking**
- **GPU acceleration** for population simulations
- **Algebraic effects** for controlled side effects
- **Linear types** for memory safety

```medlang
// MedLang source
param CL: Clearance = 25 L/h varies LogNormal(CV: 30%)
```

```sounio
// Generated Sounio
let CL: L_per_h = sample(LogNormal { 
    mean: 25.0, 
    cv: 0.30 
}) -> effect[Prob, GPU]
```

### 6.2 Julia Backend (Development)

For rapid prototyping and access to Julia ecosystem:

```julia
# Generated Julia
CL = 25.0  # L/h - NOTE: No unit checking!
CL_dist = LogNormal(log(25.0), 0.30)
```

### 6.3 PharmML Backend (Regulatory)

For FDA/EMA submission compatibility:

```xml
<!-- Generated PharmML -->
<Parameter symbId="CL">
    <Distribution>
        <LogNormal>
            <Mean><Scalar>25</Scalar></Mean>
            <CV><Scalar>0.30</Scalar></CV>
        </LogNormal>
    </Distribution>
</Parameter>
```

---

## 7. Standard Library

### 7.1 Built-in Functions

```medlang
// Mathematical
log(x), log10(x), exp(x), sqrt(x), pow(x, n)
abs(x), min(x, y), max(x, y), clamp(x, lo, hi)

// PK-specific
auc(C, t)                    // Trapezoidal AUC
auc_inf(C, t, ke)           // AUC extrapolated to infinity
cmax(C, t)                   // Maximum concentration
tmax(C, t)                   // Time of Cmax
half_life(ke)                // 0.693 / ke
mrt(auc, aumc)               // Mean residence time

// DDI
r_model(I, Ki, fm)           // Basic R-model: 1/(fm/(1+I/Ki) + (1-fm))
mbi_ratio(I, kinact, KI, kdeg, fm)  // MBI AUC ratio
induction_ratio(I, Emax, EC50, fm)  // Induction AUC ratio

// Clearance scaling
allometric(CL_ref, weight, weight_ref, exp)  // Allometric scaling
maturation(age, TM50, Hill)  // Pediatric maturation

// Unit conversion
to_L_h(cl: mL/min) -> L/h    // mL/min to L/h
to_uM(c: ng/mL, MW) -> µM    // Mass conc to molar
```

### 7.2 Built-in Models

```medlang
import std.pk.one_compartment
import std.pk.two_compartment
import std.pk.three_compartment

import std.absorption.first_order
import std.absorption.zero_order
import std.absorption.transit_compartment
import std.absorption.weibull

import std.elimination.linear
import std.elimination.michaelis_menten
import std.elimination.parallel

import std.ddi.competitive_inhibition
import std.ddi.mechanism_based_inhibition
import std.ddi.induction
```

---

## 8. Example: Complete DDI Model

```medlang
/// Midazolam-Ketoconazole DDI Model
/// Demonstrates gut-wall and hepatic CYP3A4 inhibition
model Midazolam_Ketoconazole_DDI {
    
    // =========================================
    // DRUG DEFINITIONS
    // =========================================
    
    drug Midazolam {
        MW: 325.8 Da
        logP: 3.9
        
        absorption {
            route: oral
            ka: 1.5 1/h
            Fa: 1.0           // Fraction absorbed
        }
        
        distribution {
            Vc: 30 L
            Vp: 50 L
            Q: 25 L/h
            fu: 0.03          // Fraction unbound
        }
        
        metabolism {
            CYP3A4: 94%, Km = 4 µM
            CYP3A5: 4%
            other: 2%
        }
        
        elimination {
            CL: 25 L/h
            fe: 0.01          // Fraction excreted unchanged
        }
        
        first_pass {
            Fg: 0.57          // Gut bioavailability
            Fh: 0.77          // Hepatic bioavailability
        }
    }
    
    drug Ketoconazole {
        MW: 531.4 Da
        
        pk {
            ka: 0.8 1/h
            Vc: 200 L
            CL: 10 L/h
            fu: 0.01
        }
        
        inhibition {
            CYP3A4: competitive, Ki = 0.015 µM
            affects: [gut, liver]
        }
    }
    
    // =========================================
    // DDI MODEL
    // =========================================
    
    ddi {
        perpetrator: Ketoconazole
        victim: Midazolam
        
        gut_wall {
            // [I]gut = Dose / 250 mL
            I_gut = Ketoconazole.dose / 250 mL
            Fg_ddi = 1 - (1 - Midazolam.Fg) / (1 + I_gut / Ketoconazole.Ki)
        }
        
        hepatic {
            // [I]h = [I]u,plasma + portal contribution
            I_hepatic = Ketoconazole.Cu + I_portal
            CLint_ratio = 1 / (1 + I_hepatic / Ketoconazole.Ki)
        }
        
        overall {
            AUC_ratio = (Fg_ddi / Midazolam.Fg) * 
                        (Fh_ddi / Midazolam.Fh) * 
                        (1 / CLint_ratio)
        }
    }
    
    // =========================================
    // SIMULATION
    // =========================================
    
    simulate {
        // Perpetrator pre-treatment
        dose Ketoconazole 400 mg oral at -24 h
        dose Ketoconazole 400 mg oral at 0 h
        
        // Victim administration
        dose Midazolam 7.5 mg oral at 1 h
        
        time: 0..48 h
        step: 0.1 h
        
        observe {
            C_midazolam_plasma as "Midazolam Cp (µM)"
            C_ketoconazole_plasma as "Ketoconazole Cp (µM)"
            AUC_ratio as "AUC Ratio"
        }
        
        // Compare with and without DDI
        scenarios {
            baseline: without Ketoconazole
            ddi: with Ketoconazole
        }
    }
    
    // =========================================
    // VALIDATION
    // =========================================
    
    validate {
        /// Clinical observation: Olkkola et al. 1994
        expected AUC_ratio: 15..16
        expected Cmax_ratio: 3..5
        
        assert AUC_ratio within 2-fold of 15.4
    }
}
```

---

## 9. Compiler Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MedLang Source                           │
│                    (.medlang files)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Lexer   │→│  Parser  │→│   AST    │→│ Type Checker │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                                │                │
│                         ┌──────────────────────┘                │
│                         ▼                                       │
│              ┌─────────────────────┐                           │
│              │   MedLang HIR       │  (High-level IR)          │
│              │   (Typed + Units)   │                           │
│              └─────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Sounio       │ │ Julia           │ │ PharmML         │
│ Backend         │ │ Backend         │ │ Backend         │
│                 │ │                 │ │                 │
│ - Unit types    │ │ - Quick dev     │ │ - Regulatory    │
│ - GPU effects   │ │ - Ecosystem     │ │ - Validation    │
│ - Linear types  │ │ - Debugging     │ │ - Standards     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   .d files      │ │   .jl files     │ │   .xml files    │
│   (Sounio)   │ │   (Julia)       │ │   (PharmML)     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## 10. Future Extensions

### 10.1 Planned Features

- **QSP Constructs**: TMDD, receptor binding, cell dynamics
- **PBPK Organs**: Tissue-specific compartments with zonation
- **Machine Learning**: GNN integration for property prediction
- **Bayesian Inference**: MCMC/VI for parameter estimation
- **Optimal Design**: D-optimal, adaptive dosing

### 10.2 Sounio-Specific Features

When targeting Sounio, MedLang can leverage:

```medlang
// GPU-accelerated population simulation
@gpu
population Virtual_Patients {
    n: 10000
    // ... runs on GPU automatically
}

// Probabilistic programming
@probabilistic
estimate {
    prior CL ~ LogNormal(25, 0.3)
    likelihood C_obs ~ Normal(C_pred, sigma)
    posterior via NUTS(1000 samples)
}

// Linear types for memory safety
@linear
resource PatientData {
    // Guaranteed single ownership
    // No memory leaks possible
}
```

---

## Appendix A: Unit Conversion Table

| From | To | Factor |
|------|-----|--------|
| mL/min | L/h | × 0.06 |
| µg/mL | mg/L | × 1.0 |
| µM | ng/mL | × MW/1000 |
| ng/mL | µM | × 1000/MW |
| pmol/min/mg | nmol/min/mg | × 0.001 |
| 1/min | 1/h | × 60 |

---

## Appendix B: Enzyme Reference

| Enzyme | kdeg (h⁻¹) | t½ (h) | Abundance (pmol/mg) |
|--------|------------|--------|---------------------|
| CYP3A4 | 0.019 | 36 | 137 |
| CYP2D6 | 0.029 | 24 | 10 |
| CYP2C9 | 0.014 | 50 | 60 |
| CYP2C19 | 0.019 | 36 | 14 |
| CYP1A2 | 0.014 | 50 | 45 |
| CYP2C8 | 0.014 | 50 | 24 |

---

*This specification is a living document. Contributions welcome.*
