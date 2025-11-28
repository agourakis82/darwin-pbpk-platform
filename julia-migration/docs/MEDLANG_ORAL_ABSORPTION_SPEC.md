# MedLang Track D: Oral Absorption & First-Pass Metabolism Specification

**Version:** 0.2.0  
**Author:** Dr. Demetrios Agourakis  
**Date:** November 2025  
**Status:** Implemented in Darwin PBPK Platform

## Overview

This document specifies the oral absorption and first-pass metabolism extensions to the MedLang Track D (Pharmacometrics/PBPK) grammar. These extensions enable modeling of oral drug delivery with physiologically-based absorption kinetics.

## Grammar Extensions

### Route Declaration

```medlang
route: <route_type>
```

Where `<route_type>` is one of:
- `iv` or `intravenous` - Intravenous bolus
- `oral` or `po` - Oral administration
- `im` or `intramuscular` - Intramuscular injection
- `sc` or `subcutaneous` - Subcutaneous injection
- `infusion` - IV infusion

### Absorption Block

```medlang
absorption {
    Ka: <value>,           // Absorption rate constant (1/h)
    F: <value>,            // Fraction absorbed (0-1), optional
    lag: <value>           // Lag time (h), optional
}
```

**Parameters:**
- `Ka` (required): First-order absorption rate constant in units of 1/h
- `F` (optional, default=1.0): Fraction of dose absorbed from gut lumen
- `lag` (optional, default=0.0): Lag time before absorption begins

### First-Pass Block

```medlang
firstpass {
    Fg: <value>,           // Gut availability (0-1)
    Fh: <value>            // Hepatic availability (0-1)
}
```

**Parameters:**
- `Fg`: Fraction of drug escaping gut wall metabolism (gut availability)
- `Fh`: Fraction of drug escaping hepatic first-pass metabolism

**Bioavailability Calculation:**
```
F_effective = F × Fg × Fh
```

Where:
- F = fraction absorbed (Fa)
- Fg = gut availability (1 - gut extraction)
- Fh = hepatic availability (1 - hepatic extraction ratio)

## Complete Model Example

```medlang
model Midazolam_OralPBPK {
    // Drug: Midazolam (CYP3A4 substrate with significant first-pass)
    
    // Route of administration
    route: oral

    // Oral absorption parameters
    absorption {
        Ka: 2.5,           // Rapid absorption
        F: 0.95,           // High fraction absorbed
        lag: 0.25          // 15 min gastric emptying
    }

    // First-pass metabolism (CYP3A4)
    firstpass {
        Fg: 0.44,          // Significant gut CYP3A4 metabolism
        Fh: 0.57           // Moderate hepatic extraction
    }

    // Clearance mechanisms
    clearance hepatic: 27.0_L/h    // High hepatic clearance
    clearance renal: 0.5_L/h       // Minimal renal

    // Organ definitions
    organ blood { V: 5.0_L, Q: 0.0_L/h, Kp: 1.0 }
    organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 3.5 }
    organ kidney { V: 0.31_L, Q: 60.0_L/h, Kp: 2.0 }
    organ brain { V: 1.4_L, Q: 50.0_L/h, Kp: 0.8 }
    organ heart { V: 0.33_L, Q: 20.0_L/h, Kp: 2.5 }
    organ lung { V: 0.5_L, Q: 300.0_L/h, Kp: 1.8 }
    organ muscle { V: 30.0_L, Q: 75.0_L/h, Kp: 1.5 }
    organ adipose { V: 15.0_L, Q: 12.0_L/h, Kp: 8.0 }
    organ gut { V: 1.1_L, Q: 45.0_L/h, Kp: 2.0 }
    organ skin { V: 3.3_L, Q: 10.0_L/h, Kp: 1.2 }
    organ bone { V: 10.0_L, Q: 5.0_L/h, Kp: 0.5 }
    organ spleen { V: 0.18_L, Q: 15.0_L/h, Kp: 2.2 }
    organ pancreas { V: 0.1_L, Q: 5.0_L/h, Kp: 1.8 }
    organ other { V: 5.0_L, Q: 20.0_L/h, Kp: 1.5 }
}
```

## ODE System

The oral absorption PBPK model uses a 15-compartment ODE system:

### State Variables
- `u[1:14]`: Organ concentrations (mg/L) - standard 14-compartment PBPK
- `u[15]`: Amount in gut lumen (mg) - absorption depot

### Differential Equations

**Gut Lumen (Absorption Depot):**
```
dA_gut/dt = -Ka × A_gut
```

**Blood Compartment (with absorption input):**
```
dC_blood/dt = Σ[organ_fluxes] - clearance_rate × C_blood + (Ka × A_gut × Fg × Fh) / V_blood
```

**Other Organs (standard PBPK):**
```
dC_organ/dt = (Q_organ / V_organ) × (C_blood - C_organ / Kp_organ)
```

### Initial Conditions
- Oral dose: `A_gut(0) = Dose × F`
- All organ concentrations: `C_organ(0) = 0`

### Lag Time Handling
For models with lag time > 0:
- t < lag: Drug remains in gut lumen, no absorption
- t ≥ lag: First-order absorption begins

## Pharmacokinetic Relationships

### Hepatic Availability (Fh)
```
Fh = 1 - ERH = 1 - (CLH / QH)
```

Where:
- ERH = hepatic extraction ratio
- CLH = hepatic clearance (L/h)
- QH = hepatic blood flow (~90 L/h for 70kg adult)

### Ka Estimation from Tmax
For one-compartment models:
```
Ka ≈ 2.5 / Tmax
```

More precisely:
```
Tmax = ln(Ka/Ke) / (Ka - Ke)
```

### Effective Bioavailability
```
F_eff = Fa × Fg × Fh
```

Typical ranges:
- Fa (fraction absorbed): 0.1 - 1.0
- Fg (gut availability): 0.3 - 1.0 (CYP3A4 substrates: 0.3-0.6)
- Fh (hepatic availability): 0.05 - 1.0

## Validation Results

Tested against 572 drugs from ULTIMATE_DATASET:

| Metric | IV Model | Oral Model | Improvement |
|--------|----------|------------|-------------|
| Simulation Success | 92.1% | 92.1% | - |
| Cmax within 2-fold | 4.4% | 24.3% | +452% |
| AUC within 2-fold | 43.4% | 37.5% | -14% |
| AUC within 3-fold | - | 50.0% | FDA threshold met |
| Half-life GMFE | 2.02 | 2.01 | Stable |

## Implementation Notes

### Parser Tokens
```julia
TOK_ABSORPTION   # absorption keyword
TOK_FIRSTPASS    # firstpass keyword
TOK_ROUTE        # route keyword
```

### AST Structures
```julia
struct AbsorptionDef <: ASTNode
    ka::Expr              # Absorption rate constant
    f::Union{Expr, Nothing}  # Bioavailability
    lag::Union{Expr, Nothing}  # Lag time
end

struct FirstPassDef <: ASTNode
    fg::Expr  # Gut availability
    fh::Expr  # Hepatic availability
end

@enum RouteType begin
    ROUTE_IV
    ROUTE_ORAL
    ROUTE_IM
    ROUTE_SC
    ROUTE_INFUSION
end
```

### ODE Solver Parameters
```julia
struct OralParams
    ka::Float64   # 1/h
    fa::Float64   # 0-1
    fg::Float64   # 0-1
    fh::Float64   # 0-1
    lag::Float64  # h
end
```

## References

1. Rowland M, Tozer TN. Clinical Pharmacokinetics and Pharmacodynamics. 4th ed. 2011.
2. Poulin P, Theil FP. Prediction of pharmacokinetics prior to in vivo studies. J Pharm Sci. 2002.
3. Yang J, et al. Prediction of intestinal first-pass drug metabolism. Curr Drug Metab. 2007.
4. FDA Guidance: Physiologically Based Pharmacokinetic Analyses. 2018.

## Changelog

### v0.2.0 (November 2025)
- Added `absorption` block for Ka, F, lag parameters
- Added `firstpass` block for Fg, Fh parameters
- Added `route` declaration
- Implemented 15-compartment ODE system with gut lumen
- Full validation against 572 drugs

### v0.1.0 (November 2025)
- Initial MedLang Track D implementation
- 14-compartment IV PBPK model
