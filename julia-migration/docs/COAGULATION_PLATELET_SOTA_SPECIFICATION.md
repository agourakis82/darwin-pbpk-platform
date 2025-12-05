# Coagulation & Platelet Module - SOTA Technical Specification

## Darwin PBPK Platform
**Date:** 2025-12-05  
**Status:** Deep Research Complete  
**Sources:** Q1 Literature 2020-2024

---

## 1. Executive Summary

This specification defines the implementation of platelet and coagulation factor compartments based on state-of-the-art QSP (Quantitative Systems Pharmacology) models from the literature.

### Key Models Referenced

| Model | Year | Features | Equations |
|-------|------|----------|-----------|
| **Wajima et al.** | 2009 | Complete coagulation network | 56 ODEs |
| **Hockin-Mann** | 2002 | Thrombin generation | 34 ODEs, 42 rate constants |
| **Hartmann et al.** | 2016 | Anticoagulant effects | 56 compartments |
| **Zhou et al.** | 2015 | FXa inhibitors (DOACs) | PT/aPTT prediction |

### Primary Sources

- [Wajima Model - Clinical Pharmacology & Therapeutics 2009](https://ascpt.onlinelibrary.wiley.com/doi/abs/10.1038/clpt.2009.87)
- [Hockin-Mann Model - BioModels BIOMD0000000335](https://www.ebi.ac.uk/biomodels/BIOMD0000000335)
- [QSP Coagulation Review 2023 - PMC10054658](https://pmc.ncbi.nlm.nih.gov/articles/PMC10054658/)
- [Hartmann QSP Model 2016](https://ascpt.onlinelibrary.wiley.com/doi/10.1002/psp4.12111)

---

## 2. Coagulation Factor Parameters

### 2.1 Plasma Concentrations (Normal Human)

| Factor | Name | Concentration | MW (kDa) | Half-life |
|--------|------|---------------|----------|-----------|
| **II** | Prothrombin | 1400 nM (100 μg/mL) | 72 | 60-72 h |
| **V** | Proaccelerin | 20-25 nM | 330 | 12-36 h |
| **VII** | Proconvertin | 10 nM (0.5 μg/mL) | 50 | 3-6 h |
| **VIIa** | Activated VII | 0.1 nM (3.6 ng/mL) | 50 | 2.5 h |
| **VIII** | Anti-hemophilic A | 0.7 nM | 280 | 12 h |
| **IX** | Christmas factor | 90 nM (5 μg/mL) | 55 | 18-24 h |
| **X** | Stuart-Prower | 170 nM (10 μg/mL) | 59 | 48 h |
| **XI** | Plasma thromboplastin | 30-60 nM | 160 | 60-80 h |
| **XII** | Hageman factor | 375 nM | 80 | 50-70 h |
| **XIII** | Fibrin-stabilizing | 70 nM | 320 | 120-200 h |
| Fibrinogen | - | 7-12 μM (2-4 g/L) | 340 | 96-120 h |
| Prothrombin | - | 1.4 μM | 72 | 60-72 h |

### 2.2 Anticoagulant Proteins

| Protein | Concentration | MW (kDa) | Function |
|---------|---------------|----------|----------|
| Antithrombin III | 2.4 μM | 58 | Inhibits IIa, Xa, IXa |
| TFPI | 2.5 nM | 42 | Inhibits TF·VIIa·Xa |
| Protein C | 65 nM | 62 | Inactivates Va, VIIIa |
| Protein S | 350 nM | 75 | Cofactor for Protein C |

### 2.3 Vitamin K-Dependent Factor Synthesis

Warfarin mechanism - inhibits VKORC1:
```
Vitamin K epoxide → (VKORC1) → Vitamin K quinone → Vitamin K hydroquinone
                     ↑ BLOCKED BY WARFARIN
```

Affected factors: II, VII, IX, X, Protein C, Protein S

**Turnover model for warfarin:**
```
dF/dt = k_syn × (1 - I_warfarin) - k_deg × F

Where:
- k_syn = synthesis rate constant
- k_deg = degradation rate (ln(2)/half-life)
- I_warfarin = S-warfarin inhibitory effect (Emax model)
```

---

## 3. Coagulation Cascade ODE Model

### 3.1 Simplified Model Structure (Based on Wajima)

The cascade is divided into:
1. **Initiation** (TF pathway)
2. **Amplification** (thrombin feedback)
3. **Propagation** (prothrombinase complex)
4. **Termination** (antithrombin, TFPI)

### 3.2 Core Reactions

```
# Tissue Factor Pathway (Extrinsic)
TF + VII ⇌ TF·VII → TF·VIIa
TF·VIIa + X → TF·VIIa + Xa
TF·VIIa + IX → TF·VIIa + IXa

# Common Pathway
Xa + Va + II → Xa·Va·II → IIa (Thrombin)
IIa + Fibrinogen → Fibrin

# Thrombin Feedback (Amplification)
IIa + V → IIa + Va
IIa + VIII → IIa + VIIIa
IIa + XI → IIa + XIa

# Tenase Complex
IXa + VIIIa + X → IXa·VIIIa + Xa

# Inhibition
ATIII + IIa → ATIII·IIa (inactive)
ATIII + Xa → ATIII·Xa (inactive)
TFPI + Xa → TFPI·Xa
TFPI·Xa + TF·VIIa → TFPI·Xa·TF·VIIa (inactive)
```

### 3.3 Michaelis-Menten Kinetics

General form for enzyme-catalyzed reactions:
```
d[Product]/dt = (Vmax × [Substrate]) / (Km + [Substrate])

# Or in terms of rate constants:
d[P]/dt = kcat × [E] × [S] / (Km + [S])
```

### 3.4 Key Rate Constants (from Hockin-Mann)

| Reaction | kcat (s⁻¹) | Km (nM) | Reference |
|----------|-----------|---------|-----------|
| Xa → IIa | 32.4 | 300 | Prothrombinase |
| IXa → Xa | 10.7 | 160 | Tenase |
| VIIa → Xa | 103 | 450 | Extrinsic |
| VIIa → IXa | 2.3 | 300 | Extrinsic |
| IIa → Va | 20 | 100 | Feedback |
| IIa → VIIIa | 20 | 100 | Feedback |
| ATIII + IIa | 7.1×10³ M⁻¹s⁻¹ | - | Inhibition |
| ATIII + Xa | 1.5×10³ M⁻¹s⁻¹ | - | Inhibition |

---

## 4. Direct Oral Anticoagulants (DOACs)

### 4.1 Factor Xa Inhibitors

| Drug | Ki (nM) | IC50 Free Xa | IC50 Prothrombinase | kon (M⁻¹s⁻¹) | koff (s⁻¹) |
|------|---------|--------------|---------------------|--------------|------------|
| **Rivaroxaban** | 0.4 | 0.7 | 2.1 | 1.7×10⁷ | 5×10⁻³ |
| **Apixaban** | 0.08 | 0.08 | 0.62 | 1.7×10⁶ | 2×10⁻³ |
| **Edoxaban** | 0.56 | 0.56 | 1.6 | - | - |
| **Betrixaban** | 0.12 | 0.12 | 0.35 | - | - |

### 4.2 Direct Thrombin Inhibitors

| Drug | Ki (nM) | IC50 | kon (M⁻¹s⁻¹) | koff (s⁻¹) |
|------|---------|------|--------------|------------|
| **Dabigatran** | 4.5 | 4.5 | 1.5×10⁵ | 6.8×10⁻⁴ |
| **Argatroban** | 39 | 19 | 6.3×10⁵ | 2.5×10⁻² |
| **Bivalirudin** | 1.9 | 1.9 | 7.4×10⁵ | 1.4×10⁻³ |

### 4.3 DOAC PD Model

```julia
# Competitive inhibition of Factor Xa
function inhibit_fxa(Xa_conc, drug_conc, Ki)
    return Xa_conc / (1 + drug_conc / Ki)
end

# Effect on thrombin generation
# Rivaroxaban IC50 for TG = 75 nM (clot-bound Xa)
```

---

## 5. Warfarin PK/PD Model

### 5.1 Pharmacokinetics

| Parameter | S-Warfarin | R-Warfarin |
|-----------|------------|------------|
| CL (L/h) | 0.18 | 0.08 |
| Vd (L) | 9.5 | 9.5 |
| t½ (h) | 32 | 45 |
| Potency ratio | 1.0 | 0.2-0.5 |

### 5.2 Pharmacodynamics - INR Model

```julia
# Transit compartment model for vitamin K-dependent factors
# Two parallel chains with different MTT

MTT1 = 27.2  # hours (rapid response factors: VII)
MTT2 = 110.9 # hours (slow response factors: II, IX, X)

# Inhibitory Emax model
I_warfarin = (Imax × C_S_warfarin^γ) / (IC50^γ + C_S_warfarin^γ)

# Factor synthesis inhibition
d[F_VK]/dt = k_syn × (1 - I_warfarin) - k_deg × [F_VK]

# INR calculation
INR = (PCA_patient / PCA_normal)^ISI

# Where PCA = Prothrombin Complex Activity
# Depends on factors II, VII, IX, X
```

### 5.3 Genetic Effects

| Genotype | Effect | Parameter |
|----------|--------|-----------|
| VKORC1 A/A | High sensitivity | IC50 ↓ 50% |
| VKORC1 B/B | Low sensitivity | IC50 ↑ 50% |
| CYP2C9 *1/*2 | Reduced metabolism | CL ↓ 20% |
| CYP2C9 *2/*2 | Poor metabolizer | CL ↓ 60% |
| CYP2C9 *3/*3 | Poor metabolizer | CL ↓ 80% |

---

## 6. Platelet Compartment

### 6.1 Physical Parameters

| Parameter | Value | Range |
|-----------|-------|-------|
| Count | 250 × 10⁹/L | 150-400 × 10⁹/L |
| Volume (MPV) | 7.5 fL | 4.5-11 fL |
| Diameter | 3.5 μm | 3-5 μm |
| Lifespan | 8-10 days | - |
| Turnover | 35 × 10⁹/day | - |

### 6.2 Granule Content

**Alpha Granules (50-80 per platelet, 10% of volume)**
| Content | Function |
|---------|----------|
| Fibrinogen | Clot formation |
| vWF | Platelet adhesion |
| Factor V | Prothrombinase |
| PAI-1 | Fibrinolysis inhibition |
| PDGF | Wound healing |
| P-selectin | Adhesion |

**Dense Granules (6-7 per platelet)**
| Content | Concentration | Function |
|---------|---------------|----------|
| ADP | 0.4-0.6 M | P2Y12 activation |
| ATP | 0.5 M | Energy |
| Serotonin (5-HT) | 65 mM | Vasoconstriction |
| Ca²⁺ | 2.2 M | Signaling |
| Polyphosphate | - | Coagulation enhancer |

### 6.3 Platelet Activation Pathways

```
                    ┌─────────────────────────────────────┐
                    │         PLATELET ACTIVATION         │
                    └─────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
   ┌─────────┐                  ┌─────────┐                   ┌─────────┐
   │ Collagen│                  │Thrombin │                   │   ADP   │
   │ (GPVI)  │                  │ (PAR-1) │                   │ (P2Y12) │
   └────┬────┘                  └────┬────┘                   └────┬────┘
        │                             │                             │
        ▼                             ▼                             ▼
   ┌─────────┐                  ┌─────────┐                   ┌─────────┐
   │   PLC   │                  │   Gq    │                   │   Gi    │
   └────┬────┘                  └────┬────┘                   └────┬────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │ Ca²⁺ release  │
                              │ COX-1 → TXA₂  │
                              │ GPIIb/IIIa    │
                              │ activation    │
                              └───────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │  AGGREGATION  │
                              │ + Fibrinogen  │
                              └───────────────┘
```

### 6.4 Antiplatelet Drug Targets

| Drug | Target | Mechanism | IC50/Ki |
|------|--------|-----------|---------|
| **Aspirin** | COX-1 | Irreversible acetylation | IC50 = 3 μM |
| **Clopidogrel** | P2Y12 | Irreversible (active metabolite) | - |
| **Prasugrel** | P2Y12 | Irreversible (active metabolite) | - |
| **Ticagrelor** | P2Y12 | Reversible | Ki = 2 nM |
| **Abciximab** | GPIIb/IIIa | mAb | Kd = 5 nM |
| **Eptifibatide** | GPIIb/IIIa | Cyclic peptide | Ki = 120 pM |

### 6.5 Platelet Aggregation Model

```julia
# ADP-induced aggregation (simplified)
struct PlateletAggregation
    # State
    resting_fraction::Float64      # 0-1
    activated_fraction::Float64    # 0-1
    aggregated_fraction::Float64   # 0-1
    
    # Parameters
    k_activation::Float64          # 1/s, ADP-dependent
    k_aggregation::Float64         # 1/s
    k_disaggregation::Float64      # 1/s
    
    # Drug effects
    p2y12_inhibition::Float64      # 0-1 (clopidogrel)
    cox1_inhibition::Float64       # 0-1 (aspirin)
    gpiib_iiia_inhibition::Float64 # 0-1 (abciximab)
end

# Aggregation ODEs
function platelet_aggregation_ode!(du, u, p, t)
    R, A, Ag = u  # Resting, Activated, Aggregated
    ADP, TXA2 = p.agonists
    
    # Activation rate
    k_act = p.k_activation * (ADP / (ADP + p.EC50_ADP)) * (1 - p.p2y12_inhibition)
    k_act += p.k_activation * (TXA2 / (TXA2 + p.EC50_TXA2)) * (1 - p.cox1_inhibition)
    
    # Aggregation rate
    k_agg = p.k_aggregation * (1 - p.gpiib_iiia_inhibition)
    
    du[1] = -k_act * R  # dR/dt
    du[2] = k_act * R - k_agg * A * A  # dA/dt (second order)
    du[3] = k_agg * A * A - p.k_disagg * Ag  # dAg/dt
end
```

---

## 7. Integration with PBPK

### 7.1 Blood Compartment Structure

```julia
struct BloodCompartmentComplete
    # Existing components
    plasma::PlasmaCompartment
    rbc::RBCCompartment
    wbc::WBCCompartment  # Already implemented (7 subtypes)
    
    # NEW: Platelet compartment
    platelets::PlateletCompartment
    
    # NEW: Coagulation factors
    coagulation::CoagulationSystem
    
    # Dynamics
    fractal_blood::FractalBloodModel  # CTRW dynamics
end
```

### 7.2 Drug-Coagulation Interactions

```julia
function calculate_drug_effect_on_coagulation(
    drug::Drug,
    coag::CoagulationSystem,
    C_plasma::Float64
)
    if drug.class == :factor_xa_inhibitor
        # Direct FXa inhibition
        Xa_activity = coag.factor_Xa / (1 + C_plasma / drug.Ki_Xa)
        return update_thrombin_generation(coag, Xa_activity)
        
    elseif drug.class == :direct_thrombin_inhibitor
        # Direct IIa inhibition
        IIa_activity = coag.thrombin / (1 + C_plasma / drug.Ki_IIa)
        return update_fibrin_formation(coag, IIa_activity)
        
    elseif drug.class == :vitamin_k_antagonist
        # VKORC1 inhibition → reduced factor synthesis
        I_effect = emax_inhibition(C_plasma, drug.IC50, drug.gamma)
        return update_vk_factor_synthesis(coag, I_effect)
    end
end
```

### 7.3 INR Calculation

```julia
function calculate_INR(coag::CoagulationSystem)
    # Prothrombin Complex Activity (PCA)
    # Depends on factors II, VII, IX, X
    
    PCA = (
        0.3 * coag.factor_VII / NORMAL_VII +
        0.3 * coag.factor_X / NORMAL_X +
        0.2 * coag.factor_II / NORMAL_II +
        0.2 * coag.factor_IX / NORMAL_IX
    )
    
    # INR = (PT_patient / PT_normal)^ISI
    # Approximated as:
    INR = (1.0 / PCA) ^ ISI  # ISI ≈ 1.0-1.4
    
    return INR
end
```

---

## 8. Implementation Roadmap

### Phase 1: Core Structures (Week 1)

```julia
# File: platelet_compartment.jl
struct PlateletCompartment
    count::Float64              # cells/L
    mpv::Float64                # fL
    activation_state::Float64   # 0-1
    aggregation_state::Float64  # 0-1
    
    # Granule contents
    adp_released::Float64       # M
    serotonin_released::Float64 # M
    txa2_generated::Float64     # M
    
    # Drug effects
    p2y12_inhibition::Float64
    cox1_inhibition::Float64
    gpiib_iiia_inhibition::Float64
end

# File: coagulation_system.jl
struct CoagulationSystem
    # Zymogens (inactive)
    factor_II::Float64   # Prothrombin
    factor_V::Float64
    factor_VII::Float64
    factor_VIII::Float64
    factor_IX::Float64
    factor_X::Float64
    factor_XI::Float64
    fibrinogen::Float64
    
    # Activated factors
    factor_IIa::Float64  # Thrombin
    factor_Va::Float64
    factor_VIIa::Float64
    factor_VIIIa::Float64
    factor_IXa::Float64
    factor_Xa::Float64
    factor_XIa::Float64
    
    # Inhibitors
    antithrombin::Float64
    tfpi::Float64
    protein_c::Float64
    protein_s::Float64
    
    # Vitamin K status
    vitamin_k::Float64
    vkorc1_activity::Float64
end
```

### Phase 2: ODE System (Week 1-2)

```julia
function coagulation_ode!(du, u, p, t)
    # Unpack state
    II, V, VII, VIII, IX, X, XI, Fg = u[1:8]      # Zymogens
    IIa, Va, VIIa, VIIIa, IXa, Xa, XIa = u[9:15]  # Active
    ATIII, TFPI = u[16:17]                         # Inhibitors
    
    # TF concentration (input)
    TF = p.tissue_factor
    
    # Extrinsic pathway
    rate_VIIa_Xa = p.kcat_VIIa_X * VIIa * X / (p.Km_VIIa_X + X)
    rate_VIIa_IXa = p.kcat_VIIa_IX * VIIa * IX / (p.Km_VIIa_IX + IX)
    
    # Tenase complex
    rate_IXa_Xa = p.kcat_tenase * IXa * VIIIa * X / (p.Km_tenase + X)
    
    # Prothrombinase complex
    rate_Xa_IIa = p.kcat_prothrombinase * Xa * Va * II / (p.Km_prothrombinase + II)
    
    # Thrombin feedback
    rate_IIa_Va = p.kcat_IIa_V * IIa * V / (p.Km_IIa_V + V)
    rate_IIa_VIIIa = p.kcat_IIa_VIII * IIa * VIII / (p.Km_IIa_VIII + VIII)
    
    # Inhibition by ATIII
    rate_ATIII_IIa = p.k_ATIII_IIa * ATIII * IIa
    rate_ATIII_Xa = p.k_ATIII_Xa * ATIII * Xa
    
    # TFPI inhibition
    rate_TFPI_Xa = p.k_TFPI_Xa * TFPI * Xa
    
    # Drug effects (DOACs)
    if p.has_fxa_inhibitor
        Xa_effective = Xa / (1 + p.C_drug / p.Ki_Xa)
        rate_Xa_IIa *= Xa_effective / Xa
    end
    
    # ODEs
    du[1] = -rate_Xa_IIa + p.k_syn_II * p.vk_synthesis  # dII/dt
    du[7] = -rate_VIIa_Xa - rate_VIIa_IXa               # dVII/dt (simplified)
    du[10] = -rate_IXa_Xa                                # dIX/dt
    du[6] = -rate_Xa_IIa                                 # dX/dt
    
    du[9] = rate_Xa_IIa - rate_ATIII_IIa                 # dIIa/dt
    du[14] = rate_VIIa_Xa + rate_IXa_Xa - rate_ATIII_Xa - rate_TFPI_Xa  # dXa/dt
    
    # ... continue for all factors
end
```

### Phase 3: Clinical Endpoints (Week 2)

```julia
# PT/INR calculation
function calculate_pt_inr(coag::CoagulationSystem)
    # PT depends on factors II, V, VII, X and fibrinogen
    # Simplified: PT ∝ 1/[Xa generation rate]
    
    pt_ratio = NORMAL_PT / estimated_pt(coag)
    inr = pt_ratio ^ ISI
    return (pt=estimated_pt(coag), inr=inr)
end

# aPTT calculation
function calculate_aptt(coag::CoagulationSystem)
    # aPTT depends on intrinsic pathway: VIII, IX, XI, XII
    # + common pathway: II, V, X, fibrinogen
    
    return estimated_aptt(coag)
end

# Anti-Xa activity (for DOAC monitoring)
function calculate_anti_xa(coag::CoagulationSystem, drug_conc::Float64, drug::Drug)
    # Anti-Xa activity in IU/mL
    # Calibrated to drug concentration
    
    return drug_conc / drug.anti_xa_calibration
end
```

---

## 9. Validation Targets

### 9.1 Thrombin Generation Curve (TGC)

| Parameter | Normal Range | Unit |
|-----------|--------------|------|
| Lag time | 3-5 | min |
| Peak thrombin | 200-400 | nM |
| Time to peak | 5-8 | min |
| Endogenous thrombin potential (ETP) | 1200-1800 | nM·min |

### 9.2 Coagulation Assays

| Test | Normal Range | Therapeutic Range (Warfarin) |
|------|--------------|------------------------------|
| PT | 11-15 s | - |
| INR | 0.9-1.1 | 2.0-3.0 |
| aPTT | 25-35 s | 1.5-2.5× normal (heparin) |

### 9.3 DOAC Concentrations

| Drug | Cmax (ng/mL) | Ctrough (ng/mL) | Anti-Xa (IU/mL) |
|------|--------------|-----------------|-----------------|
| Rivaroxaban 20mg | 200-270 | 20-40 | 0.5-1.5 |
| Apixaban 5mg BID | 100-150 | 50-100 | 0.5-1.0 |
| Edoxaban 60mg | 150-200 | 20-40 | 0.3-0.8 |

---

## 10. References

### Primary Literature

1. Wajima T, et al. (2009) A comprehensive model for the humoral coagulation network in humans. *Clin Pharmacol Ther* 86:290-298.

2. Hockin MF, et al. (2002) A model for the stoichiometric regulation of blood coagulation. *J Biol Chem* 277:18322-18333.

3. Hartmann R, et al. (2016) QSP model to predict effects of commonly used anticoagulants. *CPT Pharmacometrics Syst Pharmacol* 5:554-564.

4. Zhou K, et al. (2015) A systems pharmacology model for FXa inhibitors. *CPT Pharmacometrics Syst Pharmacol* 5:146-154.

5. [Blood 2023 - Rethinking Coagulation](https://ashpublications.org/blood/article/142/25/2133/498486)

6. [J Thromb Haemost 2024 - Mathematical Models of Coagulation](https://www.jthjournal.org/article/S1538-7836(24)00167-3/fulltext)

### Databases

- BioModels: [BIOMD0000000335](https://www.ebi.ac.uk/biomodels/BIOMD0000000335) (Hockin-Mann)
- DrugBank: Rivaroxaban, Apixaban, Dabigatran, Warfarin entries

---

*Darwin PBPK Platform - Q1 Scientific Rigor*
