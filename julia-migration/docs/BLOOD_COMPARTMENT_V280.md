# Blood Compartment v2.8.0 - Non-Critical Features & DOID/ICD Integration

## Overview

Version 2.8.0 completes the Blood Compartment with 6 additional non-critical modules, achieving comprehensive SOTA coverage for blood-related PBPK modeling.

## New Modules in v2.8.0

### 1. Immunoglobulin Isotypes (`immunoglobulin_isotypes.jl`)

Complete immunoglobulin modeling beyond IgG:

- **IgG Subclasses**: IgG1, IgG2, IgG3, IgG4 with distinct Fc receptor affinities
- **IgM**: Pentameric structure, 10 binding sites, superior complement activation
- **IgA**: Dimeric secretory form, mucosal immunity
- **IgE**: Mast cell binding, allergic/parasitic responses
- **Complement System**: C1q, C3, C4 concentrations, classical pathway activation
- **Fc Receptor Binding**: FcγRI, FcγRIIa, FcγRIIb, FcγRIIIa affinities

```julia
igg1 = create_igg_subclass("Rituximab", 1)
igm = create_igm("Anti-A")
complement = ComplementSystem()
clearance = calculate_isotype_clearance(igg1, 10.0)  # mg/L
```

### 2. Acute Phase Response (`acute_phase_response.jl`)

IL-6-driven acute phase protein dynamics:

- **Positive APPs**: CRP (100-1000× increase), SAA (1000×), AAG (2-4×), ferritin
- **Negative APPs**: Albumin (↓30-50%), transferrin, transthyretin
- **Triggers**: Sepsis, surgery, trauma, infection, inflammation
- **Severity Levels**: Mild, moderate, severe
- **Time Course**: Dynamic protein changes over 72+ hours

```julia
state = create_acute_phase_state(:sepsis; severity=:severe)
changes = calculate_protein_changes(state)
time_course = get_time_course(:sepsis, :severe, 72.0)
```

### 3. RBC Aging (`rbc_aging.jl`)

Red blood cell age-dependent effects:

- **Age Distribution**: Cohort-based modeling (0-120 days)
- **Reticulocyte Effects**: Enhanced transport capacity in young RBCs
- **RDW Impact**: Red cell distribution width effects on PK variability
- **Transporter Expression**: Age-weighted GLUT1, ENT1, AE1 activity
- **Disease States**: Hemolytic anemia, sickle cell, thalassemia

```julia
population = create_normal_rbc_population()
hemolytic = create_disease_population(:hemolytic_anemia)
transport = calculate_age_weighted_transport(population, :GLUT1)
dist = get_age_distribution(population)
```

### 4. Spleen RES Clearance (`spleen_res_clearance.jl`)

Reticuloendothelial system clearance:

- **Splenic Macrophages**: Red pulp, white pulp macrophage pools
- **RES Capacity**: Liver (70%), spleen (15%), bone marrow (10%), lymph nodes
- **Fc Receptor Clearance**: FcγRI, FcγRIII-mediated uptake
- **Particle Size Effects**: Size-dependent splenic filtration
- **Splenectomy**: Post-splenectomy PK adjustments

```julia
spleen = create_normal_spleen()
splenomegaly = create_disease_spleen(:splenomegaly)
res = RESCapacity()
post_spx = apply_splenectomy(res)
clearance = calculate_res_clearance(100.0, 0.8, res)
```

### 5. Circadian Effects (`circadian_effects.jl`)

Chronopharmacokinetic modeling:

- **Parameter Rhythms**: Albumin, AAG, GFR, hepatic blood flow
- **Cosinor Model**: Amplitude, acrophase, period for each parameter
- **Chronotypes**: Morning, intermediate, evening adjustments
- **Optimal Dosing**: Drug-specific best administration times
- **Chronotherapy**: Benefit estimation for timed dosing

```julia
params = create_default_parameters()
variation = simulate_circadian_variation(:albumin, 48.0)
optimal = calculate_optimal_dosing_time(:hepatic)
chronotype = get_chronotype_adjustment(:morning)
```

### 6. Disease Ontology PK (`disease_ontology_pk.jl`)

**Key Feature**: DOID + ICD-10 + ICD-11 integration for standardized disease-PK mapping.

#### Supported Diseases (19 total):

| Category | Diseases |
|----------|----------|
| **Renal** | CKD (all stages), ESRD, AKI |
| **Hepatic** | Cirrhosis, Alcoholic liver disease, NAFLD |
| **Diabetes** | Type 1, Type 2, General DM |
| **Cardiac** | Heart failure |
| **Inflammatory** | Rheumatoid arthritis, SLE, IBD |
| **Critical Illness** | Sepsis, Burns |
| **Other** | Pregnancy, Obesity, Cancer |

#### Usage Examples:

```julia
# Lookup by DOID
t2dm = get_pk_adjustments_by_doid("DOID:9352")
# Returns: DiseasePKProfile with GFR, hepatic, fu adjustments

# Lookup by ICD-10
ckd4 = get_pk_adjustments_by_icd10("N18.4")
# Returns: CKD Stage 4 profile (GFR 15-29 mL/min)

# Lookup by ICD-11
cirrhosis = get_pk_adjustments_by_icd11("DB93.1")

# Search by name
results = search_disease_pk("kidney")

# Combine comorbidities
combined = combine_disease_profiles([t2dm, ckd])
# Takes worst-case for each parameter
```

#### PK Adjustments Provided:

- `gfr_adjustment`: Renal function multiplier
- `hepatic_adjustment`: Hepatic metabolism multiplier
- `fu_acidic_adjustment`: Free fraction for acidic drugs
- `fu_basic_adjustment`: Free fraction for basic drugs
- `vd_adjustment`: Volume of distribution multiplier
- `absorption_adjustment`: Oral absorption multiplier
- `albumin_concentration`: Expected albumin (g/L)
- `aag_concentration`: Expected AAG (g/L)
- `special_considerations`: Clinical notes
- `evidence_level`: :high, :moderate, :low, :extrapolated

## Clinical Relevance

### Disease-Specific Dosing

The DOID/ICD integration enables:

1. **EHR Integration**: Direct lookup from patient diagnoses
2. **Regulatory Compliance**: Standardized disease identification
3. **Population PBPK**: Disease covariates for simulations
4. **Polypharmacy**: Comorbidity-aware dosing

### Example: Septic Patient

```julia
# Get sepsis PK adjustments
sepsis = get_pk_adjustments_by_doid("DOID:0080559")

# Key changes:
# - GFR: 60% (AKI common)
# - fu_acidic: 2.0× (hypoalbuminemia)
# - fu_basic: 0.5× (elevated AAG)
# - Vd: 1.8× (capillary leak)
# - Albumin: 20 g/L
# - AAG: 2.5 g/L

# Combine with acute phase response
apr_state = create_acute_phase_state(:sepsis; severity=:severe)
# Dynamic tracking of protein changes
```

## Module Summary

| Module | Key Features | Clinical Application |
|--------|--------------|---------------------|
| Immunoglobulin Isotypes | IgG/M/A/E, complement, FcR | mAb pharmacology |
| Acute Phase Response | IL-6, CRP, AAG dynamics | Inflammation effects |
| RBC Aging | Age distribution, RDW | Drug transport variability |
| Spleen RES | Macrophage clearance | Particle/complex removal |
| Circadian Effects | Cosinor rhythms | Chronotherapy optimization |
| Disease Ontology PK | DOID/ICD-10/ICD-11 | Standardized disease PK |

## Version History

- **v2.7.0**: Lipoprotein binding, RBC transporters, disease state binding, mAb PBPK
- **v2.8.0**: Non-critical modules (6) + DOID/ICD integration

## References

1. Schriml LM et al. (2019) Human Disease Ontology 2018 update. Nucleic Acids Res.
2. WHO ICD-11 Reference Guide (2022)
3. Roberts JA et al. (2014) Individualised antibiotic dosing in sepsis. Lancet Infect Dis.
4. Blanchet B et al. (2008) Drug disposition in burn patients. Clin Pharmacokinet.
5. Abduljalil K et al. (2012) Drug PK in pregnancy. Clin Pharmacokinet.
