# Blood Compartment Implementation v2.6.0

## Overview

Darwin PBPK Platform v2.6.0 introduces a comprehensive **Blood Compartment** module with state-of-the-art mechanistic models for drug distribution in blood, coagulation dynamics, and sensitivity analysis.

## New Modules

### 1. Blood Binding (`blood_binding.jl`)

Mechanistic drug distribution in blood components following PK-Sim methodology.

**Features:**
- **Blood-to-Plasma Ratio (B:P)**: Rodgers-Rowland equations with lipophilicity-based partitioning
- **RBC Binding**: pH-dependent ion trapping, membrane partitioning
- **Plasma Protein Binding**: Albumin (acidic drugs), AAG (basic drugs)
- **Platelet Partitioning**: Surface binding and intracellular accumulation
- **WBC Binding**: Drug-specific accumulation ratios

**Key Functions:**
```julia
calculate_bp_ratio(drug_props, blood_props)
calculate_rbc_partition(drug_props, blood_props)
calculate_fu_blood(fu_plasma, bp_ratio)
```

### 2. Drug-Specific WBC Binding

Comprehensive database for drugs with significant leukocyte accumulation.

**Supported Drugs:**
| Drug | WBC:Plasma Ratio | Mechanism |
|------|-----------------|-----------|
| Chloroquine | 7.0 | Lysosomotropic (pH trap) |
| Hydroxychloroquine | 6.0 | Lysosomotropic |
| Azithromycin | 100.0 | Extreme accumulation |
| Tenofovir | 5.0 | PBMC targeting |
| Dolutegravir | 1.5 | CD4+ preference |
| Darunavir | 2.0 | Lymphocyte binding |
| Ritonavir | 1.8 | Booster effects |
| Maraviroc | 3.0 | CCR5 binding |

**Key Functions:**
```julia
get_wbc_accumulation(drug_name)
calculate_drug_specific_wbc_partition(drug_name, params)
calculate_intracellular_drug_amount(plasma_conc, accumulation, wbc_count)
calculate_reservoir_effect(wbc_amount, elimination_rate, wbc_half_life)
```

### 3. Hemodynamics (`hemodynamics.jl`)

Blood flow mechanics and shear-dependent processes.

**Features:**
- **Shear Stress Calculation**: Poiseuille flow in vessels
- **Shear-Induced Platelet Activation (SIPA)**: Threshold-based activation
- **vWF Unfolding**: Shear-dependent A1 domain exposure
- **Carreau-Yasuda Viscosity**: Non-Newtonian blood rheology

**Key Functions:**
```julia
calculate_wall_shear_stress(vessel)
shear_induced_platelet_activation(shear_rate, exposure_time, baseline)
vwf_unfolding_probability(shear_stress)
```

### 4. Coagulation Extended (`coagulation_extended.jl`)

Enhanced coagulation cascade with critical fixes.

**Features:**
- **FXI Feedback Loop**: Fixes Hockin-Mann model limitation at low TF
- **Contact Pathway**: FXII activation, kallikrein amplification
- **Platelet Surface Enhancement**: 300,000× prothrombinase acceleration
- **Antithrombin Dynamics**: Time-dependent factor inhibition

**Key Structures:**
```julia
ContactPathway  # FXII, PK, HK concentrations
FXIFeedback     # Thrombin-FXI activation rates
```

### 5. TGA Validation (`tga_validation.jl`)

Clinical validation against Thrombin Generation Assay data.

**Clinical Datasets:**
- Healthy controls (1pM TF, 5pM TF)
- Hemophilia A/B
- Warfarin (INR 2.0, 3.0)
- DOACs (Rivaroxaban, Apixaban)
- Factor XI deficiency

**Validation Metrics:**
- AAFE (Average Absolute Fold Error)
- GMFE (Geometric Mean Fold Error)
- R² correlation
- Within 2-fold percentage

**Key Functions:**
```julia
compare_to_clinical(simulated::TGAParameters, clinical::ClinicalTGADataset)
validate_coagulation_model(simulations, datasets)
calculate_goodness_of_fit(observed, predicted)
```

### 6. Lattice Boltzmann CFD (`lattice_boltzmann.jl`)

Computational fluid dynamics for blood flow simulation.

**Features:**
- **D2Q9 Lattice**: 2D nine-velocity model
- **Vessel Geometries**: Straight tube, stenosis, bifurcation, curved
- **Non-Newtonian Viscosity**: Carreau-Yasuda model for blood
- **Wall Shear Stress**: Extraction at vessel boundaries
- **Hematocrit Correction**: Viscosity adjustment for RBC content

**Key Functions:**
```julia
create_lbm_simulation(geometry, fluid, bc)
run_lbm_simulation!(sim, n_steps)
extract_wall_shear_stress(sim)
carreau_yasuda_viscosity(shear_rate, fluid)
```

### 7. Sensitivity Analysis (`sensitivity_analysis.jl`)

Comprehensive parameter sensitivity framework.

**Methods:**
| Method | Type | Use Case |
|--------|------|----------|
| OAT | Local | Quick screening |
| Morris | Global | Factor prioritization |
| Sobol | Global | Variance decomposition |
| PRCC | Global | Correlation analysis |

**Key Functions:**
```julia
one_at_a_time_sensitivity(model, params, outputs)
morris_screening(model, params, outputs; n_trajectories=10)
sobol_sensitivity(model, params, outputs; n_samples=1024)
prcc_analysis(model, params, outputs; n_samples=200)
latin_hypercube_sample(params, n_samples)
default_coagulation_parameters()
```

## Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| Blood Binding | 122 | PASS |
| Hemodynamics | 122 | PASS |
| Coagulation Extended | 122 | PASS |
| Drug-specific WBC | 24 | PASS |
| TGA Validation | 31 | PASS |
| Lattice Boltzmann | 34 | PASS |
| Sensitivity Analysis | 60 | PASS |
| **Total** | **271** | **PASS** |

## Usage Examples

### Blood-to-Plasma Ratio
```julia
using DarwinPBPK

drug = create_drug_properties(
    name = "Metoprolol",
    logP = 1.88,
    pKa = 9.68,
    fu_plasma = 0.88,
    charge_type = :base
)

blood = BloodProperties()
bp_ratio = calculate_bp_ratio(drug, blood)
```

### Sensitivity Analysis
```julia
using DarwinPBPK

# Define parameter ranges
params = [
    ParameterRange("CL", 10.0, 5.0, 20.0; distribution=:lognormal, std=0.3),
    ParameterRange("Vd", 50.0, 25.0, 100.0; distribution=:uniform),
    ParameterRange("ka", 1.0, 0.5, 2.0; distribution=:normal, std=0.2)
]

# Run Morris screening
result = morris_screening(pk_model, params, ["Cmax", "AUC"]; n_trajectories=20)

# Get influential parameters
ranking = result.rankings["Cmax"]
```

### TGA Validation
```julia
using DarwinPBPK

# Simulate TGA
simulated = TGAParameters(
    lag_time = 3.5,
    time_to_peak = 9.0,
    peak_thrombin = 320.0,
    etp = 1800.0,
    velocity_index = 100.0,
    width_50 = 8.0,
    start_tail = 20.0,
    tf_concentration = 5.0,
    patient_condition = "healthy"
)

# Compare to clinical data
result = compare_to_clinical(simulated, HEALTHY_TGA_5PM_TF)
println("All criteria met: ", result["acceptance_criteria"]["all_criteria_met"])
```

## References

1. Rodgers T, Rowland M. (2006) Physiologically based pharmacokinetic modelling 2: Predicting the tissue distribution of acids, very weak bases, neutrals and zwitterions. J Pharm Sci 95:1238-1257

2. Hemker HC et al. (2006) Calibrated Automated Thrombin Generation Measurement in Clotting Plasma. Pathophysiol Haemost Thromb 35:4-9

3. Hockin MF et al. (2002) A Model for the Stoichiometric Regulation of Blood Coagulation. J Biol Chem 277:18322-18333

4. Saltelli A et al. (2008) Global Sensitivity Analysis: The Primer. Wiley

5. Chen S, Doolen GD. (1998) Lattice Boltzmann Method for Fluid Flows. Annu Rev Fluid Mech 30:329-364

---

**Version**: 2.6.0  
**Date**: 2025-12-05  
**Author**: Darwin PBPK Platform Team
