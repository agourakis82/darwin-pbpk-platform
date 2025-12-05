# TGA Validation Module - User Guide

## Overview

The **Thrombin Generation Assay (TGA) Validation Module** (`tga_validation.jl`) provides comprehensive tools for validating coagulation cascade models against clinical TGA data. TGA is the gold standard for assessing global hemostatic function and is widely used in clinical research and drug development.

**Location**: `/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration/src/DarwinPBPK/compartments/tga_validation.jl`

## Key Features

### 1. TGA Parameter Extraction
- **Lag time**: Time to reach 2% of peak thrombin (minutes)
- **Time to peak (ttPeak)**: Time to maximum thrombin concentration (minutes)
- **Peak thrombin**: Maximum thrombin concentration (nM)
- **ETP**: Endogenous Thrombin Potential - area under curve (nM·min)
- **Velocity Index**: Maximum rate of thrombin generation (nM/min)

### 2. Clinical Reference Datasets
Nine curated datasets from peer-reviewed literature:

| Dataset | Condition | N | TF | Reference |
|---------|-----------|---|----|--------------|
| `HEALTHY_TGA_1PM_TF` | Normal healthy | 123 | 1 pM | Hemker 2006; Ninivaggi 2012 |
| `HEALTHY_TGA_5PM_TF` | Normal healthy | 123 | 5 pM | Hemker 2006 |
| `HEMOPHILIA_A_TGA` | Hemophilia A (FVIII<1%) | 45 | 1 pM | Loeffen 2018; Dargaud 2010 |
| `HEMOPHILIA_B_TGA` | Hemophilia B (FIX<1%) | 28 | 1 pM | Loeffen 2018 |
| `WARFARIN_INR2_TGA` | Warfarin (INR 2.0-2.5) | 67 | 5 pM | Dargaud 2021; Al Dieri 2012 |
| `WARFARIN_INR3_TGA` | Warfarin (INR 2.5-3.5) | 42 | 5 pM | Dargaud 2021 |
| `DOAC_RIVAROXABAN_TGA` | Rivaroxaban (therapeutic) | 52 | 5 pM | Dargaud 2021; Tripodi 2015 |
| `DOAC_APIXABAN_TGA` | Apixaban (therapeutic) | 38 | 5 pM | Dargaud 2021 |
| `FXI_DEFICIENCY_TGA` | FXI deficiency (15-30%) | 35 | 1 pM | Livnat 2006; Bolton-Maggs 2008 |

### 3. Validation Metrics

Following FDA/EMA guidance for PBPK model validation:

- **AFE**: Average Fold Error (geometric mean of predicted/observed)
- **AAFE**: Absolute Average Fold Error (target: < 2.0)
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination (target: > 0.7)
- **Within-fold criteria**: Fraction within 2-fold (target: ≥ 80%)

### 4. Acceptance Criteria

A model passes validation if:
1. AAFE < 2.0 (within 2-fold on average)
2. R² > 0.7 (good correlation)
3. ≥ 80% of predictions within 2-fold
4. ≥ 80% of datasets pass all criteria

## Quick Start

### Basic Usage

```julia
using DarwinPBPK

# Step 1: Extract TGA parameters from simulated thrombin curve
time_points = collect(0.0:0.1:30.0)  # 0-30 minutes
thrombin_curve = [...]  # Your simulated thrombin concentrations (nM)

tga_params = extract_tga_parameters(
    thrombin_curve, 
    time_points,
    tf_concentration=1.0,
    patient_condition="simulated_healthy"
)

println("Lag time: $(tga_params.lag_time) min")
println("Peak thrombin: $(tga_params.peak_thrombin) nM")
println("ETP: $(tga_params.etp) nM·min")

# Step 2: Compare to clinical reference
comparison = compare_to_clinical(tga_params, HEALTHY_TGA_1PM_TF)

# Step 3: Check acceptance criteria
metrics = comparison["overall_metrics"]
println("AAFE: $(metrics.aafe)")
println("R²: $(metrics.r_squared)")
println("Pass: $(comparison["acceptance_criteria"]["all_criteria_met"])")
```

### Multi-Dataset Validation

```julia
# Simulate multiple scenarios
simulations = Dict(
    "healthy" => extract_tga_parameters(healthy_curve, times),
    "hemophilia_A" => extract_tga_parameters(hemophilia_curve, times),
    "warfarin_INR2" => extract_tga_parameters(warfarin_curve, times)
)

# Select clinical datasets
datasets = [
    HEALTHY_TGA_1PM_TF,
    HEMOPHILIA_A_TGA,
    WARFARIN_INR2_TGA
]

# Run comprehensive validation
results = validate_coagulation_model(simulations, datasets)

# Print summary
print_validation_summary(results)

# Check overall acceptance
if results["summary"]["overall_model_acceptable"]
    println("✓ Model acceptable for clinical use")
else
    println("✗ Model needs improvement")
end
```

## API Reference

### Data Structures

#### `TGAParameters`
```julia
struct TGAParameters
    lag_time::Float64          # minutes
    time_to_peak::Float64      # minutes
    peak_thrombin::Float64     # nM
    etp::Float64               # nM·min
    velocity_index::Float64    # nM/min
    width_50::Float64          # minutes (width at 50% peak)
    start_tail::Float64        # minutes (tail start)
    tf_concentration::Float64  # pM
    patient_condition::String
end
```

#### `ClinicalTGADataset`
```julia
struct ClinicalTGADataset
    name::String
    n_subjects::Int
    lag_time_mean::Float64
    time_to_peak_mean::Float64
    peak_thrombin_mean::Float64
    etp_mean::Float64
    velocity_index_mean::Float64
    # ... (standard deviations)
    tf_concentration::Float64
    phospholipids::Float64
    condition::String
    reference::String
end
```

#### `ValidationMetrics`
```julia
struct ValidationMetrics
    afe::Float64               # Average Fold Error
    aafe::Float64              # Absolute Average Fold Error
    rmse::Float64              # Root Mean Square Error
    mae::Float64               # Mean Absolute Error
    r_squared::Float64         # R²
    mean_prediction_error::Float64
    mean_relative_error::Float64
    within_2fold::Float64      # Fraction within 2-fold
    within_3fold::Float64      # Fraction within 3-fold
    n_observations::Int
end
```

### Core Functions

#### `extract_tga_parameters`
```julia
extract_tga_parameters(
    thrombin_curve::Vector{Float64},
    time_points::Vector{Float64};
    tf_concentration::Float64=1.0,
    patient_condition::String="simulated"
) -> TGAParameters
```

**Purpose**: Extract all TGA endpoints from a simulated thrombin generation curve.

**Algorithm**:
1. Peak thrombin: `maximum(curve)`
2. Lag time: Time when curve reaches 2% of peak
3. ETP: Trapezoidal integration of area under curve
4. Velocity Index: Maximum derivative (dII/dt)

**Example**:
```julia
time = 0.0:0.1:30.0
thrombin = generate_thrombin_curve(...)
params = extract_tga_parameters(thrombin, time, tf_concentration=5.0)
```

#### `compare_to_clinical`
```julia
compare_to_clinical(
    simulated::TGAParameters,
    reference::ClinicalTGADataset
) -> Dict{String, Any}
```

**Purpose**: Compare simulated TGA parameters to clinical reference dataset.

**Returns**: Dictionary containing:
- `parameter_comparisons`: Parameter-wise fold errors and z-scores
- `overall_metrics`: ValidationMetrics struct
- `acceptance_criteria`: Pass/fail for each criterion

**Example**:
```julia
comparison = compare_to_clinical(my_simulation, HEALTHY_TGA_1PM_TF)
if comparison["acceptance_criteria"]["all_criteria_met"]
    println("✓ Validation passed")
end
```

#### `calculate_goodness_of_fit`
```julia
calculate_goodness_of_fit(
    predicted::Vector{Float64},
    observed::Vector{Float64}
) -> ValidationMetrics
```

**Purpose**: Calculate comprehensive goodness-of-fit metrics.

**Example**:
```julia
pred = [100.0, 200.0, 300.0]
obs = [105.0, 190.0, 310.0]
metrics = calculate_goodness_of_fit(pred, obs)
println("AAFE: $(metrics.aafe)")  # Should be ~1.05
```

#### `validate_coagulation_model`
```julia
validate_coagulation_model(
    model_simulations::Dict{String, TGAParameters},
    clinical_datasets::Vector{ClinicalTGADataset}
) -> Dict{String, Any}
```

**Purpose**: Comprehensive validation across multiple clinical datasets.

**Returns**: Dictionary containing:
- `validation_results`: Vector of individual dataset comparisons
- `summary`: Overall summary statistics
- `timestamp`: Validation timestamp

**Example**:
```julia
sims = Dict("healthy" => sim1, "hemophilia" => sim2)
datasets = [HEALTHY_TGA_1PM_TF, HEMOPHILIA_A_TGA]
results = validate_coagulation_model(sims, datasets)
print_validation_summary(results)
```

### Helper Functions

#### `calculate_prediction_error`
```julia
calculate_prediction_error(
    predicted::Float64,
    observed::Float64
) -> Tuple{Float64, Float64}
```
Returns `(fold_error, absolute_fold_error)`.

#### `get_all_clinical_datasets`
```julia
get_all_clinical_datasets() -> Vector{ClinicalTGADataset}
```
Returns all 9 available clinical datasets.

#### `print_validation_summary`
```julia
print_validation_summary(validation_results::Dict{String, Any})
```
Pretty-print validation results to console with pass/fail indicators.

## Clinical Interpretation

### TGA Parameter Ranges

| Condition | Lag (min) | ttPeak (min) | Peak (nM) | ETP (nM·min) | VI (nM/min) |
|-----------|-----------|--------------|-----------|--------------|-------------|
| Normal (1pM TF) | 3.1-4.3 | 8.3-10.7 | 254-370 | 1521-2109 | 77-113 |
| Normal (5pM TF) | 2.3-3.1 | 5.5-7.1 | 312-456 | 1757-2433 | 123-179 |
| Hemophilia A | 7.3-13.7 | 16.5-27.5 | 57-113 | 315-645 | 7.5-16.5 |
| Warfarin INR 2 | 3.6-6.0 | 8.1-12.3 | 176-280 | 940-1430 | 44-72 |
| Rivaroxaban | 4.0-7.0 | 9.3-14.3 | 147-243 | 790-1260 | 36-60 |

### Physiological Relationships

1. **Higher TF → Faster, stronger response**
   - 5pM TF: Shorter lag, higher peak vs 1pM TF

2. **Hemophilia → Severely impaired**
   - Peak thrombin < 30% of normal
   - ETP < 30% of normal
   - Prolonged lag time (>10 min)

3. **Anticoagulation → Dose-dependent suppression**
   - Warfarin INR 3 > INR 2 (more suppression)
   - Apixaban > Rivaroxaban (more potent FXa inhibitor)

4. **ETP is most sensitive marker**
   - Best correlates with bleeding risk
   - Most stable parameter (low CV%)

## Integration with Coagulation Module

The TGA validation module is designed to work seamlessly with the coagulation cascade model (`coagulation.jl`):

```julia
using DarwinPBPK

# Create coagulation system
coag_system = create_coagulation_system()

# Simulate thrombin generation
time_points, thrombin_curve = thrombin_generation_assay(
    coag_system,
    tf_concentration=1.0,  # pM
    duration=30.0          # minutes
)

# Extract TGA parameters
tga_params = extract_tga_parameters(thrombin_curve, time_points)

# Validate against clinical data
comparison = compare_to_clinical(tga_params, HEALTHY_TGA_1PM_TF)
```

## Validation Workflow

### Recommended Workflow for Model Development

```julia
# 1. Define scenarios to validate
scenarios = [
    ("healthy", healthy_system, HEALTHY_TGA_1PM_TF),
    ("hemophilia_A", hemophilia_system, HEMOPHILIA_A_TGA),
    ("warfarin", warfarin_system, WARFARIN_INR2_TGA),
    ("rivaroxaban", rivaroxaban_system, DOAC_RIVAROXABAN_TGA)
]

# 2. Run simulations
simulations = Dict{String, TGAParameters}()
for (name, system, _) in scenarios
    time, thrombin = thrombin_generation_assay(system)
    simulations[name] = extract_tga_parameters(thrombin, time)
end

# 3. Validate
datasets = [s[3] for s in scenarios]
results = validate_coagulation_model(simulations, datasets)

# 4. Analyze results
print_validation_summary(results)

# 5. Iterate if needed
if !results["summary"]["overall_model_acceptable"]
    # Identify problematic parameters
    for result in results["validation_results"]
        if !result["acceptance_criteria"]["all_criteria_met"]
            println("Failed: $(result["reference_dataset"])")
            # Analyze parameter_comparisons to identify issues
        end
    end
end
```

## Troubleshooting

### Common Issues

#### 1. AAFE > 2.0 (Systematic bias)
- **Cause**: Model systematically over- or under-predicts thrombin
- **Solution**: Recalibrate kinetic rate constants (kcat, Km values)

#### 2. Low R² (Poor correlation)
- **Cause**: Model captures wrong dynamics
- **Solution**: Review feedback mechanisms (thrombin-mediated activation)

#### 3. Low within-2fold percentage
- **Cause**: High variability across parameters
- **Solution**: Check individual parameter fold errors to identify outliers

#### 4. Hemophilia validation fails
- **Cause**: Intrinsic pathway (FVIIIa, FIXa) incorrectly modeled
- **Solution**: Review tenase complex formation kinetics

#### 5. Anticoagulant validation fails
- **Cause**: Drug Ki or inhibition mechanism incorrect
- **Solution**: Verify DOAC/warfarin parameters against literature

## References

### Key Literature

1. **Hemker HC et al. (2006)** - *Pathophysiol Haemost Thromb*
   - "Calibrated Automated Thrombography (CAT)"
   - Established CAT methodology and normal ranges

2. **Ninivaggi M et al. (2012)** - *J Thromb Haemost*
   - "Reference values for thrombin generation"
   - N=123 healthy subjects, comprehensive covariates analysis

3. **Loeffen R et al. (2018)** - *Haemophilia*
   - "Clinical relevance of thrombin generation in hemophilia"
   - Hemophilia A/B TGA reference values

4. **Dargaud Y et al. (2021)** - *Blood Coag Fibrinol*
   - "Thrombin generation in anticoagulated patients"
   - Warfarin and DOAC reference values

5. **Al Dieri R et al. (2012)** - *J Thromb Haemost*
   - "The thrombogram in rare bleeding disorders"
   - Factor deficiencies and anticoagulation

6. **Tripodi A et al. (2015)** - *J Thromb Haemost*
   - "Thrombin generation assay and direct oral anticoagulants"
   - Rivaroxaban, apixaban, dabigatran effects

## Testing

Run the test suite:
```bash
cd julia-migration
julia --project=. test/test_tga_validation.jl
```

Run the comprehensive example:
```bash
julia --project=. examples/tga_validation_example.jl
```

## File Locations

- **Module**: `/julia-migration/src/DarwinPBPK/compartments/tga_validation.jl`
- **Tests**: `/julia-migration/test/test_tga_validation.jl`
- **Example**: `/julia-migration/examples/tga_validation_example.jl`
- **Documentation**: `/julia-migration/docs/TGA_VALIDATION_GUIDE.md` (this file)

## License

Part of the Darwin PBPK Platform.
Author: Dr. Demetrios Agourakis
Date: 2025-12-05
