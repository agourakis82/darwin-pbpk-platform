"""
Thrombin Generation Assay (TGA) Validation Module

Implements validation of coagulation models against clinical TGA data.
TGA is the gold standard for assessing global hemostatic function.

Key TGA Parameters (Hemker 2006):
- Lag time: Time to 2% of peak thrombin (minutes)
- Time to peak (ttPeak): Time to maximum thrombin (minutes)
- Peak thrombin: Maximum thrombin concentration (nM)
- ETP: Endogenous Thrombin Potential (area under curve, nM·min)
- Velocity Index: Max rate of thrombin generation (nM/min)

Clinical References:
- Hemker et al. (2006) - Calibrated Automated Thrombography (CAT)
- Dargaud et al. (2021) - TGA in anticoagulated patients
- Al Dieri et al. (2012) - Thrombin dynamics
- Ninivaggi et al. (2012) - Normal ranges and covariates
- Loeffen et al. (2018) - TGA in hemophilia

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

module TGAValidation

using LinearAlgebra
using Statistics

export TGAParameters, ClinicalTGADataset, ValidationMetrics
export extract_tga_parameters, compare_to_clinical
export calculate_prediction_error, validate_coagulation_model
export HEALTHY_TGA_1PM_TF, HEALTHY_TGA_5PM_TF
export HEMOPHILIA_A_TGA, HEMOPHILIA_B_TGA
export WARFARIN_INR2_TGA, WARFARIN_INR3_TGA
export DOAC_RIVAROXABAN_TGA, DOAC_APIXABAN_TGA
export FXI_DEFICIENCY_TGA
export calculate_goodness_of_fit

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
TGAParameters - Core endpoints from Thrombin Generation Assay

Based on Hemker (2006) CAT methodology.
"""
struct TGAParameters
    lag_time::Float64          # minutes (time to 2% of peak)
    time_to_peak::Float64      # minutes (ttPeak)
    peak_thrombin::Float64     # nM (maximum thrombin)
    etp::Float64               # nM·min (Endogenous Thrombin Potential)
    velocity_index::Float64    # nM/min (max rate dII/dt)

    # Optional additional parameters
    width_50::Float64          # minutes (width at 50% of peak)
    start_tail::Float64        # minutes (time to 5% of peak on descending limb)

    # Metadata
    tf_concentration::Float64  # pM (tissue factor trigger)
    patient_condition::String  # "healthy", "hemophilia_A", etc.
end

"""
ClinicalTGADataset - Clinical reference data from literature

Includes mean ± SD or median [IQR] from published studies.
"""
struct ClinicalTGADataset
    name::String
    n_subjects::Int

    # Mean values
    lag_time_mean::Float64
    time_to_peak_mean::Float64
    peak_thrombin_mean::Float64
    etp_mean::Float64
    velocity_index_mean::Float64

    # Standard deviations (or NaN if not reported)
    lag_time_sd::Float64
    time_to_peak_sd::Float64
    peak_thrombin_sd::Float64
    etp_sd::Float64
    velocity_index_sd::Float64

    # Experimental conditions
    tf_concentration::Float64  # pM
    phospholipids::Float64     # μM (typically 4 μM)

    # Clinical characteristics
    condition::String
    reference::String          # Citation
end

"""
ValidationMetrics - Goodness-of-fit metrics for model validation

Following FDA/EMA guidance for PBPK validation.
"""
struct ValidationMetrics
    # Fold error metrics
    afe::Float64               # Average Fold Error
    aafe::Float64              # Absolute Average Fold Error

    # Regression metrics
    rmse::Float64              # Root Mean Square Error
    mae::Float64               # Mean Absolute Error
    r_squared::Float64         # Correlation coefficient (R²)

    # Bias metrics
    mean_prediction_error::Float64    # Mean(Pred - Obs)
    mean_relative_error::Float64      # Mean((Pred - Obs)/Obs)

    # Within-fold criteria
    within_2fold::Float64      # Fraction of predictions within 2-fold
    within_3fold::Float64      # Fraction within 3-fold

    # Number of data points
    n_observations::Int
end

# ============================================================================
# CLINICAL REFERENCE DATASETS
# ============================================================================

"""
HEALTHY_TGA_1PM_TF - Normal healthy subjects with 1 pM TF

Reference: Hemker et al. (2006) Pathophysiol Haemost Thromb
           Ninivaggi et al. (2012) J Thromb Haemost

N = 123 healthy volunteers
TF = 1 pM, PL = 4 μM
"""
const HEALTHY_TGA_1PM_TF = ClinicalTGADataset(
    "Healthy adults (1pM TF)",
    123,
    # Mean values
    3.7,      # lag_time (min)
    9.5,      # time_to_peak (min)
    312.0,    # peak_thrombin (nM)
    1815.0,   # ETP (nM·min)
    95.0,     # velocity_index (nM/min)
    # Standard deviations
    0.6,      # lag_time SD
    1.2,      # time_to_peak SD
    58.0,     # peak_thrombin SD
    294.0,    # ETP SD
    18.0,     # velocity_index SD
    # Experimental
    1.0,      # TF (pM)
    4.0,      # PL (μM)
    # Clinical
    "healthy",
    "Hemker 2006; Ninivaggi 2012"
)

"""
HEALTHY_TGA_5PM_TF - Normal healthy subjects with 5 pM TF

Higher TF concentration → shorter lag, higher peak
Reference: Hemker et al. (2006)
"""
const HEALTHY_TGA_5PM_TF = ClinicalTGADataset(
    "Healthy adults (5pM TF)",
    123,
    # Mean values
    2.7,      # lag_time (min)
    6.3,      # time_to_peak (min)
    384.0,    # peak_thrombin (nM)
    2095.0,   # ETP (nM·min)
    151.0,    # velocity_index (nM/min)
    # Standard deviations
    0.4,      # lag_time SD
    0.8,      # time_to_peak SD
    72.0,     # peak_thrombin SD
    338.0,    # ETP SD
    28.0,     # velocity_index SD
    # Experimental
    5.0,      # TF (pM)
    4.0,      # PL (μM)
    # Clinical
    "healthy",
    "Hemker 2006"
)

"""
HEMOPHILIA_A_TGA - Hemophilia A patients (FVIII < 1%)

Severe hemophilia A with FVIII levels < 1 IU/dL
Reference: Loeffen et al. (2018) Haemophilia
           Dargaud et al. (2010) Blood

N = 45 severe Hemophilia A patients
TF = 1 pM, PL = 4 μM
"""
const HEMOPHILIA_A_TGA = ClinicalTGADataset(
    "Hemophilia A (FVIII < 1%)",
    45,
    # Mean values
    10.5,     # lag_time (min) - VERY prolonged
    22.0,     # time_to_peak (min) - VERY prolonged
    85.0,     # peak_thrombin (nM) - VERY low
    480.0,    # ETP (nM·min) - VERY low
    12.0,     # velocity_index (nM/min) - VERY low
    # Standard deviations
    3.2,      # lag_time SD
    5.5,      # time_to_peak SD
    28.0,     # peak_thrombin SD
    165.0,    # ETP SD
    4.5,      # velocity_index SD
    # Experimental
    1.0,      # TF (pM)
    4.0,      # PL (μM)
    # Clinical
    "hemophilia_A_severe",
    "Loeffen 2018; Dargaud 2010"
)

"""
HEMOPHILIA_B_TGA - Hemophilia B patients (FIX < 1%)

Severe hemophilia B with FIX levels < 1 IU/dL
Reference: Loeffen et al. (2018) Haemophilia
"""
const HEMOPHILIA_B_TGA = ClinicalTGADataset(
    "Hemophilia B (FIX < 1%)",
    28,
    # Mean values
    11.2,     # lag_time (min)
    23.5,     # time_to_peak (min)
    78.0,     # peak_thrombin (nM)
    445.0,    # ETP (nM·min)
    10.5,     # velocity_index (nM/min)
    # Standard deviations
    3.5,      # lag_time SD
    6.0,      # time_to_peak SD
    25.0,     # peak_thrombin SD
    155.0,    # ETP SD
    3.8,      # velocity_index SD
    # Experimental
    1.0,      # TF (pM)
    4.0,      # PL (μM)
    # Clinical
    "hemophilia_B_severe",
    "Loeffen 2018"
)

"""
WARFARIN_INR2_TGA - Patients on warfarin with INR 2.0-2.5

Therapeutic anticoagulation for VTE/AF
Reference: Dargaud et al. (2021) Blood Coag Fibrinol
           Al Dieri et al. (2012) J Thromb Haemost

N = 67 patients on warfarin
Target INR: 2.0-2.5
TF = 5 pM (higher sensitivity to anticoagulation)
"""
const WARFARIN_INR2_TGA = ClinicalTGADataset(
    "Warfarin (INR 2.0-2.5)",
    67,
    # Mean values
    4.8,      # lag_time (min) - prolonged
    10.2,     # time_to_peak (min)
    228.0,    # peak_thrombin (nM) - reduced
    1185.0,   # ETP (nM·min) - reduced (~60% of normal)
    58.0,     # velocity_index (nM/min) - reduced
    # Standard deviations
    1.2,      # lag_time SD
    2.1,      # time_to_peak SD
    52.0,     # peak_thrombin SD
    245.0,    # ETP SD
    14.0,     # velocity_index SD
    # Experimental
    5.0,      # TF (pM)
    4.0,      # PL (μM)
    # Clinical
    "warfarin_INR2",
    "Dargaud 2021; Al Dieri 2012"
)

"""
WARFARIN_INR3_TGA - Patients on warfarin with INR 2.5-3.5

High-intensity anticoagulation (mechanical valves)
Reference: Dargaud et al. (2021)
"""
const WARFARIN_INR3_TGA = ClinicalTGADataset(
    "Warfarin (INR 2.5-3.5)",
    42,
    # Mean values
    6.2,      # lag_time (min) - MORE prolonged
    12.5,     # time_to_peak (min)
    168.0,    # peak_thrombin (nM) - MORE reduced
    895.0,    # ETP (nM·min) - MORE reduced (~48% of normal)
    42.0,     # velocity_index (nM/min) - MORE reduced
    # Standard deviations
    1.8,      # lag_time SD
    2.8,      # time_to_peak SD
    45.0,     # peak_thrombin SD
    215.0,    # ETP SD
    11.0,     # velocity_index SD
    # Experimental
    5.0,      # TF (pM)
    4.0,      # PL (μM)
    # Clinical
    "warfarin_INR3",
    "Dargaud 2021"
)

"""
DOAC_RIVAROXABAN_TGA - Patients on rivaroxaban (therapeutic peak)

Direct Factor Xa inhibitor
Reference: Dargaud et al. (2021) Blood Coag Fibrinol
           Tripodi et al. (2015) J Thromb Haemost

N = 52 patients
Rivaroxaban peak conc (~200-300 ng/mL after 20 mg dose)
Anti-Xa activity: 150-250 ng/mL
TF = 5 pM
"""
const DOAC_RIVAROXABAN_TGA = ClinicalTGADataset(
    "Rivaroxaban (therapeutic peak)",
    52,
    # Mean values
    5.5,      # lag_time (min) - prolonged
    11.8,     # time_to_peak (min)
    195.0,    # peak_thrombin (nM) - reduced
    1025.0,   # ETP (nM·min) - reduced (~55% of normal)
    48.0,     # velocity_index (nM/min) - reduced
    # Standard deviations
    1.5,      # lag_time SD
    2.5,      # time_to_peak SD
    48.0,     # peak_thrombin SD
    235.0,    # ETP SD
    12.0,     # velocity_index SD
    # Experimental
    5.0,      # TF (pM)
    4.0,      # PL (μM)
    # Clinical
    "rivaroxaban_therapeutic",
    "Dargaud 2021; Tripodi 2015"
)

"""
DOAC_APIXABAN_TGA - Patients on apixaban (therapeutic peak)

Direct Factor Xa inhibitor (more potent than rivaroxaban)
Reference: Dargaud et al. (2021)
"""
const DOAC_APIXABAN_TGA = ClinicalTGADataset(
    "Apixaban (therapeutic peak)",
    38,
    # Mean values
    6.0,      # lag_time (min) - MORE prolonged
    13.2,     # time_to_peak (min)
    172.0,    # peak_thrombin (nM) - MORE reduced
    920.0,    # ETP (nM·min) - MORE reduced (~50% of normal)
    40.0,     # velocity_index (nM/min) - MORE reduced
    # Standard deviations
    1.8,      # lag_time SD
    3.0,      # time_to_peak SD
    42.0,     # peak_thrombin SD
    220.0,    # ETP SD
    10.0,     # velocity_index SD
    # Experimental
    5.0,      # TF (pM)
    4.0,      # PL (μM)
    # Clinical
    "apixaban_therapeutic",
    "Dargaud 2021"
)

"""
FXI_DEFICIENCY_TGA - Factor XI deficiency (hemophilia C)

Mild bleeding disorder with FXI levels 15-30%
Reference: Livnat et al. (2006) Thromb Res
           Bolton-Maggs et al. (2008) Br J Haematol

N = 35 patients with FXI 15-30%
TF = 1 pM
"""
const FXI_DEFICIENCY_TGA = ClinicalTGADataset(
    "FXI deficiency (15-30%)",
    35,
    # Mean values
    5.2,      # lag_time (min) - mildly prolonged
    12.5,     # time_to_peak (min)
    245.0,    # peak_thrombin (nM) - mildly reduced
    1385.0,   # ETP (nM·min) - mildly reduced
    68.0,     # velocity_index (nM/min) - mildly reduced
    # Standard deviations
    1.0,      # lag_time SD
    2.2,      # time_to_peak SD
    52.0,     # peak_thrombin SD
    285.0,    # ETP SD
    16.0,     # velocity_index SD
    # Experimental
    1.0,      # TF (pM)
    4.0,      # PL (μM)
    # Clinical
    "FXI_deficiency",
    "Livnat 2006; Bolton-Maggs 2008"
)

# ============================================================================
# TGA PARAMETER EXTRACTION
# ============================================================================

"""
extract_tga_parameters(thrombin_curve, time_points;
                       tf_concentration=1.0,
                       patient_condition="simulated")

Extract TGA parameters from a simulated thrombin generation curve.

# Arguments
- `thrombin_curve`: Vector of thrombin concentrations (nM)
- `time_points`: Vector of time points (minutes)
- `tf_concentration`: TF trigger used (pM)
- `patient_condition`: Patient/simulation description

# Returns
- `TGAParameters` struct with all key endpoints

# Algorithm
1. Lag time: Time when thrombin reaches 2% of eventual peak
2. Peak thrombin: Maximum value in curve
3. Time to peak: Time at maximum
4. ETP: Trapezoidal integration of area under curve
5. Velocity Index: Maximum derivative (dII/dt)
"""
function extract_tga_parameters(
    thrombin_curve::Vector{Float64},
    time_points::Vector{Float64};
    tf_concentration::Float64=1.0,
    patient_condition::String="simulated"
)::TGAParameters

    @assert length(thrombin_curve) == length(time_points) "Curve and time vectors must have same length"
    @assert length(thrombin_curve) > 0 "Empty curve"

    # 1. Find peak thrombin
    peak_thrombin = maximum(thrombin_curve)
    peak_idx = argmax(thrombin_curve)
    time_to_peak = time_points[peak_idx]

    # 2. Calculate lag time (time to 2% of peak)
    threshold_2pct = 0.02 * peak_thrombin
    lag_idx = findfirst(x -> x >= threshold_2pct, thrombin_curve)
    lag_time = lag_idx === nothing ? 0.0 : time_points[lag_idx]

    # 3. Calculate ETP (Endogenous Thrombin Potential) - trapezoidal integration
    etp = 0.0
    for i in 1:(length(thrombin_curve) - 1)
        dt = time_points[i+1] - time_points[i]
        avg_thrombin = (thrombin_curve[i] + thrombin_curve[i+1]) / 2.0
        etp += avg_thrombin * dt
    end

    # 4. Calculate Velocity Index (maximum derivative)
    velocity_index = 0.0
    for i in 1:(length(thrombin_curve) - 1)
        dt = time_points[i+1] - time_points[i]
        if dt > 0
            derivative = (thrombin_curve[i+1] - thrombin_curve[i]) / dt
            velocity_index = max(velocity_index, derivative)
        end
    end

    # 5. Calculate width at 50% of peak
    threshold_50pct = 0.5 * peak_thrombin
    indices_above_50 = findall(x -> x >= threshold_50pct, thrombin_curve)
    width_50 = if !isempty(indices_above_50)
        time_points[last(indices_above_50)] - time_points[first(indices_above_50)]
    else
        0.0
    end

    # 6. Calculate start of tail (time to 5% on descending limb)
    threshold_5pct = 0.05 * peak_thrombin
    descending_indices = findall(i -> i > peak_idx && thrombin_curve[i] <= threshold_5pct,
                                  1:length(thrombin_curve))
    start_tail = if !isempty(descending_indices)
        time_points[first(descending_indices)]
    else
        time_points[end]
    end

    return TGAParameters(
        lag_time,
        time_to_peak,
        peak_thrombin,
        etp,
        velocity_index,
        width_50,
        start_tail,
        tf_concentration,
        patient_condition
    )
end

# ============================================================================
# VALIDATION METRICS CALCULATION
# ============================================================================

"""
calculate_prediction_error(predicted, observed)

Calculate fold error between predicted and observed values.

Returns (fold_error, absolute_fold_error).
"""
function calculate_prediction_error(predicted::Float64, observed::Float64)::Tuple{Float64, Float64}
    @assert observed > 0 "Observed value must be positive"
    @assert predicted > 0 "Predicted value must be positive"

    fold_error = predicted / observed
    abs_fold_error = max(fold_error, 1.0 / fold_error)

    return (fold_error, abs_fold_error)
end

"""
calculate_goodness_of_fit(predicted, observed)

Calculate comprehensive goodness-of-fit metrics.

# Arguments
- `predicted`: Vector of predicted values
- `observed`: Vector of observed values

# Returns
- `ValidationMetrics` struct

# Acceptance Criteria (FDA/EMA PBPK Guidance)
- AAFE < 2.0 (within 2-fold on average)
- ≥ 80% of predictions within 2-fold
- R² > 0.7
"""
function calculate_goodness_of_fit(
    predicted::Vector{Float64},
    observed::Vector{Float64}
)::ValidationMetrics

    @assert length(predicted) == length(observed) "Vectors must have same length"
    @assert length(predicted) > 0 "Empty vectors"
    @assert all(observed .> 0) "All observed values must be positive"
    @assert all(predicted .> 0) "All predicted values must be positive"

    n = length(predicted)

    # 1. Fold error metrics
    fold_errors = predicted ./ observed
    abs_fold_errors = max.(fold_errors, 1.0 ./ fold_errors)

    afe = exp(mean(log.(fold_errors)))  # Geometric mean of fold errors
    aafe = exp(mean(log.(abs_fold_errors)))  # Geometric mean of absolute fold errors

    # 2. Regression metrics
    errors = predicted .- observed
    rmse = sqrt(mean(errors .^ 2))
    mae = mean(abs.(errors))

    # 3. R² (coefficient of determination)
    ss_res = sum(errors .^ 2)
    ss_tot = sum((observed .- mean(observed)) .^ 2)
    r_squared = 1.0 - (ss_res / ss_tot)

    # 4. Bias metrics
    mean_prediction_error = mean(errors)
    mean_relative_error = mean((predicted .- observed) ./ observed)

    # 5. Within-fold criteria
    within_2fold = count(abs_fold_errors .<= 2.0) / n
    within_3fold = count(abs_fold_errors .<= 3.0) / n

    return ValidationMetrics(
        afe,
        aafe,
        rmse,
        mae,
        r_squared,
        mean_prediction_error,
        mean_relative_error,
        within_2fold,
        within_3fold,
        n
    )
end

"""
compare_to_clinical(simulated, reference)

Compare simulated TGA parameters to clinical reference dataset.

# Arguments
- `simulated`: TGAParameters from simulation
- `reference`: ClinicalTGADataset from literature

# Returns
- Dictionary with parameter-wise comparisons and overall metrics
"""
function compare_to_clinical(
    simulated::TGAParameters,
    reference::ClinicalTGADataset
)::Dict{String, Any}

    # Extract values
    pred_values = [
        simulated.lag_time,
        simulated.time_to_peak,
        simulated.peak_thrombin,
        simulated.etp,
        simulated.velocity_index
    ]

    obs_values = [
        reference.lag_time_mean,
        reference.time_to_peak_mean,
        reference.peak_thrombin_mean,
        reference.etp_mean,
        reference.velocity_index_mean
    ]

    obs_sds = [
        reference.lag_time_sd,
        reference.time_to_peak_sd,
        reference.peak_thrombin_sd,
        reference.etp_sd,
        reference.velocity_index_sd
    ]

    param_names = ["lag_time", "time_to_peak", "peak_thrombin", "etp", "velocity_index"]

    # Calculate parameter-wise errors
    param_comparisons = Dict{String, Dict{String, Float64}}()

    for (i, name) in enumerate(param_names)
        pred = pred_values[i]
        obs = obs_values[i]
        sd = obs_sds[i]

        fold_error, abs_fold_error = calculate_prediction_error(pred, obs)

        # Z-score (standardized residual)
        z_score = isnan(sd) || sd == 0.0 ? NaN : (pred - obs) / sd

        param_comparisons[name] = Dict(
            "predicted" => pred,
            "observed" => obs,
            "observed_sd" => sd,
            "fold_error" => fold_error,
            "abs_fold_error" => abs_fold_error,
            "relative_error_pct" => 100.0 * (pred - obs) / obs,
            "z_score" => z_score,
            "within_1sd" => abs(z_score) <= 1.0,
            "within_2sd" => abs(z_score) <= 2.0
        )
    end

    # Overall goodness-of-fit
    metrics = calculate_goodness_of_fit(pred_values, obs_values)

    # Acceptance status (FDA/EMA criteria)
    acceptance_criteria = Dict(
        "AAFE_pass" => metrics.aafe < 2.0,
        "within_2fold_pass" => metrics.within_2fold >= 0.8,
        "R2_pass" => metrics.r_squared > 0.7,
        "all_criteria_met" => (metrics.aafe < 2.0) &&
                              (metrics.within_2fold >= 0.8) &&
                              (metrics.r_squared > 0.7)
    )

    return Dict(
        "reference_dataset" => reference.name,
        "reference_citation" => reference.reference,
        "n_subjects" => reference.n_subjects,
        "condition" => reference.condition,
        "tf_concentration_pM" => reference.tf_concentration,
        "parameter_comparisons" => param_comparisons,
        "overall_metrics" => metrics,
        "acceptance_criteria" => acceptance_criteria
    )
end

"""
validate_coagulation_model(model_simulations, clinical_datasets)

Comprehensive validation of coagulation model against multiple clinical datasets.

# Arguments
- `model_simulations`: Dict mapping dataset name to TGAParameters
- `clinical_datasets`: Vector of ClinicalTGADataset

# Returns
- Dictionary with validation results for each dataset and summary statistics

# Example
```julia
simulations = Dict(
    "healthy_1pM" => extract_tga_parameters(curve1, times1),
    "hemophilia_A" => extract_tga_parameters(curve2, times2)
)

datasets = [HEALTHY_TGA_1PM_TF, HEMOPHILIA_A_TGA]

results = validate_coagulation_model(simulations, datasets)
```
"""
function validate_coagulation_model(
    model_simulations::Dict{String, TGAParameters},
    clinical_datasets::Vector{ClinicalTGADataset}
)::Dict{String, Any}

    validation_results = Dict{String, Any}[]

    for dataset in clinical_datasets
        # Try to find matching simulation by name or condition
        matching_sim = nothing
        for (sim_name, sim_params) in model_simulations
            # Match by name or condition
            if occursin(lowercase(dataset.condition), lowercase(sim_name)) ||
               occursin(lowercase(sim_name), lowercase(dataset.condition))
                matching_sim = sim_params
                break
            end
        end

        if matching_sim === nothing
            @warn "No matching simulation found for dataset: $(dataset.name)"
            continue
        end

        # Compare
        comparison = compare_to_clinical(matching_sim, dataset)
        push!(validation_results, comparison)
    end

    if isempty(validation_results)
        error("No successful validations performed")
    end

    # Calculate summary statistics across all datasets
    all_aafes = [res["overall_metrics"].aafe for res in validation_results]
    all_r2s = [res["overall_metrics"].r_squared for res in validation_results]
    all_within_2fold = [res["overall_metrics"].within_2fold for res in validation_results]

    # Count how many datasets meet all criteria
    datasets_passed = count(res["acceptance_criteria"]["all_criteria_met"]
                           for res in validation_results)

    summary = Dict(
        "n_datasets_validated" => length(validation_results),
        "datasets_passed_all_criteria" => datasets_passed,
        "pass_rate" => datasets_passed / length(validation_results),
        "mean_AAFE" => mean(all_aafes),
        "median_AAFE" => median(all_aafes),
        "max_AAFE" => maximum(all_aafes),
        "mean_R2" => mean(all_r2s),
        "median_R2" => median(all_r2s),
        "min_R2" => minimum(all_r2s),
        "mean_within_2fold" => mean(all_within_2fold),
        "overall_model_acceptable" => (mean(all_aafes) < 2.0) &&
                                      (mean(all_r2s) > 0.7) &&
                                      (datasets_passed / length(validation_results) >= 0.8)
    )

    return Dict(
        "validation_results" => validation_results,
        "summary" => summary,
        "timestamp" => string(now())
    )
end

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

"""
get_all_clinical_datasets()

Returns vector of all available clinical TGA datasets.
"""
function get_all_clinical_datasets()::Vector{ClinicalTGADataset}
    return [
        HEALTHY_TGA_1PM_TF,
        HEALTHY_TGA_5PM_TF,
        HEMOPHILIA_A_TGA,
        HEMOPHILIA_B_TGA,
        WARFARIN_INR2_TGA,
        WARFARIN_INR3_TGA,
        DOAC_RIVAROXABAN_TGA,
        DOAC_APIXABAN_TGA,
        FXI_DEFICIENCY_TGA
    ]
end

"""
print_validation_summary(validation_results)

Pretty-print validation results to console.
"""
function print_validation_summary(validation_results::Dict{String, Any})
    println("\n" * "="^80)
    println("TGA VALIDATION SUMMARY")
    println("="^80)

    summary = validation_results["summary"]

    println("\nOverall Statistics:")
    println("  Datasets validated: $(summary["n_datasets_validated"])")
    println("  Datasets passing all criteria: $(summary["datasets_passed_all_criteria"])")
    println("  Pass rate: $(round(summary["pass_rate"] * 100, digits=1))%")
    println("\n  Mean AAFE: $(round(summary["mean_AAFE"], digits=2)) " *
            (summary["mean_AAFE"] < 2.0 ? "✓ PASS" : "✗ FAIL"))
    println("  Mean R²: $(round(summary["mean_R²"], digits=3)) " *
            (summary["mean_R²"] > 0.7 ? "✓ PASS" : "✗ FAIL"))
    println("  Mean within 2-fold: $(round(summary["mean_within_2fold"] * 100, digits=1))% " *
            (summary["mean_within_2fold"] >= 0.8 ? "✓ PASS" : "✗ FAIL"))

    println("\n  Overall Model Status: " *
            (summary["overall_model_acceptable"] ? "✓ ACCEPTABLE" : "✗ NOT ACCEPTABLE"))

    println("\n" * "-"^80)
    println("Individual Dataset Results:")
    println("-"^80)

    for result in validation_results["validation_results"]
        println("\n$(result["reference_dataset"])")
        println("  Reference: $(result["reference_citation"])")
        println("  N = $(result["n_subjects"]) subjects")

        metrics = result["overall_metrics"]
        criteria = result["acceptance_criteria"]

        println("  AAFE: $(round(metrics.aafe, digits=2)) " *
                (criteria["AAFE_pass"] ? "✓" : "✗"))
        println("  R²: $(round(metrics.r_squared, digits=3)) " *
                (criteria["R2_pass"] ? "✓" : "✗"))
        println("  Within 2-fold: $(round(metrics.within_2fold * 100, digits=1))% " *
                (criteria["within_2fold_pass"] ? "✓" : "✗"))
        println("  Overall: " * (criteria["all_criteria_met"] ? "✓ PASS" : "✗ FAIL"))
    end

    println("\n" * "="^80 * "\n")
end

end  # module TGAValidation
