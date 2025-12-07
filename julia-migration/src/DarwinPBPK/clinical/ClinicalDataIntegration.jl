# ===========================================================================
# CLINICAL TRIAL DATA INTEGRATION MODULE
# ===========================================================================
# Standardized import, processing, and analysis of clinical PK data for
# PBPK model validation and parameter estimation.
#
# Supports:
# - CDISC SDTM/ADaM format import
# - Noncompartmental Analysis (NCA)
# - Individual and population PK data
# - Bioequivalence assessment
# - Literature data extraction
#
# Author: Dr. Demetrios Agourakis
# Date: December 2025
# Version: 1.0.0
# ===========================================================================

module ClinicalDataIntegration

using Statistics
using DataFrames
using CSV
using Dates
using Distributions

export ClinicalStudy, PKObservation, Subject, DosingRecord
export NCAResult, BioequivalenceResult, ValidationMetrics
export load_clinical_study, load_sdtm_pc, load_adam_adpc
export compute_nca, compute_nca_population
export validate_model_against_clinical, bioequivalence_assessment
export literature_pk_entry, aggregate_literature_data
export ClinicalPKDatabase, add_study!, query_studies

# ===========================================================================
# Data Structures
# ===========================================================================

"""
Individual subject in a clinical study.
"""
struct Subject
    id::String
    age::Float64              # years
    weight::Float64           # kg
    height::Float64           # cm
    sex::Symbol               # :male, :female
    race::String
    bmi::Float64              # kg/m²
    creatinine_clearance::Float64  # mL/min (renal function)
    child_pugh::Symbol        # :A, :B, :C, :normal (hepatic function)
    genotype::Dict{String, String}  # CYP genotypes
    comorbidities::Vector{String}
end

"""
Construct Subject with automatic BMI calculation.
"""
function Subject(;
    id::String,
    age::Float64,
    weight::Float64,
    height::Float64,
    sex::Symbol,
    race::String = "Unknown",
    creatinine_clearance::Float64 = 120.0,
    child_pugh::Symbol = :normal,
    genotype::Dict{String, String} = Dict{String, String}(),
    comorbidities::Vector{String} = String[]
)
    bmi = weight / (height / 100)^2
    return Subject(id, age, weight, height, sex, race, bmi,
                   creatinine_clearance, child_pugh, genotype, comorbidities)
end

"""
Dosing record for a subject.
"""
struct DosingRecord
    time::Float64             # hours from first dose
    amount::Float64           # mg
    route::Symbol             # :oral, :iv, :im, :sc
    formulation::String       # e.g., "tablet", "solution", "capsule"
    fed_state::Symbol         # :fasted, :fed, :unknown
end

"""
Single PK observation.
"""
struct PKObservation
    subject_id::String
    time::Float64             # hours post-dose
    concentration::Float64    # ng/mL or μg/mL
    concentration_unit::String
    sample_type::Symbol       # :plasma, :serum, :blood, :urine
    analyte::String           # parent drug or metabolite
    blq::Bool                 # below limit of quantification
    lloq::Float64             # lower limit of quantification
end

"""
Complete clinical study data.
"""
struct ClinicalStudy
    study_id::String
    drug_name::String
    phase::String             # "Phase I", "Phase II", etc.
    design::String            # "parallel", "crossover", "single-dose"
    subjects::Vector{Subject}
    dosing::Dict{String, Vector{DosingRecord}}  # subject_id => dosing records
    observations::Vector{PKObservation}
    metadata::Dict{String, Any}
end

"""
Noncompartmental analysis results.
"""
struct NCAResult
    subject_id::String
    cmax::Float64             # Maximum concentration
    tmax::Float64             # Time of Cmax
    auc_0_t::Float64          # AUC from 0 to last measurable
    auc_0_inf::Float64        # AUC extrapolated to infinity
    auc_extrap_pct::Float64   # % AUC extrapolated
    half_life::Float64        # Terminal half-life
    lambda_z::Float64         # Terminal rate constant
    cl_f::Float64             # Apparent clearance (CL/F)
    vz_f::Float64             # Apparent volume (Vz/F)
    mrt::Float64              # Mean residence time
    c0::Float64               # Extrapolated C0 (IV only)
    dose::Float64
    route::Symbol
end

"""
Bioequivalence assessment results.
"""
struct BioequivalenceResult
    test_product::String
    reference_product::String
    parameter::String         # "Cmax", "AUC0t", "AUC0inf"
    geometric_mean_ratio::Float64
    ci_90_lower::Float64
    ci_90_upper::Float64
    is_bioequivalent::Bool    # 80-125% criterion
    n_subjects::Int
    cv_percent::Float64       # Intra-subject CV
end

"""
Model validation metrics against clinical data.
"""
struct ValidationMetrics
    drug_name::String
    n_subjects::Int
    n_observations::Int

    # Prediction accuracy
    gmfe::Float64             # Geometric mean fold error
    aafe::Float64             # Absolute average fold error
    afe::Float64              # Average fold error (bias)
    rmse::Float64             # Root mean square error

    # Categorical accuracy
    within_2fold::Float64     # % within 2-fold
    within_1_5fold::Float64   # % within 1.5-fold
    within_1_25fold::Float64  # % within 1.25-fold (BE criterion)

    # By parameter
    cmax_fe::Float64          # Cmax fold error
    auc_fe::Float64           # AUC fold error
    tmax_fe::Float64          # Tmax fold error

    # Correlation
    r_squared::Float64
    concordance_correlation::Float64
end

# ===========================================================================
# Data Loading Functions
# ===========================================================================

"""
    load_clinical_study(filepath; format=:auto) -> ClinicalStudy

Load clinical PK data from file.

Supported formats:
- `:csv` - Generic CSV with standard columns
- `:sdtm` - CDISC SDTM PC domain
- `:adam` - CDISC ADaM ADPC dataset
- `:nonmem` - NONMEM-style dataset
"""
function load_clinical_study(
    filepath::String;
    format::Symbol = :auto,
    drug_name::String = "Unknown",
    study_id::String = "STUDY001"
)::ClinicalStudy
    # Auto-detect format
    if format == :auto
        if occursin("sdtm", lowercase(filepath)) || occursin("pc.csv", lowercase(filepath))
            format = :sdtm
        elseif occursin("adam", lowercase(filepath)) || occursin("adpc", lowercase(filepath))
            format = :adam
        else
            format = :csv
        end
    end

    if format == :sdtm
        return load_sdtm_pc(filepath; drug_name, study_id)
    elseif format == :adam
        return load_adam_adpc(filepath; drug_name, study_id)
    else
        return load_generic_csv(filepath; drug_name, study_id)
    end
end

"""
Load CDISC SDTM PC (Pharmacokinetic Concentrations) domain.
"""
function load_sdtm_pc(
    filepath::String;
    drug_name::String = "Unknown",
    study_id::String = "STUDY001"
)::ClinicalStudy
    df = CSV.read(filepath, DataFrame)

    # Standard SDTM PC columns
    required_cols = [:USUBJID, :PCTPTNUM, :PCSTRESN, :PCSTRESU]

    observations = PKObservation[]
    subject_ids = Set{String}()

    for row in eachrow(df)
        subj_id = string(get(row, :USUBJID, ""))
        push!(subject_ids, subj_id)

        time = get(row, :PCTPTNUM, 0.0)
        conc = get(row, :PCSTRESN, NaN)
        unit = string(get(row, :PCSTRESU, "ng/mL"))
        analyte = string(get(row, :PCTESTCD, drug_name))
        blq = get(row, :PCSTAT, "") == "BLQ"
        lloq = get(row, :PCLLOQ, 0.0)

        if !isnan(conc) || blq
            push!(observations, PKObservation(
                subj_id, time, isnan(conc) ? 0.0 : conc, unit,
                :plasma, analyte, blq, lloq
            ))
        end
    end

    # Create minimal subjects (demographics would come from DM domain)
    subjects = [Subject(id=id, age=40.0, weight=70.0, height=170.0, sex=:unknown)
                for id in subject_ids]

    return ClinicalStudy(
        study_id, drug_name, "Unknown", "Unknown",
        subjects, Dict{String, Vector{DosingRecord}}(),
        observations, Dict{String, Any}("format" => "SDTM")
    )
end

"""
Load CDISC ADaM ADPC (Analysis Dataset for PK Concentrations).
"""
function load_adam_adpc(
    filepath::String;
    drug_name::String = "Unknown",
    study_id::String = "STUDY001"
)::ClinicalStudy
    df = CSV.read(filepath, DataFrame)

    observations = PKObservation[]
    subjects_dict = Dict{String, Subject}()
    dosing_dict = Dict{String, Vector{DosingRecord}}()

    for row in eachrow(df)
        subj_id = string(get(row, :USUBJID, get(row, :SUBJID, "")))

        # Extract subject demographics if available
        if !haskey(subjects_dict, subj_id)
            age = get(row, :AGE, 40.0)
            weight = get(row, :WEIGHTBL, get(row, :WEIGHT, 70.0))
            height = get(row, :HEIGHTBL, get(row, :HEIGHT, 170.0))
            sex = get(row, :SEX, "U") == "M" ? :male :
                  get(row, :SEX, "U") == "F" ? :female : :unknown

            subjects_dict[subj_id] = Subject(
                id=subj_id, age=Float64(age), weight=Float64(weight),
                height=Float64(height), sex=sex
            )
        end

        # Extract observation
        time = get(row, :AFRLT, get(row, :NFRLT, get(row, :TIME, 0.0)))
        conc = get(row, :AVAL, get(row, :DV, NaN))
        unit = string(get(row, :AVALU, "ng/mL"))
        blq = get(row, :ABLFL, "") == "Y" || get(row, :BLQ, 0) == 1

        if !ismissing(conc) && !isnan(conc)
            push!(observations, PKObservation(
                subj_id, Float64(time), Float64(conc), unit,
                :plasma, drug_name, blq, 0.0
            ))
        end

        # Extract dosing if available
        dose = get(row, :DOSEA, get(row, :AMT, nothing))
        if !isnothing(dose) && dose > 0
            if !haskey(dosing_dict, subj_id)
                dosing_dict[subj_id] = DosingRecord[]
            end
            route = get(row, :ROUTE, "ORAL") == "ORAL" ? :oral : :iv
            push!(dosing_dict[subj_id], DosingRecord(
                Float64(time), Float64(dose), route, "tablet", :unknown
            ))
        end
    end

    return ClinicalStudy(
        study_id, drug_name, "Unknown", "Unknown",
        collect(values(subjects_dict)), dosing_dict,
        observations, Dict{String, Any}("format" => "ADaM")
    )
end

"""
Load generic CSV format.
"""
function load_generic_csv(
    filepath::String;
    drug_name::String = "Unknown",
    study_id::String = "STUDY001"
)::ClinicalStudy
    df = CSV.read(filepath, DataFrame)

    # Try to identify columns
    id_col = findfirst(c -> lowercase(string(c)) in ["id", "subject", "subjid", "usubjid"], names(df))
    time_col = findfirst(c -> lowercase(string(c)) in ["time", "tad", "hour", "hours"], names(df))
    conc_col = findfirst(c -> lowercase(string(c)) in ["conc", "concentration", "dv", "cp"], names(df))

    if isnothing(id_col) || isnothing(time_col) || isnothing(conc_col)
        error("Could not identify required columns (ID, TIME, CONC)")
    end

    observations = PKObservation[]
    subject_ids = Set{String}()

    for row in eachrow(df)
        subj_id = string(row[id_col])
        push!(subject_ids, subj_id)

        time = Float64(row[time_col])
        conc = Float64(row[conc_col])

        push!(observations, PKObservation(
            subj_id, time, conc, "ng/mL", :plasma, drug_name, false, 0.0
        ))
    end

    subjects = [Subject(id=id, age=40.0, weight=70.0, height=170.0, sex=:unknown)
                for id in subject_ids]

    return ClinicalStudy(
        study_id, drug_name, "Unknown", "Unknown",
        subjects, Dict{String, Vector{DosingRecord}}(),
        observations, Dict{String, Any}("format" => "CSV")
    )
end

# ===========================================================================
# Noncompartmental Analysis (NCA)
# ===========================================================================

"""
    compute_nca(observations, dose; route=:oral) -> NCAResult

Perform noncompartmental analysis on concentration-time data.

# Arguments
- `observations`: Vector of PKObservation for single subject
- `dose`: Dose amount in mg
- `route`: Administration route

# Returns
- `NCAResult`: NCA parameters
"""
function compute_nca(
    observations::Vector{PKObservation},
    dose::Float64;
    route::Symbol = :oral,
    lloq::Float64 = 0.0
)::NCAResult
    # Filter and sort observations
    obs = filter(o -> !o.blq && o.concentration > lloq, observations)
    sort!(obs, by=o -> o.time)

    if isempty(obs)
        return NCAResult(
            observations[1].subject_id,
            NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN,
            dose, route
        )
    end

    times = [o.time for o in obs]
    concs = [o.concentration for o in obs]
    subject_id = obs[1].subject_id

    # Cmax and Tmax
    cmax_idx = argmax(concs)
    cmax = concs[cmax_idx]
    tmax = times[cmax_idx]

    # AUC by linear-log trapezoidal rule
    auc_0_t = compute_auc_linear_log(times, concs)

    # Terminal phase (use last 3+ points after Cmax)
    term_start = max(cmax_idx + 1, length(concs) - 3)
    if term_start < length(concs) && all(concs[term_start:end] .> 0)
        term_times = times[term_start:end]
        term_concs = log.(concs[term_start:end])

        # Linear regression for lambda_z
        n = length(term_times)
        sum_x = sum(term_times)
        sum_y = sum(term_concs)
        sum_xy = sum(term_times .* term_concs)
        sum_x2 = sum(term_times.^2)

        lambda_z = -(n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
        lambda_z = max(lambda_z, 1e-6)  # Ensure positive

        half_life = log(2) / lambda_z

        # AUC extrapolation
        clast = concs[end]
        auc_extrap = clast / lambda_z
        auc_0_inf = auc_0_t + auc_extrap
        auc_extrap_pct = 100 * auc_extrap / auc_0_inf
    else
        lambda_z = NaN
        half_life = NaN
        auc_0_inf = auc_0_t
        auc_extrap_pct = 0.0
    end

    # Clearance and Volume
    cl_f = dose / auc_0_inf  # CL/F for oral, CL for IV
    vz_f = isnan(lambda_z) ? NaN : cl_f / lambda_z

    # MRT
    aumc_0_t = compute_aumc(times, concs)
    mrt = aumc_0_t / auc_0_t

    # C0 (IV only - back extrapolation)
    c0 = route == :iv && !isnan(lambda_z) ? concs[1] * exp(lambda_z * times[1]) : NaN

    return NCAResult(
        subject_id, cmax, tmax, auc_0_t, auc_0_inf, auc_extrap_pct,
        half_life, lambda_z, cl_f, vz_f, mrt, c0, dose, route
    )
end

"""
Compute AUC using linear-log trapezoidal method.
"""
function compute_auc_linear_log(times::Vector{Float64}, concs::Vector{Float64})::Float64
    auc = 0.0
    for i in 2:length(times)
        dt = times[i] - times[i-1]
        c1, c2 = concs[i-1], concs[i]

        if c1 > 0 && c2 > 0 && c2 < c1  # Log trapezoidal for descending
            auc += dt * (c1 - c2) / log(c1 / c2)
        else  # Linear trapezoidal
            auc += dt * (c1 + c2) / 2
        end
    end
    return auc
end

"""
Compute AUMC (area under moment curve).
"""
function compute_aumc(times::Vector{Float64}, concs::Vector{Float64})::Float64
    aumc = 0.0
    for i in 2:length(times)
        dt = times[i] - times[i-1]
        tc1 = times[i-1] * concs[i-1]
        tc2 = times[i] * concs[i]
        aumc += dt * (tc1 + tc2) / 2
    end
    return aumc
end

"""
    compute_nca_population(study; route=:oral) -> Vector{NCAResult}

Compute NCA for all subjects in a study.
"""
function compute_nca_population(
    study::ClinicalStudy;
    route::Symbol = :oral,
    default_dose::Float64 = 100.0
)::Vector{NCAResult}
    results = NCAResult[]

    for subject in study.subjects
        # Get observations for this subject
        subj_obs = filter(o -> o.subject_id == subject.id, study.observations)

        if isempty(subj_obs)
            continue
        end

        # Get dose
        dose = default_dose
        if haskey(study.dosing, subject.id) && !isempty(study.dosing[subject.id])
            dose = study.dosing[subject.id][1].amount
            route = study.dosing[subject.id][1].route
        end

        nca = compute_nca(subj_obs, dose; route)
        push!(results, nca)
    end

    return results
end

# ===========================================================================
# Model Validation
# ===========================================================================

"""
    validate_model_against_clinical(predicted, observed) -> ValidationMetrics

Compare model predictions against clinical observations.

# Arguments
- `predicted`: Dict with "cmax", "auc", "tmax" predictions
- `observed`: Vector of NCAResult from clinical data

# Returns
- `ValidationMetrics`: Comprehensive validation statistics
"""
function validate_model_against_clinical(
    predicted::Dict{String, Vector{Float64}},
    observed::Vector{NCAResult};
    drug_name::String = "Unknown"
)::ValidationMetrics
    n_subjects = length(observed)

    # Extract observed values
    obs_cmax = [r.cmax for r in observed if !isnan(r.cmax)]
    obs_auc = [r.auc_0_inf for r in observed if !isnan(r.auc_0_inf)]
    obs_tmax = [r.tmax for r in observed if !isnan(r.tmax)]

    # Get predicted values (ensure same length)
    pred_cmax = predicted["cmax"][1:length(obs_cmax)]
    pred_auc = predicted["auc"][1:length(obs_auc)]
    pred_tmax = get(predicted, "tmax", fill(NaN, length(obs_tmax)))[1:length(obs_tmax)]

    # Compute fold errors
    fe_cmax = [max(p/o, o/p) for (p, o) in zip(pred_cmax, obs_cmax) if o > 0 && p > 0]
    fe_auc = [max(p/o, o/p) for (p, o) in zip(pred_auc, obs_auc) if o > 0 && p > 0]
    fe_tmax = [max(p/o, o/p) for (p, o) in zip(pred_tmax, obs_tmax) if o > 0 && p > 0 && !isnan(p)]

    all_fe = vcat(fe_cmax, fe_auc)

    # GMFE (geometric mean fold error)
    gmfe = isempty(all_fe) ? NaN : exp(mean(log.(all_fe)))

    # AAFE (absolute average fold error)
    aafe = isempty(all_fe) ? NaN : mean(all_fe)

    # AFE (average fold error - for bias detection)
    ratios = vcat(
        [p/o for (p, o) in zip(pred_cmax, obs_cmax) if o > 0],
        [p/o for (p, o) in zip(pred_auc, obs_auc) if o > 0]
    )
    afe = isempty(ratios) ? NaN : exp(mean(log.(ratios)))

    # RMSE
    all_pred = vcat(pred_cmax, pred_auc)
    all_obs = vcat(obs_cmax, obs_auc)
    rmse = sqrt(mean((all_pred .- all_obs).^2))

    # Categorical accuracy
    within_2fold = isempty(all_fe) ? NaN : 100 * sum(all_fe .<= 2.0) / length(all_fe)
    within_1_5fold = isempty(all_fe) ? NaN : 100 * sum(all_fe .<= 1.5) / length(all_fe)
    within_1_25fold = isempty(all_fe) ? NaN : 100 * sum(all_fe .<= 1.25) / length(all_fe)

    # Individual parameter FE
    cmax_fe = isempty(fe_cmax) ? NaN : exp(mean(log.(fe_cmax)))
    auc_fe = isempty(fe_auc) ? NaN : exp(mean(log.(fe_auc)))
    tmax_fe = isempty(fe_tmax) ? NaN : exp(mean(log.(fe_tmax)))

    # Correlation
    r_squared = cor(all_pred, all_obs)^2

    # Concordance correlation coefficient
    mean_pred, mean_obs = mean(all_pred), mean(all_obs)
    var_pred, var_obs = var(all_pred), var(all_obs)
    cov_po = cov(all_pred, all_obs)
    ccc = 2 * cov_po / (var_pred + var_obs + (mean_pred - mean_obs)^2)

    return ValidationMetrics(
        drug_name, n_subjects, length(all_obs),
        gmfe, aafe, afe, rmse,
        within_2fold, within_1_5fold, within_1_25fold,
        cmax_fe, auc_fe, tmax_fe,
        r_squared, ccc
    )
end

# ===========================================================================
# Bioequivalence Assessment
# ===========================================================================

"""
    bioequivalence_assessment(test_nca, ref_nca; parameters=[:cmax, :auc]) -> Vector{BioequivalenceResult}

Perform bioequivalence assessment using average bioequivalence approach.
"""
function bioequivalence_assessment(
    test_nca::Vector{NCAResult},
    ref_nca::Vector{NCAResult};
    test_name::String = "Test",
    ref_name::String = "Reference",
    parameters::Vector{Symbol} = [:cmax, :auc_0_inf]
)::Vector{BioequivalenceResult}
    results = BioequivalenceResult[]

    for param in parameters
        # Extract values
        test_vals = if param == :cmax
            [r.cmax for r in test_nca if !isnan(r.cmax)]
        elseif param == :auc_0_inf || param == :auc
            [r.auc_0_inf for r in test_nca if !isnan(r.auc_0_inf)]
        elseif param == :auc_0_t
            [r.auc_0_t for r in test_nca if !isnan(r.auc_0_t)]
        else
            Float64[]
        end

        ref_vals = if param == :cmax
            [r.cmax for r in ref_nca if !isnan(r.cmax)]
        elseif param == :auc_0_inf || param == :auc
            [r.auc_0_inf for r in ref_nca if !isnan(r.auc_0_inf)]
        elseif param == :auc_0_t
            [r.auc_0_t for r in ref_nca if !isnan(r.auc_0_t)]
        else
            Float64[]
        end

        n = min(length(test_vals), length(ref_vals))
        if n < 2
            continue
        end

        test_vals = test_vals[1:n]
        ref_vals = ref_vals[1:n]

        # Log-transform
        log_test = log.(test_vals)
        log_ref = log.(ref_vals)

        # Difference
        diff = log_test .- log_ref
        mean_diff = mean(diff)
        se_diff = std(diff) / sqrt(n)

        # 90% CI (two one-sided tests)
        t_crit = quantile(TDist(n - 1), 0.95)
        ci_lower = exp(mean_diff - t_crit * se_diff)
        ci_upper = exp(mean_diff + t_crit * se_diff)

        gmr = exp(mean_diff)

        # Intra-subject CV
        cv_pct = 100 * sqrt(exp(var(diff)) - 1)

        # BE criterion: 80-125%
        is_be = ci_lower >= 0.80 && ci_upper <= 1.25

        push!(results, BioequivalenceResult(
            test_name, ref_name, string(param),
            gmr, ci_lower, ci_upper, is_be, n, cv_pct
        ))
    end

    return results
end

# ===========================================================================
# Literature Data Management
# ===========================================================================

"""
Literature PK data entry.
"""
struct LiteraturePKEntry
    drug_name::String
    pmid::String              # PubMed ID
    doi::String
    parameter::String         # "Cmax", "AUC", "CL", etc.
    value::Float64
    unit::String
    cv_percent::Float64
    n_subjects::Int
    population::String        # "healthy", "patients", "renally impaired"
    dose::Float64
    route::Symbol
    formulation::String
end

"""
Create a literature PK entry.
"""
function literature_pk_entry(;
    drug_name::String,
    parameter::String,
    value::Float64,
    unit::String,
    pmid::String = "",
    doi::String = "",
    cv_percent::Float64 = NaN,
    n_subjects::Int = 0,
    population::String = "healthy",
    dose::Float64 = NaN,
    route::Symbol = :oral,
    formulation::String = "tablet"
)::LiteraturePKEntry
    return LiteraturePKEntry(
        drug_name, pmid, doi, parameter, value, unit,
        cv_percent, n_subjects, population, dose, route, formulation
    )
end

"""
    aggregate_literature_data(entries) -> Dict

Aggregate literature data by drug and parameter.
"""
function aggregate_literature_data(
    entries::Vector{LiteraturePKEntry}
)::Dict{String, Dict{String, NamedTuple}}
    result = Dict{String, Dict{String, NamedTuple}}()

    for drug in unique([e.drug_name for e in entries])
        drug_entries = filter(e -> e.drug_name == drug, entries)
        result[drug] = Dict{String, NamedTuple}()

        for param in unique([e.parameter for e in drug_entries])
            param_entries = filter(e -> e.parameter == param, drug_entries)
            values = [e.value for e in param_entries]

            result[drug][param] = (
                mean = mean(values),
                std = std(values),
                min = minimum(values),
                max = maximum(values),
                n_studies = length(param_entries),
                unit = param_entries[1].unit
            )
        end
    end

    return result
end

# ===========================================================================
# Clinical PK Database
# ===========================================================================

"""
In-memory database for clinical PK studies.
"""
mutable struct ClinicalPKDatabase
    studies::Dict{String, ClinicalStudy}
    literature::Vector{LiteraturePKEntry}
    nca_results::Dict{String, Vector{NCAResult}}
end

"""
Create empty database.
"""
ClinicalPKDatabase() = ClinicalPKDatabase(
    Dict{String, ClinicalStudy}(),
    LiteraturePKEntry[],
    Dict{String, Vector{NCAResult}}()
)

"""
Add study to database.
"""
function add_study!(db::ClinicalPKDatabase, study::ClinicalStudy)
    db.studies[study.study_id] = study

    # Auto-compute NCA
    nca = compute_nca_population(study)
    if !isempty(nca)
        db.nca_results[study.study_id] = nca
    end

    return db
end

"""
Query studies by drug name.
"""
function query_studies(
    db::ClinicalPKDatabase,
    drug_name::String
)::Vector{ClinicalStudy}
    return [s for s in values(db.studies) if lowercase(s.drug_name) == lowercase(drug_name)]
end

end # module
