# ===========================================================================
# REGULATORY SUBMISSION REPORT GENERATOR
# ===========================================================================
# Generates FDA/EMA-compliant PBPK model documentation for regulatory
# submissions following ICH M12, FDA PBPK Guidance, and EMA guidelines.
#
# Report Types:
# - FDA PBPK Model Qualification Report
# - EMA PBPK Model Report
# - ICH M12 DDI Assessment Report
# - Model Validation Summary
#
# References:
# - FDA Guidance: Physiologically Based Pharmacokinetic Analyses (2018)
# - EMA Guideline: PBPK Modelling (2018)
# - ICH M12: Drug Interaction Studies (2024)
#
# Author: Dr. Sounio Agourakis
# Date: December 2025
# Version: 1.0.0
# ===========================================================================

module RegulatoryReports

using Dates
using Printf
using Statistics

export RegulatoryReport, ModelDocumentation, ValidationSummary
export DDIAssessment, SensitivitySummary, PopulationCovariates
export generate_fda_report, generate_ema_report, generate_ich_m12_report
export generate_validation_summary, generate_ddi_assessment
export export_report_markdown, export_report_latex, export_report_json
export FDA_ACCEPTANCE_CRITERIA, EMA_ACCEPTANCE_CRITERIA

# ===========================================================================
# Constants - Regulatory Acceptance Criteria
# ===========================================================================

"""
FDA PBPK Model Acceptance Criteria (2018 Guidance).
"""
const FDA_ACCEPTANCE_CRITERIA = Dict{String, Any}(
    "aafe_threshold" => 2.0,           # AAFE should be < 2
    "gmfe_threshold" => 2.0,           # GMFE should be < 2
    "within_2fold_threshold" => 0.80,  # ≥80% within 2-fold
    "auc_ratio_range" => (0.5, 2.0),   # DDI AUC ratio prediction
    "cmax_ratio_range" => (0.5, 2.0),  # DDI Cmax ratio prediction
    "ddi_classification" => Dict(
        "strong_inhibitor" => 5.0,     # AUC ratio ≥ 5
        "moderate_inhibitor" => 2.0,   # AUC ratio 2-5
        "weak_inhibitor" => 1.25,      # AUC ratio 1.25-2
        "no_inhibition" => 1.25        # AUC ratio < 1.25
    )
)

"""
EMA PBPK Model Acceptance Criteria (2018 Guideline).
"""
const EMA_ACCEPTANCE_CRITERIA = Dict{String, Any}(
    "gmfe_threshold" => 2.0,
    "aafe_threshold" => 2.0,           # Same as FDA
    "within_2fold_threshold" => 0.80,
    "prediction_bias" => (-0.3, 0.3),  # Log scale
    "sensitivity_threshold" => 0.1,    # 10% change in output
    "population_variability" => 0.5    # 50% CV acceptable
)

# ===========================================================================
# Data Structures
# ===========================================================================

"""
Model documentation metadata.
"""
struct ModelDocumentation
    model_name::String
    version::String
    author::String
    date::Date
    software::String
    software_version::String
    drug_name::String
    indication::String
    sponsor::String
    regulatory_agency::Symbol  # :fda, :ema, :pmda
end

"""
Validation summary for PBPK model.
"""
struct ValidationSummary
    n_studies::Int
    n_subjects::Int
    n_observations::Int

    # Primary metrics
    gmfe::Float64
    aafe::Float64
    afe::Float64

    # Categorical accuracy
    within_2fold::Float64
    within_1_5fold::Float64

    # By parameter
    cmax_gmfe::Float64
    auc_gmfe::Float64
    tmax_gmfe::Float64
    cl_gmfe::Float64

    # Pass/fail
    meets_criteria::Bool
    criteria_used::String
end

"""
DDI assessment for regulatory submission.
"""
struct DDIAssessment
    perpetrator::String
    victim::String
    mechanism::String           # "CYP3A4 inhibition", etc.

    # Predicted DDI
    predicted_auc_ratio::Float64
    predicted_cmax_ratio::Float64

    # Observed DDI (if available)
    observed_auc_ratio::Float64
    observed_cmax_ratio::Float64

    # Prediction accuracy
    auc_fe::Float64             # Fold error
    cmax_fe::Float64

    # Classification
    predicted_class::String     # "strong", "moderate", "weak", "no effect"
    observed_class::String
    classification_match::Bool

    # Dose adjustment recommendation
    dose_recommendation::String
end

"""
Sensitivity analysis summary.
"""
struct SensitivitySummary
    parameter::String
    baseline_value::Float64
    range_tested::Tuple{Float64, Float64}
    impact_on_auc::Float64      # % change in AUC
    impact_on_cmax::Float64     # % change in Cmax
    is_sensitive::Bool          # > 10% impact
    tornado_rank::Int           # Rank in tornado plot
end

"""
Population covariate analysis.
"""
struct PopulationCovariates
    covariate::String           # "age", "weight", "renal_function", etc.
    effect_on_cl::Float64       # % change per unit
    effect_on_vd::Float64
    clinical_relevance::String  # "Yes - dose adjustment", "No", "Monitor"
    subgroup_recommendations::Vector{String}
end

"""
Complete regulatory report.
"""
struct RegulatoryReport
    documentation::ModelDocumentation
    executive_summary::String
    model_description::String
    validation_summary::ValidationSummary
    ddi_assessments::Vector{DDIAssessment}
    sensitivity_analysis::Vector{SensitivitySummary}
    population_analysis::Vector{PopulationCovariates}
    conclusions::String
    appendices::Dict{String, Any}
end

# ===========================================================================
# Report Generation Functions
# ===========================================================================

"""
    generate_fda_report(model_info, validation_data, ddi_data) -> RegulatoryReport

Generate FDA-compliant PBPK model report following 2018 guidance.
"""
function generate_fda_report(
    model_info::ModelDocumentation,
    validation_data::ValidationSummary,
    ddi_data::Vector{DDIAssessment} = DDIAssessment[];
    sensitivity_data::Vector{SensitivitySummary} = SensitivitySummary[],
    population_data::Vector{PopulationCovariates} = PopulationCovariates[]
)::RegulatoryReport

    # Generate executive summary
    exec_summary = generate_executive_summary(model_info, validation_data, :fda)

    # Generate model description
    model_desc = generate_model_description(model_info)

    # Generate conclusions
    conclusions = generate_conclusions(validation_data, ddi_data, :fda)

    # Appendices
    appendices = Dict{String, Any}(
        "acceptance_criteria" => FDA_ACCEPTANCE_CRITERIA,
        "generation_date" => now(),
        "software" => "Darwin PBPK Platform v2.11.0"
    )

    return RegulatoryReport(
        model_info,
        exec_summary,
        model_desc,
        validation_data,
        ddi_data,
        sensitivity_data,
        population_data,
        conclusions,
        appendices
    )
end

"""
    generate_ema_report(model_info, validation_data) -> RegulatoryReport

Generate EMA-compliant PBPK model report following 2018 guideline.
"""
function generate_ema_report(
    model_info::ModelDocumentation,
    validation_data::ValidationSummary,
    ddi_data::Vector{DDIAssessment} = DDIAssessment[];
    sensitivity_data::Vector{SensitivitySummary} = SensitivitySummary[],
    population_data::Vector{PopulationCovariates} = PopulationCovariates[]
)::RegulatoryReport

    exec_summary = generate_executive_summary(model_info, validation_data, :ema)
    model_desc = generate_model_description(model_info)
    conclusions = generate_conclusions(validation_data, ddi_data, :ema)

    appendices = Dict{String, Any}(
        "acceptance_criteria" => EMA_ACCEPTANCE_CRITERIA,
        "generation_date" => now(),
        "software" => "Darwin PBPK Platform v2.11.0"
    )

    return RegulatoryReport(
        model_info,
        exec_summary,
        model_desc,
        validation_data,
        ddi_data,
        sensitivity_data,
        population_data,
        conclusions,
        appendices
    )
end

"""
    generate_ich_m12_report(ddi_assessments) -> String

Generate ICH M12-compliant DDI assessment report.
"""
function generate_ich_m12_report(
    ddi_assessments::Vector{DDIAssessment};
    drug_name::String = "Test Drug"
)::String
    buf = IOBuffer()

    println(buf, "# ICH M12 Drug Interaction Assessment Report")
    println(buf, "## Drug: $drug_name")
    println(buf, "## Date: $(today())")
    println(buf)

    println(buf, "### 1. In Vitro Assessment Summary")
    println(buf)

    println(buf, "### 2. Clinical DDI Study Results")
    println(buf)

    println(buf, "| Perpetrator | Victim | Mechanism | AUC Ratio (Pred) | AUC Ratio (Obs) | Classification |")
    println(buf, "|-------------|--------|-----------|------------------|-----------------|----------------|")

    for ddi in ddi_assessments
        println(buf, "| $(ddi.perpetrator) | $(ddi.victim) | $(ddi.mechanism) | " *
                     "$(@sprintf("%.2f", ddi.predicted_auc_ratio)) | " *
                     "$(@sprintf("%.2f", ddi.observed_auc_ratio)) | $(ddi.predicted_class) |")
    end
    println(buf)

    println(buf, "### 3. PBPK Model Predictions")
    println(buf)

    println(buf, "### 4. Label Recommendations")
    println(buf)
    for ddi in ddi_assessments
        if !isempty(ddi.dose_recommendation)
            println(buf, "- **$(ddi.perpetrator) + $(ddi.victim)**: $(ddi.dose_recommendation)")
        end
    end

    return String(take!(buf))
end

# ===========================================================================
# Helper Functions
# ===========================================================================

"""
Generate executive summary section.
"""
function generate_executive_summary(
    model_info::ModelDocumentation,
    validation::ValidationSummary,
    agency::Symbol
)::String
    criteria = agency == :fda ? FDA_ACCEPTANCE_CRITERIA : EMA_ACCEPTANCE_CRITERIA

    status = validation.meets_criteria ? "MEETS" : "DOES NOT MEET"

    return """
## Executive Summary

This report documents the development and validation of a physiologically based
pharmacokinetic (PBPK) model for **$(model_info.drug_name)** intended for
**$(model_info.indication)**.

### Key Findings

- **Model Version**: $(model_info.version)
- **Software**: $(model_info.software) $(model_info.software_version)
- **Validation Status**: Model $status $(uppercase(string(agency))) acceptance criteria

### Validation Metrics

| Metric | Value | Criterion | Status |
|--------|-------|-----------|--------|
| GMFE | $(@sprintf("%.2f", validation.gmfe)) | < $(criteria["gmfe_threshold"]) | $(validation.gmfe < criteria["gmfe_threshold"] ? "PASS" : "FAIL") |
| AAFE | $(@sprintf("%.2f", validation.aafe)) | < $(criteria["aafe_threshold"]) | $(validation.aafe < criteria["aafe_threshold"] ? "PASS" : "FAIL") |
| Within 2-fold | $(@sprintf("%.0f%%", validation.within_2fold * 100)) | ≥ $(Int(criteria["within_2fold_threshold"] * 100))% | $(validation.within_2fold >= criteria["within_2fold_threshold"] ? "PASS" : "FAIL") |

### Data Summary

- Number of clinical studies: $(validation.n_studies)
- Number of subjects: $(validation.n_subjects)
- Number of observations: $(validation.n_observations)
"""
end

"""
Generate model description section.
"""
function generate_model_description(model_info::ModelDocumentation)::String
    return """
## Model Description

### 2.1 Model Structure

The PBPK model for $(model_info.drug_name) was developed using $(model_info.software)
version $(model_info.software_version). The model incorporates:

- Physiological parameters from population databases
- Drug-specific parameters from in vitro and clinical studies
- Absorption, distribution, metabolism, and elimination (ADME) processes

### 2.2 Model Parameters

[Parameter table to be populated from model data]

### 2.3 Model Assumptions

1. Well-stirred model for hepatic clearance
2. Perfusion-limited distribution to tissues
3. Linear pharmacokinetics within therapeutic dose range

### 2.4 Model Verification

The model was verified against observed clinical data as described in Section 3.
"""
end

"""
Generate conclusions section.
"""
function generate_conclusions(
    validation::ValidationSummary,
    ddi_data::Vector{DDIAssessment},
    agency::Symbol
)::String
    criteria_name = agency == :fda ? "FDA 2018 Guidance" : "EMA 2018 Guideline"

    buf = IOBuffer()

    println(buf, "## Conclusions")
    println(buf)

    if validation.meets_criteria
        println(buf, "The PBPK model has been successfully validated against clinical data and ")
        println(buf, "**meets the acceptance criteria** specified in the $criteria_name.")
    else
        println(buf, "The PBPK model validation shows some deviations from the acceptance criteria ")
        println(buf, "specified in the $criteria_name. Additional validation may be required.")
    end
    println(buf)

    println(buf, "### Model Applications")
    println(buf)
    println(buf, "Based on the validation results, this model is suitable for:")
    println(buf)
    if validation.gmfe < 1.5
        println(buf, "- DDI predictions with high confidence")
        println(buf, "- Dose selection for special populations")
        println(buf, "- Waiver of clinical DDI studies (where applicable)")
    else
        println(buf, "- Qualitative DDI assessment")
        println(buf, "- Hypothesis generation for clinical studies")
    end
    println(buf)

    if !isempty(ddi_data)
        println(buf, "### DDI Conclusions")
        println(buf)
        for ddi in ddi_data
            println(buf, "- **$(ddi.perpetrator) + $(ddi.victim)**: $(ddi.predicted_class) interaction")
            if !isempty(ddi.dose_recommendation)
                println(buf, "  - Recommendation: $(ddi.dose_recommendation)")
            end
        end
    end

    return String(take!(buf))
end

# ===========================================================================
# Validation Summary Generation
# ===========================================================================

"""
    generate_validation_summary(predictions, observations; criteria=:fda) -> ValidationSummary

Generate validation summary from predicted vs observed data.
"""
function generate_validation_summary(
    predicted::Dict{String, Vector{Float64}},
    observed::Dict{String, Vector{Float64}};
    n_studies::Int = 1,
    criteria::Symbol = :fda
)::ValidationSummary

    # Calculate metrics for each parameter
    metrics = Dict{String, Float64}()

    all_fe = Float64[]

    for param in ["cmax", "auc", "tmax", "cl"]
        if haskey(predicted, param) && haskey(observed, param)
            pred = predicted[param]
            obs = observed[param]

            # Filter valid pairs
            valid_idx = findall(i -> pred[i] > 0 && obs[i] > 0, 1:min(length(pred), length(obs)))

            if !isempty(valid_idx)
                fe = [max(pred[i]/obs[i], obs[i]/pred[i]) for i in valid_idx]
                metrics["$(param)_gmfe"] = exp(mean(log.(fe)))
                append!(all_fe, fe)
            else
                metrics["$(param)_gmfe"] = NaN
            end
        else
            metrics["$(param)_gmfe"] = NaN
        end
    end

    # Overall metrics
    gmfe = isempty(all_fe) ? NaN : exp(mean(log.(all_fe)))
    aafe = isempty(all_fe) ? NaN : mean(all_fe)

    # AFE (for bias)
    all_ratios = Float64[]
    for param in ["cmax", "auc"]
        if haskey(predicted, param) && haskey(observed, param)
            for (p, o) in zip(predicted[param], observed[param])
                if p > 0 && o > 0
                    push!(all_ratios, p / o)
                end
            end
        end
    end
    afe = isempty(all_ratios) ? NaN : exp(mean(log.(all_ratios)))

    # Categorical accuracy
    within_2fold = isempty(all_fe) ? NaN : sum(all_fe .<= 2.0) / length(all_fe)
    within_1_5fold = isempty(all_fe) ? NaN : sum(all_fe .<= 1.5) / length(all_fe)

    # Check criteria
    threshold = criteria == :fda ? FDA_ACCEPTANCE_CRITERIA : EMA_ACCEPTANCE_CRITERIA
    meets = gmfe < threshold["gmfe_threshold"] && within_2fold >= threshold["within_2fold_threshold"]

    n_subjects = haskey(observed, "cmax") ? length(observed["cmax"]) : 0
    n_obs = sum(length(v) for v in values(observed))

    return ValidationSummary(
        n_studies, n_subjects, n_obs,
        gmfe, aafe, afe,
        within_2fold, within_1_5fold,
        get(metrics, "cmax_gmfe", NaN),
        get(metrics, "auc_gmfe", NaN),
        get(metrics, "tmax_gmfe", NaN),
        get(metrics, "cl_gmfe", NaN),
        meets,
        string(criteria)
    )
end

# ===========================================================================
# DDI Assessment
# ===========================================================================

"""
    generate_ddi_assessment(perpetrator, victim, pred_auc, obs_auc; mechanism="Unknown") -> DDIAssessment

Generate DDI assessment with classification.
"""
function generate_ddi_assessment(
    perpetrator::String,
    victim::String,
    predicted_auc_ratio::Float64,
    observed_auc_ratio::Float64;
    predicted_cmax_ratio::Float64 = NaN,
    observed_cmax_ratio::Float64 = NaN,
    mechanism::String = "Unknown"
)::DDIAssessment

    # Calculate fold errors
    auc_fe = max(predicted_auc_ratio / observed_auc_ratio,
                 observed_auc_ratio / predicted_auc_ratio)

    cmax_fe = if !isnan(predicted_cmax_ratio) && !isnan(observed_cmax_ratio)
        max(predicted_cmax_ratio / observed_cmax_ratio,
            observed_cmax_ratio / predicted_cmax_ratio)
    else
        NaN
    end

    # Classify DDI
    pred_class = classify_ddi(predicted_auc_ratio)
    obs_class = classify_ddi(observed_auc_ratio)

    classification_match = pred_class == obs_class

    # Dose recommendation
    dose_rec = generate_dose_recommendation(pred_class, mechanism)

    return DDIAssessment(
        perpetrator, victim, mechanism,
        predicted_auc_ratio, predicted_cmax_ratio,
        observed_auc_ratio, observed_cmax_ratio,
        auc_fe, cmax_fe,
        pred_class, obs_class, classification_match,
        dose_rec
    )
end

"""
Classify DDI based on AUC ratio.
"""
function classify_ddi(auc_ratio::Float64)::String
    if auc_ratio >= 5.0
        return "strong inhibitor"
    elseif auc_ratio >= 2.0
        return "moderate inhibitor"
    elseif auc_ratio >= 1.25
        return "weak inhibitor"
    elseif auc_ratio <= 0.5
        return "strong inducer"
    elseif auc_ratio <= 0.8
        return "moderate inducer"
    else
        return "no clinically significant interaction"
    end
end

"""
Generate dose recommendation based on DDI classification.
"""
function generate_dose_recommendation(ddi_class::String, mechanism::String)::String
    if occursin("strong inhibitor", ddi_class)
        return "Avoid concomitant use or reduce dose by 75%"
    elseif occursin("moderate inhibitor", ddi_class)
        return "Reduce dose by 50% during concomitant use"
    elseif occursin("weak inhibitor", ddi_class)
        return "No dose adjustment required; monitor for adverse effects"
    elseif occursin("strong inducer", ddi_class)
        return "Avoid concomitant use or increase dose"
    elseif occursin("moderate inducer", ddi_class)
        return "Consider dose increase; monitor for efficacy"
    else
        return "No dose adjustment required"
    end
end

# ===========================================================================
# Export Functions
# ===========================================================================

"""
    export_report_markdown(report) -> String

Export regulatory report to Markdown format.
"""
function export_report_markdown(report::RegulatoryReport)::String
    buf = IOBuffer()

    # Title
    println(buf, "# PBPK Model Report: $(report.documentation.drug_name)")
    println(buf)
    println(buf, "**Sponsor**: $(report.documentation.sponsor)")
    println(buf, "**Date**: $(report.documentation.date)")
    println(buf, "**Model Version**: $(report.documentation.version)")
    println(buf)

    # Executive Summary
    println(buf, report.executive_summary)
    println(buf)

    # Model Description
    println(buf, report.model_description)
    println(buf)

    # Validation Section
    println(buf, "## Model Validation")
    println(buf)
    println(buf, "### Validation Metrics Summary")
    println(buf)
    println(buf, "| Metric | Value |")
    println(buf, "|--------|-------|")
    println(buf, "| GMFE | $(@sprintf("%.3f", report.validation_summary.gmfe)) |")
    println(buf, "| AAFE | $(@sprintf("%.3f", report.validation_summary.aafe)) |")
    println(buf, "| AFE (Bias) | $(@sprintf("%.3f", report.validation_summary.afe)) |")
    println(buf, "| Within 2-fold | $(@sprintf("%.1f%%", report.validation_summary.within_2fold * 100)) |")
    println(buf, "| Within 1.5-fold | $(@sprintf("%.1f%%", report.validation_summary.within_1_5fold * 100)) |")
    println(buf)

    # DDI Section
    if !isempty(report.ddi_assessments)
        println(buf, "## Drug-Drug Interaction Assessment")
        println(buf)
        println(buf, "| Perpetrator | Victim | Pred AUC Ratio | Obs AUC Ratio | FE | Classification |")
        println(buf, "|-------------|--------|----------------|---------------|-----|----------------|")
        for ddi in report.ddi_assessments
            println(buf, "| $(ddi.perpetrator) | $(ddi.victim) | " *
                        "$(@sprintf("%.2f", ddi.predicted_auc_ratio)) | " *
                        "$(@sprintf("%.2f", ddi.observed_auc_ratio)) | " *
                        "$(@sprintf("%.2f", ddi.auc_fe)) | $(ddi.predicted_class) |")
        end
        println(buf)
    end

    # Sensitivity Analysis
    if !isempty(report.sensitivity_analysis)
        println(buf, "## Sensitivity Analysis")
        println(buf)
        println(buf, "| Parameter | Impact on AUC | Impact on Cmax | Sensitive |")
        println(buf, "|-----------|---------------|----------------|-----------|")
        for sa in report.sensitivity_analysis
            println(buf, "| $(sa.parameter) | $(@sprintf("%.1f%%", sa.impact_on_auc)) | " *
                        "$(@sprintf("%.1f%%", sa.impact_on_cmax)) | $(sa.is_sensitive ? "Yes" : "No") |")
        end
        println(buf)
    end

    # Conclusions
    println(buf, report.conclusions)

    return String(take!(buf))
end

"""
    export_report_json(report) -> String

Export regulatory report to JSON format.
"""
function export_report_json(report::RegulatoryReport)::String
    # Convert to dictionary structure
    dict = Dict{String, Any}(
        "metadata" => Dict(
            "drug_name" => report.documentation.drug_name,
            "model_version" => report.documentation.version,
            "software" => report.documentation.software,
            "date" => string(report.documentation.date),
            "sponsor" => report.documentation.sponsor,
            "agency" => string(report.documentation.regulatory_agency)
        ),
        "validation" => Dict(
            "gmfe" => report.validation_summary.gmfe,
            "aafe" => report.validation_summary.aafe,
            "afe" => report.validation_summary.afe,
            "within_2fold" => report.validation_summary.within_2fold,
            "within_1_5fold" => report.validation_summary.within_1_5fold,
            "meets_criteria" => report.validation_summary.meets_criteria,
            "n_studies" => report.validation_summary.n_studies,
            "n_subjects" => report.validation_summary.n_subjects
        ),
        "ddi_assessments" => [
            Dict(
                "perpetrator" => ddi.perpetrator,
                "victim" => ddi.victim,
                "mechanism" => ddi.mechanism,
                "predicted_auc_ratio" => ddi.predicted_auc_ratio,
                "observed_auc_ratio" => ddi.observed_auc_ratio,
                "classification" => ddi.predicted_class,
                "recommendation" => ddi.dose_recommendation
            ) for ddi in report.ddi_assessments
        ]
    )

    # Simple JSON serialization
    return dict_to_json(dict)
end

"""
Simple dictionary to JSON conversion.
"""
function dict_to_json(d::Dict; indent::Int = 2)::String
    buf = IOBuffer()
    _json_write(buf, d, 0, indent)
    return String(take!(buf))
end

function _json_write(io::IO, d::Dict, level::Int, indent::Int)
    println(io, "{")
    pairs = collect(d)
    for (i, (k, v)) in enumerate(pairs)
        print(io, " "^((level + 1) * indent), "\"$k\": ")
        _json_write(io, v, level + 1, indent)
        if i < length(pairs)
            println(io, ",")
        else
            println(io)
        end
    end
    print(io, " "^(level * indent), "}")
end

function _json_write(io::IO, v::Vector, level::Int, indent::Int)
    if isempty(v)
        print(io, "[]")
        return
    end
    println(io, "[")
    for (i, item) in enumerate(v)
        print(io, " "^((level + 1) * indent))
        _json_write(io, item, level + 1, indent)
        if i < length(v)
            println(io, ",")
        else
            println(io)
        end
    end
    print(io, " "^(level * indent), "]")
end

function _json_write(io::IO, v::String, level::Int, indent::Int)
    print(io, "\"", escape_string(v), "\"")
end

function _json_write(io::IO, v::Number, level::Int, indent::Int)
    if isnan(v)
        print(io, "null")
    else
        print(io, v)
    end
end

function _json_write(io::IO, v::Bool, level::Int, indent::Int)
    print(io, v ? "true" : "false")
end

function _json_write(io::IO, v::Any, level::Int, indent::Int)
    print(io, "\"", string(v), "\"")
end

end # module
