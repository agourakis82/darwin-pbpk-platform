# External PK Datasets Integration Module
# Provides access to curated public pharmacokinetic datasets for validation
#
# Data Sources:
# 1. Open-Systems-Pharmacology (OSP) Database - DDI and Pediatrics
# 2. Zenodo Beta-Lactam Critically Ill Dataset
# 3. PK-DB REST API (caffeine, morphine, midazolam, etc.)
#
# Version: 1.0.0
# License: MIT (code) / CC-BY for data attribution

module ExternalDatasets

using HTTP
using JSON3
using CSV
using DataFrames
using Dates

export ExternalDataSource, PKDBStudy, OSPRecord, ZenodoRecord
export list_available_datasets, load_osp_ddi, load_osp_pediatrics
export load_zenodo_betalactam, query_pkdb_api
export get_ddi_auc_ratios, get_pediatric_clearance
export validate_against_external_data

# ============================================================================
# Data Source Structures
# ============================================================================

"""
    ExternalDataSource

Metadata about an external pharmacokinetic data source.
"""
struct ExternalDataSource
    name::String
    description::String
    url::String
    license::String
    n_records::Int
    drugs::Vector{String}
    data_types::Vector{Symbol}  # :concentration, :auc_ratio, :clearance, :fu, :rb
    citation::String
end

"""
    PKDBStudy

A study from the PK-DB database.
"""
struct PKDBStudy
    sid::String
    name::String
    pmid::String
    n_individuals::Int
    n_groups::Int
    n_outputs::Int
    substances::Vector{String}
    has_timecourse::Bool
end

"""
    OSPRecord

A record from the Open-Systems-Pharmacology database.
"""
struct OSPRecord
    id::Int
    study_id::String
    reference::String
    victim::String
    perpetrator::String
    mechanism::String
    auc_ratio::Float64
    cmax_ratio::Union{Float64, Missing}
    dose::Float64
    dose_unit::String
    route::String
end

"""
    ZenodoRecord

A record from a Zenodo dataset.
"""
struct ZenodoRecord
    study_id::Int
    author::String
    year::Int
    country::String
    drug::String
    n_patients::Int
    patient_type::String
    covariates::Dict{String, Any}
end

# ============================================================================
# Available Datasets Catalog
# ============================================================================

const AVAILABLE_DATASETS = Dict{Symbol, ExternalDataSource}(
    :osp_ddi => ExternalDataSource(
        "OSP Drug-Drug Interactions",
        "Curated DDI data from peer-reviewed literature for PBPK model qualification",
        "https://github.com/Open-Systems-Pharmacology/Database-for-observed-data",
        "Open Source (OSP Foundation)",
        634,
        ["Midazolam", "Alfentanil", "Triazolam", "Alprazolam", "Rifampicin", "Itraconazole", "Clarithromycin", "Erythromycin", "Fluconazole", "Verapamil"],
        [:auc_ratio, :cmax_ratio],
        "Open Systems Pharmacology Suite. Database for observed data. GitHub, 2024."
    ),
    :osp_pediatrics => ExternalDataSource(
        "OSP Pediatric PK",
        "Pediatric pharmacokinetic parameters from literature",
        "https://github.com/Open-Systems-Pharmacology/Database-for-observed-data",
        "Open Source (OSP Foundation)",
        277,
        ["Sufentanil", "Fentanyl", "Alfentanil", "Morphine", "Midazolam"],
        [:clearance, :auc, :cmax],
        "Open Systems Pharmacology Suite. Pediatric Database. GitHub, 2024."
    ),
    :zenodo_betalactam => ExternalDataSource(
        "Beta-Lactam PK in Critically Ill",
        "Population PK studies of beta-lactam antibiotics in ICU patients",
        "https://zenodo.org/records/8241522",
        "CC-BY 4.0",
        151,
        ["Piperacillin", "Meropenem", "Cefepime", "Doripenem", "Ceftazidime", "Ceftriaxone"],
        [:clearance, :volume_distribution, :pta],
        "Abdul-Aziz MH et al. Beta-lactam PK systematic review data. Zenodo, 2023. DOI:10.5281/zenodo.8241522"
    ),
    :pkdb => ExternalDataSource(
        "PK-DB (Pharmacokinetics Database)",
        "Open database for pharmacokinetics information from clinical trials",
        "https://pk-db.com",
        "MIT (code) / CC-BY-SA 4.0 (data)",
        796,  # studies
        ["Caffeine", "Morphine", "Codeine", "Midazolam", "Acetaminophen", "Diazepam", "Simvastatin", "Glucose"],
        [:concentration, :auc, :clearance, :half_life, :cmax, :tmax],
        "König M et al. PK-DB: pharmacokinetics database. Nucleic Acids Res. 2021;49(D1):D1358-D1363. PMID:33151297"
    )
)

"""
    list_available_datasets()

List all available external datasets with their metadata.
"""
function list_available_datasets()
    println("=" ^ 80)
    println("AVAILABLE EXTERNAL PK DATASETS FOR VALIDATION")
    println("=" ^ 80)

    for (key, ds) in AVAILABLE_DATASETS
        println("\n[$key]")
        println("  Name: $(ds.name)")
        println("  Records: $(ds.n_records)")
        println("  Drugs: $(join(ds.drugs[1:min(5, length(ds.drugs))], ", "))$(length(ds.drugs) > 5 ? "..." : "")")
        println("  Data Types: $(join(string.(ds.data_types), ", "))")
        println("  License: $(ds.license)")
        println("  URL: $(ds.url)")
    end

    return AVAILABLE_DATASETS
end

# ============================================================================
# OSP Database Loading
# ============================================================================

const DATA_DIR = joinpath(@__DIR__, "..", "..", "..", "data", "external_pk_datasets")

"""
    load_osp_ddi(; filter_drug::Union{String, Nothing}=nothing)

Load the OSP Drug-Drug Interaction database.
Returns a DataFrame with AUC ratios and Cmax ratios.
"""
function load_osp_ddi(; filter_drug::Union{String, Nothing}=nothing)
    filepath = joinpath(DATA_DIR, "OSP_DDI.csv")

    if !isfile(filepath)
        error("OSP DDI dataset not found. Run download script first: $filepath")
    end

    df = CSV.read(filepath, DataFrame)

    # Clean column names
    rename!(df, Symbol("Study ID") => :study_id)

    if !isnothing(filter_drug)
        filter_lower = lowercase(filter_drug)
        df = filter(row ->
            lowercase(string(get(row, :Victim, ""))) == filter_lower ||
            lowercase(string(get(row, :Perpetrator, ""))) == filter_lower,
            df
        )
    end

    return df
end

"""
    load_osp_pediatrics(; filter_drug::Union{String, Nothing}=nothing)

Load the OSP Pediatric PK database.
Returns a DataFrame with clearance and AUC values.
"""
function load_osp_pediatrics(; filter_drug::Union{String, Nothing}=nothing)
    filepath = joinpath(DATA_DIR, "OSP_Pediatrics.csv")

    if !isfile(filepath)
        error("OSP Pediatrics dataset not found. Run download script first: $filepath")
    end

    df = CSV.read(filepath, DataFrame)

    if !isnothing(filter_drug)
        filter_lower = lowercase(filter_drug)
        df = filter(row ->
            lowercase(string(get(row, :Analyte, ""))) == filter_lower,
            df
        )
    end

    return df
end

# ============================================================================
# Zenodo Beta-Lactam Dataset
# ============================================================================

"""
    load_zenodo_betalactam()

Load the Zenodo beta-lactam critically ill dataset.
Returns two DataFrames: covariates and outcomes.
"""
function load_zenodo_betalactam()
    cov_path = joinpath(DATA_DIR, "Zenodo_BetaLactam_CriticallyIll_covariates.csv")
    out_path = joinpath(DATA_DIR, "Zenodo_BetaLactam_CriticallyIll_outcomes.csv")

    if !isfile(cov_path) || !isfile(out_path)
        error("Zenodo beta-lactam dataset not found. Download from: https://zenodo.org/records/8241522")
    end

    covariates = CSV.read(cov_path, DataFrame)
    outcomes = CSV.read(out_path, DataFrame)

    return (covariates=covariates, outcomes=outcomes)
end

# ============================================================================
# PK-DB REST API Integration
# ============================================================================

const PKDB_API_BASE = "https://pk-db.com/api/v1"

"""
    query_pkdb_api(endpoint::String; params::Dict=Dict())

Query the PK-DB REST API.
"""
function query_pkdb_api(endpoint::String; params::Dict=Dict())
    url = "$PKDB_API_BASE/$endpoint/"

    if !isempty(params)
        query_string = join(["$k=$v" for (k, v) in params], "&")
        url = "$url?$query_string"
    end

    try
        response = HTTP.get(url; headers=["Accept" => "application/json"])
        return JSON3.read(String(response.body))
    catch e
        @warn "PK-DB API query failed: $e"
        return nothing
    end
end

"""
    list_pkdb_studies(; substance::Union{String, Nothing}=nothing, page::Int=1)

List studies from PK-DB.
"""
function list_pkdb_studies(; substance::Union{String, Nothing}=nothing, page::Int=1)
    params = Dict("format" => "json", "page" => string(page))

    result = query_pkdb_api("studies"; params=params)

    if isnothing(result)
        return PKDBStudy[]
    end

    studies = PKDBStudy[]
    for study in result.data.data
        push!(studies, PKDBStudy(
            string(study.sid),
            string(study.name),
            haskey(study, :reference) && !isnothing(study.reference) ?
                string(get(study.reference, :pmid, "")) : "",
            study.individual_count,
            study.group_count,
            study.output_count,
            [string(s.name) for s in get(study, :substances, [])],
            study.timecourse_count > 0
        ))
    end

    return studies
end

"""
    get_pkdb_study_detail(sid::String)

Get detailed information about a specific PK-DB study.
"""
function get_pkdb_study_detail(sid::String)
    return query_pkdb_api("studies/$sid"; params=Dict("format" => "json"))
end

# ============================================================================
# Validation Helper Functions
# ============================================================================

"""
    get_ddi_auc_ratios(victim::String, perpetrator::String)

Get AUC ratios for a specific drug-drug interaction from OSP database.
"""
function get_ddi_auc_ratios(victim::String, perpetrator::String)
    df = load_osp_ddi()

    victim_lower = lowercase(victim)
    perp_lower = lowercase(perpetrator)

    filtered = filter(row ->
        lowercase(string(get(row, :Victim, ""))) == victim_lower &&
        lowercase(string(get(row, :Perpetrator, ""))) == perp_lower,
        df
    )

    if nrow(filtered) == 0
        return (n=0, mean=nothing, std=nothing, min=nothing, max=nothing, studies=String[])
    end

    auc_col = Symbol("AUCR Avg")
    auc_values = [row[auc_col] for row in eachrow(filtered) if !ismissing(row[auc_col])]

    studies = unique([string(row[:study_id]) for row in eachrow(filtered)])

    return (
        n = length(auc_values),
        mean = mean(auc_values),
        std = length(auc_values) > 1 ? std(auc_values) : 0.0,
        min = minimum(auc_values),
        max = maximum(auc_values),
        studies = studies
    )
end

"""
    get_pediatric_clearance(drug::String)

Get pediatric clearance values from OSP database.
"""
function get_pediatric_clearance(drug::String)
    df = load_osp_pediatrics(filter_drug=drug)

    if nrow(df) == 0
        return nothing
    end

    cl_col = Symbol("CL Avg")
    cl_values = [row[cl_col] for row in eachrow(df) if !ismissing(row[cl_col])]

    if isempty(cl_values)
        return nothing
    end

    return (
        n = length(cl_values),
        mean = mean(cl_values),
        std = length(cl_values) > 1 ? std(cl_values) : 0.0,
        unit = df[1, Symbol("CL AvgUnit")],
        studies = unique([string(row[:Study]) for row in eachrow(df)])
    )
end

# ============================================================================
# Validation Against External Data
# ============================================================================

"""
    ExternalValidationResult

Result of validating a model prediction against external data.
"""
struct ExternalValidationResult
    parameter::Symbol
    predicted::Float64
    observed_mean::Float64
    observed_std::Float64
    observed_n::Int
    percent_error::Float64
    within_2sd::Bool
    data_source::Symbol
    studies::Vector{String}
end

"""
    validate_against_external_data(predictions::Dict, data_source::Symbol)

Validate model predictions against external dataset.
Returns a vector of ExternalValidationResult.
"""
function validate_against_external_data(predictions::Dict, data_source::Symbol)
    if data_source ∉ keys(AVAILABLE_DATASETS)
        error("Unknown data source: $data_source. Available: $(keys(AVAILABLE_DATASETS))")
    end

    results = ExternalValidationResult[]

    # Implement validation logic based on data source
    # This is a placeholder - specific validation depends on what's being validated

    return results
end

# ============================================================================
# Summary Statistics
# ============================================================================

"""
    summarize_external_datasets()

Print summary statistics of all loaded external datasets.
"""
function summarize_external_datasets()
    println("\n" * "=" ^ 80)
    println("EXTERNAL PK DATASETS - SUMMARY")
    println("=" ^ 80)

    # OSP DDI
    try
        ddi = load_osp_ddi()
        println("\n[OSP DDI Database]")
        println("  Total records: $(nrow(ddi))")
        println("  Unique victims: $(length(unique(ddi.Victim)))")
        println("  Unique perpetrators: $(length(unique(ddi.Perpetrator)))")

        # Top interactions
        println("  Top victim drugs: $(join(first(sort(combine(groupby(ddi, :Victim), nrow => :n), :n, rev=true), 5).Victim, ", "))")
    catch e
        println("\n[OSP DDI Database] - Not loaded: $e")
    end

    # OSP Pediatrics
    try
        ped = load_osp_pediatrics()
        println("\n[OSP Pediatrics Database]")
        println("  Total records: $(nrow(ped))")
        println("  Unique analytes: $(length(unique(ped.Analyte)))")
    catch e
        println("\n[OSP Pediatrics Database] - Not loaded: $e")
    end

    # Zenodo Beta-Lactam
    try
        bl = load_zenodo_betalactam()
        println("\n[Zenodo Beta-Lactam ICU Dataset]")
        println("  Studies: $(nrow(bl.covariates))")
        println("  Outcome records: $(nrow(bl.outcomes))")
        println("  Drugs: $(join(unique(bl.covariates.betalactam_studied), ", "))")
    catch e
        println("\n[Zenodo Beta-Lactam ICU Dataset] - Not loaded: $e")
    end

    # PK-DB
    try
        studies = list_pkdb_studies(page=1)
        println("\n[PK-DB API]")
        println("  Studies (page 1): $(length(studies))")
        if !isempty(studies)
            println("  Example study: $(studies[1].name) ($(studies[1].n_individuals) individuals)")
        end
    catch e
        println("\n[PK-DB API] - Not accessible: $e")
    end

    println("\n" * "=" ^ 80)
end

end # module ExternalDatasets
