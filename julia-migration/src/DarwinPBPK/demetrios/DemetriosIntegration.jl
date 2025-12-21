# ===========================================================================
# DEMETRIOS INTEGRATION MODULE
# ===========================================================================
# FFI bindings and integration between Julia DarwinPBPK and Demetrios compiler.
#
# This module provides:
# - Compilation of Demetrios PBPK models
# - Data exchange via shared JSON format
# - Calling Demetrios functions from Julia
# - Importing Demetrios simulation results
#
# Author: Dr. Demetrios Agourakis
# Date: December 2025
# Version: 1.0.0
# ===========================================================================

module DemetriosIntegration

using JSON3
using Dates
using UUIDs

export DemetriosCompiler, DemetriosModel, DemetriosResult
export compile_demetrios, run_demetrios_pbpk, load_demetrios_result
export drug_to_demetrios, patient_to_demetrios, params_to_demetrios
export DemetriosDataFormat, export_for_demetrios, import_from_demetrios

# ===========================================================================
# Constants
# ===========================================================================

const DEMETRIOS_COMPILER_PATH = Ref{String}("")
const DEMETRIOS_VERSION = "0.78.1"

"""
    set_demetrios_path!(path::String)

Set the path to the Demetrios compiler binary.
"""
function set_demetrios_path!(path::String)
    if !isfile(path)
        error("Demetrios compiler not found at: $path")
    end
    DEMETRIOS_COMPILER_PATH[] = path
    @info "Demetrios compiler set to: $path"
end

"""
    get_demetrios_path()

Get the current Demetrios compiler path, auto-detecting if not set.
"""
function get_demetrios_path()
    if isempty(DEMETRIOS_COMPILER_PATH[])
        # Try to find in common locations
        candidates = [
            joinpath(@__DIR__, "..", "..", "..", "..", "Darwin-demetrios", "compiler", "target", "release", "dc"),
            joinpath(homedir(), ".demetrios", "bin", "dc"),
            "/usr/local/bin/dc",
        ]

        for candidate in candidates
            if isfile(candidate)
                DEMETRIOS_COMPILER_PATH[] = candidate
                @info "Auto-detected Demetrios compiler at: $candidate"
                return candidate
            end
        end

        error("Demetrios compiler not found. Set path with set_demetrios_path!()")
    end
    return DEMETRIOS_COMPILER_PATH[]
end

# ===========================================================================
# Compiler Interface
# ===========================================================================

"""
    DemetriosCompiler

Handle to the Demetrios compiler with configuration.
"""
struct DemetriosCompiler
    path::String
    version::String
    features::Vector{Symbol}
    output_dir::String
end

function DemetriosCompiler(;
    path::String = get_demetrios_path(),
    features::Vector{Symbol} = [:units, :epistemic, :effects],
    output_dir::String = tempdir()
)
    # Get version
    version_output = read(`$path --version`, String)
    version = match(r"demetrios (\d+\.\d+\.\d+)", version_output)
    ver_str = version !== nothing ? version.captures[1] : "unknown"

    DemetriosCompiler(path, ver_str, features, output_dir)
end

"""
    compile_demetrios(compiler::DemetriosCompiler, source_file::String; target=:check)

Compile a Demetrios source file.

Targets:
- `:check` - Type check only
- `:jit` - JIT compile and return function pointer
- `:llvm` - Compile to LLVM IR
- `:object` - Compile to object file
"""
function compile_demetrios(
    compiler::DemetriosCompiler,
    source_file::String;
    target::Symbol = :check,
    show_types::Bool = false,
    show_ast::Bool = false
)
    if !isfile(source_file)
        error("Source file not found: $source_file")
    end

    cmd_args = [compiler.path]

    # Target-specific commands
    if target == :check
        push!(cmd_args, "check")
    elseif target == :jit
        push!(cmd_args, "run")
    elseif target == :llvm
        push!(cmd_args, "compile", "--emit=llvm-ir")
    elseif target == :object
        push!(cmd_args, "compile", "--emit=obj")
    else
        error("Unknown target: $target")
    end

    push!(cmd_args, source_file)

    if show_types
        push!(cmd_args, "--show-types")
    end
    if show_ast
        push!(cmd_args, "--show-ast")
    end

    # Run compiler
    try
        output = read(Cmd(cmd_args), String)
        return (success=true, output=output, errors=nothing)
    catch e
        if e isa ProcessFailedException
            return (success=false, output="", errors=String(e.procs[1].cmd))
        end
        rethrow(e)
    end
end

# ===========================================================================
# Model Wrapper
# ===========================================================================

"""
    DemetriosModel

Wrapper for a compiled Demetrios PBPK model.
"""
struct DemetriosModel
    name::String
    source_file::String
    compiled::Bool
    entry_point::String
    input_schema::Dict{String, Any}
    output_schema::Dict{String, Any}
end

"""
    load_demetrios_model(source_file::String; entry_point="main")

Load and validate a Demetrios PBPK model.
"""
function load_demetrios_model(source_file::String; entry_point::String = "main")
    compiler = DemetriosCompiler()
    result = compile_demetrios(compiler, source_file, target=:check, show_types=true)

    if !result.success
        error("Failed to compile Demetrios model: $(result.errors)")
    end

    # Extract model name from file
    name = basename(source_file) |> x -> replace(x, ".d" => "")

    # Parse type information from compiler output to build schemas
    input_schema = parse_input_schema(result.output)
    output_schema = parse_output_schema(result.output)

    DemetriosModel(name, source_file, true, entry_point, input_schema, output_schema)
end

function parse_input_schema(compiler_output::String)
    # Placeholder - would parse actual type info from compiler
    Dict{String, Any}(
        "drug" => Dict("type" => "Drug", "required" => true),
        "patient" => Dict("type" => "Patient", "required" => true),
        "dose" => Dict("type" => "mg", "required" => true),
        "duration" => Dict("type" => "h", "required" => true),
    )
end

function parse_output_schema(compiler_output::String)
    Dict{String, Any}(
        "times" => Dict("type" => "Vec<h>"),
        "plasma_conc" => Dict("type" => "Vec<Knowledge[mg_per_L]>"),
        "metrics" => Dict("type" => "PKMetrics"),
    )
end

# ===========================================================================
# Data Format Exchange
# ===========================================================================

"""
    DemetriosDataFormat

Shared data format for Julia ↔ Demetrios exchange.
Uses JSON with schema validation.
"""
module DemetriosDataFormat

using JSON3
using Dates

export DrugData, PatientData, PBPKParamsData, SimulationRequest, SimulationResult

"""
Drug data in Demetrios-compatible format.
"""
struct DrugData
    name::String
    smiles::String
    mw::Float64
    logp::Float64
    tpsa::Float64
    fu_plasma::Float64
    bp_ratio::Float64
    pka_acidic::Union{Float64, Nothing}
    pka_basic::Union{Float64, Nothing}
    drug_class::String  # "acidic", "basic", "neutral", "zwitterionic"

    # Epistemic metadata
    confidence::Dict{String, Float64}
    provenance::Dict{String, String}
end

function DrugData(;
    name::String,
    smiles::String = "",
    mw::Float64,
    logp::Float64,
    tpsa::Float64 = 0.0,
    fu_plasma::Float64,
    bp_ratio::Float64 = 0.55,
    pka_acidic::Union{Float64, Nothing} = nothing,
    pka_basic::Union{Float64, Nothing} = nothing,
    drug_class::String = "neutral",
    confidence::Dict{String, Float64} = Dict("mw" => 0.99, "logp" => 0.90, "fu_plasma" => 0.85),
    provenance::Dict{String, String} = Dict("source" => "darwin_pbpk")
)
    DrugData(name, smiles, mw, logp, tpsa, fu_plasma, bp_ratio, pka_acidic, pka_basic, drug_class, confidence, provenance)
end

"""
Patient data in Demetrios-compatible format.
"""
struct PatientData
    id::String
    weight_kg::Float64
    height_cm::Float64
    age_years::Float64
    sex::String  # "male" or "female"
    egfr_ml_min::Float64
    liver_function::String  # "normal", "mild", "moderate", "severe"

    # Optional covariates
    bsa_m2::Union{Float64, Nothing}
    bmi::Union{Float64, Nothing}
    albumin_g_l::Union{Float64, Nothing}
end

function PatientData(;
    id::String = "SUBJ001",
    weight_kg::Float64 = 70.0,
    height_cm::Float64 = 170.0,
    age_years::Float64 = 35.0,
    sex::String = "male",
    egfr_ml_min::Float64 = 100.0,
    liver_function::String = "normal",
    bsa_m2::Union{Float64, Nothing} = nothing,
    bmi::Union{Float64, Nothing} = nothing,
    albumin_g_l::Union{Float64, Nothing} = nothing
)
    PatientData(id, weight_kg, height_cm, age_years, sex, egfr_ml_min, liver_function, bsa_m2, bmi, albumin_g_l)
end

"""
PBPK parameters in Demetrios-compatible format.
"""
struct PBPKParamsData
    cl_hepatic_l_h::Float64
    cl_renal_l_h::Float64
    vd_l::Float64
    ka_per_h::Float64
    f_oral::Float64
    kp_values::Dict{String, Float64}  # organ => Kp

    # Epistemic metadata
    confidence::Dict{String, Float64}
end

"""
Simulation request to send to Demetrios.
"""
struct SimulationRequest
    id::String
    timestamp::String
    model::String  # "darwin_pbpk_14comp", "mechanistic_ddi", etc.
    drug::DrugData
    patient::PatientData
    params::Union{PBPKParamsData, Nothing}
    dose_mg::Float64
    route::String  # "oral", "iv", "im"
    duration_h::Float64
    dt_h::Float64
    options::Dict{String, Any}
end

function SimulationRequest(;
    model::String,
    drug::DrugData,
    patient::PatientData,
    dose_mg::Float64,
    route::String = "oral",
    duration_h::Float64 = 24.0,
    dt_h::Float64 = 0.1,
    params::Union{PBPKParamsData, Nothing} = nothing,
    options::Dict{String, Any} = Dict()
)
    SimulationRequest(
        string(uuid4()),
        Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        model,
        drug,
        patient,
        params,
        dose_mg,
        route,
        duration_h,
        dt_h,
        options
    )
end

"""
Simulation result from Demetrios.
"""
struct SimulationResult
    id::String
    request_id::String
    success::Bool
    error_message::Union{String, Nothing}

    # PK results
    times_h::Vector{Float64}
    plasma_conc_mg_l::Vector{Float64}
    confidence::Vector{Float64}

    # Metrics
    cmax_mg_l::Float64
    cmax_confidence::Float64
    tmax_h::Float64
    auc_mg_h_l::Float64
    auc_confidence::Float64
    half_life_h::Float64
    half_life_confidence::Float64

    # Metadata
    computation_time_ms::Float64
    demetrios_version::String
    provenance::Dict{String, String}
end

end  # module DemetriosDataFormat

using .DemetriosDataFormat

# ===========================================================================
# Julia → Demetrios Conversion
# ===========================================================================

"""
    drug_to_demetrios(drug::Any) -> DrugData

Convert a Julia Drug struct to Demetrios format.
"""
function drug_to_demetrios(drug)
    # Handle different Julia drug representations
    if hasfield(typeof(drug), :name)
        name = drug.name
    elseif hasfield(typeof(drug), :drug_name)
        name = drug.drug_name
    else
        name = "unknown"
    end

    DrugData(
        name = string(name),
        smiles = get(drug, :smiles, ""),
        mw = Float64(get(drug, :mw, get(drug, :molecular_weight, 300.0))),
        logp = Float64(get(drug, :logp, get(drug, :logP, 2.0))),
        tpsa = Float64(get(drug, :tpsa, 50.0)),
        fu_plasma = Float64(get(drug, :fu, get(drug, :fu_plasma, 0.5))),
        bp_ratio = Float64(get(drug, :bp_ratio, get(drug, :blood_plasma_ratio, 0.55))),
        pka_acidic = get(drug, :pka_acidic, nothing),
        pka_basic = get(drug, :pka_basic, nothing),
        drug_class = string(get(drug, :drug_class, "neutral")),
        confidence = Dict(
            "mw" => 0.99,
            "logp" => get(drug, :logp_confidence, 0.90),
            "fu_plasma" => get(drug, :fu_confidence, 0.85),
        ),
        provenance = Dict("source" => "darwin_pbpk_julia")
    )
end

"""
    patient_to_demetrios(patient::Any) -> PatientData

Convert a Julia Patient struct to Demetrios format.
"""
function patient_to_demetrios(patient)
    PatientData(
        id = string(get(patient, :id, get(patient, :patient_id, "SUBJ001"))),
        weight_kg = Float64(get(patient, :weight, get(patient, :body_weight, 70.0))),
        height_cm = Float64(get(patient, :height, 170.0)),
        age_years = Float64(get(patient, :age, 35.0)),
        sex = lowercase(string(get(patient, :sex, get(patient, :gender, "male")))),
        egfr_ml_min = Float64(get(patient, :egfr, get(patient, :gfr, 100.0))),
        liver_function = string(get(patient, :liver_function, "normal")),
    )
end

"""
    params_to_demetrios(params::Any) -> PBPKParamsData

Convert Julia PBPK parameters to Demetrios format.
"""
function params_to_demetrios(params)
    # Extract Kp values
    kp_values = Dict{String, Float64}()
    organs = ["liver", "kidney", "brain", "heart", "lung", "muscle", "adipose", "gut", "skin", "bone", "spleen", "pancreas", "rest"]

    for organ in organs
        kp_key = Symbol("kp_$organ")
        if hasfield(typeof(params), kp_key)
            kp_values[organ] = Float64(getfield(params, kp_key))
        end
    end

    PBPKParamsData(
        Float64(get(params, :cl_hepatic, get(params, :clearance_hepatic, 10.0))),
        Float64(get(params, :cl_renal, get(params, :clearance_renal, 1.0))),
        Float64(get(params, :vd, get(params, :volume_distribution, 70.0))),
        Float64(get(params, :ka, get(params, :absorption_rate, 1.0))),
        Float64(get(params, :f_oral, get(params, :bioavailability, 0.5))),
        kp_values,
        Dict("cl_hepatic" => 0.80, "vd" => 0.85, "ka" => 0.75)
    )
end

# ===========================================================================
# Export/Import Functions
# ===========================================================================

"""
    export_for_demetrios(request::SimulationRequest, filepath::String)

Export a simulation request to JSON for Demetrios consumption.
"""
function export_for_demetrios(request::SimulationRequest, filepath::String)
    json_str = JSON3.write(request)
    open(filepath, "w") do f
        write(f, json_str)
    end
    @info "Exported simulation request to: $filepath"
    filepath
end

"""
    import_from_demetrios(filepath::String) -> SimulationResult

Import a simulation result from Demetrios JSON output.
"""
function import_from_demetrios(filepath::String)
    json_str = read(filepath, String)
    result = JSON3.read(json_str, SimulationResult)
    @info "Imported simulation result: $(result.id)"
    result
end

# ===========================================================================
# Run Demetrios PBPK
# ===========================================================================

"""
    run_demetrios_pbpk(model::DemetriosModel, request::SimulationRequest) -> SimulationResult

Run a Demetrios PBPK simulation via subprocess.
"""
function run_demetrios_pbpk(model::DemetriosModel, request::SimulationRequest)
    compiler = DemetriosCompiler()

    # Write request to temp file
    request_file = joinpath(tempdir(), "demetrios_request_$(request.id).json")
    result_file = joinpath(tempdir(), "demetrios_result_$(request.id).json")

    export_for_demetrios(request, request_file)

    # Run Demetrios with input
    start_time = time()
    try
        run(`$(compiler.path) run $(model.source_file) --input $request_file --output $result_file`)
        computation_time = (time() - start_time) * 1000  # ms

        # Load result
        result = import_from_demetrios(result_file)

        # Clean up
        rm(request_file, force=true)
        rm(result_file, force=true)

        return result

    catch e
        @error "Demetrios execution failed" exception=e

        # Return error result
        return SimulationResult(
            string(uuid4()),
            request.id,
            false,
            string(e),
            Float64[], Float64[], Float64[],
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
            DEMETRIOS_VERSION,
            Dict("error" => "execution_failed")
        )
    end
end

"""
    run_demetrios_pbpk(model_name::String; drug, patient, dose_mg, kwargs...)

Convenience function to run Demetrios PBPK with keyword arguments.
"""
function run_demetrios_pbpk(
    model_name::String;
    drug,
    patient,
    dose_mg::Float64,
    route::String = "oral",
    duration_h::Float64 = 24.0,
    dt_h::Float64 = 0.1
)
    # Find model source file
    pbpk_dir = joinpath(@__DIR__, "..", "..", "..", "..", "Darwin-demetrios", "examples", "pbpk")
    source_file = joinpath(pbpk_dir, "$model_name.d")

    if !isfile(source_file)
        error("Demetrios model not found: $source_file")
    end

    model = load_demetrios_model(source_file)

    request = SimulationRequest(
        model = model_name,
        drug = drug_to_demetrios(drug),
        patient = patient_to_demetrios(patient),
        dose_mg = dose_mg,
        route = route,
        duration_h = duration_h,
        dt_h = dt_h
    )

    run_demetrios_pbpk(model, request)
end

# ===========================================================================
# Batch Processing
# ===========================================================================

"""
    run_demetrios_batch(model_name::String, drugs::Vector, patient; kwargs...)

Run Demetrios PBPK for multiple drugs in batch.
Returns a vector of SimulationResult.
"""
function run_demetrios_batch(
    model_name::String,
    drugs::Vector,
    patient;
    dose_mg::Float64 = 100.0,
    kwargs...
)
    results = SimulationResult[]

    @info "Running Demetrios batch for $(length(drugs)) drugs..."

    for (i, drug) in enumerate(drugs)
        @info "Processing drug $i/$(length(drugs)): $(drug.name)"

        result = run_demetrios_pbpk(
            model_name;
            drug = drug,
            patient = patient,
            dose_mg = dose_mg,
            kwargs...
        )

        push!(results, result)
    end

    @info "Batch complete: $(count(r -> r.success, results))/$(length(results)) successful"
    results
end

# ===========================================================================
# Comparison Utilities
# ===========================================================================

"""
    compare_julia_demetrios(julia_result, demetrios_result)

Compare simulation results between Julia and Demetrios implementations.
"""
function compare_julia_demetrios(julia_result, demetrios_result::SimulationResult)
    # Extract Julia metrics
    julia_cmax = julia_result.metrics.cmax
    julia_tmax = julia_result.metrics.tmax
    julia_auc = julia_result.metrics.auc

    # Demetrios metrics
    dem_cmax = demetrios_result.cmax_mg_l
    dem_tmax = demetrios_result.tmax_h
    dem_auc = demetrios_result.auc_mg_h_l

    # Calculate fold errors
    cmax_fe = dem_cmax / julia_cmax
    auc_fe = dem_auc / julia_auc
    tmax_diff = abs(dem_tmax - julia_tmax)

    Dict(
        "cmax_julia" => julia_cmax,
        "cmax_demetrios" => dem_cmax,
        "cmax_fold_error" => cmax_fe,
        "auc_julia" => julia_auc,
        "auc_demetrios" => dem_auc,
        "auc_fold_error" => auc_fe,
        "tmax_julia" => julia_tmax,
        "tmax_demetrios" => dem_tmax,
        "tmax_difference_h" => tmax_diff,
        "agreement" => abs(log10(cmax_fe)) < 0.1 && abs(log10(auc_fe)) < 0.1
    )
end

end  # module DemetriosIntegration
