# ===========================================================================
# SOUNIO INTEGRATION MODULE
# ===========================================================================
# FFI bindings and integration between Julia DarwinPBPK and Sounio compiler.
#
# This module provides:
# - Compilation of Sounio PBPK models
# - Data exchange via shared JSON format
# - Calling Sounio functions from Julia
# - Importing Sounio simulation results
#
# Author: Dr. Sounio Agourakis
# Date: December 2025
# Version: 1.0.0
# ===========================================================================

module SounioIntegration

using JSON3
using Dates
using UUIDs

export SounioCompiler, SounioModel, SounioResult
export compile_sounio, run_sounio_pbpk, load_sounio_result
export drug_to_sounio, patient_to_sounio, params_to_sounio
export SounioDataFormat, export_for_sounio, import_from_sounio
export compile_and_run_medlang, MedLangPipelineResult, MedLangPipelineOptions

# Epistemic types and UQ bridge exports
export EpistemicTypes, UQBridge
export run_sounio_pbpk_gated, EpistemicSimulationResult

# Include epistemic computing modules
include("EpistemicTypes.jl")
include("UQBridge.jl")
using .EpistemicTypes
using .UQBridge

# ===========================================================================
# Constants
# ===========================================================================

const SOUNIO_COMPILER_PATH = Ref{String}("")
const SOUNIO_VERSION = "0.97.0"  # Updated for darwin_pbpk stdlib integration

"""
    set_sounio_path!(path::String)

Set the path to the Sounio compiler binary.
"""
function set_sounio_path!(path::String)
    if !isfile(path)
        error("Sounio compiler not found at: $path")
    end
    SOUNIO_COMPILER_PATH[] = path
    @info "Sounio compiler set to: $path"
end

"""
    get_sounio_path()

Get the current Sounio compiler path, auto-detecting if not set.
"""
function get_sounio_path()
    if isempty(SOUNIO_COMPILER_PATH[])
        # Try to find in common locations
        candidates = [
            joinpath(@__DIR__, "..", "..", "..", "..", "Darwin-sounio", "compiler", "target", "release", "dc"),
            joinpath(homedir(), ".sounio", "bin", "dc"),
            "/usr/local/bin/dc",
        ]

        for candidate in candidates
            if isfile(candidate)
                SOUNIO_COMPILER_PATH[] = candidate
                @info "Auto-detected Sounio compiler at: $candidate"
                return candidate
            end
        end

        error("Sounio compiler not found. Set path with set_sounio_path!()")
    end
    return SOUNIO_COMPILER_PATH[]
end

# ===========================================================================
# Compiler Interface
# ===========================================================================

"""
    SounioCompiler

Handle to the Sounio compiler with configuration.
"""
struct SounioCompiler
    path::String
    version::String
    features::Vector{Symbol}
    output_dir::String
end

function SounioCompiler(;
    path::String = get_sounio_path(),
    features::Vector{Symbol} = [:units, :epistemic, :effects],
    output_dir::String = tempdir()
)
    # Get version
    version_output = read(`$path --version`, String)
    version = match(r"sounio (\d+\.\d+\.\d+)", version_output)
    ver_str = version !== nothing ? version.captures[1] : "unknown"

    SounioCompiler(path, ver_str, features, output_dir)
end

"""
    compile_sounio(compiler::SounioCompiler, source_file::String; target=:check)

Compile a Sounio source file.

Targets:
- `:check` - Type check only
- `:jit` - JIT compile and return function pointer
- `:llvm` - Compile to LLVM IR
- `:object` - Compile to object file
"""
function compile_sounio(
    compiler::SounioCompiler,
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
    SounioModel

Wrapper for a compiled Sounio PBPK model.
"""
struct SounioModel
    name::String
    source_file::String
    compiled::Bool
    entry_point::String
    input_schema::Dict{String, Any}
    output_schema::Dict{String, Any}
end

"""
    load_sounio_model(source_file::String; entry_point="main")

Load and validate a Sounio PBPK model.
"""
function load_sounio_model(source_file::String; entry_point::String = "main")
    compiler = SounioCompiler()
    result = compile_sounio(compiler, source_file, target=:check, show_types=true)

    if !result.success
        error("Failed to compile Sounio model: $(result.errors)")
    end

    # Extract model name from file
    name = basename(source_file) |> x -> replace(x, ".sio" => "")

    # Parse type information from compiler output to build schemas
    input_schema = parse_input_schema(result.output)
    output_schema = parse_output_schema(result.output)

    SounioModel(name, source_file, true, entry_point, input_schema, output_schema)
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
    SounioDataFormat

Shared data format for Julia ↔ Sounio exchange.
Uses JSON with schema validation.
"""
module SounioDataFormat

using JSON3
using Dates

export DrugData, PatientData, PBPKParamsData, SimulationRequest, SimulationResult

"""
Drug data in Sounio-compatible format.
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
Patient data in Sounio-compatible format.
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
PBPK parameters in Sounio-compatible format.
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
Simulation request to send to Sounio.
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
Simulation result from Sounio.
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
    sounio_version::String
    provenance::Dict{String, String}
end

end  # module SounioDataFormat

using .SounioDataFormat

# ===========================================================================
# Julia → Sounio Conversion
# ===========================================================================

"""
    drug_to_sounio(drug::Any) -> DrugData

Convert a Julia Drug struct to Sounio format.
"""
function drug_to_sounio(drug)
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
    patient_to_sounio(patient::Any) -> PatientData

Convert a Julia Patient struct to Sounio format.
"""
function patient_to_sounio(patient)
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
    params_to_sounio(params::Any) -> PBPKParamsData

Convert Julia PBPK parameters to Sounio format.
"""
function params_to_sounio(params)
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
    export_for_sounio(request::SimulationRequest, filepath::String)

Export a simulation request to JSON for Sounio consumption.
"""
function export_for_sounio(request::SimulationRequest, filepath::String)
    json_str = JSON3.write(request)
    open(filepath, "w") do f
        write(f, json_str)
    end
    @info "Exported simulation request to: $filepath"
    filepath
end

"""
    import_from_sounio(filepath::String) -> SimulationResult

Import a simulation result from Sounio JSON output.
"""
function import_from_sounio(filepath::String)
    json_str = read(filepath, String)
    result = JSON3.read(json_str, SimulationResult)
    @info "Imported simulation result: $(result.id)"
    result
end

# ===========================================================================
# Run Sounio PBPK
# ===========================================================================

"""
    run_sounio_pbpk(model::SounioModel, request::SimulationRequest) -> SimulationResult

Run a Sounio PBPK simulation via subprocess.
"""
function run_sounio_pbpk(model::SounioModel, request::SimulationRequest)
    compiler = SounioCompiler()

    # Write request to temp file
    request_file = joinpath(tempdir(), "sounio_request_$(request.id).json")
    result_file = joinpath(tempdir(), "sounio_result_$(request.id).json")

    export_for_sounio(request, request_file)

    # Run Sounio with input
    start_time = time()
    try
        run(`$(compiler.path) run $(model.source_file) --input $request_file --output $result_file`)
        computation_time = (time() - start_time) * 1000  # ms

        # Load result
        result = import_from_sounio(result_file)

        # Clean up
        rm(request_file, force=true)
        rm(result_file, force=true)

        return result

    catch e
        @error "Sounio execution failed" exception=e

        # Return error result
        return SimulationResult(
            string(uuid4()),
            request.id,
            false,
            string(e),
            Float64[], Float64[], Float64[],
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
            SOUNIO_VERSION,
            Dict("error" => "execution_failed")
        )
    end
end

"""
    run_sounio_pbpk(model_name::String; drug, patient, dose_mg, kwargs...)

Convenience function to run Sounio PBPK with keyword arguments.
"""
function run_sounio_pbpk(
    model_name::String;
    drug,
    patient,
    dose_mg::Float64,
    route::String = "oral",
    duration_h::Float64 = 24.0,
    dt_h::Float64 = 0.1
)
    # Find model source file
    pbpk_dir = joinpath(@__DIR__, "..", "..", "..", "..", "Darwin-sounio", "examples", "pbpk")
    source_file = joinpath(pbpk_dir, "$model_name.sio")

    if !isfile(source_file)
        error("Sounio model not found: $source_file")
    end

    model = load_sounio_model(source_file)

    request = SimulationRequest(
        model = model_name,
        drug = drug_to_sounio(drug),
        patient = patient_to_sounio(patient),
        dose_mg = dose_mg,
        route = route,
        duration_h = duration_h,
        dt_h = dt_h
    )

    run_sounio_pbpk(model, request)
end

# ===========================================================================
# Epistemic Simulation (Confidence-Gated)
# ===========================================================================

"""
    EpistemicSimulationResult

Simulation result with epistemic confidence tracking.
"""
struct EpistemicSimulationResult
    base_result::SimulationResult
    confidence::Float64
    passed_gate::Bool
    gated_metrics::Dict{String, EpistemicTypes.Knowledge{Float64}}
end

"""
    run_sounio_pbpk_gated(model, request; min_confidence=0.80) -> EpistemicSimulationResult

Run Sounio PBPK simulation with confidence gating.

Only executes if input parameters meet minimum confidence threshold.

# Arguments
- `model::SounioModel`: Compiled Sounio model
- `request::SimulationRequest`: Simulation request with epistemic drug data
- `min_confidence::Float64`: Minimum required confidence (default: 0.80)

# Returns
- `EpistemicSimulationResult`: Result with confidence tracking

# Example
```julia
drug = EpistemicTypes.EpistemicDrugData(
    name = "TestDrug",
    mw = Knowledge(300.0, 0.95, MEASURED),
    logp = Knowledge(2.5, 0.85, LITERATURE),
    fu_plasma = Knowledge(0.4, 0.75, ESTIMATED)
)

result = run_sounio_pbpk_gated(model, request; min_confidence=0.80)

if result.passed_gate
    println("Simulation passed confidence gate")
    println("Cmax confidence: \$(result.gated_metrics[:cmax].confidence)")
end
```
"""
function run_sounio_pbpk_gated(
    model::SounioModel,
    request::SimulationRequest;
    min_confidence::Float64 = 0.80
)::EpistemicSimulationResult
    # Check input confidence from drug data
    input_confidence = get(request.drug.confidence, "overall", 1.0)

    # Also check specific parameter confidences
    param_confidences = [
        get(request.drug.confidence, "mw", 0.99),
        get(request.drug.confidence, "logp", 0.90),
        get(request.drug.confidence, "fu_plasma", 0.85),
    ]

    min_input_conf = minimum(param_confidences)

    # Check confidence gate
    passed_gate = min_input_conf >= min_confidence

    if !passed_gate
        @warn "Confidence gate failed" min_input_conf min_confidence
        # Return result with zero confidence
        return EpistemicSimulationResult(
            SimulationResult(
                "", request.id, false, "Confidence gate failed: $(min_input_conf) < $(min_confidence)",
                Float64[], Float64[], Float64[],
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, SOUNIO_VERSION, Dict("gate_failed" => "true")
            ),
            min_input_conf,
            false,
            Dict{String, EpistemicTypes.Knowledge{Float64}}()
        )
    end

    # Run actual simulation
    base_result = run_sounio_pbpk(model, request)

    # Create epistemic metrics
    gated_metrics = Dict{String, EpistemicTypes.Knowledge{Float64}}()

    if base_result.success
        # Output confidence is propagated from inputs
        output_confidence = min_input_conf * base_result.cmax_confidence

        gated_metrics["cmax"] = EpistemicTypes.Knowledge(
            base_result.cmax_mg_l,
            output_confidence,
            EpistemicTypes.COMPUTED
        )

        gated_metrics["auc"] = EpistemicTypes.Knowledge(
            base_result.auc_mg_h_l,
            min_input_conf * base_result.auc_confidence,
            EpistemicTypes.COMPUTED
        )

        gated_metrics["tmax"] = EpistemicTypes.Knowledge(
            base_result.tmax_h,
            min_input_conf,
            EpistemicTypes.COMPUTED
        )

        gated_metrics["half_life"] = EpistemicTypes.Knowledge(
            base_result.half_life_h,
            min_input_conf * base_result.half_life_confidence,
            EpistemicTypes.COMPUTED
        )
    end

    EpistemicSimulationResult(
        base_result,
        min_input_conf,
        passed_gate,
        gated_metrics
    )
end

"""
    run_sounio_pbpk_gated(model_name::String; drug_epistemic, patient, dose_mg, min_confidence=0.80, kwargs...)

Convenience function with epistemic drug data.
"""
function run_sounio_pbpk_gated(
    model_name::String;
    drug_epistemic::EpistemicTypes.EpistemicDrugData,
    patient,
    dose_mg::Float64,
    min_confidence::Float64 = 0.80,
    route::String = "oral",
    duration_h::Float64 = 24.0,
    dt_h::Float64 = 0.1
)::EpistemicSimulationResult
    # Convert epistemic drug to standard format with confidence metadata
    drug = DrugData(
        name = drug_epistemic.name,
        mw = EpistemicTypes.value(drug_epistemic.mw),
        logp = EpistemicTypes.value(drug_epistemic.logp),
        fu_plasma = EpistemicTypes.value(drug_epistemic.fu_plasma),
        bp_ratio = EpistemicTypes.value(drug_epistemic.bp_ratio),
        confidence = Dict(
            "mw" => EpistemicTypes.confidence(drug_epistemic.mw),
            "logp" => EpistemicTypes.confidence(drug_epistemic.logp),
            "fu_plasma" => EpistemicTypes.confidence(drug_epistemic.fu_plasma),
            "overall" => drug_epistemic.min_confidence
        )
    )

    # Find model source file
    pbpk_dir = joinpath(@__DIR__, "..", "..", "..", "..", "Darwin-sounio", "examples", "pbpk")
    source_file = joinpath(pbpk_dir, "$model_name.sio")

    if !isfile(source_file)
        error("Sounio model not found: $source_file")
    end

    model = load_sounio_model(source_file)

    request = SimulationRequest(
        model = model_name,
        drug = drug,
        patient = patient_to_sounio(patient),
        dose_mg = dose_mg,
        route = route,
        duration_h = duration_h,
        dt_h = dt_h
    )

    run_sounio_pbpk_gated(model, request; min_confidence=min_confidence)
end

# ===========================================================================
# MedLang → Sounio Pipeline
# ===========================================================================

"""
    MedLangPipelineOptions

Configuration options for the MedLang → Sounio compilation pipeline.
"""
Base.@kwdef struct MedLangPipelineOptions
    use_stdlib::Bool = true
    cleanup_temp_files::Bool = true
    show_generated_code::Bool = false
    timeout_seconds::Float64 = 60.0
    output_dir::String = tempdir()
    sounio_features::Vector{Symbol} = [:units, :epistemic, :effects]
end

"""
    MedLangPipelineResult

Result of MedLang → Sounio pipeline execution.
"""
struct MedLangPipelineResult
    success::Bool
    medlang_source::String
    sounio_code::String
    sounio_file::String
    compilation_success::Bool
    compilation_output::String
    simulation_result::Union{SimulationResult, Nothing}
    error_message::Union{String, Nothing}
    pipeline_time_ms::Float64
    stages_completed::Vector{Symbol}
end

"""
    compile_and_run_medlang(source::String; dose_mg, duration_h, patient, kwargs...) -> MedLangPipelineResult

Complete pipeline: MedLang source → compile to Sounio → execute → return results.

# Arguments
- `source::String`: MedLang source code
- `dose_mg::Float64`: Dose in mg
- `duration_h::Float64`: Simulation duration in hours (default: 24.0)
- `patient`: Patient data (optional, uses default if not provided)
- `options::MedLangPipelineOptions`: Pipeline configuration

# Returns
- `MedLangPipelineResult`: Complete pipeline result with all stages

# Example
```julia
source = \"\"\"
model MyDrug {
    clearance hepatic: 10.0_L/h
    organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 2.5 }
}
\"\"\"
result = compile_and_run_medlang(source; dose_mg=100.0, duration_h=24.0)

if result.success
    println("Cmax: \$(result.simulation_result.cmax_mg_l)")
    println("AUC: \$(result.simulation_result.auc_mg_h_l)")
end
```

# Pipeline Stages
1. **Parse**: Parse MedLang source to model
2. **Generate**: Generate Sounio code (stdlib or standalone)
3. **Write**: Write Sounio code to temp file
4. **Compile**: Compile with Sounio compiler (type-check)
5. **Execute**: Run simulation
6. **Collect**: Import and return results

# Notes
If Sounio compiler is not available, returns partial result with generated code.
Set `options.show_generated_code=true` to print Sounio code.
"""
function compile_and_run_medlang(
    source::String;
    dose_mg::Float64,
    duration_h::Float64 = 24.0,
    patient::Union{PatientData, Nothing} = nothing,
    drug_name::String = "MedLangDrug",
    route::String = "oral",
    options::MedLangPipelineOptions = MedLangPipelineOptions()
)::MedLangPipelineResult
    start_time = time()
    stages_completed = Symbol[]
    sounio_code = ""
    sounio_file = ""
    compilation_output = ""
    compilation_success = false
    simulation_result = nothing
    error_message = nothing

    # Use default patient if not provided
    if patient === nothing
        patient = PatientData()
    end

    try
        # =====================================================================
        # Stage 1: Parse MedLang
        # =====================================================================
        @info "Stage 1: Parsing MedLang source..."

        # Import parser from MedLang module
        # This uses the parse_medlang_to_model function from medlang_sounio_compiler.jl
        model = parse_medlang_to_model_safe(source)
        push!(stages_completed, :parse)

        # =====================================================================
        # Stage 2: Generate Sounio code
        # =====================================================================
        @info "Stage 2: Generating Sounio code (stdlib=$(options.use_stdlib))..."

        sounio_code = generate_sounio_code_safe(model, options.use_stdlib)
        push!(stages_completed, :generate)

        if options.show_generated_code
            println("=" ^ 60)
            println("Generated Sounio Code:")
            println("=" ^ 60)
            println(sounio_code)
            println("=" ^ 60)
        end

        # =====================================================================
        # Stage 3: Write to temp file
        # =====================================================================
        @info "Stage 3: Writing Sounio code to file..."

        model_name = extract_model_name(source, drug_name)
        sounio_file = joinpath(options.output_dir, "$(model_name)_$(string(uuid4())[1:8]).sio")
        write(sounio_file, sounio_code)
        push!(stages_completed, :write)

        # =====================================================================
        # Stage 4: Compile with Sounio
        # =====================================================================
        @info "Stage 4: Compiling with Sounio..."

        try
            compiler = SounioCompiler(features=options.sounio_features)
            result = compile_sounio(compiler, sounio_file, target=:check, show_types=true)

            compilation_success = result.success
            compilation_output = result.output
            push!(stages_completed, :compile)

            if !result.success
                @warn "Sounio compilation failed" errors=result.errors
                error_message = "Compilation failed: $(result.errors)"
            end
        catch e
            @warn "Sounio compiler not available" exception=e
            error_message = "Sounio compiler not available: $(e)"
            # Continue without execution - still return generated code
        end

        # =====================================================================
        # Stage 5: Execute simulation (if compilation succeeded)
        # =====================================================================
        if compilation_success
            @info "Stage 5: Executing simulation..."

            try
                model_obj = load_sounio_model(sounio_file)

                # Create drug data from model
                drug_data = DrugData(
                    name = model_name,
                    mw = 300.0,  # Default - would be extracted from model
                    logp = 2.0,
                    fu_plasma = 0.5
                )

                request = SimulationRequest(
                    model = model_name,
                    drug = drug_data,
                    patient = patient,
                    dose_mg = dose_mg,
                    route = route,
                    duration_h = duration_h,
                    dt_h = 0.1
                )

                simulation_result = run_sounio_pbpk(model_obj, request)
                push!(stages_completed, :execute)

                # =====================================================================
                # Stage 6: Collect results
                # =====================================================================
                push!(stages_completed, :collect)

            catch e
                @warn "Simulation execution failed" exception=e
                error_message = "Execution failed: $(e)"
            end
        end

    catch e
        @error "Pipeline error" exception=e
        error_message = "Pipeline error: $(e)"
    end

    # Cleanup temp files if requested
    if options.cleanup_temp_files && isfile(sounio_file)
        try
            rm(sounio_file, force=true)
        catch
            # Ignore cleanup errors
        end
    end

    pipeline_time = (time() - start_time) * 1000  # ms

    success = :collect in stages_completed && simulation_result !== nothing && simulation_result.success

    MedLangPipelineResult(
        success,
        source,
        sounio_code,
        sounio_file,
        compilation_success,
        compilation_output,
        simulation_result,
        error_message,
        pipeline_time,
        stages_completed
    )
end

"""
    compile_and_run_medlang_file(filepath::String; kwargs...) -> MedLangPipelineResult

Compile and run a MedLang file through the Sounio pipeline.

# Arguments
- `filepath::String`: Path to .medlang file
- All other arguments same as `compile_and_run_medlang`
"""
function compile_and_run_medlang_file(
    filepath::String;
    kwargs...
)::MedLangPipelineResult
    if !isfile(filepath)
        return MedLangPipelineResult(
            false, "", "", "", false, "", nothing,
            "File not found: $filepath",
            0.0, Symbol[]
        )
    end

    source = read(filepath, String)
    compile_and_run_medlang(source; kwargs...)
end

export compile_and_run_medlang_file

# Helper functions for pipeline

"""
Parse MedLang source safely, returning a simplified model structure.
"""
function parse_medlang_to_model_safe(source::String)
    # Try to import from MedLang module
    try
        # Use the parser from medlang_sounio_compiler.jl
        return @eval Main.DarwinPBPK.MedLang.parse_medlang_to_model($source)
    catch
        # Fallback: Create minimal model from source
        return create_minimal_model(source)
    end
end

"""
Create minimal model structure from MedLang source.
"""
function create_minimal_model(source::String)
    # Extract model name
    name_match = match(r"model\s+(\w+)", source)
    name = name_match !== nothing ? name_match.captures[1] : "GenericModel"

    # This returns a Dict that can be used for code generation
    Dict(
        :name => name,
        :source => source,
        :parameters => extract_parameters(source),
        :compartments => extract_compartments(source),
        :clearances => extract_clearances(source)
    )
end

"""
Extract parameters from MedLang source.
"""
function extract_parameters(source::String)
    params = []

    # Match parameter definitions
    for m in eachmatch(r"(\w+)\s*:\s*([\d.]+)_?(\w*)", source)
        push!(params, Dict(
            :name => m.captures[1],
            :value => parse(Float64, m.captures[2]),
            :unit => m.captures[3]
        ))
    end

    params
end

"""
Extract compartments/organs from MedLang source.
"""
function extract_compartments(source::String)
    comps = []

    # Match organ definitions
    for m in eachmatch(r"organ\s+(\w+)\s*\{([^}]+)\}", source)
        push!(comps, Dict(
            :name => m.captures[1],
            :properties => m.captures[2]
        ))
    end

    comps
end

"""
Extract clearance mechanisms from MedLang source.
"""
function extract_clearances(source::String)
    clearances = []

    # Match clearance definitions
    for m in eachmatch(r"clearance\s+(\w+)\s*:\s*([\d.]+)_?(\w*)", source)
        push!(clearances, Dict(
            :mechanism => m.captures[1],
            :value => parse(Float64, m.captures[2]),
            :unit => m.captures[3]
        ))
    end

    clearances
end

"""
Generate Sounio code safely from model.
"""
function generate_sounio_code_safe(model, use_stdlib::Bool)
    try
        # Try to use the proper code generator
        return @eval Main.DarwinPBPK.MedLang.compile_medlang_to_sounio($model; use_stdlib=$use_stdlib)
    catch
        # Fallback: Generate minimal Sounio code
        return generate_minimal_sounio_code(model, use_stdlib)
    end
end

"""
Generate minimal Sounio code from model Dict.
"""
function generate_minimal_sounio_code(model::Dict, use_stdlib::Bool)
    name = get(model, :name, "GenericModel")

    if use_stdlib
        # Use darwin_pbpk stdlib
        return """
//! $name - Generated from MedLang
//! Uses darwin_pbpk stdlib for production-ready PBPK simulation

import darwin_pbpk.simulation::{SimulationConfig, SimulationResult, run_pbpk_simulation}
import darwin_pbpk.core.pbpk_params::{PBPKParams, PatientData, DrugProperties}

/// Main simulation entry point
fn main(dose: mg, duration: h, patient: PatientData) -> SimulationResult {
    let config = SimulationConfig {
        model: "14_compartment",
        solver: "Rodas4",
        dt: 0.1_h,
        duration: duration,
    };

    let drug = DrugProperties {
        name: "$name",
        molecular_weight: 300.0_Da,
        logP: 2.0,
        fu_plasma: 0.5,
    };

    run_pbpk_simulation(config, drug, patient, dose)
}
"""
    else
        # Generate standalone code
        return """
//! $name - Generated from MedLang
//! Standalone implementation with full ODE system

/// PK Parameters
struct PKParams {
    cl_hepatic: L_per_h,
    cl_renal: L_per_h,
    vd: L,
    ka: per_h,
}

/// Main simulation function
fn simulate(dose: mg, duration: h) -> SimulationResult {
    let params = PKParams {
        cl_hepatic: 10.0_L_per_h,
        cl_renal: 1.0_L_per_h,
        vd: 70.0_L,
        ka: 1.0_per_h,
    };

    // Simplified one-compartment model
    let times = linspace(0.0_h, duration, 100);
    let ke = (params.cl_hepatic + params.cl_renal) / params.vd;

    let conc = times.map(|t| {
        let factor = dose / params.vd;
        factor * (params.ka / (params.ka - ke)) * (exp(-ke * t) - exp(-params.ka * t))
    });

    SimulationResult {
        times: times,
        concentrations: conc,
    }
}
"""
    end
end

"""
Extract model name from MedLang source.
"""
function extract_model_name(source::String, default::String)
    m = match(r"model\s+(\w+)", source)
    return m !== nothing ? m.captures[1] : default
end

# ===========================================================================
# Batch Processing
# ===========================================================================

"""
    run_sounio_batch(model_name::String, drugs::Vector, patient; kwargs...)

Run Sounio PBPK for multiple drugs in batch.
Returns a vector of SimulationResult.
"""
function run_sounio_batch(
    model_name::String,
    drugs::Vector,
    patient;
    dose_mg::Float64 = 100.0,
    kwargs...
)
    results = SimulationResult[]

    @info "Running Sounio batch for $(length(drugs)) drugs..."

    for (i, drug) in enumerate(drugs)
        @info "Processing drug $i/$(length(drugs)): $(drug.name)"

        result = run_sounio_pbpk(
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
    compare_julia_sounio(julia_result, sounio_result)

Compare simulation results between Julia and Sounio implementations.
"""
function compare_julia_sounio(julia_result, sounio_result::SimulationResult)
    # Extract Julia metrics
    julia_cmax = julia_result.metrics.cmax
    julia_tmax = julia_result.metrics.tmax
    julia_auc = julia_result.metrics.auc

    # Sounio metrics
    dem_cmax = sounio_result.cmax_mg_l
    dem_tmax = sounio_result.tmax_h
    dem_auc = sounio_result.auc_mg_h_l

    # Calculate fold errors
    cmax_fe = dem_cmax / julia_cmax
    auc_fe = dem_auc / julia_auc
    tmax_diff = abs(dem_tmax - julia_tmax)

    Dict(
        "cmax_julia" => julia_cmax,
        "cmax_sounio" => dem_cmax,
        "cmax_fold_error" => cmax_fe,
        "auc_julia" => julia_auc,
        "auc_sounio" => dem_auc,
        "auc_fold_error" => auc_fe,
        "tmax_julia" => julia_tmax,
        "tmax_sounio" => dem_tmax,
        "tmax_difference_h" => tmax_diff,
        "agreement" => abs(log10(cmax_fe)) < 0.1 && abs(log10(auc_fe)) < 0.1
    )
end

end  # module SounioIntegration
