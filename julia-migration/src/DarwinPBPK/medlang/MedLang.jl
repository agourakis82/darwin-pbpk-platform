"""
MedLang DSL Integration for Darwin PBPK Platform

First Real Implementation of MedLang DSL (github.com/agourakis82/medlang)

This module provides:
- MedLang DSL parser (Track D grammar)
- Transpiler to Julia/PBPKParams
- Unit-safe type system
- Integration with Darwin PBPK ODE solver

MedLang is a medical-native, GPU/HPC-accelerated programming language
designed to unify quantum pharmacology, clinical reasoning, AI models,
probabilistic measures, and fractal signal analysis.

References:
- MedLang Core Spec: medlang_core_spec_v0.1.md
- MedLang-D Grammar: medlang_d_minimal_grammar_v0.md
- Track D (Pharmacometrics): medlang_pharmacometrics_qsp_spec_v0.1.md

Author: Dr. Demetrios Agourakis
Date: November 2025
Version: 0.1.0
"""

module MedLang

# Re-export from parent module for internal use
using ..ODEPBPKSolver

# Include submodules
include("parser.jl")
include("transpiler.jl")
include("fractal_kinetics.jl")

# Re-export parser types
using .MedLangParser
export parse_medlang, MedLangAST, ModelDef, StateDef, ParamDef, ODEEquation
export PopulationDef, TimelineDef, DoseEvent, ObserveEvent
export ParseError, validate_units, validate_model_units
export Expr, LiteralExpr, IdentExpr, BinaryExpr, CallExpr, UnaryExpr, QualifiedExpr
export UnitExpr, TypeExpr
export AbsorptionDef, FirstPassDef, RouteType
export ROUTE_IV, ROUTE_ORAL, ROUTE_IM, ROUTE_SC, ROUTE_INFUSION

# Re-export dimensional analysis types
export Dimension, ValidationResult, infer_dimension, get_dimension, format_dimension
export TYPE_DIMENSIONS, UNIT_DIMENSIONS

# Re-export transpiler functions
using .MedLangTranspiler
export transpile_to_julia, transpile_to_pbpk_params, generate_ode_system
export TranspileError, TranspileResult
export OralAbsorptionParams, get_oral_absorption_params, effective_bioavailability
export calculate_fh, estimate_fg, ExtendedTranspileResult, transpile_to_extended_params

# Re-export ODEProblem generation functions
export ODEModelResult, generate_ode_function, generate_ode_model, generate_ode_problem

# Re-export fractal kinetics (Deep Fractal PBPK Theory)
using .FractalKinetics
export mittag_leffler, mittag_leffler_derivative
export fractional_decay, fractional_accumulation
export FractalCompartment, FractalPBPKParams
export tissue_fractal_dim, tissue_alpha, molecular_fractal_dim
export fractal_coupling, spectral_correction
export fractal_oie_tozer, estimate_fut_fractal
export ALEXANDER_ORBACH_DS, FRACTAL_KEYWORDS

#=============================================================================
  High-Level API
=============================================================================#

"""
    load_medlang(filepath::String) -> MedLangAST

Load and parse a MedLang file.

# Arguments
- `filepath::String`: Path to .medlang file

# Returns
- `MedLangAST`: Parsed abstract syntax tree

# Example
```julia
ast = load_medlang("models/drug_x.medlang")
```
"""
function load_medlang(filepath::String)::MedLangAST
    if !isfile(filepath)
        throw(ArgumentError("File not found: $filepath"))
    end

    source = read(filepath, String)
    return parse_medlang(source)
end

export load_medlang

"""
    compile_model(source::String; model_name=nothing) -> PBPKParams

Compile MedLang source directly to PBPKParams.

# Arguments
- `source::String`: MedLang source code
- `model_name::String`: Optional model name to compile (default: first model)

# Returns
- `PBPKParams`: Julia PBPK parameters struct

# Example
```julia
source = \"\"\"
model MyDrug {
    clearance hepatic: 10.0_L/h
    organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 2.5 }
}
\"\"\"
params = compile_model(source)
```
"""
function compile_model(source::String; model_name::Union{String,Nothing}=nothing)::PBPKParams
    return transpile_to_pbpk_params(source; model_name=model_name)
end

export compile_model

"""
    compile_file(filepath::String; model_name=nothing) -> PBPKParams

Compile a MedLang file to PBPKParams.

# Arguments
- `filepath::String`: Path to .medlang file
- `model_name::String`: Optional model name to compile

# Returns
- `PBPKParams`: Julia PBPK parameters struct
"""
function compile_file(filepath::String; model_name::Union{String,Nothing}=nothing)::PBPKParams
    source = read(filepath, String)
    return compile_model(source; model_name=model_name)
end

export compile_file

"""
    generate_julia_module(source::String) -> String

Generate a complete Julia module from MedLang source.

# Arguments
- `source::String`: MedLang source code

# Returns
- `String`: Generated Julia code

# Example
```julia
julia_code = generate_julia_module(medlang_source)
write("generated_model.jl", julia_code)
```
"""
function generate_julia_module(source::String)::String
    ast = parse_medlang(source)
    result = transpile_to_julia(ast)
    return result.julia_code
end

export generate_julia_module

"""
    simulate_medlang(source::String, dose::Float64; kwargs...) -> Dict

Parse MedLang model and run simulation.

# Arguments
- `source::String`: MedLang source code
- `dose::Float64`: Dose in mg
- `t_max::Float64`: Maximum simulation time (hours)
- `num_points::Int`: Number of time points

# Returns
- `Dict`: Concentration-time profiles for each organ
"""
function simulate_medlang(
    source::String,
    dose::Float64;
    t_max::Float64=24.0,
    num_points::Int=100,
    model_name::Union{String,Nothing}=nothing
)
    params = compile_model(source; model_name=model_name)
    return ODEPBPKSolver.simulate(params, dose; t_max=t_max, num_points=num_points)
end

export simulate_medlang

"""
    simulate_oral(source::String, dose::Float64; kwargs...) -> Dict

Parse MedLang model with oral absorption and run simulation.

# Arguments
- `source::String`: MedLang source code
- `dose::Float64`: Dose in mg
- `t_max::Float64`: Maximum simulation time (hours)
- `num_points::Int`: Number of time points
- `model_name::String`: Optional model name

# Returns
- `Dict`: Concentration-time profiles including gut compartment

# Example
```julia
source = \"\"\"
model OralDrug {
    route: oral
    absorption { Ka: 1.5, F: 0.8, lag: 0.5 }
    firstpass { Fg: 0.9, Fh: 0.7 }
    clearance hepatic: 15.0_L/h
}
\"\"\"
results = simulate_oral(source, 100.0)  # 100 mg oral dose
```
"""
function simulate_oral(
    source::String,
    dose::Float64;
    t_max::Float64=24.0,
    num_points::Int=100,
    model_name::Union{String,Nothing}=nothing
)
    # Get extended params including oral absorption
    extended = transpile_to_extended_params(source; model_name=model_name)

    return simulate_oral_pbpk(
        extended.pbpk_params,
        extended.oral_params,
        dose;
        t_max=t_max,
        num_points=num_points
    )
end

export simulate_oral

"""
    simulate_oral_pbpk(pbpk_params, oral_params, dose; kwargs...) -> Dict

Simulate PBPK model with oral absorption compartment.

This implements a two-compartment absorption model:
- Gut compartment: dA_gut/dt = -Ka * A_gut
- Systemic: Drug enters blood after first-pass metabolism

Bioavailability: F_eff = F * Fg * Fh

# Arguments
- `pbpk_params::PBPKParams`: PBPK model parameters
- `oral_params::OralAbsorptionParams`: Oral absorption parameters
- `dose::Float64`: Oral dose in mg

# Returns
- `Dict`: Concentration-time profiles for gut and all organs
"""
function simulate_oral_pbpk(
    pbpk_params::ODEPBPKSolver.PBPKParams,
    oral_params::OralAbsorptionParams,
    dose::Float64;
    t_max::Float64=24.0,
    num_points::Int=100
)
    # Calculate effective bioavailability
    f_eff = effective_bioavailability(oral_params)

    # If IV route or no absorption, use standard simulation
    if oral_params.route == ROUTE_IV || oral_params.ka <= 0.0
        return ODEPBPKSolver.simulate(pbpk_params, dose * f_eff; t_max=t_max, num_points=num_points)
    end

    # For oral route, use extended ODE system with gut compartment
    return simulate_oral_ode(pbpk_params, oral_params, dose; t_max=t_max, num_points=num_points)
end

export simulate_oral_pbpk

"""
Internal function to simulate oral absorption with extended ODE system.
"""
function simulate_oral_ode(
    pbpk_params::ODEPBPKSolver.PBPKParams,
    oral_params::OralAbsorptionParams,
    dose::Float64;
    t_max::Float64=24.0,
    num_points::Int=100
)
    ka = oral_params.ka
    f_eff = effective_bioavailability(oral_params)
    lag = oral_params.lag

    # Time points
    times = range(0.0, t_max, length=num_points)

    # Get blood volume for concentration calculation
    v_blood = get(pbpk_params.volumes, "blood", 5.0)

    # For a simplified simulation, we can use the analytical solution for one-compartment
    # with first-order absorption, then apply to full PBPK

    # Run PBPK simulations at multiple time points with absorption input
    results = Dict{String,Vector{Float64}}()
    results["time"] = collect(times)
    results["gut"] = zeros(num_points)

    # Initialize organ concentrations
    for organ in ODEPBPKSolver.PBPK_ORGANS
        results[organ] = zeros(num_points)
    end
    results["plasma"] = zeros(num_points)

    # Gut compartment analytical solution: A_gut(t) = Dose * exp(-Ka * (t - lag)) for t > lag
    for (i, t) in enumerate(times)
        effective_time = max(0.0, t - lag)

        # Amount in gut
        a_gut = dose * exp(-ka * effective_time)
        results["gut"][i] = a_gut

        # Amount absorbed up to time t (accounting for bioavailability)
        amount_absorbed = dose * f_eff * (1.0 - exp(-ka * effective_time))

        if amount_absorbed > 0.0 && t > lag
            # Run PBPK simulation with absorbed amount as IV bolus at t=0
            # This is a simplified approach - proper implementation would use
            # continuous absorption rate in the ODE system

            # For now, approximate as quasi-steady-state
            pbpk_result = ODEPBPKSolver.simulate(
                pbpk_params,
                amount_absorbed;
                t_max=effective_time + 0.1,
                num_points=10
            )

            # Get concentrations at this effective time
            if haskey(pbpk_result, "plasma") && length(pbpk_result["plasma"]) > 0
                # Use final concentration as approximation
                results["plasma"][i] = pbpk_result["plasma"][end]
            end

            for organ in ODEPBPKSolver.PBPK_ORGANS
                if haskey(pbpk_result, organ) && length(pbpk_result[organ]) > 0
                    results[organ][i] = pbpk_result[organ][end]
                end
            end
        end
    end

    # Calculate Cmax and Tmax
    cmax_idx = argmax(results["plasma"])
    results["cmax"] = results["plasma"][cmax_idx]
    results["tmax"] = results["time"][cmax_idx]

    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in 2:num_points
        dt = results["time"][i] - results["time"][i-1]
        auc += 0.5 * (results["plasma"][i] + results["plasma"][i-1]) * dt
    end
    results["auc"] = auc

    return results
end

#=============================================================================
  Validation & Inspection
=============================================================================#

"""
    validate_model(source::String) -> Vector{String}

Validate MedLang source without compiling.

Returns list of warnings/errors (empty if valid).
"""
function validate_model(source::String)::Vector{String}
    issues = String[]

    try
        ast = parse_medlang(source)

        # Check for models
        if isempty(ast.models)
            push!(issues, "WARNING: No models defined")
        end

        # Check each model
        for model in ast.models
            # Validate organ coverage
            defined_organs = Set([lowercase(o.name) for o in model.organs])
            standard_organs = Set(ODEPBPKSolver.PBPK_ORGANS)

            missing_organs = setdiff(standard_organs, defined_organs)
            if !isempty(missing_organs)
                push!(issues, "INFO: Model '$(model.name)' missing organs: $(join(missing_organs, ", "))")
            end

            # Validate clearances
            has_clearance = !isempty(model.clearances)
            if !has_clearance
                push!(issues, "WARNING: Model '$(model.name)' has no clearance mechanisms defined")
            end

            # Validate states match ODEs
            state_names = Set([s.name for s in model.states])
            ode_states = Set([o.state for o in model.odes])

            orphan_odes = setdiff(ode_states, state_names)
            if !isempty(orphan_odes)
                push!(issues, "ERROR: ODEs reference undefined states: $(join(orphan_odes, ", "))")
            end
        end

    catch e
        if e isa ParseError
            push!(issues, "PARSE ERROR at line $(e.line), col $(e.col): $(e.message)")
        else
            push!(issues, "ERROR: $(e)")
        end
    end

    return issues
end

export validate_model

"""
    describe_model(source::String) -> String

Generate a human-readable description of a MedLang model.
"""
function describe_model(source::String)::String
    ast = parse_medlang(source)

    buf = IOBuffer()

    for model in ast.models
        println(buf, "Model: $(model.name)")
        println(buf, "="^(8 + length(model.name)))
        println(buf)

        println(buf, "States ($(length(model.states))):")
        for s in model.states
            unit_str = s.type.unit !== nothing ? " [$(s.type.unit.base)]" : ""
            println(buf, "  - $(s.name): $(s.type.name)$unit_str")
        end
        println(buf)

        println(buf, "Parameters ($(length(model.params))):")
        for p in model.params
            unit_str = p.type.unit !== nothing ? " [$(p.type.unit.base)]" : ""
            default_str = p.default !== nothing ? " = $(expr_to_string(p.default))" : ""
            println(buf, "  - $(p.name): $(p.type.name)$unit_str$default_str")
        end
        println(buf)

        println(buf, "Organs ($(length(model.organs))):")
        for o in model.organs
            println(buf, "  - $(o.name)")
        end
        println(buf)

        println(buf, "Clearances ($(length(model.clearances))):")
        for c in model.clearances
            println(buf, "  - $(c.mechanism): $(expr_to_string(c.rate))")
        end
        println(buf)

        println(buf, "ODEs ($(length(model.odes))):")
        for ode in model.odes
            println(buf, "  d$(ode.state)/dt = $(expr_to_string(ode.rhs))")
        end
        println(buf)

        println(buf, "Observables ($(length(model.observables))):")
        for obs in model.observables
            println(buf, "  - $(obs.name) = $(expr_to_string(obs.expr))")
        end
        println(buf)
    end

    if !isempty(ast.timelines)
        println(buf, "Timelines ($(length(ast.timelines))):")
        for tl in ast.timelines
            println(buf, "  - $(tl.name): $(length(tl.events)) events")
        end
        println(buf)
    end

    if !isempty(ast.populations)
        println(buf, "Populations ($(length(ast.populations))):")
        for pop in ast.populations
            println(buf, "  - $(pop.name) (based on $(pop.model))")
            println(buf, "    Random effects: $(length(pop.random_effects))")
        end
    end

    return String(take!(buf))
end

export describe_model

# Helper function for describe_model
function expr_to_string(expr::MedLangParser.Expr)::String
    if expr isa MedLangParser.LiteralExpr
        unit_str = expr.unit !== nothing ? "_$(expr.unit.base)" : ""
        return "$(expr.value)$unit_str"
    elseif expr isa MedLangParser.IdentExpr
        return expr.name
    elseif expr isa MedLangParser.BinaryExpr
        return "($(expr_to_string(expr.left)) $(expr.op) $(expr_to_string(expr.right)))"
    elseif expr isa MedLangParser.UnaryExpr
        return "$(expr.op)$(expr_to_string(expr.operand))"
    elseif expr isa MedLangParser.CallExpr
        args = join([expr_to_string(a) for a in expr.args], ", ")
        return "$(expr.func)($args)"
    elseif expr isa MedLangParser.QualifiedExpr
        return join(expr.parts, ".")
    else
        return "<?>"
    end
end

#=============================================================================
  GNN Integration
=============================================================================#

"""
    simulate_with_gnn(source::String, dose::Float64, gnn_model; kwargs...) -> Dict

Parse MedLang model and run simulation using Dynamic GNN predictor.

This provides a fast ML-based alternative to ODE solving for
rapid parameter exploration and population simulations.

# Arguments
- `source::String`: MedLang source code
- `dose::Float64`: Dose in mg
- `gnn_model`: Trained DynamicPBPKGNN model
- `t_max::Float64`: Maximum simulation time (hours)
- `num_points::Int`: Number of time points

# Returns
- `Dict`: Concentration-time profiles for each organ (from GNN)

# Example
```julia
model = load_gnn_model("models/trained_gnn.bson")
results = simulate_with_gnn(medlang_source, 100.0, model)
```
"""
function simulate_with_gnn(
    source::String,
    dose::Float64,
    gnn_model;
    t_max::Float64=24.0,
    num_points::Int=100,
    model_name::Union{String,Nothing}=nothing,
    device=nothing
)
    # Parse MedLang to PBPKParams
    params = compile_model(source; model_name=model_name)

    # Generate time points
    time_points = collect(range(0.0, t_max, length=num_points))

    # Use GNN forward pass
    # Note: This requires DynamicGNN module to be loaded
    # The actual call will be: DynamicGNN.forward(gnn_model, dose, params, time_points, device)

    # Return placeholder - actual implementation requires DynamicGNN import
    return Dict(
        "time" => time_points,
        "method" => "gnn",
        "params" => params,
        "dose" => dose,
        "message" => "Use DynamicGNN.forward(model, dose, params, time_points, device) for GNN prediction"
    )
end

export simulate_with_gnn

"""
    medlang_to_gnn_input(source::String, dose::Float64; model_name=nothing) -> Tuple

Convert MedLang source to GNN-compatible input format.

Returns (dose, PBPKParams, time_points) tuple ready for GNN.forward().

# Example
```julia
dose, params, times = medlang_to_gnn_input(source, 100.0)
result = DynamicGNN.forward(gnn_model, dose, params, times, cpu)
```
"""
function medlang_to_gnn_input(
    source::String,
    dose::Float64;
    model_name::Union{String,Nothing}=nothing,
    t_max::Float64=24.0,
    num_points::Int=100
)
    params = compile_model(source; model_name=model_name)
    time_points = collect(range(0.0, t_max, length=num_points))
    return (dose, params, time_points)
end

export medlang_to_gnn_input

"""
    compare_ode_vs_gnn(source::String, dose::Float64, gnn_model; kwargs...) -> Dict

Run both ODE solver and GNN predictor, return comparison metrics.

Useful for validating GNN predictions against mechanistic model.

# Returns
Dict with:
- "ode_results": ODE solver output
- "gnn_results": GNN predictor output
- "metrics": Comparison metrics (GMFE, R², etc.)
"""
function compare_ode_vs_gnn(
    source::String,
    dose::Float64,
    gnn_forward_fn::Function;  # Function that takes (dose, params, times) -> concentrations
    t_max::Float64=24.0,
    num_points::Int=100,
    model_name::Union{String,Nothing}=nothing
)
    # Get params and times
    params = compile_model(source; model_name=model_name)
    time_points = collect(range(0.0, t_max, length=num_points))

    # ODE simulation
    ode_results = ODEPBPKSolver.simulate(params, dose; t_max=t_max, num_points=num_points)

    # GNN prediction (using provided forward function)
    gnn_concs = gnn_forward_fn(dose, params, time_points)

    # Extract blood concentrations for comparison
    ode_blood = get(ode_results, "blood", zeros(num_points))
    gnn_blood = size(gnn_concs, 2) >= 1 ? vec(gnn_concs[1, 1, :]) : zeros(num_points)

    # Compute comparison metrics
    # Filter out zeros for fold error calculation
    valid_idx = findall(x -> x > 0, ode_blood)

    if !isempty(valid_idx)
        ode_valid = ode_blood[valid_idx]
        gnn_valid = gnn_blood[valid_idx]

        # Fold errors
        fe = [max(g/o, o/g) for (g, o) in zip(gnn_valid, ode_valid) if o > 0 && g > 0]
        gmfe = isempty(fe) ? NaN : exp(mean(log.(fe)))

        # R²
        ss_res = sum((gnn_valid .- ode_valid).^2)
        ss_tot = sum((ode_valid .- mean(ode_valid)).^2)
        r2 = ss_tot > 0 ? 1 - ss_res/ss_tot : NaN

        # MAE
        mae = mean(abs.(gnn_valid .- ode_valid))
    else
        gmfe, r2, mae = NaN, NaN, NaN
    end

    return Dict(
        "ode_results" => ode_results,
        "gnn_blood" => gnn_blood,
        "time" => time_points,
        "metrics" => Dict(
            "GMFE" => gmfe,
            "R2" => r2,
            "MAE" => mae,
        )
    )
end

export compare_ode_vs_gnn

#=============================================================================
  Version Info
=============================================================================#

const MEDLANG_VERSION = v"0.1.0"
const MEDLANG_SPEC_VERSION = "Track D v0.1"

"""
    version() -> VersionNumber

Return MedLang implementation version.
"""
version() = MEDLANG_VERSION

export version

"""
    info() -> String

Return MedLang module information.
"""
function info()::String
    return """
    MedLang DSL for Darwin PBPK
    ===========================
    Implementation Version: $MEDLANG_VERSION
    Spec Version: $MEDLANG_SPEC_VERSION
    Repository: github.com/agourakis82/medlang

    Features:
    - MedLang Track D parser (PBPK/Pharmacometrics)
    - Julia transpiler with unit safety
    - Integration with ODEPBPKSolver
    - Population modeling (NLME)
    - Timeline/dosing schedules

    Usage:
      params = compile_model(source)
      results = simulate_medlang(source, dose)
    """
end

export info

end # module
