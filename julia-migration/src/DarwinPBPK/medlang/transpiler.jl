"""
MedLang to Julia Transpiler for Darwin PBPK

Transpiles MedLang AST to Julia code and PBPKParams structs.
Implements the compilation path: MedLang → CIR → Julia

Features:
- AST → PBPKParams struct generation
- Unit validation and conversion
- ODE system generation
- Population model expansion
- Timeline → dosing schedule
- Oral absorption and first-pass metabolism

Author: Dr. Demetrios Agourakis
Date: November 2025
"""

module MedLangTranspiler

using ..MedLangParser: parse_medlang, MedLangAST, ModelDef, StateDef, ParamDef
using ..MedLangParser: ODEEquation, PopulationDef, TimelineDef, DoseEvent, ObserveEvent
using ..MedLangParser: Expr, LiteralExpr, IdentExpr, BinaryExpr, UnaryExpr, CallExpr, QualifiedExpr
using ..MedLangParser: TypeExpr, UnitExpr, OrganDef, ClearanceDef, ObsDef, RandomEffectDef
using ..MedLangParser: AbsorptionDef, FirstPassDef, RouteType
using ..MedLangParser: ROUTE_IV, ROUTE_ORAL, ROUTE_IM, ROUTE_SC, ROUTE_INFUSION
using ..ODEPBPKSolver: PBPKParams, PBPK_ORGANS, NUM_ORGANS

export transpile_to_julia, transpile_to_pbpk_params, generate_ode_system
export TranspileError, TranspileResult
export OralAbsorptionParams, get_oral_absorption_params, effective_bioavailability
export calculate_fh, estimate_fg, ExtendedTranspileResult, transpile_to_extended_params

#=============================================================================
  Transpile Error
=============================================================================#

struct TranspileError <: Exception
    message::String
    context::String
end

Base.showerror(io::IO, e::TranspileError) = print(io, "TranspileError: $(e.message)")

#=============================================================================
  Transpile Result
=============================================================================#

struct TranspileResult
    julia_code::String
    pbpk_params::Union{PBPKParams, Nothing}
    warnings::Vector{String}
end

#=============================================================================
  Oral Absorption Parameters
=============================================================================#

"""
Parameters for oral absorption and first-pass metabolism.

# Fields
- `ka::Float64`: Absorption rate constant (1/h)
- `f::Float64`: Bioavailability (0-1), default 1.0
- `lag::Float64`: Lag time (h), default 0.0
- `fg::Float64`: Fraction escaping gut metabolism (0-1), default 1.0
- `fh::Float64`: Fraction escaping hepatic first-pass (0-1), default 1.0
- `route::RouteType`: Route of administration

Bioavailability is calculated as: F = Fa × Fg × Fh
Where:
- Fa = fraction absorbed (typically assumed 1.0 for complete absorption)
- Fg = fraction escaping gut wall metabolism
- Fh = fraction escaping hepatic first-pass metabolism (= 1 - ERH)
- ERH = hepatic extraction ratio = CLH / QH
"""
struct OralAbsorptionParams
    ka::Float64       # Absorption rate constant (1/h)
    f::Float64        # Bioavailability (fraction)
    lag::Float64      # Lag time (h)
    fg::Float64       # Gut availability (fraction)
    fh::Float64       # Hepatic availability (fraction)
    route::RouteType  # Route of administration
end

# Default constructor for IV (no absorption)
OralAbsorptionParams() = OralAbsorptionParams(0.0, 1.0, 0.0, 1.0, 1.0, ROUTE_IV)

# Constructor for oral dosing with default first-pass
function OralAbsorptionParams(ka::Float64; f=1.0, lag=0.0, fg=1.0, fh=1.0)
    OralAbsorptionParams(ka, f, lag, fg, fh, ROUTE_ORAL)
end

"""
Calculate effective bioavailability: F_eff = f × fg × fh
"""
function effective_bioavailability(params::OralAbsorptionParams)::Float64
    return params.f * params.fg * params.fh
end

"""
Calculate hepatic availability from clearance and blood flow.
Fh = 1 - ERH = 1 - (CLH / QH)

# Arguments
- `cl_hepatic::Float64`: Hepatic clearance (L/h)
- `q_hepatic::Float64`: Hepatic blood flow (L/h), typically ~90 L/h for 70kg adult

# Returns
- `Float64`: Hepatic availability (0-1)
"""
function calculate_fh(cl_hepatic::Float64, q_hepatic::Float64=90.0)::Float64
    erh = min(1.0, cl_hepatic / q_hepatic)  # Extraction ratio can't exceed 1
    return 1.0 - erh
end

"""
Estimate gut availability (Fg) from intestinal metabolism.
For CYP3A4 substrates, Fg can range from 0.3 to 1.0.

# Arguments
- `cyp3a4_substrate::Bool`: Whether drug is CYP3A4 substrate
- `gut_metabolism_rate::Float64`: Optional explicit gut metabolism rate

# Returns
- `Float64`: Gut availability (0-1)
"""
function estimate_fg(; cyp3a4_substrate::Bool=false, gut_metabolism_rate::Float64=0.0)::Float64
    if gut_metabolism_rate > 0.0
        # Use explicit metabolism rate
        return max(0.1, 1.0 - gut_metabolism_rate)
    elseif cyp3a4_substrate
        # CYP3A4 substrates typically have Fg = 0.4-0.8
        return 0.6  # Conservative estimate
    else
        return 1.0  # No significant gut metabolism
    end
end

#=============================================================================
  Unit Conversion
=============================================================================#

# Standard unit conversions to base units
const UNIT_CONVERSIONS = Dict{String, Tuple{Float64, String}}(
    # Mass → mg
    "g" => (1000.0, "mg"),
    "kg" => (1e6, "mg"),
    "ug" => (0.001, "mg"),
    "ng" => (1e-6, "mg"),
    "pg" => (1e-9, "mg"),
    "mg" => (1.0, "mg"),

    # Volume → L
    "mL" => (0.001, "L"),
    "uL" => (1e-6, "L"),
    "dL" => (0.1, "L"),
    "L" => (1.0, "L"),

    # Time → h
    "min" => (1/60, "h"),
    "s" => (1/3600, "h"),
    "d" => (24.0, "h"),
    "h" => (1.0, "h"),

    # Clearance → L/h
    "L/h" => (1.0, "L/h"),
    "mL/min" => (0.06, "L/h"),  # 0.001 * 60
    "L/h/kg" => (1.0, "L/h"),   # Per kg, will multiply by weight

    # Rate → 1/h
    "1/h" => (1.0, "1/h"),
    "1/min" => (60.0, "1/h"),
)

function convert_unit(value::Float64, from_unit::String, to_unit::String)::Float64
    if from_unit == to_unit
        return value
    end

    if haskey(UNIT_CONVERSIONS, from_unit)
        factor, base = UNIT_CONVERSIONS[from_unit]
        base_value = value * factor

        if base == to_unit
            return base_value
        end

        # Try to convert from base to target
        if haskey(UNIT_CONVERSIONS, to_unit)
            to_factor, to_base = UNIT_CONVERSIONS[to_unit]
            if to_base == base
                return base_value / to_factor
            end
        end
    end

    @warn "Cannot convert $from_unit to $to_unit, returning original value"
    return value
end

#=============================================================================
  Expression Evaluation
=============================================================================#

"""
Evaluate a constant expression at transpile time.
Returns (value, unit) tuple.
"""
function eval_const_expr(expr::Expr, env::Dict{String, Tuple{Float64, String}})::Tuple{Float64, Union{String, Nothing}}
    if expr isa LiteralExpr
        if expr.unit !== nothing
            return (expr.value, expr.unit.base)
        else
            return (Float64(expr.value), nothing)
        end
    elseif expr isa IdentExpr
        if haskey(env, expr.name)
            return env[expr.name]
        else
            throw(TranspileError("Unknown identifier: $(expr.name)", "eval_const_expr"))
        end
    elseif expr isa BinaryExpr
        left_val, left_unit = eval_const_expr(expr.left, env)
        right_val, right_unit = eval_const_expr(expr.right, env)

        result = if expr.op == :+
            left_val + right_val
        elseif expr.op == :-
            left_val - right_val
        elseif expr.op == :*
            left_val * right_val
        elseif expr.op == :/
            left_val / right_val
        elseif expr.op == :^
            left_val ^ right_val
        else
            throw(TranspileError("Unknown operator: $(expr.op)", "eval_const_expr"))
        end

        # Unit inference (simplified)
        result_unit = if expr.op in (:+, :-)
            left_unit  # Same units required
        elseif expr.op == :*
            # Compound unit
            if left_unit !== nothing && right_unit !== nothing
                "$(left_unit)*$(right_unit)"
            else
                left_unit !== nothing ? left_unit : right_unit
            end
        elseif expr.op == :/
            if left_unit !== nothing && right_unit !== nothing
                "$(left_unit)/$(right_unit)"
            else
                left_unit
            end
        else
            nothing
        end

        return (result, result_unit)
    elseif expr isa UnaryExpr
        val, unit = eval_const_expr(expr.operand, env)
        if expr.op == :-
            return (-val, unit)
        else
            return (val, unit)
        end
    elseif expr isa CallExpr
        # Built-in functions
        if expr.func == "exp"
            arg_val, _ = eval_const_expr(expr.args[1], env)
            return (exp(arg_val), nothing)
        elseif expr.func == "log"
            arg_val, _ = eval_const_expr(expr.args[1], env)
            return (log(arg_val), nothing)
        elseif expr.func == "sqrt"
            arg_val, unit = eval_const_expr(expr.args[1], env)
            return (sqrt(arg_val), unit)
        elseif expr.func == "pow"
            base_val, unit = eval_const_expr(expr.args[1], env)
            exp_val, _ = eval_const_expr(expr.args[2], env)
            return (base_val ^ exp_val, unit)
        else
            throw(TranspileError("Unknown function: $(expr.func)", "eval_const_expr"))
        end
    else
        throw(TranspileError("Cannot evaluate expression type: $(typeof(expr))", "eval_const_expr"))
    end
end

#=============================================================================
  Expression to Julia Code
=============================================================================#

function expr_to_julia(expr::Expr)::String
    if expr isa LiteralExpr
        if expr.value isa String
            return "\"$(expr.value)\""
        else
            return string(expr.value)
        end
    elseif expr isa IdentExpr
        return expr.name
    elseif expr isa QualifiedExpr
        return join(expr.parts, ".")
    elseif expr isa BinaryExpr
        left = expr_to_julia(expr.left)
        right = expr_to_julia(expr.right)
        op = string(expr.op)
        return "($left $op $right)"
    elseif expr isa UnaryExpr
        operand = expr_to_julia(expr.operand)
        return "$(expr.op)$operand"
    elseif expr isa CallExpr
        args = join([expr_to_julia(a) for a in expr.args], ", ")
        return "$(expr.func)($args)"
    else
        return "???"
    end
end

#=============================================================================
  Model Transpilation
=============================================================================#

"""
Transpile a MedLang model definition to PBPKParams.
"""
function transpile_model_to_pbpk(model::ModelDef)::Tuple{PBPKParams, Vector{String}}
    warnings = String[]

    # Build environment from parameters
    env = Dict{String, Tuple{Float64, String}}()
    for param in model.params
        if param.default !== nothing
            try
                val, unit = eval_const_expr(param.default, env)
                env[param.name] = (val, unit !== nothing ? unit : "")
            catch e
                push!(warnings, "Could not evaluate param $(param.name): $e")
            end
        end
    end

    # Extract volumes and blood flows from organ definitions
    volumes = Dict{String, Float64}()
    blood_flows = Dict{String, Float64}()
    partition_coeffs = Dict{String, Float64}()

    for organ in model.organs
        name = lowercase(organ.name)

        try
            vol_val, vol_unit = eval_const_expr(organ.volume, env)
            volumes[name] = vol_unit !== nothing ? convert_unit(vol_val, vol_unit, "L") : vol_val
        catch e
            push!(warnings, "Could not evaluate volume for $name: $e")
        end

        try
            flow_val, flow_unit = eval_const_expr(organ.blood_flow, env)
            blood_flows[name] = flow_unit !== nothing ? convert_unit(flow_val, flow_unit, "L/h") : flow_val
        catch e
            push!(warnings, "Could not evaluate blood flow for $name: $e")
        end

        try
            kp_val, _ = eval_const_expr(organ.partition_coeff, env)
            partition_coeffs[name] = kp_val
        catch e
            push!(warnings, "Could not evaluate Kp for $name: $e")
        end
    end

    # Extract clearances
    clearance_hepatic = 0.0
    clearance_renal = 0.0

    for cl in model.clearances
        try
            cl_val, cl_unit = eval_const_expr(cl.rate, env)
            cl_converted = cl_unit !== nothing ? convert_unit(cl_val, cl_unit, "L/h") : cl_val

            if cl.mechanism == :hepatic
                clearance_hepatic = cl_converted
            elseif cl.mechanism == :renal
                clearance_renal = cl_converted
            end
        catch e
            push!(warnings, "Could not evaluate clearance $(cl.mechanism): $e")
        end
    end

    # Also check parameters for clearances (alternative definition method)
    for param in model.params
        if param.default !== nothing
            name_lower = lowercase(param.name)
            try
                val, unit = eval_const_expr(param.default, env)
                val_converted = unit !== nothing ? convert_unit(val, unit, "L/h") : val

                if contains(name_lower, "cl") && contains(name_lower, "hepatic")
                    clearance_hepatic = val_converted
                elseif contains(name_lower, "cl") && contains(name_lower, "renal")
                    clearance_renal = val_converted
                elseif name_lower == "cl" && clearance_hepatic == 0.0
                    # Generic CL defaults to hepatic
                    clearance_hepatic = val_converted
                end
            catch
                # Ignore evaluation errors for params
            end
        end
    end

    # Create PBPKParams
    params = PBPKParams(
        volumes = volumes,
        blood_flows = blood_flows,
        clearance_hepatic = clearance_hepatic,
        clearance_renal = clearance_renal,
        partition_coeffs = partition_coeffs
    )

    return (params, warnings)
end

"""
Generate Julia ODE system code from MedLang model.
"""
function generate_ode_system(model::ModelDef)::String
    code = IOBuffer()

    # Function signature
    println(code, "function $(model.name)_ode!(du, u, p, t)")

    # State variable unpacking
    println(code, "    # State variables")
    for (i, state) in enumerate(model.states)
        println(code, "    $(state.name) = u[$i]")
    end
    println(code)

    # Parameter unpacking
    println(code, "    # Parameters")
    for param in model.params
        println(code, "    $(param.name) = p.$(param.name)")
    end
    println(code)

    # ODE equations
    println(code, "    # Differential equations")
    for (i, ode) in enumerate(model.odes)
        rhs = expr_to_julia(ode.rhs)
        println(code, "    du[$i] = $rhs")
    end
    println(code)

    println(code, "    return nothing")
    println(code, "end")

    return String(take!(code))
end

"""
Generate Julia struct for model parameters.
"""
function generate_param_struct(model::ModelDef)::String
    code = IOBuffer()

    println(code, "struct $(model.name)Params")
    for param in model.params
        type_str = "Float64"  # Default to Float64
        println(code, "    $(param.name)::$type_str")
    end
    println(code, "end")
    println(code)

    # Constructor with defaults
    println(code, "function $(model.name)Params(;")
    for (i, param) in enumerate(model.params)
        default = param.default !== nothing ? expr_to_julia(param.default) : "0.0"
        comma = i < length(model.params) ? "," : ""
        println(code, "    $(param.name) = $default$comma")
    end
    println(code, ")")
    println(code, "    return $(model.name)Params($(join([p.name for p in model.params], ", ")))")
    println(code, "end")

    return String(take!(code))
end

"""
Generate complete Julia module from MedLang AST.
"""
function transpile_to_julia(ast::MedLangAST)::TranspileResult
    code = IOBuffer()
    all_warnings = String[]

    println(code, "# Generated from MedLang DSL")
    println(code, "# Do not edit manually - regenerate from .medlang source")
    println(code, "")
    println(code, "module GeneratedPBPK")
    println(code, "")
    println(code, "using DifferentialEquations")
    println(code, "using StaticArrays")
    println(code, "")

    # Generate each model
    for model in ast.models
        println(code, "# ============ Model: $(model.name) ============")
        println(code, "")

        # Parameter struct
        println(code, generate_param_struct(model))
        println(code, "")

        # ODE system
        println(code, generate_ode_system(model))
        println(code, "")
    end

    # Generate timeline utilities
    if !isempty(ast.timelines)
        println(code, "# ============ Timelines ============")
        println(code, "")

        for timeline in ast.timelines
            println(code, "const $(timeline.name)_EVENTS = [")
            for event in timeline.events
                if event isa DoseEvent
                    time_str = expr_to_julia(event.time)
                    amount_str = expr_to_julia(event.amount)
                    println(code, "    (:dose, $time_str, $amount_str, \"$(event.target)\"),")
                elseif event isa ObserveEvent
                    time_str = expr_to_julia(event.time)
                    println(code, "    (:observe, $time_str, \"$(event.observable)\"),")
                end
            end
            println(code, "]")
            println(code, "")
        end
    end

    println(code, "end # module")

    return TranspileResult(String(take!(code)), nothing, all_warnings)
end

"""
Transpile MedLang source directly to PBPKParams.

# Arguments
- `source::String`: MedLang source code
- `model_name::String`: Name of the model to extract (default: first model)

# Returns
- `PBPKParams`: Julia PBPKParams struct

# Example
```julia
source = \"\"\"
model MyDrug {
    clearance hepatic: 10.0_L/h
    clearance renal: 2.5_L/h

    organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 2.5 }
    organ kidney { V: 0.31_L, Q: 60.0_L/h, Kp: 1.8 }
}
\"\"\"
params = transpile_to_pbpk_params(source)
```
"""
function transpile_to_pbpk_params(source::String; model_name::Union{String, Nothing}=nothing)::PBPKParams
    ast = parse_medlang(source)

    if isempty(ast.models)
        throw(TranspileError("No models found in source", "transpile_to_pbpk_params"))
    end

    model = if model_name !== nothing
        idx = findfirst(m -> m.name == model_name, ast.models)
        if idx === nothing
            throw(TranspileError("Model '$model_name' not found", "transpile_to_pbpk_params"))
        end
        ast.models[idx]
    else
        ast.models[1]
    end

    params, warnings = transpile_model_to_pbpk(model)

    for w in warnings
        @warn w
    end

    return params
end

#=============================================================================
  Oral Absorption Parameter Extraction
=============================================================================#

"""
Extract oral absorption parameters from a MedLang model.

# Arguments
- `model::ModelDef`: Parsed MedLang model
- `env::Dict`: Environment with evaluated parameters

# Returns
- `OralAbsorptionParams`: Oral absorption parameters
"""
function get_oral_absorption_params(model::ModelDef, env::Dict{String, Tuple{Float64, String}}=Dict{String, Tuple{Float64, String}}())::OralAbsorptionParams
    # Default values
    ka = 1.0    # Default Ka = 1.0 1/h
    f = 1.0     # Default F = 1.0 (complete bioavailability)
    lag = 0.0   # Default lag = 0 h
    fg = 1.0    # Default Fg = 1.0 (no gut metabolism)
    fh = 1.0    # Default Fh = 1.0 (no hepatic first-pass)
    route = model.route

    # Extract from absorption block if present
    if model.absorption !== nothing
        try
            ka_val, ka_unit = eval_const_expr(model.absorption.ka, env)
            # Convert to 1/h if needed
            if ka_unit !== nothing && ka_unit != "1/h"
                ka = convert_unit(ka_val, ka_unit, "1/h")
            else
                ka = ka_val
            end
        catch e
            @warn "Could not evaluate Ka: $e, using default 1.0 1/h"
        end

        if model.absorption.f !== nothing
            try
                f_val, _ = eval_const_expr(model.absorption.f, env)
                f = clamp(f_val, 0.0, 1.0)
            catch e
                @warn "Could not evaluate F: $e, using default 1.0"
            end
        end

        if model.absorption.lag !== nothing
            try
                lag_val, lag_unit = eval_const_expr(model.absorption.lag, env)
                if lag_unit !== nothing && lag_unit != "h"
                    lag = convert_unit(lag_val, lag_unit, "h")
                else
                    lag = lag_val
                end
            catch e
                @warn "Could not evaluate lag: $e, using default 0.0 h"
            end
        end
    end

    # Extract from firstpass block if present
    if model.firstpass !== nothing
        try
            fg_val, _ = eval_const_expr(model.firstpass.fg, env)
            fg = clamp(fg_val, 0.0, 1.0)
        catch e
            @warn "Could not evaluate Fg: $e, using default 1.0"
        end

        try
            fh_val, _ = eval_const_expr(model.firstpass.fh, env)
            fh = clamp(fh_val, 0.0, 1.0)
        catch e
            @warn "Could not evaluate Fh: $e, using default 1.0"
        end
    end

    # If route is oral but no absorption block, set reasonable defaults
    if route == ROUTE_ORAL && model.absorption === nothing
        ka = 1.0  # Typical oral Ka
    end

    return OralAbsorptionParams(ka, f, lag, fg, fh, route)
end

"""
Extended transpilation result including oral absorption parameters.
"""
struct ExtendedTranspileResult
    pbpk_params::PBPKParams
    oral_params::OralAbsorptionParams
    warnings::Vector{String}
end

"""
Transpile MedLang source to both PBPKParams and OralAbsorptionParams.

# Arguments
- `source::String`: MedLang source code
- `model_name::String`: Name of the model to extract (default: first model)

# Returns
- `ExtendedTranspileResult`: Contains both PBPKParams and OralAbsorptionParams
"""
function transpile_to_extended_params(source::String; model_name::Union{String, Nothing}=nothing)::ExtendedTranspileResult
    ast = parse_medlang(source)

    if isempty(ast.models)
        throw(TranspileError("No models found in source", "transpile_to_extended_params"))
    end

    model = if model_name !== nothing
        idx = findfirst(m -> m.name == model_name, ast.models)
        if idx === nothing
            throw(TranspileError("Model '$model_name' not found", "transpile_to_extended_params"))
        end
        ast.models[idx]
    else
        ast.models[1]
    end

    # Build environment from parameters
    env = Dict{String, Tuple{Float64, String}}()
    for param in model.params
        if param.default !== nothing
            try
                val, unit = eval_const_expr(param.default, env)
                env[param.name] = (val, unit !== nothing ? unit : "")
            catch
                # Ignore evaluation errors
            end
        end
    end

    pbpk_params, warnings = transpile_model_to_pbpk(model)
    oral_params = get_oral_absorption_params(model, env)

    return ExtendedTranspileResult(pbpk_params, oral_params, warnings)
end

end # module
