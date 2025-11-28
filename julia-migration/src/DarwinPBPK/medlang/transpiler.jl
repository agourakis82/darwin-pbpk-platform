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
# SOTA v0.2 Neural-Symbolic AST types
using ..MedLangParser: CompoundDef, NeuralNetSpec, NeuralODEDef, MechanisticODEDef
using ..MedLangParser: InferenceDef, PharmacodynamicsDef, TargetDef
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
    pbpk_params::Union{PBPKParams,Nothing}
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
const UNIT_CONVERSIONS = Dict{String,Tuple{Float64,String}}(
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
    "min" => (1 / 60, "h"),
    "s" => (1 / 3600, "h"),
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
function eval_const_expr(expr::Expr, env::Dict{String,Tuple{Float64,String}})::Tuple{Float64,Union{String,Nothing}}
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
            left_val^right_val
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
            return (base_val^exp_val, unit)
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
function transpile_model_to_pbpk(model::ModelDef)::Tuple{PBPKParams,Vector{String}}
    warnings = String[]

    # Build environment from parameters
    env = Dict{String,Tuple{Float64,String}}()
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
    volumes = Dict{String,Float64}()
    blood_flows = Dict{String,Float64}()
    partition_coeffs = Dict{String,Float64}()

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
        volumes=volumes,
        blood_flows=blood_flows,
        clearance_hepatic=clearance_hepatic,
        clearance_renal=clearance_renal,
        partition_coeffs=partition_coeffs
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
function transpile_to_pbpk_params(source::String; model_name::Union{String,Nothing}=nothing)::PBPKParams
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
function get_oral_absorption_params(model::ModelDef, env::Dict{String,Tuple{Float64,String}}=Dict{String,Tuple{Float64,String}}())::OralAbsorptionParams
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
function transpile_to_extended_params(source::String; model_name::Union{String,Nothing}=nothing)::ExtendedTranspileResult
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
    env = Dict{String,Tuple{Float64,String}}()
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

#=============================================================================
  DifferentialEquations.jl Integration - ODEProblem Generation
=============================================================================#

"""
Result of generating an ODE model from MedLang.

Contains everything needed to simulate the model with DifferentialEquations.jl.
"""
struct ODEModelResult
    ode_func::Function           # The ODE function f!(du, u, p, t)
    u0::Vector{Float64}          # Initial conditions
    params::NamedTuple           # Parameters as NamedTuple
    tspan::Tuple{Float64,Float64} # Default time span
    state_names::Vector{String}  # Names of state variables
    param_names::Vector{String}  # Names of parameters
    model_name::String           # Name of the model
end

"""
Evaluate MedLang expression at runtime with given context.
"""
function eval_expr_runtime(expr, ctx::Dict{String,Float64})::Float64
    if expr isa LiteralExpr
        return Float64(expr.value)

    elseif expr isa IdentExpr
        return get(ctx, expr.name, 0.0)

    elseif expr isa QualifiedExpr
        full_name = join(expr.parts, ".")
        return get(ctx, full_name, 0.0)

    elseif expr isa BinaryExpr
        left = eval_expr_runtime(expr.left, ctx)
        right = eval_expr_runtime(expr.right, ctx)

        if expr.op == :+
            return left + right
        elseif expr.op == :-
            return left - right
        elseif expr.op == :*
            return left * right
        elseif expr.op == :/
            return right != 0.0 ? left / right : 0.0
        elseif expr.op == :^
            return left^right
        end

    elseif expr isa UnaryExpr
        operand = eval_expr_runtime(expr.operand, ctx)
        if expr.op == :-
            return -operand
        end
        return operand

    elseif expr isa CallExpr
        args = [eval_expr_runtime(a, ctx) for a in expr.args]

        if expr.func == "exp"
            return exp(args[1])
        elseif expr.func == "log"
            return log(max(1e-300, args[1]))
        elseif expr.func == "sqrt"
            return sqrt(max(0.0, args[1]))
        elseif expr.func == "sin"
            return sin(args[1])
        elseif expr.func == "cos"
            return cos(args[1])
        elseif expr.func == "pow" && length(args) >= 2
            return args[1]^args[2]
        elseif expr.func == "abs"
            return abs(args[1])
        elseif expr.func == "max" && length(args) >= 2
            return max(args[1], args[2])
        elseif expr.func == "min" && length(args) >= 2
            return min(args[1], args[2])
        end
    end

    return 0.0
end

"""
    generate_ode_function(model::ModelDef) -> Function

Generate an ODE function from a MedLang model definition.

Returns a function with signature f!(du, u, p, t) suitable for DifferentialEquations.jl.
"""
function generate_ode_function(model::ModelDef)::Function
    # Build parameter name to index mapping
    param_names = [p.name for p in model.params]
    state_names = [s.name for s in model.states]

    # Pre-process ODE RHS expressions
    ode_exprs = [(ode.state, ode.rhs) for ode in model.odes]

    # Create the ODE function dynamically
    function ode_func!(du, u, p, t)
        # Build evaluation context
        ctx = Dict{String,Float64}()

        # Add state variables to context
        for (i, name) in enumerate(state_names)
            ctx[name] = u[i]
        end

        # Add parameters to context (p is a NamedTuple)
        for name in param_names
            ctx[name] = getfield(p, Symbol(name))
        end

        # Add time
        ctx["t"] = t

        # Evaluate each ODE equation
        for (i, (state, rhs)) in enumerate(ode_exprs)
            du[i] = eval_expr_runtime(rhs, ctx)
        end

        return nothing
    end

    return ode_func!
end

"""
    generate_ode_model(source::String; kwargs...) -> ODEModelResult

Generate complete ODE model information from MedLang source.

Returns all components needed to construct and customize an ODEProblem.

# Arguments
- `source::String`: MedLang source code
- `model_name::String`: Name of model to compile (default: first model)
- `dose::Float64`: Initial dose to override first state (default: 0.0, no override)

# Returns
- `ODEModelResult`: Contains ODE function, initial conditions, parameters, etc.
"""
function generate_ode_model(
    source::String;
    model_name::Union{String,Nothing}=nothing,
    dose::Float64=0.0
)::ODEModelResult
    ast = parse_medlang(source)

    if isempty(ast.models)
        throw(TranspileError("No models found in source", "generate_ode_model"))
    end

    # Select model
    model = if model_name !== nothing
        idx = findfirst(m -> m.name == model_name, ast.models)
        if idx === nothing
            throw(TranspileError("Model '$model_name' not found", "generate_ode_model"))
        end
        ast.models[idx]
    else
        ast.models[1]
    end

    # Build environment for parameter evaluation
    env = Dict{String,Tuple{Float64,String}}()
    for param in model.params
        if param.default !== nothing
            try
                val, unit = eval_const_expr(param.default, env)
                env[param.name] = (val, unit !== nothing ? unit : "")
            catch
                env[param.name] = (0.0, "")
            end
        end
    end

    # Extract initial conditions from states
    state_names = String[]
    u0 = Float64[]
    for state in model.states
        push!(state_names, state.name)
        if state.initial !== nothing
            try
                val, _ = eval_const_expr(state.initial, env)
                push!(u0, val)
            catch
                push!(u0, 0.0)
            end
        else
            push!(u0, 0.0)
        end
    end

    # Override first state with dose if specified
    if dose > 0.0 && !isempty(u0)
        u0[1] = dose
    end

    # Extract parameters as NamedTuple
    param_names = String[]
    param_values = Float64[]
    for param in model.params
        push!(param_names, param.name)
        val = get(env, param.name, (0.0, ""))[1]
        push!(param_values, val)
    end

    # Create NamedTuple for parameters
    param_symbols = tuple(Symbol.(param_names)...)
    params = NamedTuple{param_symbols}(tuple(param_values...))

    # Generate ODE function
    ode_func = generate_ode_function(model)

    return ODEModelResult(
        ode_func,
        u0,
        params,
        (0.0, 24.0),
        state_names,
        param_names,
        model.name
    )
end

"""
    generate_ode_problem(source::String; kwargs...) -> ODEProblem

Generate a DifferentialEquations.jl ODEProblem from MedLang source.

This is the primary integration point for using MedLang models with
the Julia DifferentialEquations ecosystem.

# Arguments
- `source::String`: MedLang source code
- `model_name::String`: Name of model to compile (default: first model)
- `tspan::Tuple`: Time span for simulation (default: (0.0, 24.0))
- `dose::Float64`: Initial dose in mg (default: 0.0)

# Returns
- `ODEProblem`: Ready-to-solve ODE problem

# Example
```julia
source = \"\"\"
model OneCmpt {
    state A_central : DoseMass = 100_mg
    param CL : Clearance = 10.0_L/h
    param V : Volume = 50.0_L

    d/dt A_central = -(CL / V) * A_central

    obs C_plasma : ConcMass = A_central / V
}
\"\"\"

using DifferentialEquations
prob = generate_ode_problem(source)
sol = solve(prob, Tsit5())
```
"""
function generate_ode_problem(
    source::String;
    model_name::Union{String,Nothing}=nothing,
    tspan::Tuple{Float64,Float64}=(0.0, 24.0),
    dose::Float64=0.0
)
    # DifferentialEquations is available via parent module
    result = generate_ode_model(source; model_name=model_name, dose=dose)

    # Create ODEProblem - caller must have DifferentialEquations loaded
    # Return a tuple that can be used to construct ODEProblem
    return (result.ode_func, result.u0, tspan, result.params)
end

export ODEModelResult, generate_ode_function, generate_ode_model, generate_ode_problem

#=============================================================================
  SOTA v0.2 Neural-Symbolic Transpilation
=============================================================================#

"""
Result of neural-symbolic transpilation.

Contains generated code for:
- Neural networks (Lux.jl)
- NeuralODE systems (DiffEqFlux.jl)
- Bayesian inference (Turing.jl)
- Molecular embeddings
"""
struct NeuralSymbolicResult
    model_name::String
    neural_networks::Dict{String,String}    # name → Lux.jl code
    neural_ode_code::String                  # DiffEqFlux NeuralODE code
    mechanistic_ode_code::String             # Standard ODE code
    inference_code::String                   # Turing.jl model code
    compound_embedding_code::String          # Molecular embedding code
    full_module_code::String                 # Complete Julia module
    warnings::Vector{String}
end

"""
Generate Lux.jl neural network code from NeuralNetSpec.
"""
function generate_lux_network(name::String, spec::NeuralNetSpec)::String
    code = IOBuffer()

    # Build chain of layers
    layers = spec.layers
    activation = spec.activation
    dropout = spec.dropout

    # Map activation names to Lux activations
    act_map = Dict(
        "relu" => "relu",
        "tanh" => "tanh",
        "swish" => "swish",
        "sigmoid" => "sigmoid",
        "gelu" => "gelu",
        "softplus" => "softplus",
        "elu" => "elu",
    )
    act_fn = get(act_map, lowercase(activation), "tanh")

    println(code, "# Neural network: $name")
    println(code, "function create_$(name)_network(input_dim::Int)")

    if isempty(layers)
        # Default network
        println(code, "    return Lux.Chain(")
        println(code, "        Lux.Dense(input_dim, 32, $act_fn),")
        println(code, "        Lux.Dense(32, 16, $act_fn),")
        println(code, "        Lux.Dense(16, 1)")
        println(code, "    )")
    else
        println(code, "    return Lux.Chain(")

        # Input layer
        println(code, "        Lux.Dense(input_dim, $(layers[1]), $act_fn),")

        # Hidden layers
        for i in 1:(length(layers)-1)
            if dropout > 0.0
                println(code, "        Lux.Dropout($dropout),")
            end
            println(code, "        Lux.Dense($(layers[i]), $(layers[i+1]), $act_fn),")
        end

        # Output layer
        println(code, "        Lux.Dense($(layers[end]), 1)")
        println(code, "    )")
    end

    println(code, "end")

    return String(take!(code))
end

"""
Generate NeuralODE code for hybrid neural-mechanistic system.
"""
function generate_neural_ode_code(model::ModelDef)::String
    code = IOBuffer()

    if isempty(model.neural_odes)
        return ""
    end

    println(code, "#=============================================================================")
    println(code, "  Neural ODE Components (DiffEqFlux.jl)")
    println(code, "=============================================================================#")
    println(code)

    # Generate each neural ODE
    for node in model.neural_odes
        println(code, "# NeuralODE: $(node.name)")
        println(code, "# State: $(node.state)")

        # Generate the neural network
        println(code, generate_lux_network("$(node.name)_nn", node.network))
        println(code)

        # Generate the NeuralODE wrapper
        println(code, "function create_$(node.name)_neuralode(;")
        println(code, "    input_dim::Int=1,")
        println(code, "    tspan::Tuple{Float64,Float64}=(0.0, 24.0),")
        println(code, "    solver=Tsit5()")
        println(code, ")")
        println(code, "    nn = create_$(node.name)_nn_network(input_dim)")
        println(code, "    rng = Random.default_rng()")
        println(code, "    ps, st = Lux.setup(rng, nn)")
        println(code, "    ")
        println(code, "    # Neural ODE dynamics")
        println(code, "    function neural_dynamics!(du, u, p, t)")
        println(code, "        du .= first(nn(u, p, st))")

        # Add constraints if present
        if !isempty(node.constraints)
            println(code, "        # Physiological constraints")
            println(code, "        du .= max.(du, -100.0)  # Prevent extreme negative rates")
        end

        println(code, "    end")
        println(code, "    ")
        println(code, "    return NeuralODE(neural_dynamics!, tspan, solver;")
        println(code, "        saveat=0.1, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))")
        println(code, "end")
        println(code)
    end

    return String(take!(code))
end

"""
Generate Turing.jl Bayesian inference code.
"""
function generate_inference_code(model::ModelDef)::String
    code = IOBuffer()

    if model.inference === nothing
        return ""
    end

    inf = model.inference

    println(code, "#=============================================================================")
    println(code, "  Bayesian Inference (Turing.jl)")
    println(code, "=============================================================================#")
    println(code)
    println(code, "using Turing")
    println(code, "using Distributions")
    println(code)

    println(code, "@model function $(model.name)_bayesian(obs_data, time_points)")
    println(code, "    # Priors")

    # Generate priors from parameters
    for param in model.params
        param_name = param.name
        # Default priors based on parameter type
        if contains(lowercase(param_name), "cl") || contains(lowercase(param_name), "clearance")
            println(code, "    $param_name ~ LogNormal(log(10.0), 0.5)  # Clearance prior")
        elseif contains(lowercase(param_name), "v") && !contains(lowercase(param_name), "max")
            println(code, "    $param_name ~ LogNormal(log(50.0), 0.5)  # Volume prior")
        elseif contains(lowercase(param_name), "ka") || contains(lowercase(param_name), "k")
            println(code, "    $param_name ~ LogNormal(log(1.0), 0.5)  # Rate constant prior")
        elseif contains(lowercase(param_name), "sigma") || contains(lowercase(param_name), "error")
            println(code, "    $param_name ~ truncated(Normal(0.0, 0.3), lower=0.0)  # Error prior")
        else
            println(code, "    $param_name ~ LogNormal(log(1.0), 1.0)  # Generic prior")
        end
    end

    println(code, "    ")
    println(code, "    # Simulate model")
    println(code, "    params = (; $(join([p.name for p in model.params], ", ")))")
    println(code, "    predicted = simulate_$(model.name)(params, time_points)")
    println(code, "    ")
    println(code, "    # Likelihood")
    println(code, "    sigma = haskey(params, :sigma) ? params.sigma : 0.1")
    println(code, "    for i in eachindex(obs_data)")
    println(code, "        obs_data[i] ~ Normal(predicted[i], sigma * predicted[i] + 0.01)")
    println(code, "    end")
    println(code, "    ")
    println(code, "    return predicted")
    println(code, "end")
    println(code)

    # Add inference utilities
    println(code, "\"\"\"")
    println(code, "Run Bayesian inference with $(inf.method) sampler.")
    println(code, "\"\"\"")
    println(code, "function run_$(model.name)_inference(obs_data, time_points;")
    println(code, "    n_samples::Int=1000, n_chains::Int=4)")

    if inf.method == "NUTS"
        target_accept = get(inf.method_params, "arg2", 0.65)
        println(code, "    model = $(model.name)_bayesian(obs_data, time_points)")
        println(code, "    chain = sample(model, NUTS($target_accept), MCMCThreads(), n_samples, n_chains)")
        println(code, "    return chain")
    elseif inf.method == "ADVI"
        println(code, "    model = $(model.name)_bayesian(obs_data, time_points)")
        println(code, "    q = vi(model, ADVI(10, 1000))")
        println(code, "    return q")
    else
        println(code, "    model = $(model.name)_bayesian(obs_data, time_points)")
        println(code, "    chain = sample(model, NUTS(0.65), n_samples)")
        println(code, "    return chain")
    end

    println(code, "end")
    println(code)

    return String(take!(code))
end

"""
Generate molecular embedding code for compound definition.
"""
function generate_compound_code(model::ModelDef)::String
    code = IOBuffer()

    if model.compound === nothing
        return ""
    end

    cmpd = model.compound

    println(code, "#=============================================================================")
    println(code, "  Molecular Embedding (SMILES-aware)")
    println(code, "=============================================================================#")
    println(code)

    println(code, "# Compound: $(cmpd.name)")
    println(code, "const $(uppercase(cmpd.name))_SMILES = \"$(cmpd.smiles)\"")
    println(code, "const $(uppercase(cmpd.name))_MW = $(cmpd.mw)")
    println(code)

    # Generate embedding function
    println(code, "\"\"\"")
    println(code, "Generate molecular embedding from SMILES.")
    println(code, "Uses fingerprint-based encoding if no neural encoder available.")
    println(code, "\"\"\"")
    println(code, "function get_$(cmpd.name)_embedding(;")
    println(code, "    embedding_dim::Int=256,")
    println(code, "    use_neural::Bool=true")
    println(code, ")::Vector{Float64}")
    println(code, "    smiles = $(uppercase(cmpd.name))_SMILES")
    println(code, "    ")
    println(code, "    if use_neural && @isdefined(ChemBERTa)")
    println(code, "        # Use neural encoder if available")
    println(code, "        return ChemBERTa.encode(smiles)")
    println(code, "    else")
    println(code, "        # Fallback: fingerprint-based embedding")
    println(code, "        return smiles_to_fingerprint(smiles, embedding_dim)")
    println(code, "    end")
    println(code, "end")
    println(code)

    # Add fingerprint utility
    println(code, "\"\"\"")
    println(code, "Convert SMILES to molecular fingerprint vector.")
    println(code, "Simple hash-based fingerprint for fallback encoding.")
    println(code, "\"\"\"")
    println(code, "function smiles_to_fingerprint(smiles::String, dim::Int=256)::Vector{Float64}")
    println(code, "    fp = zeros(Float64, dim)")
    println(code, "    ")
    println(code, "    # Character-level features")
    println(code, "    for (i, c) in enumerate(smiles)")
    println(code, "        idx = (Int(c) * (i + 1)) % dim + 1")
    println(code, "        fp[idx] += 1.0")
    println(code, "    end")
    println(code, "    ")
    println(code, "    # Substructure features (2-grams, 3-grams)")
    println(code, "    for n in 2:min(4, length(smiles))")
    println(code, "        for i in 1:(length(smiles)-n+1)")
    println(code, "            substr = smiles[i:i+n-1]")
    println(code, "            idx = abs(hash(substr)) % dim + 1")
    println(code, "            fp[idx] += 1.0 / n")
    println(code, "        end")
    println(code, "    end")
    println(code, "    ")
    println(code, "    # Normalize")
    println(code, "    norm = sqrt(sum(fp.^2) + 1e-8)")
    println(code, "    return fp ./ norm")
    println(code, "end")
    println(code)

    return String(take!(code))
end

"""
Generate pharmacodynamics model code.
"""
function generate_pd_code(model::ModelDef)::String
    code = IOBuffer()

    if isempty(model.pharmacodynamics)
        return ""
    end

    println(code, "#=============================================================================")
    println(code, "  Pharmacodynamics Models")
    println(code, "=============================================================================#")
    println(code)

    for pd in model.pharmacodynamics
        println(code, "# PD Model: $(pd.name) ($(pd.model))")

        if pd.model == "Emax"
            println(code, "function $(pd.name)_effect(C::Float64;")
            println(code, "    E0::Float64=0.0,")
            println(code, "    Emax::Float64=100.0,")
            println(code, "    EC50::Float64=10.0,")
            println(code, "    hill::Float64=1.0")
            println(code, ")::Float64")
            println(code, "    return E0 + Emax * C^hill / (EC50^hill + C^hill)")
            println(code, "end")
        elseif pd.model == "linear"
            println(code, "function $(pd.name)_effect(C::Float64;")
            println(code, "    E0::Float64=0.0,")
            println(code, "    slope::Float64=1.0")
            println(code, ")::Float64")
            println(code, "    return E0 + slope * C")
            println(code, "end")
        elseif pd.model == "sigmoid"
            println(code, "function $(pd.name)_effect(C::Float64;")
            println(code, "    E0::Float64=0.0,")
            println(code, "    Emax::Float64=100.0,")
            println(code, "    EC50::Float64=10.0,")
            println(code, "    gamma::Float64=1.0")
            println(code, ")::Float64")
            println(code, "    return E0 + Emax / (1.0 + (EC50/C)^gamma)")
            println(code, "end")
        else
            # Default to Emax
            println(code, "function $(pd.name)_effect(C::Float64; kwargs...)::Float64")
            println(code, "    # Generic PD model")
            println(code, "    return C")
            println(code, "end")
        end
        println(code)
    end

    return String(take!(code))
end

"""
Transpile MedLang model with SOTA neural-symbolic features.

Generates complete Julia module with:
- Neural networks (Lux.jl)
- NeuralODE hybrid systems (DiffEqFlux.jl)
- Bayesian inference (Turing.jl)
- Molecular embeddings
- Pharmacodynamics models

# Arguments
- `source::String`: MedLang source code
- `model_name::String`: Model to transpile (default: first)

# Returns
- `NeuralSymbolicResult`: Complete transpilation result
"""
function transpile_neural_symbolic(
    source::String;
    model_name::Union{String,Nothing}=nothing
)::NeuralSymbolicResult
    ast = parse_medlang(source)
    warnings = String[]

    if isempty(ast.models)
        throw(TranspileError("No models found in source", "transpile_neural_symbolic"))
    end

    model = if model_name !== nothing
        idx = findfirst(m -> m.name == model_name, ast.models)
        if idx === nothing
            throw(TranspileError("Model '$model_name' not found", "transpile_neural_symbolic"))
        end
        ast.models[idx]
    else
        ast.models[1]
    end

    # Generate individual components
    neural_networks = Dict{String,String}()
    for node in model.neural_odes
        neural_networks[node.name] = generate_lux_network("$(node.name)_nn", node.network)
    end

    neural_ode_code = generate_neural_ode_code(model)
    mechanistic_ode_code = generate_ode_system(model)
    inference_code = generate_inference_code(model)
    compound_code = generate_compound_code(model)
    pd_code = generate_pd_code(model)

    # Generate full module
    full_code = IOBuffer()

    println(full_code, "\"\"\"")
    println(full_code, "Generated Neural-Symbolic PBPK Model: $(model.name)")
    println(full_code, "")
    println(full_code, "This module was auto-generated from MedLang DSL source.")
    println(full_code, "Features: Neural ODE, Bayesian inference, molecular embeddings")
    println(full_code, "\"\"\"")
    println(full_code, "module $(model.name)NeuralPBPK")
    println(full_code)
    println(full_code, "# Dependencies")
    println(full_code, "using DifferentialEquations")
    println(full_code, "using Lux")
    println(full_code, "using Random")
    println(full_code, "using Zygote")

    if !isempty(model.neural_odes)
        println(full_code, "using DiffEqFlux: NeuralODE, InterpolatingAdjoint, ZygoteVJP")
    end

    if model.inference !== nothing
        println(full_code, "using Turing")
        println(full_code, "using Distributions")
    end

    println(full_code)

    # Add compound/molecular code
    if !isempty(compound_code)
        println(full_code, compound_code)
    end

    # Add neural networks and NeuralODE code
    if !isempty(neural_ode_code)
        println(full_code, neural_ode_code)
    end

    # Add mechanistic ODE code
    if !isempty(mechanistic_ode_code)
        println(full_code, "# Mechanistic ODE System")
        println(full_code, mechanistic_ode_code)
        println(full_code)
    end

    # Add PD models
    if !isempty(pd_code)
        println(full_code, pd_code)
    end

    # Add inference code
    if !isempty(inference_code)
        println(full_code, inference_code)
    end

    # Add hybrid simulation function
    println(full_code, "#=============================================================================")
    println(full_code, "  Hybrid Neural-Mechanistic Simulation")
    println(full_code, "=============================================================================#")
    println(full_code)
    println(full_code, "\"\"\"")
    println(full_code, "Run hybrid neural-mechanistic simulation.")
    println(full_code, "Combines mechanistic ODEs with learned neural components.")
    println(full_code, "\"\"\"")
    println(full_code, "function simulate_hybrid(;")
    println(full_code, "    dose::Float64=100.0,")
    println(full_code, "    tspan::Tuple{Float64,Float64}=(0.0, 24.0),")
    println(full_code, "    saveat::Float64=0.1,")
    println(full_code, "    neural_params=nothing")
    println(full_code, ")")
    println(full_code, "    # Initialize state")
    println(full_code, "    u0 = [dose]  # Initial concentration")
    println(full_code, "    ")
    println(full_code, "    # Default parameters")
    println(full_code, "    p = (CL=10.0, V=50.0)")
    println(full_code, "    ")
    println(full_code, "    # Solve ODE")
    println(full_code, "    prob = ODEProblem($(model.name)_ode!, u0, tspan, p)")
    println(full_code, "    sol = solve(prob, Tsit5(); saveat=saveat)")
    println(full_code, "    ")
    println(full_code, "    return sol")
    println(full_code, "end")
    println(full_code)

    println(full_code, "end # module")

    return NeuralSymbolicResult(
        model.name,
        neural_networks,
        neural_ode_code,
        mechanistic_ode_code,
        inference_code,
        compound_code,
        String(take!(full_code)),
        warnings
    )
end

export NeuralSymbolicResult, transpile_neural_symbolic
export generate_lux_network, generate_neural_ode_code, generate_inference_code
export generate_compound_code, generate_pd_code

end # module
