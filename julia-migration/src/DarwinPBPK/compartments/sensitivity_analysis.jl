"""
Sensitivity Analysis Module for PBPK Models

Implements comprehensive local and global sensitivity analysis methods:
- Local: One-at-a-time (OAT), elasticity coefficients, normalized sensitivity
- Global: Sobol indices, Morris screening, PRCC
- Advanced sampling: Latin Hypercube, Sobol sequences, Morris trajectories
- Integration with coagulation and other PBPK compartment models

Methods based on SOTA literature:
- Saltelli et al. (2008) - Global Sensitivity Analysis
- Sobol (2001) - Variance-based sensitivity
- Morris (1991) - Elementary effects screening
- Marino et al. (2008) - PRCC for biological models

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

module SensitivityAnalysis

using LinearAlgebra
using Statistics
using Random
using Distributions

export ParameterRange, SensitivityResult, SensitivityConfig
export one_at_a_time_sensitivity, calculate_elasticity, normalized_sensitivity_coefficient
export sobol_sensitivity, morris_screening, prcc_analysis
export latin_hypercube_sample, sobol_sequence, morris_trajectories
export rank_parameters, identify_influential_parameters
export sensitivity_tornado_plot_data, sensitivity_heatmap_data
export coagulation_sensitivity_wrapper, default_coagulation_parameters

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    ParameterRange

Defines a parameter for sensitivity analysis with its range and distribution.

# Fields
- `name::String`: Parameter name
- `nominal::Float64`: Nominal/baseline value
- `min::Float64`: Minimum value (lower bound)
- `max::Float64`: Maximum value (upper bound)
- `distribution::Symbol`: Distribution type (:uniform, :normal, :lognormal)
- `std::Float64`: Standard deviation (for :normal, :lognormal)
"""
struct ParameterRange
    name::String
    nominal::Float64
    min::Float64
    max::Float64
    distribution::Symbol  # :uniform, :normal, :lognormal
    std::Float64  # For normal/lognormal

    function ParameterRange(name, nominal, min, max; distribution=:uniform, std=0.0)
        @assert min < max "min must be less than max"
        @assert min <= nominal <= max "nominal must be within [min, max]"
        new(name, nominal, min, max, distribution, std)
    end
end

"""
    SensitivityResult

Stores results from sensitivity analysis.

# Fields
- `method::Symbol`: Analysis method (:oat, :sobol, :morris, :prcc)
- `parameters::Vector{String}`: Parameter names
- `outputs::Vector{String}`: Output variable names
- `sensitivities::Dict{String, Dict{String, Float64}}`: Nested dict [output][param] => sensitivity
- `indices::Dict{String, Any}`: Method-specific indices (S1, ST, μ*, σ, etc.)
- `rankings::Dict{String, Vector{Tuple{String, Float64}}}`: Ranked parameters by output
- `metadata::Dict{String, Any}`: Additional info (n_samples, confidence intervals, etc.)
"""
struct SensitivityResult
    method::Symbol
    parameters::Vector{String}
    outputs::Vector{String}
    sensitivities::Dict{String, Dict{String, Float64}}
    indices::Dict{String, Any}
    rankings::Dict{String, Vector{Tuple{String, Float64}}}
    metadata::Dict{String, Any}
end

"""
    SensitivityConfig

Configuration for sensitivity analysis.

# Fields
- `method::Symbol`: :oat, :sobol, :morris, :prcc
- `n_samples::Int`: Number of samples (for global methods)
- `n_trajectories::Int`: Number of trajectories (Morris)
- `n_levels::Int`: Number of levels (Morris)
- `perturbation::Float64`: Perturbation size (OAT, default 0.01 = 1%)
- `outputs::Vector{String}`: Output variables to analyze
- `parallel::Bool`: Use parallel computation (not implemented)
- `seed::Int`: Random seed for reproducibility
"""
struct SensitivityConfig
    method::Symbol
    n_samples::Int
    n_trajectories::Int
    n_levels::Int
    perturbation::Float64
    outputs::Vector{String}
    parallel::Bool
    seed::Int

    function SensitivityConfig(;
        method=:sobol,
        n_samples=1024,
        n_trajectories=10,
        n_levels=4,
        perturbation=0.01,
        outputs=String[],
        parallel=false,
        seed=42
    )
        new(method, n_samples, n_trajectories, n_levels, perturbation, outputs, parallel, seed)
    end
end

# ============================================================================
# LOCAL SENSITIVITY ANALYSIS
# ============================================================================

"""
    one_at_a_time_sensitivity(model::Function, params::Vector{ParameterRange},
                              outputs::Vector{String}; perturbation=0.01)

Performs One-At-a-Time (OAT) local sensitivity analysis.

Perturbs each parameter individually while holding others at nominal values.
Returns normalized sensitivity coefficients: (ΔY/Y₀) / (ΔX/X₀).

# Arguments
- `model`: Function that takes Dict{String, Float64} and returns Dict{String, Float64}
- `params`: Vector of ParameterRange defining parameters
- `outputs`: Output variable names to analyze
- `perturbation`: Relative perturbation size (default 0.01 = 1%)

# Returns
- `SensitivityResult` with OAT sensitivities
"""
function one_at_a_time_sensitivity(
    model::Function,
    params::Vector{ParameterRange},
    outputs::Vector{String};
    perturbation::Float64=0.01
)
    n_params = length(params)

    # Baseline simulation
    baseline_dict = Dict(p.name => p.nominal for p in params)
    baseline_output = model(baseline_dict)

    # Pre-allocate storage
    sensitivities = Dict{String, Dict{String, Float64}}()
    for out in outputs
        sensitivities[out] = Dict{String, Float64}()
    end

    # Perturb each parameter
    for (i, param) in enumerate(params)
        # Create perturbed parameter dict
        perturbed_dict = copy(baseline_dict)
        delta = param.nominal * perturbation
        perturbed_dict[param.name] = param.nominal + delta

        # Ensure within bounds
        perturbed_dict[param.name] = clamp(perturbed_dict[param.name], param.min, param.max)

        # Simulate with perturbation
        perturbed_output = model(perturbed_dict)

        # Calculate normalized sensitivity for each output
        for out in outputs
            if haskey(baseline_output, out) && haskey(perturbed_output, out)
                y0 = baseline_output[out]
                y1 = perturbed_output[out]

                # Normalized sensitivity: (ΔY/Y₀) / (ΔX/X₀)
                if y0 != 0.0
                    sens = ((y1 - y0) / y0) / perturbation
                else
                    sens = 0.0
                end

                sensitivities[out][param.name] = sens
            end
        end
    end

    # Rank parameters by absolute sensitivity
    rankings = Dict{String, Vector{Tuple{String, Float64}}}()
    for out in outputs
        sorted_params = sort(
            [(k, v) for (k, v) in sensitivities[out]],
            by=x->abs(x[2]),
            rev=true
        )
        rankings[out] = sorted_params
    end

    metadata = Dict{String, Any}(
        "perturbation" => perturbation,
        "n_parameters" => n_params,
        "baseline_output" => baseline_output
    )

    return SensitivityResult(
        :oat,
        [p.name for p in params],
        outputs,
        sensitivities,
        Dict{String, Any}(),
        rankings,
        metadata
    )
end

"""
    calculate_elasticity(model::Function, params::Dict{String, Float64},
                        param_name::String, output_name::String; δ=1e-6)

Calculates elasticity coefficient: E = (∂Y/∂X) × (X/Y).

Measures the percentage change in output Y for a 1% change in parameter X.

# Returns
- Elasticity coefficient (dimensionless)
"""
function calculate_elasticity(
    model::Function,
    params::Dict{String, Float64},
    param_name::String,
    output_name::String;
    δ::Float64=1e-6
)
    # Baseline
    y0 = model(params)[output_name]
    x0 = params[param_name]

    # Perturbed (use relative perturbation)
    params_pert = copy(params)
    params_pert[param_name] = x0 * (1.0 + δ)
    y1 = model(params_pert)[output_name]

    # Elasticity: (dy/dx) * (x/y)
    if y0 != 0.0
        dy_dx = (y1 - y0) / (x0 * δ)
        elasticity = dy_dx * (x0 / y0)
    else
        elasticity = 0.0
    end

    return elasticity
end

"""
    normalized_sensitivity_coefficient(model::Function, params::Dict{String, Float64},
                                       param_name::String, output_name::String; δ=0.01)

Calculates normalized sensitivity coefficient (NSC): (ΔY/Y) / (ΔX/X).

Similar to elasticity but uses finite differences with larger perturbation.
"""
function normalized_sensitivity_coefficient(
    model::Function,
    params::Dict{String, Float64},
    param_name::String,
    output_name::String;
    δ::Float64=0.01
)
    # Baseline
    y0 = model(params)[output_name]
    x0 = params[param_name]

    # Perturbed
    params_pert = copy(params)
    params_pert[param_name] = x0 * (1.0 + δ)
    y1 = model(params_pert)[output_name]

    # NSC
    if y0 != 0.0 && x0 != 0.0
        nsc = ((y1 - y0) / y0) / ((params_pert[param_name] - x0) / x0)
    else
        nsc = 0.0
    end

    return nsc
end

# ============================================================================
# GLOBAL SENSITIVITY ANALYSIS - SOBOL INDICES
# ============================================================================

"""
    sobol_sensitivity(model::Function, params::Vector{ParameterRange},
                      outputs::Vector{String}; n_samples=1024, seed=42)

Performs variance-based global sensitivity analysis using Sobol indices.

Computes first-order (S1) and total-order (ST) indices using Saltelli sampling scheme.
- S1: Main effect of parameter alone
- ST: Total effect including interactions

# Arguments
- `n_samples`: Base sample size (total evaluations = n_samples × (2p + 2))

# Returns
- `SensitivityResult` with S1 and ST indices
"""
function sobol_sensitivity(
    model::Function,
    params::Vector{ParameterRange},
    outputs::Vector{String};
    n_samples::Int=1024,
    seed::Int=42
)
    Random.seed!(seed)
    n_params = length(params)

    # Generate Saltelli sampling matrices
    # Need 3 matrices: A, B, and AB_i for each parameter
    # Total evaluations: n_samples × (2p + 2)
    A = latin_hypercube_sample(params, n_samples)  # n × p
    B = latin_hypercube_sample(params, n_samples)  # n × p

    # Pre-allocate outputs
    Y_A = zeros(n_samples, length(outputs))
    Y_B = zeros(n_samples, length(outputs))
    Y_AB = zeros(n_samples, n_params, length(outputs))

    # Evaluate model at A matrix
    for i in 1:n_samples
        params_dict = Dict(params[j].name => A[i, j] for j in 1:n_params)
        result = model(params_dict)
        for (k, out) in enumerate(outputs)
            Y_A[i, k] = result[out]
        end
    end

    # Evaluate model at B matrix
    for i in 1:n_samples
        params_dict = Dict(params[j].name => B[i, j] for j in 1:n_params)
        result = model(params_dict)
        for (k, out) in enumerate(outputs)
            Y_B[i, k] = result[out]
        end
    end

    # Evaluate model at AB_i matrices (A with i-th column from B)
    for j in 1:n_params
        for i in 1:n_samples
            AB_i = copy(A[i, :])
            AB_i[j] = B[i, j]  # Replace j-th parameter

            params_dict = Dict(params[k].name => AB_i[k] for k in 1:n_params)
            result = model(params_dict)
            for (m, out) in enumerate(outputs)
                Y_AB[i, j, m] = result[out]
            end
        end
    end

    # Calculate Sobol indices
    S1 = Dict{String, Dict{String, Float64}}()
    ST = Dict{String, Dict{String, Float64}}()

    for (k, out) in enumerate(outputs)
        S1[out] = Dict{String, Float64}()
        ST[out] = Dict{String, Float64}()

        # Variance of Y
        V_Y = var(vcat(Y_A[:, k], Y_B[:, k]))

        if V_Y > 1e-12  # Avoid division by zero
            for j in 1:n_params
                param_name = params[j].name

                # First-order index: S1 = V[E(Y|Xi)] / V(Y)
                # Estimator: (1/n) Σ(f(B)_i * (f(AB_i)_i - f(A)_i)) / V(Y)
                numerator_s1 = mean(Y_B[:, k] .* (Y_AB[:, j, k] .- Y_A[:, k]))
                S1[out][param_name] = numerator_s1 / V_Y

                # Total-order index: ST = 1 - V[E(Y|X~i)] / V(Y)
                # Estimator: (1/2n) Σ(f(A)_i - f(AB_i)_i)^2 / V(Y)
                numerator_st = 0.5 * mean((Y_A[:, k] .- Y_AB[:, j, k]).^2)
                ST[out][param_name] = numerator_st / V_Y
            end
        else
            # No variance - all parameters have zero sensitivity
            for j in 1:n_params
                S1[out][params[j].name] = 0.0
                ST[out][params[j].name] = 0.0
            end
        end
    end

    # Rank by total-order index (includes interactions)
    rankings = Dict{String, Vector{Tuple{String, Float64}}}()
    for out in outputs
        sorted_params = sort(
            [(k, v) for (k, v) in ST[out]],
            by=x->abs(x[2]),
            rev=true
        )
        rankings[out] = sorted_params
    end

    metadata = Dict{String, Any}(
        "n_samples" => n_samples,
        "n_evaluations" => n_samples * (2 * n_params + 2),
        "sampling_scheme" => "Saltelli"
    )

    indices = Dict{String, Any}(
        "S1" => S1,
        "ST" => ST
    )

    # Use ST for main sensitivities dict
    return SensitivityResult(
        :sobol,
        [p.name for p in params],
        outputs,
        ST,
        indices,
        rankings,
        metadata
    )
end

# ============================================================================
# GLOBAL SENSITIVITY ANALYSIS - MORRIS METHOD
# ============================================================================

"""
    morris_screening(model::Function, params::Vector{ParameterRange},
                     outputs::Vector{String}; n_trajectories=10, n_levels=4, seed=42)

Performs Morris screening using elementary effects method.

Efficient screening method to identify important parameters.
Returns μ* (mean absolute effect) and σ (standard deviation of effects).

High μ* → important parameter
High σ → non-linear or interacting parameter

# Arguments
- `n_trajectories`: Number of independent trajectories (r)
- `n_levels`: Number of grid levels (typically 4 or 6)

# Returns
- `SensitivityResult` with μ*, σ indices
"""
function morris_screening(
    model::Function,
    params::Vector{ParameterRange},
    outputs::Vector{String};
    n_trajectories::Int=10,
    n_levels::Int=4,
    seed::Int=42
)
    Random.seed!(seed)
    n_params = length(params)

    # Storage for elementary effects
    EE = zeros(n_trajectories, n_params, length(outputs))

    # Generate r trajectories
    trajectories = morris_trajectories(params, n_trajectories, n_levels)

    # For each trajectory
    for r in 1:n_trajectories
        traj = trajectories[r]  # (p+1) × p matrix

        # Baseline point
        params_dict = Dict(params[j].name => traj[1, j] for j in 1:n_params)
        y_prev = model(params_dict)

        # For each step in trajectory
        for i in 1:n_params
            params_dict = Dict(params[j].name => traj[i+1, j] for j in 1:n_params)
            y_curr = model(params_dict)

            # Find which parameter changed
            changed_idx = findfirst(traj[i+1, :] .!= traj[i, :])

            if changed_idx !== nothing
                delta = traj[i+1, changed_idx] - traj[i, changed_idx]

                # Calculate elementary effect for each output
                for (k, out) in enumerate(outputs)
                    if haskey(y_prev, out) && haskey(y_curr, out)
                        dy = y_curr[out] - y_prev[out]
                        EE[r, changed_idx, k] = dy / delta
                    end
                end
            end

            y_prev = y_curr
        end
    end

    # Calculate Morris statistics
    μ_star = Dict{String, Dict{String, Float64}}()  # Mean of absolute effects
    σ = Dict{String, Dict{String, Float64}}()        # Standard deviation

    for (k, out) in enumerate(outputs)
        μ_star[out] = Dict{String, Float64}()
        σ[out] = Dict{String, Float64}()

        for j in 1:n_params
            param_name = params[j].name
            effects = EE[:, j, k]

            μ_star[out][param_name] = mean(abs.(effects))
            σ[out][param_name] = std(effects)
        end
    end

    # Rank by μ* (mean absolute effect)
    rankings = Dict{String, Vector{Tuple{String, Float64}}}()
    for out in outputs
        sorted_params = sort(
            [(k, v) for (k, v) in μ_star[out]],
            by=x->x[2],
            rev=true
        )
        rankings[out] = sorted_params
    end

    metadata = Dict{String, Any}(
        "n_trajectories" => n_trajectories,
        "n_levels" => n_levels,
        "n_evaluations" => n_trajectories * (n_params + 1)
    )

    indices = Dict{String, Any}(
        "μ_star" => μ_star,
        "σ" => σ,
        "elementary_effects" => EE
    )

    return SensitivityResult(
        :morris,
        [p.name for p in params],
        outputs,
        μ_star,
        indices,
        rankings,
        metadata
    )
end

# ============================================================================
# GLOBAL SENSITIVITY ANALYSIS - PRCC
# ============================================================================

"""
    prcc_analysis(model::Function, params::Vector{ParameterRange},
                  outputs::Vector{String}; n_samples=1000, seed=42)

Performs Partial Rank Correlation Coefficient (PRCC) analysis.

PRCC measures monotonic relationship between parameter and output,
controlling for other parameters. Useful for non-linear models.

# Returns
- `SensitivityResult` with PRCC values (-1 to 1)
"""
function prcc_analysis(
    model::Function,
    params::Vector{ParameterRange},
    outputs::Vector{String};
    n_samples::Int=1000,
    seed::Int=42
)
    Random.seed!(seed)
    n_params = length(params)

    # Generate Latin Hypercube samples
    X = latin_hypercube_sample(params, n_samples)

    # Pre-allocate output matrix
    Y = zeros(n_samples, length(outputs))

    # Run model simulations
    for i in 1:n_samples
        params_dict = Dict(params[j].name => X[i, j] for j in 1:n_params)
        result = model(params_dict)
        for (k, out) in enumerate(outputs)
            Y[i, k] = result[out]
        end
    end

    # Calculate PRCC for each output
    prcc_values = Dict{String, Dict{String, Float64}}()

    for (k, out) in enumerate(outputs)
        prcc_values[out] = Dict{String, Float64}()

        # Convert to ranks
        X_ranks = hcat([sortperm(sortperm(X[:, j])) for j in 1:n_params]...)
        Y_rank = sortperm(sortperm(Y[:, k]))

        # For each parameter
        for j in 1:n_params
            param_name = params[j].name

            # Get ranks for parameter j and output
            x_j = X_ranks[:, j]
            y = Y_rank

            # Get ranks for other parameters (to control for)
            other_indices = setdiff(1:n_params, j)
            X_other = X_ranks[:, other_indices]

            # Calculate PRCC using partial correlation
            prcc = calculate_prcc(x_j, y, X_other)
            prcc_values[out][param_name] = prcc
        end
    end

    # Rank by absolute PRCC
    rankings = Dict{String, Vector{Tuple{String, Float64}}}()
    for out in outputs
        sorted_params = sort(
            [(k, v) for (k, v) in prcc_values[out]],
            by=x->abs(x[2]),
            rev=true
        )
        rankings[out] = sorted_params
    end

    metadata = Dict{String, Any}(
        "n_samples" => n_samples,
        "method" => "PRCC"
    )

    indices = Dict{String, Any}(
        "PRCC" => prcc_values
    )

    return SensitivityResult(
        :prcc,
        [p.name for p in params],
        outputs,
        prcc_values,
        indices,
        rankings,
        metadata
    )
end

"""
    calculate_prcc(x::Vector{Float64}, y::Vector{Float64}, Z::Matrix{Float64})

Calculates Partial Rank Correlation Coefficient.

Uses linear regression to remove effect of Z, then correlates residuals.
"""
function calculate_prcc(x::Vector{<:Real}, y::Vector{<:Real}, Z::Matrix{<:Real})
    n = length(x)

    # Convert to Float64 for calculations
    x_f = Float64.(x)
    y_f = Float64.(y)
    Z_f = Float64.(Z)

    if size(Z_f, 2) == 0
        # No variables to control for - return simple correlation
        return cor(x_f, y_f)
    end

    # Add intercept to Z
    Z_aug = hcat(ones(n), Z_f)

    # Regress x on Z
    β_x = Z_aug \ x_f
    res_x = x_f .- Z_aug * β_x

    # Regress y on Z
    β_y = Z_aug \ y_f
    res_y = y_f .- Z_aug * β_y

    # Correlation of residuals
    return cor(res_x, res_y)
end

# ============================================================================
# SAMPLING FUNCTIONS
# ============================================================================

"""
    latin_hypercube_sample(params::Vector{ParameterRange}, n::Int)

Generates Latin Hypercube samples for parameters.

Returns n × p matrix where each column is uniformly sampled in [min, max].
"""
function latin_hypercube_sample(params::Vector{ParameterRange}, n::Int)
    p = length(params)
    samples = zeros(n, p)

    for j in 1:p
        param = params[j]

        # Generate LHS in [0, 1]
        segments = randperm(n)
        lhs = (segments .- 1.0 .+ rand(n)) ./ n

        # Scale to parameter range
        if param.distribution == :uniform
            samples[:, j] = param.min .+ lhs .* (param.max - param.min)
        elseif param.distribution == :normal
            # Normal distribution truncated to [min, max]
            dist = truncated(Normal(param.nominal, param.std), param.min, param.max)
            samples[:, j] = quantile.(dist, lhs)
        elseif param.distribution == :lognormal
            # Lognormal distribution
            μ = log(param.nominal)
            σ = param.std
            dist = truncated(LogNormal(μ, σ), param.min, param.max)
            samples[:, j] = quantile.(dist, lhs)
        end
    end

    return samples
end

"""
    sobol_sequence(params::Vector{ParameterRange}, n::Int)

Generates Sobol quasi-random sequence for parameters.

Note: Simplified implementation. For production, use Sobol.jl package.
"""
function sobol_sequence(params::Vector{ParameterRange}, n::Int)
    # Simplified: Use Latin Hypercube as approximation
    # For true Sobol sequence, integrate Sobol.jl
    return latin_hypercube_sample(params, n)
end

"""
    morris_trajectories(params::Vector{ParameterRange}, r::Int, levels::Int)

Generates r Morris trajectories for parameters.

Each trajectory has (p+1) points, where one parameter changes at each step.

# Returns
- Vector of r matrices, each (p+1) × p
"""
function morris_trajectories(params::Vector{ParameterRange}, r::Int, levels::Int)
    p = length(params)
    trajectories = Vector{Matrix{Float64}}(undef, r)

    # Grid levels
    grid = collect(range(0.0, 1.0, length=levels))

    for i in 1:r
        traj = zeros(p + 1, p)

        # Start from random grid point
        for j in 1:p
            traj[1, j] = rand(grid)
        end

        # Random permutation of parameters
        perm = randperm(p)

        # Build trajectory
        for step in 1:p
            traj[step + 1, :] = traj[step, :]
            param_idx = perm[step]

            # Perturb by Δ = levels/(2(levels-1))
            Δ = levels / (2.0 * (levels - 1))

            # Randomly choose direction
            direction = rand([-1, 1])
            new_val = traj[step, param_idx] + direction * Δ

            # Keep in [0, 1]
            new_val = clamp(new_val, 0.0, 1.0)
            traj[step + 1, param_idx] = new_val
        end

        # Scale to parameter ranges
        traj_scaled = zeros(p + 1, p)
        for j in 1:p
            param = params[j]
            traj_scaled[:, j] = param.min .+ traj[:, j] .* (param.max - param.min)
        end

        trajectories[i] = traj_scaled
    end

    return trajectories
end

# ============================================================================
# OUTPUT ANALYSIS
# ============================================================================

"""
    rank_parameters(result::SensitivityResult)

Returns parameters ranked by importance for each output.

# Returns
- Dict{String, Vector{Tuple{String, Float64}}} mapping output => [(param, score), ...]
"""
function rank_parameters(result::SensitivityResult)
    return result.rankings
end

"""
    identify_influential_parameters(result::SensitivityResult, threshold::Float64)

Identifies parameters with sensitivity above threshold.

# Returns
- Dict{String, Vector{String}} mapping output => [influential_params]
"""
function identify_influential_parameters(result::SensitivityResult, threshold::Float64)
    influential = Dict{String, Vector{String}}()

    for out in result.outputs
        influential[out] = String[]

        for (param, score) in result.rankings[out]
            if abs(score) >= threshold
                push!(influential[out], param)
            end
        end
    end

    return influential
end

"""
    sensitivity_tornado_plot_data(result::SensitivityResult, output::String)

Prepares data for tornado diagram (ranked horizontal bar chart).

# Returns
- Vector of (parameter_name, sensitivity_value) sorted by absolute value
"""
function sensitivity_tornado_plot_data(result::SensitivityResult, output::String)
    return result.rankings[output]
end

"""
    sensitivity_heatmap_data(result::SensitivityResult)

Prepares data for sensitivity heatmap (parameters × outputs).

# Returns
- Named tuple (parameters, outputs, matrix) where matrix[i,j] = sensitivity of param i to output j
"""
function sensitivity_heatmap_data(result::SensitivityResult)
    n_params = length(result.parameters)
    n_outputs = length(result.outputs)

    matrix = zeros(n_params, n_outputs)

    for (j, out) in enumerate(result.outputs)
        for (i, param) in enumerate(result.parameters)
            if haskey(result.sensitivities[out], param)
                matrix[i, j] = result.sensitivities[out][param]
            end
        end
    end

    return (parameters=result.parameters, outputs=result.outputs, matrix=matrix)
end

# ============================================================================
# COAGULATION MODEL INTEGRATION
# ============================================================================

"""
    coagulation_sensitivity_wrapper(params_dict::Dict{String, Float64})

Wrapper function for coagulation model sensitivity analysis.

Simulates thrombin generation and returns key outputs:
- peak_thrombin: Maximum thrombin concentration (nM)
- lag_time: Time to 10 nM thrombin (min)
- ttp: Time to peak thrombin (min)
- etp: Endogenous thrombin potential (nM·min)

# Arguments
- `params_dict`: Dict with factor concentrations (e.g., "II", "V", "VII", etc.)

# Returns
- Dict with output metrics
"""
function coagulation_sensitivity_wrapper(params_dict::Dict{String, Float64})
    # This is a placeholder - integrate with actual coagulation model
    # from DarwinPBPK.Coagulation module

    # Extract key factors
    fII = get(params_dict, "II", 1400.0)
    fV = get(params_dict, "V", 22.0)
    fVII = get(params_dict, "VII", 10.0)
    fVIII = get(params_dict, "VIII", 0.7)
    fIX = get(params_dict, "IX", 90.0)
    fX = get(params_dict, "X", 170.0)

    # Simplified thrombin generation model
    # In production, call simulate_coagulation! from Coagulation module

    # Mock outputs (replace with actual simulation)
    peak_thrombin = fII * 0.3 * (1.0 + 0.2 * fV / 22.0) * (1.0 + 0.15 * fVII / 10.0)
    lag_time = 5.0 - 0.5 * (fVII / 10.0) + 0.3 * (fVIII / 0.7)
    ttp = 8.0 + 0.2 * lag_time
    etp = peak_thrombin * 15.0  # Approximate area under curve

    return Dict(
        "peak_thrombin" => peak_thrombin,
        "lag_time" => max(lag_time, 0.1),
        "ttp" => ttp,
        "etp" => etp
    )
end

"""
    default_coagulation_parameters()

Returns default parameter ranges for coagulation sensitivity analysis.

Varies factors ±50% around normal values (similar to clinical variability).
"""
function default_coagulation_parameters()
    # From Coagulation module NORMAL_FACTOR_CONCENTRATIONS
    return [
        ParameterRange("II", 1400.0, 700.0, 2100.0),      # Prothrombin
        ParameterRange("V", 22.0, 11.0, 33.0),            # Factor V
        ParameterRange("VII", 10.0, 5.0, 15.0),           # Factor VII
        ParameterRange("VIII", 0.7, 0.35, 1.05),          # Factor VIII
        ParameterRange("IX", 90.0, 45.0, 135.0),          # Factor IX
        ParameterRange("X", 170.0, 85.0, 255.0),          # Factor X
        ParameterRange("ATIII", 2400.0, 1200.0, 3600.0),  # Antithrombin
        ParameterRange("TFPI", 2.5, 1.25, 3.75),          # TFPI
    ]
end

end # module SensitivityAnalysis
