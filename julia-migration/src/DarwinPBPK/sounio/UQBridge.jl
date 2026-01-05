# ===========================================================================
# UQ BRIDGE MODULE
# ===========================================================================
# Bridge between Julia UQ methods and Sounio epistemic computing.
#
# This module provides conversion between:
# - Bayesian posterior distributions → Knowledge{T}
# - MC-Dropout uncertainty → Knowledge{T}
# - Evidential learning output → Knowledge{T}
# - Ensemble predictions → Knowledge{T}
#
# Author: Dr. Sounio Agourakis
# Date: January 2026
# Version: 1.0.0
# ===========================================================================

module UQBridge

using Statistics
using Distributions

# Import from sibling module
include("EpistemicTypes.jl")
using .EpistemicTypes

export bayesian_to_knowledge, mcdropout_to_knowledge, evidential_to_knowledge
export ensemble_to_knowledge, bootstrap_to_knowledge
export UQResult, convert_uq_result
export confidence_from_std, confidence_from_cv, confidence_from_credible_interval
export PosteriorSummary, summarize_posterior

# ===========================================================================
# Confidence Conversion Functions
# ===========================================================================

"""
    confidence_from_std(mean::Float64, std::Float64; max_cv::Float64=1.0) -> Float64

Convert standard deviation to confidence score.

Uses coefficient of variation (CV = std/|mean|) mapped to [0, 1].
Lower CV = higher confidence.

# Formula
ε = 1 - min(CV / max_cv, 1.0)

# Arguments
- `mean::Float64`: Mean value
- `std::Float64`: Standard deviation
- `max_cv::Float64`: Maximum CV considered (CV ≥ max_cv → ε = 0)
"""
function confidence_from_std(mean::Float64, std::Float64; max_cv::Float64=1.0)::Float64
    if abs(mean) < 1e-10
        # For near-zero mean, use absolute uncertainty
        return max(0.0, 1.0 - std)
    end

    cv = std / abs(mean)
    return max(0.0, min(1.0, 1.0 - cv / max_cv))
end

"""
    confidence_from_cv(cv::Float64; max_cv::Float64=1.0) -> Float64

Convert coefficient of variation to confidence score.
"""
function confidence_from_cv(cv::Float64; max_cv::Float64=1.0)::Float64
    return max(0.0, min(1.0, 1.0 - cv / max_cv))
end

"""
    confidence_from_credible_interval(mean::Float64, lower::Float64, upper::Float64; coverage::Float64=0.95) -> Float64

Convert credible interval width to confidence score.

Narrower intervals (relative to mean) = higher confidence.
"""
function confidence_from_credible_interval(
    mean::Float64,
    lower::Float64,
    upper::Float64;
    coverage::Float64=0.95
)::Float64
    if abs(mean) < 1e-10
        width = upper - lower
        return max(0.0, 1.0 - width)
    end

    width = upper - lower
    relative_width = width / abs(mean)

    # Map relative width to confidence
    # width = 0 → ε = 1.0
    # width = 2*mean → ε = 0.5
    # width = 4*mean → ε = 0.0
    return max(0.0, min(1.0, 1.0 - relative_width / 4.0))
end

# ===========================================================================
# Bayesian Posterior Conversion
# ===========================================================================

"""
    PosteriorSummary

Summary statistics from a Bayesian posterior distribution.
"""
struct PosteriorSummary
    mean::Float64
    median::Float64
    std::Float64
    lower_ci::Float64  # Lower credible interval
    upper_ci::Float64  # Upper credible interval
    ci_coverage::Float64
    ess::Float64       # Effective sample size
    rhat::Float64      # Convergence diagnostic
end

"""
    summarize_posterior(samples::Vector{Float64}; ci_coverage::Float64=0.95) -> PosteriorSummary

Compute summary statistics from MCMC samples.
"""
function summarize_posterior(samples::Vector{Float64}; ci_coverage::Float64=0.95)::PosteriorSummary
    n = length(samples)

    mean_val = mean(samples)
    median_val = median(samples)
    std_val = std(samples)

    # Credible interval
    alpha = (1.0 - ci_coverage) / 2.0
    lower_ci = quantile(samples, alpha)
    upper_ci = quantile(samples, 1.0 - alpha)

    # Effective sample size (simplified)
    ess = Float64(n)  # Simplified - proper ESS requires autocorrelation

    # R-hat (simplified - single chain)
    rhat = 1.0

    PosteriorSummary(mean_val, median_val, std_val, lower_ci, upper_ci, ci_coverage, ess, rhat)
end

"""
    bayesian_to_knowledge(samples::Vector{Float64}; ci_coverage::Float64=0.95) -> Knowledge{Float64}

Convert Bayesian posterior samples to Knowledge.

# Arguments
- `samples::Vector{Float64}`: MCMC posterior samples
- `ci_coverage::Float64`: Credible interval coverage (default: 95%)

# Returns
Knowledge with:
- value: Posterior mean
- confidence: Based on relative credible interval width
- provenance: COMPUTED
"""
function bayesian_to_knowledge(
    samples::Vector{Float64};
    ci_coverage::Float64=0.95
)::Knowledge{Float64}
    summary = summarize_posterior(samples; ci_coverage=ci_coverage)

    # Confidence from credible interval
    ε = confidence_from_credible_interval(
        summary.mean,
        summary.lower_ci,
        summary.upper_ci;
        coverage=ci_coverage
    )

    # Adjust for convergence (R-hat > 1.1 reduces confidence)
    if summary.rhat > 1.1
        ε *= 0.8
    end

    # Create Knowledge with metadata
    k = Knowledge(summary.mean, ε, COMPUTED)
    k = with_metadata(k, "std", summary.std)
    k = with_metadata(k, "ci_lower", summary.lower_ci)
    k = with_metadata(k, "ci_upper", summary.upper_ci)
    k = with_metadata(k, "ess", summary.ess)
    k = with_metadata(k, "uq_method", "bayesian")

    return k
end

"""
    bayesian_to_knowledge(posterior, param_idx::Int; kwargs...) -> Knowledge{Float64}

Convert specific parameter from Turing.jl posterior to Knowledge.

# Arguments
- `posterior`: Turing.jl Chains object or similar
- `param_idx::Int`: Parameter index in chain
"""
function bayesian_to_knowledge(
    posterior,
    param_idx::Int;
    ci_coverage::Float64=0.95
)::Knowledge{Float64}
    # Extract samples for parameter
    if hasmethod(getindex, Tuple{typeof(posterior), Int})
        samples = vec(posterior[:, param_idx, :])
    else
        # Fallback for different posterior formats
        samples = vec(collect(posterior))
    end

    return bayesian_to_knowledge(samples; ci_coverage=ci_coverage)
end

"""
    bayesian_to_knowledge(posterior, param_name::Symbol; kwargs...) -> Knowledge{Float64}

Convert named parameter from Turing.jl posterior to Knowledge.
"""
function bayesian_to_knowledge(
    posterior,
    param_name::Symbol;
    ci_coverage::Float64=0.95
)::Knowledge{Float64}
    # Try to extract by name
    if hasmethod(getindex, Tuple{typeof(posterior), Symbol})
        samples = vec(posterior[param_name])
    else
        error("Cannot extract parameter $param_name from posterior")
    end

    return bayesian_to_knowledge(samples; ci_coverage=ci_coverage)
end

# ===========================================================================
# MC-Dropout Conversion
# ===========================================================================

"""
    mcdropout_to_knowledge(predictions::Vector{Float64}) -> Knowledge{Float64}

Convert MC-Dropout predictions to Knowledge.

MC-Dropout produces multiple predictions with dropout enabled during inference.
The variance in predictions reflects model uncertainty.

# Arguments
- `predictions::Vector{Float64}`: Multiple forward passes with dropout

# Returns
Knowledge with:
- value: Mean prediction
- confidence: Based on prediction variance
- provenance: ESTIMATED
"""
function mcdropout_to_knowledge(predictions::Vector{Float64})::Knowledge{Float64}
    mean_pred = mean(predictions)
    std_pred = std(predictions)

    # Confidence from standard deviation
    ε = confidence_from_std(mean_pred, std_pred; max_cv=0.5)

    k = Knowledge(mean_pred, ε, ESTIMATED)
    k = with_metadata(k, "std", std_pred)
    k = with_metadata(k, "n_samples", length(predictions))
    k = with_metadata(k, "uq_method", "mc_dropout")

    return k
end

"""
    mcdropout_to_knowledge(predictions::Matrix{Float64}, idx::Int) -> Knowledge{Float64}

Convert MC-Dropout predictions for specific output index.

# Arguments
- `predictions::Matrix{Float64}`: Shape (n_outputs, n_samples)
- `idx::Int`: Output index
"""
function mcdropout_to_knowledge(predictions::Matrix{Float64}, idx::Int)::Knowledge{Float64}
    return mcdropout_to_knowledge(vec(predictions[idx, :]))
end

# ===========================================================================
# Evidential Learning Conversion
# ===========================================================================

"""
    evidential_to_knowledge(result::NamedTuple) -> Knowledge{Float64}

Convert evidential learning output to Knowledge.

Evidential deep learning outputs (μ, ν, α, β) for each prediction,
representing a Normal-Inverse-Gamma distribution.

# Arguments
- `result::NamedTuple`: Evidential output with fields (mu, nu, alpha, beta)

# Returns
Knowledge with confidence based on evidential uncertainty.
"""
function evidential_to_knowledge(result::NamedTuple)::Knowledge{Float64}
    mu = result.mu      # Mean prediction
    nu = result.nu      # Pseudo-count (affects confidence)
    alpha = result.alpha # Shape parameter
    beta = result.beta   # Scale parameter

    # Epistemic uncertainty from evidential parameters
    # Higher nu and alpha → more confident
    # Confidence based on inverse of variance
    var_epistemic = beta / (nu * (alpha - 1))

    if isnan(var_epistemic) || isinf(var_epistemic) || var_epistemic < 0
        var_epistemic = 1.0
    end

    # Map variance to confidence
    ε = confidence_from_std(mu, sqrt(var_epistemic); max_cv=1.0)

    # Adjust by pseudo-count (nu > 1 increases confidence)
    ε = min(1.0, ε * min(2.0, nu))

    k = Knowledge(mu, ε, ESTIMATED)
    k = with_metadata(k, "nu", nu)
    k = with_metadata(k, "alpha", alpha)
    k = with_metadata(k, "beta", beta)
    k = with_metadata(k, "epistemic_var", var_epistemic)
    k = with_metadata(k, "uq_method", "evidential")

    return k
end

"""
    evidential_to_knowledge(mu, nu, alpha, beta) -> Knowledge{Float64}

Direct conversion from evidential parameters.
"""
function evidential_to_knowledge(
    mu::Float64,
    nu::Float64,
    alpha::Float64,
    beta::Float64
)::Knowledge{Float64}
    return evidential_to_knowledge((mu=mu, nu=nu, alpha=alpha, beta=beta))
end

# ===========================================================================
# Ensemble Conversion
# ===========================================================================

"""
    ensemble_to_knowledge(predictions::Vector{Float64}; weights=nothing) -> Knowledge{Float64}

Convert ensemble model predictions to Knowledge.

# Arguments
- `predictions::Vector{Float64}`: Predictions from ensemble members
- `weights::Union{Vector{Float64}, Nothing}`: Optional member weights

# Returns
Knowledge with confidence based on ensemble agreement.
"""
function ensemble_to_knowledge(
    predictions::Vector{Float64};
    weights::Union{Vector{Float64}, Nothing}=nothing
)::Knowledge{Float64}
    if weights === nothing
        mean_pred = mean(predictions)
    else
        @assert length(weights) == length(predictions)
        mean_pred = sum(predictions .* weights) / sum(weights)
    end

    std_pred = std(predictions)

    # Confidence from ensemble agreement
    ε = confidence_from_std(mean_pred, std_pred; max_cv=0.3)

    k = Knowledge(mean_pred, ε, COMPUTED)
    k = with_metadata(k, "std", std_pred)
    k = with_metadata(k, "n_members", length(predictions))
    k = with_metadata(k, "uq_method", "ensemble")

    return k
end

# ===========================================================================
# Bootstrap Conversion
# ===========================================================================

"""
    bootstrap_to_knowledge(samples::Vector{Float64}; ci_coverage::Float64=0.95) -> Knowledge{Float64}

Convert bootstrap samples to Knowledge.

Bootstrap provides non-parametric uncertainty estimates.
"""
function bootstrap_to_knowledge(
    samples::Vector{Float64};
    ci_coverage::Float64=0.95
)::Knowledge{Float64}
    mean_val = mean(samples)
    std_val = std(samples)

    # Bootstrap confidence interval
    alpha = (1.0 - ci_coverage) / 2.0
    lower_ci = quantile(samples, alpha)
    upper_ci = quantile(samples, 1.0 - alpha)

    ε = confidence_from_credible_interval(mean_val, lower_ci, upper_ci; coverage=ci_coverage)

    k = Knowledge(mean_val, ε, COMPUTED)
    k = with_metadata(k, "std", std_val)
    k = with_metadata(k, "ci_lower", lower_ci)
    k = with_metadata(k, "ci_upper", upper_ci)
    k = with_metadata(k, "n_bootstrap", length(samples))
    k = with_metadata(k, "uq_method", "bootstrap")

    return k
end

# ===========================================================================
# Generic UQ Result Conversion
# ===========================================================================

"""
    UQResult

Generic container for UQ results from various methods.
"""
struct UQResult
    mean::Float64
    uncertainty::Float64  # std, variance, or similar
    method::Symbol        # :bayesian, :mc_dropout, :evidential, :ensemble, :bootstrap
    additional::Dict{String, Any}
end

"""
    convert_uq_result(result::UQResult) -> Knowledge{Float64}

Convert generic UQ result to Knowledge.
"""
function convert_uq_result(result::UQResult)::Knowledge{Float64}
    # Interpret uncertainty based on method
    if result.method == :bayesian
        ε = confidence_from_std(result.mean, result.uncertainty; max_cv=1.0)
        prov = COMPUTED
    elseif result.method == :mc_dropout
        ε = confidence_from_std(result.mean, result.uncertainty; max_cv=0.5)
        prov = ESTIMATED
    elseif result.method == :evidential
        ε = confidence_from_std(result.mean, sqrt(result.uncertainty); max_cv=1.0)
        prov = ESTIMATED
    elseif result.method == :ensemble
        ε = confidence_from_std(result.mean, result.uncertainty; max_cv=0.3)
        prov = COMPUTED
    elseif result.method == :bootstrap
        ε = confidence_from_std(result.mean, result.uncertainty; max_cv=1.0)
        prov = COMPUTED
    else
        ε = confidence_from_std(result.mean, result.uncertainty)
        prov = ESTIMATED
    end

    k = Knowledge(result.mean, ε, prov)
    k = with_metadata(k, "uq_method", string(result.method))

    for (key, val) in result.additional
        k = with_metadata(k, key, val)
    end

    return k
end

# ===========================================================================
# Batch Conversion
# ===========================================================================

"""
    convert_uq_batch(results::Vector{UQResult}) -> Vector{Knowledge{Float64}}

Convert batch of UQ results to Knowledge vector.
"""
function convert_uq_batch(results::Vector{UQResult})::Vector{Knowledge{Float64}}
    return [convert_uq_result(r) for r in results]
end

export convert_uq_batch

"""
    bayesian_posterior_to_epistemic_params(posterior; param_names) -> Dict{Symbol, Knowledge{Float64}}

Convert full Bayesian posterior to dictionary of epistemic parameters.

# Arguments
- `posterior`: Turing.jl Chains or similar
- `param_names::Vector{Symbol}`: Parameter names to extract

# Returns
Dict mapping parameter names to Knowledge values.
"""
function bayesian_posterior_to_epistemic_params(
    posterior,
    param_names::Vector{Symbol}
)::Dict{Symbol, Knowledge{Float64}}
    result = Dict{Symbol, Knowledge{Float64}}()

    for name in param_names
        try
            result[name] = bayesian_to_knowledge(posterior, name)
        catch e
            @warn "Failed to convert parameter $name" exception=e
        end
    end

    return result
end

export bayesian_posterior_to_epistemic_params

end  # module UQBridge
