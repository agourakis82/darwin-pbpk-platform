"""
Uncertainty Calibration Metrics for PBPK Models

SOTA Q1 2025 Implementation:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability diagrams
- Continuous Ranked Probability Score (CRPS)
- Sharpness metrics

Proper uncertainty calibration is essential for regulatory acceptance.
A well-calibrated model's 95% credible intervals should contain 95%
of true values.

Author: Dr. Sounio Agourakis + AI Assistant
Date: December 2025

References:
- Guo et al., 2017 (On Calibration of Modern Neural Networks)
- Kuleshov et al., 2018 (Accurate Uncertainties for Deep Learning)
- Gneiting & Raftery, 2007 (Strictly Proper Scoring Rules)
"""

module Calibration

using Statistics
using Distributions
using Random
using LinearAlgebra

export CalibrationResult, CalibrationMetrics
export expected_calibration_error, maximum_calibration_error
export reliability_diagram, calibration_curve
export continuous_ranked_probability_score, sharpness
export recalibrate_predictions, isotonic_calibration
export full_calibration_analysis

#=============================================================================
  Data Structures
=============================================================================#

"""
Result of calibration analysis.
"""
struct CalibrationResult
    ece::Float64                  # Expected Calibration Error
    mce::Float64                  # Maximum Calibration Error
    sharpness::Float64            # Average prediction interval width
    coverage_90::Float64          # Actual coverage at 90% CI
    coverage_95::Float64          # Actual coverage at 95% CI
    crps::Float64                 # Continuous Ranked Probability Score
    is_well_calibrated::Bool      # ECE < 0.05 and proper coverage
    reliability_data::Dict{String, Any}  # Contains both vectors and Bool
end

"""
Comprehensive calibration metrics.
"""
struct CalibrationMetrics
    n_samples::Int
    z_scores::Vector{Float64}
    expected_coverage::Vector{Float64}
    observed_coverage::Vector{Float64}
    bin_edges::Vector{Float64}
    bin_counts::Vector{Int}
    ece::Float64
    mce::Float64
end

#=============================================================================
  Core Calibration Metrics
=============================================================================#

"""
Expected Calibration Error (ECE).

Measures the average discrepancy between predicted confidence and
observed accuracy across all confidence bins.

For regression: Computes expected vs observed coverage at different
credible interval levels.

# Arguments
- `pred_means::Vector{Float64}`: Predicted means
- `pred_stds::Vector{Float64}`: Predicted standard deviations
- `observed::Vector{Float64}`: True values
- `n_bins::Int`: Number of confidence bins (default: 10)

# Returns
- ECE value in [0, 1] where 0 = perfect calibration

# Example
```julia
ece = expected_calibration_error(pred_means, pred_stds, observed)
if ece < 0.05
    println("Model is well-calibrated")
end
```
"""
function expected_calibration_error(
    pred_means::AbstractVector{Float64},
    pred_stds::AbstractVector{Float64},
    observed::AbstractVector{Float64};
    n_bins::Int = 10
)::Float64
    n = length(pred_means)
    @assert length(pred_stds) == n && length(observed) == n "Length mismatch"

    # Compute z-scores
    z_scores = (observed .- pred_means) ./ (pred_stds .+ 1e-10)

    # Coverage levels to check
    coverage_levels = range(0.1, 0.95, length=n_bins)

    total_error = 0.0
    total_weight = 0.0

    for level in coverage_levels
        # Expected: level fraction should be within interval
        z_crit = quantile(Normal(), (1 + level) / 2)

        # Observed coverage
        observed_coverage = mean(abs.(z_scores) .< z_crit)

        # Weighted error (weight by level to emphasize high-confidence regions)
        error = abs(level - observed_coverage)
        weight = level

        total_error += weight * error
        total_weight += weight
    end

    return total_error / total_weight
end

"""
Maximum Calibration Error (MCE).

The maximum discrepancy between predicted and observed confidence
across all bins. More sensitive to worst-case miscalibration.
"""
function maximum_calibration_error(
    pred_means::AbstractVector{Float64},
    pred_stds::AbstractVector{Float64},
    observed::AbstractVector{Float64};
    n_bins::Int = 10
)::Float64
    n = length(pred_means)
    z_scores = (observed .- pred_means) ./ (pred_stds .+ 1e-10)

    coverage_levels = range(0.1, 0.99, length=n_bins)
    max_error = 0.0

    for level in coverage_levels
        z_crit = quantile(Normal(), (1 + level) / 2)
        observed_coverage = mean(abs.(z_scores) .< z_crit)
        error = abs(level - observed_coverage)
        max_error = max(max_error, error)
    end

    return max_error
end

"""
Compute calibration metrics at multiple levels.
"""
function calibration_metrics(
    pred_means::AbstractVector{Float64},
    pred_stds::AbstractVector{Float64},
    observed::AbstractVector{Float64};
    n_bins::Int = 10
)::CalibrationMetrics
    n = length(pred_means)
    z_scores = (observed .- pred_means) ./ (pred_stds .+ 1e-10)

    coverage_levels = collect(range(0.1, 0.99, length=n_bins))
    observed_coverage = Float64[]

    for level in coverage_levels
        z_crit = quantile(Normal(), (1 + level) / 2)
        push!(observed_coverage, mean(abs.(z_scores) .< z_crit))
    end

    ece = expected_calibration_error(pred_means, pred_stds, observed; n_bins=n_bins)
    mce = maximum_calibration_error(pred_means, pred_stds, observed; n_bins=n_bins)

    # Bin counts for z-scores
    bin_edges = collect(range(-4, 4, length=n_bins+1))
    bin_counts = zeros(Int, n_bins)
    for z in z_scores
        for i in 1:n_bins
            if bin_edges[i] <= z < bin_edges[i+1]
                bin_counts[i] += 1
                break
            end
        end
    end

    return CalibrationMetrics(
        n,
        z_scores,
        coverage_levels,
        observed_coverage,
        bin_edges,
        bin_counts,
        ece,
        mce
    )
end

export calibration_metrics

#=============================================================================
  Reliability Diagram
=============================================================================#

"""
Generate data for reliability diagram visualization.

A reliability diagram plots expected confidence vs observed accuracy.
Perfect calibration = diagonal line.

# Returns
Dict with:
- expected: Expected coverage levels
- observed: Observed coverage at each level
- perfect: Diagonal reference line
- gap: Calibration gap at each level
"""
function reliability_diagram(
    pred_means::AbstractVector{Float64},
    pred_stds::AbstractVector{Float64},
    observed::AbstractVector{Float64};
    n_bins::Int = 10
)::Dict{String, Any}
    z_scores = (observed .- pred_means) ./ (pred_stds .+ 1e-10)

    expected = collect(range(0.1, 0.99, length=n_bins))
    observed_cov = Float64[]

    for level in expected
        z_crit = quantile(Normal(), (1 + level) / 2)
        push!(observed_cov, mean(abs.(z_scores) .< z_crit))
    end

    gap = observed_cov .- expected

    return Dict{String, Any}(
        "expected" => expected,
        "observed" => observed_cov,
        "perfect" => expected,
        "gap" => gap,
        "is_overconfident" => mean(gap) < 0,  # Observed < Expected = overconfident
    )
end

"""
Generate calibration curve (alternative visualization).

Plots sorted predictions against their empirical CDF.
"""
function calibration_curve(
    pred_means::AbstractVector{Float64},
    pred_stds::AbstractVector{Float64},
    observed::AbstractVector{Float64}
)::Dict{String, Vector{Float64}}
    n = length(pred_means)

    # Compute cumulative probability for each observation
    cdf_values = [cdf(Normal(pred_means[i], pred_stds[i] + 1e-10), observed[i]) for i in 1:n]

    # Sort CDF values
    sorted_cdf = sort(cdf_values)

    # Empirical CDF (uniform if well-calibrated)
    empirical = collect(range(0, 1, length=n))

    return Dict(
        "theoretical" => sorted_cdf,
        "empirical" => empirical,
        "ks_statistic" => maximum(abs.(sorted_cdf .- empirical)),
    )
end

#=============================================================================
  Scoring Rules
=============================================================================#

"""
Continuous Ranked Probability Score (CRPS).

A strictly proper scoring rule for probabilistic predictions.
Lower is better. For Gaussian predictive distributions:

CRPS = σ * (z * (2Φ(z) - 1) + 2φ(z) - 1/√π)

where z = (y - μ) / σ

# Arguments
- `pred_mean::Float64`: Predicted mean
- `pred_std::Float64`: Predicted standard deviation
- `observed::Float64`: True value

# Returns
- CRPS value (lower = better)
"""
function crps_gaussian(
    pred_mean::Float64,
    pred_std::Float64,
    observed::Float64
)::Float64
    σ = pred_std + 1e-10
    z = (observed - pred_mean) / σ

    Φ_z = cdf(Normal(), z)
    φ_z = pdf(Normal(), z)

    return σ * (z * (2 * Φ_z - 1) + 2 * φ_z - 1 / sqrt(π))
end

"""
Mean CRPS over multiple predictions.
"""
function continuous_ranked_probability_score(
    pred_means::AbstractVector{Float64},
    pred_stds::AbstractVector{Float64},
    observed::AbstractVector{Float64}
)::Float64
    n = length(pred_means)
    crps_values = [crps_gaussian(pred_means[i], pred_stds[i], observed[i]) for i in 1:n]
    return mean(crps_values)
end

"""
Interval Score (for prediction intervals).

Penalizes both miscoverage and interval width.

IS_α(l, u, y) = (u - l) + (2/α)(l - y)⁺ + (2/α)(y - u)⁺
"""
function interval_score(
    lower::Float64,
    upper::Float64,
    observed::Float64;
    alpha::Float64 = 0.05  # For 95% CI
)::Float64
    width = upper - lower

    # Penalties for being outside interval
    penalty_low = max(0.0, lower - observed) * (2 / alpha)
    penalty_high = max(0.0, observed - upper) * (2 / alpha)

    return width + penalty_low + penalty_high
end

"""
Mean interval score over multiple predictions.
"""
function mean_interval_score(
    lowers::AbstractVector{Float64},
    uppers::AbstractVector{Float64},
    observed::AbstractVector{Float64};
    alpha::Float64 = 0.05
)::Float64
    n = length(observed)
    scores = [interval_score(lowers[i], uppers[i], observed[i]; alpha=alpha) for i in 1:n]
    return mean(scores)
end

export crps_gaussian, interval_score, mean_interval_score

#=============================================================================
  Sharpness Metrics
=============================================================================#

"""
Sharpness - average width of prediction intervals.

Sharper (narrower) intervals are better, conditional on being calibrated.
"""
function sharpness(
    pred_stds::AbstractVector{Float64};
    confidence_level::Float64 = 0.95
)::Float64
    z = quantile(Normal(), (1 + confidence_level) / 2)
    interval_widths = 2 * z .* pred_stds
    return mean(interval_widths)
end

"""
Coefficient of Variation of interval widths.

Measures consistency of uncertainty estimates.
Lower CV = more consistent uncertainty.
"""
function uncertainty_cv(pred_stds::AbstractVector{Float64})::Float64
    return std(pred_stds) / (mean(pred_stds) + 1e-10)
end

export uncertainty_cv

#=============================================================================
  Recalibration Methods
=============================================================================#

"""
Simple scaling recalibration.

Finds a scaling factor s such that s * σ gives proper coverage.
"""
function recalibrate_scaling(
    pred_means::AbstractVector{Float64},
    pred_stds::AbstractVector{Float64},
    observed::AbstractVector{Float64};
    target_coverage::Float64 = 0.95
)::Float64
    z_scores = (observed .- pred_means) ./ (pred_stds .+ 1e-10)
    z_crit = quantile(Normal(), (1 + target_coverage) / 2)

    # Find scaling factor
    # We want: P(|z/s| < z_crit) = target_coverage
    # This means s = quantile(|z|, target_coverage) / z_crit

    abs_z_sorted = sort(abs.(z_scores))
    empirical_quantile = abs_z_sorted[Int(ceil(target_coverage * length(z_scores)))]

    scale_factor = empirical_quantile / z_crit

    return scale_factor
end

"""
Apply recalibration to predictions.
"""
function recalibrate_predictions(
    pred_means::AbstractVector{Float64},
    pred_stds::AbstractVector{Float64},
    observed::AbstractVector{Float64}
)::Vector{Float64}
    scale = recalibrate_scaling(pred_means, pred_stds, observed)
    return pred_stds .* scale
end

"""
Isotonic regression for calibration.

Non-parametric calibration using isotonic regression.
"""
function isotonic_calibration(
    predicted_confidences::AbstractVector{Float64},
    actual_outcomes::AbstractVector{Float64}  # 1 if within predicted CI, 0 otherwise
)::Vector{Float64}
    n = length(predicted_confidences)

    # Sort by predicted confidence
    sorted_idx = sortperm(predicted_confidences)
    sorted_pred = predicted_confidences[sorted_idx]
    sorted_actual = actual_outcomes[sorted_idx]

    # Pool Adjacent Violators Algorithm (PAVA)
    calibrated = copy(sorted_actual)

    i = 1
    while i < n
        j = i
        # Find pool of violators
        while j < n && calibrated[j] > calibrated[j + 1]
            j += 1
        end

        if j > i
            # Average the pool
            pool_mean = mean(calibrated[i:j])
            calibrated[i:j] .= pool_mean
        end
        i = j + 1
    end

    # Restore original order
    result = similar(calibrated)
    for (orig_idx, sorted_i) in enumerate(sorted_idx)
        result[sorted_i] = calibrated[orig_idx]
    end

    return result
end

#=============================================================================
  Full Calibration Analysis
=============================================================================#

"""
Comprehensive calibration analysis.

Computes all calibration metrics and returns a complete CalibrationResult.

# Arguments
- `pred_means`: Predicted means
- `pred_stds`: Predicted standard deviations
- `observed`: True values
- `n_bins`: Number of bins for ECE/reliability diagram

# Returns
CalibrationResult with all metrics and diagnostic data
"""
function full_calibration_analysis(
    pred_means::AbstractVector{Float64},
    pred_stds::AbstractVector{Float64},
    observed::AbstractVector{Float64};
    n_bins::Int = 10
)::CalibrationResult
    # Core metrics
    ece = expected_calibration_error(pred_means, pred_stds, observed; n_bins=n_bins)
    mce = maximum_calibration_error(pred_means, pred_stds, observed; n_bins=n_bins)

    # Sharpness
    sharp = sharpness(pred_stds)

    # Coverage at key levels
    z_scores = (observed .- pred_means) ./ (pred_stds .+ 1e-10)

    z_90 = quantile(Normal(), 0.95)  # For 90% CI
    z_95 = quantile(Normal(), 0.975)  # For 95% CI

    coverage_90 = mean(abs.(z_scores) .< z_90)
    coverage_95 = mean(abs.(z_scores) .< z_95)

    # CRPS
    crps = continuous_ranked_probability_score(pred_means, pred_stds, observed)

    # Reliability diagram data
    rel_data = reliability_diagram(pred_means, pred_stds, observed; n_bins=n_bins)

    # Well-calibrated if:
    # 1. ECE < 0.05
    # 2. 95% CI coverage between 0.90 and 1.0
    # 3. 90% CI coverage between 0.85 and 0.95
    is_calibrated = ece < 0.05 &&
                    0.90 <= coverage_95 <= 1.0 &&
                    0.85 <= coverage_90 <= 0.95

    return CalibrationResult(
        ece,
        mce,
        sharp,
        coverage_90,
        coverage_95,
        crps,
        is_calibrated,
        rel_data
    )
end

#=============================================================================
  PK-Specific Calibration
=============================================================================#

"""
Calibration analysis for PK parameters (log-space).

Many PK parameters are log-normally distributed, so calibration
should be assessed in log-space.
"""
function pk_calibration_analysis(
    pred_means::AbstractVector{Float64},
    pred_stds::AbstractVector{Float64},
    observed::AbstractVector{Float64};
    log_transform::Bool = true,
    n_bins::Int = 10
)::CalibrationResult
    if log_transform
        # Transform to log-space
        log_pred = log.(pred_means .+ 1e-10)
        log_obs = log.(observed .+ 1e-10)

        # Approximate log-space std using delta method
        # Var(log(X)) ≈ (σ/μ)² for small CV
        log_stds = pred_stds ./ (pred_means .+ 1e-10)

        return full_calibration_analysis(log_pred, log_stds, log_obs; n_bins=n_bins)
    else
        return full_calibration_analysis(pred_means, pred_stds, observed; n_bins=n_bins)
    end
end

"""
Multi-parameter calibration summary.

Analyzes calibration for multiple PK parameters (CL, Vd, F, etc.).
"""
function multi_parameter_calibration(
    predictions::Dict{String, Tuple{Vector{Float64}, Vector{Float64}}},  # param => (means, stds)
    observations::Dict{String, Vector{Float64}};  # param => observed
    log_params::Vector{String} = ["CL", "Vd", "t_half"]  # Parameters to analyze in log-space
)::Dict{String, CalibrationResult}
    results = Dict{String, CalibrationResult}()

    for (param, (means, stds)) in predictions
        if haskey(observations, param)
            obs = observations[param]
            log_transform = param in log_params
            results[param] = pk_calibration_analysis(means, stds, obs; log_transform=log_transform)
        end
    end

    return results
end

export pk_calibration_analysis, multi_parameter_calibration

end # module
