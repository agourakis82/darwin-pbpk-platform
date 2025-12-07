"""
MC-Dropout for Epistemic Uncertainty Quantification

SOTA Q1 2025 Implementation:
- Monte Carlo Dropout for neural network uncertainty
- Epistemic uncertainty estimation without retraining
- Prediction intervals from multiple forward passes
- Integration with GNN/PBPK models

MC-Dropout (Gal & Ghahramani, 2016) provides a Bayesian approximation
by keeping dropout active during inference and running multiple forward
passes to estimate predictive uncertainty.

Author: Dr. Demetrios Agourakis + AI Assistant
Date: December 2025

References:
- Gal & Ghahramani, 2016 (Dropout as a Bayesian Approximation)
- Kendall & Gal, 2017 (What Uncertainties Do We Need?)
"""

module MCDropout

using Flux
using Statistics
using Random
using Distributions

export MCDropoutWrapper, predict_with_uncertainty
export UncertaintyResult, decompose_uncertainty
export calibration_metrics, prediction_intervals

#=============================================================================
  Uncertainty Result Structure
=============================================================================#

"""
Result structure for uncertainty-aware predictions.

Contains point estimates and uncertainty decomposition.
"""
struct UncertaintyResult
    mean::Vector{Float64}       # Mean prediction
    std::Vector{Float64}        # Total standard deviation
    epistemic::Vector{Float64}  # Epistemic (model) uncertainty
    aleatoric::Vector{Float64}  # Aleatoric (data) uncertainty (if available)
    samples::Matrix{Float64}    # Raw MC samples [n_samples, n_predictions]
    n_samples::Int              # Number of MC samples used
    confidence_level::Float64   # CI level (default 0.95)
    ci_lower::Vector{Float64}   # Lower confidence bound
    ci_upper::Vector{Float64}   # Upper confidence bound
end

"""
Create UncertaintyResult from MC samples.

Samples should be a matrix where:
- First dimension is number of MC samples
- Second dimension is number of predictions

Returns UncertaintyResult with Vector{Float64} fields.
"""
function UncertaintyResult(
    samples::AbstractArray;
    confidence_level::Float64 = 0.95,
    aleatoric::Union{AbstractArray, Nothing} = nothing
)
    α = (1 - confidence_level) / 2
    n_samples = size(samples, 1)
    n_predictions = size(samples, 2)

    # Compute statistics - ensure Vector{Float64} output
    mean_pred = Vector{Float64}(vec(mean(samples, dims=1)))
    std_pred = Vector{Float64}(vec(std(samples, dims=1)))

    # Epistemic uncertainty = variance of predictions
    epistemic = copy(std_pred)

    # Aleatoric uncertainty (if provided)
    if aleatoric === nothing
        aleatoric_val = zeros(Float64, n_predictions)
    else
        aleatoric_val = Vector{Float64}(vec(aleatoric))
    end

    # Confidence intervals from samples
    ci_lower = Vector{Float64}([quantile(samples[:, j], α) for j in 1:n_predictions])
    ci_upper = Vector{Float64}([quantile(samples[:, j], 1 - α) for j in 1:n_predictions])

    # Convert samples to Float64 matrix
    samples_f64 = Matrix{Float64}(samples)

    return UncertaintyResult(
        mean_pred,
        std_pred,
        epistemic,
        aleatoric_val,
        samples_f64,
        n_samples,
        confidence_level,
        ci_lower,
        ci_upper
    )
end

#=============================================================================
  MC-Dropout Wrapper
=============================================================================#

"""
MC-Dropout wrapper for any Flux model.

Wraps a model to enable uncertainty estimation via Monte Carlo dropout.
The wrapper ensures dropout remains active during inference.

# Usage
```julia
model = Chain(Dense(10, 64, relu), Dropout(0.1), Dense(64, 1))
mc_model = MCDropoutWrapper(model; dropout_rate=0.1)

# Single prediction with uncertainty
result = predict_with_uncertainty(mc_model, x; n_samples=50)
println("Prediction: \$(result.mean) ± \$(result.std)")
```
"""
struct MCDropoutWrapper{M}
    model::M
    dropout_rate::Float64

    function MCDropoutWrapper(model::M; dropout_rate::Float64 = 0.1) where M
        if dropout_rate < 0 || dropout_rate > 1
            throw(ArgumentError("dropout_rate must be in [0, 1]"))
        end
        new{M}(model, dropout_rate)
    end
end

Flux.@functor MCDropoutWrapper

"""
Standard forward pass (uses model's default behavior).
"""
function (wrapper::MCDropoutWrapper)(x)
    return wrapper.model(x)
end

"""
Forward pass with dropout forced on.

This is the key to MC-Dropout: keeping dropout active during inference.
"""
function forward_with_dropout(wrapper::MCDropoutWrapper, x)
    # Set model to training mode (enables dropout)
    Flux.trainmode!(wrapper.model, true)
    result = wrapper.model(x)
    # Reset to eval mode
    Flux.testmode!(wrapper.model)
    return result
end

"""
Predict with Monte Carlo dropout uncertainty estimation.

# Arguments
- `wrapper::MCDropoutWrapper`: Wrapped model with dropout
- `x`: Input data
- `n_samples::Int`: Number of MC forward passes (default: 50)
- `confidence_level::Float64`: Confidence level for intervals (default: 0.95)

# Returns
- `UncertaintyResult`: Predictions with uncertainty estimates
"""
function predict_with_uncertainty(
    wrapper::MCDropoutWrapper,
    x;
    n_samples::Int = 50,
    confidence_level::Float64 = 0.95
)::UncertaintyResult
    # Collect MC samples
    samples = []

    for _ in 1:n_samples
        pred = forward_with_dropout(wrapper, x)
        push!(samples, pred)
    end

    # Stack samples
    sample_array = stack(samples)  # [n_samples, output_dims...]

    return UncertaintyResult(sample_array; confidence_level = confidence_level)
end

"""
Batch prediction with uncertainty.

Processes multiple inputs efficiently.
"""
function predict_batch_with_uncertainty(
    wrapper::MCDropoutWrapper,
    X::AbstractMatrix;  # [features, batch_size]
    n_samples::Int = 50,
    confidence_level::Float64 = 0.95
)::Vector{UncertaintyResult}
    batch_size = size(X, 2)
    results = Vector{UncertaintyResult}(undef, batch_size)

    # Collect all samples for batch
    all_samples = []
    for _ in 1:n_samples
        preds = forward_with_dropout(wrapper, X)
        push!(all_samples, preds)
    end

    # Stack and reshape: [n_samples, output_dim, batch_size]
    sample_array = stack(all_samples)

    # Create result for each batch item
    for i in 1:batch_size
        item_samples = sample_array[:, :, i]
        results[i] = UncertaintyResult(item_samples; confidence_level = confidence_level)
    end

    return results
end

export predict_batch_with_uncertainty

#=============================================================================
  Uncertainty Decomposition
=============================================================================#

"""
Decompose total uncertainty into epistemic and aleatoric components.

For models that output both mean and variance (heteroscedastic models),
we can separate:
- Epistemic: Variance of the means (model uncertainty)
- Aleatoric: Mean of the variances (data uncertainty)

# Arguments
- `mean_samples`: MC samples of mean predictions [n_samples, ...]
- `var_samples`: MC samples of variance predictions [n_samples, ...]

# Returns
Dict with:
- total: Total predictive variance
- epistemic: Model uncertainty
- aleatoric: Data/noise uncertainty
- ratio: Epistemic / Total (high = model uncertain, low = data noisy)
"""
function decompose_uncertainty(
    mean_samples::AbstractArray,
    var_samples::AbstractArray
)::Dict{String, Any}
    # Epistemic = Var(E[y|x,w]) - variance of means
    epistemic = var(mean_samples, dims=1)

    # Aleatoric = E[Var(y|x,w)] - mean of variances
    aleatoric = mean(var_samples, dims=1)

    # Total = Epistemic + Aleatoric
    total = epistemic .+ aleatoric

    # Ratio
    ratio = epistemic ./ (total .+ 1e-10)

    return Dict(
        "total" => dropdims(total, dims=1),
        "epistemic" => dropdims(epistemic, dims=1),
        "aleatoric" => dropdims(aleatoric, dims=1),
        "ratio" => dropdims(ratio, dims=1),
    )
end

#=============================================================================
  Calibration Metrics
=============================================================================#

"""
Compute calibration metrics for uncertainty estimates.

A well-calibrated model should have:
- 95% of true values within 95% CI
- Flat reliability diagram

# Arguments
- `predictions`: Vector of UncertaintyResult
- `true_values`: Vector of observed values

# Returns
Dict with calibration metrics
"""
function calibration_metrics(
    predictions::Vector{<:UncertaintyResult},
    true_values::AbstractVector
)::Dict{String, Any}
    n = length(predictions)
    @assert length(true_values) == n "Length mismatch"

    # Extract means and stds
    pred_means = [p.mean isa Number ? p.mean : p.mean[1] for p in predictions]
    pred_stds = [p.std isa Number ? p.std : p.std[1] for p in predictions]

    # Compute z-scores
    z_scores = (true_values .- pred_means) ./ (pred_stds .+ 1e-10)

    # Expected vs observed coverage at different levels
    coverage_levels = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
    expected_coverage = Float64[]
    observed_coverage = Float64[]

    for level in coverage_levels
        z_crit = quantile(Normal(), (1 + level) / 2)
        expected = level
        observed = mean(abs.(z_scores) .< z_crit)
        push!(expected_coverage, expected)
        push!(observed_coverage, observed)
    end

    # Expected Calibration Error (ECE)
    ece = mean(abs.(expected_coverage .- observed_coverage))

    # Mean Calibration Error at 95%
    mce_95 = abs(0.95 - observed_coverage[5])

    # Sharpness (average CI width)
    ci_widths = [p.ci_upper isa Number ?
                 p.ci_upper - p.ci_lower :
                 mean(p.ci_upper .- p.ci_lower)
                 for p in predictions]
    sharpness = mean(ci_widths)

    # RMSE of z-scores from N(0,1)
    z_rmse = sqrt(mean(z_scores.^2))

    return Dict(
        "ECE" => ece,
        "MCE_95" => mce_95,
        "sharpness" => sharpness,
        "z_score_mean" => mean(z_scores),
        "z_score_std" => std(z_scores),
        "z_score_rmse" => z_rmse,
        "coverage_levels" => coverage_levels,
        "expected_coverage" => expected_coverage,
        "observed_coverage" => observed_coverage,
        "is_well_calibrated" => ece < 0.05 && abs(z_rmse - 1.0) < 0.2,
    )
end

"""
Generate data for reliability diagram plotting.
"""
function reliability_diagram_data(
    predictions::Vector{<:UncertaintyResult},
    true_values::AbstractVector;
    n_bins::Int = 10
)::Dict{String, Vector{Float64}}
    pred_means = [p.mean isa Number ? p.mean : p.mean[1] for p in predictions]
    pred_stds = [p.std isa Number ? p.std : p.std[1] for p in predictions]

    expected = collect(range(0.1, 0.99, length=n_bins))
    observed = Float64[]

    z_scores = (true_values .- pred_means) ./ (pred_stds .+ 1e-10)

    for level in expected
        z_crit = quantile(Normal(), (1 + level) / 2)
        coverage = mean(abs.(z_scores) .< z_crit)
        push!(observed, coverage)
    end

    return Dict(
        "expected" => expected,
        "observed" => observed,
        "perfect" => expected,  # Diagonal line
    )
end

export reliability_diagram_data

#=============================================================================
  Prediction Intervals
=============================================================================#

"""
Compute prediction intervals from MC samples.

# Arguments
- `result::UncertaintyResult`: MC dropout result
- `confidence_levels::Vector{Float64}`: CI levels to compute

# Returns
Dict mapping level → (lower, upper) bounds
"""
function prediction_intervals(
    result::UncertaintyResult;
    confidence_levels::Vector{Float64} = [0.5, 0.75, 0.9, 0.95]
)::Dict{Float64, Tuple}
    intervals = Dict{Float64, Tuple}()

    for level in confidence_levels
        α = (1 - level) / 2
        lower = mapslices(x -> quantile(x, α), result.samples, dims=1)
        upper = mapslices(x -> quantile(x, 1 - α), result.samples, dims=1)
        intervals[level] = (dropdims(lower, dims=1), dropdims(upper, dims=1))
    end

    return intervals
end

#=============================================================================
  PBPK-Specific Functions
=============================================================================#

"""
MC-Dropout for PBPK GNN predictions.

Specialized wrapper for DynamicPBPKGNN that handles:
- Multi-organ predictions
- Time-series outputs
- PK parameter uncertainty
"""
function pbpk_predict_with_uncertainty(
    gnn_forward_fn::Function,
    doses::Vector{Float64},
    params::Vector,
    time_points::Vector{Vector{Float64}};
    n_samples::Int = 50,
    confidence_level::Float64 = 0.95
)::Dict{String, Any}
    batch_size = length(doses)

    # Collect MC samples
    all_predictions = []

    for _ in 1:n_samples
        # Forward pass with dropout active
        result = gnn_forward_fn(doses, params, time_points)
        push!(all_predictions, result["concentrations"])
    end

    # Stack: [n_samples, batch_size, n_organs, n_times]
    pred_stack = cat(all_predictions..., dims=4)
    pred_stack = permutedims(pred_stack, (4, 1, 2, 3))

    # Compute statistics
    pred_mean = dropdims(mean(pred_stack, dims=1), dims=1)
    pred_std = dropdims(std(pred_stack, dims=1), dims=1)

    # Confidence intervals
    α = (1 - confidence_level) / 2
    ci_lower = dropdims(mapslices(x -> quantile(x, α), pred_stack, dims=1), dims=1)
    ci_upper = dropdims(mapslices(x -> quantile(x, 1 - α), pred_stack, dims=1), dims=1)

    return Dict{String, Any}(
        "mean" => pred_mean,
        "std" => pred_std,
        "epistemic_uncertainty" => pred_std,  # Epistemic from MC variance
        "ci_lower" => ci_lower,
        "ci_upper" => ci_upper,
        "n_samples" => n_samples,
        "confidence_level" => confidence_level,
        "all_samples" => pred_stack,
    )
end

"""
Compute PK parameter uncertainties (Cmax, AUC, Tmax) from MC samples.
"""
function pk_parameter_uncertainty(
    pbpk_result::Dict{String, Any},
    time_points::Vector{Float64};
    blood_idx::Int = 1
)::Dict{String, Any}
    pred_stack = pbpk_result["all_samples"]  # [n_samples, batch_size, n_organs, n_times]
    n_samples = size(pred_stack, 1)
    batch_size = size(pred_stack, 2)

    # Extract blood concentrations
    cmax_samples = zeros(n_samples, batch_size)
    tmax_samples = zeros(n_samples, batch_size)
    auc_samples = zeros(n_samples, batch_size)

    for mc in 1:n_samples
        for b in 1:batch_size
            conc_profile = vec(pred_stack[mc, b, blood_idx, :])

            # Cmax
            cmax_samples[mc, b] = maximum(conc_profile)

            # Tmax
            tmax_samples[mc, b] = time_points[argmax(conc_profile)]

            # AUC (trapezoidal)
            auc = 0.0
            for t in 2:length(time_points)
                dt = time_points[t] - time_points[t-1]
                auc += 0.5 * (conc_profile[t] + conc_profile[t-1]) * dt
            end
            auc_samples[mc, b] = auc
        end
    end

    # Create uncertainty results for each parameter
    function summarize_param(samples)
        mean_val = vec(mean(samples, dims=1))
        std_val = vec(std(samples, dims=1))
        ci_lower = vec(mapslices(x -> quantile(x, 0.025), samples, dims=1))
        ci_upper = vec(mapslices(x -> quantile(x, 0.975), samples, dims=1))
        return Dict(
            "mean" => mean_val,
            "std" => std_val,
            "ci_lower" => ci_lower,
            "ci_upper" => ci_upper,
        )
    end

    return Dict{String, Any}(
        "cmax" => summarize_param(cmax_samples),
        "tmax" => summarize_param(tmax_samples),
        "auc" => summarize_param(auc_samples),
    )
end

export pbpk_predict_with_uncertainty, pk_parameter_uncertainty

#=============================================================================
  Utilities
=============================================================================#

"""
Add dropout layers to an existing model for MC-Dropout.

Inserts Dropout layers after each Dense layer in a Chain.
"""
function add_dropout(model::Chain; dropout_rate::Float64 = 0.1)::Chain
    layers = []
    for layer in model.layers
        push!(layers, layer)
        if layer isa Dense
            push!(layers, Dropout(dropout_rate))
        end
    end
    return Chain(layers...)
end

"""
Check if a model has dropout layers.
"""
function has_dropout(model)::Bool
    for layer in Flux.modules(model)
        if layer isa Dropout
            return true
        end
    end
    return false
end

export add_dropout, has_dropout

end # module
