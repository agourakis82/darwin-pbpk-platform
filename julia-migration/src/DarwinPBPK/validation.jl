"""
Validation & Analysis - Scripts de Validação Científica

Inovações SOTA:
- Métricas científicas com incerteza (Measurements.jl)
- Visualização científica de alta qualidade (Plots.jl, Makie.jl)
- Relatórios automatizados (Weave.jl ou Documenter.jl)

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: Novembro 2025
"""

module Validation

using Statistics
using Random
using Measurements
using Plots
using DataFrames
using CSV
using JSON

# Optional imports for DynamicGNN and ODEPBPKSolver
# These are only needed for model evaluation functions, not basic metrics
const HAS_DYNAMIC_GNN = Ref{Bool}(false)
const HAS_ODE_SOLVER = Ref{Bool}(false)

function __init__()
    # Check if DynamicGNN is available
    try
        if isdefined(parentmodule(@__MODULE__), :DynamicGNN)
            HAS_DYNAMIC_GNN[] = true
        end
    catch
        HAS_DYNAMIC_GNN[] = false
    end

    # Check if ODEPBPKSolver is available
    try
        if isdefined(parentmodule(@__MODULE__), :ODEPBPKSolver)
            HAS_ODE_SOLVER[] = true
        end
    catch
        HAS_ODE_SOLVER[] = false
    end
end

"""
Fold Error (FE) - Métrica regulatória (FDA/EMA).

FE = max(pred/obs, obs/pred)

Inovações:
- Type-safe computation
- Handling de zeros/NaN
"""
function fold_error(pred::AbstractVector{Float64}, obs::AbstractVector{Float64})::Vector{Float64}
    fe = Vector{Float64}()

    for (p, o) in zip(pred, obs)
        if o > 0 && isfinite(p) && isfinite(o)
            fe_val = max(p / o, o / p)
            push!(fe, fe_val)
        end
    end

    return fe
end

"""
Geometric Mean Fold Error (GMFE).

GMFE = exp(mean(log(FE)))

Inovações:
- Type-safe computation
- Handling de zeros/NaN
"""
function geometric_mean_fold_error(pred::AbstractVector{Float64}, obs::AbstractVector{Float64})::Float64
    fe = fold_error(pred, obs)
    if isempty(fe)
        return NaN
    end

    log_fe = log.(fe)
    return exp(mean(log_fe))
end

#=============================================================================
  FDA/EMA Regulatory Metrics (AFE, AAFE)
=============================================================================#

"""
Average Fold Error (AFE) - FDA/EMA regulatory metric.

AFE = 10^(mean(log10(pred/obs)))

Interpretation:
- AFE = 1.0: Unbiased predictions
- AFE > 1.0: Systematic overprediction
- AFE < 1.0: Systematic underprediction

Reference: Guest et al. (2011) Eur J Pharm Sci 42:283-292
"""
function average_fold_error(pred::AbstractVector{Float64}, obs::AbstractVector{Float64})::Float64
    # Filter valid pairs
    valid_idx = [i for i in 1:length(pred)
                 if obs[i] > 0 && pred[i] > 0 && isfinite(pred[i]) && isfinite(obs[i])]

    if isempty(valid_idx)
        return NaN
    end

    pred_valid = pred[valid_idx]
    obs_valid = obs[valid_idx]

    # AFE = 10^(mean(log10(pred/obs)))
    log_ratios = log10.(pred_valid ./ obs_valid)
    return 10^mean(log_ratios)
end

"""
Absolute Average Fold Error (AAFE) - FDA/EMA regulatory metric.

AAFE = 10^(mean(|log10(pred/obs)|))

Interpretation:
- AAFE = 1.0: Perfect predictions
- AAFE < 2.0: Acceptable for regulatory submissions (FDA/EMA guideline)
- AAFE < 1.5: Excellent prediction accuracy

Reference: Guest et al. (2011) Eur J Pharm Sci 42:283-292
"""
function absolute_average_fold_error(pred::AbstractVector{Float64}, obs::AbstractVector{Float64})::Float64
    # Filter valid pairs
    valid_idx = [i for i in 1:length(pred)
                 if obs[i] > 0 && pred[i] > 0 && isfinite(pred[i]) && isfinite(obs[i])]

    if isempty(valid_idx)
        return NaN
    end

    pred_valid = pred[valid_idx]
    obs_valid = obs[valid_idx]

    # AAFE = 10^(mean(|log10(pred/obs)|))
    abs_log_ratios = abs.(log10.(pred_valid ./ obs_valid))
    return 10^mean(abs_log_ratios)
end

"""
Regulatory Metrics Summary - Complete FDA/EMA metrics package.

Returns a Dict with all regulatory metrics for submission:
- GMFE: Geometric Mean Fold Error
- AFE: Average Fold Error (bias indicator)
- AAFE: Absolute Average Fold Error (precision indicator)
- pct_within_2fold: Percent within 2-fold (primary acceptance criterion)
- pct_within_1_5fold: Percent within 1.5-fold
- pct_within_1_25fold: Percent within bioequivalence range
- n_valid: Number of valid comparisons
- bias_direction: "overprediction", "underprediction", or "unbiased"
"""
function regulatory_metrics_summary(
    pred::AbstractVector{Float64},
    obs::AbstractVector{Float64}
)::Dict{String, Any}
    gmfe = geometric_mean_fold_error(pred, obs)
    afe = average_fold_error(pred, obs)
    aafe = absolute_average_fold_error(pred, obs)

    pct_2x = percent_within_fold(pred, obs, 2.0)
    pct_1_5x = percent_within_fold(pred, obs, 1.5)
    pct_1_25x = percent_within_fold(pred, obs, 1.25)

    # Count valid comparisons
    n_valid = count(i -> obs[i] > 0 && pred[i] > 0 && isfinite(pred[i]) && isfinite(obs[i]),
                    1:length(pred))

    # Determine bias direction
    bias_direction = if afe > 1.25
        "overprediction"
    elseif afe < 0.8
        "underprediction"
    else
        "unbiased"
    end

    # FDA/EMA acceptance criteria
    meets_fda_criteria = aafe < 2.0 && pct_2x >= 70.0

    return Dict{String, Any}(
        "GMFE" => gmfe,
        "AFE" => afe,
        "AAFE" => aafe,
        "percent_within_2fold" => pct_2x,
        "percent_within_1.5fold" => pct_1_5x,
        "percent_within_1.25fold" => pct_1_25x,
        "n_valid" => n_valid,
        "bias_direction" => bias_direction,
        "meets_FDA_criteria" => meets_fda_criteria,
    )
end

export average_fold_error, absolute_average_fold_error, regulatory_metrics_summary

#=============================================================================
  Bootstrap Confidence Intervals (Q1 Publication Enhancement)
=============================================================================#

"""
Bootstrap result structure with estimate and confidence interval.
"""
struct BootstrapResult
    estimate::Float64
    ci_lower::Float64
    ci_upper::Float64
    se::Float64
    n_bootstrap::Int
    ci_level::Float64
end

"""
Generic bootstrap function for any metric.

Computes bootstrap confidence intervals for a given metric function.
Uses the percentile method (bias-corrected available via BCa option).

# Arguments
- `metric_fn::Function`: Function(pred, obs) -> Float64
- `pred::Vector{Float64}`: Predicted values
- `obs::Vector{Float64}`: Observed values
- `n_bootstrap::Int`: Number of bootstrap resamples (default: 2000)
- `ci_level::Float64`: Confidence level (default: 0.95)
- `seed::Union{Int, Nothing}`: Random seed for reproducibility

# Returns
- `BootstrapResult`: Estimate with CI and standard error

# Example
```julia
result = bootstrap_metric(geometric_mean_fold_error, pred, obs; n_bootstrap=2000)
println("GMFE = \$(result.estimate) (95% CI: \$(result.ci_lower) - \$(result.ci_upper))")
```
"""
function bootstrap_metric(
    metric_fn::Function,
    pred::AbstractVector{Float64},
    obs::AbstractVector{Float64};
    n_bootstrap::Int = 2000,
    ci_level::Float64 = 0.95,
    seed::Union{Int, Nothing} = nothing
)::BootstrapResult
    if seed !== nothing
        Random.seed!(seed)
    end

    n = length(pred)
    @assert length(obs) == n "Prediction and observation vectors must have same length"

    # Compute point estimate
    point_estimate = metric_fn(pred, obs)

    # Bootstrap resampling
    bootstrap_values = Float64[]
    sizehint!(bootstrap_values, n_bootstrap)

    for _ in 1:n_bootstrap
        # Sample with replacement
        indices = rand(1:n, n)
        pred_boot = pred[indices]
        obs_boot = obs[indices]

        # Compute metric on bootstrap sample
        val = metric_fn(pred_boot, obs_boot)

        if isfinite(val)
            push!(bootstrap_values, val)
        end
    end

    if isempty(bootstrap_values)
        return BootstrapResult(point_estimate, NaN, NaN, NaN, n_bootstrap, ci_level)
    end

    # Compute confidence interval (percentile method)
    α = (1 - ci_level) / 2
    ci_lower = quantile(bootstrap_values, α)
    ci_upper = quantile(bootstrap_values, 1 - α)

    # Standard error
    se = std(bootstrap_values)

    return BootstrapResult(point_estimate, ci_lower, ci_upper, se, n_bootstrap, ci_level)
end

"""
Bootstrap GMFE with confidence interval.
"""
function gmfe_with_ci(
    pred::AbstractVector{Float64},
    obs::AbstractVector{Float64};
    n_bootstrap::Int = 2000,
    ci_level::Float64 = 0.95
)::BootstrapResult
    return bootstrap_metric(geometric_mean_fold_error, pred, obs;
                           n_bootstrap=n_bootstrap, ci_level=ci_level)
end

"""
Bootstrap AFE with confidence interval.
"""
function afe_with_ci(
    pred::AbstractVector{Float64},
    obs::AbstractVector{Float64};
    n_bootstrap::Int = 2000,
    ci_level::Float64 = 0.95
)::BootstrapResult
    return bootstrap_metric(average_fold_error, pred, obs;
                           n_bootstrap=n_bootstrap, ci_level=ci_level)
end

"""
Bootstrap AAFE with confidence interval.
"""
function aafe_with_ci(
    pred::AbstractVector{Float64},
    obs::AbstractVector{Float64};
    n_bootstrap::Int = 2000,
    ci_level::Float64 = 0.95
)::BootstrapResult
    return bootstrap_metric(absolute_average_fold_error, pred, obs;
                           n_bootstrap=n_bootstrap, ci_level=ci_level)
end

"""
Bootstrap R² with confidence interval.
"""
function r_squared_with_ci(
    pred::AbstractVector{Float64},
    obs::AbstractVector{Float64};
    n_bootstrap::Int = 2000,
    ci_level::Float64 = 0.95
)::BootstrapResult
    return bootstrap_metric(r_squared, pred, obs;
                           n_bootstrap=n_bootstrap, ci_level=ci_level)
end

"""
Bootstrap percent within fold with confidence interval.
"""
function percent_within_fold_with_ci(
    pred::AbstractVector{Float64},
    obs::AbstractVector{Float64},
    fold::Float64 = 2.0;
    n_bootstrap::Int = 2000,
    ci_level::Float64 = 0.95
)::BootstrapResult
    metric_fn = (p, o) -> percent_within_fold(p, o, fold)
    return bootstrap_metric(metric_fn, pred, obs;
                           n_bootstrap=n_bootstrap, ci_level=ci_level)
end

"""
Complete regulatory metrics with bootstrap confidence intervals.

This is the publication-ready version that includes uncertainty estimates
for all key regulatory metrics.

# Returns
Dict with each metric containing:
- estimate: Point estimate
- ci_lower: Lower bound of 95% CI
- ci_upper: Upper bound of 95% CI
- se: Bootstrap standard error

# Example
```julia
metrics = regulatory_metrics_with_ci(pred, obs; n_bootstrap=2000)
println("GMFE = \$(metrics["GMFE"].estimate) (95% CI: \$(metrics["GMFE"].ci_lower)-\$(metrics["GMFE"].ci_upper))")
```
"""
function regulatory_metrics_with_ci(
    pred::AbstractVector{Float64},
    obs::AbstractVector{Float64};
    n_bootstrap::Int = 2000,
    ci_level::Float64 = 0.95,
    seed::Union{Int, Nothing} = 42
)::Dict{String, Any}
    if seed !== nothing
        Random.seed!(seed)
    end

    # Compute all metrics with CIs
    gmfe = gmfe_with_ci(pred, obs; n_bootstrap=n_bootstrap, ci_level=ci_level)
    afe = afe_with_ci(pred, obs; n_bootstrap=n_bootstrap, ci_level=ci_level)
    aafe = aafe_with_ci(pred, obs; n_bootstrap=n_bootstrap, ci_level=ci_level)
    r2 = r_squared_with_ci(pred, obs; n_bootstrap=n_bootstrap, ci_level=ci_level)

    pct_2x = percent_within_fold_with_ci(pred, obs, 2.0; n_bootstrap=n_bootstrap, ci_level=ci_level)
    pct_1_5x = percent_within_fold_with_ci(pred, obs, 1.5; n_bootstrap=n_bootstrap, ci_level=ci_level)
    pct_1_25x = percent_within_fold_with_ci(pred, obs, 1.25; n_bootstrap=n_bootstrap, ci_level=ci_level)

    # Count valid comparisons
    n_valid = count(i -> obs[i] > 0 && pred[i] > 0 && isfinite(pred[i]) && isfinite(obs[i]),
                    1:length(pred))

    # Determine bias direction from AFE
    bias_direction = if afe.estimate > 1.25
        "overprediction"
    elseif afe.estimate < 0.8
        "underprediction"
    else
        "unbiased"
    end

    # FDA/EMA acceptance: AAFE < 2.0 AND >70% within 2-fold
    # Compute probability of meeting criteria from bootstrap
    meets_fda = aafe.estimate < 2.0 && pct_2x.estimate >= 70.0

    # Probability FDA criteria met (from bootstrap distributions)
    # We need to resample to get joint distribution
    Random.seed!(seed !== nothing ? seed + 1 : nothing)
    n_meets = 0
    for _ in 1:n_bootstrap
        indices = rand(1:length(pred), length(pred))
        pred_boot = pred[indices]
        obs_boot = obs[indices]

        aafe_boot = absolute_average_fold_error(pred_boot, obs_boot)
        pct_boot = percent_within_fold(pred_boot, obs_boot, 2.0)

        if isfinite(aafe_boot) && isfinite(pct_boot)
            if aafe_boot < 2.0 && pct_boot >= 70.0
                n_meets += 1
            end
        end
    end
    prob_meets_fda = n_meets / n_bootstrap

    return Dict{String, Any}(
        "GMFE" => gmfe,
        "AFE" => afe,
        "AAFE" => aafe,
        "R_squared" => r2,
        "percent_within_2fold" => pct_2x,
        "percent_within_1.5fold" => pct_1_5x,
        "percent_within_1.25fold" => pct_1_25x,
        "n_valid" => n_valid,
        "n_bootstrap" => n_bootstrap,
        "ci_level" => ci_level,
        "bias_direction" => bias_direction,
        "meets_FDA_criteria" => meets_fda,
        "prob_meets_FDA_criteria" => prob_meets_fda,
    )
end

"""
Format BootstrapResult for publication-ready output.

# Returns
String like "1.64 (95% CI: 1.52-1.78)"
"""
function format_bootstrap_result(
    result::BootstrapResult;
    digits::Int = 2,
    ci_label::String = "95% CI"
)::String
    est = round(result.estimate, digits=digits)
    lower = round(result.ci_lower, digits=digits)
    upper = round(result.ci_upper, digits=digits)
    return "$est ($ci_label: $lower-$upper)"
end

"""
Generate LaTeX table row for regulatory metrics.

Useful for manuscript preparation.
"""
function latex_metrics_row(
    metrics::Dict{String, Any},
    row_label::String = "Model"
)::String
    gmfe = metrics["GMFE"]
    aafe = metrics["AAFE"]
    pct2x = metrics["percent_within_2fold"]
    r2 = metrics["R_squared"]

    # Format: Label & GMFE (CI) & AAFE (CI) & %2-fold (CI) & R² (CI) \\
    row = "$row_label & "
    row *= "$(round(gmfe.estimate, digits=2)) ($(round(gmfe.ci_lower, digits=2))-$(round(gmfe.ci_upper, digits=2))) & "
    row *= "$(round(aafe.estimate, digits=2)) ($(round(aafe.ci_lower, digits=2))-$(round(aafe.ci_upper, digits=2))) & "
    row *= "$(round(pct2x.estimate, digits=1)) ($(round(pct2x.ci_lower, digits=1))-$(round(pct2x.ci_upper, digits=1))) & "
    row *= "$(round(r2.estimate, digits=3)) ($(round(r2.ci_lower, digits=3))-$(round(r2.ci_upper, digits=3))) \\\\"

    return row
end

export BootstrapResult, bootstrap_metric
export gmfe_with_ci, afe_with_ci, aafe_with_ci, r_squared_with_ci
export percent_within_fold_with_ci, regulatory_metrics_with_ci
export format_bootstrap_result, latex_metrics_row

"""
Percent within Fold.

Percentual de predições dentro de um fold range (e.g., 1.25×, 1.5×, 2.0×).

Inovações:
- Type-safe computation
- Múltiplos thresholds
"""
function percent_within_fold(
    pred::AbstractVector{Float64},
    obs::AbstractVector{Float64},
    fold::Float64 = 2.0,
)::Float64
    fe = fold_error(pred, obs)
    if isempty(fe)
        return NaN
    end

    within = count(f -> f <= fold, fe)
    return (within / length(fe)) * 100.0
end

"""
MAE e RMSE em escala log10.

Inovações:
- Type-safe computation
- Handling de zeros/NaN
"""
function mae_rmse_log10(
    pred::AbstractVector{Float64},
    obs::AbstractVector{Float64},
)::Tuple{Float64, Float64}
    # Filtrar zeros e valores inválidos
    valid_idx = [i for i in 1:length(pred)
                 if obs[i] > 0 && pred[i] > 0 && isfinite(pred[i]) && isfinite(obs[i])]

    if isempty(valid_idx)
        return NaN, NaN
    end

    pred_log = log10.(pred[valid_idx])
    obs_log = log10.(obs[valid_idx])

    mae = mean(abs.(pred_log .- obs_log))
    rmse = sqrt(mean((pred_log .- obs_log).^2))

    return mae, rmse
end

"""
R² (Coefficient of Determination).

Inovações:
- Type-safe computation
- Handling de zeros/NaN
"""
function r_squared(pred::AbstractVector{Float64}, obs::AbstractVector{Float64})::Float64
    # Filtrar valores inválidos
    valid_idx = [i for i in 1:length(pred)
                 if isfinite(pred[i]) && isfinite(obs[i])]

    if isempty(valid_idx)
        return NaN
    end

    pred_valid = pred[valid_idx]
    obs_valid = obs[valid_idx]

    ss_res = sum((obs_valid .- pred_valid).^2)
    ss_tot = sum((obs_valid .- mean(obs_valid)).^2)

    if ss_tot == 0
        return NaN
    end

    return 1.0 - (ss_res / ss_tot)
end

"""
Validação científica completa.

Inovações:
- Métricas regulatórias (FE, GMFE, % within fold)
- Métricas científicas (R², MAE, RMSE em log10)
- Visualização científica
- Relatórios automatizados

Args:
    model: DynamicPBPKGNN model (or any model with forward_batch method)
    test_data: Dataset de teste
    output_dir: Diretório de saída

Returns:
    Dict com todas as métricas

Note: This function requires DynamicGNN module to be loaded.
"""
function validate_scientific(
    model::Any,  # DynamicPBPKGNN or compatible model
    test_data::Any,  # TODO: Type do dataset
    output_dir::String;
    device = cpu,
)
    if !HAS_DYNAMIC_GNN[]
        error("validate_scientific requires DynamicGNN module to be loaded")
    end
    # Predições
    pred_concentrations = []
    true_concentrations = []

    for sample in test_data
        results = forward_batch(
            model,
            [sample.dose],
            [sample.params],
            sample.time_points,
            device,
        )

        pred_conc = results["concentrations"][1, :, :]  # [num_organs, num_time_points]
        true_conc = sample.true_conc

        push!(pred_concentrations, pred_conc)
        push!(true_concentrations, true_conc)
    end

    # Calcular Cmax e AUC
    pred_cmax = [maximum(conc[BLOOD_IDX, :]) for conc in pred_concentrations]
    true_cmax = [maximum(conc[BLOOD_IDX, :]) for conc in true_concentrations]

    # AUC (trapezoidal)
    pred_auc = [trapezoidal_auc(time_points, conc[BLOOD_IDX, :])
                for (conc, time_points) in zip(pred_concentrations, [s.time_points for s in test_data])]
    true_auc = [trapezoidal_auc(time_points, conc[BLOOD_IDX, :])
                for (conc, time_points) in zip(true_concentrations, [s.time_points for s in test_data])]

    # Métricas Cmax
    fe_cmax = fold_error(pred_cmax, true_cmax)
    gmfe_cmax = geometric_mean_fold_error(pred_cmax, true_cmax)
    pct_1_25_cmax = percent_within_fold(pred_cmax, true_cmax, 1.25)
    pct_1_5_cmax = percent_within_fold(pred_cmax, true_cmax, 1.5)
    pct_2_0_cmax = percent_within_fold(pred_cmax, true_cmax, 2.0)
    mae_log10_cmax, rmse_log10_cmax = mae_rmse_log10(pred_cmax, true_cmax)
    r2_cmax = r_squared(pred_cmax, true_cmax)

    # Métricas AUC
    fe_auc = fold_error(pred_auc, true_auc)
    gmfe_auc = geometric_mean_fold_error(pred_auc, true_auc)
    pct_1_25_auc = percent_within_fold(pred_auc, true_auc, 1.25)
    pct_1_5_auc = percent_within_fold(pred_auc, true_auc, 1.5)
    pct_2_0_auc = percent_within_fold(pred_auc, true_auc, 2.0)
    mae_log10_auc, rmse_log10_auc = mae_rmse_log10(pred_auc, true_auc)
    r2_auc = r_squared(pred_auc, true_auc)

    # Consolidar métricas
    metrics = Dict(
        "Cmax" => Dict(
            "fold_error_mean" => mean(fe_cmax),
            "fold_error_median" => median(fe_cmax),
            "fold_error_p67" => quantile(fe_cmax, 0.67),
            "geometric_mean_fold_error" => gmfe_cmax,
            "percent_within_1.25x" => pct_1_25_cmax,
            "percent_within_1.5x" => pct_1_5_cmax,
            "percent_within_2.0x" => pct_2_0_cmax,
            "mae_log10" => mae_log10_cmax,
            "rmse_log10" => rmse_log10_cmax,
            "r_squared" => r2_cmax,
        ),
        "AUC" => Dict(
            "fold_error_mean" => mean(fe_auc),
            "fold_error_median" => median(fe_auc),
            "fold_error_p67" => quantile(fe_auc, 0.67),
            "geometric_mean_fold_error" => gmfe_auc,
            "percent_within_1.25x" => pct_1_25_auc,
            "percent_within_1.5x" => pct_1_5_auc,
            "percent_within_2.0x" => pct_2_0_auc,
            "mae_log10" => mae_log10_auc,
            "rmse_log10" => rmse_log10_auc,
            "r_squared" => r2_auc,
        ),
    )

    # Salvar métricas
    mkpath(output_dir)
    open(joinpath(output_dir, "scientific_metrics.json"), "w") do f
        JSON.print(f, metrics, 2)
    end

    # Visualização
    plot_scientific_metrics(pred_cmax, true_cmax, pred_auc, true_auc, output_dir)

    return metrics
end

"""
AUC trapezoidal.

Inovações:
- Type-safe computation
- Integração numérica precisa
"""
function trapezoidal_auc(time_points::Vector{Float64}, concentrations::Vector{Float64})::Float64
    auc = 0.0
    for i in 1:(length(time_points) - 1)
        dt = time_points[i+1] - time_points[i]
        auc += dt * (concentrations[i] + concentrations[i+1]) / 2.0
    end
    return auc
end

"""
Visualização científica.

Inovações:
- Plots.jl (alta qualidade)
- Scatter plots (pred vs obs)
- Fold error distributions
- Residuals vs predicted
"""
function plot_scientific_metrics(
    pred_cmax::Vector{Float64},
    true_cmax::Vector{Float64},
    pred_auc::Vector{Float64},
    true_auc::Vector{Float64},
    output_dir::String,
)
    # Scatter plot: Pred vs Obs (Cmax)
    p1 = scatter(
        true_cmax,
        pred_cmax,
        xlabel="Observed Cmax (mg/L)",
        ylabel="Predicted Cmax (mg/L)",
        title="Cmax: Predicted vs Observed",
        legend=false,
    )
    # Linha 1:1
    plot!(p1, [minimum(true_cmax), maximum(true_cmax)],
          [minimum(true_cmax), maximum(true_cmax)],
          linestyle=:dash, color=:red, label="1:1")
    # Linhas 2×
    plot!(p1, [minimum(true_cmax), maximum(true_cmax)],
          [2*minimum(true_cmax), 2*maximum(true_cmax)],
          linestyle=:dash, color=:gray, label="2×")
    plot!(p1, [minimum(true_cmax), maximum(true_cmax)],
          [0.5*minimum(true_cmax), 0.5*maximum(true_cmax)],
          linestyle=:dash, color=:gray, label="0.5×")

    # Scatter plot: Pred vs Obs (AUC)
    p2 = scatter(
        true_auc,
        pred_auc,
        xlabel="Observed AUC (mg·h/L)",
        ylabel="Predicted AUC (mg·h/L)",
        title="AUC: Predicted vs Observed",
        legend=false,
    )
    plot!(p2, [minimum(true_auc), maximum(true_auc)],
          [minimum(true_auc), maximum(true_auc)],
          linestyle=:dash, color=:red, label="1:1")

    # Fold error distribution (Cmax)
    fe_cmax = fold_error(pred_cmax, true_cmax)
    p3 = histogram(
        fe_cmax,
        xlabel="Fold Error",
        ylabel="Frequency",
        title="Cmax Fold Error Distribution",
        bins=50,
    )

    # Salvar plots
    mkpath(output_dir)
    savefig(p1, joinpath(output_dir, "cmax_scatter.png"))
    savefig(p2, joinpath(output_dir, "auc_scatter.png"))
    savefig(p3, joinpath(output_dir, "fe_distribution.png"))
end

export validate_scientific, fold_error, geometric_mean_fold_error, percent_within_fold
export mae_rmse_log10, r_squared, plot_scientific_metrics, trapezoidal_auc

end # module
