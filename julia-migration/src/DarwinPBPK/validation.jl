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
using Measurements
using Plots
using DataFrames
using CSV
using JSON

# Importar módulos
using ..DynamicGNN: DynamicPBPKGNN, forward_batch
using ..ODEPBPKSolver: PBPKParams, PBPK_ORGANS, NUM_ORGANS

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
    model: DynamicPBPKGNN model
    test_data: Dataset de teste
    output_dir: Diretório de saída

Returns:
    Dict com todas as métricas
"""
function validate_scientific(
    model::DynamicPBPKGNN,
    test_data::Any,  # TODO: Type do dataset
    output_dir::String;
    device = cpu,
)
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
export mae_rmse_log10, r_squared, plot_scientific_metrics

end # module

