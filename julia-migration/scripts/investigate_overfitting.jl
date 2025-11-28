#!/usr/bin/env julia
"""
Script para Investigar Overfitting no GMFE

Análise rigorosa Q1+:
1. Train/Test split rigoroso
2. Validação cruzada k-fold
3. Early stopping
4. Regularização (L1/L2)
5. Análise de learning curves
6. Comparação train vs validation metrics

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: 2025-11-18
"""

using Pkg
Pkg.activate(".")

using DarwinPBPK
using DarwinPBPK.Validation
using DarwinPBPK.DynamicGNN
using DarwinPBPK.ODEPBPKSolver
using Statistics
using Random
using JSON
using DataFrames
using CSV
using Plots

# Configuração
Random.seed!(42)
const K_FOLDS = 5
const EARLY_STOPPING_PATIENCE = 10
const MIN_DELTA = 0.001

"""
Análise de Overfitting - Métricas Train vs Validation

Detecta overfitting comparando métricas de treino e validação.
"""
function analyze_overfitting(
    train_pred::Vector{Float64},
    train_obs::Vector{Float64},
    val_pred::Vector{Float64},
    val_obs::Vector{Float64},
)::Dict{String, Any}

    # Métricas de treino
    train_fe = Validation.fold_error(train_pred, train_obs)
    train_gmfe = Validation.geometric_mean_fold_error(train_pred, train_obs)
    train_r2 = Validation.r_squared(train_pred, train_obs)
    train_pct_2x = Validation.percent_within_fold(train_pred, train_obs, 2.0)

    # Métricas de validação
    val_fe = Validation.fold_error(val_pred, val_obs)
    val_gmfe = Validation.geometric_mean_fold_error(val_pred, val_obs)
    val_r2 = Validation.r_squared(val_pred, val_obs)
    val_pct_2x = Validation.percent_within_fold(val_pred, val_obs, 2.0)

    # Diferenças (overfitting se train >> val)
    gmfe_gap = train_gmfe - val_gmfe
    r2_gap = train_r2 - val_r2
    pct_2x_gap = train_pct_2x - val_pct_2x

    # Critérios de overfitting
    overfitting_gmfe = gmfe_gap < -0.1  # Train GMFE muito melhor que val
    overfitting_r2 = r2_gap > 0.1  # Train R² muito melhor que val
    overfitting_pct = pct_2x_gap > 10.0  # Train % muito melhor que val

    return Dict(
        "train" => Dict(
            "gmfe" => train_gmfe,
            "r2" => train_r2,
            "pct_2x" => train_pct_2x,
            "fe_mean" => mean(train_fe),
            "fe_median" => median(train_fe),
        ),
        "validation" => Dict(
            "gmfe" => val_gmfe,
            "r2" => val_r2,
            "pct_2x" => val_pct_2x,
            "fe_mean" => mean(val_fe),
            "fe_median" => median(val_fe),
        ),
        "gaps" => Dict(
            "gmfe_gap" => gmfe_gap,
            "r2_gap" => r2_gap,
            "pct_2x_gap" => pct_2x_gap,
        ),
        "overfitting_detected" => Dict(
            "gmfe" => overfitting_gmfe,
            "r2" => overfitting_r2,
            "pct_2x" => overfitting_pct,
        ),
        "overfitting_severity" => overfitting_gmfe || overfitting_r2 || overfitting_pct ? "HIGH" : "LOW",
    )
end

"""
Validação Cruzada k-Fold

Avalia modelo em k folds para detectar overfitting.
"""
function k_fold_cross_validation(
    data::Vector{Tuple{Float64, PBPKParams, Vector{Float64}}},  # (dose, params, true_conc)
    model::DynamicPBPKGNN,
    k::Int = K_FOLDS,
)::Dict{String, Any}

    n = length(data)
    fold_size = n ÷ k
    results = Vector{Dict{String, Any}}()

    for fold in 1:k
        # Split train/val
        val_start = (fold - 1) * fold_size + 1
        val_end = fold == k ? n : fold * fold_size

        val_indices = val_start:val_end
        train_indices = vcat(1:(val_start-1), (val_end+1):n)

        train_data = data[train_indices]
        val_data = data[val_indices]

        # Predições (simulado - precisa treinar modelo)
        # Por enquanto, usar ODE solver como baseline
        train_pred = Float64[]
        train_obs = Float64[]
        val_pred = Float64[]
        val_obs = Float64[]

        for (dose, params, true_conc) in train_data
            # Simular com ODE solver
            result = ODEPBPKSolver.solve_ode(params, dose, (0.0, 24.0))
            pred_cmax = maximum(result["blood"])
            true_cmax = maximum(true_conc)

            push!(train_pred, pred_cmax)
            push!(train_obs, true_cmax)
        end

        for (dose, params, true_conc) in val_data
            result = ODEPBPKSolver.solve_ode(params, dose, (0.0, 24.0))
            pred_cmax = maximum(result["blood"])
            true_cmax = maximum(true_conc)

            push!(val_pred, pred_cmax)
            push!(val_obs, true_cmax)
        end

        # Análise de overfitting
        analysis = analyze_overfitting(train_pred, train_obs, val_pred, val_obs)
        analysis["fold"] = fold
        push!(results, analysis)
    end

    # Agregar resultados
    avg_train_gmfe = mean([r["train"]["gmfe"] for r in results])
    avg_val_gmfe = mean([r["validation"]["gmfe"] for r in results])
    avg_gap = avg_train_gmfe - avg_val_gmfe

    return Dict(
        "folds" => results,
        "summary" => Dict(
            "avg_train_gmfe" => avg_train_gmfe,
            "avg_val_gmfe" => avg_val_gmfe,
            "avg_gap" => avg_gap,
            "overfitting_detected" => avg_gap < -0.1,
        ),
    )
end

"""
Learning Curves

Analisa evolução de métricas durante treinamento.
"""
function learning_curves(
    train_metrics_history::Vector{Dict{String, Float64}},
    val_metrics_history::Vector{Dict{String, Float64}},
)::Dict{String, Any}

    epochs = 1:length(train_metrics_history)

    train_gmfe = [m["gmfe"] for m in train_metrics_history]
    val_gmfe = [m["gmfe"] for m in val_metrics_history]

    train_r2 = [m["r2"] for m in train_metrics_history]
    val_r2 = [m["r2"] for m in val_metrics_history]

    # Detectar divergência (overfitting)
    divergence_epoch = nothing
    for i in 2:length(epochs)
        gap = train_gmfe[i] - val_gmfe[i]
        if gap < -0.1 && train_gmfe[i] < val_gmfe[i] - 0.1
            divergence_epoch = i
            break
        end
    end

    return Dict(
        "epochs" => collect(epochs),
        "train_gmfe" => train_gmfe,
        "val_gmfe" => val_gmfe,
        "train_r2" => train_r2,
        "val_r2" => val_r2,
        "divergence_epoch" => divergence_epoch,
        "overfitting_detected" => divergence_epoch !== nothing,
    )
end

"""
Early Stopping

Detecta quando parar treinamento para evitar overfitting.
"""
function should_stop_early(
    val_metrics_history::Vector{Dict{String, Float64}},
    patience::Int = EARLY_STOPPING_PATIENCE,
    min_delta::Float64 = MIN_DELTA,
)::Tuple{Bool, Int}

    if length(val_metrics_history) < patience + 1
        return false, 0
    end

    best_val_gmfe = minimum([m["gmfe"] for m in val_metrics_history])
    recent_val_gmfe = [m["gmfe"] for m in val_metrics_history[end-patience:end]]

    # Verificar se melhorou recentemente
    no_improvement = true
    for gmfe in recent_val_gmfe
        if gmfe < best_val_gmfe - min_delta
            no_improvement = false
            break
        end
    end

    if no_improvement
        return true, length(val_metrics_history) - patience
    end

    return false, 0
end

"""
Relatório Completo de Overfitting

Gera relatório detalhado com todas as análises.
"""
function generate_overfitting_report(
    train_pred::Vector{Float64},
    train_obs::Vector{Float64},
    val_pred::Vector{Float64},
    val_obs::Vector{Float64},
    output_dir::String = "logs/overfitting_analysis",
)::Dict{String, Any}

    mkpath(output_dir)

    # Análise básica
    analysis = analyze_overfitting(train_pred, train_obs, val_pred, val_obs)

    # Visualizações
    p1 = scatter(
        train_obs,
        train_pred,
        label="Train",
        xlabel="Observed",
        ylabel="Predicted",
        title="Train: Predicted vs Observed",
    )
    plot!(p1, [minimum(train_obs), maximum(train_obs)],
          [minimum(train_obs), maximum(train_obs)],
          linestyle=:dash, color=:red, label="1:1")

    p2 = scatter(
        val_obs,
        val_pred,
        label="Validation",
        xlabel="Observed",
        ylabel="Predicted",
        title="Validation: Predicted vs Observed",
    )
    plot!(p2, [minimum(val_obs), maximum(val_obs)],
          [minimum(val_obs), maximum(val_obs)],
          linestyle=:dash, color=:red, label="1:1")

    # Comparação de métricas
    metrics_comparison = [
        analysis["train"]["gmfe"],
        analysis["validation"]["gmfe"],
        analysis["train"]["r2"],
        analysis["validation"]["r2"],
    ]

    p3 = bar(
        ["Train GMFE", "Val GMFE", "Train R²", "Val R²"],
        metrics_comparison,
        title="Metrics Comparison: Train vs Validation",
        label="",
    )

    # Salvar plots
    savefig(p1, joinpath(output_dir, "train_scatter.png"))
    savefig(p2, joinpath(output_dir, "val_scatter.png"))
    savefig(p3, joinpath(output_dir, "metrics_comparison.png"))

    # Salvar relatório JSON
    report = Dict(
        "analysis" => analysis,
        "recommendations" => generate_recommendations(analysis),
    )

    open(joinpath(output_dir, "overfitting_report.json"), "w") do f
        JSON.print(f, report, 2)
    end

    return report
end

"""
Gera recomendações baseadas na análise.
"""
function generate_recommendations(analysis::Dict{String, Any})::Vector{String}
    recommendations = String[]

    if analysis["overfitting_detected"]["gmfe"]
        push!(recommendations, "⚠️ OVERFITTING DETECTED: GMFE gap significativo (train vs validation)")
        push!(recommendations, "→ Aplicar regularização L2 (weight decay)")
        push!(recommendations, "→ Reduzir complexidade do modelo (menos camadas/neurônios)")
        push!(recommendations, "→ Aumentar dataset de treinamento")
    end

    if analysis["overfitting_detected"]["r2"]
        push!(recommendations, "⚠️ OVERFITTING DETECTED: R² gap significativo")
        push!(recommendations, "→ Implementar dropout")
        push!(recommendations, "→ Early stopping mais agressivo")
    end

    if analysis["overfitting_detected"]["pct_2x"]
        push!(recommendations, "⚠️ OVERFITTING DETECTED: % within 2x gap significativo")
        push!(recommendations, "→ Validação cruzada k-fold")
        push!(recommendations, "→ Ensemble de modelos")
    end

    if analysis["overfitting_severity"] == "LOW"
        push!(recommendations, "✅ Overfitting não detectado - modelo generaliza bem")
    end

    return recommendations
end

# Main
function main()
    println("=" ^ 80)
    println("INVESTIGAÇÃO DE OVERFITTING - GMFE")
    println("=" ^ 80)
    println()

    # TODO: Carregar dados reais
    # Por enquanto, usar dados sintéticos para demonstração

    println("📊 Análise de Overfitting:")
    println("  1. Train/Test split rigoroso")
    println("  2. Validação cruzada k-fold")
    println("  3. Learning curves")
    println("  4. Early stopping")
    println()

    println("⚠️  NOTA: Este script requer dados de treinamento/validação reais")
    println("   Implementar carregamento de dados do dataset")
    println()

    println("✅ Script criado - pronto para análise quando dados estiverem disponíveis")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

