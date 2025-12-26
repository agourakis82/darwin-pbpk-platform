#!/usr/bin/env julia
"""
Comparação de Métricas: Antes vs. Depois da Regularização

Compara métricas de modelos treinados com e sem regularização.

Autor: Dr. Sounio Agourakis + AI Assistant
Data: 2025-11-18
"""

using Pkg
Pkg.activate(".")

using JSON
using Plots
using Statistics

# Caminhos dos resultados
const BEFORE_REGULARIZATION = "models/dynamic_gnn_v4_compound/evaluation_scientific/scientific_eval.json"
const AFTER_REGULARIZATION = "models/dynamic_gnn_regularized/validation_metrics.json"
const EXPERIMENTAL = "models/dynamic_gnn_v4_compound/revalidation/revalidation_results.json"

"""
Carrega métricas de JSON.
"""
function load_metrics(json_path::String)::Dict{String, Any}
    if !isfile(json_path)
        return Dict()
    end
    return JSON.parsefile(json_path)
end

"""
Compara métricas antes vs. depois.
"""
function compare_metrics(
    before::Dict{String, Any},
    after::Dict{String, Any},
    experimental::Dict{String, Any},
    output_dir::String = "julia-migration/logs/comparison",
)
    mkpath(output_dir)

    println("=" ^ 80)
    println("COMPARAÇÃO: ANTES vs. DEPOIS DA REGULARIZAÇÃO")
    println("=" ^ 80)
    println()

    # Extrair métricas
    before_cmax_gmfe = haskey(before, "model_metrics") ?
        before["model_metrics"]["geometric_mean_fold_error"] : nothing
    before_auc_gmfe = haskey(before, "model_metrics") ?
        before["model_metrics"]["geometric_mean_fold_error"] : nothing

    after_cmax_gmfe = haskey(after, "cmax") ?
        after["cmax"]["geometric_mean_fold_error"] : nothing
    after_auc_gmfe = haskey(after, "auc") ?
        after["auc"]["geometric_mean_fold_error"] : nothing

    exp_cmax_gmfe = haskey(experimental, "Fine-tuned") ?
        experimental["Fine-tuned"]["cmax"]["gmfe"] : nothing
    exp_auc_gmfe = haskey(experimental, "Fine-tuned") ?
        experimental["Fine-tuned"]["auc"]["gmfe"] : nothing

    # Comparação
    comparison = Dict{String, Any}()

    if before_cmax_gmfe !== nothing && after_cmax_gmfe !== nothing
        improvement_cmax = ((before_cmax_gmfe - after_cmax_gmfe) / before_cmax_gmfe) * 100
        comparison["cmax"] = Dict(
            "before" => before_cmax_gmfe,
            "after" => after_cmax_gmfe,
            "improvement_percent" => improvement_cmax,
            "experimental" => exp_cmax_gmfe,
        )
    end

    if before_auc_gmfe !== nothing && after_auc_gmfe !== nothing
        improvement_auc = ((before_auc_gmfe - after_auc_gmfe) / before_auc_gmfe) * 100
        comparison["auc"] = Dict(
            "before" => before_auc_gmfe,
            "after" => after_auc_gmfe,
            "improvement_percent" => improvement_auc,
            "experimental" => exp_auc_gmfe,
        )
    end

    # Salvar comparação
    open(joinpath(output_dir, "comparison.json"), "w") do f
        JSON.print(f, comparison, 2)
    end

    # Visualização
    if haskey(comparison, "cmax") && haskey(comparison, "auc")
        metrics = ["Cmax GMFE", "AUC GMFE"]
        before_vals = [comparison["cmax"]["before"], comparison["auc"]["before"]]
        after_vals = [comparison["cmax"]["after"], comparison["auc"]["after"]]
        exp_vals = [comparison["cmax"]["experimental"], comparison["auc"]["experimental"]]

        p = groupedbar(
            metrics,
            [before_vals after_vals exp_vals],
            label=["Antes" "Depois" "Experimental"],
            title="GMFE: Comparação Antes vs. Depois",
            yscale=:log10,
        )
        savefig(p, joinpath(output_dir, "comparison.png"))
    end

    # Relatório Markdown
    md = "# 📊 Comparação: Antes vs. Depois da Regularização\n\n"
    md *= "**Data:** 2025-11-18\n\n"
    md *= "---\n\n"

    if haskey(comparison, "cmax")
        cmax = comparison["cmax"]
        md *= "## Cmax\n\n"
        md *= "| Métrica | Antes | Depois | Melhoria | Experimental |\n"
        md *= "|---------|-------|--------|----------|--------------|\n"
        md *= "| GMFE | $(round(cmax["before"], digits=3)) | $(round(cmax["after"], digits=3)) | $(round(cmax["improvement_percent"], digits=1))% | $(round(cmax["experimental"], digits=2)) |\n"
        md *= "\n"
    end

    if haskey(comparison, "auc")
        auc = comparison["auc"]
        md *= "## AUC\n\n"
        md *= "| Métrica | Antes | Depois | Melhoria | Experimental |\n"
        md *= "|---------|-------|--------|----------|--------------|\n"
        md *= "| GMFE | $(round(auc["before"], digits=3)) | $(round(auc["after"], digits=3)) | $(round(auc["improvement_percent"], digits=1))% | $(round(auc["experimental"], digits=2)) |\n"
        md *= "\n"
    end

    open(joinpath(output_dir, "comparison_report.md"), "w") do f
        write(f, md)
    end

    # Print resumo
    println("📊 Comparação:")
    if haskey(comparison, "cmax")
        cmax = comparison["cmax"]
        println("   Cmax GMFE:")
        println("     - Antes: $(round(cmax["before"], digits=3))")
        println("     - Depois: $(round(cmax["after"], digits=3))")
        println("     - Melhoria: $(round(cmax["improvement_percent"], digits=1))%")
        println("     - Experimental: $(round(cmax["experimental"], digits=2))")
        println()
    end

    if haskey(comparison, "auc")
        auc = comparison["auc"]
        println("   AUC GMFE:")
        println("     - Antes: $(round(auc["before"], digits=3))")
        println("     - Depois: $(round(auc["after"], digits=3))")
        println("     - Melhoria: $(round(auc["improvement_percent"], digits=1))%")
        println("     - Experimental: $(round(auc["experimental"], digits=2))")
        println()
    end

    println("📁 Relatório salvo em: $output_dir/")

    return comparison
end

# Main
function main()
    println("📊 Comparação: Antes vs. Depois da Regularização")
    println()

    # Carregar métricas
    before = load_metrics(BEFORE_REGULARIZATION)
    after = load_metrics(AFTER_REGULARIZATION)
    experimental = load_metrics(EXPERIMENTAL)

    if isempty(before) || isempty(after)
        println("⚠️  Arquivos não encontrados:")
        println("   - Antes: $BEFORE_REGULARIZATION")
        println("   - Depois: $AFTER_REGULARIZATION")
        println()
        println("💡 Execute primeiro o treinamento com regularização:")
        println("   julia julia-migration/scripts/training/train_with_regularization.jl")
        return
    end

    # Comparar
    comparison = compare_metrics(before, after, experimental)

    println("✅ Comparação completa!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

