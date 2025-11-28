#!/usr/bin/env julia
"""
Análise de Overfitting a partir de Resultados JSON Existentes

Analisa resultados de validação existentes para detectar overfitting.

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: 2025-11-18
"""

using JSON
using Statistics
using Plots

# Caminhos dos arquivos
const SYNTHETIC_RESULTS = "models/dynamic_gnn_v4_compound/evaluation_scientific/scientific_eval.json"
const EXPERIMENTAL_RESULTS = "models/dynamic_gnn_v4_compound/revalidation/revalidation_results.json"

"""
Carrega e analisa resultados JSON.
"""
function load_and_analyze(json_path::String)::Dict{String, Any}
    if !isfile(json_path)
        println("⚠️  Arquivo não encontrado: $json_path")
        return Dict()
    end

    data = JSON.parsefile(json_path)
    return data
end

"""
Analisa overfitting comparando resultados sintéticos vs experimentais.
"""
function analyze_overfitting_gap(
    synthetic::Dict{String, Any},
    experimental::Dict{String, Any},
)::Dict{String, Any}

    results = Dict{String, Any}()

    # Extrair GMFE
    if haskey(synthetic, "cmax") && haskey(synthetic["cmax"], "geometric_mean_fold_error")
        synth_gmfe_cmax = synthetic["cmax"]["geometric_mean_fold_error"]
    else
        synth_gmfe_cmax = nothing
    end

    if haskey(experimental, "cmax") && haskey(experimental["cmax"], "gmfe")
        exp_gmfe_cmax = experimental["cmax"]["gmfe"]
    else
        exp_gmfe_cmax = nothing
    end

    if haskey(synthetic, "auc") && haskey(synthetic["auc"], "geometric_mean_fold_error")
        synth_gmfe_auc = synthetic["auc"]["geometric_mean_fold_error"]
    else
        synth_gmfe_auc = nothing
    end

    if haskey(experimental, "auc") && haskey(experimental["auc"], "gmfe")
        exp_gmfe_auc = experimental["auc"]["gmfe"]
    else
        exp_gmfe_auc = nothing
    end

    # Calcular gaps
    if synth_gmfe_cmax !== nothing && exp_gmfe_cmax !== nothing
        gap_cmax = exp_gmfe_cmax - synth_gmfe_cmax
        gap_ratio_cmax = exp_gmfe_cmax / synth_gmfe_cmax

        results["cmax"] = Dict(
            "synthetic_gmfe" => synth_gmfe_cmax,
            "experimental_gmfe" => exp_gmfe_cmax,
            "gap" => gap_cmax,
            "gap_ratio" => gap_ratio_cmax,
            "overfitting_detected" => gap_ratio_cmax > 10.0,  # Gap > 10× indica overfitting
        )
    end

    if synth_gmfe_auc !== nothing && exp_gmfe_auc !== nothing
        gap_auc = exp_gmfe_auc - synth_gmfe_auc
        gap_ratio_auc = exp_gmfe_auc / synth_gmfe_auc

        results["auc"] = Dict(
            "synthetic_gmfe" => synth_gmfe_auc,
            "experimental_gmfe" => exp_gmfe_auc,
            "gap" => gap_auc,
            "gap_ratio" => gap_ratio_auc,
            "overfitting_detected" => gap_ratio_auc > 10.0,
        )
    end

    return results
end

"""
Gera visualizações de overfitting.
"""
function plot_overfitting_analysis(analysis::Dict{String, Any}, output_dir::String)
    mkpath(output_dir)

    # Plot: GMFE Comparison
    if haskey(analysis, "cmax") && haskey(analysis, "auc")
        metrics = ["Synthetic Cmax", "Experimental Cmax", "Synthetic AUC", "Experimental AUC"]
        values = [
            analysis["cmax"]["synthetic_gmfe"],
            analysis["cmax"]["experimental_gmfe"],
            analysis["auc"]["synthetic_gmfe"],
            analysis["auc"]["experimental_gmfe"],
        ]

        p = bar(metrics, values, title="GMFE: Synthetic vs Experimental", label="GMFE", yscale=:log10)
        savefig(p, joinpath(output_dir, "gmfe_comparison.png"))
    end

    # Plot: Gap Ratio
    if haskey(analysis, "cmax") && haskey(analysis, "auc")
        metrics = ["Cmax Gap Ratio", "AUC Gap Ratio"]
        ratios = [
            analysis["cmax"]["gap_ratio"],
            analysis["auc"]["gap_ratio"],
        ]

        p = bar(metrics, ratios, title="Overfitting Gap Ratio (Experimental/Synthetic)", label="Ratio")
        hline!(p, [10.0], linestyle=:dash, color=:red, label="Overfitting Threshold (10×)")
        savefig(p, joinpath(output_dir, "gap_ratio.png"))
    end
end

"""
Gera relatório de overfitting.
"""
function generate_overfitting_report(analysis::Dict{String, Any}, output_dir::String)
    mkpath(output_dir)

    report = Dict(
        "analysis" => analysis,
        "summary" => Dict(
            "overfitting_detected_cmax" => haskey(analysis, "cmax") && analysis["cmax"]["overfitting_detected"],
            "overfitting_detected_auc" => haskey(analysis, "auc") && analysis["auc"]["overfitting_detected"],
            "severity" => "HIGH",
        ),
        "recommendations" => generate_recommendations(analysis),
    )

    # Salvar JSON
    open(joinpath(output_dir, "overfitting_analysis.json"), "w") do f
        JSON.print(f, report, 2)
    end

    # Salvar Markdown
    md_report = generate_markdown_report(report)
    open(joinpath(output_dir, "overfitting_report.md"), "w") do f
        write(f, md_report)
    end

    return report
end

"""
Gera recomendações baseadas na análise.
"""
function generate_recommendations(analysis::Dict{String, Any})::Vector{String}
    recommendations = String[]

    if haskey(analysis, "cmax") && analysis["cmax"]["overfitting_detected"]
        push!(recommendations, "🚨 OVERFITTING CRÍTICO em Cmax:")
        push!(recommendations, "   - Gap ratio: $(round(analysis["cmax"]["gap_ratio"], digits=2))×")
        push!(recommendations, "   - Ação: Implementar regularização L2 (weight decay)")
        push!(recommendations, "   - Ação: Adicionar dropout (0.2-0.5)")
        push!(recommendations, "   - Ação: Reduzir complexidade do modelo")
        push!(recommendations, "   - Ação: Aumentar dataset de treinamento")
    end

    if haskey(analysis, "auc") && analysis["auc"]["overfitting_detected"]
        push!(recommendations, "🚨 OVERFITTING CRÍTICO em AUC:")
        push!(recommendations, "   - Gap ratio: $(round(analysis["auc"]["gap_ratio"], digits=2))×")
        push!(recommendations, "   - Ação: Early stopping mais agressivo")
        push!(recommendations, "   - Ação: Validação cruzada k-fold")
        push!(recommendations, "   - Ação: Ensemble de modelos")
    end

    return recommendations
end

"""
Gera relatório Markdown.
"""
function generate_markdown_report(report::Dict{String, Any})::String
    md = "# 🔍 Relatório de Análise de Overfitting\n\n"
    md *= "**Data:** $(Dates.now())\n\n"
    md *= "---\n\n"

    md *= "## 📊 Resumo Executivo\n\n"

    if report["summary"]["overfitting_detected_cmax"] || report["summary"]["overfitting_detected_auc"]
        md *= "**🚨 OVERFITTING DETECTADO**\n\n"
        md *= "O modelo apresenta overfitting significativo:\n"

        if report["summary"]["overfitting_detected_cmax"]
            analysis = report["analysis"]["cmax"]
            md *= "- **Cmax:** Gap ratio de $(round(analysis["gap_ratio"], digits=2))×\n"
        end

        if report["summary"]["overfitting_detected_auc"]
            analysis = report["analysis"]["auc"]
            md *= "- **AUC:** Gap ratio de $(round(analysis["gap_ratio"], digits=2))×\n"
        end
    else
        md *= "**✅ Overfitting não detectado**\n\n"
    end

    md *= "\n---\n\n"
    md *= "## 📈 Análise Detalhada\n\n"

    if haskey(report["analysis"], "cmax")
        analysis = report["analysis"]["cmax"]
        md *= "### Cmax\n\n"
        md *= "| Métrica | Valor |\n"
        md *= "|---------|-------|\n"
        md *= "| GMFE (Sintético) | $(round(analysis["synthetic_gmfe"], digits=3)) |\n"
        md *= "| GMFE (Experimental) | $(round(analysis["experimental_gmfe"], digits=3)) |\n"
        md *= "| Gap | $(round(analysis["gap"], digits=3)) |\n"
        md *= "| Gap Ratio | $(round(analysis["gap_ratio"], digits=2))× |\n"
        md *= "| Overfitting Detectado | $(analysis["overfitting_detected"] ? "✅ Sim" : "❌ Não") |\n"
        md *= "\n"
    end

    if haskey(report["analysis"], "auc")
        analysis = report["analysis"]["auc"]
        md *= "### AUC\n\n"
        md *= "| Métrica | Valor |\n"
        md *= "|---------|-------|\n"
        md *= "| GMFE (Sintético) | $(round(analysis["synthetic_gmfe"], digits=3)) |\n"
        md *= "| GMFE (Experimental) | $(round(analysis["experimental_gmfe"], digits=3)) |\n"
        md *= "| Gap | $(round(analysis["gap"], digits=3)) |\n"
        md *= "| Gap Ratio | $(round(analysis["gap_ratio"], digits=2))× |\n"
        md *= "| Overfitting Detectado | $(analysis["overfitting_detected"] ? "✅ Sim" : "❌ Não") |\n"
        md *= "\n"
    end

    md *= "\n---\n\n"
    md *= "## 💡 Recomendações\n\n"

    for rec in report["recommendations"]
        md *= "- $rec\n"
    end

    return md
end

# Main
function main()
    println("=" ^ 80)
    println("ANÁLISE DE OVERFITTING - Resultados Existentes")
    println("=" ^ 80)
    println()

    # Carregar resultados
    println("📊 Carregando resultados...")
    synthetic = load_and_analyze(SYNTHETIC_RESULTS)
    experimental = load_and_analyze(EXPERIMENTAL_RESULTS)

    if isempty(synthetic) || isempty(experimental)
        println("⚠️  Não foi possível carregar todos os resultados")
        return
    end

    println("✅ Resultados carregados")
    println()

    # Analisar overfitting
    println("🔍 Analisando overfitting...")
    analysis = analyze_overfitting_gap(synthetic, experimental)

    if isempty(analysis)
        println("⚠️  Não foi possível analisar overfitting (estrutura de dados diferente)")
        return
    end

    println("✅ Análise completa")
    println()

    # Gerar visualizações
    output_dir = "julia-migration/logs/overfitting_analysis"
    println("📊 Gerando visualizações...")
    plot_overfitting_analysis(analysis, output_dir)
    println("✅ Visualizações salvas em $output_dir")
    println()

    # Gerar relatório
    println("📝 Gerando relatório...")
    report = generate_overfitting_report(analysis, output_dir)
    println("✅ Relatório salvo em $output_dir")
    println()

    # Resumo
    println("=" ^ 80)
    println("RESUMO")
    println("=" ^ 80)

    if haskey(analysis, "cmax")
        cmax = analysis["cmax"]
        println("Cmax:")
        println("  - GMFE Sintético: $(round(cmax["synthetic_gmfe"], digits=3))")
        println("  - GMFE Experimental: $(round(cmax["experimental_gmfe"], digits=3))")
        println("  - Gap Ratio: $(round(cmax["gap_ratio"], digits=2))×")
        println("  - Overfitting: $(cmax["overfitting_detected"] ? "✅ DETECTADO" : "❌ Não detectado")")
        println()
    end

    if haskey(analysis, "auc")
        auc = analysis["auc"]
        println("AUC:")
        println("  - GMFE Sintético: $(round(auc["synthetic_gmfe"], digits=3))")
        println("  - GMFE Experimental: $(round(auc["experimental_gmfe"], digits=3))")
        println("  - Gap Ratio: $(round(auc["gap_ratio"], digits=2))×")
        println("  - Overfitting: $(auc["overfitting_detected"] ? "✅ DETECTADO" : "❌ Não detectado")")
        println()
    end

    println("📁 Relatório completo: $output_dir/overfitting_report.md")
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Dates
    main()
end

