#!/usr/bin/env julia
"""
Scientific Evaluation Script (Julia version)

Migrado de: scripts/evaluate_dynamic_gnn_scientific.py

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: 2025-11-18
"""

using Pkg
Pkg.activate(@__DIR__)

using ArgParse
using BSON
using CUDA
using Statistics
using Plots

# Importar módulos
include(joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK.jl"))
using .DarwinPBPK
using .DarwinPBPK.Validation
using .DarwinPBPK.DynamicGNN
using .DarwinPBPK.Training

function parse_args()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--checkpoint", "-c"
            help = "Path to model checkpoint"
            arg_type = String
            required = true
        "--dataset", "-d"
            help = "Path to validation dataset"
            arg_type = String
            required = true
        "--output-dir", "-o"
            help = "Output directory for results"
            arg_type = String
            default = "results/evaluation"
    end

    return parse_args(s)
end

function main()
    args = parse_args()

    println("=" ^ 80)
    println("SCIENTIFIC EVALUATION (Julia)")
    println("=" ^ 80)
    println()

    # Carregar modelo
    println("📦 Carregando modelo...")
    BSON.@load args["checkpoint"] model
    println("  ✅ Modelo carregado")

    # Carregar dataset
    println("📊 Carregando dataset...")
    dataset = Training.PBPKDataset(args["dataset"])
    println("  ✅ Dataset carregado: ", length(dataset), " amostras")

    # Avaliar
    println("🔬 Executando avaliação científica...")
    metrics, pred_flat, true_flat = Validation.evaluate_model_scientific(
        model,
        dataset,
        cpu,  # device
    )

    # Imprimir métricas
    println()
    println("=" ^ 80)
    println("RESULTADOS:")
    println("=" ^ 80)
    for (key, value) in metrics
        println(@sprintf("  %s: %.6f", key, value))
    end
    println()

    # Salvar resultados
    mkpath(args["output-dir"])
    results_path = joinpath(args["output-dir"], "metrics.json")
    open(results_path, "w") do f
        JSON.print(f, metrics, 2)
    end
    println("  💾 Resultados salvos em: ", results_path)

    println()
    println("=" ^ 80)
    println("✅ AVALIAÇÃO COMPLETA!")
    println("=" ^ 80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

