#!/usr/bin/env julia
"""
Training script for Dynamic GNN PBPK model (Julia version)

Migrado de: scripts/train_dynamic_gnn_pbpk.py

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: 2025-11-18
"""

using Pkg
Pkg.activate(@__DIR__)

using ArgParse
using Random
using ProgressMeter
using BSON
using CUDA

# Importar módulos do DarwinPBPK
include(joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK.jl"))
using .DarwinPBPK
using .DarwinPBPK.Training
using .DarwinPBPK.DynamicGNN
using .DarwinPBPK.ODEPBPKSolver

function parse_args()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--dataset", "-d"
            help = "Path to dataset NPZ file"
            arg_type = String
            required = true
        "--output-dir", "-o"
            help = "Output directory for checkpoints"
            arg_type = String
            default = "models/dynamic_gnn"
        "--epochs", "-e"
            help = "Number of training epochs"
            arg_type = Int
            default = 100
        "--batch-size", "-b"
            help = "Batch size"
            arg_type = Int
            default = 32
        "--learning-rate", "-l"
            help = "Learning rate"
            arg_type = Float64
            default = 1e-4
        "--split-strategy"
            help = "Data split strategy (random, compound, group)"
            arg_type = String
            default = "compound"
        "--device"
            help = "Device (cpu, cuda)"
            arg_type = String
            default = "cuda"
    end

    return parse_args(s)
end

function main()
    args = parse_args()

    println("=" ^ 80)
    println("TRAINING DYNAMIC GNN PBPK MODEL (Julia)")
    println("=" ^ 80)
    println()

    # Carregar dataset
    println("📦 Carregando dataset...")
    dataset = Training.PBPKDataset(args["dataset"])
    println("  ✅ Dataset carregado: ", length(dataset), " amostras")

    # Split dataset
    println("📊 Dividindo dataset (estratégia: ", args["split-strategy"], ")...")
    # TODO: Implementar split logic
    train_size = Int(floor(0.8 * length(dataset)))
    train_data = dataset[1:train_size]
    val_data = dataset[train_size+1:end]
    println("  ✅ Train: ", length(train_data), " | Val: ", length(val_data))

    # Criar modelo
    println("🧠 Criando modelo...")
    model = DynamicGNN.DynamicPBPKGNN(
        node_dim = 16,
        edge_dim = 4,
        hidden_dim = 64,
        num_gnn_layers = 3,
        num_temporal_steps = 100,
        dt = 0.1,
        use_attention = true,
    )
    println("  ✅ Modelo criado")

    # Setup device
    device = args["device"] == "cuda" && CUDA.functional() ? gpu : cpu
    println("  📱 Device: ", device)

    # Setup optimizer
    println("⚙️  Configurando otimizador...")
    opt = Flux.Adam(args["learning-rate"])
    println("  ✅ Optimizer: Adam(lr=", args["learning-rate"], ")")

    # Training loop
    println()
    println("🚀 Iniciando treinamento...")
    println("=" ^ 80)

    best_val_loss = Inf
    for epoch in 1:args["epochs"]
        # Train epoch
        train_loss, train_metrics = Training.train_epoch!(
            model,
            train_data,
            opt,
            device,
        )

        # Validate
        val_loss, val_metrics = Training.validate_epoch(
            model,
            val_data,
            device,
        )

        println(@sprintf(
            "Epoch %3d/%d | Train Loss: %.6f | Val Loss: %.6f",
            epoch,
            args["epochs"],
            train_loss,
            val_loss
        ))

        # Save best model
        if val_loss < best_val_loss
            best_val_loss = val_loss
            checkpoint_path = joinpath(args["output-dir"], "best_model.bson")
            mkpath(dirname(checkpoint_path))
            BSON.@save checkpoint_path model epoch val_loss
            println("  💾 Melhor modelo salvo!")
        end
    end

    println()
    println("=" ^ 80)
    println("✅ TREINAMENTO COMPLETO!")
    println("=" ^ 80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

