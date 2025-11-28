#!/usr/bin/env julia
"""
Treinamento com Regularização Anti-Overfitting - VERSÃO GPU

Script otimizado para GPU (CUDA.jl) com:
- Regularização L2 (weight decay)
- Dropout
- Early stopping
- Gradient clipping
- Aceleração GPU (CUDA)

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: 2025-11-18
"""

using Pkg
Pkg.activate(".")

# Carregar módulos locais via include
const PROJECT_ROOT = dirname(dirname(@__DIR__))
push!(LOAD_PATH, joinpath(PROJECT_ROOT, "src"))

include(joinpath(PROJECT_ROOT, "src", "DarwinPBPK.jl"))
using .DarwinPBPK
using .DarwinPBPK.DynamicGNN
using .DarwinPBPK.Training
using .DarwinPBPK.ODEPBPKSolver
using .DarwinPBPK.Validation

using Random
using JSON
using ProgressMeter

# Tentar carregar CUDA
const CUDA_AVAILABLE = try
    using CUDA
    CUDA.functional()
catch
    false
end

# Definir gpu e cpu
if CUDA_AVAILABLE
    const gpu = CUDA.gpu
    const cpu = CUDA.cpu
else
    using Flux
    const gpu = Flux.cpu  # Fallback para CPU
    const cpu = Flux.cpu
end

# Carregar dataset diretamente (sem módulo separado)
using NPZ

function load_npz_dataset(npz_path::String)
    if !isfile(npz_path)
        error("Arquivo não encontrado: $npz_path")
    end

    data = NPZ.npzread(npz_path)

    # Extrair dados
    doses = Float64[]
    params = ODEPBPKSolver.PBPKParams[]
    true_concs = Matrix{Float64}[]
    time_points = Vector{Float64}[]

    if haskey(data, "doses") && haskey(data, "concentrations")
        doses_arr = data["doses"]
        concs_arr = data["concentrations"]

        # Time points
        t_points = haskey(data, "time_points") ? vec(data["time_points"]) : collect(0.0:0.5:24.0)

        # Clearances
        cl_hepatic = haskey(data, "clearance_hepatic") ? vec(data["clearance_hepatic"]) :
                     haskey(data, "clearances_hepatic") ? vec(data["clearances_hepatic"]) :
                     fill(10.0, length(doses_arr))

        cl_renal = haskey(data, "clearance_renal") ? vec(data["clearance_renal"]) :
                   haskey(data, "clearances_renal") ? vec(data["clearances_renal"]) :
                   fill(5.0, length(doses_arr))

        # Partition coefficients
        kp_arr = haskey(data, "partition_coeffs") ? data["partition_coeffs"] :
                 ones(Float64, length(doses_arr), ODEPBPKSolver.NUM_ORGANS)

        # Processar cada amostra
        for i in 1:length(doses_arr)
            dose = Float64(doses_arr[i])

            kp_dict = Dict{String, Float64}()
            for (j, organ) in enumerate(ODEPBPKSolver.PBPK_ORGANS)
                kp_dict[organ] = Float64(kp_arr[i, j])
            end

            p = ODEPBPKSolver.PBPKParams(
                clearance_hepatic=Float64(cl_hepatic[i]),
                clearance_renal=Float64(cl_renal[i]),
                partition_coeffs=kp_dict,
            )

            # Concentrações
            conc_matrix = Float64.(concs_arr[i, :, :])'  # [num_organs, num_time_points]

            push!(doses, dose)
            push!(params, p)
            push!(true_concs, conc_matrix)
            push!(time_points, t_points)
        end
    else
        error("Estrutura NPZ não reconhecida. Chaves: $(keys(data))")
    end

    return Training.PBPKDataset(doses, params, true_concs, time_points)
end

# Configuração
Random.seed!(42)
const DEVICE = CUDA_AVAILABLE ? gpu : cpu

println("=" ^ 80)
println("TREINAMENTO COM REGULARIZAÇÃO ANTI-OVERFITTING (GPU)")
println("=" ^ 80)
println()
println("📊 Configuração do Sistema:")
println("   - GPU: $(CUDA_AVAILABLE ? "✅ CUDA disponível ($(CUDA.name(CUDA.device())))" : "❌ CPU apenas")")
println("   - Device: $(DEVICE == gpu ? "GPU" : "CPU")")
println()

# Main
function main()
    println("🚀 Treinamento com Regularização Anti-Overfitting")
    println()

    # Carregar dados
    println("📂 Carregando dados...")

    # Tentar carregar dataset específico primeiro
    dataset_paths = [
        "data/processed/pbpk_enriched/dynamic_gnn_dataset_enriched_v4.npz",
        joinpath(homedir(), "darwin-pbpk-platform/data/processed/pbpk_enriched/dynamic_gnn_dataset_enriched_v4.npz"),
    ]

    train_data = nothing
    val_data = nothing
    test_data = nothing

    for path in dataset_paths
        if isfile(path)
            println("✅ Dataset encontrado: $path")
            try
                dataset = load_npz_dataset(path)

                # Split train/val/test (80/10/10)
                n_total = length(dataset.doses)
                n_train = Int(floor(n_total * 0.8))
                n_val = Int(floor(n_total * 0.1))
                n_test = n_total - n_train - n_val

                # Shuffle indices
                indices = Random.shuffle(Random.MersenneTwister(42), 1:n_total)
                train_idx = indices[1:n_train]
                val_idx = indices[n_train+1:n_train+n_val]
                test_idx = indices[n_train+n_val+1:end]

                # Criar splits
                train_data = Training.PBPKDataset(
                    dataset.doses[train_idx],
                    dataset.params[train_idx],
                    dataset.true_concentrations[train_idx],
                    dataset.time_points[train_idx],
                )

                val_data = Training.PBPKDataset(
                    dataset.doses[val_idx],
                    dataset.params[val_idx],
                    dataset.true_concentrations[val_idx],
                    dataset.time_points[val_idx],
                )

                test_data = Training.PBPKDataset(
                    dataset.doses[test_idx],
                    dataset.params[test_idx],
                    dataset.true_concentrations[test_idx],
                    dataset.time_points[test_idx],
                )

                println("✅ Dataset carregado e dividido:")
                println("   - Train: $n_train amostras")
                println("   - Val: $n_val amostras")
                println("   - Test: $n_test amostras")
                break
            catch e
                println("   ⚠️  Erro ao carregar: $e")
                continue
            end
        end
    end

    if train_data === nothing
        println("⚠️  Nenhum dataset encontrado nos caminhos padrão")
        println("   Gerando dados sintéticos para demonstração...")

        # Gerar dados sintéticos
        train_data = generate_synthetic_data(1000)
        val_data = generate_synthetic_data(200)
        test_data = generate_synthetic_data(200)

        println("✅ Dados sintéticos gerados:")
        println("   - Train: $(length(train_data.doses)) amostras")
        println("   - Val: $(length(val_data.doses)) amostras")
        println("   - Test: $(length(test_data.doses)) amostras")
    end

    println()

    # Treinar
    output_dir = "models/dynamic_gnn_regularized_gpu"
    println("🎯 Iniciando treinamento...")
    println("   Output: $output_dir")
    println()

    model, history = train_with_regularization(
        train_data,
        val_data,
        output_dir;
        num_epochs=100,
        batch_size=CUDA_AVAILABLE ? 32 : 16,  # Batch maior com GPU
        learning_rate=1e-3,
        weight_decay=1e-5,
        use_dropout=true,
        dropout_rate=0.2,
        early_stopping_patience=10,
        device=DEVICE,
    )

    # Validar
    println()
    metrics = validate_model(model, test_data, output_dir)

    println("=" ^ 80)
    println("✅ TREINAMENTO E VALIDAÇÃO COMPLETOS")
    println("=" ^ 80)
    println()
    println("📁 Resultados salvos em: $output_dir/")

    # Mostrar métricas
    if haskey(metrics, "cmax")
        println()
        println("📊 Métricas Finais:")
        println("   Cmax GMFE: $(round(metrics["cmax"]["geometric_mean_fold_error"], digits=3))")
        println("   AUC GMFE: $(round(metrics["auc"]["geometric_mean_fold_error"], digits=3))")
    end
end

"""
Gera dados sintéticos para demonstração.
"""
function generate_synthetic_data(n_samples::Int)::Training.PBPKDataset
    doses = Float64[]
    params = ODEPBPKSolver.PBPKParams[]
    true_concs = Matrix{Float64}[]
    time_points = Vector{Float64}[]

    for i in 1:n_samples
        dose = 50.0 + rand() * 150.0  # 50-200 mg
        p = ODEPBPKSolver.PBPKParams(
            clearance_hepatic=5.0 + rand() * 15.0,
            clearance_renal=2.0 + rand() * 8.0,
        )

        # Simular concentrações verdadeiras (usar ODE solver)
        result = ODEPBPKSolver.solve_ode(p, dose, (0.0, 24.0))
        conc_matrix = hcat([result[organ] for organ in ODEPBPKSolver.PBPK_ORGANS]...)'
        t_points = result["time"]

        push!(doses, dose)
        push!(params, p)
        push!(true_concs, conc_matrix)
        push!(time_points, t_points)
    end

    return Training.PBPKDataset(doses, params, true_concs, time_points)
end

"""
Treina modelo com regularização (GPU-ready).
"""
function train_with_regularization(
    train_data::Training.PBPKDataset,
    val_data::Training.PBPKDataset,
    output_dir::String = "models/dynamic_gnn_regularized";
    num_epochs::Int = 100,
    batch_size::Int = 32,
    learning_rate::Float64 = 1e-3,
    weight_decay::Float64 = 1e-5,
    use_dropout::Bool = true,
    dropout_rate::Float64 = 0.2,
    clip_grad_norm::Float64 = 1.0,
    early_stopping_patience::Int = 10,
    device = cpu,
)
    println("=" ^ 80)
    println("TREINAMENTO COM REGULARIZAÇÃO ANTI-OVERFITTING")
    println("=" ^ 80)
    println()

    # Criar modelo
    model = DynamicGNN.DynamicPBPKGNN(
        hidden_dim=64,
        num_gnn_layers=3,
        num_temporal_steps=100,
    ) |> device  # Mover para device (GPU/CPU)

    println("📊 Configuração:")
    println("   - Device: $(device == gpu ? "GPU ✅" : "CPU")")
    println("   - Epochs: $num_epochs")
    println("   - Batch size: $batch_size")
    println("   - Learning rate: $learning_rate")
    println("   - Weight decay (L2): $weight_decay")
    println("   - Dropout: $(use_dropout ? "Sim ($dropout_rate)" : "Não")")
    println("   - Gradient clipping: $clip_grad_norm")
    println("   - Early stopping patience: $early_stopping_patience")
    println()

    # Treinar
    history = Training.train_model(
        model,
        train_data,
        val_data,
        num_epochs,
        batch_size,
        learning_rate,
        device;
        weight_decay=weight_decay,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        clip_grad_norm=clip_grad_norm,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=joinpath(output_dir, "checkpoints"),
    )

    # Salvar modelo final
    mkpath(output_dir)
    using BSON
    BSON.@save joinpath(output_dir, "best_model.bson") model

    # Salvar histórico
    open(joinpath(output_dir, "training_history.json"), "w") do f
        JSON.print(f, Dict(
            "train_loss" => history["train_loss_history"],
            "val_loss" => history["val_loss_history"],
        ), 2)
    end

    println()
    println("✅ Treinamento completo!")
    println("📁 Modelo salvo em: $output_dir/")

    return model, history
end

"""
Valida modelo em dados de teste.
"""
function validate_model(
    model::DynamicGNN.DynamicPBPKGNN,
    test_data::Training.PBPKDataset,
    output_dir::String,
)
    println("=" ^ 80)
    println("VALIDAÇÃO DO MODELO")
    println("=" ^ 80)
    println()

    pred_cmax = Float64[]
    true_cmax = Float64[]
    pred_auc = Float64[]
    true_auc = Float64[]

    for i in 1:length(test_data.doses)
        result = DynamicGNN.forward(
            model,
            test_data.doses[i],
            test_data.params[i],
            test_data.time_points[i],
            DEVICE,
        )

        concs = result["concentrations"][1, :, :]  # [num_organs, num_time_points]
        blood_idx = ODEPBPKSolver.BLOOD_IDX

        # Cmax
        pred_cmax_val = maximum(concs[blood_idx, :])
        true_cmax_val = maximum(test_data.true_concentrations[i][blood_idx, :])
        push!(pred_cmax, pred_cmax_val)
        push!(true_cmax, true_cmax_val)

        # AUC (trapezoidal)
        pred_auc_val = Validation.trapezoidal_auc(
            test_data.time_points[i],
            concs[blood_idx, :],
        )
        true_auc_val = Validation.trapezoidal_auc(
            test_data.time_points[i],
            test_data.true_concentrations[i][blood_idx, :],
        )
        push!(pred_auc, pred_auc_val)
        push!(true_auc, true_auc_val)
    end

    # Calcular métricas
    cmax_gmfe = Validation.geometric_mean_fold_error(pred_cmax, true_cmax)
    cmax_r2 = Validation.r_squared(pred_cmax, true_cmax)
    cmax_pct_2x = Validation.percent_within_fold(pred_cmax, true_cmax, 2.0)

    auc_gmfe = Validation.geometric_mean_fold_error(pred_auc, true_auc)
    auc_r2 = Validation.r_squared(pred_auc, true_auc)
    auc_pct_2x = Validation.percent_within_fold(pred_auc, true_auc, 2.0)

    metrics = Dict(
        "cmax" => Dict(
            "geometric_mean_fold_error" => cmax_gmfe,
            "r_squared" => cmax_r2,
            "percent_within_2.0x" => cmax_pct_2x,
        ),
        "auc" => Dict(
            "geometric_mean_fold_error" => auc_gmfe,
            "r_squared" => auc_r2,
            "percent_within_2.0x" => auc_pct_2x,
        ),
    )

    # Salvar métricas
    mkpath(output_dir)
    open(joinpath(output_dir, "validation_metrics.json"), "w") do f
        JSON.print(f, metrics, 2)
    end

    println("📊 Métricas de Validação:")
    println("   Cmax:")
    println("     - GMFE: $(round(cmax_gmfe, digits=3))")
    println("     - R²: $(round(cmax_r2, digits=3))")
    println("     - % within 2.0x: $(round(cmax_pct_2x, digits=1))%")
    println("   AUC:")
    println("     - GMFE: $(round(auc_gmfe, digits=3))")
    println("     - R²: $(round(auc_r2, digits=3))")
    println("     - % within 2.0x: $(round(auc_pct_2x, digits=1))%")
    println()

    return metrics
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

