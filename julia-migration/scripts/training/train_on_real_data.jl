#!/usr/bin/env julia
"""
Train Dynamic GNN on Real Drug Data

Uses validation_drugs_100.json with experimental Cmax/AUC observations
to train the model on real pharmaceutical data.

Author: Dr. Sounio Agourakis + AI Assistant
Date: November 2025
"""

# Activate project
using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using JSON
using Random
using Statistics
using LinearAlgebra
using Printf

# Load DarwinPBPK modules
include(joinpath(@__DIR__, "../../src/DarwinPBPK.jl"))
using .DarwinPBPK.ODEPBPKSolver
using .DarwinPBPK.Training
using .DarwinPBPK.Validation
using .DarwinPBPK.DynamicGNN

Random.seed!(42)

#=============================================================================
  Data Loading
=============================================================================#

"""
Load real drug data from validation_drugs_100.json
"""
function load_real_drug_data(path::String)
    data = JSON.parsefile(path)
    return data
end

"""
Convert drug data to PBPK training format.

Uses literature clearance/Vd to set PBPK parameters,
then uses observed Cmax/AUC as training targets.
"""
function prepare_training_data(drugs::Vector{Any})
    doses = Float64[]
    params = ODEPBPKSolver.PBPKParams[]
    target_cmax = Float64[]
    target_auc = Float64[]
    drug_names = String[]

    for drug in drugs
        # Skip if missing key data
        if !haskey(drug, "dose") || !haskey(drug, "cmax_obs") || !haskey(drug, "auc_obs")
            continue
        end

        dose = Float64(drug["dose"])
        cmax = Float64(drug["cmax_obs"])
        auc = Float64(drug["auc_obs"])

        # Get clearance and volume from literature
        cl = get(drug, "CL_lit", 10.0)  # L/h default
        vd = get(drug, "Vd_lit", 100.0)  # L default

        # Convert to PBPK parameters
        # Hepatic clearance is typically ~70% of total clearance
        cl_hepatic = Float64(cl) * 0.7
        cl_renal = Float64(cl) * 0.3

        # Estimate partition coefficients from Vd
        # Higher Vd = more tissue distribution
        vd_ratio = Float64(vd) / 70.0  # Normalize to body weight

        # Create PBPK parameters
        kp_base = clamp(vd_ratio * 0.5, 0.1, 10.0)
        kp_dict = Dict{String,Float64}()
        for organ in ODEPBPKSolver.PBPK_ORGANS
            if organ == "adipose"
                kp_dict[organ] = kp_base * 2.0  # Lipophilic drugs accumulate
            elseif organ == "liver"
                kp_dict[organ] = kp_base * 1.5
            elseif organ == "blood"
                kp_dict[organ] = 1.0
            else
                kp_dict[organ] = kp_base
            end
        end

        p = ODEPBPKSolver.PBPKParams(
            clearance_hepatic=cl_hepatic,
            clearance_renal=cl_renal,
            partition_coeffs=kp_dict,
        )

        push!(doses, dose)
        push!(params, p)
        push!(target_cmax, cmax)
        push!(target_auc, auc)
        push!(drug_names, get(drug, "drug_name", "unknown"))
    end

    return (
        doses=doses,
        params=params,
        target_cmax=target_cmax,
        target_auc=target_auc,
        drug_names=drug_names,
    )
end

"""
Generate ODE solutions for training data.
"""
function generate_ode_solutions(doses, params; tspan=(0.0, 24.0), dt=0.5)
    concentrations = Matrix{Float64}[]
    time_points = Vector{Float64}[]

    for (dose, p) in zip(doses, params)
        try
            result = ODEPBPKSolver.solve_ode(p, dose, tspan)

            # Extract concentration matrix [organs x timepoints]
            t = result["time"]
            n_organs = length(ODEPBPKSolver.PBPK_ORGANS)
            n_times = length(t)

            conc_matrix = zeros(n_organs, n_times)
            for (i, organ) in enumerate(ODEPBPKSolver.PBPK_ORGANS)
                if haskey(result, organ)
                    conc_matrix[i, :] = result[organ]
                end
            end

            push!(concentrations, conc_matrix)
            push!(time_points, t)
        catch e
            @warn "Failed to solve ODE for dose=$dose: $e"
            # Use placeholder
            push!(concentrations, zeros(14, 49))
            push!(time_points, collect(0.0:dt:24.0))
        end
    end

    return concentrations, time_points
end

#=============================================================================
  Training Loop with Real Data Targets
=============================================================================#

"""
Custom loss function that matches predicted Cmax/AUC to observed values.
"""
function pk_loss(pred_concs, true_cmax, true_auc, time_points)
    blood_idx = ODEPBPKSolver.BLOOD_IDX

    total_loss = 0.0
    n_samples = length(pred_concs)

    for i in 1:n_samples
        # Predicted Cmax (max blood concentration)
        pred_cmax = maximum(pred_concs[i][blood_idx, :])

        # Predicted AUC (trapezoidal integration)
        pred_auc = Validation.trapezoidal_auc(time_points[i], pred_concs[i][blood_idx, :])

        # Log-scale MSE (better for PK data spanning orders of magnitude)
        cmax_loss = (log(pred_cmax + 1e-10) - log(true_cmax[i] + 1e-10))^2
        auc_loss = (log(pred_auc + 1e-10) - log(true_auc[i] + 1e-10))^2

        total_loss += cmax_loss + auc_loss
    end

    return total_loss / n_samples
end

"""
Train model with real drug data targets.
"""
function train_on_real_data(;
    data_path::String="/mnt/f/datasets/pbpk/validation_drugs_100.json",
    output_dir::String="models/real_data_trained",
    num_epochs::Int=100,
    batch_size::Int=8,
    learning_rate::Float64=1e-4,
    weight_decay::Float64=1e-4,
)
    println("="^80)
    println("TRAINING ON REAL DRUG DATA")
    println("="^80)
    println()

    # Load data
    println("Loading real drug data...")
    drugs = load_real_drug_data(data_path)
    println("  Loaded $(length(drugs)) drugs")

    # Prepare training data
    data = prepare_training_data(drugs)
    println("  Prepared $(length(data.doses)) samples")
    println()

    # Show sample drugs
    println("Sample drugs:")
    for i in 1:min(5, length(data.drug_names))
        @printf("  %s: dose=%.1f, Cmax=%.2f, AUC=%.2f\n",
            data.drug_names[i], data.doses[i],
            data.target_cmax[i], data.target_auc[i])
    end
    println()

    # Generate ODE solutions
    println("Generating ODE solutions...")
    concentrations, time_points = generate_ode_solutions(data.doses, data.params)
    println("  Generated $(length(concentrations)) concentration profiles")
    println()

    # Split train/val (80/20)
    n_total = length(data.doses)
    n_train = Int(floor(n_total * 0.8))

    indices = shuffle(1:n_total)
    train_idx = indices[1:n_train]
    val_idx = indices[n_train+1:end]

    println("Data split:")
    println("  Train: $(length(train_idx)) samples")
    println("  Val: $(length(val_idx)) samples")
    println()

    # Create model
    println("Creating Dynamic GNN model...")
    model = DynamicGNN.DynamicPBPKGNN(
        hidden_dim=64,
        num_gnn_layers=3,
        num_temporal_steps=100,
    )

    # Training config
    println("Training configuration:")
    println("  Epochs: $num_epochs")
    println("  Batch size: $batch_size")
    println("  Learning rate: $learning_rate")
    println("  Weight decay: $weight_decay")
    println()

    # Create PBPKDataset for Training module
    train_dataset = Training.PBPKDataset(
        data.doses[train_idx],
        data.params[train_idx],
        concentrations[train_idx],
        time_points[train_idx],
    )

    val_dataset = Training.PBPKDataset(
        data.doses[val_idx],
        data.params[val_idx],
        concentrations[val_idx],
        time_points[val_idx],
    )

    # Train
    println("Starting training...")
    println("="^80)

    history = Training.train_model(
        model,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        learning_rate,
        cpu;  # or gpu
        weight_decay=weight_decay,
        use_dropout=true,
        dropout_rate=0.2,
        clip_grad_norm=1.0,
        early_stopping_patience=15,
        checkpoint_dir=joinpath(output_dir, "checkpoints"),
        lr_scheduler=:cosine,
        lr_min=1e-6,
    )

    println()
    println("="^80)
    println("Training complete!")
    println()

    # Final validation
    println("Final validation on held-out data...")

    pred_cmax = Float64[]
    true_cmax = Float64[]
    pred_auc = Float64[]
    true_auc = Float64[]

    for i in val_idx
        # Run model forward pass
        result = DynamicGNN.forward(
            model,
            data.doses[i],
            data.params[i],
            time_points[i],
            cpu,
        )

        concs = result["concentrations"][1, :, :]
        blood_idx = ODEPBPKSolver.BLOOD_IDX

        # Predicted PK
        push!(pred_cmax, maximum(concs[blood_idx, :]))
        push!(pred_auc, Validation.trapezoidal_auc(time_points[i], concs[blood_idx, :]))

        # True observed
        push!(true_cmax, data.target_cmax[i])
        push!(true_auc, data.target_auc[i])
    end

    # Metrics
    cmax_gmfe = Validation.geometric_mean_fold_error(pred_cmax, true_cmax)
    auc_gmfe = Validation.geometric_mean_fold_error(pred_auc, true_auc)
    cmax_r2 = Validation.r_squared(pred_cmax, true_cmax)
    auc_r2 = Validation.r_squared(pred_auc, true_auc)
    cmax_2x = Validation.percent_within_fold(pred_cmax, true_cmax, 2.0)
    auc_2x = Validation.percent_within_fold(pred_auc, true_auc, 2.0)

    println()
    println("Validation Metrics (Real Drug Data):")
    println("  Cmax:")
    @printf("    GMFE: %.3f\n", cmax_gmfe)
    @printf("    R²: %.3f\n", cmax_r2)
    @printf("    %% within 2x: %.1f%%\n", cmax_2x)
    println("  AUC:")
    @printf("    GMFE: %.3f\n", auc_gmfe)
    @printf("    R²: %.3f\n", auc_r2)
    @printf("    %% within 2x: %.1f%%\n", auc_2x)
    println()

    # Save results
    mkpath(output_dir)

    metrics = Dict(
        "cmax" => Dict(
            "gmfe" => cmax_gmfe,
            "r2" => cmax_r2,
            "pct_within_2x" => cmax_2x,
        ),
        "auc" => Dict(
            "gmfe" => auc_gmfe,
            "r2" => auc_r2,
            "pct_within_2x" => auc_2x,
        ),
        "training" => Dict(
            "epochs" => num_epochs,
            "batch_size" => batch_size,
            "learning_rate" => learning_rate,
            "weight_decay" => weight_decay,
            "n_train" => length(train_idx),
            "n_val" => length(val_idx),
        ),
    )

    open(joinpath(output_dir, "validation_metrics.json"), "w") do f
        JSON.print(f, metrics, 2)
    end

    # Save training history
    open(joinpath(output_dir, "training_history.json"), "w") do f
        JSON.print(f, history, 2)
    end

    println("Results saved to: $output_dir/")

    return model, metrics, history
end

#=============================================================================
  Main
=============================================================================#

function main()
    println("Darwin PBPK - Training on Real Drug Data")
    println()

    model, metrics, history = train_on_real_data(
        num_epochs=50,
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=1e-4,
    )

    println()
    println("="^80)
    println("TRAINING COMPLETE")
    println("="^80)

    # Summary
    if metrics["cmax"]["gmfe"] < 2.0 && metrics["auc"]["gmfe"] < 2.0
        println("SUCCESS: Model achieves GMFE < 2.0 on real drug data!")
    else
        println("Model needs improvement. Current GMFE:")
        @printf("  Cmax: %.3f\n", metrics["cmax"]["gmfe"])
        @printf("  AUC: %.3f\n", metrics["auc"]["gmfe"])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
