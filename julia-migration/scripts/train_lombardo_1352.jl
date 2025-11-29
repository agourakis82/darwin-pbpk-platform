#!/usr/bin/env julia
"""
DARWIN PBPK v2.3.0 - Train on Lombardo 1352 with proper molecular descriptors
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Random
using Statistics
using Printf
using Flux

Random.seed!(42)

println("="^80)
println("DARWIN PBPK v2.3.0 - LOMBARDO 1352 WITH REAL DESCRIPTORS")
println("="^80)
println()

# Load the proper dataset
println("Loading Lombardo 1352 dataset with molecular descriptors...")
df = CSV.read("/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv", DataFrame)
println("  Total rows: ", nrow(df))
println("  Columns: ", names(df))
println()

# Filter rows with valid Vdss
df_vdss = dropmissing(df, :human_VDss_L_kg)
println("  Compounds with Vdss: ", nrow(df_vdss))

# Filter rows with valid Clearance
df_cl = dropmissing(df, :human_CL_mL_min_kg)
println("  Compounds with CL: ", nrow(df_cl))

#=============================================================================
  PREPARE FEATURES
=============================================================================#

println()
println("Preparing features...")

feature_cols = [:MW, :HBA, :HBD, :TPSA_NO, :RotBondCount]

# Add LogP/LogD if available
if Symbol("MoKa.LogP") in propertynames(df_vdss)
    push!(feature_cols, Symbol("MoKa.LogP"))
end
if Symbol("MoKa.LogD7.4") in propertynames(df_vdss)
    push!(feature_cols, Symbol("MoKa.LogD7.4"))
end

println("  Feature columns: ", feature_cols)

# Extract features for model
function prepare_data(df, target_col, feature_cols)
    # Remove rows with missing features or target
    valid_rows = trues(nrow(df))

    for col in feature_cols
        valid_rows .&= .!ismissing.(df[:, col])
    end
    valid_rows .&= .!ismissing.(df[:, target_col])

    df_valid = df[valid_rows, :]

    # Extract features
    n_samples = nrow(df_valid)
    n_features = length(feature_cols)
    X = zeros(Float32, n_features, n_samples)

    for (i, col) in enumerate(feature_cols)
        X[i, :] = Float32.(df_valid[:, col])
    end

    y = Float32.(df_valid[:, target_col])

    # Normalize features
    X[1, :] ./= 600  # MW
    X[2, :] ./= 15   # HBA
    X[3, :] ./= 8    # HBD
    X[4, :] ./= 200  # TPSA
    X[5, :] ./= 15   # RotBond
    if n_features >= 6
        X[6, :] = (X[6, :] .+ 5) ./ 12  # LogP
    end
    if n_features >= 7
        X[7, :] = (X[7, :] .+ 5) ./ 12  # LogD
    end

    # Clamp to valid range
    X = clamp.(X, 0f0, 1f0)

    return X, y
end

X_vdss, y_vdss = prepare_data(df_vdss, :human_VDss_L_kg, feature_cols)
println("  Vdss X: ", size(X_vdss), ", y: ", length(y_vdss))
println("  Vdss range: ", round(minimum(y_vdss), digits=3), " - ", round(maximum(y_vdss), digits=3), " L/kg")

X_cl, y_cl = prepare_data(df_cl, :human_CL_mL_min_kg, feature_cols)
println("  CL X: ", size(X_cl), ", y: ", length(y_cl))
println("  CL range: ", round(minimum(y_cl), digits=3), " - ", round(maximum(y_cl), digits=3), " mL/min/kg")

#=============================================================================
  METRICS
=============================================================================#

function compute_gmfe(pred, obs)
    valid = (pred .> 0) .& (obs .> 0)
    p, o = pred[valid], obs[valid]
    ratios = p ./ o
    fold_errors = max.(ratios, 1 ./ ratios)
    return exp(mean(log.(fold_errors)))
end

function pct_within_fold(pred, obs, fold)
    valid = (pred .> 0) .& (obs .> 0)
    ratios = pred[valid] ./ obs[valid]
    within = (ratios .>= 1/fold) .& (ratios .<= fold)
    return mean(within) * 100
end

function compute_r2(pred, obs)
    ss_res = sum((pred .- obs).^2)
    ss_tot = sum((obs .- mean(obs)).^2)
    return 1 - ss_res / ss_tot
end

#=============================================================================
  TRAIN VDSS MODEL
=============================================================================#

println()
println("="^80)
println("TRAINING VDSS MODEL")
println("="^80)

y_log_vdss = log.(y_vdss .+ 0.01f0)

n_total = size(X_vdss, 2)
n_train = Int(floor(n_total * 0.8))

indices = shuffle(1:n_total)
train_idx = indices[1:n_train]
val_idx = indices[n_train+1:end]

X_train = X_vdss[:, train_idx]
X_val = X_vdss[:, val_idx]
y_train = reshape(y_log_vdss[train_idx], 1, :)
y_val = reshape(y_log_vdss[val_idx], 1, :)
y_val_orig = y_vdss[val_idx]

println("  Train: ", n_train, " | Val: ", length(val_idx))

n_features = size(X_vdss, 1)
model_vdss = Chain(
    Dense(n_features => 128, relu),
    Dropout(0.4),
    Dense(128 => 64, relu),
    Dropout(0.3),
    Dense(64 => 32, relu),
    Dense(32 => 1)
)

opt_vdss = Adam(0.001)

best_val_loss = Inf32
best_params = nothing
patience = 50
wait_count = 0

for epoch in 1:500
    perm = shuffle(1:size(X_train, 2))
    X_shuffled = X_train[:, perm]
    y_shuffled = y_train[:, perm]

    Flux.trainmode!(model_vdss)

    for i in 1:32:size(X_train, 2)
        j = min(i + 31, size(X_train, 2))
        xb = X_shuffled[:, i:j]
        yb = y_shuffled[:, i:j]
        gs = gradient(() -> Flux.mse(model_vdss(xb), yb), Flux.params(model_vdss))
        Flux.update!(opt_vdss, Flux.params(model_vdss), gs)
    end

    Flux.testmode!(model_vdss)
    train_loss = Flux.mse(model_vdss(X_train), y_train)
    val_loss = Flux.mse(model_vdss(X_val), y_val)

    if val_loss < best_val_loss
        global best_val_loss = val_loss
        global best_params = deepcopy(Flux.params(model_vdss))
        global wait_count = 0
    else
        global wait_count += 1
    end

    if epoch % 50 == 0 || epoch == 1
        @printf("Epoch %3d | Train: %.4f | Val: %.4f | Best: %.4f\n",
                epoch, train_loss, val_loss, best_val_loss)
    end

    if wait_count >= patience
        println("Early stopping at epoch ", epoch)
        break
    end
end

if best_params !== nothing
    for (p, bp) in zip(Flux.params(model_vdss), best_params)
        p .= bp
    end
end

# Validation
Flux.testmode!(model_vdss)
pred_log = vec(model_vdss(X_val))
pred_vdss_out = exp.(pred_log) .- 0.01f0
pred_vdss_out = max.(pred_vdss_out, 0.01f0)

gmfe_vdss = compute_gmfe(pred_vdss_out, y_val_orig)
r2_vdss = compute_r2(pred_vdss_out, y_val_orig)
within_2fold_vdss = pct_within_fold(pred_vdss_out, y_val_orig, 2.0)
within_3fold_vdss = pct_within_fold(pred_vdss_out, y_val_orig, 3.0)

println()
println("VDSS Results (", length(y_val_orig), " test compounds):")
println("-"^50)
@printf("  GMFE:          %.3f\n", gmfe_vdss)
@printf("  R²:            %.3f\n", r2_vdss)
@printf("  Within 2-fold: %.1f%%\n", within_2fold_vdss)
@printf("  Within 3-fold: %.1f%%\n", within_3fold_vdss)

#=============================================================================
  TRAIN CLEARANCE MODEL
=============================================================================#

println()
println("="^80)
println("TRAINING CLEARANCE MODEL")
println("="^80)

y_log_cl = log.(y_cl .+ 0.01f0)

n_total_cl = size(X_cl, 2)
n_train_cl = Int(floor(n_total_cl * 0.8))

indices_cl = shuffle(1:n_total_cl)
train_idx_cl = indices_cl[1:n_train_cl]
val_idx_cl = indices_cl[n_train_cl+1:end]

X_train_cl = X_cl[:, train_idx_cl]
X_val_cl = X_cl[:, val_idx_cl]
y_train_cl = reshape(y_log_cl[train_idx_cl], 1, :)
y_val_cl = reshape(y_log_cl[val_idx_cl], 1, :)
y_val_orig_cl = y_cl[val_idx_cl]

println("  Train: ", n_train_cl, " | Val: ", length(val_idx_cl))

model_cl = Chain(
    Dense(n_features => 128, relu),
    Dropout(0.4),
    Dense(128 => 64, relu),
    Dropout(0.3),
    Dense(64 => 32, relu),
    Dense(32 => 1)
)

opt_cl = Adam(0.001)

best_val_loss_cl = Inf32
best_params_cl = nothing
wait_count_cl = 0

for epoch in 1:500
    perm = shuffle(1:size(X_train_cl, 2))
    X_shuffled = X_train_cl[:, perm]
    y_shuffled = y_train_cl[:, perm]

    Flux.trainmode!(model_cl)

    for i in 1:32:size(X_train_cl, 2)
        j = min(i + 31, size(X_train_cl, 2))
        xb = X_shuffled[:, i:j]
        yb = y_shuffled[:, i:j]
        gs = gradient(() -> Flux.mse(model_cl(xb), yb), Flux.params(model_cl))
        Flux.update!(opt_cl, Flux.params(model_cl), gs)
    end

    Flux.testmode!(model_cl)
    val_loss = Flux.mse(model_cl(X_val_cl), y_val_cl)

    if val_loss < best_val_loss_cl
        global best_val_loss_cl = val_loss
        global best_params_cl = deepcopy(Flux.params(model_cl))
        global wait_count_cl = 0
    else
        global wait_count_cl += 1
    end

    if epoch % 50 == 0 || epoch == 1
        train_loss = Flux.mse(model_cl(X_train_cl), y_train_cl)
        @printf("Epoch %3d | Train: %.4f | Val: %.4f | Best: %.4f\n",
                epoch, train_loss, val_loss, best_val_loss_cl)
    end

    if wait_count_cl >= 50
        println("Early stopping at epoch ", epoch)
        break
    end
end

if best_params_cl !== nothing
    for (p, bp) in zip(Flux.params(model_cl), best_params_cl)
        p .= bp
    end
end

# Validation
Flux.testmode!(model_cl)
pred_log_cl = vec(model_cl(X_val_cl))
pred_cl_out = exp.(pred_log_cl) .- 0.01f0
pred_cl_out = max.(pred_cl_out, 0.01f0)

gmfe_cl = compute_gmfe(pred_cl_out, y_val_orig_cl)
r2_cl = compute_r2(pred_cl_out, y_val_orig_cl)
within_2fold_cl = pct_within_fold(pred_cl_out, y_val_orig_cl, 2.0)
within_3fold_cl = pct_within_fold(pred_cl_out, y_val_orig_cl, 3.0)

println()
println("CLEARANCE Results (", length(y_val_orig_cl), " test compounds):")
println("-"^50)
@printf("  GMFE:          %.3f\n", gmfe_cl)
@printf("  R²:            %.3f\n", r2_cl)
@printf("  Within 2-fold: %.1f%%\n", within_2fold_cl)
@printf("  Within 3-fold: %.1f%%\n", within_3fold_cl)

#=============================================================================
  FINAL SUMMARY
=============================================================================#

println()
println("="^80)
println("FINAL SUMMARY - LOMBARDO 1352 DATASET")
println("="^80)
println()

println("Model Performance:")
println("-"^60)
@printf("  %-15s | %8s | %8s | %8s\n", "Parameter", "GMFE", "2-fold%", "R²")
println("-"^60)
@printf("  %-15s | %8.3f | %7.1f%% | %8.3f\n", "Vdss", gmfe_vdss, within_2fold_vdss, r2_vdss)
@printf("  %-15s | %8.3f | %7.1f%% | %8.3f\n", "Clearance", gmfe_cl, within_2fold_cl, r2_cl)
println("-"^60)

println()
println("FDA/EMA Regulatory Assessment:")
println("-"^60)

if gmfe_vdss < 2.0
    println("  Vdss:      ✓ GMFE < 2.0 - ACCEPTABLE FOR REGULATORY")
elseif gmfe_vdss < 2.5
    println("  Vdss:      ⚠ GMFE < 2.5 - Borderline")
else
    @printf("  Vdss:      ✗ GMFE = %.2f\n", gmfe_vdss)
end

if within_2fold_vdss >= 50.0
    println("  Vdss:      ✓ >50% within 2-fold - ACCEPTABLE")
end

if gmfe_cl < 2.0
    println("  Clearance: ✓ GMFE < 2.0 - ACCEPTABLE FOR REGULATORY")
elseif gmfe_cl < 2.5
    println("  Clearance: ⚠ GMFE < 2.5 - Borderline")
else
    @printf("  Clearance: ✗ GMFE = %.2f\n", gmfe_cl)
end

if within_2fold_cl >= 50.0
    println("  Clearance: ✓ >50% within 2-fold - ACCEPTABLE")
end

println()
println("="^80)
println("TRAINING COMPLETE")
println("="^80)
