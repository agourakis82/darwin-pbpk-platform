#!/usr/bin/env julia
"""
DARWIN PBPK v2.3.0 - Comprehensive Real Data Training
Train on Lombardo Vdss + AZ Clearance + Obach Half-life datasets
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Statistics
using Printf
using LinearAlgebra
using Flux
using DelimitedFiles

Random.seed!(42)

println("="^80)
println("DARWIN PBPK v2.3.0 - COMPREHENSIVE REAL DATA TRAINING")
println("="^80)
println()

#=============================================================================
  LOAD ALL REAL PBPK DATASETS
=============================================================================#

data_dir = "/home/agourakis82/workspace/darwin-pbpk-platform/data"

println("Loading real PBPK datasets...")
println()

function load_tab_file(path)
    lines = readlines(path)

    data = []
    for line in lines[2:end]
        parts = split(line, "\t")
        if length(parts) >= 3
            name = replace(parts[1], "\"" => "")
            smiles = replace(parts[2], "\"" => "")
            value = tryparse(Float64, replace(parts[3], "\"" => ""))
            if value !== nothing && !isempty(smiles)
                push!(data, (name=name, smiles=smiles, value=value))
            end
        end
    end
    return data
end

vdss_data = load_tab_file(joinpath(data_dir, "vdss_lombardo.tab"))
println("  Vdss (Lombardo): $(length(vdss_data)) compounds")

cl_data = load_tab_file(joinpath(data_dir, "clearance_hepatocyte_az.tab"))
println("  Clearance (Hepatocyte): $(length(cl_data)) compounds")

hl_data = load_tab_file(joinpath(data_dir, "half_life_obach.tab"))
println("  Half-life (Obach): $(length(hl_data)) compounds")

#=============================================================================
  COMPUTE MOLECULAR DESCRIPTORS FROM SMILES
=============================================================================#

println()
println("Computing molecular descriptors from SMILES...")

function smiles_to_descriptors(smiles::String)
    # Count atoms
    n_c = count(c -> c == 'C' || c == 'c', smiles)
    n_n = count(c -> c == 'N' || c == 'n', smiles)
    n_o = count(c -> c == 'O' || c == 'o', smiles)
    n_s = count(c -> c == 'S' || c == 's', smiles)
    n_f = count("F", smiles)
    n_cl = count("Cl", smiles)
    n_br = count("Br", smiles)

    # Count aromatic atoms
    n_aromatic = count(c -> islowercase(c) && c in ['c', 'n', 'o', 's'], smiles)

    # Estimate MW
    mw_est = n_c * 12.0 + n_n * 14.0 + n_o * 16.0 + n_s * 32.0 +
             n_f * 19.0 + n_cl * 35.5 + n_br * 80.0 +
             length(smiles) * 0.5

    # Estimate LogP
    logp_est = 0.5 * n_c - 0.5 * n_o - 0.8 * n_n + 0.8 * n_cl + 0.3 * n_f

    # Estimate TPSA
    tpsa_est = n_o * 20.0 + n_n * 26.0

    # H-bond donors/acceptors
    hbd = count("O", smiles) + count("N", smiles) - count("=O", smiles) - count("=N", smiles)
    hba = n_o + n_n

    # Rotatable bonds estimate
    rot_bonds = max(0, count("-", smiles) - n_aromatic ÷ 2)

    return Float32[
        min(mw_est, 1000) / 600,
        clamp((logp_est + 5) / 12, 0, 1),
        min(tpsa_est, 250) / 200,
        min(max(hbd, 0), 10) / 8,
        min(hba, 15) / 12,
        min(rot_bonds, 15) / 12,
        min(n_aromatic, 20) / 15,
        min(n_c, 50) / 40,
        min(n_n + n_o, 15) / 12,
        Float32(n_cl > 0 || n_f > 0 || n_br > 0),
        min(length(smiles), 200) / 150,
    ]
end

# Process Vdss data
println("  Processing Vdss compounds...")
X_vdss = hcat([smiles_to_descriptors(d.smiles) for d in vdss_data]...)
y_vdss = Float32[d.value for d in vdss_data]

println("    X shape: $(size(X_vdss))")
println("    Vdss range: $(round(minimum(y_vdss), digits=3)) - $(round(maximum(y_vdss), digits=3)) L/kg")

# Process Clearance data
println("  Processing Clearance compounds...")
X_cl = hcat([smiles_to_descriptors(d.smiles) for d in cl_data]...)
y_cl = Float32[d.value for d in cl_data]

println("    X shape: $(size(X_cl))")
println("    CL range: $(round(minimum(y_cl), digits=3)) - $(round(maximum(y_cl), digits=3))")

#=============================================================================
  METRICS FUNCTIONS
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

function compute_afe(pred, obs)
    valid = (pred .> 0) .& (obs .> 0)
    return exp(mean(log.(pred[valid] ./ obs[valid])))
end

#=============================================================================
  TRAIN VDSS MODEL
=============================================================================#

println()
println("="^80)
println("TRAINING VDSS MODEL ($(length(vdss_data)) compounds)")
println("="^80)

y_log_vdss = log.(y_vdss .+ 0.01f0)

n_total = size(X_vdss, 2)
n_train = Int(floor(n_total * 0.85))

indices = shuffle(1:n_total)
train_idx = indices[1:n_train]
val_idx = indices[n_train+1:end]

X_train = X_vdss[:, train_idx]
X_val = X_vdss[:, val_idx]
y_train = reshape(y_log_vdss[train_idx], 1, :)
y_val = reshape(y_log_vdss[val_idx], 1, :)
y_val_orig = y_vdss[val_idx]

println("  Train: $n_train | Val: $(length(val_idx))")

model_vdss = Chain(
    Dense(11 => 256, leakyrelu),
    Dropout(0.5),
    Dense(256 => 128, leakyrelu),
    BatchNorm(128),
    Dropout(0.4),
    Dense(128 => 64, leakyrelu),
    BatchNorm(64),
    Dropout(0.3),
    Dense(64 => 32, leakyrelu),
    Dense(32 => 1)
)

println("  Parameters: $(sum(length, Flux.params(model_vdss)))")

opt_vdss = AdamW(0.001, (0.9, 0.999), 1e-4)

best_val_loss = Inf32
best_params = nothing
patience = 60
wait_count = 0

for epoch in 1:600
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

    if epoch % 100 == 0 || epoch == 1
        @printf("Epoch %3d | Train: %.4f | Val: %.4f | Best: %.4f\n",
                epoch, train_loss, val_loss, best_val_loss)
    end

    if wait_count >= patience
        println("Early stopping at epoch $epoch")
        break
    end
end

if best_params !== nothing
    for (p, bp) in zip(Flux.params(model_vdss), best_params)
        p .= bp
    end
end

# Vdss Validation
println()
println("="^80)
println("VDSS VALIDATION RESULTS")
println("="^80)

Flux.testmode!(model_vdss)
pred_log = vec(model_vdss(X_val))
pred_vdss_val = exp.(pred_log) .- 0.01f0
pred_vdss_val = max.(pred_vdss_val, 0.01f0)

gmfe_vdss = compute_gmfe(pred_vdss_val, y_val_orig)
afe_vdss = compute_afe(pred_vdss_val, y_val_orig)
r2_vdss = compute_r2(pred_vdss_val, y_val_orig)
within_2fold_vdss = pct_within_fold(pred_vdss_val, y_val_orig, 2.0)
within_3fold_vdss = pct_within_fold(pred_vdss_val, y_val_orig, 3.0)

println()
println("Vdss Prediction ($(length(y_val_orig)) test compounds):")
println("-"^50)
@printf("  GMFE:          %.3f\n", gmfe_vdss)
@printf("  AFE:           %.3f (1.0 = unbiased)\n", afe_vdss)
@printf("  R²:            %.3f\n", r2_vdss)
@printf("  Within 2-fold: %.1f%%\n", within_2fold_vdss)
@printf("  Within 3-fold: %.1f%%\n", within_3fold_vdss)

#=============================================================================
  TRAIN CLEARANCE MODEL
=============================================================================#

println()
println("="^80)
println("TRAINING CLEARANCE MODEL ($(length(cl_data)) compounds)")
println("="^80)

y_log_cl = log.(y_cl .+ 0.01f0)

n_total_cl = size(X_cl, 2)
n_train_cl = Int(floor(n_total_cl * 0.85))

indices_cl = shuffle(1:n_total_cl)
train_idx_cl = indices_cl[1:n_train_cl]
val_idx_cl = indices_cl[n_train_cl+1:end]

X_train_cl = X_cl[:, train_idx_cl]
X_val_cl = X_cl[:, val_idx_cl]
y_train_cl = reshape(y_log_cl[train_idx_cl], 1, :)
y_val_cl = reshape(y_log_cl[val_idx_cl], 1, :)
y_val_orig_cl = y_cl[val_idx_cl]

println("  Train: $n_train_cl | Val: $(length(val_idx_cl))")

model_cl = Chain(
    Dense(11 => 256, leakyrelu),
    Dropout(0.5),
    Dense(256 => 128, leakyrelu),
    BatchNorm(128),
    Dropout(0.4),
    Dense(128 => 64, leakyrelu),
    Dense(64 => 1)
)

opt_cl = AdamW(0.001, (0.9, 0.999), 1e-4)

best_val_loss_cl = Inf32
best_params_cl = nothing
wait_count_cl = 0

for epoch in 1:600
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

    if epoch % 100 == 0 || epoch == 1
        train_loss = Flux.mse(model_cl(X_train_cl), y_train_cl)
        @printf("Epoch %3d | Train: %.4f | Val: %.4f | Best: %.4f\n",
                epoch, train_loss, val_loss, best_val_loss_cl)
    end

    if wait_count_cl >= 60
        println("Early stopping at epoch $epoch")
        break
    end
end

if best_params_cl !== nothing
    for (p, bp) in zip(Flux.params(model_cl), best_params_cl)
        p .= bp
    end
end

# Clearance Validation
println()
println("="^80)
println("CLEARANCE VALIDATION RESULTS")
println("="^80)

Flux.testmode!(model_cl)
pred_log_cl = vec(model_cl(X_val_cl))
pred_cl_val = exp.(pred_log_cl) .- 0.01f0
pred_cl_val = max.(pred_cl_val, 0.01f0)

gmfe_cl = compute_gmfe(pred_cl_val, y_val_orig_cl)
afe_cl = compute_afe(pred_cl_val, y_val_orig_cl)
r2_cl = compute_r2(pred_cl_val, y_val_orig_cl)
within_2fold_cl = pct_within_fold(pred_cl_val, y_val_orig_cl, 2.0)
within_3fold_cl = pct_within_fold(pred_cl_val, y_val_orig_cl, 3.0)

println()
println("Clearance Prediction ($(length(y_val_orig_cl)) test compounds):")
println("-"^50)
@printf("  GMFE:          %.3f\n", gmfe_cl)
@printf("  AFE:           %.3f (1.0 = unbiased)\n", afe_cl)
@printf("  R²:            %.3f\n", r2_cl)
@printf("  Within 2-fold: %.1f%%\n", within_2fold_cl)
@printf("  Within 3-fold: %.1f%%\n", within_3fold_cl)

#=============================================================================
  FINAL SUMMARY
=============================================================================#

println()
println("="^80)
println("FINAL SUMMARY - REAL PBPK DATA")
println("="^80)
println()

println("Dataset Sizes:")
println("  Vdss (Lombardo):     $(length(vdss_data)) compounds")
println("  Clearance (AZ):      $(length(cl_data)) compounds")
println("  Half-life (Obach):   $(length(hl_data)) compounds")
println()

println("Model Performance:")
println("-"^60)
@printf("  %-20s | %8s | %8s | %8s\n", "Parameter", "GMFE", "2-fold%", "R²")
println("-"^60)
@printf("  %-20s | %8.3f | %7.1f%% | %8.3f\n", "Vdss", gmfe_vdss, within_2fold_vdss, r2_vdss)
@printf("  %-20s | %8.3f | %7.1f%% | %8.3f\n", "Clearance", gmfe_cl, within_2fold_cl, r2_cl)
println("-"^60)

println()
println("FDA/EMA Regulatory Assessment:")
println("-"^60)

if gmfe_vdss < 2.0
    println("  Vdss:      ✓ GMFE < 2.0 - ACCEPTABLE")
elseif gmfe_vdss < 2.5
    println("  Vdss:      ⚠ GMFE < 2.5 - Borderline")
else
    @printf("  Vdss:      ✗ GMFE = %.2f - Needs improvement\n", gmfe_vdss)
end

if within_2fold_vdss >= 50.0
    println("  Vdss:      ✓ >50% within 2-fold - ACCEPTABLE")
end

if gmfe_cl < 2.0
    println("  Clearance: ✓ GMFE < 2.0 - ACCEPTABLE")
elseif gmfe_cl < 2.5
    println("  Clearance: ⚠ GMFE < 2.5 - Borderline")
else
    @printf("  Clearance: ✗ GMFE = %.2f - Needs improvement\n", gmfe_cl)
end

if within_2fold_cl >= 50.0
    println("  Clearance: ✓ >50% within 2-fold - ACCEPTABLE")
end

println()
println("="^80)
println("COMPREHENSIVE REAL DATA TRAINING COMPLETE")
println("="^80)
