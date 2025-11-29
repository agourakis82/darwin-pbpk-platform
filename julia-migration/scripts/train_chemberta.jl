#!/usr/bin/env julia
"""
DARWIN PBPK - ChemBERTa Embeddings Training
Push for GMFE < 2.0
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NPZ, CSV, DataFrames, Random, Statistics, Printf, Flux, LinearAlgebra

println("="^80)
println("DARWIN PBPK - ChemBERTa HYBRID MODEL")
println("="^80)
println()

# Load ChemBERTa embeddings
println("Loading ChemBERTa embeddings (768-dim)...")
embeddings = NPZ.npzread("/home/agourakis82/workspace/darwin-pbpk-platform/data/embeddings/chemberta_embeddings.npy")
println("  Embeddings shape: ", size(embeddings))

# Load data
println("Loading compound data...")
df = CSV.read("/home/agourakis82/workspace/darwin-pbpk-platform/data/embeddings/consolidated_pbpk_v1.csv", DataFrame)
println("  Total compounds: ", nrow(df))

# Filter for compounds with Vd
vd_mask = .!ismissing.(df.vd)
df_vd = df[vd_mask, :]
println("  Compounds with Vd: ", nrow(df_vd))

# Get indices
vd_indices = findall(vd_mask)

# Get embeddings for Vd compounds
X_emb = Float32.(embeddings[vd_indices, :])'  # [768 x N]
y_vd = Float32.(df_vd.vd)
y_log_vd = log.(y_vd .+ 0.01f0)

println("  X_emb shape: ", size(X_emb))
println("  Vd range: ", round(minimum(y_vd), digits=3), " - ", round(maximum(y_vd), digits=3))
println()

# Create splits
Random.seed!(42)
n_total = length(y_vd)
indices = shuffle(1:n_total)
n_train = Int(floor(n_total * 0.8))
n_val = Int(floor(n_total * 0.1))

train_idx = indices[1:n_train]
val_idx = indices[n_train+1:n_train+n_val]
test_idx = indices[n_train+n_val+1:end]

X_train = X_emb[:, train_idx]
X_val = X_emb[:, val_idx]
X_test = X_emb[:, test_idx]

y_train = reshape(y_log_vd[train_idx], 1, :)
y_val = reshape(y_log_vd[val_idx], 1, :)
y_val_orig = y_vd[val_idx]
y_test_orig = y_vd[test_idx]

println("Split:")
println("  Train: ", size(X_train, 2))
println("  Val: ", size(X_val, 2))
println("  Test: ", size(X_test, 2))

# Metrics
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

# Model
println()
println("="^80)
println("TRAINING ChemBERTa MODEL")
println("="^80)

model = Chain(
    Dense(768 => 256, relu),
    Dropout(0.4),
    Dense(256 => 128, relu),
    Dropout(0.3),
    Dense(128 => 64, relu),
    Dense(64 => 1)
)

println("  Parameters: ", sum(length, Flux.params(model)))

opt = Adam(0.0005)

best_val_loss = Inf32
best_params = nothing
patience = 40
wait_count = 0

for epoch in 1:300
    perm = shuffle(1:size(X_train, 2))
    X_shuffled = X_train[:, perm]
    y_shuffled = y_train[:, perm]

    Flux.trainmode!(model)

    for i in 1:64:size(X_train, 2)
        j = min(i + 63, size(X_train, 2))
        xb = X_shuffled[:, i:j]
        yb = y_shuffled[:, i:j]
        gs = gradient(() -> Flux.mse(model(xb), yb), Flux.params(model))
        Flux.update!(opt, Flux.params(model), gs)
    end

    Flux.testmode!(model)
    train_loss = Flux.mse(model(X_train), y_train)
    val_loss = Flux.mse(model(X_val), y_val)

    if val_loss < best_val_loss
        global best_val_loss = val_loss
        global best_params = deepcopy(Flux.params(model))
        global wait_count = 0
    else
        global wait_count += 1
    end

    if epoch % 30 == 0 || epoch == 1
        @printf("Epoch %3d | Train: %.4f | Val: %.4f | Best: %.4f\n",
                epoch, train_loss, val_loss, best_val_loss)
    end

    if wait_count >= patience
        println("Early stopping at epoch ", epoch)
        break
    end
end

if best_params !== nothing
    for (p, bp) in zip(Flux.params(model), best_params)
        p .= bp
    end
end

# Validation
println()
println("="^80)
println("VALIDATION RESULTS")
println("="^80)

Flux.testmode!(model)

# Validation set
pred_log_val = vec(model(X_val))
pred_val = max.(exp.(pred_log_val) .- 0.01f0, 0.01f0)

gmfe_val = compute_gmfe(pred_val, y_val_orig)
within_2fold_val = pct_within_fold(pred_val, y_val_orig, 2.0)
within_3fold_val = pct_within_fold(pred_val, y_val_orig, 3.0)

println()
println("Validation Set (", length(y_val_orig), " compounds):")
println("-"^50)
@printf("  GMFE:          %.3f\n", gmfe_val)
@printf("  Within 2-fold: %.1f%%\n", within_2fold_val)
@printf("  Within 3-fold: %.1f%%\n", within_3fold_val)

# Test set
pred_log_test = vec(model(X_test))
pred_test = max.(exp.(pred_log_test) .- 0.01f0, 0.01f0)

gmfe_test = compute_gmfe(pred_test, y_test_orig)
within_2fold_test = pct_within_fold(pred_test, y_test_orig, 2.0)
within_3fold_test = pct_within_fold(pred_test, y_test_orig, 3.0)

println()
println("Test Set - HELD OUT (", length(y_test_orig), " compounds):")
println("-"^50)
@printf("  GMFE:          %.3f\n", gmfe_test)
@printf("  Within 2-fold: %.1f%%\n", within_2fold_test)
@printf("  Within 3-fold: %.1f%%\n", within_3fold_test)

println()
println("="^80)
println("FINAL ASSESSMENT")
println("="^80)

if gmfe_test < 2.0
    println("*** GMFE < 2.0 ON HELD-OUT TEST SET - FDA ACCEPTABLE! ***")
elseif gmfe_test < 2.2
    @printf("  GMFE = %.3f - Within 10%% of regulatory threshold\n", gmfe_test)
else
    @printf("  GMFE = %.3f - Above regulatory threshold\n", gmfe_test)
end

if within_2fold_test >= 50.0
    println("  >50%% within 2-fold on test set - ACCEPTABLE")
end

println()
println("="^80)
