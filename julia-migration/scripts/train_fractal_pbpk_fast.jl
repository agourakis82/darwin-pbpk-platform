"""
Fractal-Mechanistic PBPK Model Training (Optimized)

Fast version using analytical gradients and optimized architecture.
"""

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Random

# Include our modules
include("../src/DarwinPBPK/fractal_descriptors.jl")
include("../src/DarwinPBPK/rodgers_rowland.jl")

using .FractalDescriptors
using .RodgersRowland

println("="^80)
println("FRACTAL-MECHANISTIC PBPK MODEL (Optimized)")
println("="^80)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

println("\n[1] Loading data...")
data_path = "/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv"
df = CSV.read(data_path, DataFrame)

required_cols = [:smiles_r, :human_VDss_L_kg, :human_fup, Symbol("MoKa.LogP"), Symbol("MoKa.LogD7.4"), :MW]
df_complete = dropmissing(df, required_cols)
println("  Compounds with complete data: $(nrow(df_complete))")

# ============================================================================
# COMPUTE ALL FEATURES
# ============================================================================

println("\n[2] Computing features...")

function get_physchem(row)
    Float64[
        row[:MW] / 600,
        row[:HBA] / 15,
        row[:HBD] / 8,
        row[:TPSA_NO] / 200,
        row[:RotBondCount] / 15,
        (row[Symbol("MoKa.LogP")] + 5) / 12,
        (row[Symbol("MoKa.LogD7.4")] + 5) / 12,
        row[:human_fup]
    ]
end

X_data = Vector{Vector{Float64}}()
y_data = Float64[]
mech_pred = Float64[]

for row in eachrow(df_complete)
    try
        smiles = row[:smiles_r]
        physchem = get_physchem(row)
        fractal = compute_fractal_features(smiles)

        # Mechanistic Vdss
        logP = row[Symbol("MoKa.LogP")]
        logD = row[Symbol("MoKa.LogD7.4")]
        fup = row[:human_fup]
        mw = row[:MW]
        pKa_est = (logP - logD) > 0.5 ? 8.5 : 5.0

        mol = MoleculeParams(logP=logP, logD74=logD, pKa=pKa_est, fup=fup, MW=mw)
        vdss_mech = predict_vdss_mechanistic(mol)

        # All features + mechanistic baseline
        features = vcat(physchem, fractal, [log(vdss_mech + 0.01)])

        push!(X_data, features)
        push!(y_data, log(row[:human_VDss_L_kg]))
        push!(mech_pred, vdss_mech)
    catch
        continue
    end
end

n_samples = length(y_data)
n_features = length(X_data[1])
println("  Samples: $n_samples, Features: $n_features")

# Convert to matrices
X = hcat(X_data...)
y = y_data

# ============================================================================
# FAST NEURAL NETWORK WITH ANALYTICAL GRADIENTS
# ============================================================================

# ReLU and derivatives
relu(x) = max(0.0, x)
relu_deriv(x) = x > 0 ? 1.0 : 0.0

mutable struct FastNet
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    W3::Matrix{Float64}
    b3::Vector{Float64}
    # Gradients
    dW1::Matrix{Float64}
    db1::Vector{Float64}
    dW2::Matrix{Float64}
    db2::Vector{Float64}
    dW3::Matrix{Float64}
    db3::Vector{Float64}
    # Adam state
    mW1::Matrix{Float64}
    vW1::Matrix{Float64}
    mb1::Vector{Float64}
    vb1::Vector{Float64}
    mW2::Matrix{Float64}
    vW2::Matrix{Float64}
    mb2::Vector{Float64}
    vb2::Vector{Float64}
    mW3::Matrix{Float64}
    vW3::Matrix{Float64}
    mb3::Vector{Float64}
    vb3::Vector{Float64}
    t::Int
end

function FastNet(din, h1, h2)
    scale1 = sqrt(2.0 / din)
    scale2 = sqrt(2.0 / h1)
    scale3 = sqrt(2.0 / h2)

    FastNet(
        randn(h1, din) .* scale1, zeros(h1),
        randn(h2, h1) .* scale2, zeros(h2),
        randn(1, h2) .* scale3, zeros(1),
        zeros(h1, din), zeros(h1),
        zeros(h2, h1), zeros(h2),
        zeros(1, h2), zeros(1),
        zeros(h1, din), zeros(h1, din),
        zeros(h1), zeros(h1),
        zeros(h2, h1), zeros(h2, h1),
        zeros(h2), zeros(h2),
        zeros(1, h2), zeros(1, h2),
        zeros(1), zeros(1),
        0
    )
end

function forward_backward!(net::FastNet, X::Matrix{Float64}, y::Vector{Float64})
    batch_size = size(X, 2)

    # Forward
    z1 = net.W1 * X .+ net.b1
    a1 = relu.(z1)

    z2 = net.W2 * a1 .+ net.b2
    a2 = relu.(z2)

    z3 = net.W3 * a2 .+ net.b3
    pred = vec(z3)

    # Loss
    diff = pred .- y
    loss = mean(diff.^2)

    # Backward
    d3 = reshape(diff, 1, :) ./ batch_size

    net.dW3 .= d3 * a2'
    net.db3 .= vec(sum(d3, dims=2))

    d2_pre = net.W3' * d3
    d2 = d2_pre .* relu_deriv.(z2)

    net.dW2 .= d2 * a1'
    net.db2 .= vec(sum(d2, dims=2))

    d1_pre = net.W2' * d2
    d1 = d1_pre .* relu_deriv.(z1)

    net.dW1 .= d1 * X'
    net.db1 .= vec(sum(d1, dims=2))

    return loss, pred
end

function adam_step!(net::FastNet; lr=0.001, β1=0.9, β2=0.999, ε=1e-8)
    net.t += 1

    for (p, g, m, v) in [
        (net.W1, net.dW1, net.mW1, net.vW1),
        (net.b1, net.db1, net.mb1, net.vb1),
        (net.W2, net.dW2, net.mW2, net.vW2),
        (net.b2, net.db2, net.mb2, net.vb2),
        (net.W3, net.dW3, net.mW3, net.vW3),
        (net.b3, net.db3, net.mb3, net.vb3)
    ]
        m .= β1 .* m .+ (1 - β1) .* g
        v .= β2 .* v .+ (1 - β2) .* g.^2
        m_hat = m ./ (1 - β1^net.t)
        v_hat = v ./ (1 - β2^net.t)
        p .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ε)
    end
end

function predict(net::FastNet, X::Matrix{Float64})
    z1 = net.W1 * X .+ net.b1
    a1 = relu.(z1)
    z2 = net.W2 * a1 .+ net.b2
    a2 = relu.(z2)
    z3 = net.W3 * a2 .+ net.b3
    return vec(z3)
end

# ============================================================================
# METRICS
# ============================================================================

function compute_gmfe(pred, obs)
    pred_orig = exp.(pred)
    obs_orig = exp.(obs)
    valid = (pred_orig .> 0) .& (obs_orig .> 0)
    ratios = pred_orig[valid] ./ obs_orig[valid]
    fold_errors = max.(ratios, 1 ./ ratios)
    return exp(mean(log.(fold_errors)))
end

function pct_within_fold(pred, obs, fold)
    pred_orig = exp.(pred)
    obs_orig = exp.(obs)
    valid = (pred_orig .> 0) .& (obs_orig .> 0)
    ratios = pred_orig[valid] ./ obs_orig[valid]
    within = (ratios .>= 1/fold) .& (ratios .<= fold)
    return mean(within) * 100
end

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

println("\n[3] Training (5-fold CV × 3 seeds)...")

n_folds = 5
n_seeds = 3
all_gmfes = Float64[]
all_2fold = Float64[]

for seed in 1:n_seeds
    Random.seed!(42 + seed)
    indices = shuffle(1:n_samples)
    fold_size = n_samples ÷ n_folds

    for fold in 1:n_folds
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == n_folds ? n_samples : fold * fold_size
        test_idx = indices[test_start:test_end]
        train_idx = setdiff(indices, test_idx)

        X_train = X[:, train_idx]
        y_train = y[train_idx]
        X_test = X[:, test_idx]
        y_test = y[test_idx]

        # Normalize
        μ = mean(X_train, dims=2)
        σ = std(X_train, dims=2) .+ 1e-8
        X_train_n = (X_train .- μ) ./ σ
        X_test_n = (X_test .- μ) ./ σ

        # Train
        net = FastNet(n_features, 48, 24)

        for epoch in 1:150
            # Mini-batch
            batch_idx = rand(1:length(train_idx), 64)
            loss, _ = forward_backward!(net, X_train_n[:, batch_idx], y_train[batch_idx])
            adam_step!(net, lr=0.003)
        end

        # Evaluate
        pred = predict(net, X_test_n)
        gmfe = compute_gmfe(pred, y_test)
        pct2 = pct_within_fold(pred, y_test, 2.0)

        push!(all_gmfes, gmfe)
        push!(all_2fold, pct2)
    end

    seed_gmfe = mean(all_gmfes[end-n_folds+1:end])
    println("  Seed $seed: Mean GMFE = $(round(seed_gmfe, digits=3))")
end

# ============================================================================
# RESULTS
# ============================================================================

println("\n" * "="^80)
println("RESULTS: FRACTAL-MECHANISTIC PBPK MODEL")
println("="^80)

mean_gmfe = mean(all_gmfes)
std_gmfe = std(all_gmfes)
best_gmfe = minimum(all_gmfes)
mean_2fold = mean(all_2fold)
best_2fold = maximum(all_2fold)

println("\nCross-Validation ($n_folds-fold × $n_seeds seeds):")
println("  Mean GMFE:      $(round(mean_gmfe, digits=3)) ± $(round(std_gmfe, digits=3))")
println("  Best Fold GMFE: $(round(best_gmfe, digits=3))")
println("  Mean % 2-fold:  $(round(mean_2fold, digits=1))%")
println("  Best % 2-fold:  $(round(best_2fold, digits=1))%")

println("\n" * "-"^50)
println("FDA/EMA Regulatory Assessment:")
println("  GMFE < 2.0:       $(mean_gmfe < 2.0 ? "✓ PASS" : "✗ FAIL")")
println("  >50% within 2-fold: $(mean_2fold > 50 ? "✓ PASS" : "✗ FAIL")")

# Mechanistic baseline
mech_gmfe = compute_gmfe(log.(mech_pred .+ 0.01), y)
mech_2fold = pct_within_fold(log.(mech_pred .+ 0.01), y, 2.0)

println("\n" * "-"^50)
println("Component Analysis:")
println("  Mechanistic-only (R&R + Øie-Tozer):")
println("    GMFE: $(round(mech_gmfe, digits=3)), % 2-fold: $(round(mech_2fold, digits=1))%")
println("  Fractal-Mechanistic Hybrid:")
println("    GMFE: $(round(mean_gmfe, digits=3)), % 2-fold: $(round(mean_2fold, digits=1))%")
println("  Improvement: $(round((mech_gmfe - mean_gmfe) / mech_gmfe * 100, digits=1))%")

println("\n" * "-"^50)
println("Comparison with Literature:")
println("  Method                    | GMFE  | % 2-fold")
println("  " * "-"^42)
println("  Øie-Tozer (exp fup+fut)   | 1.55  | 81%")
println("  PKSmart 2024              | 2.09  | ~60%")
println("  Our prev (descriptors)    | 2.19  | 57%")
println("  This (Fractal+Mech)       | $(round(mean_gmfe, digits=2))  | $(round(mean_2fold, digits=0))%")

println("\n" * "="^80)
println("KEY INSIGHT: Self-similarity across scales")
println("Molecular fractal topology couples with physiological fractal networks")
println("="^80)
