"""
STABLE ENSEMBLE MODEL

Problem: Single neural networks have high variance (GMFE ranges from 1.0 to 4.0).
Solution: Ensemble of diverse models to reduce variance.

Ensemble strategy:
1. Multiple seeds (reduce random initialization variance)
2. Multiple architectures (reduce architecture bias)
3. Median prediction (robust to outliers)
"""

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Random
using Printf

include("../../src/DarwinPBPK/fractal_descriptors.jl")
include("../../src/DarwinPBPK/fractional_pbpk.jl")

using .FractalDescriptors
using .FractionalPBPK

println("="^80)
println("STABLE ENSEMBLE MODEL")
println("Goal: Reproducible GMFE < 2.0 across ALL seeds")
println("="^80)

# ============================================================================
# FEATURES (using baseline - proven equivalent to fractal in ablation)
# ============================================================================

function compute_features(row)
    fup = row[:human_fup]
    logP = row[Symbol("MoKa.LogP")]
    logD = row[Symbol("MoKa.LogD7.4")]
    MW = row[:MW]
    TPSA = row[:TPSA_NO]
    HBD = Float64(row[:HBD])
    HBA = Float64(row[:HBA])
    RB = Float64(row[:RotBondCount])
    P = 10^logD

    # Basic physicochemical
    features = Float64[
        MW / 500, HBA / 12, HBD / 6, TPSA / 150, RB / 12,
        (logP + 3) / 10, (logD + 3) / 10,
        log10(fup + 0.001) / 3 + 1, fup,
        P / (P + 1),  # Membrane permeability proxy
        TPSA / MW * 10,
        (HBD + HBA) / 20,
    ]

    # Simple mechanistic estimate (Øie-Tozer inspired)
    Vp, Ve, Vr = 0.04, 0.17, 0.39
    fut_est = clamp(0.5 / (1 + 0.1 * P), 0.01, 0.99)
    vdss_est = Vp + Ve * (fup / fut_est) + Vr * (fup / fut_est)

    push!(features, log(vdss_est + 0.01))
    push!(features, fup / fut_est)
    push!(features, log(fup + 0.001) - log(fut_est + 0.001))

    return features
end

# ============================================================================
# NEURAL NETWORK
# ============================================================================

relu(x) = max(0.0, x)
leaky_relu(x) = x > 0 ? x : 0.01x  # More stable than ReLU

mutable struct Net
    layers::Vector{Tuple{Matrix{Float64}, Vector{Float64}}}
    # Adam state
    m::Vector{Tuple{Matrix{Float64}, Vector{Float64}}}
    v::Vector{Tuple{Matrix{Float64}, Vector{Float64}}}
    t::Int
end

function Net(dims::Vector{Int})
    layers = Tuple{Matrix{Float64}, Vector{Float64}}[]
    m_state = Tuple{Matrix{Float64}, Vector{Float64}}[]
    v_state = Tuple{Matrix{Float64}, Vector{Float64}}[]

    for i in 1:length(dims)-1
        # Xavier initialization
        scale = sqrt(2.0 / (dims[i] + dims[i+1]))
        W = randn(dims[i+1], dims[i]) .* scale
        b = zeros(dims[i+1])
        push!(layers, (W, b))
        push!(m_state, (zeros(size(W)), zeros(size(b))))
        push!(v_state, (zeros(size(W)), zeros(size(b))))
    end

    Net(layers, m_state, v_state, 0)
end

function forward(net::Net, X)
    a = X
    for (i, (W, b)) in enumerate(net.layers)
        z = W * a .+ b
        a = i < length(net.layers) ? leaky_relu.(z) : z
    end
    return vec(a)
end

function train_step!(net::Net, X, y; lr=0.001, λ=0.0001, clip=0.5)
    n = size(X, 2)
    L = length(net.layers)

    # Forward pass - store activations
    activations = [X]
    z_vals = []
    a = X
    for (i, (W, b)) in enumerate(net.layers)
        z = W * a .+ b
        push!(z_vals, z)
        a = i < L ? leaky_relu.(z) : z
        push!(activations, a)
    end

    pred = vec(a)

    # Backward pass
    δ = reshape((pred .- y) ./ n, 1, :)
    grads = []

    for i in L:-1:1
        W, b = net.layers[i]
        a_prev = activations[i]

        dW = δ * a_prev' .+ 2λ .* W
        db = vec(sum(δ, dims=2))
        pushfirst!(grads, (dW, db))

        if i > 1
            z = z_vals[i-1]
            δ = (W' * δ) .* (z .> 0 .+ 0.01 .* (z .<= 0))
        end
    end

    # Gradient clipping (global norm)
    total_norm = sqrt(sum(norm(g[1])^2 + norm(g[2])^2 for g in grads))
    if total_norm > clip
        scale = clip / total_norm
        grads = [(g[1] .* scale, g[2] .* scale) for g in grads]
    end

    # Adam update
    net.t += 1
    β1, β2, ε = 0.9, 0.999, 1e-8

    for i in 1:L
        dW, db = grads[i]
        W, b = net.layers[i]
        mW, mb = net.m[i]
        vW, vb = net.v[i]

        # Update moments
        mW .= β1 .* mW .+ (1-β1) .* dW
        vW .= β2 .* vW .+ (1-β2) .* dW.^2
        mb .= β1 .* mb .+ (1-β1) .* db
        vb .= β2 .* vb .+ (1-β2) .* db.^2

        # Bias correction
        mW_hat = mW ./ (1 - β1^net.t)
        vW_hat = vW ./ (1 - β2^net.t)
        mb_hat = mb ./ (1 - β1^net.t)
        vb_hat = vb ./ (1 - β2^net.t)

        # Update
        W .-= lr .* mW_hat ./ (sqrt.(vW_hat) .+ ε)
        b .-= lr .* mb_hat ./ (sqrt.(vb_hat) .+ ε)

        net.layers[i] = (W, b)
        net.m[i] = (mW, mb)
        net.v[i] = (vW, vb)
    end

    return mean((pred .- y).^2)
end

# ============================================================================
# ENSEMBLE
# ============================================================================

struct Ensemble
    nets::Vector{Net}
end

function train_ensemble(X_train, y_train, n_features;
                        n_models=20, epochs=400, lr=0.001)

    # Diverse architectures
    architectures = [
        [n_features, 64, 32, 1],
        [n_features, 48, 24, 1],
        [n_features, 32, 16, 1],
        [n_features, 64, 32, 16, 1],
        [n_features, 48, 1],
    ]

    nets = Net[]

    for i in 1:n_models
        arch = architectures[mod1(i, length(architectures))]
        net = Net(arch)

        # Train with different batch sizes and learning rates
        batch_size = [32, 64, 128][mod1(i, 3)]
        lr_scale = [0.5, 1.0, 1.5][mod1(i, 3)]

        for ep in 1:epochs
            # Learning rate decay
            current_lr = lr * lr_scale * (1.0 - ep / (epochs * 1.2))

            idx = rand(1:size(X_train, 2), batch_size)
            train_step!(net, X_train[:, idx], y_train[idx], lr=current_lr)
        end

        push!(nets, net)
    end

    Ensemble(nets)
end

function predict_ensemble(ens::Ensemble, X)
    predictions = hcat([forward(net, X) for net in ens.nets]...)
    # Use median for robustness to outliers
    return vec(median(predictions, dims=2))
end

# ============================================================================
# METRICS
# ============================================================================

function gmfe(pred, obs)
    p, o = exp.(pred), exp.(obs)
    fe = max.(p./o, o./p)
    exp(mean(log.(fe)))
end

pct_fold(pred, obs, f) = mean((exp.(pred)./exp.(obs) .>= 1/f) .&
                              (exp.(pred)./exp.(obs) .<= f)) * 100

# ============================================================================
# LOAD DATA
# ============================================================================

println("\n[1] Loading data...")
df = CSV.read("/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv", DataFrame)
df = dropmissing(df, [:smiles_r, :human_VDss_L_kg, :human_fup,
                      Symbol("MoKa.LogP"), Symbol("MoKa.LogD7.4"), :MW])

X_data, y_data = Vector{Vector{Float64}}(), Float64[]

for row in eachrow(df)
    try
        push!(X_data, compute_features(row))
        push!(y_data, log(row[:human_VDss_L_kg]))
    catch; continue; end
end

n, n_features = length(y_data), length(X_data[1])
X, y = hcat(X_data...), y_data
println("  Samples: $n, Features: $n_features")

# ============================================================================
# CROSS-VALIDATION: Ensemble vs Single Model
# ============================================================================

println("\n[2] Comparing Ensemble vs Single Model (5-fold × 10 seeds)...")

n_folds, n_seeds = 5, 10
single_results, ensemble_results = Float64[], Float64[]
single_2fold, ensemble_2fold = Float64[], Float64[]

for seed in 1:n_seeds
    Random.seed!(seed * 123)
    idx = shuffle(1:n)
    fs = n ÷ n_folds

    seed_single, seed_ensemble = Float64[], Float64[]

    for fold in 1:n_folds
        te_idx = idx[(fold-1)*fs+1 : fold==n_folds ? n : fold*fs]
        tr_idx = setdiff(idx, te_idx)

        # Normalize
        μ = mean(X[:, tr_idx], dims=2)
        σ = std(X[:, tr_idx], dims=2) .+ 1e-8
        Xtr = (X[:, tr_idx] .- μ) ./ σ
        Xte = (X[:, te_idx] .- μ) ./ σ
        ytr, yte = y[tr_idx], y[te_idx]

        # Single model
        Random.seed!(seed * 1000 + fold)
        single_net = Net([n_features, 64, 32, 1])
        for ep in 1:400
            bi = rand(1:length(tr_idx), 64)
            train_step!(single_net, Xtr[:, bi], ytr[bi])
        end
        pred_single = forward(single_net, Xte)
        g_single = gmfe(pred_single, yte)
        push!(single_results, g_single)
        push!(single_2fold, pct_fold(pred_single, yte, 2.0))
        push!(seed_single, g_single)

        # Ensemble (20 models)
        Random.seed!(seed * 2000 + fold)
        ens = train_ensemble(Xtr, ytr, n_features, n_models=20, epochs=400)
        pred_ens = predict_ensemble(ens, Xte)
        g_ens = gmfe(pred_ens, yte)
        push!(ensemble_results, g_ens)
        push!(ensemble_2fold, pct_fold(pred_ens, yte, 2.0))
        push!(seed_ensemble, g_ens)
    end

    @printf("  Seed %2d: Single=%.3f±%.3f, Ensemble=%.3f±%.3f\n",
            seed, mean(seed_single), std(seed_single),
            mean(seed_ensemble), std(seed_ensemble))
end

# ============================================================================
# RESULTS
# ============================================================================

println("\n" * "="^80)
println("RESULTS: ENSEMBLE vs SINGLE MODEL")
println("="^80)

println("\nSINGLE MODEL:")
println("  Mean GMFE: $(round(mean(single_results), digits=3)) ± $(round(std(single_results), digits=3))")
println("  Best fold: $(round(minimum(single_results), digits=3))")
println("  Worst fold: $(round(maximum(single_results), digits=3))")
println("  Mean % 2-fold: $(round(mean(single_2fold), digits=1))%")

println("\nENSEMBLE (20 models):")
println("  Mean GMFE: $(round(mean(ensemble_results), digits=3)) ± $(round(std(ensemble_results), digits=3))")
println("  Best fold: $(round(minimum(ensemble_results), digits=3))")
println("  Worst fold: $(round(maximum(ensemble_results), digits=3))")
println("  Mean % 2-fold: $(round(mean(ensemble_2fold), digits=1))%")

# Variance reduction
var_reduction = 1 - std(ensemble_results) / std(single_results)
println("\nVariance reduction: $(round(var_reduction * 100, digits=1))%")

# How many folds achieve GMFE < 2.0?
single_pass = sum(single_results .< 2.0) / length(single_results) * 100
ensemble_pass = sum(ensemble_results .< 2.0) / length(ensemble_results) * 100
println("\n% folds with GMFE < 2.0:")
println("  Single: $(round(single_pass, digits=1))%")
println("  Ensemble: $(round(ensemble_pass, digits=1))%")

println("\n" * "="^80)
if ensemble_pass > 80
    println("✓ ENSEMBLE ACHIEVES REPRODUCIBLE GMFE < 2.0 ($(round(ensemble_pass,digits=0))% of folds)")
else
    println("○ ENSEMBLE IMPROVES BUT NOT FULLY REPRODUCIBLE ($(round(ensemble_pass,digits=0))% < 2.0)")
end
println("="^80)

# Statistical comparison
using SpecialFunctions: erf
diff = single_results .- ensemble_results
t = mean(diff) / (std(diff) / sqrt(length(diff)))
p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))

println("\nPaired t-test (Single vs Ensemble):")
println("  Mean improvement: $(round(mean(diff), digits=3))")
println("  t = $(round(t, digits=3)), p = $(round(p, digits=4))")
if p < 0.05 && mean(diff) > 0
    println("  ✓ Ensemble SIGNIFICANTLY better than single model")
end
