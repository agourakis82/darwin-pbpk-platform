"""
ABLATION STUDY v2: With Adam Optimizer

Previous ablation used simple SGD - both conditions performed poorly (GMFE ~2.65).
This version uses Adam optimizer which achieved GMFE ~1.8 in earlier experiments.

Question: With proper optimization, do fractal features help?
"""

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Random
using Printf
using SpecialFunctions: erf

include("../../src/DarwinPBPK/fractal_descriptors.jl")
include("../../src/DarwinPBPK/fractional_pbpk.jl")

using .FractalDescriptors
using .FractionalPBPK

println("="^80)
println("ABLATION STUDY v2: With Adam Optimizer")
println("="^80)

# ============================================================================
# FEATURE COMPUTATION (same as before)
# ============================================================================

function compute_baseline_features(row)
    fup = row[:human_fup]
    logP = row[Symbol("MoKa.LogP")]
    logD = row[Symbol("MoKa.LogD7.4")]
    MW = row[:MW]
    TPSA = row[:TPSA_NO]
    HBD = Float64(row[:HBD])
    HBA = Float64(row[:HBA])
    RB = Float64(row[:RotBondCount])

    Float64[
        MW / 500, HBA / 12, HBD / 6, TPSA / 150, RB / 12,
        (logP + 3) / 10, (logD + 3) / 10,
        log10(fup + 0.001) / 3 + 1, fup,
        10^logD / (10^logD + 1),
        TPSA / MW * 10,
        (HBD + HBA) / 20,
    ]
end

function compute_fractal_features(row, frac_desc)
    fup = row[:human_fup]
    logD = row[Symbol("MoKa.LogD7.4")]
    MW = row[:MW]
    TPSA = row[:TPSA_NO]
    HBD = Float64(row[:HBD])
    HBA = Float64(row[:HBA])
    RB = Float64(row[:RotBondCount])
    P = 10^logD

    d_f_mol = molecular_fractal_dim(MW, RB, TPSA, HBD, HBA)
    d_f_tissue = 2.70
    α_tissue = 0.80
    d_s = 4/3

    coupling = exp(-abs(d_f_mol - d_f_tissue)^2 / 0.09)
    fut = clamp(1 / (1 + 0.1 * P) * (d_f_tissue / 3)^α_tissue, 0.001, 0.99)

    Float64[
        d_f_mol / 3,
        frac_desc["fractal_dim"] / 3,
        frac_desc["topological_entropy"] / 2,
        frac_desc["branching_complexity"],
        frac_desc["fragment_self_similarity"],
        coupling,
        (d_f_mol - d_f_tissue) / 0.5,
        (d_f_mol / d_f_tissue)^α_tissue,
        d_s / 2,
        (fup / fut)^(d_s/2) * coupling,
        log(fut + 0.01) / 3 + 1,
        log(0.04 + 0.17 * (fup/fut)^(d_s/2) * coupling + 0.39 * (fup/fut) * coupling + 0.01),
    ]
end

# ============================================================================
# NEURAL NETWORK WITH ADAM
# ============================================================================

relu(x) = max(0.0, x)

mutable struct AdamNet
    W1::Matrix{Float64}; b1::Vector{Float64}
    W2::Matrix{Float64}; b2::Vector{Float64}
    W3::Matrix{Float64}; b3::Vector{Float64}
    # Adam state
    mW1::Matrix{Float64}; vW1::Matrix{Float64}; mb1::Vector{Float64}; vb1::Vector{Float64}
    mW2::Matrix{Float64}; vW2::Matrix{Float64}; mb2::Vector{Float64}; vb2::Vector{Float64}
    mW3::Matrix{Float64}; vW3::Matrix{Float64}; mb3::Vector{Float64}; vb3::Vector{Float64}
    t::Int
end

function AdamNet(din, h1=64, h2=32)
    AdamNet(
        randn(h1, din) .* sqrt(2/din), zeros(h1),
        randn(h2, h1) .* sqrt(2/h1), zeros(h2),
        randn(1, h2) .* sqrt(1/h2), zeros(1),
        zeros(h1, din), zeros(h1, din), zeros(h1), zeros(h1),
        zeros(h2, h1), zeros(h2, h1), zeros(h2), zeros(h2),
        zeros(1, h2), zeros(1, h2), zeros(1), zeros(1),
        0
    )
end

function forward(net::AdamNet, X)
    a1 = relu.(net.W1 * X .+ net.b1)
    a2 = relu.(net.W2 * a1 .+ net.b2)
    return vec(net.W3 * a2 .+ net.b3)
end

function train_step!(net::AdamNet, X, y; lr=0.002, λ=0.0003, clip=1.0)
    n = size(X, 2)

    # Forward
    z1 = net.W1 * X .+ net.b1
    a1 = relu.(z1)
    z2 = net.W2 * a1 .+ net.b2
    a2 = relu.(z2)
    pred = vec(net.W3 * a2 .+ net.b3)

    # Backward
    diff = (pred .- y) ./ n

    d3 = reshape(diff, 1, :)
    dW3 = d3 * a2' .+ 2λ .* net.W3
    db3 = vec(sum(d3, dims=2))

    d2 = (net.W3' * d3) .* (z2 .> 0)
    dW2 = d2 * a1' .+ 2λ .* net.W2
    db2 = vec(sum(d2, dims=2))

    d1 = (net.W2' * d2) .* (z1 .> 0)
    dW1 = d1 * X' .+ 2λ .* net.W1
    db1 = vec(sum(d1, dims=2))

    # Gradient clipping
    for g in [dW1, db1, dW2, db2, dW3, db3]
        gn = norm(g)
        gn > clip && (g .*= clip / gn)
    end

    # Adam update
    net.t += 1
    β1, β2, ε = 0.9, 0.999, 1e-8

    for (W, dW, m, v) in [(net.W1, dW1, net.mW1, net.vW1),
                          (net.W2, dW2, net.mW2, net.vW2),
                          (net.W3, dW3, net.mW3, net.vW3)]
        m .= β1 .* m .+ (1-β1) .* dW
        v .= β2 .* v .+ (1-β2) .* dW.^2
        m_hat = m ./ (1 - β1^net.t)
        v_hat = v ./ (1 - β2^net.t)
        W .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ε)
    end

    for (b, db, m, v) in [(net.b1, db1, net.mb1, net.vb1),
                          (net.b2, db2, net.mb2, net.vb2),
                          (net.b3, db3, net.mb3, net.vb3)]
        m .= β1 .* m .+ (1-β1) .* db
        v .= β2 .* v .+ (1-β2) .* db.^2
        m_hat = m ./ (1 - β1^net.t)
        v_hat = v ./ (1 - β2^net.t)
        b .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ε)
    end

    return mean((pred .- y).^2)
end

# ============================================================================
# METRICS
# ============================================================================

function gmfe(pred, obs)
    p, o = exp.(pred), exp.(obs)
    fe = max.(p./o, o./p)
    return exp(mean(log.(fe)))
end

function pct_within_fold(pred, obs, fold)
    p, o = exp.(pred), exp.(obs)
    ratio = p ./ o
    mean((ratio .>= 1/fold) .& (ratio .<= fold)) * 100
end

# ============================================================================
# LOAD DATA
# ============================================================================

println("\n[1] Loading data...")
df = CSV.read("/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv", DataFrame)
df = dropmissing(df, [:smiles_r, :human_VDss_L_kg, :human_fup,
                      Symbol("MoKa.LogP"), Symbol("MoKa.LogD7.4"), :MW])

baseline_X, fractal_X, y_all = Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), Float64[]

for row in eachrow(df)
    try
        base = compute_baseline_features(row)
        frac_desc = compute_all_fractal_descriptors(row[:smiles_r])
        frac = compute_fractal_features(row, frac_desc)
        push!(baseline_X, base)
        push!(fractal_X, vcat(base, frac))
        push!(y_all, log(row[:human_VDss_L_kg]))
    catch; continue; end
end

n = length(y_all)
n_base, n_frac = length(baseline_X[1]), length(fractal_X[1])
X_base, X_frac, y = hcat(baseline_X...), hcat(fractal_X...), y_all

println("  Samples: $n, Baseline: $n_base feat, Fractal: $n_frac feat")

# ============================================================================
# ABLATION WITH ADAM
# ============================================================================

println("\n[2] Ablation study with Adam optimizer...")
println("    5-fold CV × 10 seeds, 500 epochs each")

n_folds, n_seeds = 5, 10
base_results, frac_results = Float64[], Float64[]
base_2fold, frac_2fold = Float64[], Float64[]

for seed in 1:n_seeds
    Random.seed!(seed * 42)
    idx = shuffle(1:n)
    fs = n ÷ n_folds

    for fold in 1:n_folds
        te_idx = idx[(fold-1)*fs+1 : fold==n_folds ? n : fold*fs]
        tr_idx = setdiff(idx, te_idx)

        # Normalize
        function norm_data(X, tr, te)
            μ = mean(X[:, tr], dims=2)
            σ = std(X[:, tr], dims=2) .+ 1e-8
            (X[:, tr] .- μ) ./ σ, (X[:, te] .- μ) ./ σ
        end

        ytr, yte = y[tr_idx], y[te_idx]

        # BASELINE with Adam
        Random.seed!(seed * 1000 + fold)
        Xtr_b, Xte_b = norm_data(X_base, tr_idx, te_idx)
        net_b = AdamNet(n_base, 64, 32)
        for ep in 1:500
            bi = rand(1:length(tr_idx), 64)
            train_step!(net_b, Xtr_b[:, bi], ytr[bi])
        end
        pred_b = forward(net_b, Xte_b)
        push!(base_results, gmfe(pred_b, yte))
        push!(base_2fold, pct_within_fold(pred_b, yte, 2.0))

        # FRACTAL with Adam (same init seed)
        Random.seed!(seed * 1000 + fold)
        Xtr_f, Xte_f = norm_data(X_frac, tr_idx, te_idx)
        net_f = AdamNet(n_frac, 64, 32)
        for ep in 1:500
            bi = rand(1:length(tr_idx), 64)
            train_step!(net_f, Xtr_f[:, bi], ytr[bi])
        end
        pred_f = forward(net_f, Xte_f)
        push!(frac_results, gmfe(pred_f, yte))
        push!(frac_2fold, pct_within_fold(pred_f, yte, 2.0))
    end

    bm = mean(base_results[end-n_folds+1:end])
    fm = mean(frac_results[end-n_folds+1:end])
    @printf("  Seed %2d: Base=%.3f, Frac=%.3f, Δ=%+.3f\n", seed, bm, fm, bm-fm)
end

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

println("\n" * "="^80)
println("RESULTS WITH ADAM OPTIMIZER")
println("="^80)

diff = base_results .- frac_results
n_pairs = length(diff)
mean_diff = mean(diff)
se_diff = std(diff) / sqrt(n_pairs)
t_stat = mean_diff / se_diff

function p_val(t, df)
    abs(t) > 4 ? 2 * exp(-0.5 * t^2) / sqrt(2π) / abs(t) :
                 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
end

p = p_val(t_stat, n_pairs - 1)

println("\nGMFE Comparison:")
println("  Baseline: $(round(mean(base_results), digits=3)) ± $(round(std(base_results), digits=3))")
println("  Fractal:  $(round(mean(frac_results), digits=3)) ± $(round(std(frac_results), digits=3))")
println("  Best baseline fold: $(round(minimum(base_results), digits=3))")
println("  Best fractal fold:  $(round(minimum(frac_results), digits=3))")
println("\nPaired test (H0: no difference):")
println("  Mean Δ: $(round(mean_diff, digits=4)) (positive = fractal better)")
println("  t = $(round(t_stat, digits=3)), p = $(round(p, digits=4))")

diff_2f = frac_2fold .- base_2fold
t_2f = mean(diff_2f) / (std(diff_2f) / sqrt(n_pairs))
p_2f = p_val(t_2f, n_pairs - 1)

println("\n% Within 2-fold:")
println("  Baseline: $(round(mean(base_2fold), digits=1))%")
println("  Fractal:  $(round(mean(frac_2fold), digits=1))%")
println("  t = $(round(t_2f, digits=3)), p = $(round(p_2f, digits=4))")

println("\n" * "="^80)
if p < 0.05 && mean_diff > 0
    println("✓ FRACTAL FEATURES SIGNIFICANTLY IMPROVE GMFE (p=$(round(p,digits=4)))")
elseif p < 0.05 && mean_diff < 0
    println("✗ FRACTAL FEATURES SIGNIFICANTLY WORSEN GMFE (p=$(round(p,digits=4)))")
else
    println("○ NO SIGNIFICANT DIFFERENCE (p=$(round(p,digits=4)))")
end
println("="^80)
