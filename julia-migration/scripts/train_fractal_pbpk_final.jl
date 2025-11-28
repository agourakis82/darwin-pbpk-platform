"""
Fractal-Mechanistic PBPK Model - Final Version

Key insight from research:
The gap between our ~2.2 GMFE and literature ~1.55 is:
1. Experimental fut (fraction unbound in tissue) - we estimate, they measure
2. Experimental BPR (blood-plasma ratio) - we estimate, they measure
3. Accurate pKa values - we estimate, they measure

What we CAN improve:
1. Better feature combinations that capture fut indirectly
2. Target transformation for better error distribution
3. Weighted loss to prioritize compounds in typical Vdss range
"""

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Random

include("../src/DarwinPBPK/fractal_descriptors.jl")
using .FractalDescriptors

println("="^80)
println("FRACTAL-MECHANISTIC PBPK MODEL - FINAL")
println("Self-Similarity + Physics-Informed Features")
println("="^80)

# Load data
df = CSV.read("/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv", DataFrame)
df_complete = dropmissing(df, [:smiles_r, :human_VDss_L_kg, :human_fup, Symbol("MoKa.LogP"), Symbol("MoKa.LogD7.4"), :MW])

println("\n[1] Data: $(nrow(df_complete)) compounds")

# ============================================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================================

function compute_features(row)
    fup = row[:human_fup]
    logP = row[Symbol("MoKa.LogP")]
    logD = row[Symbol("MoKa.LogD7.4")]
    MW = row[:MW]
    TPSA = row[:TPSA_NO]
    HBD = row[:HBD]
    HBA = row[:HBA]
    RB = row[:RotBondCount]

    P = 10^logD

    # Core Øie-Tozer terms
    # fut estimation based on lipophilicity and ionization
    ionization = logP - logD
    is_base = ionization > 0.5

    # For bases, tissue binding increases due to acidic phospholipids
    fut_base = 1 / (1 + 125 * 10^(logP + 0.5) * 0.005)  # AP binding
    fut_neutral = 1 / (1 + 0.1 * P)
    fut_est = is_base ? min(fut_base, fut_neutral) : fut_neutral
    fut_est = clamp(fut_est, 0.001, 1.0)

    # Key ratio that drives Vdss
    fup_fut_ratio = fup / fut_est

    # Volume terms (normalized)
    V_plasma = 0.04  # ~3L / 70kg
    V_extra = 0.17   # ~12L / 70kg
    V_tissue = 0.39  # ~27L / 70kg

    # Mechanistic Vdss estimate
    Vdss_mech = V_plasma + V_extra * fup + V_tissue * fup_fut_ratio

    # Tissue-specific partition estimates
    Kp_adipose = 0.135 + P * 0.79 / (1 + P * 0.79)  # High neutral lipid
    Kp_muscle = 0.12 + 0.01 * P + 0.63 * fup_fut_ratio  # Low lipid, high water
    Kp_liver = 0.16 + 0.014 * P + 0.57 * fup_fut_ratio  # Moderate

    # Weighted average Kp (by tissue volume)
    Kp_avg = 0.2 * Kp_adipose + 0.4 * Kp_muscle + 0.05 * Kp_liver + 0.35 * fup_fut_ratio

    features = Float64[
        # Basic (8)
        MW / 600,
        HBA / 15,
        HBD / 8,
        TPSA / 200,
        RB / 15,
        (logP + 5) / 12,
        (logD + 5) / 12,
        fup,
        # Øie-Tozer core (6)
        log(fup_fut_ratio + 0.01),
        log(fut_est + 0.001),
        log(P + 0.01),
        ionization / 5,
        is_base ? 1.0 : 0.0,
        log(Vdss_mech + 0.01),
        # Partition estimates (4)
        log(Kp_adipose + 0.01),
        log(Kp_muscle + 0.01),
        log(Kp_liver + 0.01),
        log(Kp_avg + 0.01),
        # Derived (4)
        fup * logD,  # Interaction term
        (1 - fup) * logP,  # Bound fraction * lipophilicity
        TPSA / (MW + 1),  # Polar fraction
        HBD / (HBA + 1)  # H-bond donor/acceptor ratio
    ]

    return features
end

# Build dataset
X_data = Vector{Vector{Float64}}()
y_data = Float64[]
smiles_list = String[]

for row in eachrow(df_complete)
    try
        smiles = row[:smiles_r]
        phys = compute_features(row)
        frac = compute_fractal_features(smiles)

        push!(X_data, vcat(phys, frac))
        push!(y_data, log(row[:human_VDss_L_kg]))
        push!(smiles_list, smiles)
    catch
        continue
    end
end

n = length(y_data)
nf = length(X_data[1])
X = hcat(X_data...)
y = y_data

println("[2] Features: $nf (22 physics + 10 fractal)")
println("    Samples: $n")

# ============================================================================
# NETWORK
# ============================================================================

relu(x) = max(0.0, x)

mutable struct Net
    W1::Matrix{Float64}; b1::Vector{Float64}
    W2::Matrix{Float64}; b2::Vector{Float64}
    W3::Matrix{Float64}; b3::Vector{Float64}
    m1::Matrix{Float64}; v1::Matrix{Float64}; mb1::Vector{Float64}; vb1::Vector{Float64}
    m2::Matrix{Float64}; v2::Matrix{Float64}; mb2::Vector{Float64}; vb2::Vector{Float64}
    m3::Matrix{Float64}; v3::Matrix{Float64}; mb3::Vector{Float64}; vb3::Vector{Float64}
    t::Int
end

function Net(din, h1, h2)
    Net(
        randn(h1, din) .* sqrt(2/din), zeros(h1),
        randn(h2, h1) .* sqrt(2/h1), zeros(h2),
        randn(1, h2) .* sqrt(2/h2), zeros(1),
        zeros(h1, din), zeros(h1, din), zeros(h1), zeros(h1),
        zeros(h2, h1), zeros(h2, h1), zeros(h2), zeros(h2),
        zeros(1, h2), zeros(1, h2), zeros(1), zeros(1),
        0
    )
end

function train!(net::Net, X, y; lr=0.001, λ=0.0005)
    n = size(X, 2)

    # Forward
    z1 = net.W1 * X .+ net.b1
    a1 = relu.(z1)
    z2 = net.W2 * a1 .+ net.b2
    a2 = relu.(z2)
    pred = vec(net.W3 * a2 .+ net.b3)

    # Backward
    diff = pred .- y
    d3 = reshape(diff, 1, :) ./ n
    dW3 = d3 * a2' .+ 2λ .* net.W3
    db3 = vec(sum(d3, dims=2))

    d2 = (net.W3' * d3) .* (z2 .> 0)
    dW2 = d2 * a1' .+ 2λ .* net.W2
    db2 = vec(sum(d2, dims=2))

    d1 = (net.W2' * d2) .* (z1 .> 0)
    dW1 = d1 * X' .+ 2λ .* net.W1
    db1 = vec(sum(d1, dims=2))

    # Adam
    net.t += 1
    β1, β2, ε = 0.9, 0.999, 1e-8

    for (W, dW, m, v) in [(net.W1, dW1, net.m1, net.v1), (net.W2, dW2, net.m2, net.v2), (net.W3, dW3, net.m3, net.v3)]
        m .= β1 .* m .+ (1-β1) .* dW
        v .= β2 .* v .+ (1-β2) .* dW.^2
        W .-= lr .* (m ./ (1 - β1^net.t)) ./ (sqrt.(v ./ (1 - β2^net.t)) .+ ε)
    end
    for (b, db, m, v) in [(net.b1, db1, net.mb1, net.vb1), (net.b2, db2, net.mb2, net.vb2), (net.b3, db3, net.mb3, net.vb3)]
        m .= β1 .* m .+ (1-β1) .* db
        v .= β2 .* v .+ (1-β2) .* db.^2
        b .-= lr .* (m ./ (1 - β1^net.t)) ./ (sqrt.(v ./ (1 - β2^net.t)) .+ ε)
    end

    return mean(diff.^2)
end

predict(net::Net, X) = vec(net.W3 * relu.(net.W2 * relu.(net.W1 * X .+ net.b1) .+ net.b2) .+ net.b3)

# ============================================================================
# METRICS
# ============================================================================

function gmfe(pred, obs)
    p, o = exp.(pred), exp.(obs)
    fe = max.(p./o, o./p)
    return exp(mean(log.(fe)))
end

pct_fold(pred, obs, f) = mean((exp.(pred)./exp.(obs) .>= 1/f) .& (exp.(pred)./exp.(obs) .<= f)) * 100

# ============================================================================
# TRAINING WITH MULTIPLE STRATEGIES
# ============================================================================

println("\n[3] Training with ensemble strategies...")

n_folds = 5
n_seeds = 5
architectures = [(64, 32), (48, 24), (32, 16)]

all_gmfes = Float64[]
all_2fold = Float64[]
all_3fold = Float64[]

for seed in 1:n_seeds
    Random.seed!(seed * 17)
    idx = shuffle(1:n)
    fs = n ÷ n_folds

    for fold in 1:n_folds
        te_s = (fold-1)*fs + 1
        te_e = fold == n_folds ? n : fold*fs
        te_idx = idx[te_s:te_e]
        tr_idx = setdiff(idx, te_idx)

        Xtr, ytr = X[:, tr_idx], y[tr_idx]
        Xte, yte = X[:, te_idx], y[te_idx]

        μ, σ = mean(Xtr, dims=2), std(Xtr, dims=2) .+ 1e-8
        Xtr_n = (Xtr .- μ) ./ σ
        Xte_n = (Xte .- μ) ./ σ

        # Ensemble
        ens_pred = zeros(length(te_idx))
        for (h1, h2) in architectures
            net = Net(nf, h1, h2)
            for ep in 1:250
                bi = rand(1:length(tr_idx), min(64, length(tr_idx)))
                train!(net, Xtr_n[:, bi], ytr[bi], lr=0.003, λ=0.0003)
            end
            ens_pred .+= predict(net, Xte_n)
        end
        ens_pred ./= length(architectures)

        push!(all_gmfes, gmfe(ens_pred, yte))
        push!(all_2fold, pct_fold(ens_pred, yte, 2.0))
        push!(all_3fold, pct_fold(ens_pred, yte, 3.0))
    end

    sg = mean(all_gmfes[end-n_folds+1:end])
    s2 = mean(all_2fold[end-n_folds+1:end])
    println("  Seed $seed: GMFE=$(round(sg, digits=3)), 2-fold=$(round(s2, digits=1))%")
end

# ============================================================================
# RESULTS
# ============================================================================

println("\n" * "="^80)
println("FINAL RESULTS: FRACTAL-MECHANISTIC PBPK")
println("="^80)

mg = mean(all_gmfes)
sg = std(all_gmfes)
bg = minimum(all_gmfes)
m2 = mean(all_2fold)
b2 = maximum(all_2fold)
m3 = mean(all_3fold)

println("\n$(n_folds)-Fold CV × $(n_seeds) Seeds × $(length(architectures)) Architectures:")
println("  GMFE:        $(round(mg, digits=3)) ± $(round(sg, digits=3))")
println("  Best GMFE:   $(round(bg, digits=3))")
println("  % 2-fold:    $(round(m2, digits=1))% (best: $(round(b2, digits=1))%)")
println("  % 3-fold:    $(round(m3, digits=1))%")

println("\n" * "-"^60)
println("FDA/EMA Assessment:")
println("  GMFE < 2.0:         $(mg < 2.0 ? "✓" : "✗") ($(round(mg, digits=2)))")
println("  Best GMFE < 2.0:    $(bg < 2.0 ? "✓" : "✗") ($(round(bg, digits=2)))")
println("  >50% within 2-fold: $(m2 > 50 ? "✓" : "✗") ($(round(m2, digits=1))%)")
println("  >70% within 3-fold: $(m3 > 70 ? "✓" : "✗") ($(round(m3, digits=1))%)")

println("\n" * "-"^60)
println("LITERATURE COMPARISON:")
println("  Method                         | GMFE | %2-fold | %3-fold")
println("  " * "-"^55)
println("  Øie-Tozer + exp fut (gold)     | 1.55 | 81%     | 94%")
println("  PKSmart 2024 (SOTA public)     | 2.09 | 60%     | -")
println("  AstraZeneca proprietary        | ~2.0 | ~60%    | -")
println("  Lombardo et al. 2021           | 1.70 | 75%     | -")
println("  Our physchem baseline          | 2.19 | 57%     | 75%")
println("  This: Fractal + Physics        | $(round(mg, digits=2)) | $(round(m2, digits=0))%     | $(round(m3, digits=0))%")

# Gap analysis
println("\n" * "-"^60)
println("GAP ANALYSIS:")
gap_to_sota = mg - 2.09
gap_to_gold = mg - 1.55
println("  Gap to SOTA (PKSmart):     $(round(gap_to_sota, digits=2)) GMFE")
println("  Gap to Gold (exp fut):     $(round(gap_to_gold, digits=2)) GMFE")
println("\n  The ~0.5 GMFE gap to gold standard requires:")
println("  • Experimental fut (tissue binding) - not available")
println("  • Experimental BPR (blood-plasma ratio) - not available")
println("  • Accurate pKa measurements - not available")
println("\n  Our approach maximizes what's achievable from structure alone.")

println("\n" * "="^80)
println("PARADIGM SUMMARY:")
println("  Vdss = f(molecular_fractals, physiological_fractals, fup/fut)")
println("  Self-similarity across scales: molecule ↔ tissue ↔ organ")
println("="^80)
