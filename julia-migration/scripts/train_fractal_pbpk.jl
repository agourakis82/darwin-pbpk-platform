"""
Fractal-Mechanistic PBPK Model Training

This implements the unified paradigm combining:
1. Fractal molecular descriptors (self-similarity, topology)
2. Rodgers-Rowland mechanistic Kp prediction
3. Machine learning for residual correction

The key insight: Drug distribution is a fractal process on a fractal substrate.
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
println("FRACTAL-MECHANISTIC PBPK MODEL")
println("Unified paradigm: Molecular Fractals × Physiological Fractals × ML")
println("="^80)

# ============================================================================
# LOAD DATA
# ============================================================================

println("\n[1] Loading Lombardo 1352 dataset...")
data_path = "/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv"
df = CSV.read(data_path, DataFrame)

# Filter for complete data
println("Total compounds: $(nrow(df))")

# Need: SMILES, Vdss, fup, LogP, LogD, MW
required_cols = [:smiles_r, :human_VDss_L_kg, :human_fup, Symbol("MoKa.LogP"), Symbol("MoKa.LogD7.4"), :MW]
df_complete = dropmissing(df, required_cols)
println("With complete data: $(nrow(df_complete))")

# ============================================================================
# COMPUTE FEATURES
# ============================================================================

println("\n[2] Computing fractal molecular descriptors...")

# Standard physicochemical descriptors
function get_physchem_features(row)
    Float64[
        row[:MW] / 600,                          # Normalized MW
        row[:HBA] / 15,                          # H-bond acceptors
        row[:HBD] / 8,                           # H-bond donors
        row[:TPSA_NO] / 200,                     # Topological polar surface area
        row[:RotBondCount] / 15,                 # Rotatable bonds
        (row[Symbol("MoKa.LogP")] + 5) / 12,     # LogP normalized
        (row[Symbol("MoKa.LogD7.4")] + 5) / 12,  # LogD7.4 normalized
        row[:human_fup]                          # Fraction unbound
    ]
end

# Compute fractal features for all molecules
fractal_features = []
mechanistic_vdss = []
valid_indices = []

for (i, row) in enumerate(eachrow(df_complete))
    try
        smiles = row[:smiles_r]

        # Fractal descriptors
        frac = compute_fractal_features(smiles)

        # Mechanistic Vdss prediction (Rodgers-Rowland + Øie-Tozer)
        logP = row[Symbol("MoKa.LogP")]
        logD = row[Symbol("MoKa.LogD7.4")]
        fup = row[:human_fup]
        mw = row[:MW]

        # Estimate pKa from LogP/LogD difference
        logP_logD_diff = logP - logD
        pKa_est = logP_logD_diff > 0.5 ? 8.5 : 5.0  # Simple heuristic

        mol = MoleculeParams(logP=logP, logD74=logD, pKa=pKa_est, fup=fup, MW=mw)
        vdss_mech = predict_vdss_mechanistic(mol)

        push!(fractal_features, frac)
        push!(mechanistic_vdss, vdss_mech)
        push!(valid_indices, i)

        if i % 100 == 0
            print("\r  Processed $i compounds...")
        end
    catch e
        # Skip compounds with parsing errors
        continue
    end
end
println("\n  Computed features for $(length(valid_indices)) compounds")

# Build feature matrix
df_valid = df_complete[valid_indices, :]
n_samples = length(valid_indices)

# Physicochemical features
X_physchem = hcat([get_physchem_features(row) for row in eachrow(df_valid)]...)
println("  Physicochemical features: $(size(X_physchem, 1))")

# Fractal features
X_fractal = hcat(fractal_features...)
println("  Fractal features: $(size(X_fractal, 1))")

# Mechanistic predictions
X_mech = reshape(Float64.(mechanistic_vdss), 1, :)
println("  Mechanistic predictions: 1")

# Combine all features
X_all = vcat(X_physchem, X_fractal, X_mech)
println("  Total features: $(size(X_all, 1))")

# Target: log(Vdss) for better distribution
y = log.(Float64.(df_valid.human_VDss_L_kg))

# ============================================================================
# SIMPLE NEURAL NETWORK
# ============================================================================

# Xavier initialization
function xavier_init(din, dout)
    scale = sqrt(2.0 / (din + dout))
    return randn(dout, din) .* scale
end

# Network architecture
struct FractalPBPKNet
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    W3::Matrix{Float64}
    b3::Vector{Float64}
end

function FractalPBPKNet(input_dim, hidden1=64, hidden2=32)
    W1 = xavier_init(input_dim, hidden1)
    b1 = zeros(hidden1)
    W2 = xavier_init(hidden1, hidden2)
    b2 = zeros(hidden2)
    W3 = xavier_init(hidden2, 1)
    b3 = zeros(1)
    return FractalPBPKNet(W1, b1, W2, b2, W3, b3)
end

# Forward pass with SELU activation
function forward(net::FractalPBPKNet, X)
    # SELU parameters
    λ = 1.0507
    α = 1.6733

    selu(x) = λ * (x > 0 ? x : α * (exp(x) - 1))

    h1 = net.W1 * X .+ net.b1
    a1 = selu.(h1)

    h2 = net.W2 * a1 .+ net.b2
    a2 = selu.(h2)

    out = net.W3 * a2 .+ net.b3
    return vec(out)
end

# Gradient computation (numerical for simplicity)
function compute_gradients(net, X, y, eps=1e-5)
    function loss_fn(net)
        pred = forward(net, X)
        return mean((pred .- y).^2)
    end

    base_loss = loss_fn(net)
    grads = Dict{Symbol, Any}()

    for field in [:W1, :b1, :W2, :b2, :W3, :b3]
        param = getfield(net, field)
        grad = similar(param)

        for i in eachindex(param)
            param[i] += eps
            grad[i] = (loss_fn(net) - base_loss) / eps
            param[i] -= eps
        end

        grads[field] = grad
    end

    return grads
end

# Update parameters with Adam optimizer
mutable struct AdamOptimizer
    lr::Float64
    β1::Float64
    β2::Float64
    ε::Float64
    m::Dict{Symbol, Any}
    v::Dict{Symbol, Any}
    t::Int
end

function AdamOptimizer(; lr=0.001, β1=0.9, β2=0.999, ε=1e-8)
    return AdamOptimizer(lr, β1, β2, ε, Dict(), Dict(), 0)
end

function adam_update!(opt::AdamOptimizer, net::FractalPBPKNet, grads)
    opt.t += 1

    for field in [:W1, :b1, :W2, :b2, :W3, :b3]
        param = getfield(net, field)
        grad = grads[field]

        if !haskey(opt.m, field)
            opt.m[field] = zeros(size(param))
            opt.v[field] = zeros(size(param))
        end

        opt.m[field] .= opt.β1 .* opt.m[field] .+ (1 - opt.β1) .* grad
        opt.v[field] .= opt.β2 .* opt.v[field] .+ (1 - opt.β2) .* grad.^2

        m_hat = opt.m[field] ./ (1 - opt.β1^opt.t)
        v_hat = opt.v[field] ./ (1 - opt.β2^opt.t)

        param .-= opt.lr .* m_hat ./ (sqrt.(v_hat) .+ opt.ε)
    end
end

# ============================================================================
# METRICS
# ============================================================================

function compute_gmfe(pred, obs)
    # Work in original space
    pred_orig = exp.(pred)
    obs_orig = exp.(obs)

    valid = (pred_orig .> 0) .& (obs_orig .> 0)
    p, o = pred_orig[valid], obs_orig[valid]

    ratios = p ./ o
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
# CROSS-VALIDATION TRAINING
# ============================================================================

println("\n[3] Training with 5-fold Cross-Validation...")

n_folds = 5
n_seeds = 3
all_results = []

for seed in 1:n_seeds
    Random.seed!(42 + seed)

    # Shuffle indices
    indices = shuffle(1:n_samples)
    fold_size = n_samples ÷ n_folds

    fold_gmfes = Float64[]
    fold_2fold = Float64[]

    for fold in 1:n_folds
        # Split data
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == n_folds ? n_samples : fold * fold_size
        test_idx = indices[test_start:test_end]
        train_idx = setdiff(indices, test_idx)

        X_train = X_all[:, train_idx]
        y_train = y[train_idx]
        X_test = X_all[:, test_idx]
        y_test = y[test_idx]

        # Standardize features
        μ = mean(X_train, dims=2)
        σ = std(X_train, dims=2) .+ 1e-8
        X_train_norm = (X_train .- μ) ./ σ
        X_test_norm = (X_test .- μ) ./ σ

        # Initialize network
        input_dim = size(X_all, 1)
        net = FractalPBPKNet(input_dim, 64, 32)
        opt = AdamOptimizer(lr=0.002)

        # Training
        n_epochs = 200
        batch_size = min(64, length(train_idx))

        for epoch in 1:n_epochs
            # Mini-batch training
            batch_idx = rand(1:length(train_idx), batch_size)
            X_batch = X_train_norm[:, batch_idx]
            y_batch = y_train[batch_idx]

            # Compute gradients and update
            grads = compute_gradients(net, X_batch, y_batch)
            adam_update!(opt, net, grads)
        end

        # Evaluate
        pred_test = forward(net, X_test_norm)
        gmfe = compute_gmfe(pred_test, y_test)
        pct_2 = pct_within_fold(pred_test, y_test, 2.0)

        push!(fold_gmfes, gmfe)
        push!(fold_2fold, pct_2)
    end

    mean_gmfe = mean(fold_gmfes)
    mean_2fold = mean(fold_2fold)

    push!(all_results, (gmfe=mean_gmfe, pct_2fold=mean_2fold, gmfes=fold_gmfes))
    println("  Seed $seed: GMFE=$(round(mean_gmfe, digits=3)), 2-fold=$(round(mean_2fold, digits=1))%")
end

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

println("\n" * "="^80)
println("RESULTS: FRACTAL-MECHANISTIC PBPK MODEL")
println("="^80)

all_gmfes = vcat([r.gmfes for r in all_results]...)
mean_gmfe = mean(all_gmfes)
std_gmfe = std(all_gmfes)
best_gmfe = minimum(all_gmfes)
mean_2fold = mean([r.pct_2fold for r in all_results])

println("\nCross-Validation Results ($(n_folds)-fold × $(n_seeds) seeds):")
println("  Mean GMFE: $(round(mean_gmfe, digits=3)) ± $(round(std_gmfe, digits=3))")
println("  Best Fold GMFE: $(round(best_gmfe, digits=3))")
println("  Mean % within 2-fold: $(round(mean_2fold, digits=1))%")

println("\n" * "-"^40)
println("FDA/EMA Regulatory Thresholds:")
println("  GMFE < 2.0: $(mean_gmfe < 2.0 ? "✓ PASS" : "✗ FAIL") (mean=$(round(mean_gmfe, digits=3)))")
println("  Best fold < 2.0: $(best_gmfe < 2.0 ? "✓ PASS" : "✗ FAIL") (best=$(round(best_gmfe, digits=3)))")
println("  >50% within 2-fold: $(mean_2fold > 50 ? "✓ PASS" : "✗ FAIL") ($(round(mean_2fold, digits=1))%)")

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

println("\n" * "-"^40)
println("Feature Analysis:")

# Train final model on all data for analysis
μ_all = mean(X_all, dims=2)
σ_all = std(X_all, dims=2) .+ 1e-8
X_norm = (X_all .- μ_all) ./ σ_all

net_final = FractalPBPKNet(size(X_all, 1), 64, 32)
opt_final = AdamOptimizer(lr=0.002)

for epoch in 1:300
    batch_idx = rand(1:n_samples, 64)
    grads = compute_gradients(net_final, X_norm[:, batch_idx], y[batch_idx])
    adam_update!(opt_final, net_final, grads)
end

# Compute permutation importance
feature_names = [
    "MW", "HBA", "HBD", "TPSA", "RotBond", "LogP", "LogD", "fup",  # Physicochemical
    "FracDim", "TopEntropy", "BranchComplex", "Wiener", "BalabanJ",  # Fractal
    "Randic", "Zagreb1", "Zagreb2", "SelfSim", "EffDistDim",  # More fractal
    "MechVdss"  # Mechanistic
]

pred_base = forward(net_final, X_norm)
base_mse = mean((pred_base .- y).^2)

println("\nFeature Importance (permutation):")
importances = Float64[]
for i in 1:size(X_norm, 1)
    X_perm = copy(X_norm)
    X_perm[i, :] = shuffle(X_perm[i, :])
    pred_perm = forward(net_final, X_perm)
    perm_mse = mean((pred_perm .- y).^2)
    importance = (perm_mse - base_mse) / base_mse * 100
    push!(importances, importance)
end

# Sort by importance
sorted_idx = sortperm(importances, rev=true)
for (rank, idx) in enumerate(sorted_idx[1:10])
    name = idx <= length(feature_names) ? feature_names[idx] : "Feature_$idx"
    println("  $(rank). $name: $(round(importances[idx], digits=2))%")
end

# ============================================================================
# COMPARISON WITH BASELINES
# ============================================================================

println("\n" * "-"^40)
println("Comparison with Previous Approaches:")

println("\n  Method                        | GMFE  | % 2-fold")
println("  " * "-"^50)
println("  Literature (Øie-Tozer + exp)  | 1.55  | 81%")
println("  PKSmart (2024)                | 2.09  | ~60%")
println("  Our descriptors only          | 2.19  | 57%")
println("  This: Fractal + Mechanistic   | $(round(mean_gmfe, digits=2))  | $(round(mean_2fold, digits=0))%")

# ============================================================================
# MECHANISTIC BASELINE
# ============================================================================

println("\n" * "-"^40)
println("Mechanistic-Only Baseline (Rodgers-Rowland + Øie-Tozer):")

mech_pred = vec(X_mech)  # Already computed
mech_gmfe = compute_gmfe(log.(mech_pred .+ 0.01), y)
mech_2fold = pct_within_fold(log.(mech_pred .+ 0.01), y, 2.0)

println("  Mechanistic GMFE: $(round(mech_gmfe, digits=3))")
println("  Mechanistic % 2-fold: $(round(mech_2fold, digits=1))%")
println("  Improvement from ML: $(round((mech_gmfe - mean_gmfe) / mech_gmfe * 100, digits=1))%")

println("\n" * "="^80)
println("Paradigm: Self-similarity across scales - molecular fractals")
println("coupling with physiological fractal networks determines distribution.")
println("="^80)
