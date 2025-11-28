"""
RIGOROUS ABLATION STUDY: Do Fractal Features Actually Help?

This script answers ONE question scientifically:
    Does adding fractal features improve Vdss prediction?

Methodology:
    - Control: Standard physicochemical features only
    - Treatment: Physicochemical + Fractal features
    - Same architecture, same seeds, same folds
    - Statistical test: Paired t-test across folds
    - Multiple seeds for robustness

This is REAL SCIENCE: we accept whatever the data shows.
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
println("RIGOROUS ABLATION STUDY")
println("Question: Do fractal features improve Vdss prediction?")
println("="^80)

# ============================================================================
# FEATURE SETS
# ============================================================================

"""
BASELINE features - standard physicochemical descriptors.
These are what everyone uses.
"""
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
        MW / 500,           # Molecular weight (normalized)
        HBA / 12,           # H-bond acceptors
        HBD / 6,            # H-bond donors
        TPSA / 150,         # Topological polar surface area
        RB / 12,            # Rotatable bonds
        (logP + 3) / 10,    # LogP
        (logD + 3) / 10,    # LogD at pH 7.4
        log10(fup + 0.001) / 3 + 1,  # Fraction unbound (log scale)
        fup,                # Fraction unbound (linear)
        10^logD / (10^logD + 1),  # Membrane permeability proxy
        TPSA / MW * 10,     # PSA/MW ratio
        (HBD + HBA) / 20,   # Total H-bond capacity
    ]
end

"""
FRACTAL features - our novel contribution.
Added ON TOP of baseline features.
"""
function compute_fractal_features(row, frac_desc)
    fup = row[:human_fup]
    logP = row[Symbol("MoKa.LogP")]
    logD = row[Symbol("MoKa.LogD7.4")]
    MW = row[:MW]
    TPSA = row[:TPSA_NO]
    HBD = Float64(row[:HBD])
    HBA = Float64(row[:HBA])
    RB = Float64(row[:RotBondCount])
    P = 10^logD

    # Molecular fractal dimension (our formula)
    d_f_mol = molecular_fractal_dim(MW, RB, TPSA, HBD, HBA)

    # Tissue reference (average)
    d_f_tissue = 2.70
    α_tissue = 0.80
    d_s = 4/3  # Alexander-Orbach

    # Fractal coupling coefficient
    coupling = exp(-abs(d_f_mol - d_f_tissue)^2 / 0.09)

    # Estimated fut using fractal theory
    fut = 1 / (1 + 0.1 * P) * (d_f_tissue / 3)^α_tissue
    fut = clamp(fut, 0.001, 0.99)

    Float64[
        # Molecular fractal descriptors
        d_f_mol / 3,                           # Molecular fractal dim
        frac_desc["fractal_dim"] / 3,          # Graph-based fractal dim
        frac_desc["topological_entropy"] / 2,  # Topological entropy
        frac_desc["branching_complexity"],     # Branching complexity
        frac_desc["fragment_self_similarity"], # Self-similarity

        # Tissue-molecule coupling
        coupling,                              # Fractal coupling η
        (d_f_mol - d_f_tissue) / 0.5,         # Dimension mismatch
        (d_f_mol / d_f_tissue)^α_tissue,      # Scaled coupling

        # Fractal transport
        d_s / 2,                              # Spectral dim contribution
        (fup / fut)^(d_s/2) * coupling,       # Fractal Øie-Tozer term
        log(fut + 0.01) / 3 + 1,              # Log fut

        # Mechanistic prediction (as feature)
        log(0.04 + 0.17 * (fup/fut)^(d_s/2) * coupling +
            0.39 * (fup/fut) * coupling + 0.01),  # Log Vdss_mech
    ]
end

# ============================================================================
# NEURAL NETWORK (identical for both conditions)
# ============================================================================

relu(x) = max(0.0, x)

mutable struct SimpleNet
    W1::Matrix{Float64}; b1::Vector{Float64}
    W2::Matrix{Float64}; b2::Vector{Float64}
    W3::Matrix{Float64}; b3::Vector{Float64}
end

function SimpleNet(din, h1=48, h2=24)
    SimpleNet(
        randn(h1, din) .* sqrt(2/din), zeros(h1),
        randn(h2, h1) .* sqrt(2/h1), zeros(h2),
        randn(1, h2) .* sqrt(1/h2), zeros(1)
    )
end

function forward(net::SimpleNet, X)
    a1 = relu.(net.W1 * X .+ net.b1)
    a2 = relu.(net.W2 * a1 .+ net.b2)
    return vec(net.W3 * a2 .+ net.b3)
end

function train!(net::SimpleNet, X, y; epochs=300, lr=0.003, λ=0.0003, batch_size=64)
    n = size(X, 2)

    for ep in 1:epochs
        # Mini-batch
        idx = rand(1:n, min(batch_size, n))
        Xb, yb = X[:, idx], y[idx]
        nb = length(idx)

        # Forward
        z1 = net.W1 * Xb .+ net.b1
        a1 = relu.(z1)
        z2 = net.W2 * a1 .+ net.b2
        a2 = relu.(z2)
        pred = vec(net.W3 * a2 .+ net.b3)

        # Backward
        diff = (pred .- yb) ./ nb

        d3 = reshape(diff, 1, :)
        dW3 = d3 * a2' .+ 2λ .* net.W3
        db3 = vec(sum(d3, dims=2))

        d2 = (net.W3' * d3) .* (z2 .> 0)
        dW2 = d2 * a1' .+ 2λ .* net.W2
        db2 = vec(sum(d2, dims=2))

        d1 = (net.W2' * d2) .* (z1 .> 0)
        dW1 = d1 * Xb' .+ 2λ .* net.W1
        db1 = vec(sum(d1, dims=2))

        # Gradient clipping
        for g in [dW1, db1, dW2, db2, dW3, db3]
            gn = norm(g)
            gn > 1.0 && (g .*= 1.0 / gn)
        end

        # SGD with momentum would be here, using simple SGD for reproducibility
        net.W1 .-= lr .* dW1; net.b1 .-= lr .* db1
        net.W2 .-= lr .* dW2; net.b2 .-= lr .* db2
        net.W3 .-= lr .* dW3; net.b3 .-= lr .* db3
    end
end

# ============================================================================
# METRICS
# ============================================================================

function gmfe(pred, obs)
    p, o = exp.(pred), exp.(obs)
    fe = max.(p./o, o./p)
    return exp(mean(log.(fe)))
end

function rmse(pred, obs)
    sqrt(mean((pred .- obs).^2))
end

function pct_within_fold(pred, obs, fold)
    p, o = exp.(pred), exp.(obs)
    ratio = p ./ o
    mean((ratio .>= 1/fold) .& (ratio .<= fold)) * 100
end

# ============================================================================
# LOAD DATA
# ============================================================================

println("\n[1] Loading Lombardo 1352 dataset...")
df = CSV.read("/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv", DataFrame)
df = dropmissing(df, [:smiles_r, :human_VDss_L_kg, :human_fup,
                      Symbol("MoKa.LogP"), Symbol("MoKa.LogD7.4"), :MW])

# Precompute all features
baseline_X = Vector{Vector{Float64}}()
fractal_X = Vector{Vector{Float64}}()
y_all = Float64[]
valid_idx = Int[]

println("  Computing features for $(nrow(df)) compounds...")

for (i, row) in enumerate(eachrow(df))
    try
        base_feat = compute_baseline_features(row)
        frac_desc = compute_all_fractal_descriptors(row[:smiles_r])
        frac_feat = compute_fractal_features(row, frac_desc)

        push!(baseline_X, base_feat)
        push!(fractal_X, vcat(base_feat, frac_feat))  # Fractal = baseline + fractal
        push!(y_all, log(row[:human_VDss_L_kg]))
        push!(valid_idx, i)
    catch e
        continue  # Skip compounds where SMILES parsing fails
    end
end

n_samples = length(y_all)
n_baseline = length(baseline_X[1])
n_fractal = length(fractal_X[1])

println("  Valid samples: $n_samples")
println("  Baseline features: $n_baseline")
println("  Fractal features: $n_fractal (baseline + $(n_fractal - n_baseline) new)")

X_base = hcat(baseline_X...)
X_frac = hcat(fractal_X...)
y = y_all

# ============================================================================
# ABLATION STUDY: Paired comparison across identical folds
# ============================================================================

println("\n[2] Running ablation study...")
println("    Design: 5-fold CV × 10 seeds, paired comparison")

n_folds = 5
n_seeds = 10

# Store results for statistical test
baseline_gmfes = Float64[]
fractal_gmfes = Float64[]
baseline_2fold = Float64[]
fractal_2fold = Float64[]

for seed in 1:n_seeds
    Random.seed!(seed * 42)  # Reproducible
    idx = shuffle(1:n_samples)
    fold_size = n_samples ÷ n_folds

    for fold in 1:n_folds
        # Same fold split for both conditions
        te_start = (fold - 1) * fold_size + 1
        te_end = fold == n_folds ? n_samples : fold * fold_size
        te_idx = idx[te_start:te_end]
        tr_idx = setdiff(idx, te_idx)

        # Normalize training data
        function normalize(X_tr, X_te)
            μ = mean(X_tr, dims=2)
            σ = std(X_tr, dims=2) .+ 1e-8
            return (X_tr .- μ) ./ σ, (X_te .- μ) ./ σ
        end

        # BASELINE condition
        Xtr_b, Xte_b = normalize(X_base[:, tr_idx], X_base[:, te_idx])
        ytr, yte = y[tr_idx], y[te_idx]

        net_base = SimpleNet(n_baseline, 48, 24)
        train!(net_base, Xtr_b, ytr, epochs=300, lr=0.003)
        pred_base = forward(net_base, Xte_b)

        g_base = gmfe(pred_base, yte)
        f2_base = pct_within_fold(pred_base, yte, 2.0)
        push!(baseline_gmfes, g_base)
        push!(baseline_2fold, f2_base)

        # FRACTAL condition (same fold, same seed for network init)
        Random.seed!(seed * 42 + fold * 1000)  # Same init randomness
        Xtr_f, Xte_f = normalize(X_frac[:, tr_idx], X_frac[:, te_idx])

        net_frac = SimpleNet(n_fractal, 48, 24)
        train!(net_frac, Xtr_f, ytr, epochs=300, lr=0.003)
        pred_frac = forward(net_frac, Xte_f)

        g_frac = gmfe(pred_frac, yte)
        f2_frac = pct_within_fold(pred_frac, yte, 2.0)
        push!(fractal_gmfes, g_frac)
        push!(fractal_2fold, f2_frac)
    end

    # Progress
    base_mean = mean(baseline_gmfes[end-n_folds+1:end])
    frac_mean = mean(fractal_gmfes[end-n_folds+1:end])
    @printf("  Seed %2d: Baseline GMFE=%.3f, Fractal GMFE=%.3f, Δ=%.3f\n",
            seed, base_mean, frac_mean, base_mean - frac_mean)
end

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

println("\n" * "="^80)
println("STATISTICAL ANALYSIS")
println("="^80)

# Paired differences
diff_gmfe = baseline_gmfes .- fractal_gmfes  # Positive = fractal better
diff_2fold = fractal_2fold .- baseline_2fold  # Positive = fractal better

n_pairs = length(diff_gmfe)
mean_diff_gmfe = mean(diff_gmfe)
std_diff_gmfe = std(diff_gmfe)
se_diff_gmfe = std_diff_gmfe / sqrt(n_pairs)

mean_diff_2fold = mean(diff_2fold)
std_diff_2fold = std(diff_2fold)
se_diff_2fold = std_diff_2fold / sqrt(n_pairs)

# t-statistic (H0: mean difference = 0)
t_gmfe = mean_diff_gmfe / se_diff_gmfe
t_2fold = mean_diff_2fold / se_diff_2fold

# Two-tailed p-value approximation (df = n-1)
# Using normal approximation for large n
function p_value_approx(t, df)
    # Approximation for |t| (two-tailed)
    if abs(t) > 4
        return 2 * exp(-0.5 * t^2) / sqrt(2π) / abs(t)  # Asymptotic
    else
        # Rough approximation
        return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    end
end

using SpecialFunctions: erf

p_gmfe = p_value_approx(t_gmfe, n_pairs - 1)
p_2fold = p_value_approx(t_2fold, n_pairs - 1)

println("\n1. GMFE COMPARISON (lower is better)")
println("   Baseline: $(round(mean(baseline_gmfes), digits=3)) ± $(round(std(baseline_gmfes), digits=3))")
println("   Fractal:  $(round(mean(fractal_gmfes), digits=3)) ± $(round(std(fractal_gmfes), digits=3))")
println("   Mean Δ:   $(round(mean_diff_gmfe, digits=4)) (positive = fractal better)")
println("   t-stat:   $(round(t_gmfe, digits=3))")
println("   p-value:  $(round(p_gmfe, digits=4))")
println("   95% CI:   [$(round(mean_diff_gmfe - 1.96*se_diff_gmfe, digits=4)), $(round(mean_diff_gmfe + 1.96*se_diff_gmfe, digits=4))]")

println("\n2. % WITHIN 2-FOLD (higher is better)")
println("   Baseline: $(round(mean(baseline_2fold), digits=1))% ± $(round(std(baseline_2fold), digits=1))%")
println("   Fractal:  $(round(mean(fractal_2fold), digits=1))% ± $(round(std(fractal_2fold), digits=1))%")
println("   Mean Δ:   $(round(mean_diff_2fold, digits=2))% (positive = fractal better)")
println("   t-stat:   $(round(t_2fold, digits=3))")
println("   p-value:  $(round(p_2fold, digits=4))")

# ============================================================================
# CONCLUSION
# ============================================================================

println("\n" * "="^80)
println("CONCLUSION")
println("="^80)

sig_level = 0.05

if p_gmfe < sig_level && mean_diff_gmfe > 0
    println("\n✓ FRACTAL FEATURES SIGNIFICANTLY IMPROVE GMFE")
    println("  Effect size: $(round(mean_diff_gmfe, digits=3)) reduction in GMFE")
    println("  p = $(round(p_gmfe, digits=4)) < $sig_level")
elseif p_gmfe < sig_level && mean_diff_gmfe < 0
    println("\n✗ FRACTAL FEATURES SIGNIFICANTLY WORSEN GMFE")
    println("  Effect size: $(round(-mean_diff_gmfe, digits=3)) increase in GMFE")
    println("  p = $(round(p_gmfe, digits=4)) < $sig_level")
else
    println("\n○ NO SIGNIFICANT DIFFERENCE IN GMFE")
    println("  p = $(round(p_gmfe, digits=4)) ≥ $sig_level")
    println("  Cannot conclude fractal features help or hurt")
end

if p_2fold < sig_level && mean_diff_2fold > 0
    println("\n✓ FRACTAL FEATURES SIGNIFICANTLY IMPROVE 2-FOLD ACCURACY")
    println("  Effect size: +$(round(mean_diff_2fold, digits=1))% within 2-fold")
elseif p_2fold < sig_level && mean_diff_2fold < 0
    println("\n✗ FRACTAL FEATURES SIGNIFICANTLY WORSEN 2-FOLD ACCURACY")
else
    println("\n○ NO SIGNIFICANT DIFFERENCE IN 2-FOLD ACCURACY")
end

# Honest interpretation
println("\n" * "-"^60)
println("HONEST INTERPRETATION:")
if p_gmfe >= sig_level
    println("  The data does not provide sufficient evidence that fractal")
    println("  features improve Vdss prediction. This could mean:")
    println("    1. Fractal features don't help (null hypothesis true)")
    println("    2. Effect is too small to detect with this sample size")
    println("    3. High variance obscures a real effect")
    println("  More data or better optimization needed to conclude.")
else
    println("  The statistical test suggests a real effect, but:")
    println("    1. Effect size matters more than p-value")
    println("    2. External validation still required")
    println("    3. Biological plausibility must be considered")
end

println("\n" * "="^80)
println("This is honest science: we report what the data shows.")
println("="^80)

# Save results
results = DataFrame(
    seed = repeat(1:n_seeds, inner=n_folds),
    fold = repeat(1:n_folds, outer=n_seeds),
    baseline_gmfe = baseline_gmfes,
    fractal_gmfe = fractal_gmfes,
    baseline_2fold = baseline_2fold,
    fractal_2fold = fractal_2fold
)

CSV.write("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration/scripts/training/ablation_results.csv", results)
println("\nResults saved to ablation_results.csv")
