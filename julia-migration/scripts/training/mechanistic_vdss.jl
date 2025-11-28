"""
MECHANISTIC VDSS PREDICTION WITH PROPER FUT ESTIMATION

Using the Rodgers & Rowland (2005, 2006) equations for tissue partition.

The key insight: Vdss depends on tissue binding (fut), which we must estimate
from physicochemical properties since we don't have experimental data.

Rodgers-Rowland equations:
- For bases: Kp = (fup/fut) * (1 + Ka*10^(pKa-pH))
- For acids: Kp = fup * (Vw + Vnl*Kow + Vnpl*Knpl) / fut
- fut = 1 / (1 + Ka*AP*10^(pKa-pH)) for bases

This script implements a hybrid approach:
1. Estimate fut using Rodgers-Rowland-inspired equations
2. Use experimental fup from Lombardo dataset
3. Combine with ML to learn corrections
"""

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Random
using Printf

println("="^70)
println("MECHANISTIC VDSS WITH RODGERS-ROWLAND FUT ESTIMATION")
println("="^70)

# ============================================================================
# LOAD DATA
# ============================================================================

println("\n[1] Loading Lombardo dataset...")
df = CSV.read("/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv", DataFrame)
df = dropmissing(df, [:smiles_r, :human_VDss_L_kg, :human_fup,
                      Symbol("MoKa.LogP"), Symbol("MoKa.LogD7.4"), :MW, :TPSA_NO])

println("  Compounds with complete data: $(nrow(df))")

# ============================================================================
# RODGERS-ROWLAND INSPIRED FUT ESTIMATION
# ============================================================================

"""
Estimate fraction unbound in tissue using Rodgers-Rowland concepts.

Key factors:
1. Lipophilicity (LogP, LogD) - drives neutral lipid partitioning
2. Ionization (LogP - LogD difference indicates ionization)
3. Polar surface area - affects phospholipid binding
4. Plasma binding - correlates with tissue binding

The actual R-R equations require tissue composition data.
We use a simplified empirical version.
"""
function estimate_fut_rr(fup, logP, logD, MW, TPSA)
    # Ionization indicator (if LogP ≈ LogD, compound is neutral)
    delta_log = logP - logD
    is_ionized = abs(delta_log) > 0.5

    # Neutral lipid partition
    P_neutral = 10^logP
    P_at_pH = 10^logD

    # Tissue water fraction
    Vw = 0.7

    # Neutral lipid fraction in tissue (average)
    Vnl = 0.02

    # Phospholipid fraction
    Vpl = 0.005

    # Acidic phospholipid concentration (for bases)
    AP = 0.5  # mg/mL

    # Estimate Ka (association constant with acidic phospholipids)
    # Higher for bases, lower for acids
    if delta_log > 1.0  # Likely basic
        Ka = 10^(0.5 * logP) * 0.001  # Empirical scaling
    elseif delta_log < -0.5  # Likely acidic
        Ka = 0.01
    else  # Neutral
        Ka = 0.1
    end

    # Simplified fut estimation
    # fut ~ 1 / (1 + Vnl*Kow + Ka*AP)

    # For neutrals and weak acids
    fut_neutral = 1 / (1 + Vnl * P_neutral + Ka * AP)

    # For bases (stronger tissue binding)
    fut_base = 1 / (1 + Vnl * P_neutral + Ka * AP * 10.0)

    # Blend based on ionization
    if delta_log > 1.0
        fut = fut_base
    elseif delta_log < -0.5
        fut = fut_neutral * 1.5  # Acids have higher fut
    else
        fut = fut_neutral
    end

    # Polar surface area correction (high TPSA = less lipophilic binding)
    psa_factor = exp(-TPSA / 200)
    fut = fut * (1 - 0.5 * psa_factor) + 0.5 * psa_factor

    # Correlation with plasma binding
    # If fup is very low, fut is also likely low
    fut = fut * (0.3 + 0.7 * (fup^0.3))

    return clamp(fut, 0.001, 0.99)
end

"""
Øie-Tozer equation for Vdss:
Vdss = Vp + Ve*(fup/fut) + Vr*(fup/fut)

Where:
- Vp = plasma volume (0.04 L/kg)
- Ve = extracellular fluid minus plasma (0.15 L/kg)
- Vr = cellular tissue volume (0.4 L/kg)
"""
function oie_tozer_vdss(fup, fut)
    Vp = 0.04
    Ve = 0.15
    Vr = 0.40

    fup_fut = fup / fut

    # Standard Øie-Tozer
    Vdss = Vp + Ve * fup_fut + Vr * fup_fut

    return Vdss
end

# ============================================================================
# COMPUTE FEATURES
# ============================================================================

println("\n[2] Computing mechanistic features...")

# Extract data
fup = df.human_fup
logP = df[!, Symbol("MoKa.LogP")]
logD = df[!, Symbol("MoKa.LogD7.4")]
MW = df.MW
TPSA = df.TPSA_NO
HBA = df.HBA
HBD = df.HBD
RB = df.RotBondCount
Vdss_obs = df.human_VDss_L_kg

n = length(Vdss_obs)

# Estimate fut for each compound
fut_est = [estimate_fut_rr(fup[i], logP[i], logD[i], MW[i], TPSA[i]) for i in 1:n]

# Calculate mechanistic Vdss
Vdss_mech = [oie_tozer_vdss(fup[i], fut_est[i]) for i in 1:n]

# Compute GMFE for mechanistic model alone
fe_mech = max.(Vdss_mech ./ Vdss_obs, Vdss_obs ./ Vdss_mech)
gmfe_mech = exp(mean(log.(fe_mech)))

println("  Mechanistic Øie-Tozer with R-R fut estimation:")
println("    GMFE: $(round(gmfe_mech, digits=3))")
println("    % within 2-fold: $(round(mean(fe_mech .<= 2) * 100, digits=1))%")
println("    % within 3-fold: $(round(mean(fe_mech .<= 3) * 100, digits=1))%")

# ============================================================================
# CREATE HYBRID FEATURES
# ============================================================================

println("\n[3] Creating hybrid features (mechanistic + descriptors)...")

function compute_hybrid_features(i)
    Float64[
        # Experimental fup (most important!)
        fup[i],
        log10(fup[i] + 1e-4),

        # Estimated fut from R-R
        fut_est[i],
        log10(fut_est[i] + 1e-4),

        # The ratio (key for Øie-Tozer)
        fup[i] / fut_est[i],
        log10(fup[i] / fut_est[i] + 1e-4),

        # Mechanistic Vdss prediction
        log10(Vdss_mech[i]),

        # Physicochemical descriptors
        MW[i] / 500,
        logP[i] / 5,
        logD[i] / 5,
        logP[i] - logD[i],  # Ionization indicator
        TPSA[i] / 150,
        HBA[i] / 10,
        HBD[i] / 5,
        RB[i] / 10,

        # Derived features
        10^logD[i] / (1 + 10^logD[i]),  # Membrane permeability
        TPSA[i] / MW[i] * 100,  # PSA/MW ratio
    ]
end

X = hcat([compute_hybrid_features(i) for i in 1:n]...)
y = log.(Vdss_obs)

nf = size(X, 1)
println("  Features: $nf")

# ============================================================================
# NEURAL NETWORK
# ============================================================================

relu(x) = max(0.0, x)

mutable struct Net
    W1::Matrix{Float64}; b1::Vector{Float64}
    W2::Matrix{Float64}; b2::Vector{Float64}
    W3::Vector{Float64}; b3::Float64
end

function Net(din, h1=32, h2=16)
    Net(
        randn(h1, din) .* sqrt(2/din), zeros(h1),
        randn(h2, h1) .* sqrt(2/h1), zeros(h2),
        randn(h2) .* sqrt(1/h2), 0.0
    )
end

function forward(net::Net, X)
    a1 = relu.(net.W1 * X .+ net.b1)
    a2 = relu.(net.W2 * a1 .+ net.b2)
    return vec(net.W3' * a2 .+ net.b3)
end

function train!(net::Net, X, y; epochs=400, lr=0.01, λ=0.001)
    n = size(X, 2)
    for ep in 1:epochs
        current_lr = lr * (1 - ep / (epochs * 1.2))

        z1 = net.W1 * X .+ net.b1
        a1 = relu.(z1)
        z2 = net.W2 * a1 .+ net.b2
        a2 = relu.(z2)
        pred = vec(net.W3' * a2 .+ net.b3)

        diff = (pred .- y) ./ n

        db3 = sum(diff)
        dW3 = a2 * diff .+ 2λ .* net.W3

        d2 = (net.W3 .* diff') .* (z2 .> 0)
        db2 = vec(sum(d2, dims=2))
        dW2 = d2 * a1' .+ 2λ .* net.W2

        d1 = (net.W2' * d2) .* (z1 .> 0)
        db1 = vec(sum(d1, dims=2))
        dW1 = d1 * X' .+ 2λ .* net.W1

        for g in [dW1, db1, dW2, db2, dW3]
            gn = norm(g)
            gn > 1.0 && (g .*= 1.0 / gn)
        end

        net.W1 .-= current_lr .* dW1
        net.b1 .-= current_lr .* db1
        net.W2 .-= current_lr .* dW2
        net.b2 .-= current_lr .* db2
        net.W3 .-= current_lr .* dW3
        net.b3 -= current_lr * db3
    end
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
# CROSS-VALIDATION
# ============================================================================

println("\n[4] Cross-validation (5-fold × 10 seeds)...")

n_folds, n_seeds = 5, 10
all_gmfes = Float64[]
all_2fold = Float64[]

for seed in 1:n_seeds
    Random.seed!(seed * 42)
    idx = shuffle(1:n)
    fs = n ÷ n_folds

    seed_gmfes = Float64[]

    for fold in 1:n_folds
        te_idx = idx[(fold-1)*fs+1 : fold==n_folds ? n : fold*fs]
        tr_idx = setdiff(idx, te_idx)

        # Normalize
        μ = mean(X[:, tr_idx], dims=2)
        σ = std(X[:, tr_idx], dims=2) .+ 1e-8
        Xtr = (X[:, tr_idx] .- μ) ./ σ
        Xte = (X[:, te_idx] .- μ) ./ σ
        ytr, yte = y[tr_idx], y[te_idx]

        # Train
        net = Net(nf, 32, 16)
        train!(net, Xtr, ytr)

        # Evaluate
        pred = forward(net, Xte)
        g = gmfe(pred, yte)
        push!(all_gmfes, g)
        push!(seed_gmfes, g)
        push!(all_2fold, pct_fold(pred, yte, 2.0))
    end

    @printf("  Seed %2d: GMFE = %.3f ± %.3f\n", seed, mean(seed_gmfes), std(seed_gmfes))
end

# ============================================================================
# RESULTS
# ============================================================================

println("\n" * "="^70)
println("RESULTS: HYBRID MECHANISTIC + ML MODEL")
println("="^70)

println("\nMechanistic Baseline (Øie-Tozer with R-R fut):")
println("  GMFE: $(round(gmfe_mech, digits=3))")
println("  % within 2-fold: $(round(mean(fe_mech .<= 2) * 100, digits=1))%")

println("\nHybrid Model (Mechanistic features + ML):")
println("  GMFE: $(round(mean(all_gmfes), digits=3)) ± $(round(std(all_gmfes), digits=3))")
println("  Best fold: $(round(minimum(all_gmfes), digits=3))")
println("  % within 2-fold: $(round(mean(all_2fold), digits=1))%")

improvement = (gmfe_mech - mean(all_gmfes)) / gmfe_mech * 100
println("\nImprovement over mechanistic: $(round(improvement, digits=1))%")

# Stability
stable_pct = sum(all_gmfes .< 3.0) / length(all_gmfes) * 100
pass_pct = sum(all_gmfes .< 2.0) / length(all_gmfes) * 100
println("\nStability:")
println("  % runs with GMFE < 3.0: $(round(stable_pct, digits=0))%")
println("  % runs with GMFE < 2.0: $(round(pass_pct, digits=0))%")

println("\n" * "="^70)
println("COMPARISON TO LITERATURE")
println("="^70)
println("  Øie-Tozer + experimental fut:  GMFE ~1.55 (gold standard)")
println("  Our mechanistic (R-R fut est): GMFE $(round(gmfe_mech, digits=2))")
println("  Our hybrid ML:                 GMFE $(round(mean(all_gmfes), digits=2))")
println("  PKSmart 2024 (SOTA):           GMFE ~2.09")
println("="^70)
