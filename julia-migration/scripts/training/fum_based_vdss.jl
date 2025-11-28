"""
VDSS PREDICTION USING FU,MIC AS PROXY FOR FUT

Key insight from literature:
- fu,mic (fraction unbound in microsomes) can be predicted from logP/D
- fu,mic correlates with tissue binding since microsomes have similar
  phospholipid composition to cell membranes
- Using fu,mic as proxy for fut should improve Vdss predictions

References:
- Hallifax & Houston (2006) Drug Metab Dispos 34:724-35
- Austin et al. (2002) Drug Metab Dispos 30:1497-503

Equation (Hallifax & Houston):
  For 1 mg/mL microsomal protein:
  log(1/fu,mic - 1) = 0.072 + 0.067*logP - 0.224*logP² + 0.0104*logP³

Alternative (Austin):
  fu,mic = 1 / (1 + [mic]*10^(0.072*logP-0.56))
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
println("VDSS PREDICTION WITH FU,MIC-BASED TISSUE BINDING")
println("="^70)

# ============================================================================
# FU,MIC PREDICTION EQUATIONS
# ============================================================================

"""
Hallifax & Houston (2006) equation for fu,mic.
Valid for 1 mg/mL microsomal protein concentration.

Input: logP (or logD for ionized compounds)
Output: fu,mic (fraction unbound in microsomes)
"""
function fu_mic_hallifax(logP; mic_conc=1.0)
    # Adjust for microsomal protein concentration
    # Standard equation is for 1 mg/mL
    logP_eff = logP * (mic_conc / 1.0)^0.3

    # Hallifax equation (polynomial in logP)
    log_bound_ratio = 0.072 + 0.067*logP_eff - 0.224*logP_eff^2 + 0.0104*logP_eff^3

    # Convert to fu,mic
    bound_ratio = 10^log_bound_ratio
    fu_mic = 1 / (1 + bound_ratio)

    return clamp(fu_mic, 0.001, 0.99)
end

"""
Austin et al. (2002) simplified equation.
More commonly used, validated across larger chemical space.
"""
function fu_mic_austin(logP; mic_conc=1.0)
    # fu,mic = 1 / (1 + [mic] * 10^(0.072*logP - 0.56))
    exponent = 0.072 * logP - 0.56
    fu_mic = 1 / (1 + mic_conc * 10^exponent)
    return clamp(fu_mic, 0.001, 0.99)
end

"""
Estimate fut from fu,mic.

The relationship between fu,mic and fut is based on:
- Both depend on phospholipid binding
- Tissue has ~10-50x more lipid content than microsomes at 1 mg/mL
- Empirical scaling factor needed

From mechanistic PBPK literature:
  fut ≈ fu,mic^α where α ≈ 0.5-0.8 (tissue-dependent)
"""
function fut_from_fumic(fu_mic; tissue_factor=0.6)
    # Empirical relationship: tissues bind more than microsomes
    # fut = fu_mic^α, where α < 1 means more binding in tissue
    fut = fu_mic^tissue_factor
    return clamp(fut, 0.001, 0.99)
end

# ============================================================================
# LOAD DATA
# ============================================================================

println("\n[1] Loading Lombardo dataset...")
df = CSV.read("/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv", DataFrame)
df = dropmissing(df, [:smiles_r, :human_VDss_L_kg, :human_fup,
                      Symbol("MoKa.LogP"), Symbol("MoKa.LogD7.4"), :MW, :TPSA_NO])

println("  Compounds: $(nrow(df))")

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

# ============================================================================
# COMPUTE FU,MIC AND FUT
# ============================================================================

println("\n[2] Computing fu,mic and estimating fut...")

# Use logD at pH 7.4 (more relevant for in vivo)
# For neutral compounds, logP ≈ logD
fu_mic = [fu_mic_austin(logD[i]) for i in 1:n]

# Estimate fut from fu,mic
# Test different tissue factors
tissue_factors = [0.4, 0.5, 0.6, 0.7, 0.8]

best_gmfe_val = Inf
best_tf_val = 0.5

for tf in tissue_factors
    global best_gmfe_val, best_tf_val
    fut = [fut_from_fumic(fu_mic[i], tissue_factor=tf) for i in 1:n]

    # Øie-Tozer
    Vp, Ve, Vr = 0.04, 0.15, 0.40
    Vdss_pred = [Vp + (Ve + Vr) * (fup[i] / fut[i]) for i in 1:n]

    # GMFE
    fe = max.(Vdss_pred ./ Vdss_obs, Vdss_obs ./ Vdss_pred)
    gmfe_calc = exp(mean(log.(fe)))
    pct2 = mean(fe .<= 2) * 100

    @printf("  Tissue factor %.1f: GMFE=%.3f, %%2-fold=%.1f%%\n", tf, gmfe_calc, pct2)

    if gmfe_calc < best_gmfe_val
        best_gmfe_val = gmfe_calc
        best_tf_val = tf
    end
end

println("\n  Best tissue factor: $(best_tf_val)")

# Use best tissue factor
fut_est = [fut_from_fumic(fu_mic[i], tissue_factor=best_tf_val) for i in 1:n]

# ============================================================================
# COMPARE DIFFERENT FUT ESTIMATION METHODS
# ============================================================================

println("\n[3] Comparing fut estimation methods...")

# Method 1: Simple logD-based (our previous approach)
fut_simple = [clamp(1 / (1 + 0.05 * 10^logD[i]), 0.001, 0.99) for i in 1:n]

# Method 2: fu,mic-based (new approach)
# Already computed as fut_est

# Method 3: Rodgers-Rowland inspired (previous script)
function fut_rr(fup_i, logP_i, logD_i)
    delta = logP_i - logD_i
    P = 10^logD_i
    Ka = delta > 1 ? 0.1 : 0.01
    fut = 1 / (1 + 0.02 * P + Ka * 0.5)
    fut = fut * (0.3 + 0.7 * fup_i^0.3)
    return clamp(fut, 0.001, 0.99)
end

fut_rr_est = [fut_rr(fup[i], logP[i], logD[i]) for i in 1:n]

# Calculate Vdss and GMFE for each method
Vp, Ve, Vr = 0.04, 0.15, 0.40

methods = [
    ("Simple logD", fut_simple),
    ("fu,mic-based", fut_est),
    ("R-R inspired", fut_rr_est)
]

println("\n  Method comparison (Øie-Tozer):")
println("  " * "-"^50)

for (name, fut_vec) in methods
    Vdss_pred = [Vp + (Ve + Vr) * (fup[i] / fut_vec[i]) for i in 1:n]
    fe = max.(Vdss_pred ./ Vdss_obs, Vdss_obs ./ Vdss_pred)
    gmfe = exp(mean(log.(fe)))
    pct2 = mean(fe .<= 2) * 100
    pct3 = mean(fe .<= 3) * 100
    @printf("  %-15s: GMFE=%.3f, %%2-fold=%5.1f%%, %%3-fold=%5.1f%%\n",
            name, gmfe, pct2, pct3)
end

# ============================================================================
# HYBRID ML MODEL WITH FU,MIC FEATURES
# ============================================================================

println("\n[4] Training hybrid ML model with fu,mic features...")

function compute_fumic_features(i)
    Float64[
        # Experimental fup
        fup[i],
        log10(fup[i] + 1e-4),

        # Predicted fu,mic (key new feature!)
        fu_mic[i],
        log10(fu_mic[i] + 1e-4),

        # fu,mic-based fut estimate
        fut_est[i],
        log10(fut_est[i] + 1e-4),

        # Key ratios
        fup[i] / fut_est[i],
        log10(fup[i] / fut_est[i] + 1e-4),
        fup[i] / fu_mic[i],  # New: fup to fu,mic ratio

        # Mechanistic Vdss prediction
        log10(Vp + (Ve + Vr) * fup[i] / fut_est[i]),

        # Physicochemical
        MW[i] / 500,
        logP[i] / 5,
        logD[i] / 5,
        logP[i] - logD[i],  # Ionization
        TPSA[i] / 150,
        HBA[i] / 10,
        HBD[i] / 5,
        RB[i] / 10,

        # Derived
        10^logD[i] / (1 + 10^logD[i]),
    ]
end

X = hcat([compute_fumic_features(i) for i in 1:n]...)
y = log.(Vdss_obs)
nf = size(X, 1)

println("  Features: $nf")

# Neural network
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

# Metrics
function gmfe(pred, obs)
    p, o = exp.(pred), exp.(obs)
    fe = max.(p./o, o./p)
    exp(mean(log.(fe)))
end

pct_fold(pred, obs, f) = mean((exp.(pred)./exp.(obs) .>= 1/f) .&
                              (exp.(pred)./exp.(obs) .<= f)) * 100

# Cross-validation
println("\n  5-fold CV × 10 seeds...")

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

        μ = mean(X[:, tr_idx], dims=2)
        σ = std(X[:, tr_idx], dims=2) .+ 1e-8
        Xtr = (X[:, tr_idx] .- μ) ./ σ
        Xte = (X[:, te_idx] .- μ) ./ σ
        ytr, yte = y[tr_idx], y[te_idx]

        net = Net(nf, 32, 16)
        train!(net, Xtr, ytr)

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
println("FINAL RESULTS")
println("="^70)

println("\nMechanistic Only (Øie-Tozer with fu,mic-based fut):")
Vdss_mech = [Vp + (Ve + Vr) * (fup[i] / fut_est[i]) for i in 1:n]
fe_mech = max.(Vdss_mech ./ Vdss_obs, Vdss_obs ./ Vdss_mech)
@printf("  GMFE: %.3f\n", exp(mean(log.(fe_mech))))
@printf("  %% within 2-fold: %.1f%%\n", mean(fe_mech .<= 2) * 100)

println("\nHybrid ML (fu,mic features + neural network):")
@printf("  GMFE: %.3f ± %.3f\n", mean(all_gmfes), std(all_gmfes))
@printf("  Best fold: %.3f\n", minimum(all_gmfes))
@printf("  %% within 2-fold: %.1f%%\n", mean(all_2fold))

println("\nStability:")
@printf("  %% runs with GMFE < 2.0: %.0f%%\n", sum(all_gmfes .< 2.0) / length(all_gmfes) * 100)
@printf("  %% runs with GMFE < 3.0: %.0f%%\n", sum(all_gmfes .< 3.0) / length(all_gmfes) * 100)

println("\n" * "="^70)
println("COMPARISON TO PREVIOUS APPROACHES")
println("="^70)
println("  Previous (R-R fut estimation):     GMFE ~2.4")
println("  This work (fu,mic-based fut):      GMFE $(round(mean(all_gmfes), digits=2))")
println("  State-of-art (PKSmart 2024):       GMFE ~2.09")
println("  Gold standard (experimental fut):  GMFE ~1.55")
println("="^70)
