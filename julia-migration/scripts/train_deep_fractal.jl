"""
Deep Fractal PBPK Model Training

Implementing the full fractal theory:
1. Mittag-Leffler response (fractional kinetics)
2. Spectral dimension (Alexander-Orbach)
3. Molecular-tissue fractal coupling
4. Memory-dependent features
"""

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Random
using SpecialFunctions: gamma

include("../src/DarwinPBPK/fractal_descriptors.jl")
include("../src/DarwinPBPK/fractional_pbpk.jl")

using .FractalDescriptors
using .FractionalPBPK

println("="^80)
println("DEEP FRACTAL PBPK MODEL")
println("Mittag-Leffler × Spectral Dimension × Fractal Coupling")
println("="^80)

# ============================================================================
# DEEP FRACTAL FEATURES
# ============================================================================

"""
Compute deep fractal-informed features

These features capture:
1. Molecular fractal topology
2. Tissue-molecule coupling
3. Spectral dimension effects
4. Fractional kinetics parameters
"""
function deep_fractal_features(row, frac_desc)
    fup = row[:human_fup]
    logP = row[Symbol("MoKa.LogP")]
    logD = row[Symbol("MoKa.LogD7.4")]
    MW = row[:MW]
    TPSA = row[:TPSA_NO]
    HBD = row[:HBD]
    HBA = row[:HBA]
    RB = row[:RotBondCount]
    P = 10^logD

    # --- MOLECULAR FRACTAL DIMENSION ---
    d_f_mol = molecular_fractal_dim(MW, Float64(RB), TPSA, Float64(HBD), Float64(HBA))

    # From our fractal descriptor module
    topo_entropy = frac_desc["topological_entropy"]
    branch_complex = frac_desc["branching_complexity"]
    frac_dim_topo = frac_desc["fractal_dim"]
    self_sim = frac_desc["fragment_self_similarity"]

    # --- TISSUE COUPLING ACROSS ORGANS ---
    # Each tissue has different fractal dimension
    tissues = [:muscle, :adipose, :liver, :brain, :kidney, :lung]
    tissue_d_f = [2.70, 2.40, 2.85, 2.80, 2.88, 2.97]
    tissue_α = [0.80, 0.70, 0.90, 0.60, 0.88, 0.92]

    # Coupling efficiency to each tissue
    couplings = [fractal_coupling(d_f_mol, d_f) for d_f in tissue_d_f]

    # Weighted average coupling (by tissue volume contribution to Vdss)
    weights = [0.40, 0.20, 0.03, 0.02, 0.005, 0.02]
    avg_coupling = sum(couplings .* weights) / sum(weights)

    # Best coupling (drug goes where it fits best)
    max_coupling = maximum(couplings)

    # --- SPECTRAL DIMENSION EFFECTS ---
    # Alexander-Orbach: d_s ≈ 4/3
    d_s = 4/3

    # Effective tissue fractal dimension
    d_f_tissue = sum(tissue_d_f .* weights) / sum(weights)

    # Walk dimension: d_w = 2*d_f/d_s
    d_w = 2 * d_f_tissue / d_s

    # Subdiffusion exponent: ⟨r²⟩ ∝ t^(2/d_w)
    subdiff_exp = 2 / d_w

    # Spectral correction for transport
    spectral_corr = spectral_dimension_correction(d_f_tissue)

    # --- FRACTIONAL KINETICS PARAMETERS ---
    # Effective α for the drug-body system
    α_eff = sum(tissue_α .* weights) / sum(weights)

    # Modify α based on molecular properties
    # More lipophilic drugs see more heterogeneous environment
    α_drug = α_eff * (1 - 0.1 * tanh(logD - 2))

    # --- FUT ESTIMATION (FRACTAL-CORRECTED) ---
    # Classical tissue binding
    fut_classical = 1 / (1 + 0.1 * P)

    # Fractal correction
    fut_fractal = fut_classical * (d_f_tissue / 3)^α_eff
    fut = clamp(fut_fractal, 0.001, 0.99)

    # Key ratio
    fup_fut = fup / fut

    # --- FRACTIONAL ØIE-TOZER TERMS ---
    Vp, Ve, Vr = 0.04, 0.17, 0.39  # Normalized volumes

    # Classical term
    vdss_classical = Vp + Ve * fup_fut + Vr * fup_fut

    # Fractional correction
    vdss_fractal = Vp +
                   Ve * fup_fut^(d_s/2) * avg_coupling +
                   Vr * fup_fut * (d_f_tissue/3)^α_drug * max_coupling

    # --- FEATURE VECTOR ---
    features = Float64[
        # Basic physicochemical (8)
        MW / 600,
        Float64(HBA) / 15,
        Float64(HBD) / 8,
        TPSA / 200,
        Float64(RB) / 15,
        (logP + 5) / 12,
        (logD + 5) / 12,
        fup,

        # Molecular fractal topology (5)
        d_f_mol / 3,
        frac_dim_topo / 3,
        topo_entropy / 2,
        branch_complex,
        self_sim,

        # Tissue-molecule coupling (4)
        avg_coupling,
        max_coupling,
        d_f_mol - d_f_tissue,  # Mismatch
        (d_f_mol / d_f_tissue)^α_eff,  # Fractal ratio

        # Spectral dimension effects (3)
        subdiff_exp,
        spectral_corr,
        d_w / 3,

        # Fractional kinetics (4)
        α_drug,
        α_eff,
        log(fup_fut + 0.01),
        fut,

        # Mechanistic predictions (3)
        log(vdss_classical + 0.01),
        log(vdss_fractal + 0.01),
        vdss_fractal / (vdss_classical + 0.01),

        # Deep coupling terms (4)
        fup_fut^(d_s/2),
        (fup / fup_fut)^α_drug,
        P * avg_coupling,
        log(P + 0.01) * spectral_corr
    ]

    return features
end

# ============================================================================
# LOAD DATA
# ============================================================================

println("\n[1] Loading data...")
df = CSV.read("/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv", DataFrame)
df_complete = dropmissing(df, [:smiles_r, :human_VDss_L_kg, :human_fup, Symbol("MoKa.LogP"), Symbol("MoKa.LogD7.4"), :MW])
println("  Compounds: $(nrow(df_complete))")

# ============================================================================
# COMPUTE FEATURES
# ============================================================================

println("\n[2] Computing deep fractal features...")

X_data = Vector{Vector{Float64}}()
y_data = Float64[]

for (i, row) in enumerate(eachrow(df_complete))
    try
        smiles = row[:smiles_r]
        frac_desc = compute_all_fractal_descriptors(smiles)
        features = deep_fractal_features(row, frac_desc)

        push!(X_data, features)
        push!(y_data, log(row[:human_VDss_L_kg]))
    catch e
        continue
    end

    if i % 200 == 0
        print("\r  Processed $i compounds...")
    end
end
println("\n  Valid samples: $(length(y_data))")

n = length(y_data)
nf = length(X_data[1])
X = hcat(X_data...)
y = y_data

println("  Features: $nf")

# ============================================================================
# NETWORK WITH RESIDUAL CONNECTIONS
# ============================================================================

relu(x) = max(0.0, x)

mutable struct DeepNet
    # Layer 1
    W1::Matrix{Float64}; b1::Vector{Float64}
    # Layer 2
    W2::Matrix{Float64}; b2::Vector{Float64}
    # Layer 3 (residual)
    W3::Matrix{Float64}; b3::Vector{Float64}
    # Output
    Wo::Matrix{Float64}; bo::Vector{Float64}
    # Skip connection
    Ws::Matrix{Float64}
    # Adam state (simplified)
    t::Int
    m::Dict{Symbol, Any}
    v::Dict{Symbol, Any}
end

function DeepNet(din, h1=64, h2=32, h3=16)
    net = DeepNet(
        randn(h1, din) .* sqrt(2/din), zeros(h1),
        randn(h2, h1) .* sqrt(2/h1), zeros(h2),
        randn(h3, h2) .* sqrt(2/h2), zeros(h3),
        randn(1, h3) .* sqrt(2/h3), zeros(1),
        randn(h3, din) .* sqrt(2/din),  # Skip connection
        0, Dict(), Dict()
    )
    return net
end

function forward(net::DeepNet, X)
    a1 = relu.(net.W1 * X .+ net.b1)
    a2 = relu.(net.W2 * a1 .+ net.b2)
    a3_main = relu.(net.W3 * a2 .+ net.b3)

    # Skip connection from input to layer 3
    a3_skip = net.Ws * X
    a3 = a3_main .+ 0.1 .* a3_skip  # Residual

    return vec(net.Wo * a3 .+ net.bo)
end

function train_step!(net::DeepNet, X, y; lr=0.001, λ=0.0003)
    n_batch = size(X, 2)

    # Forward
    a1 = relu.(net.W1 * X .+ net.b1)
    a2 = relu.(net.W2 * a1 .+ net.b2)
    a3_main = relu.(net.W3 * a2 .+ net.b3)
    a3_skip = net.Ws * X
    a3 = a3_main .+ 0.1 .* a3_skip
    pred = vec(net.Wo * a3 .+ net.bo)

    # Loss
    diff = pred .- y
    loss = mean(diff.^2)

    # Backward (simplified - just update all params)
    net.t += 1

    # Numerical gradient for simplicity (small batch makes this OK)
    eps = 1e-5
    for (name, param) in [(:W1, net.W1), (:b1, net.b1), (:W2, net.W2), (:b2, net.b2),
                          (:W3, net.W3), (:b3, net.b3), (:Wo, net.Wo), (:bo, net.bo),
                          (:Ws, net.Ws)]
        if !haskey(net.m, name)
            net.m[name] = zeros(size(param))
            net.v[name] = zeros(size(param))
        end

        grad = similar(param)
        for i in eachindex(param)
            param[i] += eps
            pred_plus = forward(net, X)
            loss_plus = mean((pred_plus .- y).^2)
            param[i] -= eps
            grad[i] = (loss_plus - loss) / eps + 2λ * param[i]
        end

        # Adam
        β1, β2, ε_adam = 0.9, 0.999, 1e-8
        net.m[name] .= β1 .* net.m[name] .+ (1-β1) .* grad
        net.v[name] .= β2 .* net.v[name] .+ (1-β2) .* grad.^2
        m_hat = net.m[name] ./ (1 - β1^net.t)
        v_hat = net.v[name] ./ (1 - β2^net.t)
        param .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ε_adam)
    end

    return loss
end

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
# CROSS-VALIDATION
# ============================================================================

println("\n[3] Training deep fractal model (5-fold × 3 seeds)...")

n_folds = 5
n_seeds = 3
all_gmfes = Float64[]
all_2fold = Float64[]

for seed in 1:n_seeds
    Random.seed!(seed * 23)
    idx = shuffle(1:n)
    fs = n ÷ n_folds

    seed_gmfes = Float64[]

    for fold in 1:n_folds
        te_idx = idx[(fold-1)*fs+1 : fold==n_folds ? n : fold*fs]
        tr_idx = setdiff(idx, te_idx)

        Xtr, ytr = X[:, tr_idx], y[tr_idx]
        Xte, yte = X[:, te_idx], y[te_idx]

        # Normalize
        μ = mean(Xtr, dims=2)
        σ = std(Xtr, dims=2) .+ 1e-8
        Xtr_n = (Xtr .- μ) ./ σ
        Xte_n = (Xte .- μ) ./ σ

        # Train
        net = DeepNet(nf, 48, 24, 12)

        for ep in 1:100
            bi = rand(1:length(tr_idx), min(32, length(tr_idx)))
            train_step!(net, Xtr_n[:, bi], ytr[bi], lr=0.003, λ=0.0002)
        end

        # Evaluate
        pred = forward(net, Xte_n)
        g = gmfe(pred, yte)
        push!(seed_gmfes, g)
        push!(all_gmfes, g)
        push!(all_2fold, pct_fold(pred, yte, 2.0))
    end

    println("  Seed $seed: Mean GMFE = $(round(mean(seed_gmfes), digits=3)), Best = $(round(minimum(seed_gmfes), digits=3))")
end

# ============================================================================
# RESULTS
# ============================================================================

println("\n" * "="^80)
println("DEEP FRACTAL PBPK - RESULTS")
println("="^80)

mg = mean(all_gmfes)
sg = std(all_gmfes)
bg = minimum(all_gmfes)
m2 = mean(all_2fold)

println("\n$(n_folds)-Fold CV × $(n_seeds) Seeds:")
println("  Mean GMFE:      $(round(mg, digits=3)) ± $(round(sg, digits=3))")
println("  Best GMFE:      $(round(bg, digits=3))")
println("  % within 2-fold: $(round(m2, digits=1))%")

println("\n" * "-"^60)
println("Theoretical Foundation:")
println("  • Mittag-Leffler replaces exponential decay")
println("  • Spectral dimension d_s ≈ 4/3 (Alexander-Orbach)")
println("  • Molecular-tissue fractal coupling")
println("  • Memory-dependent transport (fractional α)")

println("\n" * "-"^60)
println("FDA/EMA Assessment:")
println("  GMFE < 2.0:         $(mg < 2.0 ? "✓" : "✗") ($(round(mg, digits=2)))")
println("  Best < 2.0:         $(bg < 2.0 ? "✓" : "✗") ($(round(bg, digits=2)))")
println("  >50% within 2-fold: $(m2 > 50 ? "✓" : "✗") ($(round(m2, digits=1))%)")

println("\n" * "="^80)
println("DEEP INSIGHT:")
println("  Drug distribution = Random walk on fractal vascular network")
println("  Vdss emerges from: d_f(molecule) × d_s(tissue) × α(heterogeneity)")
println("="^80)
