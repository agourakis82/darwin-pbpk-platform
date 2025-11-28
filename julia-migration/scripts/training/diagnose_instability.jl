"""
DIAGNOSE TRAINING INSTABILITY

Some folds explode to GMFE > 10 while others achieve GMFE ~1.0.
Let's understand why.

Hypotheses:
1. Outliers in the data
2. Feature scaling issues
3. Gradient explosion
4. Bad initialization
"""

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Random
using Printf

println("="^80)
println("DIAGNOSING TRAINING INSTABILITY")
println("="^80)

# ============================================================================
# LOAD DATA
# ============================================================================

println("\n[1] Loading and analyzing data...")
df = CSV.read("/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv", DataFrame)
df = dropmissing(df, [:smiles_r, :human_VDss_L_kg, :human_fup,
                      Symbol("MoKa.LogP"), Symbol("MoKa.LogD7.4"), :MW])

vdss_values = df.human_VDss_L_kg
log_vdss = log.(vdss_values)

println("\n  Vdss distribution:")
println("    Min: $(round(minimum(vdss_values), digits=3)) L/kg")
println("    Max: $(round(maximum(vdss_values), digits=3)) L/kg")
println("    Median: $(round(median(vdss_values), digits=3)) L/kg")
println("    Mean: $(round(mean(vdss_values), digits=3)) L/kg")

println("\n  Log(Vdss) distribution:")
println("    Min: $(round(minimum(log_vdss), digits=2))")
println("    Max: $(round(maximum(log_vdss), digits=2))")
println("    Std: $(round(std(log_vdss), digits=2))")

# Find extreme values
extreme_low = vdss_values .< 0.05
extreme_high = vdss_values .> 50
println("\n  Extreme values:")
println("    Vdss < 0.05 L/kg: $(sum(extreme_low)) compounds")
println("    Vdss > 50 L/kg: $(sum(extreme_high)) compounds")

# ============================================================================
# HYPOTHESIS 1: Outliers cause instability
# ============================================================================

println("\n" * "="^80)
println("[2] Testing: Do outliers cause instability?")
println("="^80)

# Simple linear regression to find high-leverage points
function simple_features(row)
    fup = row[:human_fup]
    logP = row[Symbol("MoKa.LogP")]
    logD = row[Symbol("MoKa.LogD7.4")]
    MW = row[:MW]
    Float64[1.0, log10(MW), logP, logD, log10(fup + 0.001)]
end

X_simple = hcat([simple_features(row) for row in eachrow(df)]...)
y_simple = log_vdss

# OLS fit
β = X_simple' \ y_simple
pred = vec(X_simple' * β)
residuals = y_simple .- pred

println("\n  Simple linear model (log(Vdss) ~ logMW + logP + logD + log(fup)):")
println("    R² = $(round(1 - var(residuals)/var(y_simple), digits=3))")
println("    RMSE = $(round(sqrt(mean(residuals.^2)), digits=3))")

# Find high-residual compounds
high_res_idx = findall(abs.(residuals) .> 2.0)
println("\n  High residual compounds (|residual| > 2.0): $(length(high_res_idx))")

if length(high_res_idx) > 0
    println("    These are likely causing training instability:")
    for i in high_res_idx[1:min(5, length(high_res_idx))]
        println("      - Compound $i: Vdss=$(round(vdss_values[i], digits=2)), residual=$(round(residuals[i], digits=2))")
    end
end

# ============================================================================
# HYPOTHESIS 2: Train with and without outliers
# ============================================================================

println("\n" * "="^80)
println("[3] Testing: Does removing outliers fix instability?")
println("="^80)

# Neural network
relu(x) = max(0.0, x)

mutable struct SimpleNet
    W1::Matrix{Float64}; b1::Vector{Float64}
    W2::Matrix{Float64}; b2::Vector{Float64}
    W3::Vector{Float64}; b3::Float64
end

function SimpleNet(din)
    SimpleNet(
        randn(32, din) .* sqrt(2/din), zeros(32),
        randn(16, 32) .* sqrt(2/32), zeros(16),
        randn(16) .* sqrt(1/16), 0.0
    )
end

function forward(net::SimpleNet, X)
    a1 = relu.(net.W1 * X .+ net.b1)
    a2 = relu.(net.W2 * a1 .+ net.b2)
    return vec(net.W3' * a2 .+ net.b3)
end

function train!(net::SimpleNet, X, y; epochs=300, lr=0.01)
    n = size(X, 2)
    for ep in 1:epochs
        # Forward
        z1 = net.W1 * X .+ net.b1
        a1 = relu.(z1)
        z2 = net.W2 * a1 .+ net.b2
        a2 = relu.(z2)
        pred = vec(net.W3' * a2 .+ net.b3)

        # Backward (simplified)
        diff = (pred .- y) ./ n

        # Output layer
        db3 = sum(diff)
        dW3 = a2 * diff

        # Hidden 2
        d2 = (net.W3 .* diff') .* (z2 .> 0)
        db2 = vec(sum(d2, dims=2))
        dW2 = d2 * a1'

        # Hidden 1
        d1 = (net.W2' * d2) .* (z1 .> 0)
        db1 = vec(sum(d1, dims=2))
        dW1 = d1 * X'

        # Clip gradients
        for g in [dW1, db1, dW2, db2, dW3]
            gn = norm(g)
            gn > 1.0 && (g .*= 1.0 / gn)
        end

        # Update
        net.W1 .-= lr .* dW1
        net.b1 .-= lr .* db1
        net.W2 .-= lr .* dW2
        net.b2 .-= lr .* db2
        net.W3 .-= lr .* dW3
        net.b3 -= lr * db3
    end
end

function gmfe(pred, obs)
    p, o = exp.(pred), exp.(obs)
    fe = max.(p./o, o./p)
    exp(mean(log.(fe)))
end

# Prepare features
function full_features(row)
    fup = row[:human_fup]
    logP = row[Symbol("MoKa.LogP")]
    logD = row[Symbol("MoKa.LogD7.4")]
    MW = row[:MW]
    TPSA = row[:TPSA_NO]
    HBD = Float64(row[:HBD])
    HBA = Float64(row[:HBA])
    RB = Float64(row[:RotBondCount])

    Float64[MW/500, HBA/12, HBD/6, TPSA/150, RB/12,
            (logP+3)/10, (logD+3)/10, fup, log10(fup+0.001)/3+1]
end

X_full = hcat([full_features(row) for row in eachrow(df)]...)
y_full = log_vdss
n_full = size(X_full, 2)

# Define outliers as |residual| > 2.0
inlier_mask = abs.(residuals) .<= 2.0
X_clean = X_full[:, inlier_mask]
y_clean = y_full[inlier_mask]
n_clean = size(X_clean, 2)

println("\n  Full dataset: $n_full compounds")
println("  Clean dataset (removing $(n_full - n_clean) outliers): $n_clean compounds")

# Compare training stability
n_trials = 20
full_gmfes = Float64[]
clean_gmfes = Float64[]

println("\n  Running $n_trials trials...")

for trial in 1:n_trials
    Random.seed!(trial)

    # Split 80/20
    idx_full = shuffle(1:n_full)
    tr_full = idx_full[1:floor(Int, 0.8*n_full)]
    te_full = idx_full[floor(Int, 0.8*n_full)+1:end]

    idx_clean = shuffle(1:n_clean)
    tr_clean = idx_clean[1:floor(Int, 0.8*n_clean)]
    te_clean = idx_clean[floor(Int, 0.8*n_clean)+1:end]

    # Normalize
    μ_full = mean(X_full[:, tr_full], dims=2)
    σ_full = std(X_full[:, tr_full], dims=2) .+ 1e-8

    μ_clean = mean(X_clean[:, tr_clean], dims=2)
    σ_clean = std(X_clean[:, tr_clean], dims=2) .+ 1e-8

    # Train on full
    net_full = SimpleNet(9)
    Xtr_f = (X_full[:, tr_full] .- μ_full) ./ σ_full
    Xte_f = (X_full[:, te_full] .- μ_full) ./ σ_full
    train!(net_full, Xtr_f, y_full[tr_full])
    g_full = gmfe(forward(net_full, Xte_f), y_full[te_full])
    push!(full_gmfes, g_full)

    # Train on clean
    net_clean = SimpleNet(9)
    Xtr_c = (X_clean[:, tr_clean] .- μ_clean) ./ σ_clean
    Xte_c = (X_clean[:, te_clean] .- μ_clean) ./ σ_clean
    train!(net_clean, Xtr_c, y_clean[tr_clean])
    g_clean = gmfe(forward(net_clean, Xte_c), y_clean[te_clean])
    push!(clean_gmfes, g_clean)
end

println("\n  FULL DATASET:")
println("    Mean GMFE: $(round(mean(full_gmfes), digits=3)) ± $(round(std(full_gmfes), digits=3))")
println("    Min: $(round(minimum(full_gmfes), digits=3)), Max: $(round(maximum(full_gmfes), digits=3))")
println("    Exploded (>5): $(sum(full_gmfes .> 5))/$(n_trials)")

println("\n  CLEAN DATASET (outliers removed):")
println("    Mean GMFE: $(round(mean(clean_gmfes), digits=3)) ± $(round(std(clean_gmfes), digits=3))")
println("    Min: $(round(minimum(clean_gmfes), digits=3)), Max: $(round(maximum(clean_gmfes), digits=3))")
println("    Exploded (>5): $(sum(clean_gmfes .> 5))/$(n_trials)")

# ============================================================================
# HYPOTHESIS 3: Robust loss function
# ============================================================================

println("\n" * "="^80)
println("[4] Testing: Does Huber loss improve stability?")
println("="^80)

function train_huber!(net::SimpleNet, X, y; epochs=300, lr=0.01, δ=1.0)
    n = size(X, 2)
    for ep in 1:epochs
        z1 = net.W1 * X .+ net.b1
        a1 = relu.(z1)
        z2 = net.W2 * a1 .+ net.b2
        a2 = relu.(z2)
        pred = vec(net.W3' * a2 .+ net.b3)

        # Huber loss gradient
        diff = pred .- y
        huber_grad = [abs(d) <= δ ? d : δ * sign(d) for d in diff] ./ n

        # Backward
        db3 = sum(huber_grad)
        dW3 = a2 * huber_grad
        d2 = (net.W3 .* huber_grad') .* (z2 .> 0)
        db2 = vec(sum(d2, dims=2))
        dW2 = d2 * a1'
        d1 = (net.W2' * d2) .* (z1 .> 0)
        db1 = vec(sum(d1, dims=2))
        dW1 = d1 * X'

        for g in [dW1, db1, dW2, db2, dW3]
            gn = norm(g)
            gn > 1.0 && (g .*= 1.0 / gn)
        end

        net.W1 .-= lr .* dW1
        net.b1 .-= lr .* db1
        net.W2 .-= lr .* dW2
        net.b2 .-= lr .* db2
        net.W3 .-= lr .* dW3
        net.b3 -= lr * db3
    end
end

huber_gmfes = Float64[]

for trial in 1:n_trials
    Random.seed!(trial)
    idx = shuffle(1:n_full)
    tr = idx[1:floor(Int, 0.8*n_full)]
    te = idx[floor(Int, 0.8*n_full)+1:end]

    μ = mean(X_full[:, tr], dims=2)
    σ = std(X_full[:, tr], dims=2) .+ 1e-8

    net = SimpleNet(9)
    Xtr = (X_full[:, tr] .- μ) ./ σ
    Xte = (X_full[:, te] .- μ) ./ σ
    train_huber!(net, Xtr, y_full[tr], δ=1.0)
    push!(huber_gmfes, gmfe(forward(net, Xte), y_full[te]))
end

println("\n  HUBER LOSS (δ=1.0):")
println("    Mean GMFE: $(round(mean(huber_gmfes), digits=3)) ± $(round(std(huber_gmfes), digits=3))")
println("    Min: $(round(minimum(huber_gmfes), digits=3)), Max: $(round(maximum(huber_gmfes), digits=3))")
println("    Exploded (>5): $(sum(huber_gmfes .> 5))/$(n_trials)")

# ============================================================================
# CONCLUSION
# ============================================================================

println("\n" * "="^80)
println("DIAGNOSIS SUMMARY")
println("="^80)

println("\n1. DATA QUALITY:")
println("   - $(length(high_res_idx)) compounds have high residuals (likely mislabeled or unusual)")
println("   - Range of Vdss spans $(round(maximum(vdss_values)/minimum(vdss_values), digits=0))× (huge dynamic range)")

println("\n2. REMOVING OUTLIERS:")
full_stable = sum(full_gmfes .< 3) / n_trials * 100
clean_stable = sum(clean_gmfes .< 3) / n_trials * 100
println("   - Full dataset: $(round(full_stable, digits=0))% stable runs")
println("   - Clean dataset: $(round(clean_stable, digits=0))% stable runs")
if clean_stable > full_stable + 10
    println("   → Removing outliers HELPS stability")
else
    println("   → Removing outliers does NOT significantly help")
end

println("\n3. HUBER LOSS:")
huber_stable = sum(huber_gmfes .< 3) / n_trials * 100
println("   - Huber loss: $(round(huber_stable, digits=0))% stable runs")
if huber_stable > full_stable + 10
    println("   → Huber loss HELPS stability")
else
    println("   → Huber loss does NOT significantly help")
end

println("\n" * "="^80)
best_approach = argmax([full_stable, clean_stable, huber_stable])
approaches = ["Full dataset + MSE", "Clean dataset + MSE", "Full dataset + Huber"]
println("RECOMMENDATION: Use $(approaches[best_approach])")
println("="^80)
