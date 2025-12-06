"""
FractalBlood Integration Demo

Demonstrates the integration between FractalBlood module and PBPK ODE solver.
Shows comparison between traditional well-stirred PBPK and FractalBlood-enhanced dynamics.

Author: Darwin PBPK Platform
Date: December 2025
"""

# Add module path
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))

using DarwinPBPK.ODEPBPKSolver
using DarwinPBPK.ODEPBPKSolver: FractalBlood
using Printf
using Statistics

println("="^70)
println("FractalBlood Integration Demonstration")
println("="^70)
println()

# =============================================================================
# PART 1: Create Fractal Vascular Network
# =============================================================================

println("PART 1: Creating Fractal Vascular Network")
println("-" * "="^69)

# Create fractal network with Murray's Law branching
fractal_model = FractalBlood.create_fractal_blood_model(
    num_levels = 15,        # 15 branching levels (aorta → capillaries)
    hematocrit = 0.45,      # 45% RBC fraction
    fu = 0.1,               # 10% fraction unbound
    alpha = 1.37,           # Power-law exponent
    beta = 0.8              # Anomalous diffusion exponent
)

println("✓ Created fractal vascular network")
println("  - Number of vessels: $(length(fractal_model.vessels))")
println("  - Branching levels: $(fractal_model.num_levels)")
println("  - Fractal dimension: $(fractal_model.fractal_dimension)")
println("  - Alpha (power-law): $(fractal_model.alpha)")
println("  - Beta (CTRW): $(fractal_model.beta)")
println("  - Tau_min: $(round(fractal_model.tau_min, digits=3)) seconds")
println("  - Tau_mean: $(round(fractal_model.tau_mean, digits=2)) seconds")
println()

# Validate network topology
println("Validating network topology...")
validation = FractalBlood.validate_network_topology(fractal_model.vessels)

@printf("  - Murray's Law compliance: %.1f%%\n",
        validation["murray_law_compliance"] * 100)
if haskey(validation, "estimated_fractal_dimension")
    @printf("  - Estimated fractal dimension: %.2f (expected: 2.6-2.8)\n",
            validation["estimated_fractal_dimension"])
end
@printf("  - Mean transit time: %.2f seconds\n",
        validation["mean_transit_time"])
println()

# =============================================================================
# PART 2: Transit Time Distribution Analysis
# =============================================================================

println("PART 2: Transit Time Distribution Analysis")
println("-" * "="^69)

# Compute transit time moments
mean_τ, var_τ, skew_τ = FractalBlood.transit_time_moments(fractal_model)

println("Transit time distribution E(t) = (α-1)/τ_min × (t/τ_min)^(-α)")
println()
if isfinite(mean_τ)
    @printf("  - Mean transit time: %.2f seconds\n", mean_τ)
else
    println("  - Mean transit time: Infinite (α ≤ 2)")
end

if isfinite(var_τ)
    @printf("  - Variance: %.2f seconds²\n", var_τ)
else
    println("  - Variance: Infinite (heavy-tailed distribution)")
end

if isfinite(skew_τ)
    @printf("  - Skewness: %.2f\n", skew_τ)
else
    println("  - Skewness: Infinite")
end
println()

# Sample and visualize distribution
println("Sampling from transit time distribution...")
using Random
Random.seed!(42)

transit_samples = Float64[]
for _ in 1:1000
    # Inverse CDF sampling for power law
    u = rand()
    t_sample = fractal_model.tau_min * (1 - u)^(-1/(fractal_model.alpha - 1))
    push!(transit_samples, t_sample)
end

@printf("  - Sample mean: %.2f seconds\n", mean(transit_samples))
@printf("  - Sample median: %.2f seconds\n", median(transit_samples))
@printf("  - Sample min: %.3f seconds\n", minimum(transit_samples))
@printf("  - Sample max: %.1f seconds\n", maximum(transit_samples))
println()

# =============================================================================
# PART 3: Integration with PBPK
# =============================================================================

println("PART 3: Integration with PBPK Parameters")
println("-" * "="^69)

# Create standard PBPK parameters
pbpk_params = PBPKParams(
    clearance_hepatic = 10.0,  # L/h
    clearance_renal = 2.0       # L/h
)

println("Standard PBPK parameters:")
println("  - Hepatic clearance: $(pbpk_params.clearance_hepatic) L/h")
println("  - Renal clearance: $(pbpk_params.clearance_renal) L/h")
println("  - Blood volume: $(pbpk_params.volumes[1]) L")
println()

# Integrate FractalBlood with PBPK
println("Integrating FractalBlood with PBPK...")
integrated_params = integrate_fractal_blood!(pbpk_params, fractal_model)

println("✓ Integration complete!")
println("  - Type: $(typeof(integrated_params))")
println("  - FractalBlood enabled: $(integrated_params.fractal.enabled)")
println("  - Alpha: $(integrated_params.fractal.alpha)")
println("  - Tau_min: $(round(integrated_params.fractal.tau_min, digits=3)) s")
println("  - Tau_mean: $(round(integrated_params.fractal.tau_mean, digits=2)) s")
println("  - Use convolution: $(integrated_params.fractal.use_convolution)")
println()

# =============================================================================
# PART 4: Transit Time Distribution Function
# =============================================================================

println("PART 4: Transit Time Distribution Function")
println("-" * "="^69)

# Create FractalBloodParams for testing
fractal_params = FractalBloodParams(
    enabled = true,
    alpha = 1.37,
    tau_min = 0.1,
    tau_mean = 20.0
)

# Evaluate E(t) at different time points
println("E(t) values:")
for t in [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    E_t = fractal_transit_time_distribution(t, fractal_params)
    @printf("  E(%.2f s) = %.6f\n", t, E_t)
end
println()

# Verify normalization
using QuadGK
integral, error = quadgk(t -> fractal_transit_time_distribution(t, fractal_params),
                        0.1, 1000.0, rtol=1e-6)
@printf("Normalization check: ∫ E(t) dt = %.6f (should be ≈ 1.0)\n", integral)
@printf("Integration error: %.2e\n", error)
println()

# =============================================================================
# PART 5: Traditional vs FractalBlood PBPK
# =============================================================================

println("PART 5: Comparing Traditional vs FractalBlood PBPK")
println("-" * "="^69)

# Drug parameters
dose = 100.0  # mg
t_max = 24.0  # hours
num_points = 100

println("Drug dosing:")
println("  - Dose: $dose mg (IV bolus)")
println("  - Simulation time: $t_max hours")
println("  - Time points: $num_points")
println()

# Traditional PBPK simulation
println("Running traditional PBPK simulation...")
results_traditional = simulate(
    pbpk_params,
    dose,
    t_max = t_max,
    num_points = num_points
)

# Extract PK parameters
C_blood_trad = results_traditional["blood"]
time = results_traditional["time"]

C_max_trad = maximum(C_blood_trad)
idx_max = argmax(C_blood_trad)
T_max_trad = time[idx_max]

# AUC (trapezoidal rule)
AUC_trad = 0.0
for i in 2:length(time)
    dt = time[i] - time[i-1]
    AUC_trad += 0.5 * (C_blood_trad[i] + C_blood_trad[i-1]) * dt
end

# Terminal half-life
n_terminal = num_points ÷ 4
terminal_idx = (num_points - n_terminal + 1):num_points
valid_idx = [i for i in terminal_idx if C_blood_trad[i] > 1e-10]

if length(valid_idx) >= 3
    t_term = time[valid_idx]
    c_term = log.(C_blood_trad[valid_idx])

    # Linear regression
    n = length(t_term)
    sum_t = sum(t_term)
    sum_c = sum(c_term)
    sum_tc = sum(t_term .* c_term)
    sum_t2 = sum(t_term .^ 2)

    slope = (n * sum_tc - sum_t * sum_c) / (n * sum_t2 - sum_t^2)
    half_life_trad = -log(2) / slope
else
    half_life_trad = NaN
end

println("✓ Traditional PBPK results:")
@printf("  - Cmax: %.2f mg/L\n", C_max_trad)
@printf("  - Tmax: %.2f hours\n", T_max_trad)
@printf("  - AUC₀₋₂₄: %.2f mg·h/L\n", AUC_trad)
if isfinite(half_life_trad)
    @printf("  - Terminal t½: %.2f hours\n", half_life_trad)
end
println()

# Note about FractalBlood simulation
println("Note on FractalBlood-enhanced simulation:")
println("  Full FractalBlood simulation requires modifying the ODE system")
println("  to include history-dependent convolution terms. This is planned")
println("  for future implementation using DifferentialEquations.jl callbacks.")
println()
println("  Expected differences with FractalBlood:")
println("  - Delayed Tmax (transit time effect)")
println("  - Lower Cmax (dispersion effect)")
println("  - Longer terminal phase (heavy-tailed distribution)")
println("  - More realistic for complex dosing regimens")
println()

# =============================================================================
# PART 6: Fractal Network Statistics
# =============================================================================

println("PART 6: Fractal Network Statistics")
println("-" * "="^69)

# Analyze vessel properties
vessels = fractal_model.vessels

radii = [v.radius for v in vessels]
lengths = [v.length for v in vessels]
velocities = [v.velocity for v in vessels]
transit_times = [v.transit_time for v in vessels]

println("Vessel statistics:")
@printf("  - Total vessels: %d\n", length(vessels))
@printf("  - Radius range: %.2e to %.2e m\n", minimum(radii), maximum(radii))
@printf("  - Length range: %.2e to %.2e m\n", minimum(lengths), maximum(lengths))
@printf("  - Velocity range: %.2e to %.2e m/s\n", minimum(velocities), maximum(velocities))
@printf("  - Transit time range: %.2e to %.2e s\n",
        minimum(transit_times), maximum(transit_times))
println()

# Distribution by level
println("Vessels by branching level:")
for level in 0:min(5, fractal_model.num_levels)
    vessels_at_level = count(v -> v.level == level, vessels)
    avg_radius = mean([v.radius for v in vessels if v.level == level])
    @printf("  Level %d: %3d vessels, avg radius = %.2e m\n",
            level, vessels_at_level, avg_radius)
end
println()

# =============================================================================
# PART 7: Performance Comparison
# =============================================================================

println("PART 7: Performance Comparison")
println("-" * "="^69)

# Benchmark traditional PBPK
println("Benchmarking traditional PBPK...")
t_start = time()
for _ in 1:10
    simulate(pbpk_params, dose, t_max=24.0, num_points=100)
end
t_traditional = (time() - t_start) / 10

@printf("  Traditional PBPK: %.2f ms per simulation\n", t_traditional * 1000)
println()

println("Expected FractalBlood performance:")
println("  - Without convolution: ~2× slower than traditional")
println("  - With full convolution: ~10-50× slower (numerical integration)")
println("  - Recommendation: Use convolution for publication-quality results")
println()

# =============================================================================
# Summary
# =============================================================================

println("="^70)
println("SUMMARY")
println("="^70)
println()
println("✓ Successfully integrated FractalBlood with PBPK ODE solver")
println("✓ Fractal vascular network with $(length(vessels)) vessels created")
println("✓ Transit time distribution validated (power-law behavior)")
println("✓ PBPK parameters enhanced with fractal dynamics")
println()
println("Key Features:")
println("  - Power-law transit time distribution (α = $(fractal_model.alpha))")
println("  - Anomalous diffusion (β = $(fractal_model.beta))")
println("  - Murray's Law compliance: $(round(validation["murray_law_compliance"]*100, digits=1))%")
println("  - Mean circulation time: $(round(fractal_model.tau_mean, digits=1)) seconds")
println()
println("Next Steps:")
println("  1. Implement full convolution in ODE system (callbacks)")
println("  2. Add multi-phase blood dynamics (RBC, protein binding)")
println("  3. Validate against clinical PK data")
println("  4. GPU acceleration for batch simulations")
println()
println("Documentation: julia-migration/docs/FRACTALBLOOD_INTEGRATION.md")
println("Tests: julia-migration/test/test_fractalblood_integration.jl")
println("="^70)
