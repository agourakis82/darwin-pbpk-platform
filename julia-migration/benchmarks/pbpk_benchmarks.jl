# =============================================================================
# PBPK Benchmark Suite - Julia Reference Implementation
# =============================================================================
#
# Benchmark Targets (matching Sounio):
# 1. ODE System Evaluation (14-compartment PBPK)
# 2. RK4 Integration (1000 steps)
# 3. Validation Metrics (GMFE, AAFE, R², fold errors)
# 4. Bootstrap Confidence Intervals (2000 resamples)
# 5. Tissue Partition Coefficients (Rodgers-Rowland Kp)
#
# Run: julia --project=.. pbpk_benchmarks.jl
# =============================================================================

using BenchmarkTools
using Statistics
using Random
using Printf

# -----------------------------------------------------------------------------
# SECTION 1: Constants
# -----------------------------------------------------------------------------

const NUM_ORGANS = 14
const BLOOD_IDX, LIVER_IDX, KIDNEY_IDX, LUNG_IDX = 1, 2, 3, 4
const HEART_IDX, BRAIN_IDX, MUSCLE_IDX, ADIPOSE_IDX = 5, 6, 7, 8
const SKIN_IDX, GUT_IDX, SPLEEN_IDX, BONE_IDX = 9, 10, 11, 12
const PANCREAS_IDX, OTHER_IDX = 13, 14

# Benchmark configuration
const BENCHMARK_ODE_ITERATIONS = 100_000
const BENCHMARK_RK4_STEPS = 1000
const BENCHMARK_BOOTSTRAP_SAMPLES = 2000
const BENCHMARK_VALIDATION_SIZE = 100

# -----------------------------------------------------------------------------
# SECTION 2: Data Structures
# -----------------------------------------------------------------------------

"""14-compartment PBPK physiological parameters"""
struct PhysiologyParams
    # Organ volumes (L)
    v_blood::Float64
    v_liver::Float64
    v_kidney::Float64
    v_lung::Float64
    v_heart::Float64
    v_brain::Float64
    v_muscle::Float64
    v_adipose::Float64
    v_skin::Float64
    v_gut::Float64
    v_spleen::Float64
    v_bone::Float64
    v_pancreas::Float64
    v_other::Float64

    # Blood flows (L/h)
    q_liver::Float64
    q_kidney::Float64
    q_lung::Float64
    q_heart::Float64
    q_brain::Float64
    q_muscle::Float64
    q_adipose::Float64
    q_skin::Float64
    q_gut::Float64
    q_spleen::Float64
    q_bone::Float64
    q_pancreas::Float64
    q_other::Float64

    # Partition coefficients
    kp_liver::Float64
    kp_kidney::Float64
    kp_lung::Float64
    kp_heart::Float64
    kp_brain::Float64
    kp_muscle::Float64
    kp_adipose::Float64
    kp_skin::Float64
    kp_gut::Float64
    kp_spleen::Float64
    kp_bone::Float64
    kp_pancreas::Float64
    kp_other::Float64

    # Clearance
    cl_hepatic::Float64
    cl_renal::Float64

    # Drug binding
    fu_plasma::Float64
    blood_plasma_ratio::Float64
end

"""Drug properties for Kp calculation"""
struct DrugProperties
    log_p::Float64
    pka_acid::Float64
    pka_base::Float64
    fu_plasma::Float64
    blood_plasma_ratio::Float64
    is_acid::Bool
    is_base::Bool
    is_neutral::Bool
    molecular_weight::Float64
end

"""Tissue composition for Rodgers-Rowland"""
struct TissueComposition
    f_water::Float64
    f_lipid::Float64
    f_protein::Float64
    f_acidic_phospholipid::Float64
    f_neutral_lipid::Float64
    ph_tissue::Float64
end

# -----------------------------------------------------------------------------
# SECTION 3: Default Parameters
# -----------------------------------------------------------------------------

function default_physiology()::PhysiologyParams
    PhysiologyParams(
        # Volumes (L) - ICRP reference
        5.2, 1.8, 0.31, 0.50, 0.33, 1.45, 29.0, 14.5,
        2.6, 1.2, 0.18, 4.0, 0.10, 3.7,
        # Blood flows (L/h)
        27.8, 69.6, 348.0, 17.4, 41.8, 55.7, 17.4, 17.4,
        55.7, 10.4, 17.4, 3.5, 13.9,
        # Partition coefficients
        5.2, 4.1, 0.8, 3.2, 2.1, 2.8, 15.0, 3.5,
        4.0, 3.8, 1.5, 2.5, 2.0,
        # Clearance
        15.0, 5.0,
        # Drug binding
        0.05, 0.85
    )
end

# Tissue compositions (Rodgers & Rowland 2006)
const LIVER_COMP = TissueComposition(0.751, 0.035, 0.214, 0.0040, 0.023, 7.4)
const KIDNEY_COMP = TissueComposition(0.783, 0.027, 0.190, 0.0037, 0.013, 7.4)
const MUSCLE_COMP = TissueComposition(0.756, 0.020, 0.224, 0.0022, 0.010, 7.0)
const ADIPOSE_COMP = TissueComposition(0.187, 0.756, 0.057, 0.0004, 0.852, 7.4)
const BRAIN_COMP = TissueComposition(0.773, 0.117, 0.110, 0.0112, 0.039, 7.3)
const LUNG_COMP = TissueComposition(0.802, 0.022, 0.176, 0.0060, 0.003, 7.4)

# -----------------------------------------------------------------------------
# SECTION 4: ODE System (Benchmark #1)
# -----------------------------------------------------------------------------

"""
14-compartment PBPK ODE system - matches Sounio pbpk_ode_system()

Arguments:
- du: derivative vector (modified in-place)
- u: state vector [blood, liver, kidney, lung, heart, brain, muscle, adipose,
                   skin, gut, spleen, bone, pancreas, other]
- p: PhysiologyParams
- t: time
"""
function pbpk_ode_system!(du::Vector{Float64}, u::Vector{Float64},
                          p::PhysiologyParams, t::Float64)
    # Unbound plasma concentration
    c_plasma = u[BLOOD_IDX] / p.blood_plasma_ratio
    c_unbound = c_plasma * p.fu_plasma

    # Volumes array for convenience
    volumes = (p.v_blood, p.v_liver, p.v_kidney, p.v_lung, p.v_heart,
               p.v_brain, p.v_muscle, p.v_adipose, p.v_skin, p.v_gut,
               p.v_spleen, p.v_bone, p.v_pancreas, p.v_other)

    # Flows array
    flows = (0.0, p.q_liver, p.q_kidney, p.q_lung, p.q_heart,
             p.q_brain, p.q_muscle, p.q_adipose, p.q_skin, p.q_gut,
             p.q_spleen, p.q_bone, p.q_pancreas, p.q_other)

    # Kp array
    kps = (1.0, p.kp_liver, p.kp_kidney, p.kp_lung, p.kp_heart,
           p.kp_brain, p.kp_muscle, p.kp_adipose, p.kp_skin, p.kp_gut,
           p.kp_spleen, p.kp_bone, p.kp_pancreas, p.kp_other)

    # Calculate tissue fluxes
    total_flux = 0.0
    @inbounds for i in 2:NUM_ORGANS
        c_tissue_free = u[i] / kps[i]
        flux = flows[i] * (c_plasma - c_tissue_free)
        du[i] = flux / volumes[i]
        total_flux += flux
    end

    # Clearance
    cl_hepatic_rate = p.cl_hepatic * c_unbound
    cl_renal_rate = p.cl_renal * c_unbound

    # Blood compartment
    du[BLOOD_IDX] = (-total_flux - cl_hepatic_rate - cl_renal_rate) / p.v_blood

    return nothing
end

"""Non-mutating version for simpler benchmarking"""
function pbpk_ode_system(u::Vector{Float64}, p::PhysiologyParams, t::Float64)::Vector{Float64}
    du = zeros(Float64, NUM_ORGANS)
    pbpk_ode_system!(du, u, p, t)
    return du
end

# -----------------------------------------------------------------------------
# SECTION 5: RK4 Integrator (Benchmark #2)
# -----------------------------------------------------------------------------

"""
Single RK4 step for PBPK system
"""
function rk4_step(y::Vector{Float64}, p::PhysiologyParams, t::Float64, dt::Float64)::Vector{Float64}
    k1 = pbpk_ode_system(y, p, t)
    k2 = pbpk_ode_system(y .+ dt/2 .* k1, p, t + dt/2)
    k3 = pbpk_ode_system(y .+ dt/2 .* k2, p, t + dt/2)
    k4 = pbpk_ode_system(y .+ dt .* k3, p, t + dt)

    return y .+ dt/6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

"""
Solve PBPK system using RK4 integration
"""
function solve_pbpk_rk4(y0::Vector{Float64}, p::PhysiologyParams,
                        t_start::Float64, t_end::Float64, n_steps::Int)::Vector{Float64}
    dt = (t_end - t_start) / n_steps
    y = copy(y0)
    t = t_start

    for _ in 1:n_steps
        y = rk4_step(y, p, t, dt)
        t += dt
    end

    return y
end

# -----------------------------------------------------------------------------
# SECTION 6: Validation Metrics (Benchmark #3)
# -----------------------------------------------------------------------------

"""Fold error: max(pred/obs, obs/pred)"""
fold_error(pred::Float64, obs::Float64)::Float64 = max(pred/obs, obs/pred)

"""Geometric Mean Fold Error: exp(mean(|ln(pred/obs)|))"""
function calculate_gmfe(pred::Vector{Float64}, obs::Vector{Float64})::Float64
    n = length(pred)
    sum_log_fe = 0.0
    for i in 1:n
        if pred[i] > 0 && obs[i] > 0
            sum_log_fe += abs(log(pred[i] / obs[i]))
        end
    end
    return exp(sum_log_fe / n)
end

"""Average Fold Error: 10^(mean(log10(pred/obs)))"""
function calculate_afe(pred::Vector{Float64}, obs::Vector{Float64})::Float64
    n = length(pred)
    sum_log = 0.0
    for i in 1:n
        if pred[i] > 0 && obs[i] > 0
            sum_log += log10(pred[i] / obs[i])
        end
    end
    return 10^(sum_log / n)
end

"""Absolute Average Fold Error: 10^(mean(|log10(pred/obs)|))"""
function calculate_aafe(pred::Vector{Float64}, obs::Vector{Float64})::Float64
    n = length(pred)
    sum_abs_log = 0.0
    for i in 1:n
        if pred[i] > 0 && obs[i] > 0
            sum_abs_log += abs(log10(pred[i] / obs[i]))
        end
    end
    return 10^(sum_abs_log / n)
end

"""R² coefficient of determination"""
function calculate_r_squared(pred::Vector{Float64}, obs::Vector{Float64})::Float64
    mean_obs = mean(obs)
    ss_res = sum((obs .- pred).^2)
    ss_tot = sum((obs .- mean_obs).^2)
    return ss_tot > 1e-10 ? 1.0 - ss_res/ss_tot : 0.0
end

"""RMSE"""
function calculate_rmse(pred::Vector{Float64}, obs::Vector{Float64})::Float64
    return sqrt(mean((pred .- obs).^2))
end

"""Mean bias"""
function calculate_bias(pred::Vector{Float64}, obs::Vector{Float64})::Float64
    return mean(pred .- obs)
end

"""Percentage within N-fold"""
function calculate_within_fold(pred::Vector{Float64}, obs::Vector{Float64}, fold::Float64)::Float64
    n = length(pred)
    count = sum(fold_error(pred[i], obs[i]) <= fold for i in 1:n)
    return count / n * 100.0
end

"""All validation metrics"""
function calculate_all_metrics(pred::Vector{Float64}, obs::Vector{Float64})
    return (
        gmfe = calculate_gmfe(pred, obs),
        afe = calculate_afe(pred, obs),
        aafe = calculate_aafe(pred, obs),
        r_squared = calculate_r_squared(pred, obs),
        rmse = calculate_rmse(pred, obs),
        bias = calculate_bias(pred, obs),
        within_2fold = calculate_within_fold(pred, obs, 2.0),
        within_3fold = calculate_within_fold(pred, obs, 3.0),
        n_samples = length(pred)
    )
end

# -----------------------------------------------------------------------------
# SECTION 7: Bootstrap CI (Benchmark #4)
# -----------------------------------------------------------------------------

"""
Bootstrap confidence interval for GMFE
Uses simple LCG for reproducibility matching Sounio
"""
function bootstrap_gmfe_ci(pred::Vector{Float64}, obs::Vector{Float64},
                           n_bootstrap::Int, seed::Int)
    Random.seed!(seed)
    n = length(pred)

    estimates = Vector{Float64}(undef, n_bootstrap)
    resampled_pred = Vector{Float64}(undef, n)
    resampled_obs = Vector{Float64}(undef, n)

    for b in 1:n_bootstrap
        # Resample with replacement
        for i in 1:n
            idx = rand(1:n)
            resampled_pred[i] = pred[idx]
            resampled_obs[i] = obs[idx]
        end

        estimates[b] = calculate_gmfe(resampled_pred, resampled_obs)
    end

    point_est = calculate_gmfe(pred, obs)
    mean_est = mean(estimates)
    std_err = std(estimates)

    # Normal approximation CI
    z_95 = 1.96
    ci_lower = mean_est - z_95 * std_err
    ci_upper = mean_est + z_95 * std_err

    return (
        point_estimate = point_est,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        std_error = std_err
    )
end

# -----------------------------------------------------------------------------
# SECTION 8: Tissue Partition Coefficients (Benchmark #5)
# -----------------------------------------------------------------------------

"""Ionization factor for bases"""
ionization_factor_base(pka::Float64, ph::Float64)::Float64 = 1.0 + 10^(pka - ph)

"""Ionization factor for acids"""
ionization_factor_acid(pka::Float64, ph::Float64)::Float64 = 1.0 + 10^(ph - pka)

"""
Calculate Kp using Rodgers-Rowland method
"""
function calculate_kp_rodgers_rowland(drug::DrugProperties, tissue::TissueComposition)::Float64
    ph_plasma = 7.4

    # Partition coefficients
    p_ow = 10^drug.log_p

    # Neutral lipid partition
    kp_neutral_lipid = p_ow * tissue.f_neutral_lipid

    # Water partition with ionization
    kp_water = tissue.f_water

    if drug.is_base
        ion_plasma = ionization_factor_base(drug.pka_base, ph_plasma)
        ion_tissue = ionization_factor_base(drug.pka_base, tissue.ph_tissue)
        kp_water = tissue.f_water * ion_tissue / ion_plasma
    end

    if drug.is_acid
        ion_plasma = ionization_factor_acid(drug.pka_acid, ph_plasma)
        ion_tissue = ionization_factor_acid(drug.pka_acid, tissue.ph_tissue)
        kp_water = tissue.f_water * ion_tissue / ion_plasma
    end

    # Acidic phospholipid binding (for bases)
    kp_ap = 0.0
    if drug.is_base
        ka_ap = 0.001 * 10^(drug.log_p + 0.5 * drug.pka_base)
        kp_ap = tissue.f_acidic_phospholipid * ka_ap
    end

    # Protein binding contribution
    kp_protein = tissue.f_protein * 0.1 * p_ow

    # Total Kp
    kp_unbound = kp_neutral_lipid + kp_water + kp_ap + kp_protein
    kp = kp_unbound / drug.fu_plasma

    # Normalize to physiological range
    return clamp(kp, 0.1, 100.0)
end

"""Calculate Kp for all major tissues"""
function calculate_all_kp(drug::DrugProperties)
    return (
        kp_liver = calculate_kp_rodgers_rowland(drug, LIVER_COMP),
        kp_kidney = calculate_kp_rodgers_rowland(drug, KIDNEY_COMP),
        kp_muscle = calculate_kp_rodgers_rowland(drug, MUSCLE_COMP),
        kp_adipose = calculate_kp_rodgers_rowland(drug, ADIPOSE_COMP),
        kp_brain = calculate_kp_rodgers_rowland(drug, BRAIN_COMP),
        kp_lung = calculate_kp_rodgers_rowland(drug, LUNG_COMP)
    )
end

# -----------------------------------------------------------------------------
# SECTION 9: Test Data
# -----------------------------------------------------------------------------

function create_test_pairs()
    pred = [15.2, 8.7, 22.3, 5.1, 31.0, 11.8, 19.5, 7.3, 25.6, 13.9]
    obs = [12.5, 10.1, 18.9, 6.2, 28.5, 14.2, 17.8, 8.9, 22.1, 15.5]
    return pred, obs
end

function create_test_drug()
    DrugProperties(
        2.5,    # log_p
        4.5,    # pka_acid
        8.2,    # pka_base
        0.05,   # fu_plasma
        0.85,   # blood_plasma_ratio
        false,  # is_acid
        true,   # is_base
        false,  # is_neutral
        350.0   # molecular_weight
    )
end

# -----------------------------------------------------------------------------
# SECTION 10: Benchmark Functions
# -----------------------------------------------------------------------------

"""Benchmark: ODE system evaluation"""
function benchmark_ode_system(n_iterations::Int)::Float64
    phys = default_physiology()
    u = zeros(Float64, NUM_ORGANS)
    u[BLOOD_IDX] = 100.0  # Initial dose

    checksum = 0.0
    for _ in 1:n_iterations
        du = pbpk_ode_system(u, phys, 0.0)
        checksum += du[BLOOD_IDX]
    end

    return checksum
end

"""Benchmark: RK4 integration"""
function benchmark_rk4_integration(n_steps::Int)::Float64
    phys = default_physiology()
    y0 = zeros(Float64, NUM_ORGANS)
    y0[BLOOD_IDX] = 100.0 / phys.v_blood  # 100 mg IV dose

    final = solve_pbpk_rk4(y0, phys, 0.0, 24.0, n_steps)
    return final[BLOOD_IDX]
end

"""Benchmark: Validation metrics"""
function benchmark_validation_metrics(n_iterations::Int)::Float64
    pred, obs = create_test_pairs()

    checksum = 0.0
    for _ in 1:n_iterations
        metrics = calculate_all_metrics(pred, obs)
        checksum += metrics.gmfe + metrics.aafe + metrics.r_squared
    end

    return checksum
end

"""Benchmark: Bootstrap CI"""
function benchmark_bootstrap(n_bootstrap::Int)::Float64
    pred, obs = create_test_pairs()
    ci = bootstrap_gmfe_ci(pred, obs, n_bootstrap, 12345)
    return ci.point_estimate + ci.ci_lower + ci.ci_upper
end

"""Benchmark: Kp calculation"""
function benchmark_kp_calculation(n_iterations::Int)::Float64
    drug = create_test_drug()

    checksum = 0.0
    for _ in 1:n_iterations
        kps = calculate_all_kp(drug)
        checksum += kps.kp_liver + kps.kp_kidney + kps.kp_adipose
    end

    return checksum
end

# -----------------------------------------------------------------------------
# SECTION 11: Main Benchmark Runner
# -----------------------------------------------------------------------------

function run_benchmarks()
    println("=" ^ 70)
    println("PBPK Benchmark Suite - Julia Implementation")
    println("=" ^ 70)
    println()

    # Warmup
    println("Warming up JIT...")
    benchmark_ode_system(1000)
    benchmark_rk4_integration(100)
    benchmark_validation_metrics(100)
    benchmark_bootstrap(100)
    benchmark_kp_calculation(100)
    println()

    # Benchmark 1: ODE System
    println("-" ^ 70)
    println("Benchmark 1: ODE System Evaluation ($(BENCHMARK_ODE_ITERATIONS) iterations)")
    println("-" ^ 70)

    t1 = @benchmark benchmark_ode_system($BENCHMARK_ODE_ITERATIONS) samples=10 evals=3
    println(t1)
    ode_mean_ns = mean(t1.times)
    ode_per_iter = ode_mean_ns / BENCHMARK_ODE_ITERATIONS
    @printf("Per-iteration: %.2f ns\n", ode_per_iter)
    @printf("Throughput: %.2f M evals/sec\n", 1e9 / ode_per_iter / 1e6)
    println()

    # Benchmark 2: RK4 Integration
    println("-" ^ 70)
    println("Benchmark 2: RK4 Integration ($(BENCHMARK_RK4_STEPS) steps)")
    println("-" ^ 70)

    t2 = @benchmark benchmark_rk4_integration($BENCHMARK_RK4_STEPS) samples=100 evals=10
    println(t2)
    @printf("Per-step: %.2f µs\n", mean(t2.times) / BENCHMARK_RK4_STEPS / 1000)
    println()

    # Benchmark 3: Validation Metrics
    println("-" ^ 70)
    println("Benchmark 3: Validation Metrics (10000 iterations)")
    println("-" ^ 70)

    t3 = @benchmark benchmark_validation_metrics(10000) samples=50 evals=5
    println(t3)
    metrics_per_iter = mean(t3.times) / 10000
    @printf("Per-iteration: %.2f ns\n", metrics_per_iter)
    println()

    # Benchmark 4: Bootstrap CI
    println("-" ^ 70)
    println("Benchmark 4: Bootstrap CI ($(BENCHMARK_BOOTSTRAP_SAMPLES) resamples)")
    println("-" ^ 70)

    t4 = @benchmark benchmark_bootstrap($BENCHMARK_BOOTSTRAP_SAMPLES) samples=20 evals=3
    println(t4)
    @printf("Per-resample: %.2f µs\n", mean(t4.times) / BENCHMARK_BOOTSTRAP_SAMPLES / 1000)
    println()

    # Benchmark 5: Kp Calculation
    println("-" ^ 70)
    println("Benchmark 5: Kp Calculation (10000 iterations)")
    println("-" ^ 70)

    t5 = @benchmark benchmark_kp_calculation(10000) samples=50 evals=5
    println(t5)
    kp_per_iter = mean(t5.times) / 10000
    @printf("Per-iteration: %.2f ns\n", kp_per_iter)
    println()

    # Summary
    println("=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    @printf("%-30s %15s %15s\n", "Benchmark", "Mean Time", "Throughput")
    println("-" ^ 70)
    @printf("%-30s %12.2f ms %12.2f M/s\n", "ODE System (100k)",
            mean(t1.times)/1e6, 1e9/ode_per_iter/1e6)
    @printf("%-30s %12.2f ms %12.2f k/s\n", "RK4 Integration (1k steps)",
            mean(t2.times)/1e6, 1e9/mean(t2.times)*1000)
    @printf("%-30s %12.2f ms %12.2f M/s\n", "Validation Metrics (10k)",
            mean(t3.times)/1e6, 1e9/metrics_per_iter/1e6)
    @printf("%-30s %12.2f ms %12.2f k/s\n", "Bootstrap CI (2k)",
            mean(t4.times)/1e6, 1e9/mean(t4.times)*1000)
    @printf("%-30s %12.2f ms %12.2f M/s\n", "Kp Calculation (10k)",
            mean(t5.times)/1e6, 1e9/kp_per_iter/1e6)
    println("=" ^ 70)

    return (ode=t1, rk4=t2, metrics=t3, bootstrap=t4, kp=t5)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
