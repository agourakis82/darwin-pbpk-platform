# =============================================================================
# PBPK Benchmark Suite - Julia (No External Dependencies)
# =============================================================================

using Statistics
using Random
using Printf

# -----------------------------------------------------------------------------
# Constants & Structures (same as full benchmark)
# -----------------------------------------------------------------------------

const NUM_ORGANS = 14
const BLOOD_IDX = 1

struct PhysiologyParams
    v_blood::Float64; v_liver::Float64; v_kidney::Float64; v_lung::Float64
    v_heart::Float64; v_brain::Float64; v_muscle::Float64; v_adipose::Float64
    v_skin::Float64; v_gut::Float64; v_spleen::Float64; v_bone::Float64
    v_pancreas::Float64; v_other::Float64
    q_liver::Float64; q_kidney::Float64; q_lung::Float64; q_heart::Float64
    q_brain::Float64; q_muscle::Float64; q_adipose::Float64; q_skin::Float64
    q_gut::Float64; q_spleen::Float64; q_bone::Float64; q_pancreas::Float64
    q_other::Float64
    kp_liver::Float64; kp_kidney::Float64; kp_lung::Float64; kp_heart::Float64
    kp_brain::Float64; kp_muscle::Float64; kp_adipose::Float64; kp_skin::Float64
    kp_gut::Float64; kp_spleen::Float64; kp_bone::Float64; kp_pancreas::Float64
    kp_other::Float64
    cl_hepatic::Float64; cl_renal::Float64
    fu_plasma::Float64; blood_plasma_ratio::Float64
end

struct DrugProperties
    log_p::Float64; pka_acid::Float64; pka_base::Float64
    fu_plasma::Float64; blood_plasma_ratio::Float64
    is_acid::Bool; is_base::Bool; is_neutral::Bool
    molecular_weight::Float64
end

struct TissueComposition
    f_water::Float64; f_lipid::Float64; f_protein::Float64
    f_acidic_phospholipid::Float64; f_neutral_lipid::Float64; ph_tissue::Float64
end

# Defaults
function default_physiology()
    PhysiologyParams(
        5.2, 1.8, 0.31, 0.50, 0.33, 1.45, 29.0, 14.5, 2.6, 1.2, 0.18, 4.0, 0.10, 3.7,
        27.8, 69.6, 348.0, 17.4, 41.8, 55.7, 17.4, 17.4, 55.7, 10.4, 17.4, 3.5, 13.9,
        5.2, 4.1, 0.8, 3.2, 2.1, 2.8, 15.0, 3.5, 4.0, 3.8, 1.5, 2.5, 2.0,
        15.0, 5.0, 0.05, 0.85
    )
end

const LIVER_COMP = TissueComposition(0.751, 0.035, 0.214, 0.0040, 0.023, 7.4)
const KIDNEY_COMP = TissueComposition(0.783, 0.027, 0.190, 0.0037, 0.013, 7.4)
const MUSCLE_COMP = TissueComposition(0.756, 0.020, 0.224, 0.0022, 0.010, 7.0)
const ADIPOSE_COMP = TissueComposition(0.187, 0.756, 0.057, 0.0004, 0.852, 7.4)
const BRAIN_COMP = TissueComposition(0.773, 0.117, 0.110, 0.0112, 0.039, 7.3)
const LUNG_COMP = TissueComposition(0.802, 0.022, 0.176, 0.0060, 0.003, 7.4)

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

function pbpk_ode_system!(du::Vector{Float64}, u::Vector{Float64}, p::PhysiologyParams, t::Float64)
    c_plasma = u[1] / p.blood_plasma_ratio
    c_unbound = c_plasma * p.fu_plasma

    volumes = (p.v_blood, p.v_liver, p.v_kidney, p.v_lung, p.v_heart,
               p.v_brain, p.v_muscle, p.v_adipose, p.v_skin, p.v_gut,
               p.v_spleen, p.v_bone, p.v_pancreas, p.v_other)
    flows = (0.0, p.q_liver, p.q_kidney, p.q_lung, p.q_heart,
             p.q_brain, p.q_muscle, p.q_adipose, p.q_skin, p.q_gut,
             p.q_spleen, p.q_bone, p.q_pancreas, p.q_other)
    kps = (1.0, p.kp_liver, p.kp_kidney, p.kp_lung, p.kp_heart,
           p.kp_brain, p.kp_muscle, p.kp_adipose, p.kp_skin, p.kp_gut,
           p.kp_spleen, p.kp_bone, p.kp_pancreas, p.kp_other)

    total_flux = 0.0
    @inbounds for i in 2:NUM_ORGANS
        c_tissue_free = u[i] / kps[i]
        flux = flows[i] * (c_plasma - c_tissue_free)
        du[i] = flux / volumes[i]
        total_flux += flux
    end

    cl_rate = (p.cl_hepatic + p.cl_renal) * c_unbound
    du[1] = (-total_flux - cl_rate) / p.v_blood
    return nothing
end

function pbpk_ode_system(u::Vector{Float64}, p::PhysiologyParams, t::Float64)
    du = zeros(Float64, NUM_ORGANS)
    pbpk_ode_system!(du, u, p, t)
    return du
end

function rk4_step(y::Vector{Float64}, p::PhysiologyParams, t::Float64, dt::Float64)
    k1 = pbpk_ode_system(y, p, t)
    k2 = pbpk_ode_system(y .+ dt/2 .* k1, p, t + dt/2)
    k3 = pbpk_ode_system(y .+ dt/2 .* k2, p, t + dt/2)
    k4 = pbpk_ode_system(y .+ dt .* k3, p, t + dt)
    return y .+ dt/6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

function solve_pbpk_rk4(y0::Vector{Float64}, p::PhysiologyParams, t_start::Float64, t_end::Float64, n_steps::Int)
    dt = (t_end - t_start) / n_steps
    y = copy(y0)
    for _ in 1:n_steps
        y = rk4_step(y, p, t_start, dt)
        t_start += dt
    end
    return y
end

# Validation metrics
fold_error(pred, obs) = max(pred/obs, obs/pred)
calculate_gmfe(pred, obs) = exp(mean(abs.(log.(pred ./ obs))))
calculate_afe(pred, obs) = 10^mean(log10.(pred ./ obs))
calculate_aafe(pred, obs) = 10^mean(abs.(log10.(pred ./ obs)))

function calculate_r_squared(pred, obs)
    mean_obs = mean(obs)
    ss_res = sum((obs .- pred).^2)
    ss_tot = sum((obs .- mean_obs).^2)
    return ss_tot > 1e-10 ? 1.0 - ss_res/ss_tot : 0.0
end

function calculate_all_metrics(pred, obs)
    return (gmfe=calculate_gmfe(pred,obs), afe=calculate_afe(pred,obs),
            aafe=calculate_aafe(pred,obs), r_squared=calculate_r_squared(pred,obs))
end

# Bootstrap
function bootstrap_gmfe_ci(pred, obs, n_bootstrap, seed)
    Random.seed!(seed)
    n = length(pred)
    estimates = zeros(n_bootstrap)
    for b in 1:n_bootstrap
        idx = rand(1:n, n)
        estimates[b] = calculate_gmfe(pred[idx], obs[idx])
    end
    point_est = calculate_gmfe(pred, obs)
    std_err = std(estimates)
    return (point_estimate=point_est, ci_lower=mean(estimates)-1.96*std_err,
            ci_upper=mean(estimates)+1.96*std_err, std_error=std_err)
end

# Kp calculation
ionization_factor_base(pka, ph) = 1.0 + 10^(pka - ph)
ionization_factor_acid(pka, ph) = 1.0 + 10^(ph - pka)

function calculate_kp_rodgers_rowland(drug::DrugProperties, tissue::TissueComposition)
    p_ow = 10^drug.log_p
    kp_neutral_lipid = p_ow * tissue.f_neutral_lipid
    kp_water = tissue.f_water

    if drug.is_base
        ion_plasma = ionization_factor_base(drug.pka_base, 7.4)
        ion_tissue = ionization_factor_base(drug.pka_base, tissue.ph_tissue)
        kp_water = tissue.f_water * ion_tissue / ion_plasma
    end

    kp_ap = drug.is_base ? tissue.f_acidic_phospholipid * 0.001 * 10^(drug.log_p + 0.5 * drug.pka_base) : 0.0
    kp_protein = tissue.f_protein * 0.1 * p_ow
    kp_unbound = kp_neutral_lipid + kp_water + kp_ap + kp_protein
    return clamp(kp_unbound / drug.fu_plasma, 0.1, 100.0)
end

function calculate_all_kp(drug)
    return (kp_liver=calculate_kp_rodgers_rowland(drug, LIVER_COMP),
            kp_kidney=calculate_kp_rodgers_rowland(drug, KIDNEY_COMP),
            kp_muscle=calculate_kp_rodgers_rowland(drug, MUSCLE_COMP),
            kp_adipose=calculate_kp_rodgers_rowland(drug, ADIPOSE_COMP),
            kp_brain=calculate_kp_rodgers_rowland(drug, BRAIN_COMP),
            kp_lung=calculate_kp_rodgers_rowland(drug, LUNG_COMP))
end

# Test data
create_test_pairs() = ([15.2, 8.7, 22.3, 5.1, 31.0, 11.8, 19.5, 7.3, 25.6, 13.9],
                       [12.5, 10.1, 18.9, 6.2, 28.5, 14.2, 17.8, 8.9, 22.1, 15.5])
create_test_drug() = DrugProperties(2.5, 4.5, 8.2, 0.05, 0.85, false, true, false, 350.0)

# -----------------------------------------------------------------------------
# Benchmark Functions
# -----------------------------------------------------------------------------

function benchmark_ode_system(n_iterations)
    phys = default_physiology()
    u = zeros(Float64, NUM_ORGANS)
    u[1] = 100.0
    checksum = 0.0
    for _ in 1:n_iterations
        du = pbpk_ode_system(u, phys, 0.0)
        checksum += du[1]
    end
    return checksum
end

function benchmark_rk4_integration(n_steps)
    phys = default_physiology()
    y0 = zeros(Float64, NUM_ORGANS)
    y0[1] = 100.0 / phys.v_blood
    final = solve_pbpk_rk4(y0, phys, 0.0, 24.0, n_steps)
    return final[1]
end

function benchmark_validation_metrics(n_iterations)
    pred, obs = create_test_pairs()
    checksum = 0.0
    for _ in 1:n_iterations
        m = calculate_all_metrics(pred, obs)
        checksum += m.gmfe + m.aafe + m.r_squared
    end
    return checksum
end

function benchmark_bootstrap(n_bootstrap)
    pred, obs = create_test_pairs()
    ci = bootstrap_gmfe_ci(pred, obs, n_bootstrap, 12345)
    return ci.point_estimate + ci.ci_lower + ci.ci_upper
end

function benchmark_kp_calculation(n_iterations)
    drug = create_test_drug()
    checksum = 0.0
    for _ in 1:n_iterations
        kps = calculate_all_kp(drug)
        checksum += kps.kp_liver + kps.kp_kidney + kps.kp_adipose
    end
    return checksum
end

# -----------------------------------------------------------------------------
# Simple Timing Function
# -----------------------------------------------------------------------------

function time_benchmark(name, f, args...; warmup=3, runs=10)
    # Warmup
    for _ in 1:warmup
        f(args...)
    end

    # Timed runs
    times = zeros(runs)
    for i in 1:runs
        t0 = time_ns()
        result = f(args...)
        t1 = time_ns()
        times[i] = (t1 - t0) / 1e6  # ms
    end

    mean_t = mean(times)
    min_t = minimum(times)
    max_t = maximum(times)
    std_t = std(times)

    return (name=name, mean_ms=mean_t, min_ms=min_t, max_ms=max_t, std_ms=std_t)
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function run_benchmarks()
    println("=" ^ 70)
    println("PBPK Benchmark Suite - Julia (Simple Timing)")
    println("=" ^ 70)
    println()

    results = []

    # Benchmark 1: ODE System (100k iterations)
    println("Running ODE System benchmark (100k iterations)...")
    r1 = time_benchmark("ODE System (100k)", benchmark_ode_system, 100_000)
    push!(results, r1)
    @printf("  Mean: %.3f ms, Min: %.3f ms, Max: %.3f ms\n", r1.mean_ms, r1.min_ms, r1.max_ms)
    @printf("  Per-iteration: %.2f ns\n", r1.mean_ms * 1e6 / 100_000)
    println()

    # Benchmark 2: RK4 Integration (1k steps)
    println("Running RK4 Integration benchmark (1k steps)...")
    r2 = time_benchmark("RK4 Integration (1k)", benchmark_rk4_integration, 1000)
    push!(results, r2)
    @printf("  Mean: %.3f ms, Min: %.3f ms, Max: %.3f ms\n", r2.mean_ms, r2.min_ms, r2.max_ms)
    @printf("  Per-step: %.2f µs\n", r2.mean_ms * 1000 / 1000)
    println()

    # Benchmark 3: Validation Metrics (10k iterations)
    println("Running Validation Metrics benchmark (10k iterations)...")
    r3 = time_benchmark("Validation Metrics (10k)", benchmark_validation_metrics, 10_000)
    push!(results, r3)
    @printf("  Mean: %.3f ms, Min: %.3f ms, Max: %.3f ms\n", r3.mean_ms, r3.min_ms, r3.max_ms)
    @printf("  Per-iteration: %.2f ns\n", r3.mean_ms * 1e6 / 10_000)
    println()

    # Benchmark 4: Bootstrap CI (2k resamples)
    println("Running Bootstrap CI benchmark (2k resamples)...")
    r4 = time_benchmark("Bootstrap CI (2k)", benchmark_bootstrap, 2000)
    push!(results, r4)
    @printf("  Mean: %.3f ms, Min: %.3f ms, Max: %.3f ms\n", r4.mean_ms, r4.min_ms, r4.max_ms)
    @printf("  Per-resample: %.2f µs\n", r4.mean_ms * 1000 / 2000)
    println()

    # Benchmark 5: Kp Calculation (10k iterations)
    println("Running Kp Calculation benchmark (10k iterations)...")
    r5 = time_benchmark("Kp Calculation (10k)", benchmark_kp_calculation, 10_000)
    push!(results, r5)
    @printf("  Mean: %.3f ms, Min: %.3f ms, Max: %.3f ms\n", r5.mean_ms, r5.min_ms, r5.max_ms)
    @printf("  Per-iteration: %.2f ns\n", r5.mean_ms * 1e6 / 10_000)
    println()

    # Summary table
    println("=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    @printf("%-25s %12s %12s %12s\n", "Benchmark", "Mean (ms)", "Min (ms)", "Max (ms)")
    println("-" ^ 70)
    for r in results
        @printf("%-25s %12.3f %12.3f %12.3f\n", r.name, r.mean_ms, r.min_ms, r.max_ms)
    end
    println("=" ^ 70)

    return results
end

# Run
run_benchmarks()
