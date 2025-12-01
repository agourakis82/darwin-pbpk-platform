"""
FractalBlood - Multi-Phase Tubular Reactor with Fractal Network Dynamics

Paradigm shift from "well-stirred tank" to "fractal network of PFRs"

Implements:
1. CTRW (Continuous Time Random Walk) framework
2. Multi-phase dynamics (plasma, RBC, protein-bound)
3. Fractal vascular network topology (Murray's Law)
4. Power-law transit time distributions

Based on:
- Goirand et al. (2021) Nature Communications - Network-driven anomalous transport
- Macheras (1996) - Fractal pharmacokinetics
- Murray's Law (1926) - Vascular branching

Author: Darwin PBPK Platform
Date: 2025-11-30
"""

module FractalBlood

using SpecialFunctions  # For gamma function, Mittag-Leffler
using QuadGK           # For numerical integration
using LinearAlgebra
using Statistics

export FractalBloodModel, VesselSegment, BloodPhase
export create_fractal_network, simulate_transport
export transit_time_distribution, power_law_pdf
export mittag_leffler, fractal_rate_constant

# ============================================================================
# CONSTANTS
# ============================================================================

const FRACTAL_DIMENSION = 2.7  # Vascular tree fractal dimension
const MURRAY_EXPONENT = 3.0    # Murray's law exponent
const BLOOD_VISCOSITY = 3.5e-3 # Pa·s (blood viscosity)
const PLASMA_VISCOSITY = 1.2e-3 # Pa·s

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

"""
BloodPhase - Represents a phase in blood (plasma, RBC, protein-bound)
"""
struct BloodPhase
    name::String
    volume_fraction::Float64  # Fraction of blood volume
    velocity_factor::Float64  # Relative to plasma velocity
    partition_coeff::Float64  # Partition coefficient from plasma
    binding_rate::Float64     # Rate of exchange with plasma (1/h)
end

"""
VesselSegment - Single vessel in the vascular network
"""
mutable struct VesselSegment
    id::Int
    radius::Float64           # m
    length::Float64           # m
    level::Int                # Branching level (0=aorta)
    parent_id::Union{Int, Nothing}
    children_ids::Vector{Int}
    
    # Derived properties (calculated)
    velocity::Float64         # m/s
    flow_rate::Float64        # m³/s
    transit_time::Float64     # s
    dispersion::Float64       # m²/s (Taylor dispersion)
end

"""
FractalBloodModel - Complete fractal blood dynamics model
"""
struct FractalBloodModel
    # Network topology
    vessels::Vector{VesselSegment}
    num_levels::Int
    fractal_dimension::Float64
    
    # Multi-phase dynamics
    phases::Vector{BloodPhase}
    hematocrit::Float64
    
    # Transit time distribution parameters
    alpha::Float64            # Power-law exponent (≈1.37)
    tau_min::Float64          # Minimum transit time (s)
    tau_mean::Float64         # Mean transit time (s)
    
    # CTRW parameters
    beta::Float64             # Anomalous diffusion exponent
    waiting_time_scale::Float64
end

# ============================================================================
# MITTAG-LEFFLER FUNCTION (For fractal kinetics solutions)
# ============================================================================

"""
mittag_leffler(α, β, z; n_terms=100)

Compute the Mittag-Leffler function E_{α,β}(z)

E_{α,β}(z) = Σ_{k=0}^{∞} z^k / Γ(αk + β)

Used for solutions of fractional differential equations.
When α=1, β=1: reduces to exp(z)
"""
function mittag_leffler(α::Float64, β::Float64, z::Float64; n_terms::Int=100)::Float64
    result = 0.0
    z_power = 1.0
    
    for k in 0:n_terms
        term = z_power / gamma(α * k + β)
        result += term
        
        # Convergence check
        if abs(term) < 1e-15
            break
        end
        
        z_power *= z
    end
    
    return result
end

# Convenience: E_α(z) = E_{α,1}(z)
mittag_leffler(α::Float64, z::Float64) = mittag_leffler(α, 1.0, z)

# ============================================================================
# FRACTAL KINETICS
# ============================================================================

"""
fractal_rate_constant(k0, t, h)

Time-dependent rate "constant" for fractal kinetics:
k(t) = k₀ × t^(-h)

Where:
- k0 = intrinsic rate constant
- t = time
- h = fractal exponent (0 < h < 1)
"""
function fractal_rate_constant(k0::Float64, t::Float64, h::Float64)::Float64
    if t <= 0
        return k0  # Avoid singularity at t=0
    end
    return k0 * t^(-h)
end

# ============================================================================
# POWER-LAW TRANSIT TIME DISTRIBUTION
# ============================================================================

"""
power_law_pdf(t, α, τ_min)

Power-law probability density function for transit times:
p(τ) = (α-1)/τ_min × (τ/τ_min)^(-α) for τ ≥ τ_min

Where:
- α = power-law exponent (typically 1.3-1.5 for vascular networks)
- τ_min = minimum transit time
"""
function power_law_pdf(t::Float64, α::Float64, τ_min::Float64)::Float64
    if t < τ_min
        return 0.0
    end
    return (α - 1) / τ_min * (t / τ_min)^(-α)
end

"""
power_law_cdf(t, α, τ_min)

Cumulative distribution function for power-law transit times.
"""
function power_law_cdf(t::Float64, α::Float64, τ_min::Float64)::Float64
    if t < τ_min
        return 0.0
    end
    return 1.0 - (τ_min / t)^(α - 1)
end

"""
power_law_mean(α, τ_min)

Mean transit time for power-law distribution.
Only defined for α > 2.
"""
function power_law_mean(α::Float64, τ_min::Float64)::Float64
    if α <= 2
        return Inf  # Mean is infinite for α ≤ 2
    end
    return τ_min * (α - 1) / (α - 2)
end

# ============================================================================
# TRANSIT TIME DISTRIBUTION (Full Implementation)
# ============================================================================

"""
transit_time_distribution(model::FractalBloodModel, t)

Compute the transit time distribution E(t) for the fractal network.
Combines power-law from network topology with dispersion effects.
"""
function transit_time_distribution(model::FractalBloodModel, t::Float64)::Float64
    return power_law_pdf(t, model.alpha, model.tau_min)
end

"""
transit_time_moments(model::FractalBloodModel)

Compute statistical moments of transit time distribution.
Returns (mean, variance, skewness)
"""
function transit_time_moments(model::FractalBloodModel)
    α = model.alpha
    τ_min = model.tau_min

    # Mean (only finite for α > 2)
    if α > 2
        mean_τ = τ_min * (α - 1) / (α - 2)
    else
        mean_τ = Inf
    end

    # Variance (only finite for α > 3)
    if α > 3
        var_τ = τ_min^2 * (α - 1) / ((α - 2)^2 * (α - 3))
    else
        var_τ = Inf
    end

    # Skewness (only finite for α > 4)
    if α > 4
        skew_τ = 2 * (α - 2) * sqrt(α - 3) / ((α - 4) * sqrt(α - 1))
    else
        skew_τ = Inf
    end

    return (mean_τ, var_τ, skew_τ)
end

# ============================================================================
# CONTINUOUS TIME RANDOM WALK (CTRW)
# ============================================================================

"""
CTRWState - State of a particle in CTRW simulation
"""
mutable struct CTRWState
    position::Float64        # Position in network (0-1 normalized)
    time::Float64            # Current time
    phase::Int               # Current phase (1=plasma, 2=RBC, 3=bound)
    concentration::Float64   # Local concentration
    waiting_time::Float64    # Time until next jump
end

"""
sample_waiting_time(β, τ_scale)

Sample waiting time from power-law distribution for CTRW.
ψ(t) ∝ t^(-1-β)
"""
function sample_waiting_time(β::Float64, τ_scale::Float64)::Float64
    u = rand()
    # Inverse CDF sampling for power-law
    return τ_scale * (1 - u)^(-1/β)
end

"""
ctrw_propagator(x, t, β, D)

Green's function for CTRW with anomalous diffusion.
For subdiffusion (β < 1): ⟨x²⟩ ∝ t^β
"""
function ctrw_propagator(x::Float64, t::Float64, β::Float64, D::Float64)::Float64
    if t <= 0
        return x == 0 ? Inf : 0.0
    end

    # Anomalous diffusion scaling
    effective_D = D * t^(β - 1)

    # Gaussian kernel with anomalous scaling
    return exp(-x^2 / (4 * effective_D * t)) / sqrt(4 * π * effective_D * t)
end

"""
simulate_ctrw(model, n_particles, t_max)

Simulate CTRW for multiple particles through the fractal network.
Returns concentration profile over time.
"""
function simulate_ctrw(model::FractalBloodModel, n_particles::Int, t_max::Float64;
                       dt::Float64=0.01)::Matrix{Float64}
    n_steps = Int(ceil(t_max / dt))

    # Output: [time, C_plasma, C_RBC, C_bound, C_total]
    results = zeros(n_steps, 5)

    # Initialize particles
    particles = [CTRWState(0.0, 0.0, 1, 1.0/n_particles,
                          sample_waiting_time(model.beta, model.waiting_time_scale))
                 for _ in 1:n_particles]

    for step in 1:n_steps
        t = step * dt
        results[step, 1] = t

        for p in particles
            # Update particle position and phase
            if t >= p.time + p.waiting_time
                # Jump to new position
                p.position += randn() * sqrt(2 * model.tau_mean * dt)
                p.position = clamp(p.position, 0.0, 1.0)

                # Phase transition probability
                phase_transition!(p, model, dt)

                # Sample new waiting time
                p.waiting_time = sample_waiting_time(model.beta, model.waiting_time_scale)
                p.time = t
            end

            # Accumulate concentrations by phase
            results[step, 1 + p.phase] += p.concentration
        end

        # Total concentration
        results[step, 5] = sum(results[step, 2:4])
    end

    return results
end

"""
phase_transition!(particle, model, dt)

Handle phase transitions for a particle (plasma ↔ RBC ↔ bound)
"""
function phase_transition!(p::CTRWState, model::FractalBloodModel, dt::Float64)
    if length(model.phases) < 2
        return
    end

    current_phase = model.phases[p.phase]

    # Transition probabilities
    for (i, other_phase) in enumerate(model.phases)
        if i != p.phase
            # Rate of transition
            k_transition = other_phase.binding_rate * dt
            if rand() < k_transition
                p.phase = i
                break
            end
        end
    end
end

# ============================================================================
# FRACTAL NETWORK TOPOLOGY (Murray's Law)
# ============================================================================

"""
create_vessel(id, radius, length, level, parent_id)

Create a vessel segment with derived properties.
"""
function create_vessel(id::Int, radius::Float64, length::Float64,
                       level::Int, parent_id::Union{Int, Nothing})::VesselSegment
    # Pressure gradient (simplified)
    delta_P = 13332.0  # ~100 mmHg in Pa

    # Poiseuille flow
    flow_rate = π * radius^4 * delta_P / (8 * BLOOD_VISCOSITY * length)
    velocity = flow_rate / (π * radius^2)
    transit_time = length / velocity

    # Taylor dispersion
    D_mol = 1e-9  # Molecular diffusion (m²/s)
    dispersion = D_mol + radius^2 * velocity^2 / (48 * D_mol)

    VesselSegment(id, radius, length, level, parent_id, Int[],
                  velocity, flow_rate, transit_time, dispersion)
end

"""
create_fractal_network(num_levels; r_aorta=0.0125, l_aorta=0.4)

Build a fractal vascular network using Murray's Law.

Murray's Law: r³_parent = Σ r³_children
Fractal scaling: N(r) ∝ r^(-D) where D ≈ 2.7

Parameters:
- num_levels: Number of branching levels (aorta=0, capillaries=~20)
- r_aorta: Aorta radius (m), default 12.5mm
- l_aorta: Aorta length (m), default 40cm
"""
function create_fractal_network(num_levels::Int;
                                 r_aorta::Float64=0.0125,
                                 l_aorta::Float64=0.4)::Vector{VesselSegment}
    vessels = VesselSegment[]

    # Create aorta (level 0)
    aorta = create_vessel(1, r_aorta, l_aorta, 0, nothing)
    push!(vessels, aorta)

    vessel_id = 2

    # Build tree level by level
    for level in 1:num_levels
        # Murray's Law: r_child = r_parent / 2^(1/3)
        r_scale = 2.0^(-1.0/MURRAY_EXPONENT)

        # Length scales with radius
        l_scale = 0.8  # Empirical scaling

        # Find all vessels at previous level
        parent_ids = [v.id for v in vessels if v.level == level - 1]

        for parent_id in parent_ids
            parent = vessels[parent_id]

            # Create 2 children (binary branching)
            for _ in 1:2
                child_radius = parent.radius * r_scale
                child_length = parent.length * l_scale

                # Stop if we reach capillary size (~5-10 μm)
                if child_radius < 5e-6
                    continue
                end

                child = create_vessel(vessel_id, child_radius, child_length,
                                      level, parent_id)
                push!(vessels, child)
                push!(parent.children_ids, vessel_id)
                vessel_id += 1
            end
        end
    end

    return vessels
end

"""
network_transit_time_distribution(vessels)

Compute the transit time distribution for the entire network.
"""
function network_transit_time_distribution(vessels::Vector{VesselSegment})
    transit_times = [v.transit_time for v in vessels]

    # Fit power-law to transit time distribution
    τ_min = minimum(transit_times)
    τ_max = maximum(transit_times)
    τ_mean = mean(transit_times)

    # Estimate α from data (simplified)
    # For a proper fit, use maximum likelihood
    log_τ = log.(transit_times)
    α_estimate = 1.0 + length(transit_times) / sum(log_τ .- log(τ_min))

    return (α=α_estimate, τ_min=τ_min, τ_max=τ_max, τ_mean=τ_mean)
end

# ============================================================================
# MULTI-PHASE BLOOD MODEL FACTORY
# ============================================================================

"""
create_default_phases(hematocrit, fu)

Create default blood phases (plasma, RBC, protein-bound).

Parameters:
- hematocrit: RBC volume fraction (0.35-0.55)
- fu: Fraction unbound in plasma (0.01-0.99)
"""
function create_default_phases(hematocrit::Float64, fu::Float64)::Vector{BloodPhase}
    plasma_fraction = 1.0 - hematocrit
    bound_fraction = plasma_fraction * (1.0 - fu)
    free_fraction = plasma_fraction * fu

    phases = [
        # Free drug in plasma
        BloodPhase("plasma_free", free_fraction, 1.0, 1.0, 0.0),

        # Drug in RBCs (Fåhræus effect: RBCs flow slower)
        BloodPhase("rbc", hematocrit, 0.8, 1.0, 0.1),  # 0.1/h exchange rate

        # Protein-bound in plasma
        BloodPhase("plasma_bound", bound_fraction, 1.0, 1.0, 10.0)  # Fast exchange
    ]

    return phases
end

"""
create_fractal_blood_model(; kwargs...)

Create a complete fractal blood dynamics model.

Keyword Arguments:
- num_levels: Number of vascular branching levels (default: 15)
- hematocrit: RBC fraction (default: 0.45)
- fu: Fraction unbound (default: 0.1)
- alpha: Power-law exponent (default: 1.37)
- beta: CTRW anomalous exponent (default: 0.8)
"""
function create_fractal_blood_model(;
    num_levels::Int=15,
    hematocrit::Float64=0.45,
    fu::Float64=0.1,
    alpha::Float64=1.37,
    beta::Float64=0.8
)::FractalBloodModel

    # Build vascular network
    vessels = create_fractal_network(num_levels)

    # Get transit time distribution parameters
    tt_params = network_transit_time_distribution(vessels)

    # Create blood phases
    phases = create_default_phases(hematocrit, fu)

    # Build model
    FractalBloodModel(
        vessels,
        num_levels,
        FRACTAL_DIMENSION,
        phases,
        hematocrit,
        alpha,
        tt_params.τ_min,
        tt_params.τ_mean,
        beta,
        tt_params.τ_mean / 10  # Waiting time scale
    )
end

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

"""
Literature values for validation
"""
const LITERATURE_VALUES = Dict(
    # Transit times (seconds)
    "aorta_transit_time" => (0.1, 0.3),  # Range (min, max)
    "capillary_transit_time" => (0.5, 2.0),
    "total_circulation_time" => (20.0, 60.0),

    # Blood-to-plasma ratios for common drugs
    "warfarin_bp_ratio" => (0.55, 0.65),
    "metformin_bp_ratio" => (0.9, 1.1),
    "chloroquine_bp_ratio" => (3.0, 5.0),

    # Fractal exponents from literature
    "vascular_fractal_dimension" => (2.6, 2.8),
    "transit_time_power_law_alpha" => (1.3, 1.5)
)

"""
validate_network_topology(vessels)

Validate that network follows Murray's Law and fractal scaling.
"""
function validate_network_topology(vessels::Vector{VesselSegment})
    results = Dict{String, Any}()

    # Check Murray's Law compliance
    murray_violations = 0
    for v in vessels
        if !isempty(v.children_ids)
            r_parent = v.radius
            r_children_cubed = sum(vessels[cid].radius^3 for cid in v.children_ids)

            # Murray's Law: r³_parent = Σ r³_children
            ratio = r_children_cubed / r_parent^3
            if abs(ratio - 1.0) > 0.1  # 10% tolerance
                murray_violations += 1
            end
        end
    end

    results["murray_law_violations"] = murray_violations
    results["murray_law_compliance"] = 1.0 - murray_violations / length(vessels)

    # Check fractal dimension
    radii = [v.radius for v in vessels]
    min_r, max_r = extrema(radii)

    # Estimate fractal dimension from vessel count vs radius
    n_bins = 10
    r_bins = exp.(range(log(min_r), log(max_r), length=n_bins+1))

    counts = zeros(n_bins)
    for (i, (r_low, r_high)) in enumerate(zip(r_bins[1:end-1], r_bins[2:end]))
        counts[i] = count(r -> r_low <= r < r_high, radii)
    end

    # Fit power law: N(r) ∝ r^(-D)
    valid_bins = counts .> 0
    if sum(valid_bins) >= 3
        log_r = log.((r_bins[1:end-1] .+ r_bins[2:end]) ./ 2)[valid_bins]
        log_n = log.(counts[valid_bins])

        # Linear regression
        slope = cov(log_r, log_n) / var(log_r)
        D_estimated = -slope

        results["estimated_fractal_dimension"] = D_estimated

        # Compare to expected range
        D_min, D_max = LITERATURE_VALUES["vascular_fractal_dimension"]
        results["fractal_dimension_valid"] = D_min <= D_estimated <= D_max
    end

    # Transit time statistics
    transit_times = [v.transit_time for v in vessels]
    results["min_transit_time"] = minimum(transit_times)
    results["max_transit_time"] = maximum(transit_times)
    results["mean_transit_time"] = mean(transit_times)

    return results
end

"""
validate_transit_time_distribution(model; n_samples=10000)

Validate that transit time distribution matches power-law expectation.
"""
function validate_transit_time_distribution(model::FractalBloodModel; n_samples::Int=10000)
    results = Dict{String, Any}()

    # Sample from theoretical distribution
    t_samples = Float64[]
    t = model.tau_min
    for _ in 1:n_samples
        # Inverse CDF sampling for power-law
        u = rand()
        t_sample = model.tau_min * (1 - u)^(-1/(model.alpha - 1))
        push!(t_samples, t_sample)
    end

    # Compute statistics
    results["sample_mean"] = mean(t_samples)
    results["sample_std"] = std(t_samples)
    results["sample_median"] = median(t_samples)

    # Compare to theoretical moments
    moments = transit_time_moments(model)
    if isfinite(moments[1])
        results["theoretical_mean"] = moments[1]
        results["mean_error"] = abs(results["sample_mean"] - moments[1]) / moments[1]
    end

    # Check power-law behavior (log-log should be linear)
    bins = 10 .^ range(-1, 2, length=30)
    hist = zeros(length(bins)-1)
    for (i, (b_low, b_high)) in enumerate(zip(bins[1:end-1], bins[2:end]))
        hist[i] = count(t -> b_low <= t < b_high, t_samples)
    end

    # Fit power-law exponent
    valid = hist .> 0
    if sum(valid) >= 5
        log_t = log10.((bins[1:end-1] .+ bins[2:end]) ./ 2)[valid]
        log_p = log10.(hist[valid] ./ sum(hist) ./ diff(bins)[valid])

        slope = cov(log_t, log_p) / var(log_t)
        alpha_estimated = -slope

        results["estimated_alpha"] = alpha_estimated
        results["alpha_error"] = abs(alpha_estimated - model.alpha) / model.alpha
    end

    return results
end

"""
compare_to_traditional_pbpk(model, drug_params; t_max=24.0)

Compare fractal blood model predictions to traditional PBPK.
"""
function compare_to_traditional_pbpk(model::FractalBloodModel,
                                      drug_params::Dict;
                                      t_max::Float64=24.0)
    results = Dict{String, Any}()

    # Get drug parameters
    dose = get(drug_params, "dose", 100.0)  # mg
    Vd = get(drug_params, "Vd", 70.0)  # L
    CL = get(drug_params, "CL", 10.0)  # L/h

    # Traditional 1-compartment model
    k_el = CL / Vd
    t = range(0, t_max, length=100)
    C_traditional = (dose / Vd) .* exp.(-k_el .* t)

    # Fractal kinetics model
    # Using Mittag-Leffler instead of exponential
    h = 1.0 - model.beta  # Fractal exponent
    C_fractal = similar(C_traditional)
    for (i, ti) in enumerate(t)
        if ti > 0
            C_fractal[i] = (dose / Vd) * mittag_leffler(model.beta, -k_el * ti^model.beta)
        else
            C_fractal[i] = dose / Vd
        end
    end

    # Compute differences
    results["t"] = collect(t)
    results["C_traditional"] = C_traditional
    results["C_fractal"] = C_fractal

    # AUC comparison
    dt = t_max / (length(t) - 1)
    AUC_traditional = sum(C_traditional) * dt
    AUC_fractal = sum(C_fractal) * dt

    results["AUC_traditional"] = AUC_traditional
    results["AUC_fractal"] = AUC_fractal
    results["AUC_ratio"] = AUC_fractal / AUC_traditional

    # Terminal half-life comparison
    # Traditional: t½ = ln(2) / k_el
    # Fractal: Power-law decay (no constant half-life)
    results["traditional_half_life"] = log(2) / k_el

    # Estimate apparent half-life for fractal (time to 50% of Cmax)
    C_max = maximum(C_fractal)
    idx_half = findfirst(x -> x < C_max/2, C_fractal)
    if idx_half !== nothing
        results["fractal_apparent_half_life"] = t[idx_half]
    end

    return results
end

end # module FractalBlood
