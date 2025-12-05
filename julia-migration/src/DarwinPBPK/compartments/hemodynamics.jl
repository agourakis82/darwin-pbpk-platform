"""
Hemodynamics Module - Blood Flow and Shear-Dependent Effects

Implements flow-dependent phenomena critical for:
- Shear-induced platelet activation (SIPA)
- vWF unfolding and platelet adhesion
- Flow-dependent drug transport
- Arterial vs venous differences

Based on:
- Fogelson & Neeves (2015) - Fluid mechanics of blood clot formation
- Casa et al. (2015) - Thrombus formation under high shear
- Bark & Bharat (2012) - Wall shear stress in cardiovascular flows

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

module Hemodynamics

using LinearAlgebra

export ShearEnvironment, VesselGeometry, FlowConditions
export calculate_wall_shear_stress, calculate_shear_rate
export shear_induced_platelet_activation, vwf_unfolding_probability
export calculate_residence_time, calculate_transport_rate
export create_vessel, get_flow_regime
export BLOOD_VISCOSITY, CRITICAL_SHEAR_RATES

# ============================================================================
# CONSTANTS - Hemodynamic Parameters
# ============================================================================

# Blood rheology
const BLOOD_VISCOSITY = 0.035              # Pa·s (3.5 cP at high shear)
const PLASMA_VISCOSITY = 0.0012            # Pa·s (1.2 cP)
const BLOOD_DENSITY = 1060.0               # kg/m³

# Critical shear rates (s⁻¹)
const CRITICAL_SHEAR_RATES = Dict(
    "platelet_activation_threshold" => 1000.0,    # Minimal SIPA
    "moderate_sipa" => 3000.0,                    # Moderate activation
    "high_sipa" => 5000.0,                        # High activation
    "pathological" => 10000.0,                    # Stenosis levels
    "vwf_unfolding" => 5000.0,                    # vWF A2 domain exposure
    "rbc_aggregation_limit" => 100.0              # Above this, no rouleaux
)

# Vessel-specific parameters
const VESSEL_PARAMETERS = Dict(
    "aorta" => Dict(
        "diameter_mm" => 25.0,
        "wall_shear_stress_Pa" => 0.5,
        "typical_shear_rate" => 300.0
    ),
    "large_artery" => Dict(
        "diameter_mm" => 4.0,
        "wall_shear_stress_Pa" => 1.5,
        "typical_shear_rate" => 500.0
    ),
    "arteriole" => Dict(
        "diameter_mm" => 0.05,
        "wall_shear_stress_Pa" => 4.0,
        "typical_shear_rate" => 1500.0
    ),
    "capillary" => Dict(
        "diameter_mm" => 0.008,
        "wall_shear_stress_Pa" => 2.0,
        "typical_shear_rate" => 500.0
    ),
    "venule" => Dict(
        "diameter_mm" => 0.02,
        "wall_shear_stress_Pa" => 0.5,
        "typical_shear_rate" => 200.0
    ),
    "large_vein" => Dict(
        "diameter_mm" => 10.0,
        "wall_shear_stress_Pa" => 0.3,
        "typical_shear_rate" => 100.0
    ),
    "coronary_artery" => Dict(
        "diameter_mm" => 3.0,
        "wall_shear_stress_Pa" => 1.5,
        "typical_shear_rate" => 600.0
    ),
    "stenotic_70pct" => Dict(
        "diameter_mm" => 1.0,  # After 70% stenosis
        "wall_shear_stress_Pa" => 30.0,
        "typical_shear_rate" => 10000.0  # Pathological!
    )
)

# Platelet activation parameters
const PLATELET_SHEAR_PARAMS = Dict(
    "tau_threshold_Pa" => 0.5,              # Threshold shear stress
    "tau_half_Pa" => 3.0,                   # Half-maximal activation
    "n_hill" => 2.0,                        # Hill coefficient
    "time_constant_s" => 0.1,               # Activation time constant
    "irreversible_threshold_Pa" => 10.0     # Irreversible activation
)

# vWF parameters
const VWF_PARAMS = Dict(
    "unfolding_shear_threshold" => 5000.0,  # s⁻¹
    "a2_exposure_shear" => 3000.0,          # s⁻¹ for A2 domain
    "gp1b_binding_rate" => 1e6,             # M⁻¹s⁻¹
    "multimer_size_threshold" => 10         # UL-vWF
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
VesselGeometry - Vessel geometric parameters
"""
struct VesselGeometry
    name::String
    diameter::Float64           # m
    length::Float64             # m
    stenosis_fraction::Float64  # 0-1 (0 = no stenosis)
    curvature::Float64          # 1/m (0 = straight)
    bifurcation_angle::Float64  # radians
end

"""
FlowConditions - Blood flow parameters
"""
mutable struct FlowConditions
    # Flow rate
    volumetric_flow::Float64    # m³/s
    mean_velocity::Float64      # m/s
    peak_velocity::Float64      # m/s (for pulsatile)

    # Derived quantities
    reynolds_number::Float64
    womersley_number::Float64   # Pulsatility parameter

    # Pulsatility
    pulsatile::Bool
    heart_rate::Float64         # beats/min
    systolic_fraction::Float64  # Fraction of cycle in systole
end

"""
ShearEnvironment - Local shear conditions
"""
mutable struct ShearEnvironment
    # Shear values
    wall_shear_stress::Float64  # Pa (τ_w)
    wall_shear_rate::Float64    # s⁻¹ (γ̇)
    max_shear_rate::Float64     # s⁻¹ (in bulk)

    # Spatial variation
    shear_gradient::Float64     # Pa/m
    oscillatory_shear_index::Float64  # OSI (0-0.5)

    # Derived activation potentials
    platelet_activation_potential::Float64  # 0-1
    vwf_unfolding_probability::Float64      # 0-1
    rbc_damage_potential::Float64           # 0-1

    # Time exposure
    residence_time::Float64     # s
    cumulative_shear_dose::Float64  # Pa·s
end

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

"""
create_vessel(type; stenosis=0.0, length=0.01)

Create vessel geometry from predefined types or custom parameters.
"""
function create_vessel(
    vessel_type::String;
    stenosis::Float64=0.0,
    length::Float64=0.01,
    curvature::Float64=0.0,
    bifurcation_angle::Float64=0.0
)::VesselGeometry

    if haskey(VESSEL_PARAMETERS, vessel_type)
        params = VESSEL_PARAMETERS[vessel_type]
        diameter = params["diameter_mm"] * 1e-3  # Convert to m
    else
        error("Unknown vessel type: $vessel_type")
    end

    # Adjust for stenosis
    effective_diameter = diameter * (1.0 - stenosis)

    return VesselGeometry(
        vessel_type,
        effective_diameter,
        length,
        stenosis,
        curvature,
        bifurcation_angle
    )
end

"""
create_flow_conditions(vessel, cardiac_output; pulsatile=true)

Create flow conditions for a vessel.

# Arguments
- `vessel`: VesselGeometry
- `cardiac_output`: Total cardiac output (L/min)
- `pulsatile`: Include pulsatility effects
"""
function create_flow_conditions(
    vessel::VesselGeometry;
    cardiac_output_Lmin::Float64=5.0,
    flow_fraction::Float64=0.05,  # Fraction of CO to this vessel
    pulsatile::Bool=true,
    heart_rate::Float64=70.0
)::FlowConditions

    # Volumetric flow
    Q = cardiac_output_Lmin * flow_fraction / 60.0 * 1e-3  # m³/s

    # Cross-sectional area
    A = π * (vessel.diameter / 2)^2

    # Mean velocity
    v_mean = Q / A

    # Peak velocity (assuming parabolic profile)
    v_peak = pulsatile ? v_mean * 1.5 : v_mean

    # Reynolds number
    Re = BLOOD_DENSITY * v_mean * vessel.diameter / BLOOD_VISCOSITY

    # Womersley number
    omega = 2π * heart_rate / 60.0
    alpha = vessel.diameter / 2 * sqrt(omega * BLOOD_DENSITY / BLOOD_VISCOSITY)

    return FlowConditions(
        Q, v_mean, v_peak,
        Re, alpha,
        pulsatile, heart_rate, 0.35
    )
end

# ============================================================================
# SHEAR CALCULATIONS
# ============================================================================

"""
calculate_wall_shear_stress(vessel, flow)

Calculate wall shear stress (τ_w) in Pa.

For Poiseuille flow: τ_w = 8μQ / (πR³)
"""
function calculate_wall_shear_stress(
    vessel::VesselGeometry,
    flow::FlowConditions
)::Float64

    R = vessel.diameter / 2
    Q = flow.volumetric_flow
    μ = BLOOD_VISCOSITY

    # Poiseuille formula
    tau_w = 8 * μ * Q / (π * R^3)

    # Correction for stenosis (acceleration through narrowing)
    if vessel.stenosis_fraction > 0
        # Shear stress increases with stenosis
        acceleration_factor = 1.0 / (1.0 - vessel.stenosis_fraction)^2
        tau_w *= acceleration_factor
    end

    # Correction for pulsatility
    if flow.pulsatile
        # Peak shear during systole
        tau_w *= 1.0 + 0.3 * sin(2π * flow.systolic_fraction)
    end

    return tau_w
end

"""
calculate_shear_rate(wall_shear_stress)

Calculate shear rate (γ̇) from wall shear stress.

γ̇ = τ_w / μ
"""
function calculate_shear_rate(tau_w::Float64)::Float64
    return tau_w / BLOOD_VISCOSITY
end

"""
calculate_shear_environment(vessel, flow)

Calculate complete shear environment for a vessel segment.
"""
function calculate_shear_environment(
    vessel::VesselGeometry,
    flow::FlowConditions
)::ShearEnvironment

    # Wall shear stress
    tau_w = calculate_wall_shear_stress(vessel, flow)

    # Shear rate
    gamma_dot = calculate_shear_rate(tau_w)

    # Max shear in bulk (approximately at wall for Poiseuille)
    gamma_max = gamma_dot * 1.2  # Slightly higher at exact wall

    # Shear gradient (simplified)
    shear_gradient = tau_w / (vessel.diameter / 2)

    # Oscillatory shear index (for disturbed flow)
    OSI = if vessel.curvature > 0 || vessel.bifurcation_angle > 0
        0.1 + 0.1 * vessel.curvature * 100  # Simplified
    else
        0.0
    end

    # Platelet activation potential
    platelet_potential = shear_induced_platelet_activation(gamma_dot, 0.0, 1.0)

    # vWF unfolding
    vwf_prob = vwf_unfolding_probability(gamma_dot)

    # RBC damage (hemolysis at very high shear)
    rbc_damage = if gamma_dot > 50000
        (gamma_dot - 50000) / 50000
    else
        0.0
    end

    # Residence time
    t_res = vessel.length / flow.mean_velocity

    # Cumulative shear dose
    shear_dose = tau_w * t_res

    return ShearEnvironment(
        tau_w, gamma_dot, gamma_max,
        shear_gradient, OSI,
        platelet_potential, vwf_prob, rbc_damage,
        t_res, shear_dose
    )
end

# ============================================================================
# SHEAR-INDUCED PLATELET ACTIVATION (SIPA)
# ============================================================================

"""
shear_induced_platelet_activation(shear_rate, exposure_time, baseline_activation)

Calculate platelet activation due to shear stress.

Models:
1. Threshold-dependent activation
2. Time-integrated response
3. Irreversible activation at high shear

Returns activation level (0-1)
"""
function shear_induced_platelet_activation(
    shear_rate::Float64,
    exposure_time::Float64,
    baseline_activation::Float64=0.0;
    include_time_integration::Bool=true
)::Float64

    params = PLATELET_SHEAR_PARAMS

    # Convert shear rate to shear stress
    tau = shear_rate * BLOOD_VISCOSITY

    # Threshold check
    if tau < params["tau_threshold_Pa"]
        return baseline_activation
    end

    # Hill equation for shear-dependent activation
    tau_half = params["tau_half_Pa"]
    n = params["n_hill"]

    # Instantaneous activation potential
    instant_activation = tau^n / (tau_half^n + tau^n)

    # Time-dependent accumulation (if exposure_time > 0)
    if include_time_integration && exposure_time > 0
        tau_time = params["time_constant_s"]
        time_factor = 1.0 - exp(-exposure_time / tau_time)
        activation = instant_activation * time_factor
    else
        activation = instant_activation
    end

    # Irreversible component (very high shear)
    if tau > params["irreversible_threshold_Pa"]
        irreversible = (tau - params["irreversible_threshold_Pa"]) / 10.0
        activation = min(1.0, activation + irreversible * 0.5)
    end

    # Combine with baseline
    total_activation = baseline_activation + (1.0 - baseline_activation) * activation

    return clamp(total_activation, 0.0, 1.0)
end

"""
cumulative_shear_activation(shear_history)

Calculate platelet activation from a history of shear exposures.

# Arguments
- `shear_history`: Vector of (shear_rate, duration) tuples
"""
function cumulative_shear_activation(
    shear_history::Vector{Tuple{Float64, Float64}}
)::Float64

    activation = 0.0

    for (gamma_dot, dt) in shear_history
        # Each exposure adds to activation
        delta_activation = shear_induced_platelet_activation(
            gamma_dot, dt, activation
        )
        activation = delta_activation
    end

    return activation
end

# ============================================================================
# vWF MECHANICS
# ============================================================================

"""
vwf_unfolding_probability(shear_rate)

Calculate probability of vWF unfolding to expose A2 domain.

Critical for:
- GPIbα binding
- ADAMTS13 cleavage
- Platelet capture under high shear
"""
function vwf_unfolding_probability(shear_rate::Float64)::Float64

    gamma_threshold = VWF_PARAMS["unfolding_shear_threshold"]
    gamma_half = gamma_threshold * 0.8

    if shear_rate < gamma_half * 0.5
        return 0.0
    end

    # Sigmoidal unfolding probability
    prob = 1.0 / (1.0 + exp(-(shear_rate - gamma_half) / (gamma_half * 0.2)))

    return prob
end

"""
vwf_mediated_platelet_capture(shear_rate, vwf_concentration, platelet_count)

Calculate rate of platelet capture via vWF-GPIbα interaction.

Important under high shear where direct platelet adhesion fails.
"""
function vwf_mediated_platelet_capture(
    shear_rate::Float64,
    vwf_concentration::Float64,  # nM
    platelet_count::Float64      # cells/L
)::Float64

    # vWF must be unfolded
    p_unfolded = vwf_unfolding_probability(shear_rate)

    if p_unfolded < 0.01
        return 0.0
    end

    # Capture rate (simplified)
    k_capture = VWF_PARAMS["gp1b_binding_rate"]

    # Effective capture depends on:
    # - vWF availability
    # - Platelet availability
    # - Contact time (inverse of shear)

    contact_factor = 1.0 / (1.0 + shear_rate / 10000.0)

    capture_rate = k_capture * p_unfolded * vwf_concentration * 1e-9 *
                   platelet_count * 1e-12 * contact_factor

    return capture_rate
end

# ============================================================================
# FLOW REGIME CLASSIFICATION
# ============================================================================

"""
get_flow_regime(shear_environment)

Classify flow regime for clinical interpretation.
"""
function get_flow_regime(env::ShearEnvironment)::Dict{String, Any}

    gamma = env.wall_shear_rate
    tau = env.wall_shear_stress

    # Flow classification
    regime = if gamma < 100
        :very_low_shear
    elseif gamma < 500
        :low_shear_venous
    elseif gamma < 2000
        :normal_arterial
    elseif gamma < 5000
        :elevated_arterial
    elseif gamma < 10000
        :high_shear_stenotic
    else
        :pathological_extreme
    end

    # Clinical implications
    implications = Dict{String, Bool}(
        "rbc_aggregation" => gamma < CRITICAL_SHEAR_RATES["rbc_aggregation_limit"],
        "platelet_activation_risk" => gamma > CRITICAL_SHEAR_RATES["platelet_activation_threshold"],
        "vwf_mediated_adhesion" => gamma > CRITICAL_SHEAR_RATES["vwf_unfolding"],
        "hemolysis_risk" => gamma > 50000,
        "thrombosis_risk" => env.platelet_activation_potential > 0.5
    )

    # Dominant platelet adhesion mechanism
    adhesion_mechanism = if gamma < 1000
        :integrin_mediated  # αIIbβ3, slow
    elseif gamma < 5000
        :mixed  # Both integrin and vWF
    else
        :vwf_mediated  # GPIbα-vWF only
    end

    return Dict(
        "regime" => regime,
        "shear_rate" => gamma,
        "shear_stress_Pa" => tau,
        "implications" => implications,
        "adhesion_mechanism" => adhesion_mechanism,
        "platelet_activation" => env.platelet_activation_potential,
        "vwf_unfolding" => env.vwf_unfolding_probability
    )
end

# ============================================================================
# TRANSPORT CALCULATIONS
# ============================================================================

"""
calculate_residence_time(vessel, flow)

Calculate residence time in vessel segment.
"""
function calculate_residence_time(
    vessel::VesselGeometry,
    flow::FlowConditions
)::Float64
    return vessel.length / flow.mean_velocity
end

"""
calculate_transport_rate(shear_rate, diffusion_coeff, particle_radius)

Calculate mass transport rate considering shear-enhanced diffusion.

Shear flow enhances radial transport of particles.
"""
function calculate_transport_rate(
    shear_rate::Float64,
    diffusion_coeff::Float64,  # m²/s
    particle_radius::Float64    # m
)::Float64

    # Peclet number
    Pe = shear_rate * particle_radius^2 / diffusion_coeff

    # Shear-enhanced diffusion (Leighton & Acrivos)
    if Pe < 1
        # Diffusion-dominated
        return diffusion_coeff
    else
        # Shear-enhanced
        D_enhanced = diffusion_coeff * (1.0 + 0.33 * Pe^0.5)
        return D_enhanced
    end
end

"""
calculate_near_wall_concentration(bulk_conc, shear_rate, particle_radius)

Calculate near-wall concentration enhancement due to margination.

Platelets and WBC marginate toward walls under flow.
"""
function calculate_near_wall_concentration(
    bulk_conc::Float64,
    shear_rate::Float64,
    particle_radius::Float64;
    particle_type::Symbol=:platelet
)::Float64

    # Margination factor depends on particle size and shear
    # Platelets: enhance near wall
    # RBC: deplete near wall (Fahraeus effect)

    margination_factor = if particle_type == :platelet
        # Platelets concentrate near walls
        1.0 + 0.5 * min(shear_rate / 1000.0, 2.0)
    elseif particle_type == :wbc
        # WBC also marginate but less than platelets
        1.0 + 0.3 * min(shear_rate / 1000.0, 1.5)
    elseif particle_type == :rbc
        # RBC deplete from wall region
        1.0 - 0.3 * min(shear_rate / 500.0, 0.3)
    else
        1.0
    end

    return bulk_conc * margination_factor
end

end  # module Hemodynamics
