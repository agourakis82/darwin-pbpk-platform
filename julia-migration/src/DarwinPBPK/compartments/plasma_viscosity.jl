"""
Plasma Viscosity Effects Module

Comprehensive blood rheology modeling for PBPK applications.
Implements non-Newtonian blood flow, viscosity determinants,
and their effects on drug distribution and clearance.

Key Features:
- Carreau-Yasuda non-Newtonian viscosity model
- Hematocrit-viscosity relationships
- Plasma protein effects (fibrinogen, globulins)
- Fåhræus-Lindqvist effect for microvessels
- Hyperviscosity syndrome modeling
- Temperature-dependent viscosity
- Shear-rate dependent effects

Equations:
- Carreau-Yasuda: η = η_∞ + (η_0 - η_∞) × [1 + (λγ̇)^a]^((n-1)/a)
- Hematocrit: μ = μ_plasma × exp(k × Hct/(1-Hct))
- Poiseuille: Q = πr⁴ΔP/(8μL)
- Fåhræus-Lindqvist: μ_tube = f(diameter, Hct)

References:
- Baskurt OK, Meiselman HJ. Blood Rheology and Hemodynamics (2003)
- Carreau PJ. Rheological equations from molecular network theories (1972)
- Fåhræus R, Lindqvist T. Am J Physiol (1931)
- Somer T, Meiselman HJ. Ann Hematol (1993) - Hyperviscosity syndromes

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module PlasmaViscosity

using Statistics

export ViscosityState, BloodRheology, PerfusionState
export CarreauYasudaParams, FahrauesLindqvistParams
export calculate_plasma_viscosity, calculate_blood_viscosity
export calculate_carreau_yasuda_viscosity, calculate_apparent_viscosity
export calculate_fahraeus_lindqvist_effect, calculate_microvascular_viscosity
export calculate_perfusion_effect, calculate_hepatic_flow
export calculate_renal_flow, calculate_tissue_perfusion
export apply_hyperviscosity_adjustments, apply_hemodilution_adjustments
export create_normal_rheology, create_hyperviscosity_state
export estimate_viscosity_from_hematocrit, estimate_viscosity_from_proteins
export NORMAL_VISCOSITY, SHEAR_RATE_RANGES, HYPERVISCOSITY_SYNDROMES
export CARREAU_YASUDA_NORMAL, FAHRAEUS_LINDQVIST_PARAMS

# ============================================================================
# CONSTANTS
# ============================================================================

"""Normal blood viscosity parameters at 37°C."""
const NORMAL_VISCOSITY = Dict{Symbol, Float64}(
    :plasma => 1.2,              # mPa·s (1.1-1.3)
    :whole_blood => 3.5,         # mPa·s at γ̇=100/s, Hct=0.42
    :serum => 1.1,               # mPa·s
    :water_37C => 0.69,          # mPa·s reference
    :relative_plasma => 1.8,     # plasma/water ratio
    :relative_blood => 5.0       # blood/water ratio
)

"""
Carreau-Yasuda model parameters for normal blood.
η = η_∞ + (η_0 - η_∞) × [1 + (λγ̇)^a]^((n-1)/a)
"""
const CARREAU_YASUDA_NORMAL = Dict{Symbol, Float64}(
    :eta_0 => 0.056,             # Pa·s (zero-shear viscosity)
    :eta_inf => 0.00345,         # Pa·s (infinite-shear viscosity)
    :lambda => 3.313,            # s (relaxation time)
    :a => 1.23,                  # Yasuda parameter
    :n => 0.3568,                # Power-law index
    :hematocrit_reference => 0.45
)

"""Fåhræus-Lindqvist effect parameters."""
const FAHRAEUS_LINDQVIST_PARAMS = Dict{Symbol, Float64}(
    :d_crit => 300.0,            # μm - above this, no effect
    :d_min => 10.0,              # μm - minimum vessel diameter
    :alpha => 0.415,             # Viscosity reduction coefficient
    :beta => 0.011               # Diameter scaling
)

"""Shear rate ranges by vascular location (1/s)."""
const SHEAR_RATE_RANGES = Dict{Symbol, Tuple{Float64, Float64}}(
    :aorta => (100.0, 500.0),
    :large_arteries => (300.0, 800.0),
    :arterioles => (500.0, 1600.0),
    :capillaries => (1000.0, 5000.0),
    :venules => (50.0, 300.0),
    :large_veins => (10.0, 50.0),
    :portal_vein => (100.0, 200.0),
    :hepatic_sinusoids => (50.0, 150.0)
)

"""Protein contributions to plasma viscosity (mPa·s per g/dL)."""
const PROTEIN_VISCOSITY_CONTRIBUTIONS = Dict{Symbol, Float64}(
    :fibrinogen => 0.21,         # Major contributor
    :igg => 0.035,               # Per g/dL
    :igm => 0.12,                # Pentamer, high contribution
    :iga => 0.04,
    :albumin => 0.005,           # Minimal contribution
    :globulins => 0.05           # Average for other globulins
)

"""Hyperviscosity syndrome definitions."""
const HYPERVISCOSITY_SYNDROMES = Dict{Symbol, Dict{Symbol, Any}}(
    :waldenstrom => Dict(
        :name => "Waldenström Macroglobulinemia",
        :protein => :igm,
        :typical_viscosity => 6.0,    # Relative to water
        :threshold_viscosity => 4.0,  # Symptoms appear
        :serum_viscosity_cp => 11.0,  # Threshold in cP
        :prevalence => 0.30,          # 30% of WM patients
        :symptoms => ["Visual disturbances", "Bleeding", "Neurological"],
        :treatment => "Plasmapheresis, rituximab"
    ),
    :multiple_myeloma => Dict(
        :name => "Multiple Myeloma Hyperviscosity",
        :protein => :igg,
        :typical_viscosity => 4.5,
        :threshold_viscosity => 4.0,
        :prevalence => 0.04,          # 2-6% of MM
        :symptoms => ["Fatigue", "Visual changes", "Mucosal bleeding"],
        :treatment => "Plasmapheresis, chemotherapy"
    ),
    :polycythemia_vera => Dict(
        :name => "Polycythemia Vera Hyperviscosity",
        :cause => :hematocrit,
        :typical_viscosity => 8.0,
        :hct_threshold => 0.55,
        :symptoms => ["Headache", "Dizziness", "Thrombosis risk"],
        :treatment => "Phlebotomy, hydroxyurea"
    ),
    :cryoglobulinemia => Dict(
        :name => "Cryoglobulinemia",
        :protein => :cryoglobulins,
        :temperature_dependent => true,
        :symptoms => ["Raynaud's", "Purpura", "Arthralgia"],
        :treatment => "Warm patient, treat underlying cause"
    )
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    CarreauYasudaParams

Parameters for Carreau-Yasuda non-Newtonian viscosity model.
"""
struct CarreauYasudaParams
    eta_0::Float64      # Zero-shear viscosity (Pa·s)
    eta_inf::Float64    # Infinite-shear viscosity (Pa·s)
    lambda::Float64     # Relaxation time (s)
    a::Float64          # Yasuda parameter
    n::Float64          # Power-law index

    function CarreauYasudaParams(;
        eta_0::Float64 = CARREAU_YASUDA_NORMAL[:eta_0],
        eta_inf::Float64 = CARREAU_YASUDA_NORMAL[:eta_inf],
        lambda::Float64 = CARREAU_YASUDA_NORMAL[:lambda],
        a::Float64 = CARREAU_YASUDA_NORMAL[:a],
        n::Float64 = CARREAU_YASUDA_NORMAL[:n]
    )
        new(eta_0, eta_inf, lambda, a, n)
    end
end

"""
    FahrauesLindqvistParams

Parameters for Fåhræus-Lindqvist effect in microvessels.
"""
struct FahrauesLindqvistParams
    d_crit::Float64     # Critical diameter (μm)
    d_min::Float64      # Minimum diameter (μm)
    alpha::Float64      # Viscosity reduction coefficient
    beta::Float64       # Diameter scaling

    function FahrauesLindqvistParams(;
        d_crit::Float64 = 300.0,
        d_min::Float64 = 10.0,
        alpha::Float64 = 0.415,
        beta::Float64 = 0.011
    )
        new(d_crit, d_min, alpha, beta)
    end
end

"""
    ViscosityState

Complete blood viscosity state.
"""
struct ViscosityState
    plasma_viscosity::Float64      # mPa·s
    whole_blood_viscosity::Float64 # mPa·s at reference shear
    serum_viscosity::Float64       # mPa·s
    hematocrit::Float64
    fibrinogen::Float64            # g/dL
    total_protein::Float64         # g/dL
    temperature::Float64           # °C
    shear_rate_reference::Float64  # 1/s
    carreau_params::CarreauYasudaParams

    function ViscosityState(;
        plasma_viscosity::Float64 = 1.2,
        whole_blood_viscosity::Float64 = 3.5,
        serum_viscosity::Float64 = 1.1,
        hematocrit::Float64 = 0.42,
        fibrinogen::Float64 = 0.3,
        total_protein::Float64 = 7.0,
        temperature::Float64 = 37.0,
        shear_rate_reference::Float64 = 100.0,
        carreau_params::CarreauYasudaParams = CarreauYasudaParams()
    )
        new(plasma_viscosity, whole_blood_viscosity, serum_viscosity,
            hematocrit, fibrinogen, total_protein, temperature,
            shear_rate_reference, carreau_params)
    end
end

"""
    BloodRheology

Complete blood rheology profile.
"""
struct BloodRheology
    viscosity::ViscosityState
    condition::Symbol              # :normal, :hyperviscosity, :hemodiluted
    syndrome::Union{Symbol, Nothing}
    yield_stress::Float64          # Pa (for flow initiation)
    thixotropy_factor::Float64     # Time-dependent viscosity
    rbc_aggregation::Float64       # 0-1 scale
    rbc_deformability::Float64     # 0-1 scale (1=normal)

    function BloodRheology(;
        viscosity::ViscosityState = ViscosityState(),
        condition::Symbol = :normal,
        syndrome::Union{Symbol, Nothing} = nothing,
        yield_stress::Float64 = 0.003,
        thixotropy_factor::Float64 = 1.0,
        rbc_aggregation::Float64 = 0.3,
        rbc_deformability::Float64 = 1.0
    )
        new(viscosity, condition, syndrome, yield_stress,
            thixotropy_factor, rbc_aggregation, rbc_deformability)
    end
end

"""
    PerfusionState

Tissue perfusion state affected by viscosity.
"""
struct PerfusionState
    hepatic_flow::Float64          # L/h
    portal_flow::Float64           # L/h
    hepatic_arterial_flow::Float64 # L/h
    renal_flow::Float64            # L/h
    cardiac_output::Float64        # L/min
    mean_arterial_pressure::Float64 # mmHg
    viscosity_factor::Float64      # vs normal

    function PerfusionState(;
        hepatic_flow::Float64 = 90.0,
        portal_flow::Float64 = 65.0,
        hepatic_arterial_flow::Float64 = 25.0,
        renal_flow::Float64 = 72.0,
        cardiac_output::Float64 = 5.0,
        mean_arterial_pressure::Float64 = 93.0,
        viscosity_factor::Float64 = 1.0
    )
        new(hepatic_flow, portal_flow, hepatic_arterial_flow,
            renal_flow, cardiac_output, mean_arterial_pressure,
            viscosity_factor)
    end
end

# ============================================================================
# CORE VISCOSITY CALCULATIONS
# ============================================================================

"""
    calculate_carreau_yasuda_viscosity(shear_rate::Float64, params::CarreauYasudaParams)

Calculate blood viscosity using Carreau-Yasuda model.

η = η_∞ + (η_0 - η_∞) × [1 + (λγ̇)^a]^((n-1)/a)

# Arguments
- `shear_rate`: Shear rate γ̇ (1/s)
- `params`: CarreauYasudaParams

# Returns
Viscosity in Pa·s
"""
function calculate_carreau_yasuda_viscosity(shear_rate::Float64,
                                            params::CarreauYasudaParams)
    if shear_rate <= 0
        return params.eta_0
    end

    term = (params.lambda * shear_rate) ^ params.a
    exponent = (params.n - 1.0) / params.a

    eta = params.eta_inf + (params.eta_0 - params.eta_inf) * (1.0 + term) ^ exponent

    return eta
end

"""
    calculate_carreau_yasuda_viscosity(shear_rate::Float64, hematocrit::Float64)

Calculate viscosity with hematocrit adjustment.
"""
function calculate_carreau_yasuda_viscosity(shear_rate::Float64, hematocrit::Float64)
    # Adjust Carreau-Yasuda parameters for hematocrit
    reference_hct = CARREAU_YASUDA_NORMAL[:hematocrit_reference]

    # Scaling factors (empirical)
    hct_ratio = hematocrit / reference_hct
    eta_0_scaled = CARREAU_YASUDA_NORMAL[:eta_0] * exp(2.5 * (hematocrit - reference_hct))
    eta_inf_scaled = CARREAU_YASUDA_NORMAL[:eta_inf] * (1.0 + 2.0 * (hematocrit - reference_hct))

    params = CarreauYasudaParams(
        eta_0 = eta_0_scaled,
        eta_inf = max(eta_inf_scaled, CARREAU_YASUDA_NORMAL[:eta_inf])
    )

    return calculate_carreau_yasuda_viscosity(shear_rate, params)
end

"""
    estimate_viscosity_from_hematocrit(hematocrit::Float64;
                                        plasma_viscosity::Float64=1.2,
                                        model::Symbol=:exponential)

Estimate whole blood viscosity from hematocrit.

Models:
- :exponential: μ = μ_plasma × exp(k × Hct/(1-Hct))
- :polynomial: μ = μ_plasma × (1 + 2.5×Hct + 7.17×Hct² + ...)
- :pries: Pries et al. empirical formula
"""
function estimate_viscosity_from_hematocrit(hematocrit::Float64;
                                            plasma_viscosity::Float64=1.2,
                                            model::Symbol=:exponential)
    if model == :exponential
        # Most commonly used
        k = 2.5  # Einstein coefficient extended
        return plasma_viscosity * exp(k * hematocrit / (1.0 - hematocrit))

    elseif model == :polynomial
        # Taylor series expansion
        h = hematocrit
        return plasma_viscosity * (1.0 + 2.5*h + 7.17*h^2 + 16.56*h^3)

    elseif model == :pries
        # Pries et al. 1992 - validated for microcirculation
        h = hematocrit * 100  # Convert to percentage
        rel_visc = 1.0 + (0.45 - 1) * ((1 - h/100)^3 - 1) / ((1 - 0.45)^3 - 1)
        return plasma_viscosity * rel_visc

    else
        error("Unknown model: $model. Use :exponential, :polynomial, or :pries")
    end
end

"""
    calculate_plasma_viscosity(fibrinogen::Float64,
                               globulins::Float64,
                               albumin::Float64;
                               temperature::Float64=37.0)

Calculate plasma viscosity from protein composition.

# Arguments
- `fibrinogen`: g/dL (normal 0.2-0.4)
- `globulins`: g/dL (normal 2.3-3.5)
- `albumin`: g/dL (normal 3.5-5.0)
- `temperature`: °C
"""
function calculate_plasma_viscosity(fibrinogen::Float64,
                                    globulins::Float64,
                                    albumin::Float64;
                                    temperature::Float64=37.0)
    # Base plasma viscosity (water at 37°C + baseline proteins)
    base = 1.0  # mPa·s

    # Protein contributions
    fib_contribution = fibrinogen * PROTEIN_VISCOSITY_CONTRIBUTIONS[:fibrinogen]
    glob_contribution = globulins * PROTEIN_VISCOSITY_CONTRIBUTIONS[:globulins]
    alb_contribution = albumin * PROTEIN_VISCOSITY_CONTRIBUTIONS[:albumin]

    viscosity = base + fib_contribution + glob_contribution + alb_contribution

    # Temperature correction (2% per °C from 37°C)
    temp_factor = 1.0 + 0.02 * (37.0 - temperature)
    viscosity *= temp_factor

    return viscosity
end

"""
    estimate_viscosity_from_proteins(igm::Float64, igg::Float64;
                                     fibrinogen::Float64=0.3,
                                     albumin::Float64=4.0)

Estimate plasma viscosity with immunoglobulin emphasis.
For hyperviscosity syndrome assessment.
"""
function estimate_viscosity_from_proteins(igm::Float64, igg::Float64;
                                          fibrinogen::Float64=0.3,
                                          albumin::Float64=4.0)
    base = 1.0

    # IgM is major contributor (pentamer)
    igm_contribution = igm * PROTEIN_VISCOSITY_CONTRIBUTIONS[:igm]
    igg_contribution = igg * PROTEIN_VISCOSITY_CONTRIBUTIONS[:igg]
    fib_contribution = fibrinogen * PROTEIN_VISCOSITY_CONTRIBUTIONS[:fibrinogen]

    return base + igm_contribution + igg_contribution + fib_contribution
end

"""
    calculate_blood_viscosity(state::ViscosityState, shear_rate::Float64)

Calculate blood viscosity at specified shear rate.
"""
function calculate_blood_viscosity(state::ViscosityState, shear_rate::Float64)
    return calculate_carreau_yasuda_viscosity(shear_rate, state.carreau_params)
end

"""
    calculate_apparent_viscosity(state::ViscosityState, vessel_type::Symbol)

Calculate apparent viscosity for specific vessel type.
"""
function calculate_apparent_viscosity(state::ViscosityState, vessel_type::Symbol)
    if !haskey(SHEAR_RATE_RANGES, vessel_type)
        error("Unknown vessel type: $vessel_type")
    end

    shear_low, shear_high = SHEAR_RATE_RANGES[vessel_type]
    shear_mean = sqrt(shear_low * shear_high)  # Geometric mean

    return calculate_blood_viscosity(state, shear_mean)
end

# ============================================================================
# FÅHRÆUS-LINDQVIST EFFECT
# ============================================================================

"""
    calculate_fahraeus_lindqvist_effect(diameter_um::Float64,
                                         hematocrit::Float64;
                                         params::FahrauesLindqvistParams=FahrauesLindqvistParams())

Calculate relative viscosity reduction in small vessels.

The Fåhræus-Lindqvist effect causes apparent viscosity to decrease
in vessels smaller than ~300 μm due to the cell-free layer.

# Returns
Viscosity reduction factor (< 1.0 for small vessels)
"""
function calculate_fahraeus_lindqvist_effect(diameter_um::Float64,
                                             hematocrit::Float64;
                                             params::FahrauesLindqvistParams=FahrauesLindqvistParams())
    if diameter_um >= params.d_crit
        return 1.0  # No effect in large vessels
    end

    if diameter_um < params.d_min
        diameter_um = params.d_min  # Limit to physical minimum
    end

    # Pries et al. formulation
    d = diameter_um
    h = hematocrit

    # Relative viscosity in tube vs large vessel
    # μ_rel = 1 + (μ_45 - 1) × f(D) × g(H)

    # D-dependent term
    c = (0.8 + exp(-0.075 * d)) * (-1.0 + 1.0 / (1.0 + 10^(-11) * d^12))
    c += 1.0 / (1.0 + 10^(-11) * d^12)

    # H-dependent term
    mu_45 = 6.0 - 2.44 * exp(-0.06 * (d^0.645))

    # Combined effect
    mu_vitro = 220.0 * exp(-1.3 * d) + 3.2 - 2.44 * exp(-0.06 * d^0.645)

    reduction = 1.0 - params.alpha * (1.0 - exp(-params.beta * diameter_um))
    reduction = max(0.3, min(1.0, reduction))  # Bound between 0.3 and 1.0

    return reduction
end

"""
    calculate_microvascular_viscosity(bulk_viscosity::Float64,
                                       diameter_um::Float64,
                                       hematocrit::Float64)

Calculate effective viscosity in microvasculature.
"""
function calculate_microvascular_viscosity(bulk_viscosity::Float64,
                                           diameter_um::Float64,
                                           hematocrit::Float64)
    fl_factor = calculate_fahraeus_lindqvist_effect(diameter_um, hematocrit)
    return bulk_viscosity * fl_factor
end

# ============================================================================
# PERFUSION EFFECTS
# ============================================================================

"""
    calculate_perfusion_effect(viscosity_state::ViscosityState;
                               reference_viscosity::Float64=3.5)

Calculate perfusion reduction due to viscosity changes.

Based on Poiseuille's Law: Q ∝ 1/μ
"""
function calculate_perfusion_effect(viscosity_state::ViscosityState;
                                    reference_viscosity::Float64=3.5)
    current_viscosity = viscosity_state.whole_blood_viscosity
    perfusion_factor = reference_viscosity / current_viscosity

    return perfusion_factor
end

"""
    calculate_hepatic_flow(viscosity_state::ViscosityState;
                           normal_flow::Float64=90.0)

Calculate hepatic blood flow adjusted for viscosity.

# Returns
Dict with total, portal, and arterial flows (L/h)
"""
function calculate_hepatic_flow(viscosity_state::ViscosityState;
                                normal_flow::Float64=90.0)
    perfusion_factor = calculate_perfusion_effect(viscosity_state)

    # Hepatic flow is relatively preserved due to dual supply
    # Portal vein more affected than hepatic artery (autoregulation)
    total_flow = normal_flow * perfusion_factor^0.7  # Less than proportional
    portal_flow = 65.0 * perfusion_factor^0.8
    arterial_flow = 25.0 * perfusion_factor^0.5  # More autoregulated

    return Dict(
        :total_flow => total_flow,
        :portal_flow => portal_flow,
        :arterial_flow => arterial_flow,
        :perfusion_factor => perfusion_factor,
        :flow_reduction => 1.0 - total_flow/normal_flow
    )
end

"""
    calculate_renal_flow(viscosity_state::ViscosityState;
                         normal_gfr::Float64=100.0)

Calculate GFR adjusted for viscosity.

GFR has significant autoregulatory capacity.
"""
function calculate_renal_flow(viscosity_state::ViscosityState;
                              normal_gfr::Float64=100.0)
    perfusion_factor = calculate_perfusion_effect(viscosity_state)

    # GFR less affected due to autoregulation
    # Empirical: GFR ∝ μ^(-0.4)
    viscosity_ratio = viscosity_state.whole_blood_viscosity / NORMAL_VISCOSITY[:whole_blood]
    gfr_factor = viscosity_ratio ^ (-0.4)

    # Bound to physiological limits
    gfr_factor = max(0.5, min(1.2, gfr_factor))

    return Dict(
        :gfr => normal_gfr * gfr_factor,
        :gfr_factor => gfr_factor,
        :renal_blood_flow => 72.0 * perfusion_factor^0.6  # L/h
    )
end

"""
    calculate_tissue_perfusion(viscosity_state::ViscosityState, tissue::Symbol)

Calculate tissue-specific perfusion adjusted for viscosity.
"""
function calculate_tissue_perfusion(viscosity_state::ViscosityState, tissue::Symbol)
    perfusion_factor = calculate_perfusion_effect(viscosity_state)

    # Tissue-specific autoregulation
    autoregulation = Dict(
        :brain => 0.3,      # Strong autoregulation
        :heart => 0.4,      # Strong
        :kidney => 0.4,     # Moderate-strong
        :liver => 0.7,      # Moderate
        :muscle => 0.9,     # Weak
        :skin => 1.0,       # None
        :adipose => 1.0,    # None
        :gut => 0.8         # Weak
    )

    auto_factor = get(autoregulation, tissue, 0.8)
    tissue_perfusion_factor = perfusion_factor ^ auto_factor

    return tissue_perfusion_factor
end

# ============================================================================
# HYPERVISCOSITY SYNDROMES
# ============================================================================

"""
    create_normal_rheology()

Create normal blood rheology state.
"""
function create_normal_rheology()
    viscosity = ViscosityState()
    return BloodRheology(viscosity=viscosity, condition=:normal)
end

"""
    create_hyperviscosity_state(syndrome::Symbol; severity::Symbol=:moderate)

Create hyperviscosity syndrome state.

# Arguments
- `syndrome`: :waldenstrom, :multiple_myeloma, :polycythemia_vera, :cryoglobulinemia
- `severity`: :mild, :moderate, :severe
"""
function create_hyperviscosity_state(syndrome::Symbol; severity::Symbol=:moderate)
    if !haskey(HYPERVISCOSITY_SYNDROMES, syndrome)
        error("Unknown syndrome: $syndrome")
    end

    syndrome_data = HYPERVISCOSITY_SYNDROMES[syndrome]

    severity_factors = Dict(
        :mild => 0.7,
        :moderate => 1.0,
        :severe => 1.5
    )
    sev_factor = get(severity_factors, severity, 1.0)

    # Calculate viscosity based on syndrome
    if syndrome == :polycythemia_vera
        hct = 0.55 * sev_factor
        hct = min(hct, 0.70)
        plasma_visc = 1.2
        blood_visc = estimate_viscosity_from_hematocrit(hct)
    else
        # Protein-mediated hyperviscosity
        hct = 0.35  # Often anemic in myeloma/WM
        base_visc = syndrome_data[:typical_viscosity] * sev_factor
        plasma_visc = base_visc
        blood_visc = plasma_visc * 2.5
    end

    # Adjust Carreau-Yasuda for hyperviscosity
    cy_params = CarreauYasudaParams(
        eta_0 = CARREAU_YASUDA_NORMAL[:eta_0] * (blood_visc / 3.5),
        eta_inf = CARREAU_YASUDA_NORMAL[:eta_inf] * (blood_visc / 3.5)^0.5
    )

    viscosity = ViscosityState(
        plasma_viscosity = plasma_visc,
        whole_blood_viscosity = blood_visc,
        hematocrit = hct,
        carreau_params = cy_params
    )

    return BloodRheology(
        viscosity = viscosity,
        condition = :hyperviscosity,
        syndrome = syndrome,
        yield_stress = 0.01,  # Increased
        rbc_aggregation = 0.6  # Increased
    )
end

"""
    apply_hyperviscosity_adjustments(rheology::BloodRheology, drug_params::Dict)

Apply hyperviscosity syndrome PK adjustments.
"""
function apply_hyperviscosity_adjustments(rheology::BloodRheology, drug_params::Dict)
    if rheology.condition != :hyperviscosity
        return drug_params
    end

    vd = get(drug_params, :vd, 1.0)
    clearance = get(drug_params, :clearance, 1.0)
    extraction_ratio = get(drug_params, :extraction_ratio, 0.5)
    fu = get(drug_params, :fu, 0.5)

    # Get perfusion effects
    hepatic = calculate_hepatic_flow(rheology.viscosity)
    renal = calculate_renal_flow(rheology.viscosity)

    # Clearance adjustments
    renal_fraction = get(drug_params, :renal_fraction, 0.3)
    hepatic_cl_factor = hepatic[:total_flow] / 90.0
    renal_cl_factor = renal[:gfr_factor]

    if extraction_ratio > 0.7
        # High extraction - flow limited
        cl_adjusted = clearance * hepatic_cl_factor
    else
        # Low extraction - less affected
        cl_adjusted = clearance * (hepatic_cl_factor ^ 0.5)
    end

    # Add renal component
    hepatic_fraction = 1.0 - renal_fraction
    cl_adjusted = clearance * (hepatic_fraction * hepatic_cl_factor +
                               renal_fraction * renal_cl_factor)

    # Vd may decrease due to reduced perfusion
    vd_adjusted = vd * (1.0 - (1.0 - hepatic[:perfusion_factor]) * 0.3)

    # Protein binding in hyperglobulinemia
    if rheology.syndrome in [:waldenstrom, :multiple_myeloma]
        # High globulin may affect binding
        fu_adjusted = fu * 0.8  # More protein binding
    else
        fu_adjusted = fu
    end

    return Dict(
        :vd_adjusted => vd_adjusted,
        :clearance_adjusted => cl_adjusted,
        :fu_adjusted => fu_adjusted,
        :hepatic_flow_factor => hepatic_cl_factor,
        :renal_factor => renal_cl_factor,
        :syndrome => rheology.syndrome,
        :viscosity_factor => rheology.viscosity.whole_blood_viscosity / NORMAL_VISCOSITY[:whole_blood],
        :special_considerations => [
            "Monitor for hyperviscosity symptoms",
            "Plasmapheresis removes 99% of protein-bound drugs",
            "Consider reduced doses for narrow therapeutic index"
        ]
    )
end

# ============================================================================
# HEMODILUTION
# ============================================================================

"""
    apply_hemodilution_adjustments(target_hct::Float64,
                                    current_hct::Float64,
                                    drug_params::Dict;
                                    ke_p::Float64=1.0)

Calculate PK changes during hemodilution (surgery, fluid resuscitation).
"""
function apply_hemodilution_adjustments(target_hct::Float64,
                                        current_hct::Float64,
                                        drug_params::Dict;
                                        ke_p::Float64=1.0)
    vd = get(drug_params, :vd, 1.0)
    clearance = get(drug_params, :clearance, 1.0)

    # Viscosity changes
    visc_current = estimate_viscosity_from_hematocrit(current_hct)
    visc_target = estimate_viscosity_from_hematocrit(target_hct)
    viscosity_ratio = visc_target / visc_current

    # Blood-plasma ratio changes
    rb_current = 1.0 - current_hct + current_hct * ke_p
    rb_target = 1.0 - target_hct + target_hct * ke_p

    # Vd increases with hemodilution (plasma volume expansion)
    vd_factor = current_hct / target_hct
    vd_adjusted = vd * vd_factor ^ 0.5

    # Clearance increases due to improved flow
    cl_factor = 1.0 / viscosity_ratio
    cl_adjusted = clearance * cl_factor ^ 0.7

    return Dict(
        :vd_adjusted => vd_adjusted,
        :clearance_adjusted => cl_adjusted,
        :viscosity_reduction => 1.0 - viscosity_ratio,
        :rb_change => rb_target / rb_current,
        :dilution_factor => target_hct / current_hct,
        :clinical_notes => [
            "Drug concentrations may dilute during fluid administration",
            "Clearance may increase due to improved perfusion",
            "Consider protein binding dilution effect"
        ]
    )
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
    classify_viscosity_state(viscosity::Float64)

Classify viscosity state.
"""
function classify_viscosity_state(viscosity::Float64)
    if viscosity < 3.0
        return :low
    elseif viscosity < 4.5
        return :normal
    elseif viscosity < 6.0
        return :elevated
    else
        return :hyperviscosity
    end
end

"""
    get_viscosity_by_vessel(rheology::BloodRheology)

Get viscosity at different vascular sites.
"""
function get_viscosity_by_vessel(rheology::BloodRheology)
    result = Dict{Symbol, Float64}()

    for (vessel, shear_range) in SHEAR_RATE_RANGES
        shear_mean = sqrt(shear_range[1] * shear_range[2])
        result[vessel] = calculate_blood_viscosity(rheology.viscosity, shear_mean)
    end

    return result
end

end # module PlasmaViscosity
