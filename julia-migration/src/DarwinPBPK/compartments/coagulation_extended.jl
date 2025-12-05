"""
Extended Coagulation Model - FXI Feedback and Shear Integration

Extends the base coagulation model with:
- FXI feedback activation by thrombin (critical missing piece)
- Contact pathway (FXII → FXIa → FIXa)
- Shear-dependent coagulation rates
- Platelet surface-dependent reactions
- Polyphosphate enhancement
- Integration with hemodynamics module

Based on:
- Lakshmanan et al. (2022) - Extended HM model with FXI
- Chatterjee et al. (2010) - Platelet surface coagulation
- Muthard & Diamond (2012) - Shear-dependent thrombus
- Zilberman-Rudenko (2017) - Polyphosphate in coagulation

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

module CoagulationExtended

using LinearAlgebra

export ExtendedCoagulationSystem, ContactPathway, PlateletSurface
export create_extended_coagulation, simulate_extended_coagulation!
export add_fxi_feedback!, add_contact_activation!, add_shear_effects!
export calculate_surface_reactions, polyphosphate_enhancement
export EXTENDED_KINETIC_PARAMS, FXI_FEEDBACK_PARAMS

# ============================================================================
# EXTENDED KINETIC PARAMETERS
# ============================================================================

# FXI feedback parameters (from Lakshmanan 2022)
const FXI_FEEDBACK_PARAMS = Dict(
    # Thrombin activates FXI (THE CRITICAL FEEDBACK)
    "kcat_IIa_XI" => 0.23,           # s⁻¹
    "Km_IIa_XI" => 2000.0,           # nM

    # FXIa activates FIX
    "kcat_XIa_IX" => 1.8,            # s⁻¹
    "Km_XIa_IX" => 150.0,            # nM

    # FXIa is inhibited by various serpins
    "k_ATIII_XIa" => 1.2e2,          # M⁻¹s⁻¹
    "k_C1inh_XIa" => 3.6e3,          # M⁻¹s⁻¹ (C1-inhibitor)
    "k_alpha1AT_XIa" => 8.0e2,       # M⁻¹s⁻¹ (α1-antitrypsin)

    # FXI autoactivation on surfaces
    "k_XI_autoactivation" => 1e-4,   # s⁻¹ (on polyanions)

    # Surface enhancement factors
    "platelet_surface_enhancement" => 300.0,  # Fold increase on platelets
    "polyphosphate_enhancement" => 100.0      # PolyP from platelet granules
)

# Contact pathway parameters
const CONTACT_PATHWAY_PARAMS = Dict(
    # FXII activation
    "k_FXII_autoactivation" => 1e-5,  # s⁻¹ (on surfaces)
    "k_kallikrein_FXII" => 5.3e5,     # M⁻¹s⁻¹

    # FXIIa activates prekallikrein
    "kcat_FXIIa_PK" => 15.0,          # s⁻¹
    "Km_FXIIa_PK" => 400.0,           # nM

    # FXIIa activates FXI
    "kcat_FXIIa_XI" => 8.5,           # s⁻¹
    "Km_FXIIa_XI" => 250.0,           # nM

    # Kallikrein/HMWK concentrations
    "prekallikrein_conc" => 450.0,    # nM
    "HMWK_conc" => 670.0,             # nM

    # Contact surface types
    "kaolin_activation" => 1.0,       # Reference
    "polyP_activation" => 0.8,        # Polyphosphate
    "collagen_activation" => 0.3,     # Collagen (mainly GPVI)
    "glass_activation" => 0.9         # Glass (in vitro artifact)
)

# Shear-dependent coagulation parameters
const SHEAR_COAGULATION_PARAMS = Dict(
    # High shear reduces coagulation time (faster mixing)
    "optimal_shear_rate" => 500.0,    # s⁻¹
    "low_shear_penalty" => 0.3,       # Factor at very low shear
    "high_shear_penalty" => 0.5,      # Factor at very high shear (washout)

    # Shear affects different reactions differently
    "surface_reactions_shear_sensitive" => true,
    "fluid_phase_shear_sensitive" => false,

    # Critical shear for TF exposure
    "TF_exposure_shear_threshold" => 1000.0  # s⁻¹
)

# Extended concentrations
const EXTENDED_CONCENTRATIONS = Dict(
    "XI" => 45.0,                     # nM (30-60 nM)
    "XII" => 375.0,                   # nM
    "prekallikrein" => 450.0,         # nM
    "HMWK" => 670.0,                  # nM (high MW kininogen)
    "C1_inhibitor" => 2500.0,         # nM
    "alpha1_antitrypsin" => 25000.0,  # nM (very high)
    "polyphosphate" => 0.0            # Released from platelets
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
ContactPathway - Contact (intrinsic) activation system
"""
mutable struct ContactPathway
    # Zymogens
    factor_XII::Float64              # nM
    prekallikrein::Float64           # nM
    factor_XI::Float64               # nM (shared with main cascade)
    HMWK::Float64                    # nM

    # Activated
    factor_XIIa::Float64             # nM
    kallikrein::Float64              # nM
    factor_XIa::Float64              # nM

    # Surface/activator
    contact_surface::Float64         # Arbitrary units (0-1)
    surface_type::Symbol             # :kaolin, :polyP, :collagen, :none

    # Inhibitor complexes
    C1inh_XIIa::Float64
    C1inh_XIa::Float64
    C1inh_kallikrein::Float64
end

"""
PlateletSurface - Platelet procoagulant surface
"""
mutable struct PlateletSurface
    # Surface availability (0-1)
    ps_exposure::Float64             # Phosphatidylserine exposure

    # Bound factors
    Va_bound::Float64                # nM
    VIIIa_bound::Float64             # nM
    Xa_bound::Float64                # nM
    IXa_bound::Float64               # nM
    IIa_bound::Float64               # nM (thrombin)

    # Complexes on surface
    tenase_complex::Float64          # IXa·VIIIa·PS
    prothrombinase_complex::Float64  # Xa·Va·PS

    # Released granule contents
    polyphosphate::Float64           # nM (from dense granules)
    factor_V_released::Float64       # From α-granules
end

"""
ExtendedCoagulationSystem - Full coagulation with all pathways
"""
mutable struct ExtendedCoagulationSystem
    # Core factors (from base model)
    factor_II::Float64
    factor_IIa::Float64
    factor_V::Float64
    factor_Va::Float64
    factor_VII::Float64
    factor_VIIa::Float64
    factor_VIII::Float64
    factor_VIIIa::Float64
    factor_IX::Float64
    factor_IXa::Float64
    factor_X::Float64
    factor_Xa::Float64
    factor_XIII::Float64
    factor_XIIIa::Float64
    fibrinogen::Float64
    fibrin::Float64

    # Contact pathway
    contact::ContactPathway

    # Platelet surface
    platelet_surface::PlateletSurface

    # Inhibitors
    antithrombin::Float64
    TFPI::Float64
    protein_C::Float64
    APC::Float64

    # Tissue factor (stimulus)
    tissue_factor::Float64

    # Shear environment
    shear_rate::Float64              # s⁻¹
    shear_factor::Float64            # Modification factor (0-1)

    # FXI feedback state
    fxi_feedback_active::Bool
    fxi_contribution::Float64        # Fraction of IXa from FXI pathway
end

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

"""
create_contact_pathway(; surface_type, surface_amount)
"""
function create_contact_pathway(;
    surface_type::Symbol=:none,
    surface_amount::Float64=0.0
)::ContactPathway

    return ContactPathway(
        EXTENDED_CONCENTRATIONS["XII"],
        EXTENDED_CONCENTRATIONS["prekallikrein"],
        EXTENDED_CONCENTRATIONS["XI"],
        EXTENDED_CONCENTRATIONS["HMWK"],
        0.0,  # XIIa
        0.0,  # kallikrein
        0.0,  # XIa
        surface_amount,
        surface_type,
        0.0, 0.0, 0.0  # inhibitor complexes
    )
end

"""
create_platelet_surface(; ps_exposure, activated_platelets)
"""
function create_platelet_surface(;
    ps_exposure::Float64=0.0,
    platelet_activation::Float64=0.0
)::PlateletSurface

    # PS exposure correlates with platelet activation
    actual_ps = max(ps_exposure, platelet_activation * 0.8)

    # PolyP released upon activation
    polyP = platelet_activation * 100.0  # Up to 100 nM

    # Factor V from α-granules
    fV_released = platelet_activation * 10.0  # Up to 10 nM

    return PlateletSurface(
        actual_ps,
        0.0, 0.0, 0.0, 0.0, 0.0,  # bound factors
        0.0, 0.0,                   # complexes
        polyP, fV_released
    )
end

"""
create_extended_coagulation(; tissue_factor, shear_rate, platelet_activation)

Create extended coagulation system with all pathways.
"""
function create_extended_coagulation(;
    tissue_factor::Float64=0.0,
    shear_rate::Float64=500.0,
    platelet_activation::Float64=0.0,
    contact_surface::Symbol=:none,
    contact_amount::Float64=0.0,
    deficiencies::Dict{String, Float64}=Dict{String, Float64}()
)::ExtendedCoagulationSystem

    # Get deficiency-adjusted concentrations
    get_conc = (name, default) -> default * get(deficiencies, name, 1.0)

    # Calculate shear factor
    shear_factor = calculate_shear_factor(shear_rate)

    return ExtendedCoagulationSystem(
        # Core factors
        get_conc("II", 1400.0),
        0.0,  # IIa
        get_conc("V", 22.0),
        0.0,
        get_conc("VII", 10.0),
        get_conc("VIIa", 0.1),
        get_conc("VIII", 0.7),
        0.0,
        get_conc("IX", 90.0),
        0.0,
        get_conc("X", 170.0),
        0.0,
        get_conc("XIII", 70.0),
        0.0,
        get_conc("fibrinogen", 9000.0),
        0.0,

        # Contact pathway
        create_contact_pathway(surface_type=contact_surface, surface_amount=contact_amount),

        # Platelet surface
        create_platelet_surface(platelet_activation=platelet_activation),

        # Inhibitors
        get_conc("ATIII", 2400.0),
        get_conc("TFPI", 2.5),
        get_conc("protein_C", 65.0),
        0.0,  # APC

        # TF and shear
        tissue_factor,
        shear_rate,
        shear_factor,

        # FXI feedback
        true,  # Active by default
        0.0    # Contribution tracking
    )
end

# ============================================================================
# SHEAR EFFECTS
# ============================================================================

"""
calculate_shear_factor(shear_rate)

Calculate shear-dependent modification factor for coagulation rates.
"""
function calculate_shear_factor(shear_rate::Float64)::Float64

    params = SHEAR_COAGULATION_PARAMS
    optimal = params["optimal_shear_rate"]

    if shear_rate < 10.0
        # Very low shear (stasis) - poor mixing
        return params["low_shear_penalty"]

    elseif shear_rate < optimal
        # Below optimal - scales up
        return params["low_shear_penalty"] +
               (1.0 - params["low_shear_penalty"]) * (shear_rate / optimal)

    elseif shear_rate < 2000.0
        # Optimal range
        return 1.0

    else
        # High shear - washout effect
        washout = (shear_rate - 2000.0) / 10000.0
        return max(params["high_shear_penalty"], 1.0 - washout)
    end
end

"""
add_shear_effects!(system, shear_rate)

Update system with new shear conditions.
"""
function add_shear_effects!(
    system::ExtendedCoagulationSystem,
    shear_rate::Float64
)
    system.shear_rate = shear_rate
    system.shear_factor = calculate_shear_factor(shear_rate)

    # High shear increases PS exposure on platelets
    if shear_rate > 3000.0
        shear_ps_contribution = (shear_rate - 3000.0) / 10000.0 * 0.3
        system.platelet_surface.ps_exposure = min(1.0,
            system.platelet_surface.ps_exposure + shear_ps_contribution)
    end

    return nothing
end

# ============================================================================
# FXI FEEDBACK SYSTEM
# ============================================================================

"""
calculate_fxi_feedback(system, dt)

Calculate FXI feedback contribution to FIXa generation.

This is the CRITICAL pathway that Hockin-Mann missed:
Thrombin → FXIa → FIXa → (Tenase) → more Xa → more Thrombin

This positive feedback loop is essential for sustained thrombin generation.
"""
function calculate_fxi_feedback(
    system::ExtendedCoagulationSystem,
    dt::Float64
)::Tuple{Float64, Float64}  # Returns (d_XIa, d_IXa_from_XI)

    if !system.fxi_feedback_active
        return (0.0, 0.0)
    end

    params = FXI_FEEDBACK_PARAMS
    contact = system.contact

    IIa = system.factor_IIa
    XI = contact.factor_XI
    XIa = contact.factor_XIa
    IX = system.factor_IX

    # 1. Thrombin activates FXI (main feedback)
    rate_IIa_XIa = params["kcat_IIa_XI"] * IIa * XI / (params["Km_IIa_XI"] + XI)

    # 2. Surface enhancement (polyphosphate from platelets)
    polyP = system.platelet_surface.polyphosphate
    surface_factor = 1.0 + polyP / 100.0 * (params["polyphosphate_enhancement"] - 1.0)

    # 3. Platelet surface enhancement
    ps = system.platelet_surface.ps_exposure
    platelet_factor = 1.0 + ps * (params["platelet_surface_enhancement"] / 100.0 - 1.0)

    # Enhanced rate
    rate_IIa_XIa_enhanced = rate_IIa_XIa * surface_factor * platelet_factor

    # 4. FXIa activates FIX
    rate_XIa_IX = params["kcat_XIa_IX"] * XIa * IX / (params["Km_XIa_IX"] + IX)

    # 5. FXIa inhibition
    ATIII = system.antithrombin
    C1inh = EXTENDED_CONCENTRATIONS["C1_inhibitor"]

    rate_ATIII_XIa = params["k_ATIII_XIa"] * ATIII * XIa * 1e-9
    rate_C1inh_XIa = params["k_C1inh_XIa"] * C1inh * XIa * 1e-9

    # Net changes
    d_XIa = rate_IIa_XIa_enhanced - rate_ATIII_XIa - rate_C1inh_XIa
    d_IXa = rate_XIa_IX

    return (d_XIa * dt, d_IXa * dt)
end

"""
add_fxi_feedback!(system, enable)

Enable or disable FXI feedback pathway.
"""
function add_fxi_feedback!(system::ExtendedCoagulationSystem, enable::Bool)
    system.fxi_feedback_active = enable
    return nothing
end

# ============================================================================
# CONTACT PATHWAY ACTIVATION
# ============================================================================

"""
calculate_contact_activation(system, dt)

Calculate contact pathway activation.

Relevant for:
- In vitro assays (aPTT uses kaolin/ellagic acid)
- Some in vivo situations (polyphosphate, collagen exposure)
- NOT the primary initiator in vivo (that's TF pathway)
"""
function calculate_contact_activation(
    system::ExtendedCoagulationSystem,
    dt::Float64
)::Tuple{Float64, Float64, Float64}  # (d_XIIa, d_kallikrein, d_XIa_from_XII)

    contact = system.contact
    params = CONTACT_PATHWAY_PARAMS

    if contact.surface_type == :none || contact.contact_surface < 0.01
        return (0.0, 0.0, 0.0)
    end

    # Surface activation factor
    surface_factor = get(params, "$(contact.surface_type)_activation", 0.5) * contact.contact_surface

    XII = contact.factor_XII
    XIIa = contact.factor_XIIa
    PK = contact.prekallikrein
    KAL = contact.kallikrein
    XI = contact.factor_XI

    # 1. FXII autoactivation on surface
    rate_XII_auto = params["k_FXII_autoactivation"] * XII * surface_factor

    # 2. Kallikrein feedback on FXII
    rate_KAL_XII = params["k_kallikrein_FXII"] * KAL * XII * 1e-9

    # 3. FXIIa activates prekallikrein
    rate_XIIa_PK = params["kcat_FXIIa_PK"] * XIIa * PK / (params["Km_FXIIa_PK"] + PK)

    # 4. FXIIa activates FXI (contact activation, separate from thrombin)
    rate_XIIa_XI = params["kcat_FXIIa_XI"] * XIIa * XI / (params["Km_FXIIa_XI"] + XI)

    # 5. C1-inhibitor consumption
    C1inh = EXTENDED_CONCENTRATIONS["C1_inhibitor"]
    rate_C1inh_XIIa = 5e3 * C1inh * XIIa * 1e-9
    rate_C1inh_KAL = 2e3 * C1inh * KAL * 1e-9

    # Net changes
    d_XIIa = (rate_XII_auto + rate_KAL_XII - rate_C1inh_XIIa) * dt
    d_KAL = (rate_XIIa_PK - rate_C1inh_KAL) * dt
    d_XIa_contact = rate_XIIa_XI * dt

    return (d_XIIa, d_KAL, d_XIa_contact)
end

"""
add_contact_activation!(system, surface_type, amount)

Add contact activation surface.
"""
function add_contact_activation!(
    system::ExtendedCoagulationSystem,
    surface_type::Symbol,
    amount::Float64
)
    system.contact.surface_type = surface_type
    system.contact.contact_surface = clamp(amount, 0.0, 1.0)
    return nothing
end

# ============================================================================
# PLATELET SURFACE REACTIONS
# ============================================================================

"""
calculate_surface_reactions(system, dt)

Calculate reactions occurring on platelet phosphatidylserine surface.

Key concept: Coagulation is MASSIVELY enhanced on activated platelet surfaces.
- Tenase: 100,000× faster on surface
- Prothrombinase: 300,000× faster on surface
"""
function calculate_surface_reactions(
    system::ExtendedCoagulationSystem,
    dt::Float64
)::Dict{String, Float64}

    ps = system.platelet_surface

    if ps.ps_exposure < 0.01
        return Dict("tenase_rate" => 0.0, "prothrombinase_rate" => 0.0)
    end

    # Surface enhancement factors
    TENASE_ENHANCEMENT = 1e5
    PROTHROMBINASE_ENHANCEMENT = 3e5

    # Available binding sites (proportional to PS exposure)
    binding_sites = ps.ps_exposure

    # Factor binding to surface (simplified equilibrium)
    # In reality: factors bind with specific Kd values

    # IXa and VIIIa form tenase on surface
    IXa_available = system.factor_IXa
    VIIIa_available = system.factor_VIIIa

    tenase_potential = min(IXa_available, VIIIa_available) * binding_sites
    ps.tenase_complex = tenase_potential * 0.9  # 90% binding

    # Xa and Va form prothrombinase on surface
    Xa_available = system.factor_Xa
    Va_available = system.factor_Va + ps.factor_V_released * 0.5  # Include platelet FV

    prothrombinase_potential = min(Xa_available, Va_available) * binding_sites
    ps.prothrombinase_complex = prothrombinase_potential * 0.95

    # Enhanced reaction rates
    # Tenase: X → Xa
    X = system.factor_X
    kcat_tenase_surface = 10.7 * TENASE_ENHANCEMENT * binding_sites
    rate_tenase = kcat_tenase_surface * ps.tenase_complex * X / (160.0 + X) * 1e-5  # Scaled

    # Prothrombinase: II → IIa
    II = system.factor_II
    kcat_proth_surface = 32.4 * PROTHROMBINASE_ENHANCEMENT * binding_sites
    rate_prothrombinase = kcat_proth_surface * ps.prothrombinase_complex * II / (300.0 + II) * 1e-5

    return Dict(
        "tenase_rate" => rate_tenase * dt,
        "prothrombinase_rate" => rate_prothrombinase * dt,
        "tenase_complex" => ps.tenase_complex,
        "prothrombinase_complex" => ps.prothrombinase_complex
    )
end

"""
polyphosphate_enhancement(polyP_conc)

Calculate enhancement factor from platelet polyphosphate.

PolyP:
- Accelerates FV activation
- Enhances FXII activation
- Inhibits TFPI
- Delays fibrinolysis
"""
function polyphosphate_enhancement(polyP_conc::Float64)::Dict{String, Float64}

    if polyP_conc < 1.0
        return Dict(
            "FV_activation" => 1.0,
            "FXII_activation" => 1.0,
            "TFPI_inhibition" => 0.0,
            "fibrinolysis_delay" => 1.0
        )
    end

    # Saturable effects
    FV_effect = 1.0 + 10.0 * polyP_conc / (polyP_conc + 50.0)
    FXII_effect = 1.0 + 5.0 * polyP_conc / (polyP_conc + 100.0)
    TFPI_inhib = 0.5 * polyP_conc / (polyP_conc + 30.0)
    fibrin_delay = 1.0 + 2.0 * polyP_conc / (polyP_conc + 80.0)

    return Dict(
        "FV_activation" => FV_effect,
        "FXII_activation" => FXII_effect,
        "TFPI_inhibition" => TFPI_inhib,
        "fibrinolysis_delay" => fibrin_delay
    )
end

# ============================================================================
# EXTENDED ODE SIMULATION
# ============================================================================

"""
simulate_extended_coagulation!(system, tspan, dt)

Simulate extended coagulation model with all pathways.
"""
function simulate_extended_coagulation!(
    system::ExtendedCoagulationSystem,
    tspan::Tuple{Float64, Float64},
    dt::Float64
)
    t_start, t_end = tspan
    times = collect(t_start:dt:t_end)
    n_steps = length(times)

    results = Vector{Dict{String, Float64}}(undef, n_steps)

    # Track FXI contribution
    total_IXa_generated = 0.0
    IXa_from_FXI = 0.0

    for (i, t) in enumerate(times)
        # Store current state
        results[i] = Dict(
            "time" => t,
            "thrombin_nM" => system.factor_IIa,
            "fibrin_nM" => system.fibrin,
            "Xa_nM" => system.factor_Xa,
            "IXa_nM" => system.factor_IXa,
            "XIa_nM" => system.contact.factor_XIa,
            "XIIa_nM" => system.contact.factor_XIIa,
            "FXI_contribution" => system.fxi_contribution,
            "ps_exposure" => system.platelet_surface.ps_exposure,
            "polyP_nM" => system.platelet_surface.polyphosphate
        )

        # === EXTRINSIC PATHWAY ===
        TF = system.tissue_factor
        VIIa = system.factor_VIIa
        X = system.factor_X
        IX = system.factor_IX

        # TF·VIIa activates X and IX
        TF_VIIa = TF * VIIa / (TF + 1.0)
        rate_Xa_extrinsic = 103.0 * TF_VIIa * X / (450.0 + X) * system.shear_factor
        rate_IXa_extrinsic = 2.3 * TF_VIIa * IX / (300.0 + IX) * system.shear_factor

        # === FXI FEEDBACK PATHWAY ===
        d_XIa, d_IXa_FXI = calculate_fxi_feedback(system, dt)
        system.contact.factor_XIa += d_XIa

        # Track FXI contribution
        if d_IXa_FXI > 0
            IXa_from_FXI += d_IXa_FXI
        end

        # === CONTACT PATHWAY ===
        d_XIIa, d_KAL, d_XIa_contact = calculate_contact_activation(system, dt)
        system.contact.factor_XIIa += d_XIIa
        system.contact.kallikrein += d_KAL
        system.contact.factor_XIa += d_XIa_contact

        # === SURFACE REACTIONS ===
        surface_rates = calculate_surface_reactions(system, dt)

        # === COMMON PATHWAY ===
        # Tenase (fluid phase + surface)
        IXa = system.factor_IXa
        VIIIa = system.factor_VIIIa
        tenase_fluid = IXa * VIIIa / (IXa + VIIIa + 1.0)
        rate_Xa_tenase = 10.7 * tenase_fluid * X / (160.0 + X) * system.shear_factor
        rate_Xa_surface = surface_rates["tenase_rate"]

        # Prothrombinase (fluid phase + surface)
        Xa = system.factor_Xa
        Va = system.factor_Va
        II = system.factor_II
        proth_fluid = Xa * Va / (Xa + Va + 1.0)
        rate_IIa_fluid = 32.4 * proth_fluid * II / (300.0 + II) * system.shear_factor
        rate_IIa_surface = surface_rates["prothrombinase_rate"]

        # === THROMBIN FEEDBACK ===
        IIa = system.factor_IIa
        V = system.factor_V
        VIII = system.factor_VIII

        rate_Va = 20.0 * IIa * V / (100.0 + V)
        rate_VIIIa = 20.0 * IIa * VIII / (100.0 + VIII)

        # Fibrin formation
        Fg = system.fibrinogen
        rate_fibrin = 84.0 * IIa * Fg / (7200.0 + Fg)

        # === INHIBITION ===
        ATIII = system.antithrombin
        rate_ATIII_IIa = 7.1e3 * ATIII * IIa * 1e-9
        rate_ATIII_Xa = 1.5e3 * ATIII * Xa * 1e-9

        # PolyP inhibits TFPI
        polyP_effects = polyphosphate_enhancement(system.platelet_surface.polyphosphate)
        TFPI_effective = system.TFPI * (1.0 - polyP_effects["TFPI_inhibition"])
        rate_TFPI_Xa = 9e5 * TFPI_effective * Xa * 1e-9

        # === UPDATE STATE ===
        # Xa
        d_Xa = (rate_Xa_extrinsic + rate_Xa_tenase + rate_Xa_surface -
                rate_ATIII_Xa - rate_TFPI_Xa) * dt
        system.factor_Xa = max(0.0, system.factor_Xa + d_Xa)

        # IXa (from TF pathway + FXI pathway)
        d_IXa = (rate_IXa_extrinsic + d_IXa_FXI) * dt
        system.factor_IXa = max(0.0, system.factor_IXa + d_IXa)
        total_IXa_generated += d_IXa

        # IIa (Thrombin)
        d_IIa = (rate_IIa_fluid + rate_IIa_surface - rate_ATIII_IIa) * dt
        system.factor_IIa = max(0.0, system.factor_IIa + d_IIa)

        # Cofactors
        system.factor_Va = max(0.0, system.factor_Va + rate_Va * dt)
        system.factor_VIIIa = max(0.0, system.factor_VIIIa + rate_VIIIa * dt)

        # Fibrin
        system.fibrin += rate_fibrin * dt

        # Substrates consumed
        system.factor_II = max(0.0, system.factor_II - (rate_IIa_fluid + rate_IIa_surface) * dt)
        system.factor_X = max(0.0, system.factor_X - (rate_Xa_extrinsic + rate_Xa_tenase) * dt)
        system.fibrinogen = max(0.0, system.fibrinogen - rate_fibrin * dt)
        system.factor_V = max(0.0, system.factor_V - rate_Va * dt)
        system.factor_VIII = max(0.0, system.factor_VIII - rate_VIIIa * dt)

        # Inhibitors consumed
        system.antithrombin = max(0.0, system.antithrombin - (rate_ATIII_IIa + rate_ATIII_Xa) * dt)

        # PS exposure increases with thrombin (PAR activation)
        if IIa > 1.0
            ps_increase = 0.01 * (IIa / (IIa + 10.0)) * dt
            system.platelet_surface.ps_exposure = min(1.0,
                system.platelet_surface.ps_exposure + ps_increase)
        end
    end

    # Calculate FXI contribution to total IXa
    if total_IXa_generated > 0
        system.fxi_contribution = IXa_from_FXI / total_IXa_generated
    end

    return times, results
end

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

"""
compare_with_without_fxi(system_params; tspan, dt)

Compare thrombin generation with and without FXI feedback.

This demonstrates why FXI feedback is critical.
"""
function compare_with_without_fxi(;
    tissue_factor::Float64=0.005,
    tspan::Tuple{Float64, Float64}=(0.0, 1200.0),
    dt::Float64=0.5
)::Dict{String, Any}

    # With FXI
    sys_with = create_extended_coagulation(tissue_factor=tissue_factor)
    add_fxi_feedback!(sys_with, true)
    times_with, results_with = simulate_extended_coagulation!(sys_with, tspan, dt)

    # Without FXI
    sys_without = create_extended_coagulation(tissue_factor=tissue_factor)
    add_fxi_feedback!(sys_without, false)
    times_without, results_without = simulate_extended_coagulation!(sys_without, tspan, dt)

    # Extract peak thrombin
    peak_with = maximum([r["thrombin_nM"] for r in results_with])
    peak_without = maximum([r["thrombin_nM"] for r in results_without])

    return Dict(
        "with_FXI" => Dict(
            "peak_thrombin" => peak_with,
            "FXI_contribution" => sys_with.fxi_contribution,
            "times" => times_with,
            "results" => results_with
        ),
        "without_FXI" => Dict(
            "peak_thrombin" => peak_without,
            "times" => times_without,
            "results" => results_without
        ),
        "FXI_amplification" => peak_with / max(peak_without, 0.01)
    )
end

end  # module CoagulationExtended
