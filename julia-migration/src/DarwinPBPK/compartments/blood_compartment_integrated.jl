"""
Blood Compartment Integrated Module

Unifies all blood-related modules into a coherent system that:
1. Propagates hematocrit → viscosity → perfusion
2. Connects disease ontology → binding adjustments
3. Time-evolves acute phase response
4. Provides dynamic PK parameters to ODE solver

This module is the INTEGRATION LAYER between isolated blood modules.

Integration Map:
┌─────────────────────────────────────────────────────────────────┐
│                    BloodCompartmentState                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Anemia/    │───▶│   Plasma     │───▶│  Perfusion   │       │
│  │ Polycythemia │    │  Viscosity   │    │   Effects    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ RBC Aging &  │───▶│    Blood     │───▶│ ODE Solver   │       │
│  │ Transporters │    │   Binding    │    │ Parameters   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         ▲                   ▲                   ▲               │
│         │                   │                   │               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Disease    │───▶│ Acute Phase  │───▶│  Time-Dep    │       │
│  │  Ontology    │    │  Response    │    │  Parameters  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘

Author: Darwin PBPK Platform
Date: 2025-12-06
"""
module BloodCompartmentIntegrated

using Statistics

# Import from sibling modules (will be loaded by parent)
# These are the modules we're integrating

export BloodCompartmentState, DrugBloodProperties, IntegratedPKAdjustments
export create_blood_state, create_blood_state_from_disease
export update_blood_state!, get_current_adjustments
export calculate_integrated_pk_parameters
export apply_time_step!, get_ode_parameters
export validate_blood_state, get_integration_summary

# ============================================================================
# CONSTANTS - Reference values for normalization
# ============================================================================

const REFERENCE_VALUES = Dict{Symbol, Float64}(
    # Hematology
    :hematocrit => 0.42,
    :hemoglobin => 14.0,           # g/dL
    :rbc_count => 4.7e12,          # /L

    # Proteins
    :albumin => 40.0,              # g/L
    :aag => 0.8,                   # g/L (alpha-1 acid glycoprotein)
    :fibrinogen => 3.0,            # g/L
    :total_protein => 70.0,        # g/L

    # Perfusion
    :hepatic_flow => 90.0,         # L/h
    :portal_flow => 65.0,          # L/h
    :hepatic_arterial => 25.0,     # L/h
    :renal_flow => 72.0,           # L/h (both kidneys)
    :gfr => 100.0,                 # mL/min
    :cardiac_output => 300.0,      # L/h (5 L/min)

    # Viscosity
    :plasma_viscosity => 1.2,      # mPa·s
    :blood_viscosity => 3.5,       # mPa·s at 100/s shear

    # Blood binding
    :fu_reference => 0.5,          # Unbound fraction reference

    # Body composition
    :blood_volume => 5.0,          # L
    :plasma_volume => 3.0          # L
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    DrugBloodProperties

Drug-specific properties affecting blood compartment behavior.
"""
struct DrugBloodProperties
    name::String
    ke_p::Float64                  # RBC:plasma partition coefficient
    fu_plasma_reference::Float64   # Unbound fraction in plasma (normal)
    charge_type::Symbol            # :acidic, :basic, :neutral
    albumin_binding::Bool          # Binds to albumin
    aag_binding::Bool              # Binds to AAG
    rbc_saturable::Bool            # Saturable RBC binding
    km_rbc::Float64                # Michaelis constant for RBC (if saturable)
    extraction_ratio::Float64      # Hepatic extraction ratio
    renal_fraction::Float64        # Fraction cleared renally

    function DrugBloodProperties(name::String;
        ke_p::Float64 = 1.0,
        fu_plasma_reference::Float64 = 0.5,
        charge_type::Symbol = :neutral,
        albumin_binding::Bool = true,
        aag_binding::Bool = false,
        rbc_saturable::Bool = false,
        km_rbc::Float64 = 100.0,
        extraction_ratio::Float64 = 0.3,
        renal_fraction::Float64 = 0.3
    )
        new(name, ke_p, fu_plasma_reference, charge_type,
            albumin_binding, aag_binding, rbc_saturable, km_rbc,
            extraction_ratio, renal_fraction)
    end
end

"""
    BloodCompartmentState

Mutable state representing all blood compartment parameters at a point in time.
Updated during simulation to reflect disease progression and acute changes.
"""
mutable struct BloodCompartmentState
    # =========== Time tracking ===========
    time::Float64                  # Current simulation time (hours)
    time_since_onset::Float64      # Time since disease/inflammation onset

    # =========== Hematology ===========
    hematocrit::Float64
    hemoglobin::Float64            # g/dL
    reticulocyte_fraction::Float64
    rbc_mean_age_days::Float64     # Mean RBC age (affects transporter expression)

    # =========== Plasma proteins ===========
    albumin_g_L::Float64
    aag_g_L::Float64
    fibrinogen_g_L::Float64
    il6_pg_mL::Float64             # IL-6 for acute phase tracking
    crp_mg_L::Float64              # CRP as inflammation marker

    # =========== Viscosity (derived) ===========
    plasma_viscosity::Float64      # mPa·s
    blood_viscosity::Float64       # mPa·s at reference shear rate
    viscosity_factor::Float64      # vs normal

    # =========== Perfusion (derived) ===========
    hepatic_flow::Float64          # L/h
    portal_flow::Float64           # L/h
    renal_blood_flow::Float64      # L/h
    gfr::Float64                   # mL/min
    cardiac_output::Float64        # L/h

    # =========== Disease state ===========
    disease_doid::String           # DOID code (empty if none)
    disease_severity::Symbol       # :none, :mild, :moderate, :severe
    is_acute_phase::Bool           # Acute phase response active
    acute_phase_trigger::Symbol    # :none, :sepsis, :surgery, :trauma, etc.

    # =========== Binding state (derived) ===========
    fu_plasma::Float64             # Current unbound fraction in plasma
    fu_blood::Float64              # Current unbound fraction in blood
    rb_current::Float64            # Current blood:plasma ratio

    # =========== Adjustment factors (for ODE) ===========
    hepatic_cl_factor::Float64     # Multiplier for hepatic clearance
    renal_cl_factor::Float64       # Multiplier for renal clearance
    vd_factor::Float64             # Multiplier for volume of distribution
    absorption_factor::Float64     # Multiplier for oral absorption

    # =========== Validation flags ===========
    is_validated::Bool
    last_update_time::Float64

    function BloodCompartmentState()
        new(
            # Time
            0.0, 0.0,
            # Hematology
            0.42, 14.0, 0.01, 60.0,
            # Proteins
            40.0, 0.8, 3.0, 5.0, 1.0,
            # Viscosity
            1.2, 3.5, 1.0,
            # Perfusion
            90.0, 65.0, 72.0, 100.0, 300.0,
            # Disease
            "", :none, false, :none,
            # Binding
            0.5, 0.5, 1.0,
            # Adjustments
            1.0, 1.0, 1.0, 1.0,
            # Validation
            false, 0.0
        )
    end
end

"""
    IntegratedPKAdjustments

Output structure with all PK adjustments for a drug at current state.
"""
struct IntegratedPKAdjustments
    # Core adjustments
    vd_adjusted::Float64
    cl_hepatic_adjusted::Float64
    cl_renal_adjusted::Float64
    fu_adjusted::Float64
    bioavailability_adjusted::Float64

    # Blood-specific
    rb::Float64                    # Blood:plasma ratio
    ke_p_effective::Float64        # Effective RBC partition (may be Hct-adjusted)

    # Flow-based
    hepatic_flow_L_h::Float64
    renal_flow_L_h::Float64
    gfr_mL_min::Float64

    # Rationale
    adjustment_factors::Dict{Symbol, Float64}
    clinical_notes::Vector{String}
    evidence_level::Symbol
end

# ============================================================================
# STATE CREATION
# ============================================================================

"""
    create_blood_state()

Create normal blood compartment state.
"""
function create_blood_state()
    return BloodCompartmentState()
end

"""
    create_blood_state_from_disease(doid::String; severity::Symbol=:moderate)

Create blood state initialized from disease ontology.

# Example
```julia
state = create_blood_state_from_disease("DOID:0080559")  # Sepsis
state = create_blood_state_from_disease("DOID:784", severity=:severe)  # CKD
```
"""
function create_blood_state_from_disease(doid::String;
                                         severity::Symbol=:moderate,
                                         disease_db::Dict=Dict())
    state = BloodCompartmentState()
    state.disease_doid = doid
    state.disease_severity = severity

    # Look up disease profile
    # This would normally call get_pk_adjustments_by_doid from disease_ontology_pk
    # For now, we implement the key diseases inline

    disease_effects = get_disease_effects(doid, severity)

    # Apply disease effects
    state.gfr = REFERENCE_VALUES[:gfr] * disease_effects[:gfr_factor]
    state.hepatic_flow = REFERENCE_VALUES[:hepatic_flow] * disease_effects[:hepatic_factor]
    state.albumin_g_L = disease_effects[:albumin_g_L]
    state.aag_g_L = disease_effects[:aag_g_L]
    state.hematocrit = disease_effects[:hematocrit]

    if disease_effects[:is_acute_phase]
        state.is_acute_phase = true
        state.acute_phase_trigger = disease_effects[:acute_trigger]
        state.il6_pg_mL = disease_effects[:il6_level]
    end

    # Recalculate derived values
    update_derived_values!(state)

    return state
end

"""
    get_disease_effects(doid::String, severity::Symbol)

Get disease-specific effects on blood parameters.
"""
function get_disease_effects(doid::String, severity::Symbol)
    severity_multiplier = Dict(:mild => 0.7, :moderate => 1.0, :severe => 1.3)
    sev = get(severity_multiplier, severity, 1.0)

    # Disease-specific effects (subset - would come from disease_ontology_pk)
    effects = Dict{Symbol, Any}(
        :gfr_factor => 1.0,
        :hepatic_factor => 1.0,
        :albumin_g_L => 40.0,
        :aag_g_L => 0.8,
        :hematocrit => 0.42,
        :is_acute_phase => false,
        :acute_trigger => :none,
        :il6_level => 5.0
    )

    # CKD
    if doid == "DOID:784" || startswith(doid, "DOID:006068")
        effects[:gfr_factor] = 0.3 * (1.0 / sev)
        effects[:albumin_g_L] = 35.0 - 5.0 * sev
        effects[:aag_g_L] = 0.8 + 0.4 * sev
        effects[:hematocrit] = 0.28

    # Sepsis
    elseif doid == "DOID:0080559"
        effects[:gfr_factor] = 0.6 * (1.0 / sev)
        effects[:hepatic_factor] = 0.7 * (1.0 / sev)
        effects[:albumin_g_L] = 25.0 - 5.0 * sev
        effects[:aag_g_L] = 2.0 + 0.5 * sev
        effects[:hematocrit] = 0.30
        effects[:is_acute_phase] = true
        effects[:acute_trigger] = :sepsis
        effects[:il6_level] = 200.0 * sev

    # Cirrhosis
    elseif doid == "DOID:5082"
        effects[:gfr_factor] = 0.85
        effects[:hepatic_factor] = 0.4 * (1.0 / sev)
        effects[:albumin_g_L] = 25.0 - 7.0 * sev
        effects[:aag_g_L] = 0.5
        effects[:hematocrit] = 0.32

    # Heart failure
    elseif doid == "DOID:6000"
        effects[:gfr_factor] = 0.7 * (1.0 / sev)
        effects[:hepatic_factor] = 0.8 * (1.0 / sev)
        effects[:albumin_g_L] = 34.0
        effects[:aag_g_L] = 1.2
        effects[:hematocrit] = 0.38

    # Polycythemia vera
    elseif doid == "DOID:8997"
        effects[:hematocrit] = 0.55 + 0.05 * sev
    end

    return effects
end

# ============================================================================
# STATE UPDATE FUNCTIONS
# ============================================================================

"""
    update_blood_state!(state::BloodCompartmentState, dt::Float64)

Advance blood state by dt hours. Updates time-dependent parameters.
"""
function update_blood_state!(state::BloodCompartmentState, dt::Float64)
    state.time += dt
    state.last_update_time = state.time

    if state.is_acute_phase
        state.time_since_onset += dt
        update_acute_phase!(state, dt)
    end

    update_derived_values!(state)
    state.is_validated = true

    return state
end

"""
    update_acute_phase!(state::BloodCompartmentState, dt::Float64)

Update acute phase proteins over time.
IL-6 → CRP, SAA, AAG (↑), Albumin (↓)
"""
function update_acute_phase!(state::BloodCompartmentState, dt::Float64)
    t = state.time_since_onset
    il6 = state.il6_pg_mL

    # IL-6 decay (half-life ~2-4 hours initially, then slower)
    if t > 24.0
        il6_decay = 0.02  # Slower decay after 24h
    else
        il6_decay = 0.1   # Faster initially
    end
    state.il6_pg_mL = max(5.0, il6 * exp(-il6_decay * dt))

    # CRP kinetics (peaks at 48-72h)
    # Production rate proportional to IL-6
    crp_production = 0.5 * log10(max(1.0, state.il6_pg_mL))  # mg/L/h
    crp_clearance = 0.05 * state.crp_mg_L  # First-order elimination
    state.crp_mg_L = max(1.0, state.crp_mg_L + (crp_production - crp_clearance) * dt)
    state.crp_mg_L = min(state.crp_mg_L, 500.0)  # Cap at severe levels

    # AAG kinetics (peaks at 48-96h)
    # Upregulated by IL-6
    aag_target = 0.8 + 2.0 * (state.il6_pg_mL / 200.0)  # Max ~2.8 g/L
    aag_target = min(aag_target, 3.0)
    aag_rate = 0.02  # Approach target over ~50h
    state.aag_g_L = state.aag_g_L + (aag_target - state.aag_g_L) * aag_rate * dt

    # Albumin kinetics (decreases during inflammation)
    alb_target = 40.0 - 15.0 * (state.il6_pg_mL / 200.0)  # Down to ~25 g/L
    alb_target = max(alb_target, 20.0)
    alb_rate = 0.01  # Slower change
    state.albumin_g_L = state.albumin_g_L + (alb_target - state.albumin_g_L) * alb_rate * dt
end

"""
    update_derived_values!(state::BloodCompartmentState)

Recalculate all derived values from primary parameters.
"""
function update_derived_values!(state::BloodCompartmentState)
    # 1. Viscosity from hematocrit
    hct = state.hematocrit
    state.plasma_viscosity = calculate_plasma_viscosity_internal(
        state.fibrinogen_g_L, state.albumin_g_L
    )
    state.blood_viscosity = calculate_blood_viscosity_internal(hct, state.plasma_viscosity)
    state.viscosity_factor = state.blood_viscosity / REFERENCE_VALUES[:blood_viscosity]

    # 2. Perfusion from viscosity
    viscosity_effect = REFERENCE_VALUES[:blood_viscosity] / state.blood_viscosity

    # Hepatic flow (moderate autoregulation)
    state.hepatic_flow = REFERENCE_VALUES[:hepatic_flow] * (viscosity_effect ^ 0.7)
    state.portal_flow = REFERENCE_VALUES[:portal_flow] * (viscosity_effect ^ 0.8)

    # Renal flow (strong autoregulation for GFR)
    renal_visc_effect = viscosity_effect ^ 0.4
    state.renal_blood_flow = REFERENCE_VALUES[:renal_flow] * (viscosity_effect ^ 0.6)
    # GFR already set by disease, only adjust for viscosity if no disease
    if state.disease_doid == ""
        state.gfr = REFERENCE_VALUES[:gfr] * renal_visc_effect
    end

    # 3. Adjustment factors for ODE
    state.hepatic_cl_factor = state.hepatic_flow / REFERENCE_VALUES[:hepatic_flow]
    state.renal_cl_factor = state.gfr / REFERENCE_VALUES[:gfr]

    # 4. Basic fu adjustment (will be refined for specific drugs)
    # Albumin effect on acidic drugs
    albumin_ratio = state.albumin_g_L / REFERENCE_VALUES[:albumin]
    # AAG effect on basic drugs
    aag_ratio = state.aag_g_L / REFERENCE_VALUES[:aag]

    # Average fu change (drug-specific calculation done in calculate_integrated_pk_parameters)
    state.fu_plasma = 0.5 / albumin_ratio  # Simplified, actual is drug-specific
    state.fu_blood = state.fu_plasma  # Will be adjusted for RBC binding
end

"""
    calculate_plasma_viscosity_internal(fibrinogen::Float64, albumin::Float64)

Internal viscosity calculation.
"""
function calculate_plasma_viscosity_internal(fibrinogen::Float64, albumin::Float64)
    base = 1.0  # mPa·s water contribution
    fib_contribution = fibrinogen * 0.07  # Per g/L
    return base + fib_contribution
end

"""
    calculate_blood_viscosity_internal(hematocrit::Float64, plasma_visc::Float64)

Internal blood viscosity calculation using exponential model.
μ = μ_plasma × exp(k × Hct/(1-Hct))
"""
function calculate_blood_viscosity_internal(hematocrit::Float64, plasma_visc::Float64)
    k = 2.5  # Einstein coefficient
    return plasma_visc * exp(k * hematocrit / (1.0 - hematocrit))
end

# ============================================================================
# PK PARAMETER CALCULATION
# ============================================================================

"""
    calculate_integrated_pk_parameters(state::BloodCompartmentState,
                                        drug::DrugBloodProperties,
                                        base_params::Dict)

Calculate fully integrated PK parameters for a drug given current blood state.

# Arguments
- `state`: Current blood compartment state
- `drug`: Drug-specific blood properties
- `base_params`: Dict with :vd, :cl_hepatic, :cl_renal, :bioavailability (reference values)

# Returns
IntegratedPKAdjustments with all adjusted parameters
"""
function calculate_integrated_pk_parameters(state::BloodCompartmentState,
                                            drug::DrugBloodProperties,
                                            base_params::Dict)
    # Extract base parameters
    vd_base = get(base_params, :vd, 1.0)
    cl_hep_base = get(base_params, :cl_hepatic, 1.0)
    cl_ren_base = get(base_params, :cl_renal, 1.0)
    f_base = get(base_params, :bioavailability, 1.0)

    notes = String[]
    factors = Dict{Symbol, Float64}()

    # ===== 1. Blood-plasma ratio (Rb) =====
    hct = state.hematocrit
    ke_p = drug.ke_p

    # Hematocrit correction for Ke_p if saturable
    ke_p_effective = ke_p
    if drug.rbc_saturable
        # At high concentrations, saturation reduces effective Ke_p
        # Simplified: use fixed Ke_p for now
        ke_p_effective = ke_p
    end

    rb = 1.0 - hct + (hct * ke_p_effective)
    factors[:rb] = rb

    if ke_p > 5.0
        push!(notes, "High RBC partitioning (Ke_p=$(round(ke_p, digits=1))): TDM should use blood concentrations")
    end

    # ===== 2. Unbound fraction (fu) =====
    fu_plasma = drug.fu_plasma_reference

    # Adjust for protein changes
    if drug.charge_type == :acidic && drug.albumin_binding
        # Acidic drugs: albumin binding
        albumin_ratio = state.albumin_g_L / REFERENCE_VALUES[:albumin]
        # fu increases when albumin decreases
        fu_plasma = drug.fu_plasma_reference / albumin_ratio
        fu_plasma = min(fu_plasma, 1.0)  # Cap at 100%
        factors[:fu_albumin_effect] = 1.0 / albumin_ratio

        if albumin_ratio < 0.75
            push!(notes, "Hypoalbuminemia: fu increased $(round((1.0/albumin_ratio - 1.0)*100))%")
        end

    elseif drug.charge_type == :basic && drug.aag_binding
        # Basic drugs: AAG binding
        aag_ratio = state.aag_g_L / REFERENCE_VALUES[:aag]
        # fu decreases when AAG increases
        fu_plasma = drug.fu_plasma_reference * (REFERENCE_VALUES[:aag] / state.aag_g_L)
        fu_plasma = max(fu_plasma, 0.01)  # Minimum 1% unbound
        factors[:fu_aag_effect] = REFERENCE_VALUES[:aag] / state.aag_g_L

        if aag_ratio > 1.5
            push!(notes, "Elevated AAG: fu decreased $(round((1.0 - factors[:fu_aag_effect])*100))%")
        end
    end

    # Blood fu considers RBC binding
    fu_blood = fu_plasma / rb
    factors[:fu_plasma] = fu_plasma
    factors[:fu_blood] = fu_blood

    # ===== 3. Volume of distribution =====
    vd_adjusted = vd_base

    # Hematocrit effect on Vd for high Ke_p drugs
    if ke_p > 2.0
        hct_reference = REFERENCE_VALUES[:hematocrit]
        rb_reference = 1.0 - hct_reference + (hct_reference * ke_p)
        vd_factor_hct = rb / rb_reference
        vd_adjusted *= vd_factor_hct
        factors[:vd_hct_effect] = vd_factor_hct
    end

    # fu effect on Vd (fu ↑ → Vd ↑ for restrictive binding)
    fu_ratio = fu_plasma / drug.fu_plasma_reference
    vd_adjusted *= fu_ratio ^ 0.5  # Partial effect
    factors[:vd_fu_effect] = fu_ratio ^ 0.5

    # ===== 4. Hepatic clearance =====
    # Well-stirred model: CL_H = Q_H × fu_b × CL_int / (Q_H + fu_b × CL_int)

    # For high extraction drugs, flow-limited
    if drug.extraction_ratio > 0.7
        # Flow-limited: CL ≈ Q_H
        cl_hep_adjusted = cl_hep_base * (state.hepatic_flow / REFERENCE_VALUES[:hepatic_flow])
        factors[:cl_hep_flow_effect] = state.hepatic_flow / REFERENCE_VALUES[:hepatic_flow]
        push!(notes, "High extraction drug: clearance flow-limited")
    else
        # Capacity-limited: CL ≈ fu_b × CL_int
        fu_effect = fu_blood / (drug.fu_plasma_reference / 1.0)  # Reference Rb ≈ 1
        flow_effect = state.hepatic_cl_factor ^ 0.3  # Partial flow effect
        cl_hep_adjusted = cl_hep_base * fu_effect * flow_effect
        factors[:cl_hep_fu_effect] = fu_effect
        factors[:cl_hep_flow_effect] = flow_effect
    end

    # ===== 5. Renal clearance =====
    # Primarily GFR-dependent for filtered drugs
    cl_ren_adjusted = cl_ren_base * state.renal_cl_factor

    # fu effect on filtration (only unbound drug filtered)
    cl_ren_adjusted *= (fu_plasma / drug.fu_plasma_reference)
    factors[:cl_ren_gfr_effect] = state.renal_cl_factor
    factors[:cl_ren_fu_effect] = fu_plasma / drug.fu_plasma_reference

    if state.gfr < 30.0
        push!(notes, "Severe renal impairment (GFR $(round(state.gfr)) mL/min): major dose reduction needed")
    elseif state.gfr < 60.0
        push!(notes, "Moderate renal impairment (GFR $(round(state.gfr)) mL/min): consider dose adjustment")
    end

    # ===== 6. Bioavailability =====
    f_adjusted = f_base * state.absorption_factor

    # Disease effects
    if state.disease_severity == :severe
        f_adjusted *= 0.8  # Reduced GI function
        push!(notes, "Severe disease may reduce oral absorption")
    end

    # ===== 7. Compile evidence level =====
    evidence = :moderate
    if state.disease_doid in ["DOID:0080559", "DOID:784", "DOID:5082"]
        evidence = :high  # Well-studied populations
    end
    if state.is_acute_phase && state.time_since_onset < 6.0
        evidence = :low  # Rapid changes, uncertain
        push!(notes, "Early acute phase: parameters changing rapidly, recheck in 6-12h")
    end

    return IntegratedPKAdjustments(
        vd_adjusted,
        cl_hep_adjusted,
        cl_ren_adjusted,
        fu_plasma,
        f_adjusted,
        rb,
        ke_p_effective,
        state.hepatic_flow,
        state.renal_blood_flow,
        state.gfr,
        factors,
        notes,
        evidence
    )
end

# ============================================================================
# ODE INTERFACE
# ============================================================================

"""
    get_ode_parameters(state::BloodCompartmentState, drug::DrugBloodProperties)

Get parameters formatted for ODE solver integration.

Returns Dict with:
- :hepatic_flow (L/h)
- :renal_flow (L/h)
- :fu_blood
- :rb
- :clearance_factors (hepatic, renal)
"""
function get_ode_parameters(state::BloodCompartmentState, drug::DrugBloodProperties)
    # Ensure state is current
    if !state.is_validated
        update_derived_values!(state)
    end

    # Calculate Rb
    rb = 1.0 - state.hematocrit + (state.hematocrit * drug.ke_p)

    # Calculate fu adjustments
    fu_plasma = drug.fu_plasma_reference
    if drug.charge_type == :acidic && drug.albumin_binding
        fu_plasma = drug.fu_plasma_reference * (REFERENCE_VALUES[:albumin] / state.albumin_g_L)
    elseif drug.charge_type == :basic && drug.aag_binding
        fu_plasma = drug.fu_plasma_reference * (REFERENCE_VALUES[:aag] / state.aag_g_L)
    end
    fu_blood = fu_plasma / rb

    return Dict{Symbol, Any}(
        :hepatic_flow => state.hepatic_flow,
        :portal_flow => state.portal_flow,
        :renal_flow => state.renal_blood_flow,
        :gfr => state.gfr,
        :fu_blood => fu_blood,
        :fu_plasma => fu_plasma,
        :rb => rb,
        :viscosity_factor => state.viscosity_factor,
        :hepatic_cl_factor => state.hepatic_cl_factor,
        :renal_cl_factor => state.renal_cl_factor,
        :vd_factor => state.vd_factor,
        :time => state.time,
        :is_acute_phase => state.is_acute_phase
    )
end

"""
    apply_time_step!(state::BloodCompartmentState, dt::Float64, drug::DrugBloodProperties)

Apply a time step and return ODE parameters for that step.
Convenience function for ODE solver callbacks.
"""
function apply_time_step!(state::BloodCompartmentState, dt::Float64, drug::DrugBloodProperties)
    update_blood_state!(state, dt)
    return get_ode_parameters(state, drug)
end

# ============================================================================
# VALIDATION & DIAGNOSTICS
# ============================================================================

"""
    validate_blood_state(state::BloodCompartmentState)

Check if blood state is physiologically valid.
Returns (is_valid::Bool, issues::Vector{String})
"""
function validate_blood_state(state::BloodCompartmentState)
    issues = String[]

    # Hematocrit bounds
    if state.hematocrit < 0.15
        push!(issues, "Hematocrit critically low ($(state.hematocrit)) - life-threatening anemia")
    elseif state.hematocrit > 0.65
        push!(issues, "Hematocrit dangerously high ($(state.hematocrit)) - severe polycythemia")
    end

    # Albumin bounds
    if state.albumin_g_L < 15.0
        push!(issues, "Albumin critically low ($(state.albumin_g_L) g/L)")
    elseif state.albumin_g_L > 55.0
        push!(issues, "Albumin unexpectedly high ($(state.albumin_g_L) g/L)")
    end

    # GFR bounds
    if state.gfr < 5.0
        push!(issues, "GFR critically low ($(state.gfr) mL/min) - dialysis indicated")
    end

    # Hepatic flow bounds
    if state.hepatic_flow < 20.0
        push!(issues, "Hepatic flow critically low ($(state.hepatic_flow) L/h)")
    end

    # Viscosity bounds
    if state.blood_viscosity > 15.0
        push!(issues, "Blood viscosity very high ($(state.blood_viscosity) mPa·s) - hyperviscosity syndrome")
    end

    return (isempty(issues), issues)
end

"""
    get_integration_summary(state::BloodCompartmentState)

Get human-readable summary of current blood state and adjustments.
"""
function get_integration_summary(state::BloodCompartmentState)
    Dict(
        :time_hours => state.time,
        :disease => state.disease_doid == "" ? "None" : state.disease_doid,
        :severity => state.disease_severity,
        :hematology => Dict(
            :hematocrit => state.hematocrit,
            :hemoglobin => state.hemoglobin,
            :viscosity_factor => round(state.viscosity_factor, digits=2)
        ),
        :proteins => Dict(
            :albumin_g_L => state.albumin_g_L,
            :aag_g_L => state.aag_g_L,
            :crp_mg_L => round(state.crp_mg_L, digits=1)
        ),
        :perfusion => Dict(
            :hepatic_flow_L_h => round(state.hepatic_flow, digits=1),
            :gfr_mL_min => round(state.gfr, digits=1)
        ),
        :adjustments => Dict(
            :hepatic_cl => round(state.hepatic_cl_factor, digits=2),
            :renal_cl => round(state.renal_cl_factor, digits=2)
        ),
        :acute_phase => state.is_acute_phase ? Dict(
            :trigger => state.acute_phase_trigger,
            :hours_since_onset => round(state.time_since_onset, digits=1),
            :il6_pg_mL => round(state.il6_pg_mL, digits=1)
        ) : nothing
    )
end

# ============================================================================
# DISEASE ONTOLOGY BRIDGE
# ============================================================================
# These functions connect disease_ontology_pk.jl with disease_state_binding.jl
# enabling seamless DOID/ICD -> PK adjustments workflow

export create_state_from_doid_profile, map_ontology_to_binding_state
export get_binding_adjustments_by_doid, calculate_fu_from_disease_code

"""
    DOID_TO_BINDING_STATE_MAP

Maps DOID disease codes to DiseaseStateBinding symbols.
Bridges the two modules for unified disease handling.
"""
const DOID_TO_BINDING_STATE_MAP = Dict{String, Symbol}(
    # Renal diseases
    "DOID:784" => :ckd_stage3,          # CKD general -> stage 3 as moderate
    "DOID:0060681" => :ckd_stage3,      # CKD stage 3
    "DOID:0060682" => :ckd_stage4,      # CKD stage 4
    "DOID:783" => :esrd,                # ESRD
    "DOID:1074" => :aki,                # AKI

    # Hepatic diseases
    "DOID:5082" => :cirrhosis_child_b,  # Cirrhosis -> Child B as moderate
    "DOID:9452" => :cirrhosis_child_a,  # Alcoholic liver disease
    "DOID:0080208" => :nafld,           # NAFLD

    # Diabetes
    "DOID:9351" => :diabetes_t2,        # DM general
    "DOID:9744" => :diabetes_t1,        # T1DM
    "DOID:9352" => :diabetes_t2,        # T2DM

    # Cardiovascular
    "DOID:6000" => :elderly,            # Heart failure (uses elderly as proxy for reduced perfusion)

    # Inflammatory
    "DOID:7148" => :rheumatoid_arthritis,  # RA
    "DOID:9074" => :sle,                   # SLE
    "DOID:0050589" => :ibd,                # IBD

    # Critical illness
    "DOID:0080559" => :sepsis,          # Sepsis
    "DOID:0050805" => :burn,            # Burns

    # Metabolic
    "DOID:9970" => :obesity,            # Obesity

    # Oncology
    "DOID:162" => :cancer_cachexia,     # Cancer

    # Pregnancy
    "DOID:0060088" => :pregnancy_t2     # Pregnancy -> T2 as moderate
)

"""
    map_ontology_to_binding_state(doid::String; severity::Symbol=:moderate)

Map a DOID code to DiseaseStateBinding state symbol.
Returns the appropriate symbol for create_disease_state().
"""
function map_ontology_to_binding_state(doid::String; severity::Symbol=:moderate)
    # Normalize DOID format
    if !startswith(doid, "DOID:")
        doid = "DOID:" * doid
    end

    if haskey(DOID_TO_BINDING_STATE_MAP, doid)
        base_state = DOID_TO_BINDING_STATE_MAP[doid]

        # Adjust for severity where applicable
        if doid == "DOID:784" && severity == :severe
            return :ckd_stage4
        elseif doid == "DOID:784" && severity == :mild
            return :ckd_stage2
        elseif doid == "DOID:5082" && severity == :severe
            return :cirrhosis_child_c
        elseif doid == "DOID:5082" && severity == :mild
            return :cirrhosis_child_a
        elseif doid == "DOID:0060088" && severity == :mild
            return :pregnancy_t1
        elseif doid == "DOID:0060088" && severity == :severe
            return :pregnancy_t3
        end

        return base_state
    end

    return :normal  # Default to normal if not found
end

"""
    create_state_from_doid_profile(doid::String, ontology_profile::Any;
                                    severity::Symbol=:moderate)

Create BloodCompartmentState from disease ontology PK profile.
Merges data from DiseasePKProfile (ontology) with detailed binding calculations.

# Arguments
- `doid`: DOID code
- `ontology_profile`: DiseasePKProfile from disease_ontology_pk module
- `severity`: Disease severity level
"""
function create_state_from_doid_profile(doid::String, ontology_profile;
                                         severity::Symbol=:moderate)
    state = create_blood_state_from_disease(doid, severity=severity)

    # Override with ontology-specific values if available
    if ontology_profile !== nothing
        # Use ontology albumin/AAG if specified
        if hasproperty(ontology_profile, :albumin_concentration)
            state.albumin_g_L = ontology_profile.albumin_concentration
        end
        if hasproperty(ontology_profile, :aag_concentration)
            state.aag_g_L = ontology_profile.aag_concentration
        end

        # Apply GFR adjustment
        if hasproperty(ontology_profile, :gfr_adjustment)
            state.gfr = REFERENCE_VALUES[:gfr] * ontology_profile.gfr_adjustment
        end

        # Apply hepatic adjustment
        if hasproperty(ontology_profile, :hepatic_adjustment)
            state.hepatic_flow = REFERENCE_VALUES[:hepatic_flow] * ontology_profile.hepatic_adjustment
        end

        # Recalculate derived values
        update_derived_values!(state)
    end

    return state
end

"""
    get_binding_adjustments_by_doid(doid::String, drug_type::Symbol;
                                     severity::Symbol=:moderate)

Get binding adjustment factors using DOID code.
Bridges disease_ontology_pk to disease_state_binding.

# Arguments
- `doid`: DOID code (e.g., "DOID:0080559" for sepsis)
- `drug_type`: :acidic, :basic, :neutral
- `severity`: :mild, :moderate, :severe

# Returns
Dict with adjustment factors:
- :fu_factor - Multiplier for fraction unbound
- :vd_factor - Volume of distribution adjustment
- :cl_factor - Clearance adjustment
- :binding_state - The underlying disease state used
"""
function get_binding_adjustments_by_doid(doid::String, drug_type::Symbol;
                                          severity::Symbol=:moderate)
    # Map DOID to binding state
    binding_state_symbol = map_ontology_to_binding_state(doid, severity=severity)

    # Get adjustment factors based on state
    state = create_blood_state_from_disease(doid, severity=severity)

    # Calculate fu adjustment based on drug type
    fu_factor = 1.0
    if drug_type == :acidic
        albumin_ratio = state.albumin_g_L / REFERENCE_VALUES[:albumin]
        fu_factor = 1.0 / albumin_ratio  # fu increases when albumin decreases
    elseif drug_type == :basic
        aag_ratio = state.aag_g_L / REFERENCE_VALUES[:aag]
        fu_factor = 1.0 / aag_ratio  # fu decreases when AAG increases
    else  # neutral
        albumin_ratio = state.albumin_g_L / REFERENCE_VALUES[:albumin]
        aag_ratio = state.aag_g_L / REFERENCE_VALUES[:aag]
        fu_factor = (1.0/albumin_ratio + 1.0/aag_ratio) / 2.0
    end

    # Calculate clearance and Vd factors
    cl_factor = (state.hepatic_cl_factor + state.renal_cl_factor) / 2.0
    vd_factor = sqrt(fu_factor)  # Vd increases with fu^0.5 for most drugs

    return Dict(
        :fu_factor => fu_factor,
        :vd_factor => vd_factor,
        :cl_factor => cl_factor,
        :hepatic_cl_factor => state.hepatic_cl_factor,
        :renal_cl_factor => state.renal_cl_factor,
        :binding_state => binding_state_symbol,
        :albumin_g_L => state.albumin_g_L,
        :aag_g_L => state.aag_g_L,
        :gfr => state.gfr
    )
end

"""
    calculate_fu_from_disease_code(fu_reference::Float64,
                                    drug_type::Symbol,
                                    doid::String;
                                    severity::Symbol=:moderate)

Calculate adjusted fraction unbound from disease code.

# Example
```julia
# Phenytoin (acidic) in cirrhosis
fu_adjusted = calculate_fu_from_disease_code(0.1, :acidic, "DOID:5082")

# Lidocaine (basic) in sepsis
fu_adjusted = calculate_fu_from_disease_code(0.3, :basic, "DOID:0080559")
```
"""
function calculate_fu_from_disease_code(fu_reference::Float64,
                                         drug_type::Symbol,
                                         doid::String;
                                         severity::Symbol=:moderate)
    adjustments = get_binding_adjustments_by_doid(doid, drug_type, severity=severity)
    fu_adjusted = fu_reference * adjustments[:fu_factor]

    # Constrain to valid range
    fu_adjusted = max(fu_adjusted, 0.001)
    fu_adjusted = min(fu_adjusted, 1.0)

    return fu_adjusted
end

# ============================================================================
# ICD CODE CONVENIENCE FUNCTIONS
# ============================================================================

"""
    ICD10_TO_DOID_QUICK

Quick ICD-10 to DOID mapping for common codes.
"""
const ICD10_TO_DOID_QUICK = Dict{String, String}(
    "N18" => "DOID:784",
    "N18.3" => "DOID:0060681",
    "N18.4" => "DOID:0060682",
    "N18.5" => "DOID:783",
    "N17" => "DOID:1074",
    "K74" => "DOID:5082",
    "E10" => "DOID:9744",
    "E11" => "DOID:9352",
    "I50" => "DOID:6000",
    "A41" => "DOID:0080559",
    "R65.2" => "DOID:0080559",
    "M05" => "DOID:7148",
    "M06" => "DOID:7148",
    "M32" => "DOID:9074",
    "K50" => "DOID:0050589",
    "K51" => "DOID:0050589",
    "E66" => "DOID:9970"
)

export get_binding_adjustments_by_icd10, create_state_from_icd10

"""
    get_binding_adjustments_by_icd10(icd10::String, drug_type::Symbol;
                                      severity::Symbol=:moderate)

Get binding adjustments using ICD-10 code.
"""
function get_binding_adjustments_by_icd10(icd10::String, drug_type::Symbol;
                                           severity::Symbol=:moderate)
    icd10_clean = uppercase(icd10)

    # Try exact match
    if haskey(ICD10_TO_DOID_QUICK, icd10_clean)
        doid = ICD10_TO_DOID_QUICK[icd10_clean]
        return get_binding_adjustments_by_doid(doid, drug_type, severity=severity)
    end

    # Try prefix match
    prefix = split(icd10_clean, ".")[1]
    if haskey(ICD10_TO_DOID_QUICK, prefix)
        doid = ICD10_TO_DOID_QUICK[prefix]
        return get_binding_adjustments_by_doid(doid, drug_type, severity=severity)
    end

    # Default: no adjustments
    return Dict(
        :fu_factor => 1.0,
        :vd_factor => 1.0,
        :cl_factor => 1.0,
        :hepatic_cl_factor => 1.0,
        :renal_cl_factor => 1.0,
        :binding_state => :normal,
        :albumin_g_L => 40.0,
        :aag_g_L => 0.8,
        :gfr => 100.0
    )
end

"""
    create_state_from_icd10(icd10::String; severity::Symbol=:moderate)

Create BloodCompartmentState from ICD-10 code.
"""
function create_state_from_icd10(icd10::String; severity::Symbol=:moderate)
    icd10_clean = uppercase(icd10)

    # Find DOID
    doid = ""
    if haskey(ICD10_TO_DOID_QUICK, icd10_clean)
        doid = ICD10_TO_DOID_QUICK[icd10_clean]
    else
        prefix = split(icd10_clean, ".")[1]
        if haskey(ICD10_TO_DOID_QUICK, prefix)
            doid = ICD10_TO_DOID_QUICK[prefix]
        end
    end

    if doid == ""
        return create_blood_state()  # Normal state
    end

    return create_blood_state_from_disease(doid, severity=severity)
end

end # module BloodCompartmentIntegrated
