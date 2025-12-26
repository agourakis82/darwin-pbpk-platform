# =============================================================================
# DDI-PBPK INTEGRATION MODULE
# =============================================================================
# Connects static DDI predictions with full ODE-based PBPK simulations
# for time-course concentration profiles under DDI conditions.
#
# KEY FEATURES:
# - Dynamic enzyme inhibition/induction over time
# - Competitive, MBI, and mixed mechanisms
# - Time-varying CLint based on perpetrator concentrations
# - Full Cp(t) profiles with and without DDI
# - Gut-wall (Fg) and hepatic (Fh) DDI separation
# - Multi-dose perpetrator scenarios
#
# Darwin PBPK Platform v2.10.0
# Author: Dr. Sounio Agourakis
# Date: November 2025
# =============================================================================

module DDIPBPKIntegration

using DifferentialEquations
using Statistics

# Import from parent modules
include("ddi_prediction.jl")
using .DDIPrediction

export DDIPBPKParams, DDIPBPKResult
export simulate_ddi_pbpk, simulate_ddi_comparison
export predict_ddi_time_course, calculate_dynamic_clint
export simulate_mbi_time_course, simulate_induction_onset
export GutWallDDI, separate_fg_fh, calculate_fg_inhibition

# =============================================================================
# CONSTANTS - PHYSIOLOGICAL PARAMETERS
# =============================================================================

const PHYSIOLOGY = (
    # Blood flows (L/h) - 70kg adult
    Qh = 90.0,              # Hepatic blood flow
    Qgut = 18.0,            # Gut blood flow (portal)
    Qportal = 72.0,         # Total portal flow

    # Volumes (L)
    V_blood = 5.0,
    V_liver = 1.8,
    V_gut = 1.2,
    V_central = 15.0,       # Typical Vc
    V_peripheral = 50.0,    # Typical Vp

    # Enzyme concentrations
    CYP3A4_liver_pmol_mg = 137.0,
    CYP3A4_gut_pmol_mg = 70.0,    # ~50% of liver
    MPPGL = 40.0,                  # mg microsomal protein per g liver
    MPPGG = 20.0,                  # mg microsomal protein per g gut

    # Degradation rates (h⁻¹)
    kdeg_CYP3A4_liver = 0.019,    # t½ = 36h
    kdeg_CYP3A4_gut = 0.029,      # t½ = 24h (faster turnover)
    kdeg_CYP2D6 = 0.029,
    kdeg_CYP2C9 = 0.014,
    kdeg_CYP1A2 = 0.014,
)

# =============================================================================
# DDI-PBPK PARAMETER STRUCTURES
# =============================================================================

"""
Complete parameters for DDI-PBPK simulation.
"""
struct DDIPBPKParams
    # Victim drug PK
    victim_name::Symbol
    victim_dose_mg::Float64
    victim_ka::Float64           # Absorption rate (h⁻¹)
    victim_F::Float64            # Baseline bioavailability
    victim_Fg::Float64           # Gut bioavailability
    victim_Fh::Float64           # Hepatic bioavailability
    victim_Vc_L::Float64         # Central volume
    victim_Vp_L::Float64         # Peripheral volume
    victim_Q_L_h::Float64        # Intercompartmental clearance
    victim_CL_L_h::Float64       # Total clearance
    victim_CLh_L_h::Float64      # Hepatic clearance
    victim_CLr_L_h::Float64      # Renal clearance
    victim_fm::Dict{Symbol, Float64}  # fm by enzyme

    # Perpetrator drug PK
    perpetrator_name::Symbol
    perpetrator_dose_mg::Float64
    perpetrator_interval_h::Float64  # Dosing interval
    perpetrator_n_doses::Int
    perpetrator_ka::Float64
    perpetrator_Vc_L::Float64
    perpetrator_CL_L_h::Float64
    perpetrator_fu::Float64      # Fraction unbound

    # DDI mechanism parameters
    mechanism::Symbol            # :reversible, :mbi, :induction, :mixed
    Ki_uM::Float64              # Reversible inhibition Ki
    kinact_h::Float64           # MBI inactivation rate
    KI_uM::Float64              # MBI KI
    Emax::Float64               # Induction Emax
    EC50_uM::Float64            # Induction EC50
    affected_enzyme::Symbol     # Primary enzyme affected

    # Simulation settings
    inhibit_gut::Bool           # Include gut-wall DDI?
    inhibit_liver::Bool         # Include hepatic DDI?
end

"""
Result of DDI-PBPK simulation.
"""
struct DDIPBPKResult
    # Time courses
    time_h::Vector{Float64}

    # Victim concentrations
    Cp_victim_alone::Vector{Float64}       # Without perpetrator
    Cp_victim_ddi::Vector{Float64}         # With perpetrator

    # Perpetrator concentrations
    Cp_perpetrator::Vector{Float64}
    Cu_perpetrator::Vector{Float64}        # Unbound

    # Dynamic enzyme activity
    enzyme_activity_liver::Vector{Float64} # Fraction of baseline
    enzyme_activity_gut::Vector{Float64}

    # PK metrics - alone
    auc_alone::Float64
    cmax_alone::Float64
    tmax_alone::Float64
    half_life_alone::Float64

    # PK metrics - with DDI
    auc_ddi::Float64
    cmax_ddi::Float64
    tmax_ddi::Float64
    half_life_ddi::Float64

    # DDI magnitude
    auc_ratio::Float64
    cmax_ratio::Float64
    cl_ratio::Float64

    # Contributions
    fg_ratio::Float64           # Gut bioavailability change
    fh_ratio::Float64           # Hepatic bioavailability change
    clh_ratio::Float64          # Hepatic clearance change

    # Metadata
    params::DDIPBPKParams
    mechanism_details::String
end

# =============================================================================
# DRUG PRESETS - VICTIM PARAMETERS
# =============================================================================

"""
Get preset PK parameters for common substrates.
"""
function get_victim_preset(drug::Symbol)::NamedTuple
    presets = Dict{Symbol, NamedTuple}(
        # CYP3A4 substrates
        :midazolam => (
            ka = 1.5, F = 0.44, Fg = 0.57, Fh = 0.77,
            Vc_L = 30.0, Vp_L = 50.0, Q_L_h = 25.0,
            CL_L_h = 25.0, CLh_L_h = 23.0, CLr_L_h = 2.0,
            fm = Dict(:CYP3A4 => 0.94, :CYP3A5 => 0.04)
        ),
        :triazolam => (
            ka = 2.0, F = 0.53, Fg = 0.70, Fh = 0.76,
            Vc_L = 35.0, Vp_L = 65.0, Q_L_h = 20.0,
            CL_L_h = 15.0, CLh_L_h = 14.0, CLr_L_h = 1.0,
            fm = Dict(:CYP3A4 => 0.92)
        ),
        :simvastatin => (
            ka = 1.0, F = 0.05, Fg = 0.30, Fh = 0.17,  # Low F, high first-pass
            Vc_L = 25.0, Vp_L = 200.0, Q_L_h = 30.0,
            CL_L_h = 40.0, CLh_L_h = 38.0, CLr_L_h = 2.0,
            fm = Dict(:CYP3A4 => 0.85, :CYP2C8 => 0.10)
        ),
        :atorvastatin => (
            ka = 0.8, F = 0.14, Fg = 0.50, Fh = 0.28,
            Vc_L = 20.0, Vp_L = 180.0, Q_L_h = 25.0,
            CL_L_h = 35.0, CLh_L_h = 33.0, CLr_L_h = 2.0,
            fm = Dict(:CYP3A4 => 0.70, :CYP2C8 => 0.20)
        ),

        # CYP2D6 substrates
        :dextromethorphan => (
            ka = 1.5, F = 0.60, Fg = 0.90, Fh = 0.67,
            Vc_L = 100.0, Vp_L = 300.0, Q_L_h = 50.0,
            CL_L_h = 80.0, CLh_L_h = 75.0, CLr_L_h = 5.0,
            fm = Dict(:CYP2D6 => 0.80, :CYP3A4 => 0.15)
        ),
        :metoprolol => (
            ka = 1.2, F = 0.50, Fg = 0.95, Fh = 0.53,
            Vc_L = 60.0, Vp_L = 200.0, Q_L_h = 40.0,
            CL_L_h = 60.0, CLh_L_h = 55.0, CLr_L_h = 5.0,
            fm = Dict(:CYP2D6 => 0.70)
        ),

        # CYP2C9 substrates
        :warfarin => (
            ka = 1.0, F = 0.95, Fg = 0.99, Fh = 0.96,  # High F, low extraction
            Vc_L = 8.0, Vp_L = 2.0, Q_L_h = 5.0,
            CL_L_h = 0.15, CLh_L_h = 0.14, CLr_L_h = 0.01,
            fm = Dict(:CYP2C9 => 0.85, :CYP3A4 => 0.10)
        ),
        :tolbutamide => (
            ka = 1.5, F = 0.93, Fg = 0.98, Fh = 0.95,
            Vc_L = 7.0, Vp_L = 3.0, Q_L_h = 4.0,
            CL_L_h = 0.8, CLh_L_h = 0.75, CLr_L_h = 0.05,
            fm = Dict(:CYP2C9 => 0.90)
        ),

        # CYP1A2 substrates
        :tizanidine => (
            ka = 1.0, F = 0.34, Fg = 0.80, Fh = 0.43,
            Vc_L = 120.0, Vp_L = 200.0, Q_L_h = 50.0,
            CL_L_h = 50.0, CLh_L_h = 48.0, CLr_L_h = 2.0,
            fm = Dict(:CYP1A2 => 0.95)
        ),
        :theophylline => (
            ka = 1.2, F = 0.96, Fg = 0.99, Fh = 0.97,
            Vc_L = 30.0, Vp_L = 10.0, Q_L_h = 10.0,
            CL_L_h = 2.5, CLh_L_h = 2.3, CLr_L_h = 0.2,
            fm = Dict(:CYP1A2 => 0.70, :CYP2E1 => 0.15)
        ),

        # CYP2C8 substrates
        :repaglinide => (
            ka = 2.0, F = 0.63, Fg = 0.90, Fh = 0.70,
            Vc_L = 30.0, Vp_L = 60.0, Q_L_h = 25.0,
            CL_L_h = 30.0, CLh_L_h = 27.0, CLr_L_h = 3.0,
            fm = Dict(:CYP2C8 => 0.60, :CYP3A4 => 0.30)
        ),
    )

    return get(presets, drug, presets[:midazolam])
end

"""
Get preset PK parameters for common perpetrators.
"""
function get_perpetrator_preset(drug::Symbol)::NamedTuple
    presets = Dict{Symbol, NamedTuple}(
        # Strong CYP3A4 inhibitors
        :itraconazole => (
            ka = 0.5, Vc_L = 800.0, CL_L_h = 20.0, fu = 0.002,
            mechanism = :reversible, Ki_uM = 0.003,
            kinact_h = 0.0, KI_uM = 0.0,
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP3A4, inhibit_gut = true
        ),
        :ketoconazole => (
            ka = 0.8, Vc_L = 200.0, CL_L_h = 10.0, fu = 0.01,
            mechanism = :reversible, Ki_uM = 0.015,
            kinact_h = 0.0, KI_uM = 0.0,
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP3A4, inhibit_gut = true
        ),
        :ritonavir => (
            ka = 1.0, Vc_L = 50.0, CL_L_h = 8.0, fu = 0.02,
            mechanism = :mbi, Ki_uM = 0.02,
            kinact_h = 2.0, KI_uM = 0.1,  # Strong MBI
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP3A4, inhibit_gut = true
        ),
        :clarithromycin => (
            ka = 1.5, Vc_L = 250.0, CL_L_h = 25.0, fu = 0.30,
            mechanism = :mbi, Ki_uM = 10.0,
            kinact_h = 0.5, KI_uM = 5.0,
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP3A4, inhibit_gut = true
        ),
        :erythromycin => (
            ka = 1.2, Vc_L = 60.0, CL_L_h = 40.0, fu = 0.20,
            mechanism = :mbi, Ki_uM = 20.0,
            kinact_h = 0.3, KI_uM = 10.0,
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP3A4, inhibit_gut = true
        ),

        # Moderate CYP3A4 inhibitors
        :fluconazole => (
            ka = 1.5, Vc_L = 50.0, CL_L_h = 1.5, fu = 0.90,
            mechanism = :reversible, Ki_uM = 10.0,  # CYP3A4
            kinact_h = 0.0, KI_uM = 0.0,
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP2C9, inhibit_gut = false  # Mainly hepatic
        ),
        :diltiazem => (
            ka = 1.0, Vc_L = 300.0, CL_L_h = 50.0, fu = 0.20,
            mechanism = :mbi, Ki_uM = 5.0,
            kinact_h = 0.2, KI_uM = 15.0,
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP3A4, inhibit_gut = true
        ),

        # CYP2D6 inhibitors
        :quinidine => (
            ka = 1.2, Vc_L = 200.0, CL_L_h = 20.0, fu = 0.10,
            mechanism = :reversible, Ki_uM = 0.05,  # Very potent
            kinact_h = 0.0, KI_uM = 0.0,
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP2D6, inhibit_gut = false
        ),
        :paroxetine => (
            ka = 0.8, Vc_L = 500.0, CL_L_h = 15.0, fu = 0.05,
            mechanism = :mbi, Ki_uM = 0.5,
            kinact_h = 0.1, KI_uM = 2.0,
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP2D6, inhibit_gut = false
        ),

        # CYP1A2 inhibitors
        :ciprofloxacin => (
            ka = 2.0, Vc_L = 200.0, CL_L_h = 30.0, fu = 0.60,
            mechanism = :reversible, Ki_uM = 0.6,  # Calibrated
            kinact_h = 0.0, KI_uM = 0.0,
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP1A2, inhibit_gut = false
        ),
        :fluvoxamine => (
            ka = 1.0, Vc_L = 1500.0, CL_L_h = 50.0, fu = 0.20,
            mechanism = :reversible, Ki_uM = 0.02,  # Very potent
            kinact_h = 0.0, KI_uM = 0.0,
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP1A2, inhibit_gut = false
        ),

        # CYP2C8 inhibitors
        :gemfibrozil => (
            ka = 1.5, Vc_L = 10.0, CL_L_h = 5.0, fu = 0.01,
            mechanism = :mbi, Ki_uM = 20.0,
            kinact_h = 0.1, KI_uM = 25.0,
            Emax = 0.0, EC50_uM = 0.0,
            enzyme = :CYP2C8, inhibit_gut = false
        ),

        # Strong inducers
        :rifampin => (
            ka = 1.0, Vc_L = 60.0, CL_L_h = 10.0, fu = 0.20,
            mechanism = :induction, Ki_uM = Inf,
            kinact_h = 0.0, KI_uM = 0.0,
            Emax = 10.0, EC50_uM = 0.3,  # Strong inducer
            enzyme = :CYP3A4, inhibit_gut = true
        ),
        :carbamazepine => (
            ka = 0.5, Vc_L = 80.0, CL_L_h = 5.0, fu = 0.25,
            mechanism = :induction, Ki_uM = Inf,
            kinact_h = 0.0, KI_uM = 0.0,
            Emax = 3.0, EC50_uM = 15.0,
            enzyme = :CYP3A4, inhibit_gut = true
        ),
    )

    return get(presets, drug, presets[:ketoconazole])
end

# =============================================================================
# GUT-WALL DDI MODEL
# =============================================================================

"""
Gut-wall DDI parameters and calculations.
"""
struct GutWallDDI
    Fg_baseline::Float64        # Baseline gut bioavailability
    CLgut_baseline::Float64     # Baseline gut clearance
    Egut_baseline::Float64      # Baseline gut extraction

    Ig_uM::Float64              # Gut lumen concentration (dose/250mL)
    Ih_uM::Float64              # Hepatic inlet concentration

    Ki_uM::Float64              # Inhibition constant

    Fg_inhibited::Float64       # Inhibited gut bioavailability
    Egut_inhibited::Float64
end

"""
Calculate gut-wall DDI effect.
Uses enterocyte concentration approximation: [I]gut ≈ Dose / 250mL
"""
function calculate_fg_inhibition(
    perpetrator_dose_mg::Float64,
    perpetrator_MW::Float64,
    Ki_uM::Float64,
    Fg_baseline::Float64
)::GutWallDDI
    # Gut lumen concentration estimate
    V_gut_lumen = 0.25  # L (250 mL)
    Ig_uM = (perpetrator_dose_mg / perpetrator_MW * 1000) / V_gut_lumen

    # Baseline gut extraction
    Egut_baseline = 1.0 - Fg_baseline

    # Competitive inhibition of gut CYP3A4
    # CLgut_inhibited = CLgut_baseline / (1 + [I]/Ki)
    inhibition_factor = 1.0 + Ig_uM / Ki_uM

    Egut_inhibited = Egut_baseline / inhibition_factor
    Fg_inhibited = 1.0 - Egut_inhibited

    # Cap at 1.0
    Fg_inhibited = min(Fg_inhibited, 1.0)

    # Hepatic inlet concentration (approximation)
    Ih_uM = Ig_uM * 0.1  # ~10% of gut reaches portal

    return GutWallDDI(
        Fg_baseline, 0.0, Egut_baseline,
        Ig_uM, Ih_uM, Ki_uM,
        Fg_inhibited, Egut_inhibited
    )
end

"""
Separate Fg and Fh contributions to total bioavailability DDI.
"""
function separate_fg_fh(
    Fg_baseline::Float64,
    Fh_baseline::Float64,
    Fg_ddi::Float64,
    Fh_ddi::Float64
)::NamedTuple
    F_baseline = Fg_baseline * Fh_baseline
    F_ddi = Fg_ddi * Fh_ddi

    auc_ratio_total = F_ddi / F_baseline
    fg_contribution = Fg_ddi / Fg_baseline
    fh_contribution = Fh_ddi / Fh_baseline

    return (
        F_baseline = F_baseline,
        F_ddi = F_ddi,
        auc_ratio_total = auc_ratio_total,
        fg_contribution = fg_contribution,
        fh_contribution = fh_contribution,
        pct_from_fg = (fg_contribution - 1.0) / (auc_ratio_total - 1.0) * 100,
        pct_from_fh = (fh_contribution - 1.0) / (auc_ratio_total - 1.0) * 100
    )
end

# =============================================================================
# DYNAMIC CLint CALCULATION
# =============================================================================

"""
Calculate dynamic CLint based on perpetrator concentration and mechanism.
"""
function calculate_dynamic_clint(
    CLint_baseline::Float64,
    Cu_perpetrator::Float64,      # Unbound perpetrator concentration
    mechanism::Symbol,
    Ki_uM::Float64,
    kinact_h::Float64,
    KI_uM::Float64,
    enzyme_fraction::Float64,     # Remaining active enzyme (0-1)
    Emax::Float64,
    EC50_uM::Float64
)::Float64

    if mechanism == :reversible
        # Competitive inhibition: CLint_eff = CLint / (1 + Cu/Ki)
        inhibition_factor = 1.0 + Cu_perpetrator / max(Ki_uM, 1e-6)
        return CLint_baseline / inhibition_factor

    elseif mechanism == :mbi
        # MBI: CLint proportional to remaining enzyme
        # Also consider reversible component if Ki < Inf
        reversible_factor = 1.0
        if Ki_uM < Inf && Ki_uM > 0
            reversible_factor = 1.0 / (1.0 + Cu_perpetrator / Ki_uM)
        end
        return CLint_baseline * enzyme_fraction * reversible_factor

    elseif mechanism == :induction
        # Induction: CLint increased proportional to induced enzyme
        induction_factor = 1.0 + Emax * Cu_perpetrator / (EC50_uM + Cu_perpetrator)
        return CLint_baseline * induction_factor

    elseif mechanism == :mixed
        # Mixed inhibition + induction (e.g., ritonavir)
        # Acute: inhibition dominates
        # Chronic: induction may partially offset
        inhibition_factor = 1.0 + Cu_perpetrator / max(Ki_uM, 1e-6)
        induction_factor = 1.0 + Emax * Cu_perpetrator / (EC50_uM + Cu_perpetrator + 1e-6)

        return CLint_baseline * induction_factor / inhibition_factor * enzyme_fraction
    end

    return CLint_baseline
end

# =============================================================================
# ODE SYSTEM FOR DDI-PBPK
# =============================================================================

"""
ODE system for victim drug PK with dynamic DDI.

State vector (7 elements):
- u[1]: Victim gut amount (mg)
- u[2]: Victim central concentration (μM)
- u[3]: Victim peripheral amount (mg)
- u[4]: Perpetrator gut amount (mg)
- u[5]: Perpetrator central concentration (μM)
- u[6]: Active enzyme fraction - liver
- u[7]: Active enzyme fraction - gut
"""
function ddi_pbpk_ode!(du, u, p, t)
    # Unpack parameters
    victim = p.victim
    perp = p.perpetrator
    ddi = p.ddi

    # State variables
    A_victim_gut = max(u[1], 0.0)
    C_victim_central = max(u[2], 0.0)
    A_victim_periph = max(u[3], 0.0)
    A_perp_gut = max(u[4], 0.0)
    C_perp_central = max(u[5], 0.0)
    E_liver = clamp(u[6], 0.0, 2.0)  # Can exceed 1 for induction
    E_gut = clamp(u[7], 0.0, 2.0)

    # Convert concentrations to amounts and vice versa
    A_victim_central = C_victim_central * victim.Vc_L * victim.MW / 1000  # mg

    # Perpetrator unbound concentration
    Cu_perp = C_perp_central * perp.fu

    # =================================
    # DYNAMIC ENZYME ACTIVITY
    # =================================

    # Enzyme degradation rate
    kdeg = get(Dict(
        :CYP3A4 => PHYSIOLOGY.kdeg_CYP3A4_liver,
        :CYP2D6 => PHYSIOLOGY.kdeg_CYP2D6,
        :CYP2C9 => PHYSIOLOGY.kdeg_CYP2C9,
        :CYP1A2 => PHYSIOLOGY.kdeg_CYP1A2,
        :CYP2C8 => PHYSIOLOGY.kdeg_CYP2C9,  # Similar
    ), ddi.enzyme, 0.02)

    kdeg_gut = kdeg * 1.5  # Faster turnover in gut

    # Enzyme dynamics based on mechanism
    if ddi.mechanism == :mbi
        # MBI: dE/dt = kdeg*(1-E) - kinact*[I]/(KI+[I])*E
        # Inactivation rate
        lambda_liver = ddi.kinact_h * Cu_perp / (ddi.KI_uM + Cu_perp + 1e-6)

        # Gut: use gut lumen concentration approximation
        # Convert mg to μM: (mg / 0.25 L) / MW * 1000 = μM
        Ig_gut = (A_perp_gut / 0.25) / perp.MW * 1000  # μM
        lambda_gut = ddi.kinact_h * Ig_gut / (ddi.KI_uM + Ig_gut + 1e-6)

        du[6] = kdeg * (1.0 - E_liver) - lambda_liver * E_liver
        du[7] = kdeg_gut * (1.0 - E_gut) - lambda_gut * E_gut

    elseif ddi.mechanism == :induction
        # Induction: delayed onset, increase enzyme
        # dE/dt = kdeg * (E_target - E)
        E_target_liver = 1.0 + ddi.Emax * Cu_perp / (ddi.EC50_uM + Cu_perp + 1e-6)
        E_target_gut = 1.0 + ddi.Emax * Cu_perp / (ddi.EC50_uM + Cu_perp + 1e-6)

        du[6] = kdeg * (E_target_liver - E_liver)
        du[7] = kdeg_gut * (E_target_gut - E_gut)

    else
        # Reversible: no enzyme change
        du[6] = 0.0
        du[7] = 0.0
    end

    # =================================
    # DYNAMIC CLEARANCE
    # =================================

    # Calculate effective Fg and Fh based on current enzyme activity
    # and perpetrator concentration

    # Gut-wall clearance (if applicable)
    if ddi.inhibit_gut
        # Ig for gut = remaining gut drug / 250mL, converted to μM
        # A_perp_gut is in mg, V_gut = 0.25 L, MW from perp.MW
        # [I]gut (μM) = (A_mg / V_L) / MW * 1000
        Ig_effective = (A_perp_gut / 0.25) / perp.MW * 1000  # μM
        CLgut_factor = if ddi.mechanism == :reversible
            1.0 / (1.0 + Ig_effective / (ddi.Ki_uM + 1e-6))
        else
            E_gut
        end
    else
        CLgut_factor = 1.0
    end

    # Effective gut bioavailability
    Egut_baseline = 1.0 - victim.Fg
    Egut_effective = Egut_baseline * CLgut_factor
    Fg_effective = 1.0 - Egut_effective

    # Hepatic clearance
    # For hepatic DDI, use hepatic inlet concentration [I]h
    # [I]h = [I]plasma + ka * Fa * Dose / (Qh * MW) * 1000 (portal contribution)
    # Simplified: use plasma + fraction of gut absorption
    Ih_portal = (A_perp_gut * perp.ka * 0.9) / (PHYSIOLOGY.Qh * perp.MW) * 1000  # μM contribution from portal
    Ih_total = Cu_perp + Ih_portal  # Total hepatic inlet unbound concentration

    if ddi.inhibit_liver
        CLh_factor = if ddi.mechanism == :reversible
            1.0 / (1.0 + Ih_total / (ddi.Ki_uM + 1e-6))
        elseif ddi.mechanism == :induction
            E_liver
        else
            E_liver * (1.0 / (1.0 + Ih_total / (max(ddi.Ki_uM, 1e-6))))
        end
    else
        CLh_factor = 1.0
    end

    CLh_effective = victim.CLh_L_h * CLh_factor
    CL_effective = CLh_effective + victim.CLr_L_h

    # Also calculate effective Fh (for high-extraction drugs)
    # Fh = 1 - Eh, and Eh changes with CLint inhibition
    Eh_baseline = 1.0 - victim.Fh
    Eh_effective = Eh_baseline * CLh_factor  # Reduced extraction
    Fh_effective = 1.0 - Eh_effective

    # =================================
    # VICTIM KINETICS
    # =================================

    # Gut: dA/dt = -ka * A
    du[1] = -victim.ka * A_victim_gut

    # Absorption rate to central (with current Fg and Fh)
    absorption_rate = victim.ka * A_victim_gut * Fg_effective * Fh_effective

    # Central compartment: dC/dt = (absorption - CL*C - Q*(C - Cp/Kp)) / Vc
    C_periph = A_victim_periph / victim.Vp_L * 1000 / victim.MW  # μM

    du[2] = (
        absorption_rate / victim.Vc_L * 1000 / victim.MW   # Absorption (μM/h)
        - CL_effective / victim.Vc_L * C_victim_central    # Elimination
        - victim.Q_L_h / victim.Vc_L * (C_victim_central - C_periph)  # Distribution
    )

    # Peripheral: dA/dt = Q * (C_central - C_periph) * Vc
    du[3] = victim.Q_L_h * (C_victim_central - C_periph) * victim.Vc_L * victim.MW / 1000

    # =================================
    # PERPETRATOR KINETICS
    # =================================

    # Simple 1-compartment for perpetrator
    du[4] = -perp.ka * A_perp_gut

    absorption_perp = perp.ka * A_perp_gut * 0.9  # Assume 90% F

    du[5] = (
        absorption_perp / perp.Vc_L * 1000 / perp.MW
        - perp.CL_L_h / perp.Vc_L * C_perp_central
    )

    return nothing
end

"""
ODE system for victim drug PK alone (no DDI).
"""
function victim_alone_ode!(du, u, p, t)
    victim = p.victim

    A_gut = max(u[1], 0.0)
    C_central = max(u[2], 0.0)
    A_periph = max(u[3], 0.0)

    C_periph = A_periph / victim.Vp_L * 1000 / victim.MW

    # Gut
    du[1] = -victim.ka * A_gut

    # Central
    absorption = victim.ka * A_gut * victim.Fg * victim.Fh

    du[2] = (
        absorption / victim.Vc_L * 1000 / victim.MW
        - victim.CL_L_h / victim.Vc_L * C_central
        - victim.Q_L_h / victim.Vc_L * (C_central - C_periph)
    )

    # Peripheral
    du[3] = victim.Q_L_h * (C_central - C_periph) * victim.Vc_L * victim.MW / 1000

    return nothing
end

# =============================================================================
# MAIN SIMULATION FUNCTIONS
# =============================================================================

"""
Build DDIPBPKParams from drug names and doses.
"""
function build_ddi_params(
    victim::Symbol,
    victim_dose_mg::Float64,
    perpetrator::Symbol,
    perpetrator_dose_mg::Float64;
    perpetrator_interval_h::Float64 = 24.0,
    perpetrator_n_doses::Int = 1,
    victim_MW::Float64 = 300.0,
    perpetrator_MW::Float64 = 400.0
)::DDIPBPKParams

    v = get_victim_preset(victim)
    p = get_perpetrator_preset(perpetrator)

    return DDIPBPKParams(
        # Victim
        victim, victim_dose_mg,
        v.ka, v.F, v.Fg, v.Fh,
        v.Vc_L, v.Vp_L, v.Q_L_h,
        v.CL_L_h, v.CLh_L_h, v.CLr_L_h,
        v.fm,
        # Perpetrator
        perpetrator, perpetrator_dose_mg, perpetrator_interval_h, perpetrator_n_doses,
        p.ka, p.Vc_L, p.CL_L_h, p.fu,
        # DDI mechanism
        p.mechanism, p.Ki_uM, p.kinact_h, p.KI_uM,
        p.Emax, p.EC50_uM, p.enzyme,
        # Settings
        p.inhibit_gut, true  # Always inhibit liver
    )
end

"""
Main DDI-PBPK simulation function.
Simulates victim drug PK with and without perpetrator.
"""
function simulate_ddi_pbpk(
    params::DDIPBPKParams;
    t_max_h::Float64 = 72.0,
    dt_h::Float64 = 0.1,
    victim_MW::Float64 = 300.0,
    perpetrator_MW::Float64 = 400.0
)::DDIPBPKResult

    time_points = collect(0.0:dt_h:t_max_h)
    n_points = length(time_points)

    # ================================
    # SIMULATE VICTIM ALONE
    # ================================

    victim_params = (
        victim = (
            ka = params.victim_ka,
            Fg = params.victim_Fg,
            Fh = params.victim_Fh,
            Vc_L = params.victim_Vc_L,
            Vp_L = params.victim_Vp_L,
            Q_L_h = params.victim_Q_L_h,
            CL_L_h = params.victim_CL_L_h,
            MW = victim_MW
        ),
    )

    # Initial conditions: all drug in gut
    u0_alone = [params.victim_dose_mg, 0.0, 0.0]

    prob_alone = ODEProblem(victim_alone_ode!, u0_alone, (0.0, t_max_h), victim_params)
    sol_alone = solve(prob_alone, Tsit5(), saveat=time_points, reltol=1e-8, abstol=1e-10)

    Cp_alone = [sol_alone[i][2] for i in 1:length(sol_alone)]

    # ================================
    # SIMULATE WITH DDI
    # ================================

    ddi_params = (
        victim = (
            ka = params.victim_ka,
            Fg = params.victim_Fg,
            Fh = params.victim_Fh,
            Vc_L = params.victim_Vc_L,
            Vp_L = params.victim_Vp_L,
            Q_L_h = params.victim_Q_L_h,
            CL_L_h = params.victim_CL_L_h,
            CLh_L_h = params.victim_CLh_L_h,
            CLr_L_h = params.victim_CLr_L_h,
            MW = victim_MW
        ),
        perpetrator = (
            ka = params.perpetrator_ka,
            Vc_L = params.perpetrator_Vc_L,
            CL_L_h = params.perpetrator_CL_L_h,
            fu = params.perpetrator_fu,
            MW = perpetrator_MW,
            dose_mg = params.perpetrator_dose_mg
        ),
        ddi = (
            mechanism = params.mechanism,
            Ki_uM = params.Ki_uM,
            kinact_h = params.kinact_h,
            KI_uM = params.KI_uM,
            Emax = params.Emax,
            EC50_uM = params.EC50_uM,
            enzyme = params.affected_enzyme,
            inhibit_gut = params.inhibit_gut,
            inhibit_liver = params.inhibit_liver
        )
    )

    # Initial conditions:
    # [victim_gut, victim_central, victim_periph, perp_gut, perp_central, E_liver, E_gut]
    u0_ddi = [
        params.victim_dose_mg,  # Victim in gut
        0.0,                     # Victim central
        0.0,                     # Victim peripheral
        params.perpetrator_dose_mg,  # Perpetrator in gut
        0.0,                     # Perpetrator central
        1.0,                     # Enzyme liver (100%)
        1.0                      # Enzyme gut (100%)
    ]

    # Handle multiple perpetrator doses
    function dose_callback(integrator)
        integrator.u[4] += params.perpetrator_dose_mg
    end

    callbacks = CallbackSet()
    if params.perpetrator_n_doses > 1
        dose_times = [params.perpetrator_interval_h * i for i in 1:(params.perpetrator_n_doses-1)]
        dose_times = filter(t -> t < t_max_h, dose_times)
        if !isempty(dose_times)
            cb = PresetTimeCallback(dose_times, dose_callback)
            callbacks = CallbackSet(cb)
        end
    end

    prob_ddi = ODEProblem(ddi_pbpk_ode!, u0_ddi, (0.0, t_max_h), ddi_params)
    sol_ddi = solve(prob_ddi, Tsit5(), callback=callbacks, saveat=time_points,
                    reltol=1e-8, abstol=1e-10)

    # Extract results
    Cp_ddi = [sol_ddi[i][2] for i in 1:length(sol_ddi)]
    Cp_perp = [sol_ddi[i][5] for i in 1:length(sol_ddi)]
    Cu_perp = Cp_perp .* params.perpetrator_fu
    E_liver = [sol_ddi[i][6] for i in 1:length(sol_ddi)]
    E_gut = [sol_ddi[i][7] for i in 1:length(sol_ddi)]

    # ================================
    # CALCULATE PK METRICS
    # ================================

    # AUC (trapezoidal)
    auc_alone = sum((Cp_alone[i] + Cp_alone[i+1]) / 2 * dt_h for i in 1:(n_points-1))
    auc_ddi = sum((Cp_ddi[i] + Cp_ddi[i+1]) / 2 * dt_h for i in 1:(n_points-1))

    # Cmax and Tmax
    cmax_alone, tmax_idx_alone = findmax(Cp_alone)
    tmax_alone = time_points[tmax_idx_alone]

    cmax_ddi, tmax_idx_ddi = findmax(Cp_ddi)
    tmax_ddi = time_points[tmax_idx_ddi]

    # Half-life (terminal)
    half_life_alone = calculate_half_life(time_points, Cp_alone)
    half_life_ddi = calculate_half_life(time_points, Cp_ddi)

    # DDI ratios
    auc_ratio = auc_ddi / max(auc_alone, 1e-10)
    cmax_ratio = cmax_ddi / max(cmax_alone, 1e-10)
    cl_ratio = auc_alone / max(auc_ddi, 1e-10)  # CL ratio is inverse of AUC ratio

    # Estimate Fg and Fh contributions
    # This is approximate based on enzyme activity at absorption time
    avg_E_gut_absorption = mean(E_gut[1:min(10, n_points)])  # First hour
    avg_E_liver_absorption = mean(E_liver[1:min(10, n_points)])

    fg_ratio = if params.inhibit_gut
        Egut_baseline = 1.0 - params.victim_Fg
        Egut_inhibited = Egut_baseline * avg_E_gut_absorption
        (1.0 - Egut_inhibited) / params.victim_Fg
    else
        1.0
    end

    fh_ratio = if params.mechanism == :induction
        1.0 / avg_E_liver_absorption  # Induction reduces F
    else
        # Inhibition increases F
        Eh_baseline = 1.0 - params.victim_Fh
        Eh_inhibited = Eh_baseline * avg_E_liver_absorption
        (1.0 - Eh_inhibited) / params.victim_Fh
    end

    clh_ratio = if params.mechanism == :induction
        avg_E_liver_absorption  # Induction increases CL
    else
        1.0 / auc_ratio / fg_ratio / fh_ratio  # Approximate
    end

    # Mechanism description
    mechanism_details = if params.mechanism == :reversible
        "Reversible competitive inhibition (Ki = $(params.Ki_uM) μM)"
    elseif params.mechanism == :mbi
        "Mechanism-based inactivation (kinact = $(params.kinact_h) h⁻¹, KI = $(params.KI_uM) μM)"
    elseif params.mechanism == :induction
        "Enzyme induction (Emax = $(params.Emax), EC50 = $(params.EC50_uM) μM)"
    else
        "Mixed mechanism"
    end

    return DDIPBPKResult(
        time_points,
        Cp_alone, Cp_ddi,
        Cp_perp, Cu_perp,
        E_liver, E_gut,
        auc_alone, cmax_alone, tmax_alone, half_life_alone,
        auc_ddi, cmax_ddi, tmax_ddi, half_life_ddi,
        auc_ratio, cmax_ratio, cl_ratio,
        fg_ratio, fh_ratio, clh_ratio,
        params, mechanism_details
    )
end

"""
Calculate terminal half-life from concentration-time data.
"""
function calculate_half_life(time::Vector{Float64}, conc::Vector{Float64})::Float64
    # Ensure same length
    n = min(length(conc), length(time))
    if n < 4
        return NaN
    end

    # Use last 25% of data
    start_idx = max(1, Int(floor(0.75 * n)))

    # Filter positive concentrations
    valid_idx = [i for i in start_idx:n if i <= length(conc) && i <= length(time) && conc[i] > 1e-10]

    if length(valid_idx) < 3
        return NaN
    end

    # Linear regression on log(C) vs t
    t_valid = [time[i] for i in valid_idx]
    logC_valid = [log(conc[i]) for i in valid_idx]

    n_pts = length(t_valid)
    sum_t = sum(t_valid)
    sum_logC = sum(logC_valid)
    sum_t_logC = sum(t_valid .* logC_valid)
    sum_t2 = sum(t_valid .^ 2)

    denom = n_pts * sum_t2 - sum_t^2
    if abs(denom) < 1e-10
        return NaN
    end

    slope = (n_pts * sum_t_logC - sum_t * sum_logC) / denom

    if slope >= 0
        return NaN  # Not elimination phase
    end

    return -log(2) / slope
end

"""
Convenience function: simulate and compare DDI scenarios.
"""
function simulate_ddi_comparison(
    victim::Symbol,
    victim_dose_mg::Float64,
    perpetrator::Symbol,
    perpetrator_dose_mg::Float64;
    perpetrator_doses::Vector{Int} = [1, 3, 7],
    t_max_h::Float64 = 72.0,
    victim_MW::Float64 = 300.0,
    perpetrator_MW::Float64 = 400.0
)::Vector{DDIPBPKResult}

    results = DDIPBPKResult[]

    for n_doses in perpetrator_doses
        params = build_ddi_params(
            victim, victim_dose_mg,
            perpetrator, perpetrator_dose_mg,
            perpetrator_n_doses = n_doses
        )

        result = simulate_ddi_pbpk(
            params,
            t_max_h = t_max_h,
            victim_MW = victim_MW,
            perpetrator_MW = perpetrator_MW
        )

        push!(results, result)
    end

    return results
end

# =============================================================================
# TIME-COURSE PREDICTIONS
# =============================================================================

"""
Predict DDI time-course without full PBPK simulation.
Useful for quick estimations.
"""
function predict_ddi_time_course(
    perpetrator_conc_time::Vector{Float64},  # Time in hours
    perpetrator_conc::Vector{Float64},       # Concentration in μM
    mechanism::Symbol,
    Ki_uM::Float64,
    kinact_h::Float64 = 0.0,
    KI_uM::Float64 = 0.0,
    kdeg_h::Float64 = 0.02,
    fm::Float64 = 0.9
)::NamedTuple

    n_points = length(perpetrator_conc_time)

    auc_ratios = zeros(n_points)
    enzyme_activity = ones(n_points)

    # Track enzyme activity over time
    E = 1.0

    for i in 1:n_points
        Cu = perpetrator_conc[i]
        dt = i > 1 ? perpetrator_conc_time[i] - perpetrator_conc_time[i-1] : 0.0

        if mechanism == :mbi && dt > 0
            # Update enzyme activity
            lambda = kinact_h * Cu / (KI_uM + Cu + 1e-6)
            dE = kdeg_h * (1.0 - E) - lambda * E
            E += dE * dt
            E = clamp(E, 0.0, 1.0)
        end

        enzyme_activity[i] = E

        # Calculate AUC ratio at this time
        if mechanism == :reversible
            R = 1.0 + Cu / Ki_uM
            auc_ratios[i] = 1.0 / (fm / R + (1 - fm))
        elseif mechanism == :mbi
            # MBI uses enzyme activity
            auc_ratios[i] = 1.0 / (fm * E + (1 - fm))
        end
    end

    return (
        time_h = perpetrator_conc_time,
        perpetrator_conc_uM = perpetrator_conc,
        auc_ratio = auc_ratios,
        enzyme_activity = enzyme_activity,
        max_auc_ratio = maximum(auc_ratios),
        time_to_max_effect_h = perpetrator_conc_time[argmax(auc_ratios)]
    )
end

"""
Simulate MBI time-course showing enzyme recovery.
"""
function simulate_mbi_time_course(
    perpetrator_cmax_uM::Float64,
    perpetrator_half_life_h::Float64,
    kinact_h::Float64,
    KI_uM::Float64,
    kdeg_h::Float64 = 0.02;
    t_max_h::Float64 = 168.0,  # 1 week
    dose_interval_h::Float64 = 24.0,
    n_doses::Int = 7
)::NamedTuple

    dt = 0.5
    time_points = collect(0.0:dt:t_max_h)
    n_points = length(time_points)

    perp_conc = zeros(n_points)
    enzyme = ones(n_points)
    auc_ratio = ones(n_points)

    ke = log(2) / perpetrator_half_life_h
    E = 1.0

    for i in 1:n_points
        t = time_points[i]

        # Perpetrator concentration (multiple doses)
        C = 0.0
        for dose in 0:(n_doses-1)
            t_since_dose = t - dose * dose_interval_h
            if t_since_dose >= 0
                C += perpetrator_cmax_uM * exp(-ke * t_since_dose)
            end
        end
        perp_conc[i] = C

        # Enzyme dynamics
        lambda = kinact_h * C / (KI_uM + C + 1e-6)
        dE = kdeg_h * (1.0 - E) - lambda * E
        E += dE * dt
        E = clamp(E, 0.0, 1.0)

        enzyme[i] = E
        auc_ratio[i] = 1.0 / E  # Simplified
    end

    return (
        time_h = time_points,
        perpetrator_conc_uM = perp_conc,
        enzyme_activity = enzyme,
        auc_ratio = auc_ratio,
        min_enzyme = minimum(enzyme),
        max_auc_ratio = maximum(auc_ratio),
        time_to_recovery_h = let
            # Find when enzyme recovers to 90%
            recovery_idx = findfirst(i -> enzyme[i] > 0.9 && time_points[i] > n_doses * dose_interval_h, 1:n_points)
            recovery_idx !== nothing ? time_points[recovery_idx] : t_max_h
        end
    )
end

"""
Simulate enzyme induction onset.
"""
function simulate_induction_onset(
    inducer_cmax_uM::Float64,
    inducer_half_life_h::Float64,
    Emax::Float64,
    EC50_uM::Float64,
    kdeg_h::Float64 = 0.02;
    t_max_h::Float64 = 336.0,  # 2 weeks
    dose_interval_h::Float64 = 24.0,
    n_doses::Int = 14
)::NamedTuple

    dt = 1.0
    time_points = collect(0.0:dt:t_max_h)
    n_points = length(time_points)

    inducer_conc = zeros(n_points)
    enzyme = ones(n_points)
    auc_ratio = ones(n_points)

    ke = log(2) / inducer_half_life_h
    E = 1.0

    for i in 1:n_points
        t = time_points[i]

        # Inducer concentration (multiple doses)
        C = 0.0
        for dose in 0:(n_doses-1)
            t_since_dose = t - dose * dose_interval_h
            if t_since_dose >= 0
                C += inducer_cmax_uM * exp(-ke * t_since_dose)
            end
        end
        inducer_conc[i] = C

        # Target enzyme level
        E_target = 1.0 + Emax * C / (EC50_uM + C + 1e-6)

        # Enzyme dynamics (first-order approach to target)
        dE = kdeg_h * (E_target - E)
        E += dE * dt

        enzyme[i] = E
        auc_ratio[i] = 1.0 / E  # AUC decreases with induction
    end

    return (
        time_h = time_points,
        inducer_conc_uM = inducer_conc,
        enzyme_activity = enzyme,
        auc_ratio = auc_ratio,
        max_enzyme = maximum(enzyme),
        min_auc_ratio = minimum(auc_ratio),
        time_to_50pct_effect_h = let
            # Find when effect reaches 50% of max
            max_E = maximum(enzyme)
            target = 1.0 + 0.5 * (max_E - 1.0)
            idx = findfirst(E -> E >= target, enzyme)
            idx !== nothing ? time_points[idx] : t_max_h
        end,
        time_to_steady_state_h = let
            # ~4-5 half-lives of enzyme
            4.0 * log(2) / kdeg_h
        end
    )
end

# =============================================================================
# EXPORTS
# =============================================================================

export DDIPBPKParams, DDIPBPKResult
export simulate_ddi_pbpk, simulate_ddi_comparison, build_ddi_params
export predict_ddi_time_course, calculate_dynamic_clint
export simulate_mbi_time_course, simulate_induction_onset
export GutWallDDI, separate_fg_fh, calculate_fg_inhibition
export get_victim_preset, get_perpetrator_preset

end # module
