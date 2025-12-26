# ===========================================================================
# MECHANISTIC DDI MODULE - ADVANCED DRUG-DRUG INTERACTION MODELING
# ===========================================================================
# Quantitative IVIVE-DDI with mechanistic enzyme kinetics, competitive
# multi-inhibitor saturation, substrate depletion, and Monte Carlo UQ.
#
# This module extends the existing DDI functionality with:
# - In vitro to in vivo extrapolation (IVIVE) for DDI prediction
# - Competitive enzyme saturation for multiple inhibitors
# - Time-dependent inhibition (TDI) with detailed kinetics
# - Substrate depletion dynamics
# - Monte Carlo uncertainty propagation
# - Combined CYP-transporter DDI modeling
#
# References:
# - FDA Guidance: Clinical DDI Studies (2020)
# - EMA Guideline: Investigation of DDI (2012)
# - Rowland-Yeo et al. (2011) IVIVE scaling factors
# - Obach et al. (2006) Mechanism-based inhibition
# - Fahmi et al. (2009) Induction prediction
#
# Author: Dr. Sounio Agourakis
# Date: December 2025
# Version: 1.0.0
# ===========================================================================

module MechanisticDDI

using Statistics
using Random
using Distributions
using LinearAlgebra

# ===========================================================================
# Exports
# ===========================================================================

export IVIVEParams, EnzymeKinetics, InhibitorParams, InducerParams
export SubstrateDepletion, MultiInhibitorState, DDIUncertainty
export DDIPredictionResult, TransporterDDI, CombinedDDI

# IVIVE functions
export calculate_fu_mic, calculate_fu_inc, scale_clint_ivive
export predict_fm_ivive, calculate_hepatic_clint, calculate_gut_clint
export mppgl_scaling, hpgl_scaling, isef_adjustment

# Inhibition functions
export competitive_inhibition, noncompetitive_inhibition, uncompetitive_inhibition
export mixed_inhibition, mechanism_based_inhibition, calculate_kinact_ki
export multi_inhibitor_saturation, calculate_fmcyp_inhibited

# Induction functions
export calculate_induction_magnitude, pxr_car_activation
export net_effect_inhibition_induction, delayed_induction_onset

# Substrate depletion
export substrate_depletion_factor, time_course_depletion

# Transporter DDI
export calculate_transporter_ddi, oatp_inhibition, pgp_inhibition, bcrp_inhibition
export combined_cyp_transporter_ddi, hepatic_uptake_inhibition

# Monte Carlo
export monte_carlo_ddi_prediction, ddi_sensitivity_analysis
export calculate_ddi_confidence_interval, classify_ddi_severity

# Integration
export predict_ddi_mechanistic, simulate_ddi_time_course
export create_ddi_report, validate_against_clinical

# Constants
export PHYSIOLOGICAL_PARAMS, CYP_ABUNDANCE, TRANSPORTER_ABUNDANCE
export TISSUE_BINDING_DEFAULTS, ENZYME_KINETIC_DEFAULTS

# ===========================================================================
# Constants - Physiological Scaling Factors
# ===========================================================================

"""
Physiological scaling parameters for IVIVE.
References: Barter et al. (2007), Howgate et al. (2006)
"""
const PHYSIOLOGICAL_PARAMS = Dict{String, Any}(
    # Liver parameters
    "liver_weight" => 1800.0,           # g (average adult)
    "liver_blood_flow" => 1450.0,       # mL/min (Qh)
    "hepatocellularity" => 120e6,       # hepatocytes per g liver (HPGL)
    "mppgl" => 40.0,                    # mg microsomal protein per g liver
    "mspppgm" => 12.5,                  # mg S9 protein per g microsomal protein

    # Intestinal parameters
    "gut_weight" => 1650.0,             # g (small intestine)
    "gut_blood_flow" => 300.0,          # mL/min (portal vein contribution)
    "enterocyte_number" => 3.0e11,      # total enterocytes
    "mppge" => 3.0,                     # mg microsomal protein per g enterocyte
    "gut_volume" => 250.0,              # mL (for Ig calculation)

    # Blood parameters
    "plasma_volume" => 3000.0,          # mL
    "blood_volume" => 5000.0,           # mL
    "albumin_conc" => 40.0,             # g/L
    "agp_conc" => 0.8,                  # g/L (alpha-1-acid glycoprotein)

    # Enzyme turnover (half-lives in hours)
    "cyp3a4_halflife" => 36.0,
    "cyp2d6_halflife" => 51.0,
    "cyp2c9_halflife" => 104.0,
    "cyp2c19_halflife" => 26.0,
    "cyp1a2_halflife" => 39.0,
    "cyp2c8_halflife" => 23.0,
    "cyp2b6_halflife" => 32.0,

    # Transporter turnover
    "pgp_halflife" => 17.0,
    "bcrp_halflife" => 36.0,
    "oatp1b1_halflife" => 45.0,
    "oatp1b3_halflife" => 40.0,
)

"""
CYP enzyme abundance in liver and intestine (pmol/mg protein).
Reference: Achour et al. (2014), Paine et al. (2006)
"""
const CYP_ABUNDANCE = Dict{Symbol, Dict{String, Float64}}(
    :CYP3A4 => Dict("liver" => 137.0, "intestine" => 70.0, "cv" => 0.40),
    :CYP3A5 => Dict("liver" => 12.0, "intestine" => 5.0, "cv" => 0.80),  # Highly variable
    :CYP2D6 => Dict("liver" => 10.0, "intestine" => 0.5, "cv" => 0.50),
    :CYP2C9 => Dict("liver" => 96.0, "intestine" => 5.0, "cv" => 0.35),
    :CYP2C19 => Dict("liver" => 14.0, "intestine" => 1.0, "cv" => 0.55),
    :CYP2C8 => Dict("liver" => 24.0, "intestine" => 0.5, "cv" => 0.45),
    :CYP1A2 => Dict("liver" => 52.0, "intestine" => 0.1, "cv" => 0.45),
    :CYP2B6 => Dict("liver" => 17.0, "intestine" => 0.5, "cv" => 0.75),
    :CYP2E1 => Dict("liver" => 49.0, "intestine" => 1.0, "cv" => 0.35),
)

"""
Transporter abundance (pmol/mg protein).
Reference: Prasad et al. (2016), Drozdzik et al. (2014)
"""
const TRANSPORTER_ABUNDANCE = Dict{Symbol, Dict{String, Float64}}(
    :OATP1B1 => Dict("liver" => 2.94, "kidney" => 0.0, "intestine" => 0.0, "cv" => 0.60),
    :OATP1B3 => Dict("liver" => 1.35, "kidney" => 0.0, "intestine" => 0.0, "cv" => 0.55),
    :OATP2B1 => Dict("liver" => 0.89, "kidney" => 0.1, "intestine" => 1.2, "cv" => 0.50),
    :OCT1 => Dict("liver" => 1.50, "kidney" => 0.5, "intestine" => 0.3, "cv" => 0.55),
    :OAT2 => Dict("liver" => 0.75, "kidney" => 1.5, "intestine" => 0.1, "cv" => 0.50),
    :P_gp => Dict("liver" => 2.50, "kidney" => 4.5, "intestine" => 3.8, "cv" => 0.45),
    :BCRP => Dict("liver" => 1.80, "kidney" => 1.2, "intestine" => 2.1, "cv" => 0.50),
    :MRP2 => Dict("liver" => 3.20, "kidney" => 0.8, "intestine" => 0.5, "cv" => 0.45),
    :MATE1 => Dict("liver" => 0.50, "kidney" => 2.8, "intestine" => 0.1, "cv" => 0.55),
    :MATE2K => Dict("liver" => 0.10, "kidney" => 1.5, "intestine" => 0.0, "cv" => 0.60),
)

"""
Default tissue binding parameters for in vitro systems.
"""
const TISSUE_BINDING_DEFAULTS = Dict{String, Float64}(
    "fu_mic_default" => 1.0,      # Unbound fraction in microsomes
    "fu_hep_default" => 1.0,      # Unbound fraction in hepatocytes
    "fu_plasma_default" => 0.5,   # Unbound fraction in plasma
    "fu_blood_default" => 0.5,    # Unbound fraction in blood
    "microsomal_protein" => 0.5,  # mg/mL in incubation
    "hepatocyte_conc" => 1.0e6,   # cells/mL in incubation
)

"""
Default enzyme kinetic parameters.
"""
const ENZYME_KINETIC_DEFAULTS = Dict{String, Float64}(
    "default_km" => 10.0,         # μM
    "default_vmax" => 100.0,      # pmol/min/pmol CYP
    "default_ki" => 1.0,          # μM (competitive inhibition)
    "default_kinact" => 0.05,     # min⁻¹ (MBI)
    "default_ki_mbi" => 5.0,      # μM (MBI)
    "default_emax" => 10.0,       # Fold induction
    "default_ec50" => 1.0,        # μM (induction)
)

# ===========================================================================
# Data Structures
# ===========================================================================

"""
IVIVE scaling parameters for a drug.
"""
struct IVIVEParams
    fu_plasma::Float64          # Unbound fraction in plasma
    fu_blood::Float64           # Unbound fraction in blood
    fu_mic::Float64             # Unbound fraction in microsomes
    fu_hep::Float64             # Unbound fraction in hepatocytes
    rb::Float64                 # Blood:plasma ratio
    logp::Float64               # Lipophilicity
    mw::Float64                 # Molecular weight (Da)
    pka::Float64                # pKa (for ionization)
    is_acid::Bool               # Acidic or basic compound
end

"""
Enzyme kinetic parameters for a CYP.
"""
struct EnzymeKinetics
    enzyme::Symbol              # :CYP3A4, :CYP2D6, etc.
    km::Float64                 # Michaelis constant (μM)
    vmax::Float64               # Maximum velocity (pmol/min/pmol CYP)
    clint::Float64              # Intrinsic clearance = Vmax/Km
    fm::Float64                 # Fraction metabolized by this enzyme
    isef::Float64               # Inter-system extrapolation factor
end

"""
Inhibitor kinetic parameters.
"""
struct InhibitorParams
    name::String
    inhibitor_type::Symbol      # :competitive, :noncompetitive, :uncompetitive, :mixed, :mbi
    ki::Float64                 # Inhibition constant (μM)
    kinact::Float64             # Inactivation rate for MBI (min⁻¹)
    ki_mbi::Float64             # Ki for MBI (μM)
    fu_plasma::Float64          # Unbound fraction
    cmax_total::Float64         # Total Cmax (μM)
    cmax_unbound::Float64       # Unbound Cmax (μM)
    ig::Float64                 # Gut concentration (μM) = Dose/250mL
    target_enzymes::Vector{Symbol}  # Which CYPs are inhibited
    target_transporters::Vector{Symbol}  # Which transporters are inhibited
end

"""
Inducer kinetic parameters.
"""
struct InducerParams
    name::String
    emax::Dict{Symbol, Float64}     # Maximum fold induction per enzyme
    ec50::Dict{Symbol, Float64}     # EC50 per enzyme (μM)
    pathway::Symbol                 # :PXR, :CAR, :AhR
    fu_plasma::Float64
    cmax_unbound::Float64
    target_enzymes::Vector{Symbol}
    onset_delay::Float64            # Days to onset
end

"""
Substrate depletion state.
"""
mutable struct SubstrateDepletion
    initial_conc::Float64
    current_conc::Float64
    depletion_rate::Float64         # μM/h
    time_elapsed::Float64           # hours
    fractional_remaining::Float64
end

"""
Multi-inhibitor saturation state.
"""
struct MultiInhibitorState
    inhibitors::Vector{InhibitorParams}
    enzyme::Symbol
    total_inhibition::Float64       # 0-1
    dominant_mechanism::Symbol
    saturation_fraction::Float64
end

"""
DDI prediction uncertainty.
"""
struct DDIUncertainty
    mean_ratio::Float64
    sd_ratio::Float64
    ci_90_lower::Float64
    ci_90_upper::Float64
    ci_95_lower::Float64
    ci_95_upper::Float64
    p_exceeds_2fold::Float64
    p_exceeds_5fold::Float64
    n_simulations::Int
end

"""
Transporter DDI parameters.
"""
struct TransporterDDI
    transporter::Symbol
    inhibition_type::Symbol         # :uptake or :efflux
    ki::Float64
    ic50::Float64
    inhibitor_conc::Float64
    fold_change::Float64
    tissue::Symbol                  # :liver, :intestine, :kidney
end

"""
Combined CYP + transporter DDI result.
"""
struct CombinedDDI
    cyp_contribution::Float64
    transporter_contribution::Float64
    total_auc_ratio::Float64
    total_cmax_ratio::Float64
    fg_change::Float64
    fh_change::Float64
    cl_oral_ratio::Float64
end

"""
Complete DDI prediction result.
"""
struct DDIPredictionResult
    perpetrator::String
    victim::String
    auc_ratio::Float64
    cmax_ratio::Float64
    cl_ratio::Float64
    mechanism::Symbol
    severity::Symbol                # :none, :weak, :moderate, :strong
    fda_classification::String
    uncertainty::DDIUncertainty
    recommendations::Vector{String}
    cyp_contributions::Dict{Symbol, Float64}
    transporter_contributions::Dict{Symbol, Float64}
end

# ===========================================================================
# IVIVE Functions
# ===========================================================================

"""
    calculate_fu_mic(fu_plasma, logp; method=:austin) -> Float64

Calculate fraction unbound in microsomes from plasma binding.
References: Austin et al. (2002), Hallifax & Houston (2006)
"""
function calculate_fu_mic(fu_plasma::Float64, logp::Float64; method::Symbol=:austin)::Float64
    if method == :austin
        # Austin et al. equation
        log_fu_mic_fu_inc = 0.53 * logp - 0.49
        fu_mic_fu_inc = 10^log_fu_mic_fu_inc
        fu_mic = 1.0 / (1.0 + fu_mic_fu_inc * (1.0/fu_plasma - 1.0))
    elseif method == :hallifax
        # Hallifax & Houston equation
        fu_mic = 1.0 / (1.0 + 0.125 * TISSUE_BINDING_DEFAULTS["microsomal_protein"] * 10^(0.072*logp^2 + 0.067*logp - 1.126))
    else
        fu_mic = fu_plasma  # Assume no additional binding
    end

    return clamp(fu_mic, 0.001, 1.0)
end

"""
    calculate_fu_inc(fu_plasma, logp, incubation_type) -> Float64

Calculate fraction unbound in hepatocyte incubations.
"""
function calculate_fu_inc(fu_plasma::Float64, logp::Float64; incubation_type::Symbol=:suspension)::Float64
    if incubation_type == :suspension
        # Lower binding than monolayers
        fu_inc = calculate_fu_mic(fu_plasma, logp; method=:austin) * 1.2
    elseif incubation_type == :sandwich
        # Sandwich cultures have more binding
        fu_inc = calculate_fu_mic(fu_plasma, logp; method=:austin) * 0.8
    else
        fu_inc = calculate_fu_mic(fu_plasma, logp; method=:austin)
    end

    return clamp(fu_inc, 0.001, 1.0)
end

"""
    scale_clint_ivive(clint_invitro, ivive_params; system=:microsomes) -> Float64

Scale in vitro CLint to in vivo using IVIVE.
Returns CLint in mL/min/kg body weight.
"""
function scale_clint_ivive(
    clint_invitro::Float64,       # μL/min/mg protein or μL/min/10^6 cells
    ivive_params::IVIVEParams;
    system::Symbol = :microsomes,
    body_weight::Float64 = 70.0
)::Float64

    pp = PHYSIOLOGICAL_PARAMS

    if system == :microsomes
        # Scale: CLint (μL/min/mg) → CLint (mL/min/kg)
        # CLint,u,invivo = CLint,u,invitro × MPPGL × liver_weight / BW
        mppgl = pp["mppgl"]  # mg protein / g liver
        liver_weight = pp["liver_weight"]  # g

        # Correct for microsomal binding
        clint_u_invitro = clint_invitro / ivive_params.fu_mic

        # Scale up
        clint_invivo = clint_u_invitro * mppgl * liver_weight / body_weight / 1000  # mL/min/kg

    elseif system == :hepatocytes
        # Scale: CLint (μL/min/10^6 cells) → CLint (mL/min/kg)
        hpgl = pp["hepatocellularity"]  # cells/g liver
        liver_weight = pp["liver_weight"]

        clint_u_invitro = clint_invitro / ivive_params.fu_hep
        clint_invivo = clint_u_invitro * hpgl / 1e6 * liver_weight / body_weight / 1000

    elseif system == :recombinant
        # Scale from recombinant CYP to liver
        # Need RAF (relative activity factor) or ISEF
        clint_invivo = clint_invitro * pp["mppgl"] * pp["liver_weight"] / body_weight / 1000

    else
        clint_invivo = clint_invitro
    end

    return clint_invivo
end

"""
    mppgl_scaling(age; sex=:average) -> Float64

Age-dependent MPPGL (mg protein per g liver).
Reference: Barter et al. (2008)
"""
function mppgl_scaling(age::Float64; sex::Symbol=:average)::Float64
    if age < 1
        # Neonates: 26 mg/g
        return 26.0
    elseif age < 5
        # Infants/toddlers: linear increase
        return 26.0 + (40.0 - 26.0) * (age - 1) / 4
    elseif age < 18
        # Children/adolescents: gradual increase to adult
        return 35.0 + (40.0 - 35.0) * (age - 5) / 13
    elseif age < 65
        # Adults: stable
        return 40.0
    else
        # Elderly: slight decline
        return max(32.0, 40.0 - 0.1 * (age - 65))
    end
end

"""
    hpgl_scaling(age) -> Float64

Age-dependent hepatocellularity (10^6 cells per g liver).
"""
function hpgl_scaling(age::Float64)::Float64
    if age < 1
        return 100e6  # Lower in neonates
    elseif age < 18
        return 100e6 + (120e6 - 100e6) * age / 18
    elseif age < 65
        return 120e6
    else
        return max(100e6, 120e6 - 0.5e6 * (age - 65))
    end
end

"""
    isef_adjustment(enzyme, substrate_class) -> Float64

Inter-system extrapolation factor (ISEF) for recombinant CYP scaling.
Reference: Chen et al. (2011)
"""
function isef_adjustment(enzyme::Symbol; substrate_class::Symbol=:general)::Float64
    # ISEF = (Activity_HLM / Activity_rCYP) × (Abundance_HLM / Abundance_rCYP)
    # Typical values:
    isef_table = Dict(
        :CYP3A4 => Dict(:general => 1.0, :high_km => 0.7, :low_km => 1.3),
        :CYP2D6 => Dict(:general => 0.4, :high_km => 0.3, :low_km => 0.5),
        :CYP2C9 => Dict(:general => 0.6, :high_km => 0.5, :low_km => 0.7),
        :CYP2C19 => Dict(:general => 0.5, :high_km => 0.4, :low_km => 0.6),
        :CYP1A2 => Dict(:general => 0.8, :high_km => 0.6, :low_km => 1.0),
        :CYP2C8 => Dict(:general => 0.7, :high_km => 0.5, :low_km => 0.9),
        :CYP2B6 => Dict(:general => 0.3, :high_km => 0.2, :low_km => 0.4),
    )

    if haskey(isef_table, enzyme) && haskey(isef_table[enzyme], substrate_class)
        return isef_table[enzyme][substrate_class]
    else
        return 1.0  # Default: no adjustment
    end
end

"""
    calculate_hepatic_clint(enzyme_kinetics, ivive_params) -> Float64

Calculate hepatic intrinsic clearance from enzyme kinetics.
"""
function calculate_hepatic_clint(
    kinetics::Vector{EnzymeKinetics},
    ivive_params::IVIVEParams
)::Float64

    total_clint = 0.0

    for ek in kinetics
        # CLint = Vmax / Km
        clint_enzyme = ek.vmax / ek.km  # μL/min/pmol CYP

        # Scale by enzyme abundance
        abundance = get(CYP_ABUNDANCE, ek.enzyme, Dict("liver" => 50.0))["liver"]

        # Apply ISEF
        clint_enzyme *= ek.isef

        # Scale to liver: pmol CYP/mg × mg/g × g liver
        mppgl = PHYSIOLOGICAL_PARAMS["mppgl"]
        liver_weight = PHYSIOLOGICAL_PARAMS["liver_weight"]

        clint_liver = clint_enzyme * abundance * mppgl * liver_weight / 1e6  # mL/min

        total_clint += clint_liver
    end

    return total_clint
end

"""
    calculate_gut_clint(enzyme_kinetics, qgut) -> Float64

Calculate intestinal intrinsic clearance (for Fg calculation).
"""
function calculate_gut_clint(
    kinetics::Vector{EnzymeKinetics};
    qgut::Float64 = 300.0  # mL/min
)::Float64

    total_clint = 0.0

    for ek in kinetics
        abundance = get(CYP_ABUNDANCE, ek.enzyme, Dict("intestine" => 0.0))["intestine"]
        if abundance > 0
            clint_enzyme = ek.vmax / ek.km * abundance * ek.isef
            mppge = PHYSIOLOGICAL_PARAMS["mppge"]
            gut_weight = PHYSIOLOGICAL_PARAMS["gut_weight"]
            total_clint += clint_enzyme * mppge * gut_weight / 1e6
        end
    end

    return total_clint
end

"""
    predict_fm_ivive(substrate_clint_by_enzyme, total_clint) -> Dict{Symbol, Float64}

Predict fraction metabolized by each enzyme using IVIVE-scaled CLint.
"""
function predict_fm_ivive(
    clint_by_enzyme::Dict{Symbol, Float64}
)::Dict{Symbol, Float64}

    total = sum(values(clint_by_enzyme))
    if total == 0
        return Dict{Symbol, Float64}()
    end

    return Dict(k => v / total for (k, v) in clint_by_enzyme)
end

# ===========================================================================
# Inhibition Kinetics
# ===========================================================================

"""
    competitive_inhibition(i_conc, ki) -> Float64

Calculate fold-change in CLint from competitive inhibition.
Returns the inhibition factor (1 + [I]/Ki)
"""
function competitive_inhibition(i_conc::Float64, ki::Float64)::Float64
    return 1.0 + i_conc / ki
end

"""
    noncompetitive_inhibition(i_conc, ki) -> Float64

Calculate fold-change from noncompetitive inhibition.
Affects Vmax, not Km.
"""
function noncompetitive_inhibition(i_conc::Float64, ki::Float64)::Float64
    return 1.0 + i_conc / ki
end

"""
    uncompetitive_inhibition(i_conc, ki, s_conc, km) -> Float64

Calculate inhibition factor for uncompetitive inhibition.
Binds only to ES complex.
"""
function uncompetitive_inhibition(i_conc::Float64, ki::Float64, s_conc::Float64, km::Float64)::Float64
    alpha = s_conc / (km + s_conc)
    return 1.0 + alpha * i_conc / ki
end

"""
    mixed_inhibition(i_conc, ki, alpha) -> Float64

Calculate inhibition for mixed-type inhibition.
alpha > 1: preferentially binds free enzyme (competitive-like)
alpha < 1: preferentially binds ES complex (uncompetitive-like)
alpha = 1: pure noncompetitive
"""
function mixed_inhibition(i_conc::Float64, ki::Float64; alpha::Float64=1.0)::Float64
    # For Km: increases by factor (1 + [I]/αKi) / (1 + [I]/Ki)
    # For Vmax: decreases by factor 1 / (1 + [I]/αKi)
    # Net effect on CLint (Vmax/Km):
    return (1.0 + i_conc / (alpha * ki)) / (1.0 + i_conc / ki)
end

"""
    mechanism_based_inhibition(kinact, ki_mbi, i_conc, kdeg; time=Inf) -> Float64

Calculate enzyme inactivation from mechanism-based inhibition (MBI).
Returns the fraction of enzyme remaining active.

At steady state: Eactive/Etotal = kdeg / (kdeg + kinact × [I]/(KI + [I]))
"""
function mechanism_based_inhibition(
    kinact::Float64,      # min⁻¹ (inactivation rate constant)
    ki_mbi::Float64,      # μM (concentration for half-maximal inactivation)
    i_conc::Float64,      # μM (inhibitor concentration)
    kdeg::Float64;        # h⁻¹ (enzyme degradation rate)
    time::Float64 = Inf   # hours (time since inhibitor exposure)
)::Float64

    # Convert kdeg to min⁻¹
    kdeg_min = kdeg / 60.0

    # Inactivation rate
    kinact_app = kinact * i_conc / (ki_mbi + i_conc)

    if isinf(time)
        # Steady state
        fraction_active = kdeg_min / (kdeg_min + kinact_app)
    else
        # Time-dependent approach to steady state
        # E(t) = Ess + (E0 - Ess) × exp(-(kdeg + kinact_app) × t)
        t_min = time * 60.0
        kobs = kdeg_min + kinact_app
        e_ss = kdeg_min / kobs
        fraction_active = e_ss + (1.0 - e_ss) * exp(-kobs * t_min)
    end

    return clamp(fraction_active, 0.0, 1.0)
end

"""
    calculate_kinact_ki(ic50_values, preincubation_times) -> Tuple{Float64, Float64}

Estimate kinact and KI from IC50 shift data.
Reference: Obach et al. (1999)
"""
function calculate_kinact_ki(
    ic50_values::Vector{Float64},       # IC50 at different preincubation times
    preincubation_times::Vector{Float64} # Minutes
)::Tuple{Float64, Float64}

    # IC50(t) = IC50(0) × exp(-kinact × t × [E] / (KI + [E]))
    # Simplified: plot ln(IC50(t)/IC50(0)) vs t to get kinact_app

    if length(ic50_values) < 2 || length(ic50_values) != length(preincubation_times)
        return (0.0, ic50_values[1])
    end

    # Linear regression on log scale
    ln_ratio = log.(ic50_values ./ ic50_values[1])

    # Slope = -kinact_app
    slope = sum(preincubation_times .* ln_ratio) / sum(preincubation_times.^2)
    kinact_app = -slope

    # KI estimation requires varying [I] - simplified here
    ki_estimate = ic50_values[1] / 2.0  # Approximate

    return (kinact_app, ki_estimate)
end

"""
    multi_inhibitor_saturation(inhibitors, enzyme, substrate_km) -> MultiInhibitorState

Calculate combined inhibition from multiple inhibitors competing for same enzyme.
Uses competitive binding model with saturation.
"""
function multi_inhibitor_saturation(
    inhibitors::Vector{InhibitorParams},
    enzyme::Symbol;
    substrate_conc::Float64 = 0.1,  # μM
    substrate_km::Float64 = 10.0     # μM
)::MultiInhibitorState

    if isempty(inhibitors)
        return MultiInhibitorState(inhibitors, enzyme, 0.0, :none, 0.0)
    end

    # Filter for inhibitors targeting this enzyme
    relevant_inhibitors = filter(inh -> enzyme in inh.target_enzymes, inhibitors)

    if isempty(relevant_inhibitors)
        return MultiInhibitorState(inhibitors, enzyme, 0.0, :none, 0.0)
    end

    # For competitive inhibitors: combined effect
    # 1 + Σ([Ii]/Ki_i) for additive approximation
    # More accurate: solve competitive binding equilibrium

    total_inhibition_term = 0.0
    dominant_inh = relevant_inhibitors[1]
    max_contribution = 0.0

    for inh in relevant_inhibitors
        if inh.inhibitor_type == :competitive || inh.inhibitor_type == :mixed
            contribution = inh.cmax_unbound / inh.ki
            total_inhibition_term += contribution
            if contribution > max_contribution
                max_contribution = contribution
                dominant_inh = inh
            end
        elseif inh.inhibitor_type == :mbi
            # MBI adds separately (irreversible)
            mbi_effect = 1.0 / mechanism_based_inhibition(
                inh.kinact, inh.ki_mbi, inh.cmax_unbound,
                log(2) / get(PHYSIOLOGICAL_PARAMS, "$(lowercase(string(enzyme)))_halflife", 36.0)
            ) - 1.0
            total_inhibition_term += mbi_effect
        end
    end

    # Calculate saturation
    # At high total inhibition, enzyme becomes saturated
    saturation = total_inhibition_term / (1.0 + total_inhibition_term)

    # Total fold-reduction in enzyme activity
    fold_reduction = 1.0 + total_inhibition_term
    inhibition_fraction = 1.0 - 1.0 / fold_reduction

    return MultiInhibitorState(
        relevant_inhibitors,
        enzyme,
        inhibition_fraction,
        dominant_inh.inhibitor_type,
        saturation
    )
end

"""
    calculate_fmcyp_inhibited(fm, inhibition_factor) -> Float64

Calculate the fraction of clearance remaining after CYP inhibition.
"""
function calculate_fmcyp_inhibited(fm::Float64, inhibition_factor::Float64)::Float64
    # CLnew/CLoriginal = fm/(inhibition_factor) + (1-fm)
    return fm / inhibition_factor + (1.0 - fm)
end

# ===========================================================================
# Induction Kinetics
# ===========================================================================

"""
    calculate_induction_magnitude(emax, ec50, inducer_conc; basal=1.0) -> Float64

Calculate fold-induction using Emax model.
"""
function calculate_induction_magnitude(
    emax::Float64,          # Maximum fold induction
    ec50::Float64,          # Concentration for half-maximal induction (μM)
    inducer_conc::Float64;  # Unbound inducer concentration (μM)
    basal::Float64 = 1.0    # Basal enzyme level
)::Float64

    # Emax model: E = Ebasal × (1 + Emax × [I] / (EC50 + [I]))
    induction = basal * (1.0 + emax * inducer_conc / (ec50 + inducer_conc))

    return induction
end

"""
    pxr_car_activation(inducer_conc, ec50, pathway; gene=:CYP3A4) -> Float64

Calculate nuclear receptor activation for induction.
PXR activates: CYP3A4, CYP2B6, CYP2C8, CYP2C9, P-gp, OATP1B1
CAR activates: CYP2B6, CYP3A4, CYP2C9, UGT1A1
AhR activates: CYP1A1, CYP1A2, CYP1B1, UGT1A1
"""
function pxr_car_activation(
    inducer_conc::Float64,
    ec50::Float64;
    pathway::Symbol = :PXR,
    gene::Symbol = :CYP3A4
)::Float64

    # Emax values depend on gene and pathway
    emax_table = Dict(
        :PXR => Dict(:CYP3A4 => 20.0, :CYP2B6 => 10.0, :CYP2C9 => 3.0, :P_gp => 4.0),
        :CAR => Dict(:CYP2B6 => 40.0, :CYP3A4 => 5.0, :UGT1A1 => 3.0),
        :AhR => Dict(:CYP1A2 => 10.0, :CYP1A1 => 50.0, :CYP1B1 => 30.0),
    )

    emax = get(get(emax_table, pathway, Dict()), gene, 2.0)

    return calculate_induction_magnitude(emax, ec50, inducer_conc)
end

"""
    net_effect_inhibition_induction(inhibition_factor, induction_fold) -> Float64

Calculate net effect when drug is both inhibitor and inducer.
Common for drugs like ritonavir, efavirenz.
"""
function net_effect_inhibition_induction(
    inhibition_factor::Float64,  # From inhibition calculation (>1 means inhibition)
    induction_fold::Float64      # From induction calculation (>1 means induction)
)::Float64

    # Net CLint change = (1/inhibition_factor) × induction_fold
    # If inhibition dominates: ratio < 1 (increased AUC)
    # If induction dominates: ratio > 1 (decreased AUC)

    return induction_fold / inhibition_factor
end

"""
    delayed_induction_onset(emax, ec50, inducer_conc, kdeg, time) -> Float64

Calculate induction magnitude accounting for delayed onset.
"""
function delayed_induction_onset(
    emax::Float64,
    ec50::Float64,
    inducer_conc::Float64,
    kdeg::Float64,          # Enzyme degradation rate (h⁻¹)
    time::Float64           # Hours since start of induction
)::Float64

    # Steady-state induction
    induction_ss = calculate_induction_magnitude(emax, ec50, inducer_conc)

    # Time course: E(t) = Ess + (E0 - Ess) × exp(-kdeg × t)
    # Assuming E0 = 1.0 (baseline)
    current_induction = induction_ss + (1.0 - induction_ss) * exp(-kdeg * time)

    return current_induction
end

# ===========================================================================
# Substrate Depletion
# ===========================================================================

"""
    substrate_depletion_factor(initial_conc, clint, volume, time) -> Float64

Calculate substrate concentration after depletion.
"""
function substrate_depletion_factor(
    initial_conc::Float64,  # μM
    clint::Float64,         # mL/min
    volume::Float64,        # mL
    time::Float64           # min
)::Float64

    # First-order depletion: C(t) = C0 × exp(-CLint/V × t)
    k_elim = clint / volume
    return exp(-k_elim * time)
end

"""
    time_course_depletion(substrate, inhibitor_conc, km, vmax, time_points) -> SubstrateDepletion

Simulate substrate depletion over time with competitive inhibition.
"""
function time_course_depletion(
    initial_substrate::Float64,
    inhibitor_conc::Float64,
    ki::Float64,
    km::Float64,
    vmax::Float64;
    time_points::Vector{Float64} = collect(0.0:0.1:24.0),  # hours
    volume::Float64 = 1000.0  # mL
)::Vector{SubstrateDepletion}

    results = SubstrateDepletion[]
    current_conc = initial_substrate

    for (i, t) in enumerate(time_points)
        if i == 1
            push!(results, SubstrateDepletion(initial_substrate, current_conc, 0.0, t, 1.0))
            continue
        end

        dt = time_points[i] - time_points[i-1]

        # Competitive inhibition affects apparent Km
        km_app = km * (1.0 + inhibitor_conc / ki)

        # Michaelis-Menten rate
        rate = vmax * current_conc / (km_app + current_conc)  # μM/h

        # Update concentration
        current_conc = max(0.0, current_conc - rate * dt)

        push!(results, SubstrateDepletion(
            initial_substrate,
            current_conc,
            rate,
            t,
            current_conc / initial_substrate
        ))
    end

    return results
end

# ===========================================================================
# Transporter DDI
# ===========================================================================

"""
    calculate_transporter_ddi(transporter, inhibitor_params, substrate_affinity) -> TransporterDDI

Calculate DDI mediated by transporter inhibition.
"""
function calculate_transporter_ddi(
    transporter::Symbol,
    inhibitor::InhibitorParams,
    substrate_km::Float64;
    tissue::Symbol = :liver
)::TransporterDDI

    if !(transporter in inhibitor.target_transporters)
        return TransporterDDI(transporter, :none, Inf, Inf, 0.0, 1.0, tissue)
    end

    ki = inhibitor.ki  # Assume same Ki for transporter
    conc = tissue == :liver ? inhibitor.cmax_unbound : inhibitor.ig

    # Uptake vs efflux changes AUC in opposite directions
    transporter_type = transporter in [:OATP1B1, :OATP1B3, :OATP2B1, :OCT1, :OAT1, :OAT3] ? :uptake : :efflux

    if transporter_type == :uptake
        # Inhibiting uptake increases systemic exposure
        fold_change = 1.0 + conc / ki
    else
        # Inhibiting efflux increases intracellular and absorption
        fold_change = 1.0 / (1.0 - conc / (conc + ki))
    end

    return TransporterDDI(
        transporter,
        transporter_type,
        ki,
        ki * 2.0,  # IC50 ≈ 2 × Ki for competitive
        conc,
        fold_change,
        tissue
    )
end

"""
    oatp_inhibition(inhibitor_conc, ki_oatp1b1, ki_oatp1b3; fm_oatp=0.5) -> Float64

Calculate AUC ratio from OATP1B1/1B3 inhibition.
"""
function oatp_inhibition(
    inhibitor_conc::Float64,
    ki_oatp1b1::Float64,
    ki_oatp1b3::Float64;
    fm_oatp1b1::Float64 = 0.3,
    fm_oatp1b3::Float64 = 0.2
)::Float64

    # Inhibiting hepatic uptake increases AUC
    oatp1b1_effect = competitive_inhibition(inhibitor_conc, ki_oatp1b1)
    oatp1b3_effect = competitive_inhibition(inhibitor_conc, ki_oatp1b3)

    # Combined effect
    cl_ratio = fm_oatp1b1 / oatp1b1_effect + fm_oatp1b3 / oatp1b3_effect +
               (1.0 - fm_oatp1b1 - fm_oatp1b3)

    return 1.0 / cl_ratio  # AUC ratio
end

"""
    pgp_inhibition(inhibitor_conc, ki_pgp; tissue=:intestine) -> Float64

Calculate change in Fg or Fh from P-gp inhibition.
"""
function pgp_inhibition(
    inhibitor_conc::Float64,
    ki_pgp::Float64;
    tissue::Symbol = :intestine,
    pgp_contribution::Float64 = 0.3  # Fraction of substrate effluxed by P-gp
)::Float64

    inhibition = 1.0 - 1.0 / competitive_inhibition(inhibitor_conc, ki_pgp)

    if tissue == :intestine
        # P-gp inhibition increases intestinal absorption (Fg)
        return 1.0 + pgp_contribution * inhibition
    else
        # Hepatic P-gp inhibition decreases biliary excretion
        return 1.0 - pgp_contribution * inhibition * 0.5  # Partial effect
    end
end

"""
    bcrp_inhibition(inhibitor_conc, ki_bcrp) -> Float64

Calculate BCRP inhibition effect on absorption.
"""
function bcrp_inhibition(
    inhibitor_conc::Float64,
    ki_bcrp::Float64;
    bcrp_contribution::Float64 = 0.2
)::Float64

    inhibition = 1.0 - 1.0 / competitive_inhibition(inhibitor_conc, ki_bcrp)
    return 1.0 + bcrp_contribution * inhibition
end

"""
    hepatic_uptake_inhibition(inhibitor, substrate_fm_uptake) -> Float64

Calculate effect of hepatic uptake transporter inhibition on clearance.
"""
function hepatic_uptake_inhibition(
    inhibitor::InhibitorParams,
    fm_uptake::Dict{Symbol, Float64}  # Fraction via each transporter
)::Float64

    total_reduction = 1.0

    for (transporter, fm) in fm_uptake
        if transporter in inhibitor.target_transporters
            reduction = competitive_inhibition(inhibitor.cmax_unbound, inhibitor.ki)
            total_reduction *= (fm / reduction + (1.0 - fm))
        end
    end

    return 1.0 / total_reduction  # AUC ratio
end

"""
    combined_cyp_transporter_ddi(cyp_inhibition, transporter_inhibition, fg_baseline, fh_baseline) -> CombinedDDI

Calculate combined CYP and transporter DDI effect on total oral clearance.
"""
function combined_cyp_transporter_ddi(
    cyp_auc_ratio::Float64,
    transporter_auc_ratio::Float64;
    fg_baseline::Float64 = 0.8,
    fh_baseline::Float64 = 0.6,
    fg_change_cyp::Float64 = 1.0,
    fg_change_transport::Float64 = 1.0
)::CombinedDDI

    # Oral CL = F × CL/F = Fa × Fg × Fh × CLhepatic

    # CYP inhibition primarily affects Fh and CLhepatic
    # Transporter inhibition affects Fg (intestinal), uptake (Fh), and CLhepatic

    # Combined effect (multiplicative)
    total_auc_ratio = cyp_auc_ratio * transporter_auc_ratio

    # Estimate individual contributions
    fg_new = min(1.0, fg_baseline * fg_change_cyp * fg_change_transport)
    fh_new = min(1.0, fh_baseline * cyp_auc_ratio^0.5)  # Approximate

    cl_oral_ratio = 1.0 / total_auc_ratio

    return CombinedDDI(
        cyp_auc_ratio,
        transporter_auc_ratio,
        total_auc_ratio,
        total_auc_ratio^0.7,  # Cmax typically less affected than AUC
        fg_new / fg_baseline,
        fh_new / fh_baseline,
        cl_oral_ratio
    )
end

# ===========================================================================
# Monte Carlo DDI Prediction
# ===========================================================================

"""
    monte_carlo_ddi_prediction(perpetrator, victim_fm, n_sim; parameters_cv) -> DDIUncertainty

Run Monte Carlo simulation for DDI prediction with uncertainty.
"""
function monte_carlo_ddi_prediction(
    perpetrator::InhibitorParams,
    victim_fm::Dict{Symbol, Float64},
    victim_km::Dict{Symbol, Float64};
    n_sim::Int = 1000,
    ki_cv::Float64 = 0.3,
    cmax_cv::Float64 = 0.25,
    fm_cv::Float64 = 0.2
)::DDIUncertainty

    Random.seed!(42)  # For reproducibility
    auc_ratios = Float64[]

    for _ in 1:n_sim
        # Sample parameters from distributions
        ki_sample = perpetrator.ki * exp(randn() * ki_cv)
        cmax_sample = perpetrator.cmax_unbound * exp(randn() * cmax_cv)

        # Calculate inhibition for each enzyme
        total_cl_ratio = 0.0

        for (enzyme, fm) in victim_fm
            if enzyme in perpetrator.target_enzymes
                # Sample fm
                fm_sample = clamp(fm * exp(randn() * fm_cv), 0.0, 1.0)

                # Calculate inhibition
                if perpetrator.inhibitor_type == :mbi
                    kdeg = log(2) / get(PHYSIOLOGICAL_PARAMS, "$(lowercase(string(enzyme)))_halflife", 36.0)
                    enzyme_active = mechanism_based_inhibition(
                        perpetrator.kinact, ki_sample, cmax_sample, kdeg
                    )
                    inhibition_factor = 1.0 / enzyme_active
                else
                    inhibition_factor = competitive_inhibition(cmax_sample, ki_sample)
                end

                total_cl_ratio += fm_sample / inhibition_factor
            else
                total_cl_ratio += fm
            end
        end

        # Add non-CYP clearance
        total_cl_ratio += (1.0 - sum(values(victim_fm)))

        push!(auc_ratios, 1.0 / total_cl_ratio)
    end

    # Calculate statistics
    mean_ratio = mean(auc_ratios)
    sd_ratio = std(auc_ratios)
    sorted = sort(auc_ratios)

    ci_90_lower = sorted[Int(ceil(n_sim * 0.05))]
    ci_90_upper = sorted[Int(floor(n_sim * 0.95))]
    ci_95_lower = sorted[Int(ceil(n_sim * 0.025))]
    ci_95_upper = sorted[Int(floor(n_sim * 0.975))]

    p_exceeds_2fold = sum(auc_ratios .> 2.0) / n_sim
    p_exceeds_5fold = sum(auc_ratios .> 5.0) / n_sim

    return DDIUncertainty(
        mean_ratio,
        sd_ratio,
        ci_90_lower,
        ci_90_upper,
        ci_95_lower,
        ci_95_upper,
        p_exceeds_2fold,
        p_exceeds_5fold,
        n_sim
    )
end

"""
    ddi_sensitivity_analysis(perpetrator, victim, parameter_ranges) -> Dict

Perform local sensitivity analysis on DDI prediction.
"""
function ddi_sensitivity_analysis(
    base_auc_ratio::Float64,
    perpetrator::InhibitorParams,
    victim_fm::Dict{Symbol, Float64};
    parameter_perturbation::Float64 = 0.1
)::Dict{String, Float64}

    sensitivities = Dict{String, Float64}()

    # Ki sensitivity
    ki_high = perpetrator.ki * (1 + parameter_perturbation)
    ki_low = perpetrator.ki * (1 - parameter_perturbation)

    # Recalculate with perturbed Ki
    # (Simplified - would need full recalculation in production)
    auc_ki_high = base_auc_ratio * perpetrator.ki / ki_high
    auc_ki_low = base_auc_ratio * perpetrator.ki / ki_low

    sensitivities["Ki"] = (auc_ki_high - auc_ki_low) / (2 * parameter_perturbation * base_auc_ratio)

    # Cmax sensitivity
    sensitivities["Cmax"] = (auc_ki_low - auc_ki_high) / (2 * parameter_perturbation * base_auc_ratio)

    # fm sensitivity for each enzyme
    for (enzyme, fm) in victim_fm
        sensitivities["fm_$(enzyme)"] = fm / sum(values(victim_fm))
    end

    return sensitivities
end

"""
    calculate_ddi_confidence_interval(uncertainty; level=0.90) -> Tuple{Float64, Float64}

Extract confidence interval from DDI uncertainty analysis.
"""
function calculate_ddi_confidence_interval(
    uncertainty::DDIUncertainty;
    level::Float64 = 0.90
)::Tuple{Float64, Float64}

    if level == 0.90
        return (uncertainty.ci_90_lower, uncertainty.ci_90_upper)
    elseif level == 0.95
        return (uncertainty.ci_95_lower, uncertainty.ci_95_upper)
    else
        # Approximate using normal distribution
        z = quantile(Normal(), (1 + level) / 2)
        lower = uncertainty.mean_ratio - z * uncertainty.sd_ratio
        upper = uncertainty.mean_ratio + z * uncertainty.sd_ratio
        return (max(1.0, lower), upper)
    end
end

"""
    classify_ddi_severity(auc_ratio; margin=0.25) -> Symbol

Classify DDI risk based on AUC ratio.
"""
function classify_ddi_severity(auc_ratio::Float64; margin::Float64 = 0.25)::Symbol
    if auc_ratio < 1.0 + margin
        return :no_interaction
    elseif auc_ratio < 2.0
        return :weak
    elseif auc_ratio < 5.0
        return :moderate
    else
        return :strong
    end
end

# ===========================================================================
# Main Prediction Functions
# ===========================================================================

"""
    predict_ddi_mechanistic(perpetrator, victim_params, ivive_params) -> DDIPredictionResult

Comprehensive mechanistic DDI prediction with IVIVE.
"""
function predict_ddi_mechanistic(
    perpetrator::InhibitorParams,
    victim_fm::Dict{Symbol, Float64},
    victim_ivive::IVIVEParams;
    include_gut::Bool = true,
    include_transporters::Bool = true,
    n_monte_carlo::Int = 1000
)::DDIPredictionResult

    # 1. Calculate hepatic CYP inhibition
    hepatic_cl_ratio = 1.0
    cyp_contributions = Dict{Symbol, Float64}()

    for (enzyme, fm) in victim_fm
        if enzyme in perpetrator.target_enzymes
            if perpetrator.inhibitor_type == :mbi
                # Mechanism-based inhibition
                kdeg = log(2) / get(PHYSIOLOGICAL_PARAMS, "$(lowercase(string(enzyme)))_halflife", 36.0)
                enzyme_active = mechanism_based_inhibition(
                    perpetrator.kinact, perpetrator.ki_mbi,
                    perpetrator.cmax_unbound, kdeg
                )
                contribution = fm / enzyme_active
            else
                # Reversible inhibition
                inhibition_factor = competitive_inhibition(perpetrator.cmax_unbound, perpetrator.ki)
                contribution = fm / inhibition_factor
            end
            cyp_contributions[enzyme] = contribution / fm
        else
            contribution = fm
            cyp_contributions[enzyme] = 1.0
        end
        hepatic_cl_ratio += contribution - fm
    end

    hepatic_auc_ratio = 1.0 / hepatic_cl_ratio

    # 2. Calculate gut wall inhibition (if applicable)
    gut_auc_ratio = 1.0
    if include_gut && perpetrator.ig > 0
        for (enzyme, fm) in victim_fm
            if enzyme in [:CYP3A4, :CYP3A5] && enzyme in perpetrator.target_enzymes
                # Gut concentration is much higher than systemic
                gut_inhibition = competitive_inhibition(perpetrator.ig, perpetrator.ki)
                fg_baseline = 0.8  # Typical Fg
                fg_change = 1.0 + (1.0 - fg_baseline) * (gut_inhibition - 1.0) / gut_inhibition
                gut_auc_ratio *= fg_change
            end
        end
    end

    # 3. Calculate transporter DDI
    transporter_auc_ratio = 1.0
    transporter_contributions = Dict{Symbol, Float64}()

    if include_transporters
        for transporter in perpetrator.target_transporters
            if transporter == :OATP1B1
                tr_ratio = oatp_inhibition(perpetrator.cmax_unbound, perpetrator.ki, perpetrator.ki * 2)
                transporter_auc_ratio *= tr_ratio
                transporter_contributions[transporter] = tr_ratio
            elseif transporter == :P_gp
                pgp_effect = pgp_inhibition(perpetrator.ig, perpetrator.ki)
                transporter_auc_ratio *= pgp_effect
                transporter_contributions[transporter] = pgp_effect
            end
        end
    end

    # 4. Combined AUC ratio
    total_auc_ratio = hepatic_auc_ratio * gut_auc_ratio * transporter_auc_ratio

    # 5. Estimate Cmax ratio (typically less affected than AUC)
    cmax_ratio = total_auc_ratio^0.6

    # 6. Monte Carlo uncertainty
    victim_km = Dict(e => 10.0 for e in keys(victim_fm))  # Default Km
    uncertainty = monte_carlo_ddi_prediction(
        perpetrator, victim_fm, victim_km;
        n_sim = n_monte_carlo
    )

    # 7. Classification
    severity = classify_ddi_severity(total_auc_ratio)

    fda_class = if total_auc_ratio >= 5.0
        "Strong Inhibitor"
    elseif total_auc_ratio >= 2.0
        "Moderate Inhibitor"
    elseif total_auc_ratio >= 1.25
        "Weak Inhibitor"
    else
        "No Significant Interaction"
    end

    # 8. Generate recommendations
    recommendations = String[]
    if severity == :strong
        push!(recommendations, "Contraindicated with sensitive substrates")
        push!(recommendations, "Consider alternative medication")
        push!(recommendations, "If unavoidable, reduce substrate dose by ≥75%")
    elseif severity == :moderate
        push!(recommendations, "Use with caution")
        push!(recommendations, "Consider substrate dose reduction of 50%")
        push!(recommendations, "Monitor for adverse effects")
    elseif severity == :weak
        push!(recommendations, "Clinically manageable interaction")
        push!(recommendations, "No dose adjustment typically required")
        push!(recommendations, "Monitor clinical response")
    end

    return DDIPredictionResult(
        perpetrator.name,
        "Victim",  # Would be passed in production
        total_auc_ratio,
        cmax_ratio,
        1.0 / total_auc_ratio,  # CL ratio
        perpetrator.inhibitor_type,
        severity,
        fda_class,
        uncertainty,
        recommendations,
        cyp_contributions,
        transporter_contributions
    )
end

"""
    simulate_ddi_time_course(perpetrator, victim, duration; dt=0.1) -> Vector{Dict}

Simulate DDI time course for multiple dosing.
"""
function simulate_ddi_time_course(
    perpetrator::InhibitorParams,
    victim_fm::Dict{Symbol, Float64};
    duration::Float64 = 168.0,  # hours (1 week)
    perpetrator_dose_interval::Float64 = 24.0,  # hours
    dt::Float64 = 0.1  # hours
)::Vector{Dict{String, Float64}}

    results = Dict{String, Float64}[]

    # State variables
    enzyme_activity = Dict(e => 1.0 for e in keys(victim_fm))
    perpetrator_conc = 0.0

    # Parameters
    perpetrator_ka = 1.0  # h⁻¹
    perpetrator_ke = 0.1  # h⁻¹

    t = 0.0
    next_dose = 0.0

    while t <= duration
        # Dose perpetrator
        if t >= next_dose
            perpetrator_conc += perpetrator.cmax_unbound
            next_dose += perpetrator_dose_interval
        end

        # Perpetrator PK (simple 1-compartment)
        perpetrator_conc *= exp(-perpetrator_ke * dt)

        # Update enzyme activity
        for enzyme in keys(enzyme_activity)
            if enzyme in perpetrator.target_enzymes
                if perpetrator.inhibitor_type == :mbi
                    kdeg = log(2) / get(PHYSIOLOGICAL_PARAMS, "$(lowercase(string(enzyme)))_halflife", 36.0)
                    kinact_app = perpetrator.kinact * perpetrator_conc / (perpetrator.ki_mbi + perpetrator_conc)

                    # dE/dt = kdeg - (kdeg + kinact_app) × E
                    dE = kdeg - (kdeg + kinact_app) * enzyme_activity[enzyme]
                    enzyme_activity[enzyme] += dE * dt
                    enzyme_activity[enzyme] = clamp(enzyme_activity[enzyme], 0.01, 1.0)
                end
            end
        end

        # Calculate current AUC ratio
        cl_ratio = sum(victim_fm[e] * enzyme_activity[e] for e in keys(victim_fm)) +
                   (1.0 - sum(values(victim_fm)))
        auc_ratio = 1.0 / cl_ratio

        push!(results, Dict(
            "time" => t,
            "perpetrator_conc" => perpetrator_conc,
            "auc_ratio" => auc_ratio,
            "enzyme_activity" => mean(values(enzyme_activity))
        ))

        t += dt
    end

    return results
end

"""
    create_ddi_report(prediction) -> String

Generate formatted DDI report for regulatory submission.
"""
function create_ddi_report(prediction::DDIPredictionResult)::String
    report = """
    ===============================================================================
    DRUG-DRUG INTERACTION PREDICTION REPORT
    ===============================================================================

    INTERACTION SUMMARY
    -------------------
    Perpetrator: $(prediction.perpetrator)
    Victim: $(prediction.victim)

    PREDICTED PHARMACOKINETIC CHANGES
    ---------------------------------
    AUC Ratio: $(round(prediction.auc_ratio, digits=2))-fold increase
    Cmax Ratio: $(round(prediction.cmax_ratio, digits=2))-fold increase
    Clearance Ratio: $(round(prediction.cl_ratio, digits=3))

    CLASSIFICATION
    --------------
    Severity: $(prediction.severity)
    FDA Classification: $(prediction.fda_classification)
    Mechanism: $(prediction.mechanism)

    UNCERTAINTY ANALYSIS (Monte Carlo, n=$(prediction.uncertainty.n_simulations))
    ----------------------------------------
    Mean AUC Ratio: $(round(prediction.uncertainty.mean_ratio, digits=2))
    SD: $(round(prediction.uncertainty.sd_ratio, digits=2))
    90% CI: [$(round(prediction.uncertainty.ci_90_lower, digits=2)), $(round(prediction.uncertainty.ci_90_upper, digits=2))]
    95% CI: [$(round(prediction.uncertainty.ci_95_lower, digits=2)), $(round(prediction.uncertainty.ci_95_upper, digits=2))]
    P(AUC ratio > 2): $(round(prediction.uncertainty.p_exceeds_2fold * 100, digits=1))%
    P(AUC ratio > 5): $(round(prediction.uncertainty.p_exceeds_5fold * 100, digits=1))%

    CYP ENZYME CONTRIBUTIONS
    ------------------------
    """

    for (enzyme, contribution) in prediction.cyp_contributions
        report *= "$(enzyme): $(round(contribution, digits=2))-fold change\n"
    end

    if !isempty(prediction.transporter_contributions)
        report *= """

    TRANSPORTER CONTRIBUTIONS
    -------------------------
    """
        for (transporter, contribution) in prediction.transporter_contributions
            report *= "$(transporter): $(round(contribution, digits=2))-fold change\n"
        end
    end

    report *= """

    CLINICAL RECOMMENDATIONS
    ------------------------
    """
    for rec in prediction.recommendations
        report *= "• $(rec)\n"
    end

    report *= """

    ===============================================================================
    Report generated by Darwin PBPK Platform - Mechanistic DDI Module v1.0.0
    ===============================================================================
    """

    return report
end

"""
    validate_against_clinical(predicted_ratio, observed_ratio; acceptance_criterion=2.0) -> Bool

Validate DDI prediction against clinical data.
"""
function validate_against_clinical(
    predicted_ratio::Float64,
    observed_ratio::Float64;
    acceptance_criterion::Float64 = 2.0
)::Tuple{Bool, Float64}

    # FDA criterion: predicted within 2-fold of observed
    fold_error = predicted_ratio / observed_ratio
    if fold_error < 1.0
        fold_error = 1.0 / fold_error
    end

    passes = fold_error <= acceptance_criterion

    return (passes, fold_error)
end

end # module
