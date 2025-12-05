"""
Platelet Compartment - Detailed Modeling with Activation and Aggregation

Models platelet physiology with:
- Physical parameters (count, MPV, lifespan)
- Granule contents (alpha, dense granules)
- Activation pathways (ADP/P2Y12, TXA2/COX-1, PAR-1)
- Aggregation dynamics
- Antiplatelet drug effects (aspirin, clopidogrel, ticagrelor)
- Integration with coagulation cascade

Based on SOTA Q1 literature 2020-2024:
- Wajima et al. (2009) - Coagulation network
- QSP models for antiplatelet agents

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

module Platelets

using LinearAlgebra
using Statistics

export PlateletCompartment, PlateletGranules, PlateletActivation
export create_platelet_compartment, activate_platelets!, aggregate_platelets!
export apply_antiplatelet_drug!, calculate_bleeding_risk
export simulate_platelet_dynamics, get_platelet_state
export NORMAL_PLATELET_COUNT, NORMAL_MPV

# ============================================================================
# CONSTANTS - From SOTA Literature
# ============================================================================

# Physical parameters
const NORMAL_PLATELET_COUNT = 250e9  # cells/L (range: 150-400 × 10⁹/L)
const NORMAL_MPV = 7.5               # fL (Mean Platelet Volume, range: 4.5-11 fL)
const PLATELET_DIAMETER = 3.5        # μm (range: 3-5 μm)
const PLATELET_LIFESPAN = 9.0        # days (8-10 days)
const PLATELET_TURNOVER = 35e9       # cells/day (replacement rate)

# Granule counts per platelet
const ALPHA_GRANULES_PER_PLATELET = 65   # 50-80 per platelet
const DENSE_GRANULES_PER_PLATELET = 7    # 6-7 per platelet

# Dense granule contents (concentrations within granule)
const ADP_GRANULE_CONCENTRATION = 0.5    # M (0.4-0.6 M)
const ATP_GRANULE_CONCENTRATION = 0.5    # M
const SEROTONIN_GRANULE_CONCENTRATION = 65e-3  # M (65 mM)
const CALCIUM_GRANULE_CONCENTRATION = 2.2 # M

# EC50 values for agonists
const EC50_ADP = 1e-6          # M (1 μM for P2Y12)
const EC50_TXA2 = 10e-9        # M (10 nM for TP receptor)
const EC50_THROMBIN = 0.1e-9   # M (0.1 nM for PAR-1)
const EC50_COLLAGEN = 1e-6     # M (1 μg/mL equivalent)

# Kinetic rate constants
const K_ACTIVATION_BASELINE = 0.1    # 1/s (baseline activation rate)
const K_AGGREGATION = 0.01           # 1/(s·cells/L) (second-order)
const K_DISAGGREGATION = 0.001       # 1/s

# Antiplatelet drug parameters (from SOTA spec)
const ASPIRIN_IC50_COX1 = 3e-6       # M (IC50 = 3 μM)
const TICAGRELOR_KI_P2Y12 = 2e-9     # M (Ki = 2 nM, reversible)
const CLOPIDOGREL_IRREVERSIBLE = true
const PRASUGREL_IRREVERSIBLE = true
const ABCIXIMAB_KD_GPIIB_IIIA = 5e-9     # M (Kd = 5 nM)
const EPTIFIBATIDE_KI_GPIIB_IIIA = 120e-12  # M (Ki = 120 pM)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
PlateletGranules - Models alpha and dense granule contents

Alpha granules: Fibrinogen, vWF, Factor V, PAI-1, PDGF, P-selectin
Dense granules: ADP, ATP, Serotonin, Ca²⁺, Polyphosphate
"""
struct PlateletGranules
    # Alpha granule contents (fraction available, 0-1)
    fibrinogen::Float64       # Available for clot formation
    vwf::Float64              # von Willebrand factor
    factor_v::Float64         # Coagulation factor V
    pai_1::Float64            # Plasminogen activator inhibitor
    pdgf::Float64             # Platelet-derived growth factor
    p_selectin::Float64       # Adhesion molecule

    # Dense granule contents (fraction available, 0-1)
    adp::Float64              # ADP for autocrine activation
    atp::Float64              # ATP
    serotonin::Float64        # 5-HT, vasoconstriction
    calcium::Float64          # Ca²⁺ signaling
    polyphosphate::Float64    # Coagulation enhancer
end

"""
PlateletActivation - Models activation state and pathways

Pathways:
1. ADP → P2Y12 → Gi → cAMP↓ → sustained aggregation
2. TXA2 → TP → Gq → Ca²⁺↑ → shape change + aggregation
3. Thrombin → PAR-1 → Gq → strongest activator
4. Collagen → GPVI → activation
"""
mutable struct PlateletActivation
    # Activation state (0-1 scale)
    resting_fraction::Float64      # Fraction of resting platelets
    activated_fraction::Float64    # Fraction of activated (shape change)
    aggregated_fraction::Float64   # Fraction in aggregates

    # Pathway activation levels (0-1 scale)
    p2y12_activation::Float64      # ADP/P2Y12 pathway
    txa2_activation::Float64       # TXA2/TP pathway
    par1_activation::Float64       # Thrombin/PAR-1 pathway
    gpvi_activation::Float64       # Collagen/GPVI pathway

    # GPIIb/IIIa activation (final common pathway)
    gpiib_iiia_active::Float64     # Fibrinogen receptor activation (0-1)

    # Agonist concentrations at platelet surface (M)
    local_adp::Float64
    local_txa2::Float64
    local_thrombin::Float64
    local_collagen::Float64
end

"""
PlateletCompartment - Complete platelet model

Integrates:
- Physical parameters
- Granule contents
- Activation state
- Antiplatelet drug effects
"""
mutable struct PlateletCompartment
    # Physical parameters
    count::Float64                    # cells/L
    mean_platelet_volume::Float64     # fL (MPV)
    volume_fraction::Float64          # Fraction of blood volume

    # Granule state
    granules::PlateletGranules

    # Activation state
    activation::PlateletActivation

    # Antiplatelet drug inhibition (0-1, 0=no inhibition)
    cox1_inhibition::Float64          # Aspirin effect
    p2y12_inhibition::Float64         # Clopidogrel/ticagrelor effect
    gpiib_iiia_inhibition::Float64    # Abciximab effect

    # Pathology state
    pathology::String                 # "normal", "thrombocytopenia", "thrombocytosis"
    pathology_severity::Float64       # 0-1 scale

    # Platelet function tests (calculated)
    aggregation_response::Float64     # % max aggregation to ADP
    closure_time::Float64             # seconds (PFA-100 equivalent)
end

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

"""
create_platelet_granules(release_fraction=0.0)

Create platelet granules with specified release fraction.
release_fraction = 0 → fully loaded granules
release_fraction = 1 → fully released (degranulated)
"""
function create_platelet_granules(release_fraction::Float64=0.0)::PlateletGranules
    available = 1.0 - release_fraction

    return PlateletGranules(
        available,  # fibrinogen
        available,  # vwf
        available,  # factor_v
        available,  # pai_1
        available,  # pdgf
        available,  # p_selectin
        available,  # adp
        available,  # atp
        available,  # serotonin
        available,  # calcium
        available   # polyphosphate
    )
end

"""
create_platelet_activation()

Create initial (resting) platelet activation state.
"""
function create_platelet_activation()::PlateletActivation
    return PlateletActivation(
        1.0,    # resting_fraction (100% resting)
        0.0,    # activated_fraction
        0.0,    # aggregated_fraction
        0.0,    # p2y12_activation
        0.0,    # txa2_activation
        0.0,    # par1_activation
        0.0,    # gpvi_activation
        0.0,    # gpiib_iiia_active
        0.0,    # local_adp
        0.0,    # local_txa2
        0.0,    # local_thrombin
        0.0     # local_collagen
    )
end

"""
create_platelet_compartment(; count, mpv, pathology, pathology_severity)

Create complete platelet compartment with physiological parameters.

# Arguments
- `count`: Platelet count (cells/L), default: 250×10⁹/L
- `mpv`: Mean platelet volume (fL), default: 7.5 fL
- `pathology`: "normal", "thrombocytopenia", "thrombocytosis", "itp"
- `pathology_severity`: 0-1 scale

# Returns
- PlateletCompartment with initialized state
"""
function create_platelet_compartment(;
    count::Float64=NORMAL_PLATELET_COUNT,
    mpv::Float64=NORMAL_MPV,
    pathology::String="normal",
    pathology_severity::Float64=0.0
)::PlateletCompartment

    # Adjust count for pathology
    adjusted_count = apply_pathology_to_count(count, pathology, pathology_severity)

    # Calculate volume fraction
    volume_fraction = (adjusted_count * mpv * 1e-15)  # fL → L

    return PlateletCompartment(
        adjusted_count,
        mpv,
        volume_fraction,
        create_platelet_granules(),
        create_platelet_activation(),
        0.0,  # cox1_inhibition
        0.0,  # p2y12_inhibition
        0.0,  # gpiib_iiia_inhibition
        pathology,
        pathology_severity,
        100.0,  # aggregation_response (% max)
        150.0   # closure_time (seconds, normal: 84-160s)
    )
end

"""
apply_pathology_to_count(count, pathology, severity)

Adjust platelet count based on pathology.
"""
function apply_pathology_to_count(
    count::Float64,
    pathology::String,
    severity::Float64
)::Float64

    if pathology == "normal"
        return count

    elseif pathology == "thrombocytopenia"
        # Severity 1.0 → count drops to 10% (severe: <50×10⁹/L)
        return count * (1.0 - severity * 0.9)

    elseif pathology == "thrombocytosis"
        # Severity 1.0 → count increases 4× (>1000×10⁹/L)
        return count * (1.0 + severity * 3.0)

    elseif pathology == "itp"  # Immune thrombocytopenia
        # Similar to thrombocytopenia but immune-mediated
        return count * (1.0 - severity * 0.95)

    else
        @warn "Unknown pathology: $pathology, using normal count"
        return count
    end
end

# ============================================================================
# ACTIVATION DYNAMICS
# ============================================================================

"""
activate_platelets!(compartment, agonists, dt)

Update platelet activation state based on agonist concentrations.

# Arguments
- `compartment`: PlateletCompartment to modify
- `agonists`: NamedTuple with (adp, txa2, thrombin, collagen) concentrations (M)
- `dt`: Time step (seconds)
"""
function activate_platelets!(
    compartment::PlateletCompartment,
    agonists::NamedTuple,
    dt::Float64
)
    act = compartment.activation

    # Update local agonist concentrations
    act.local_adp = get(agonists, :adp, 0.0)
    act.local_txa2 = get(agonists, :txa2, 0.0)
    act.local_thrombin = get(agonists, :thrombin, 0.0)
    act.local_collagen = get(agonists, :collagen, 0.0)

    # Calculate pathway activations (Hill equation, n=1)
    # With drug inhibition effects

    # P2Y12 pathway (ADP) - blocked by clopidogrel/ticagrelor
    adp_effect = act.local_adp / (EC50_ADP + act.local_adp)
    act.p2y12_activation = adp_effect * (1.0 - compartment.p2y12_inhibition)

    # TXA2 pathway - blocked by aspirin (COX-1)
    txa2_effect = act.local_txa2 / (EC50_TXA2 + act.local_txa2)
    act.txa2_activation = txa2_effect * (1.0 - compartment.cox1_inhibition)

    # PAR-1 pathway (thrombin) - strongest activator
    thrombin_effect = act.local_thrombin / (EC50_THROMBIN + act.local_thrombin)
    act.par1_activation = thrombin_effect

    # GPVI pathway (collagen)
    collagen_effect = act.local_collagen / (EC50_COLLAGEN + act.local_collagen)
    act.gpvi_activation = collagen_effect

    # Total activation signal (weighted combination)
    total_signal = (
        0.3 * act.p2y12_activation +    # ADP is amplifier
        0.2 * act.txa2_activation +     # TXA2 is amplifier
        0.35 * act.par1_activation +    # Thrombin is primary
        0.15 * act.gpvi_activation      # Collagen initiates
    )

    # GPIIb/IIIa activation (final common pathway)
    # Blocked by abciximab/eptifibatide
    act.gpiib_iiia_active = min(1.0, total_signal * 1.5) * (1.0 - compartment.gpiib_iiia_inhibition)

    # Update platelet fractions
    activation_rate = K_ACTIVATION_BASELINE * total_signal

    # Resting → Activated
    d_activated = activation_rate * act.resting_fraction * dt
    act.resting_fraction = max(0.0, act.resting_fraction - d_activated)
    act.activated_fraction = min(1.0, act.activated_fraction + d_activated)

    return nothing
end

"""
aggregate_platelets!(compartment, fibrinogen_conc, dt)

Update platelet aggregation based on GPIIb/IIIa activation and fibrinogen.

# Arguments
- `compartment`: PlateletCompartment to modify
- `fibrinogen_conc`: Plasma fibrinogen concentration (M), normal ~7-12 μM
- `dt`: Time step (seconds)
"""
function aggregate_platelets!(
    compartment::PlateletCompartment,
    fibrinogen_conc::Float64,
    dt::Float64
)
    act = compartment.activation

    # Aggregation requires GPIIb/IIIa activation AND fibrinogen
    # Second-order kinetics (activated platelets interact)

    fibrinogen_factor = fibrinogen_conc / (fibrinogen_conc + 5e-6)  # Km ~5 μM

    # Aggregation rate
    k_agg = K_AGGREGATION * act.gpiib_iiia_active * fibrinogen_factor

    # Activated → Aggregated (second-order in activated fraction)
    d_aggregated = k_agg * act.activated_fraction^2 * dt

    # Disaggregation
    d_disaggregated = K_DISAGGREGATION * act.aggregated_fraction * dt

    # Update fractions
    net_change = d_aggregated - d_disaggregated
    act.activated_fraction = max(0.0, act.activated_fraction - d_aggregated)
    act.aggregated_fraction = clamp(act.aggregated_fraction + net_change, 0.0, 1.0)

    # Granule release during aggregation
    release_granules!(compartment, act.aggregated_fraction)

    return nothing
end

"""
release_granules!(compartment, aggregation_level)

Release granule contents based on aggregation level.
"""
function release_granules!(
    compartment::PlateletCompartment,
    aggregation_level::Float64
)
    # Granules are released proportionally to aggregation
    # Dense granules release faster than alpha granules

    release_factor_dense = min(1.0, aggregation_level * 2.0)
    release_factor_alpha = min(1.0, aggregation_level * 1.5)

    g = compartment.granules

    # Create new granules with reduced contents
    compartment.granules = PlateletGranules(
        g.fibrinogen * (1.0 - release_factor_alpha * 0.5),
        g.vwf * (1.0 - release_factor_alpha * 0.5),
        g.factor_v * (1.0 - release_factor_alpha * 0.5),
        g.pai_1 * (1.0 - release_factor_alpha * 0.5),
        g.pdgf * (1.0 - release_factor_alpha * 0.3),
        g.p_selectin * (1.0 - release_factor_alpha * 0.7),
        g.adp * (1.0 - release_factor_dense * 0.8),
        g.atp * (1.0 - release_factor_dense * 0.8),
        g.serotonin * (1.0 - release_factor_dense * 0.9),
        g.calcium * (1.0 - release_factor_dense * 0.7),
        g.polyphosphate * (1.0 - release_factor_dense * 0.6)
    )

    return nothing
end

# ============================================================================
# ANTIPLATELET DRUG EFFECTS
# ============================================================================

"""
apply_antiplatelet_drug!(compartment, drug_name, concentration)

Apply antiplatelet drug effect to platelet compartment.

# Arguments
- `compartment`: PlateletCompartment to modify
- `drug_name`: Drug name (aspirin, clopidogrel, prasugrel, ticagrelor, abciximab, eptifibatide)
- `concentration`: Drug concentration at platelet (M)
"""
function apply_antiplatelet_drug!(
    compartment::PlateletCompartment,
    drug_name::String,
    concentration::Float64
)
    drug = lowercase(drug_name)

    if drug == "aspirin"
        # Irreversible COX-1 inhibition (Emax model with Hill=1)
        compartment.cox1_inhibition = concentration / (ASPIRIN_IC50_COX1 + concentration)

    elseif drug in ["clopidogrel", "prasugrel"]
        # Irreversible P2Y12 inhibition (active metabolite)
        # Model as cumulative effect based on dose history
        # Simplified: assume concentration represents active metabolite
        compartment.p2y12_inhibition = min(1.0,
            compartment.p2y12_inhibition + concentration / 1e-6 * 0.1)

    elseif drug == "ticagrelor"
        # Reversible P2Y12 inhibition
        compartment.p2y12_inhibition = concentration / (TICAGRELOR_KI_P2Y12 + concentration)

    elseif drug == "abciximab"
        # GPIIb/IIIa inhibition (monoclonal antibody)
        compartment.gpiib_iiia_inhibition = concentration / (ABCIXIMAB_KD_GPIIB_IIIA + concentration)

    elseif drug == "eptifibatide"
        # GPIIb/IIIa inhibition (cyclic peptide)
        compartment.gpiib_iiia_inhibition = concentration / (EPTIFIBATIDE_KI_GPIIB_IIIA + concentration)

    else
        @warn "Unknown antiplatelet drug: $drug_name"
    end

    # Update platelet function tests
    update_platelet_function_tests!(compartment)

    return nothing
end

"""
update_platelet_function_tests!(compartment)

Update calculated platelet function test values.
"""
function update_platelet_function_tests!(compartment::PlateletCompartment)
    # Aggregation response (% of max to ADP)
    # Reduced by P2Y12 inhibition
    compartment.aggregation_response = 100.0 * (1.0 - compartment.p2y12_inhibition * 0.9)

    # Closure time (PFA-100 equivalent)
    # Increased by all antiplatelet drugs
    total_inhibition = (
        compartment.cox1_inhibition * 0.4 +
        compartment.p2y12_inhibition * 0.4 +
        compartment.gpiib_iiia_inhibition * 0.6
    )
    # Normal: 84-160s, prolonged >200s
    compartment.closure_time = 120.0 / (1.0 - total_inhibition * 0.7)

    return nothing
end

# ============================================================================
# SIMULATION
# ============================================================================

"""
simulate_platelet_dynamics(compartment, agonist_profile, tspan, dt)

Simulate platelet activation and aggregation over time.

# Arguments
- `compartment`: Initial PlateletCompartment
- `agonist_profile`: Function(t) → NamedTuple(adp, txa2, thrombin, collagen)
- `tspan`: (t_start, t_end) in seconds
- `dt`: Time step in seconds

# Returns
- Vector of time points
- Vector of PlateletCompartment states (copies)
"""
function simulate_platelet_dynamics(
    compartment::PlateletCompartment,
    agonist_profile::Function,
    tspan::Tuple{Float64, Float64},
    dt::Float64;
    fibrinogen_conc::Float64=10e-6  # 10 μM (normal)
)
    t_start, t_end = tspan
    times = collect(t_start:dt:t_end)
    n_steps = length(times)

    # Deep copy initial compartment
    current = deepcopy(compartment)

    # Store results
    results = Vector{Dict{String, Float64}}(undef, n_steps)

    for (i, t) in enumerate(times)
        # Get agonist concentrations at time t
        agonists = agonist_profile(t)

        # Update activation
        activate_platelets!(current, agonists, dt)

        # Update aggregation
        aggregate_platelets!(current, fibrinogen_conc, dt)

        # Store state
        results[i] = Dict(
            "time" => t,
            "resting" => current.activation.resting_fraction,
            "activated" => current.activation.activated_fraction,
            "aggregated" => current.activation.aggregated_fraction,
            "gpiib_iiia" => current.activation.gpiib_iiia_active,
            "adp_released" => 1.0 - current.granules.adp,
            "txa2_activation" => current.activation.txa2_activation,
            "p2y12_activation" => current.activation.p2y12_activation
        )
    end

    return times, results
end

# ============================================================================
# CLINICAL ENDPOINTS
# ============================================================================

"""
calculate_bleeding_risk(compartment)

Calculate relative bleeding risk based on platelet state.

# Returns
- bleeding_risk: Relative risk (1.0 = normal)
"""
function calculate_bleeding_risk(compartment::PlateletCompartment)::Float64
    # Factors increasing bleeding risk:
    # 1. Low platelet count
    # 2. Antiplatelet drug effects
    # 3. Impaired aggregation

    # Count factor (exponential increase below 50×10⁹/L)
    count_ratio = compartment.count / NORMAL_PLATELET_COUNT
    count_factor = if count_ratio > 0.5
        1.0
    elseif count_ratio > 0.2
        1.0 + (0.5 - count_ratio) / 0.3 * 2.0  # Up to 3×
    else
        3.0 + (0.2 - count_ratio) / 0.2 * 7.0  # Up to 10×
    end

    # Drug inhibition factor
    drug_factor = (
        1.0 +
        compartment.cox1_inhibition * 1.5 +
        compartment.p2y12_inhibition * 2.0 +
        compartment.gpiib_iiia_inhibition * 3.0
    )

    # Aggregation factor
    aggregation_factor = 100.0 / max(10.0, compartment.aggregation_response)

    return count_factor * drug_factor * aggregation_factor
end

"""
get_platelet_state(compartment)

Get summary of current platelet state for reporting.

# Returns
- Dict with all relevant parameters
"""
function get_platelet_state(compartment::PlateletCompartment)::Dict{String, Any}
    return Dict(
        "count" => compartment.count,
        "count_per_uL" => compartment.count / 1e6,  # Convert to /μL
        "mpv_fL" => compartment.mean_platelet_volume,
        "volume_fraction" => compartment.volume_fraction,

        "activation" => Dict(
            "resting_pct" => compartment.activation.resting_fraction * 100,
            "activated_pct" => compartment.activation.activated_fraction * 100,
            "aggregated_pct" => compartment.activation.aggregated_fraction * 100,
            "gpiib_iiia_active" => compartment.activation.gpiib_iiia_active
        ),

        "drug_inhibition" => Dict(
            "cox1" => compartment.cox1_inhibition,
            "p2y12" => compartment.p2y12_inhibition,
            "gpiib_iiia" => compartment.gpiib_iiia_inhibition
        ),

        "function_tests" => Dict(
            "aggregation_response_pct" => compartment.aggregation_response,
            "closure_time_s" => compartment.closure_time
        ),

        "clinical" => Dict(
            "bleeding_risk" => calculate_bleeding_risk(compartment),
            "pathology" => compartment.pathology,
            "pathology_severity" => compartment.pathology_severity
        ),

        "granules" => Dict(
            "adp_available" => compartment.granules.adp,
            "serotonin_available" => compartment.granules.serotonin,
            "factor_v_available" => compartment.granules.factor_v
        )
    )
end

end  # module Platelets
