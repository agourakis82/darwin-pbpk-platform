# ===========================================================================
# RBC DYNAMICS INTEGRATED MODULE
# ===========================================================================
# Closes the loop between RBC production, aging, clearance, and drug PK.
#
# This module integrates:
# - Hematopoiesis (bone marrow RBC production)
# - EPO feedback loop (hypoxia sensing → EPO → erythropoiesis)
# - RBC aging with organ-specific clearance
# - Spleen/liver RBC sequestration and destruction
# - Age-dependent transporter expression for drug PK
# - Iron/bilirubin metabolism from RBC turnover
#
# The key insight: RBC dynamics MUST be modeled as a closed loop:
#   RBC Clearance → ↓Hct → ↑EPO → ↑Reticulocytes → ↑Hct
#
# References:
# - Finch et al. (1970) Regulation of erythropoiesis
# - Palis (2014) Primitive and definitive erythropoiesis
# - Higgins (2015) Red blood cell population dynamics
# - Hunt et al. (2024) Transporter ontogeny
#
# Author: Dr. Demetrios Agourakis
# Date: December 2025
# Version: 1.0.0
# ===========================================================================

module RBCDynamicsIntegrated

using Statistics
using LinearAlgebra

# We'll reference sibling modules via parent when included in DarwinPBPK
# For standalone testing, these won't be available

export RBCDynamicsState, HematopoiesisParams, EPOFeedback, OrganRBCClearance
export IronMetabolism, BilirubinState, RBCTurnoverResult
export initialize_rbc_dynamics, update_rbc_dynamics!, simulate_rbc_dynamics
export calculate_organ_rbc_clearance, calculate_epo_response
export calculate_reticulocyte_release, calculate_age_weighted_transport
export get_effective_hematocrit, get_rbc_mediated_clearance
export apply_rbc_dynamics_to_pk, create_rbc_ode_system
export NORMAL_RBC_PARAMS, DISEASE_RBC_DYNAMICS

# ===========================================================================
# Constants
# ===========================================================================

"""
Normal physiological RBC parameters.
"""
const NORMAL_RBC_PARAMS = Dict{String, Float64}(
    # RBC production
    "rbc_production_rate" => 2.0e11,     # cells/day (normal ~200 billion/day)
    "reticulocyte_maturation_time" => 1.0, # days in circulation before mature
    "bone_marrow_transit_time" => 5.0,   # days from stem cell to reticulocyte release

    # RBC lifespan
    "normal_rbc_lifespan" => 120.0,      # days
    "senescence_threshold" => 100.0,     # days when clearance accelerates
    "deformability_half_life" => 90.0,   # days (exponential decline)

    # EPO dynamics
    "epo_baseline" => 10.0,              # mU/mL (normal serum EPO)
    "epo_half_life" => 5.0,              # hours
    "epo_max_stimulation" => 100.0,      # max fold increase from baseline
    "hct_setpoint" => 0.45,              # target hematocrit
    "epo_sensitivity" => 0.1,            # EPO response per % Hct below setpoint

    # Clearance distribution
    "spleen_clearance_fraction" => 0.90, # 90% of aged RBCs cleared by spleen
    "liver_clearance_fraction" => 0.08,  # 8% by liver Kupffer cells
    "other_clearance_fraction" => 0.02,  # 2% by other RES tissues

    # Iron metabolism
    "iron_per_rbc" => 1.0e-12,           # g Fe per RBC (~1 pg)
    "iron_recycling_efficiency" => 0.95, # 95% of iron recycled
    "daily_iron_loss" => 1.0e-3,         # g/day (1 mg basal loss)

    # Bilirubin
    "bilirubin_per_rbc" => 3.4e-11,      # g bilirubin per RBC
    "bilirubin_conjugation_rate" => 0.5, # fraction/hour (hepatic UGT)
    "normal_total_bilirubin" => 1.0,     # mg/dL
)

"""
Disease-specific RBC dynamics parameters.
"""
const DISEASE_RBC_DYNAMICS = Dict{Symbol, Dict{String, Float64}}(
    :normal => Dict(
        "rbc_lifespan" => 120.0,
        "production_rate_factor" => 1.0,
        "clearance_rate_factor" => 1.0,
        "epo_response_factor" => 1.0
    ),
    :sickle_cell => Dict(
        "rbc_lifespan" => 17.0,           # Drastically shortened
        "production_rate_factor" => 3.0,   # Compensatory increase
        "clearance_rate_factor" => 7.0,    # ~7x normal clearance
        "epo_response_factor" => 0.8,      # Slightly blunted
        "splenic_sequestration" => 0.4     # 40% of RBCs sequestered
    ),
    :hemolytic_anemia => Dict(
        "rbc_lifespan" => 30.0,
        "production_rate_factor" => 2.5,
        "clearance_rate_factor" => 4.0,
        "epo_response_factor" => 1.2       # Enhanced EPO response
    ),
    :thalassemia_major => Dict(
        "rbc_lifespan" => 45.0,
        "production_rate_factor" => 1.5,   # Ineffective erythropoiesis
        "clearance_rate_factor" => 2.5,
        "epo_response_factor" => 0.6,      # Blunted due to iron overload
        "ineffective_erythropoiesis" => 0.5  # 50% die in marrow
    ),
    :aplastic_anemia => Dict(
        "rbc_lifespan" => 120.0,           # Normal lifespan
        "production_rate_factor" => 0.1,   # Severely reduced production
        "clearance_rate_factor" => 1.0,
        "epo_response_factor" => 0.1       # No response (no stem cells)
    ),
    :ckd_anemia => Dict(
        "rbc_lifespan" => 80.0,            # Slightly shortened
        "production_rate_factor" => 0.5,   # Reduced EPO from kidneys
        "clearance_rate_factor" => 1.5,    # Uremic toxins
        "epo_response_factor" => 0.3       # Poor EPO production
    ),
    :polycythemia_vera => Dict(
        "rbc_lifespan" => 120.0,
        "production_rate_factor" => 2.0,   # Autonomous production
        "clearance_rate_factor" => 1.0,
        "epo_response_factor" => 0.0       # EPO-independent (JAK2 mutation)
    ),
    :spherocytosis => Dict(
        "rbc_lifespan" => 20.0,            # Membrane defect
        "production_rate_factor" => 2.5,
        "clearance_rate_factor" => 6.0,    # Splenic trapping
        "epo_response_factor" => 1.0
    ),
    :g6pd_deficiency => Dict(
        "rbc_lifespan" => 60.0,            # Baseline; shorter during crisis
        "production_rate_factor" => 1.5,
        "clearance_rate_factor" => 2.0,
        "epo_response_factor" => 1.0,
        "oxidative_stress_sensitivity" => 5.0  # Fold increase in clearance during stress
    )
)

# ===========================================================================
# Data Structures
# ===========================================================================

"""
EPO feedback system state.
"""
mutable struct EPOFeedback
    epo_level::Float64           # Current EPO level (mU/mL)
    epo_production_rate::Float64 # Kidney EPO production (mU/mL/h)
    hct_setpoint::Float64        # Target hematocrit
    sensitivity::Float64         # Response sensitivity
    max_stimulation::Float64     # Maximum fold increase
    is_exogenous::Bool          # Whether on EPO therapy
    exogenous_dose::Float64      # EPO dose if on therapy (IU/week)
end

"""
Hematopoiesis (bone marrow) parameters.
"""
mutable struct HematopoiesisParams
    stem_cell_pool::Float64      # Relative stem cell capacity (1.0 = normal)
    erythroid_fraction::Float64  # Fraction of marrow devoted to erythropoiesis
    maturation_time::Float64     # Days from CFU-E to reticulocyte
    release_rate::Float64        # Reticulocytes released per day
    ineffective_fraction::Float64 # Fraction dying in marrow (thalassemia)
    epo_responsiveness::Float64  # Response to EPO (0-1, disease-dependent)
end

"""
Iron metabolism state.
"""
mutable struct IronMetabolism
    serum_iron::Float64          # μg/dL
    ferritin::Float64            # ng/mL (storage)
    transferrin_saturation::Float64  # % (0-1)
    total_body_iron::Float64     # g
    recycled_iron_rate::Float64  # g/day from RBC destruction
    absorption_rate::Float64     # g/day from GI
    loss_rate::Float64           # g/day (basal + pathological)
end

"""
Bilirubin metabolism state.
"""
mutable struct BilirubinState
    unconjugated::Float64        # mg/dL (indirect)
    conjugated::Float64          # mg/dL (direct)
    total::Float64               # mg/dL
    production_rate::Float64     # mg/day from RBC destruction
    conjugation_rate::Float64    # mg/day (hepatic capacity)
    excretion_rate::Float64      # mg/day (biliary)
end

"""
Organ-specific RBC clearance.
"""
struct OrganRBCClearance
    spleen_clearance::Float64    # cells/day
    liver_clearance::Float64     # cells/day
    other_clearance::Float64     # cells/day
    total_clearance::Float64     # cells/day
    clearance_by_age::Vector{Float64}  # Clearance rate per age bin
end

"""
RBC age distribution with 120 daily bins.
"""
mutable struct RBCAgeDistribution
    counts::Vector{Float64}      # RBC count in each age bin (0-119 days)
    total_count::Float64         # Total RBC count
    mean_age::Float64            # Mean RBC age (days)
    reticulocyte_count::Float64  # Count of cells < 1 day old
end

"""
Complete RBC dynamics state.
"""
mutable struct RBCDynamicsState
    # Time
    time::Float64                # Current time (days)

    # RBC population
    age_distribution::RBCAgeDistribution
    hematocrit::Float64          # Current Hct (0-1)
    hemoglobin::Float64          # g/dL
    rbc_count::Float64           # 10^12/L
    mcv::Float64                 # fL (mean cell volume)

    # Production
    hematopoiesis::HematopoiesisParams
    daily_production::Float64    # cells/day actually produced

    # Feedback
    epo::EPOFeedback

    # Clearance
    organ_clearance::OrganRBCClearance

    # Metabolism
    iron::IronMetabolism
    bilirubin::BilirubinState

    # Disease state
    disease::Symbol
    disease_params::Dict{String, Float64}

    # Transporters (age-weighted)
    effective_band3::Float64     # Age-weighted Band3 expression
    effective_glut1::Float64     # Age-weighted GLUT1
    effective_ent1::Float64      # Age-weighted ENT1
end

"""
Result of RBC turnover simulation step.
"""
struct RBCTurnoverResult
    new_hematocrit::Float64
    new_rbc_count::Float64
    reticulocyte_fraction::Float64
    cells_produced::Float64
    cells_cleared::Float64
    net_change::Float64
    epo_level::Float64
    bilirubin_total::Float64
    mean_rbc_age::Float64
end

# ===========================================================================
# Initialization Functions
# ===========================================================================

"""
    initialize_rbc_dynamics(; disease=:normal, hematocrit=0.45) -> RBCDynamicsState

Initialize a complete RBC dynamics state.
"""
function initialize_rbc_dynamics(;
    disease::Symbol = :normal,
    hematocrit::Float64 = 0.45,
    hemoglobin::Float64 = 15.0,
    on_epo_therapy::Bool = false,
    epo_dose::Float64 = 0.0
)::RBCDynamicsState

    # Get disease parameters
    disease_params = get(DISEASE_RBC_DYNAMICS, disease, DISEASE_RBC_DYNAMICS[:normal])
    lifespan = disease_params["rbc_lifespan"]

    # Initialize age distribution (steady state)
    # At steady state, equal numbers in each age bin (rectangular distribution)
    n_bins = Int(ceil(lifespan))
    counts = fill(1.0 / n_bins, 120)  # Normalized

    # Scale to actual RBC count (5 × 10^12 cells/L × 5L blood = 2.5 × 10^13 total)
    total_rbc = 2.5e13 * (hematocrit / 0.45)  # Scale by Hct
    counts = counts .* total_rbc

    # Zero out bins beyond lifespan
    for i in (n_bins+1):120
        counts[i] = 0.0
    end

    age_dist = RBCAgeDistribution(
        counts,
        sum(counts),
        mean(1:120, weights(counts ./ sum(counts))),
        counts[1]  # Reticulocytes = day 0
    )

    # Hematopoiesis
    base_production = NORMAL_RBC_PARAMS["rbc_production_rate"]
    hematopoiesis = HematopoiesisParams(
        1.0,                                    # stem_cell_pool
        0.25,                                   # erythroid_fraction (25% of marrow)
        NORMAL_RBC_PARAMS["bone_marrow_transit_time"],
        base_production * disease_params["production_rate_factor"],
        get(disease_params, "ineffective_erythropoiesis", 0.0),
        disease_params["epo_response_factor"]
    )

    # EPO feedback
    epo = EPOFeedback(
        NORMAL_RBC_PARAMS["epo_baseline"],
        NORMAL_RBC_PARAMS["epo_baseline"] / 24.0,  # Steady state production
        NORMAL_RBC_PARAMS["hct_setpoint"],
        NORMAL_RBC_PARAMS["epo_sensitivity"],
        NORMAL_RBC_PARAMS["epo_max_stimulation"],
        on_epo_therapy,
        epo_dose
    )

    # Organ clearance (steady state = production)
    daily_clearance = hematopoiesis.release_rate
    organ_clearance = OrganRBCClearance(
        daily_clearance * NORMAL_RBC_PARAMS["spleen_clearance_fraction"],
        daily_clearance * NORMAL_RBC_PARAMS["liver_clearance_fraction"],
        daily_clearance * NORMAL_RBC_PARAMS["other_clearance_fraction"],
        daily_clearance,
        calculate_age_clearance_profile(lifespan)
    )

    # Iron metabolism
    iron = IronMetabolism(
        100.0,      # serum_iron μg/dL
        100.0,      # ferritin ng/mL
        0.30,       # transferrin_saturation
        4.0,        # total_body_iron g
        daily_clearance * NORMAL_RBC_PARAMS["iron_per_rbc"] *
            NORMAL_RBC_PARAMS["iron_recycling_efficiency"],
        NORMAL_RBC_PARAMS["daily_iron_loss"],
        NORMAL_RBC_PARAMS["daily_iron_loss"]
    )

    # Bilirubin
    bili_production = daily_clearance * NORMAL_RBC_PARAMS["bilirubin_per_rbc"] * 1e3  # mg/day
    bilirubin = BilirubinState(
        0.7,        # unconjugated mg/dL
        0.3,        # conjugated mg/dL
        1.0,        # total mg/dL
        bili_production,
        bili_production,  # At steady state
        bili_production
    )

    # Age-weighted transporter expression
    effective_band3 = calculate_age_weighted_transporter(age_dist, :band3)
    effective_glut1 = calculate_age_weighted_transporter(age_dist, :glut1)
    effective_ent1 = calculate_age_weighted_transporter(age_dist, :ent1)

    return RBCDynamicsState(
        0.0,                    # time
        age_dist,
        hematocrit,
        hemoglobin,
        hematocrit * 5.0 / 0.45 * 1e12,  # rbc_count (scaled)
        90.0,                   # mcv
        hematopoiesis,
        hematopoiesis.release_rate,
        epo,
        organ_clearance,
        iron,
        bilirubin,
        disease,
        disease_params,
        effective_band3,
        effective_glut1,
        effective_ent1
    )
end

"""
Calculate age-dependent clearance profile.
Clearance accelerates as RBCs age past senescence threshold.
For short-lifespan diseases (e.g., sickle cell), the profile is scaled accordingly.
"""
function calculate_age_clearance_profile(lifespan::Float64)::Vector{Float64}
    profile = zeros(120)

    # Adjust senescence threshold based on lifespan
    # For normal RBCs: senescence at ~83% of lifespan (100/120)
    # Apply same proportion for disease states
    senescence = lifespan * 0.83

    for age in 1:120
        if age < senescence
            # Minimal clearance of young RBCs (fraction per day)
            profile[age] = 0.001
        elseif age < lifespan
            # Smoothly increasing clearance from senescence to lifespan
            # Use sigmoid rather than exponential to avoid overflow
            fraction = (age - senescence) / max(1.0, lifespan - senescence)
            profile[age] = 0.01 + 0.99 * fraction^2  # Quadratic ramp to ~1.0
        else
            # Complete clearance past lifespan
            profile[age] = 1.0
        end
    end

    return profile
end

# ===========================================================================
# EPO Feedback Functions
# ===========================================================================

"""
    calculate_epo_response(hematocrit, epo_state) -> Float64

Calculate EPO level based on hematocrit deviation from setpoint.
Hypoxia-inducible factor (HIF) pathway simulation.
"""
function calculate_epo_response(
    hematocrit::Float64,
    epo::EPOFeedback
)::Float64

    if epo.is_exogenous
        # On EPO therapy - add exogenous contribution
        # Typical EPO: 10,000 IU/week ≈ 1400 IU/day
        exogenous_contribution = epo.exogenous_dose / 7.0 / 100.0  # Scale to mU/mL
        return epo.epo_level + exogenous_contribution
    end

    # Endogenous EPO response
    hct_deficit = epo.hct_setpoint - hematocrit

    if hct_deficit <= 0
        # Above setpoint - suppress EPO
        suppression = exp(-5.0 * abs(hct_deficit))
        return epo.epo_level * suppression
    else
        # Below setpoint - increase EPO (sigmoid response)
        # HIF stabilization increases exponentially with hypoxia
        stimulation = 1.0 + epo.max_stimulation * (1.0 - exp(-epo.sensitivity * hct_deficit * 100))
        return epo.epo_level * stimulation
    end
end

"""
    calculate_reticulocyte_release(epo_level, hematopoiesis) -> Float64

Calculate reticulocyte release rate based on EPO level.
"""
function calculate_reticulocyte_release(
    epo_level::Float64,
    hematopoiesis::HematopoiesisParams
)::Float64

    baseline_epo = NORMAL_RBC_PARAMS["epo_baseline"]
    baseline_production = NORMAL_RBC_PARAMS["rbc_production_rate"]

    # EPO dose-response (Michaelis-Menten like)
    # EC50 ≈ 30 mU/mL for maximal erythropoiesis
    ec50 = 30.0
    epo_effect = epo_level / (epo_level + ec50)

    # Maximum stimulation is ~3-5x baseline
    max_fold = 4.0

    # Calculate production
    production = baseline_production * (1.0 + (max_fold - 1.0) * epo_effect)

    # Apply stem cell pool limitation
    production *= hematopoiesis.stem_cell_pool

    # Apply EPO responsiveness (disease-dependent)
    production *= hematopoiesis.epo_responsiveness

    # Account for ineffective erythropoiesis
    production *= (1.0 - hematopoiesis.ineffective_fraction)

    return production
end

# ===========================================================================
# Organ Clearance Functions
# ===========================================================================

"""
    calculate_organ_rbc_clearance(state) -> OrganRBCClearance

Calculate RBC clearance by spleen, liver, and other tissues.
The clearance_by_age represents the FRACTION of cells in each age bin cleared per day.
"""
function calculate_organ_rbc_clearance(
    state::RBCDynamicsState
)::OrganRBCClearance

    age_dist = state.age_distribution
    clearance_profile = state.organ_clearance.clearance_by_age
    disease_factor = state.disease_params["clearance_rate_factor"]

    # clearance_profile is the base fraction cleared per day for each age
    # disease_factor scales this (e.g., 7x for sickle cell)
    # But we cap it to prevent >100% clearance per day
    clearance_by_age = zeros(120)
    for age in 1:120
        # Clearance fraction per day, capped at 1.0 (100%)
        frac = min(1.0, clearance_profile[age] * disease_factor)
        # Actual cells cleared from this age bin
        clearance_by_age[age] = age_dist.counts[age] * frac
    end

    total_clearance = sum(clearance_by_age)

    # Distribute to organs
    spleen_frac = NORMAL_RBC_PARAMS["spleen_clearance_fraction"]
    liver_frac = NORMAL_RBC_PARAMS["liver_clearance_fraction"]
    other_frac = NORMAL_RBC_PARAMS["other_clearance_fraction"]

    # Disease-specific modifications
    if state.disease == :spherocytosis || state.disease == :sickle_cell
        # Increased splenic trapping
        spleen_frac = min(0.98, spleen_frac * 1.5)
        liver_frac = 1.0 - spleen_frac - 0.02
    elseif state.disease == :ckd_anemia
        # More liver/other clearance (uremic toxins)
        liver_frac *= 1.5
        spleen_frac = 1.0 - liver_frac - 0.05
    end

    return OrganRBCClearance(
        total_clearance * spleen_frac,
        total_clearance * liver_frac,
        total_clearance * other_frac,
        total_clearance,
        clearance_by_age
    )
end

# ===========================================================================
# Transporter Integration
# ===========================================================================

"""
Calculate age-weighted transporter expression.
Young RBCs (reticulocytes) have higher expression than aged cells.
"""
function calculate_age_weighted_transporter(
    age_dist::RBCAgeDistribution,
    transporter::Symbol
)::Float64

    # Age-dependent expression profiles (relative to young RBC)
    # Data from Bosman et al. (2008), D'Alessandro et al. (2013)

    function expression_at_age(age::Int, trans::Symbol)::Float64
        if trans == :band3
            # Band3 decreases ~40% over lifespan
            return 1.0 - 0.004 * age
        elseif trans == :glut1
            # GLUT1 decreases ~50% over lifespan
            return max(0.5, 1.0 - 0.005 * age)
        elseif trans == :ent1
            # ENT1 decreases ~60% over lifespan
            return max(0.4, 1.0 - 0.006 * age)
        else
            return 1.0
        end
    end

    # Weight by RBC count in each age bin
    total = 0.0
    weighted_sum = 0.0

    for age in 1:120
        count = age_dist.counts[age]
        expr = expression_at_age(age, transporter)
        weighted_sum += count * expr
        total += count
    end

    return total > 0 ? weighted_sum / total : 1.0
end

"""
    calculate_age_weighted_transport(state, drug_ke_p) -> Float64

Calculate effective drug RBC partitioning accounting for RBC age distribution.
"""
function calculate_age_weighted_transport(
    state::RBCDynamicsState,
    base_ke_p::Float64;
    transporter_dependent::Symbol = :none
)::Float64

    if transporter_dependent == :none
        # Passive diffusion - not age-dependent
        return base_ke_p
    elseif transporter_dependent == :band3
        return base_ke_p * state.effective_band3
    elseif transporter_dependent == :glut1
        return base_ke_p * state.effective_glut1
    elseif transporter_dependent == :ent1
        return base_ke_p * state.effective_ent1
    else
        return base_ke_p
    end
end

# ===========================================================================
# Dynamic Simulation
# ===========================================================================

"""
    update_rbc_dynamics!(state, dt) -> RBCTurnoverResult

Update RBC dynamics state for time step dt (in days).
This is the core simulation function that closes the feedback loop.

Uses a simplified lifespan-based clearance model for numerical stability.
"""
function update_rbc_dynamics!(
    state::RBCDynamicsState,
    dt::Float64
)::RBCTurnoverResult

    # 1. Calculate current EPO level based on Hct
    new_epo = calculate_epo_response(state.hematocrit, state.epo)
    # Cap EPO at physiological maximum (~1000 mU/mL in severe anemia)
    new_epo = min(new_epo, 1000.0)
    state.epo.epo_level = new_epo

    # 2. Calculate reticulocyte release based on EPO
    reticulocyte_release = calculate_reticulocyte_release(new_epo, state.hematopoiesis)
    cells_produced = reticulocyte_release * dt

    # 3. Calculate clearance based on mean lifespan (stable model)
    # At steady state: clearance rate = total cells / lifespan
    lifespan = state.disease_params["rbc_lifespan"]

    # Daily clearance = total / lifespan (first-order kinetics)
    daily_clearance_rate = state.age_distribution.total_count / lifespan
    cells_cleared = daily_clearance_rate * dt

    # Safety: max 20% clearance per day to prevent numerical instability
    max_clearance = state.age_distribution.total_count * 0.20 * dt
    cells_cleared = min(cells_cleared, max_clearance)

    # 4. Apply clearance proportionally across age bins
    if state.age_distribution.total_count > 0 && cells_cleared > 0
        reduction_fraction = cells_cleared / state.age_distribution.total_count
        for age in 1:120
            state.age_distribution.counts[age] *= (1.0 - reduction_fraction)
        end
    end

    # 5. Age the RBC population (shift distribution)
    age_rbc_population!(state.age_distribution, dt)

    # 6. Add new reticulocytes
    state.age_distribution.counts[1] += cells_produced

    # Update organ clearance record for tracking
    clearance = OrganRBCClearance(
        cells_cleared * NORMAL_RBC_PARAMS["spleen_clearance_fraction"],
        cells_cleared * NORMAL_RBC_PARAMS["liver_clearance_fraction"],
        cells_cleared * NORMAL_RBC_PARAMS["other_clearance_fraction"],
        cells_cleared,
        state.organ_clearance.clearance_by_age
    )

    # 6. Update totals
    state.age_distribution.total_count = sum(state.age_distribution.counts)
    state.age_distribution.reticulocyte_count = state.age_distribution.counts[1]

    # Calculate mean age
    if state.age_distribution.total_count > 0
        state.age_distribution.mean_age = sum((1:120) .* state.age_distribution.counts) /
                                          state.age_distribution.total_count
    end

    # 7. Update hematocrit
    # Hct = RBC volume / blood volume
    # Assume 5L blood, 90 fL per RBC
    blood_volume = 5.0  # L
    rbc_volume = state.age_distribution.total_count * state.mcv * 1e-15  # L
    new_hct = rbc_volume / blood_volume
    new_hct = clamp(new_hct, 0.10, 0.70)  # Physiological limits

    state.hematocrit = new_hct
    state.rbc_count = state.age_distribution.total_count / 1e12 / blood_volume
    state.hemoglobin = new_hct * 33.0  # Approximate: Hb ≈ Hct × 33 g/dL

    # 8. Update iron metabolism
    iron_released = cells_cleared * NORMAL_RBC_PARAMS["iron_per_rbc"]
    state.iron.recycled_iron_rate = iron_released * NORMAL_RBC_PARAMS["iron_recycling_efficiency"]

    # 9. Update bilirubin
    bilirubin_produced = cells_cleared * NORMAL_RBC_PARAMS["bilirubin_per_rbc"] * 1e3 * dt  # mg
    state.bilirubin.production_rate = bilirubin_produced / dt

    # Simple bilirubin dynamics: production - conjugation - excretion
    bili_change = (state.bilirubin.production_rate - state.bilirubin.conjugation_rate) * dt / 5000  # Scale to mg/dL
    state.bilirubin.unconjugated += bili_change
    state.bilirubin.unconjugated = max(0.1, state.bilirubin.unconjugated)
    state.bilirubin.total = state.bilirubin.unconjugated + state.bilirubin.conjugated

    # 10. Update transporter expression
    state.effective_band3 = calculate_age_weighted_transporter(state.age_distribution, :band3)
    state.effective_glut1 = calculate_age_weighted_transporter(state.age_distribution, :glut1)
    state.effective_ent1 = calculate_age_weighted_transporter(state.age_distribution, :ent1)

    # 11. Update organ clearance for next step
    state.organ_clearance = clearance

    # 12. Advance time
    state.time += dt

    # Return result
    retic_fraction = state.age_distribution.reticulocyte_count /
                     max(1.0, state.age_distribution.total_count)

    return RBCTurnoverResult(
        new_hct,
        state.rbc_count,
        retic_fraction,
        cells_produced,
        cells_cleared,
        cells_produced - cells_cleared,
        new_epo,
        state.bilirubin.total,
        state.age_distribution.mean_age
    )
end

"""
Age the RBC population by shifting the distribution.
"""
function age_rbc_population!(age_dist::RBCAgeDistribution, dt::Float64)
    # For simplicity, if dt ≈ 1 day, shift by one bin
    # For fractional days, interpolate

    if dt >= 1.0
        # Full day(s) - shift bins
        n_days = Int(floor(dt))
        for _ in 1:n_days
            # Shift right (older)
            for age in 120:-1:2
                age_dist.counts[age] = age_dist.counts[age-1]
            end
            age_dist.counts[1] = 0.0  # New reticulocytes added separately
        end
    end

    # Handle fractional day (approximate by proportional aging)
    frac = dt - floor(dt)
    if frac > 0.01
        # Partially shift based on fraction
        new_counts = copy(age_dist.counts)
        for age in 2:120
            new_counts[age] = (1-frac) * age_dist.counts[age] + frac * age_dist.counts[age-1]
        end
        new_counts[1] = (1-frac) * age_dist.counts[1]
        age_dist.counts = new_counts
    end
end

"""
    simulate_rbc_dynamics(state, duration; dt=1.0) -> Vector{RBCTurnoverResult}

Simulate RBC dynamics over a duration (days).
"""
function simulate_rbc_dynamics(
    state::RBCDynamicsState,
    duration::Float64;
    dt::Float64 = 1.0
)::Vector{RBCTurnoverResult}

    results = RBCTurnoverResult[]
    t = 0.0

    while t < duration
        result = update_rbc_dynamics!(state, dt)
        push!(results, result)
        t += dt
    end

    return results
end

# ===========================================================================
# PK Integration Functions
# ===========================================================================

"""
    apply_rbc_dynamics_to_pk(state, pk_params) -> Dict

Apply RBC dynamics state to PK parameters.
Returns adjusted clearance, volume, and blood:plasma ratio.
"""
function apply_rbc_dynamics_to_pk(
    state::RBCDynamicsState,
    cl_base::Float64,
    vd_base::Float64,
    ke_p::Float64;
    is_hepatically_cleared::Bool = true,
    transporter::Symbol = :none
)::Dict{String, Float64}

    # 1. Calculate effective Ke_p based on RBC age
    effective_ke_p = calculate_age_weighted_transport(state, ke_p; transporter_dependent=transporter)

    # 2. Calculate blood:plasma ratio
    rb = 1.0 - state.hematocrit + state.hematocrit * effective_ke_p

    # 3. Adjust clearance for hematocrit (viscosity effect)
    # Higher Hct → higher viscosity → lower hepatic blood flow → lower CL for high-extraction drugs
    hct_normal = 0.45
    viscosity_factor = exp(2.5 * (state.hematocrit - hct_normal))  # Exponential model

    cl_adjusted = cl_base
    if is_hepatically_cleared
        # Well-stirred model: CL = Q × E, where Q affected by viscosity
        cl_adjusted = cl_base / viscosity_factor
    end

    # 4. Adjust Vd for Rb
    # Vd = Vp + Vt × (fu/fut) + Vrbc × Ke_p
    # Simplified: Vd scales with Rb
    vd_adjusted = vd_base * rb

    # 5. Check for bilirubin displacement
    # High bilirubin can displace drugs from albumin
    if state.bilirubin.unconjugated > 5.0  # Significant hyperbilirubinemia
        fu_increase = 1.0 + 0.1 * (state.bilirubin.unconjugated - 5.0)
        # This would affect protein binding - simplified here
        cl_adjusted *= fu_increase  # Increased fu → increased CL
        vd_adjusted *= fu_increase  # Increased fu → increased Vd
    end

    return Dict(
        "cl_adjusted" => cl_adjusted,
        "vd_adjusted" => vd_adjusted,
        "rb" => rb,
        "effective_ke_p" => effective_ke_p,
        "viscosity_factor" => viscosity_factor,
        "hematocrit" => state.hematocrit,
        "reticulocyte_fraction" => state.age_distribution.reticulocyte_count /
                                   max(1.0, state.age_distribution.total_count),
        "mean_rbc_age" => state.age_distribution.mean_age,
        "bilirubin" => state.bilirubin.total
    )
end

"""
    get_effective_hematocrit(state) -> Float64

Get current effective hematocrit from RBC dynamics state.
"""
function get_effective_hematocrit(state::RBCDynamicsState)::Float64
    return state.hematocrit
end

"""
    get_rbc_mediated_clearance(state) -> Dict

Get organ-specific RBC-mediated clearance contributions.
This is the clearance of the RBCs themselves, which affects drugs bound to RBCs.
"""
function get_rbc_mediated_clearance(state::RBCDynamicsState)::Dict{String, Float64}
    return Dict(
        "spleen_rbc_clearance" => state.organ_clearance.spleen_clearance,
        "liver_rbc_clearance" => state.organ_clearance.liver_clearance,
        "other_rbc_clearance" => state.organ_clearance.other_clearance,
        "total_rbc_clearance" => state.organ_clearance.total_clearance,
        "rbc_turnover_days" => state.age_distribution.total_count /
                               max(1.0, state.organ_clearance.total_clearance)
    )
end

# ===========================================================================
# ODE System Generator
# ===========================================================================

"""
    create_rbc_ode_system(disease) -> Function

Create an ODE function for RBC dynamics that can be integrated with DifferentialEquations.jl.

State vector: [RBC_count, Hematocrit, EPO_level, Reticulocyte_count, Bilirubin]
"""
function create_rbc_ode_system(
    disease::Symbol = :normal
)::Function

    params = get(DISEASE_RBC_DYNAMICS, disease, DISEASE_RBC_DYNAMICS[:normal])
    lifespan = params["rbc_lifespan"]
    prod_factor = params["production_rate_factor"]
    clear_factor = params["clearance_rate_factor"]
    epo_response = params["epo_response_factor"]

    function rbc_ode!(du, u, p, t)
        # u = [RBC_count (×10^12), Hct, EPO (mU/mL), Retic_fraction, Bilirubin (mg/dL)]
        rbc = u[1]
        hct = u[2]
        epo = u[3]
        retic = u[4]
        bili = u[5]

        # EPO response to anemia
        hct_setpoint = 0.45
        epo_baseline = 10.0
        epo_production = epo_baseline * (1.0 + 10.0 * max(0, hct_setpoint - hct)) * epo_response
        epo_clearance = epo / 5.0 * 24  # t1/2 = 5 hours

        # RBC production (driven by EPO)
        base_production = 0.2  # ×10^12/day at baseline
        epo_effect = epo / (epo + 30.0)  # Michaelis-Menten
        production = base_production * (1.0 + 3.0 * epo_effect) * prod_factor

        # RBC clearance (age-dependent, approximated)
        clearance_rate = rbc / lifespan * clear_factor

        # Reticulocyte dynamics
        retic_production = production / rbc  # As fraction
        retic_maturation = retic / 1.0  # 1 day maturation

        # Bilirubin from RBC destruction
        bili_production = clearance_rate * 0.034  # ~34 mg bilirubin per 10^12 RBCs
        bili_conjugation = bili * 0.5  # Hepatic conjugation rate

        # ODEs
        du[1] = production - clearance_rate  # dRBC/dt
        du[2] = (rbc * 90e-15 / 5.0 - hct) / 1.0  # dHct/dt (equilibration)
        du[3] = epo_production - epo_clearance  # dEPO/dt
        du[4] = retic_production - retic_maturation  # dRetic/dt
        du[5] = bili_production - bili_conjugation  # dBili/dt
    end

    return rbc_ode!
end

# ===========================================================================
# Utility: Weighted Mean
# ===========================================================================

"""
Weighted mean calculation helper.
"""
function mean(values, weights::Base.Generator)
    w = collect(weights)
    return sum(values .* w) / sum(w)
end

function weights(v::Vector{Float64})
    return (x for x in v)
end

end # module
