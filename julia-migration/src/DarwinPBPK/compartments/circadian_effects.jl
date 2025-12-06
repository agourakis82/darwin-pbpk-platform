"""
Circadian Rhythm Effects Module

Models time-of-day-dependent changes in blood parameters and
drug disposition (chronopharmacokinetics).

Key Features:
- Albumin/AAG diurnal variation
- Cortisol rhythm effects on GR
- Leukocyte trafficking rhythms
- Platelet aggregability cycles
- Hepatic blood flow variation

Clinical Relevance:
- Explains inter-occasion variability
- Important for chronotherapy optimization
- Affects anticoagulant monitoring
- Relevant for shift workers/ICU patients

References:
- Lemmer B (2006) Chronopharmacology
- Levi F (2010) Chronotherapy of cancer
- Dallmann R (2014) Circadian rhythms and pharmacology

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module CircadianEffects

using Statistics

export CircadianState, CircadianParameter
export create_default_parameters, get_circadian_factor, simulate_circadian_variation
export calculate_optimal_dosing_time, get_chronotype_adjustment
export calculate_circadian_pk_effect, calculate_chronotherapy_benefit
export CIRCADIAN_PARAMETERS, CHRONOTYPE_SHIFTS

# ============================================================================
# CONSTANTS
# ============================================================================

# Reference time (hours from midnight)
const CORTISOL_PEAK = 8.0          # 8 AM
const CORTISOL_NADIR = 0.0         # Midnight
const MELATONIN_PEAK = 3.0         # 3 AM
const BODY_TEMP_PEAK = 18.0        # 6 PM
const BODY_TEMP_NADIR = 4.0        # 4 AM

# Amplitude of variation (as fraction of mean)
const ALBUMIN_AMPLITUDE = 0.05     # ±5% variation
const AAG_AMPLITUDE = 0.10         # ±10% variation (higher)
const CORTISOL_AMPLITUDE = 0.40    # ±40% (large variation)
const HEPATIC_FLOW_AMPLITUDE = 0.15 # ±15%
const GFR_AMPLITUDE = 0.10         # ±10%

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    CircadianParameter

Single circadian-varying parameter.

# Fields
- `name::Symbol`: Parameter name
- `baseline::Float64`: 24-hour mean value
- `amplitude::Float64`: Fractional amplitude (0-1)
- `acrophase::Float64`: Time of peak (hours from midnight)
- `period::Float64`: Period (hours, typically 24)
"""
struct CircadianParameter
    name::Symbol
    baseline::Float64
    amplitude::Float64
    acrophase::Float64
    period::Float64

    function CircadianParameter(name::Symbol;
        baseline = 1.0,
        amplitude = 0.1,
        acrophase = 12.0,
        period = 24.0
    )
        new(name, baseline, amplitude, acrophase, period)
    end
end

"""
    CircadianState

Complete circadian state for a subject.

# Fields
- `current_time::Float64`: Time of day (hours from midnight)
- `chronotype::Symbol`: :morning, :intermediate, :evening
- `parameters::Dict{Symbol, CircadianParameter}`: All parameters
- `shift_worker::Bool`: Disrupted rhythm
- `jet_lag_hours::Float64`: Phase shift (hours)
"""
mutable struct CircadianState
    current_time::Float64
    chronotype::Symbol
    parameters::Dict{Symbol, CircadianParameter}
    shift_worker::Bool
    jet_lag_hours::Float64

    function CircadianState(;
        current_time = 8.0,
        chronotype = :intermediate,
        shift_worker = false,
        jet_lag_hours = 0.0
    )
        params = create_default_parameters()
        new(current_time, chronotype, params, shift_worker, jet_lag_hours)
    end
end

# ============================================================================
# CIRCADIAN PARAMETERS DATABASE
# ============================================================================

"""
Default circadian parameters for blood and PK.
"""
const CIRCADIAN_PARAMETERS = Dict{Symbol, Dict{Symbol, Any}}(
    # Plasma proteins
    :albumin => Dict(
        :baseline => 40.0,      # g/L
        :amplitude => 0.05,
        :acrophase => 16.0,     # Peak 4 PM
        :unit => "g/L"
    ),

    :aag => Dict(
        :baseline => 0.8,       # g/L
        :amplitude => 0.10,
        :acrophase => 8.0,      # Peak 8 AM
        :unit => "g/L"
    ),

    # Hormones
    :cortisol => Dict(
        :baseline => 400.0,     # nmol/L
        :amplitude => 0.50,
        :acrophase => 8.0,      # Peak 8 AM
        :unit => "nmol/L"
    ),

    :melatonin => Dict(
        :baseline => 50.0,      # pg/mL
        :amplitude => 0.80,
        :acrophase => 3.0,      # Peak 3 AM
        :unit => "pg/mL"
    ),

    :growth_hormone => Dict(
        :baseline => 2.0,       # ng/mL
        :amplitude => 0.70,
        :acrophase => 1.0,      # Peak 1 AM (during sleep)
        :unit => "ng/mL"
    ),

    # Blood cells
    :neutrophils => Dict(
        :baseline => 4.0e9,     # cells/L
        :amplitude => 0.30,
        :acrophase => 20.0,     # Peak 8 PM
        :unit => "cells/L"
    ),

    :lymphocytes => Dict(
        :baseline => 2.0e9,
        :amplitude => 0.25,
        :acrophase => 2.0,      # Peak 2 AM
        :unit => "cells/L"
    ),

    :platelets => Dict(
        :baseline => 250.0e9,
        :amplitude => 0.08,
        :acrophase => 14.0,     # Peak 2 PM
        :unit => "cells/L"
    ),

    # Platelet function
    :platelet_aggregability => Dict(
        :baseline => 1.0,       # Relative
        :amplitude => 0.20,
        :acrophase => 9.0,      # Peak 9 AM (MI risk)
        :unit => "relative"
    ),

    # Coagulation
    :fibrinogen => Dict(
        :baseline => 3.0,       # g/L
        :amplitude => 0.08,
        :acrophase => 10.0,
        :unit => "g/L"
    ),

    :pai1 => Dict(              # Plasminogen activator inhibitor
        :baseline => 20.0,      # ng/mL
        :amplitude => 0.50,
        :acrophase => 6.0,      # Peak 6 AM
        :unit => "ng/mL"
    ),

    # Hemodynamics
    :hepatic_blood_flow => Dict(
        :baseline => 1500.0,    # mL/min
        :amplitude => 0.15,
        :acrophase => 14.0,     # Peak afternoon
        :unit => "mL/min"
    ),

    :renal_blood_flow => Dict(
        :baseline => 1200.0,
        :amplitude => 0.10,
        :acrophase => 12.0,
        :unit => "mL/min"
    ),

    :gfr => Dict(
        :baseline => 100.0,     # mL/min
        :amplitude => 0.10,
        :acrophase => 14.0,
        :unit => "mL/min"
    ),

    :blood_pressure_systolic => Dict(
        :baseline => 120.0,     # mmHg
        :amplitude => 0.10,
        :acrophase => 10.0,
        :unit => "mmHg"
    ),

    :heart_rate => Dict(
        :baseline => 70.0,      # bpm
        :amplitude => 0.12,
        :acrophase => 14.0,
        :unit => "bpm"
    ),

    # Drug metabolism
    :cyp3a4_activity => Dict(
        :baseline => 1.0,
        :amplitude => 0.20,
        :acrophase => 15.0,     # Peak 3 PM
        :unit => "relative"
    ),

    :cyp2d6_activity => Dict(
        :baseline => 1.0,
        :amplitude => 0.10,
        :acrophase => 12.0,
        :unit => "relative"
    ),

    :p_glycoprotein => Dict(
        :baseline => 1.0,
        :amplitude => 0.15,
        :acrophase => 20.0,     # Peak evening
        :unit => "relative"
    ),

    # Gastric
    :gastric_ph => Dict(
        :baseline => 2.0,
        :amplitude => 0.30,
        :acrophase => 23.0,     # Highest at night
        :unit => "pH"
    ),

    :gastric_emptying => Dict(
        :baseline => 1.0,
        :amplitude => 0.15,
        :acrophase => 8.0,      # Fastest in morning
        :unit => "relative"
    )
)

"""
Chronotype phase shifts (hours).
"""
const CHRONOTYPE_SHIFTS = Dict{Symbol, Float64}(
    :morning => -1.5,       # Morning types peak earlier
    :intermediate => 0.0,
    :evening => 2.0         # Evening types peak later
)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

"""
    create_default_parameters()

Create default circadian parameters.
"""
function create_default_parameters()
    params = Dict{Symbol, CircadianParameter}()

    for (name, p) in CIRCADIAN_PARAMETERS
        params[name] = CircadianParameter(name;
            baseline = p[:baseline],
            amplitude = p[:amplitude],
            acrophase = p[:acrophase],
            period = 24.0
        )
    end

    return params
end

"""
    get_circadian_factor(param::CircadianParameter, time::Float64;
                         phase_shift::Float64=0.0)

Get circadian factor at given time.

# Arguments
- `param`: Circadian parameter
- `time`: Time of day (hours from midnight)
- `phase_shift`: Phase shift (hours)

# Returns
Current value at given time
"""
function get_circadian_factor(param::CircadianParameter, time::Float64;
                               phase_shift::Float64=0.0)
    # Cosinor model: Y(t) = M + A*cos(2π(t-φ)/τ)
    # M = baseline (MESOR), A = amplitude, φ = acrophase, τ = period

    adjusted_acrophase = param.acrophase + phase_shift

    # Normalize to 0-24h
    adjusted_acrophase = mod(adjusted_acrophase, param.period)

    # Calculate factor
    phase = 2π * (time - adjusted_acrophase) / param.period
    factor = param.baseline * (1.0 + param.amplitude * cos(phase))

    return factor
end

"""
    get_circadian_factor(state::CircadianState, param_name::Symbol)

Get circadian factor for parameter at current time.
"""
function get_circadian_factor(state::CircadianState, param_name::Symbol)
    if !haskey(state.parameters, param_name)
        return 1.0
    end

    param = state.parameters[param_name]

    # Apply chronotype shift
    phase_shift = get(CHRONOTYPE_SHIFTS, state.chronotype, 0.0)

    # Apply jet lag
    phase_shift += state.jet_lag_hours

    # Shift workers have disrupted rhythm
    amplitude_mod = state.shift_worker ? 0.5 : 1.0

    # Get base factor
    factor = get_circadian_factor(param, state.current_time; phase_shift=phase_shift)

    # Dampen amplitude for shift workers
    if state.shift_worker
        factor = param.baseline + (factor - param.baseline) * amplitude_mod
    end

    return factor
end

# ============================================================================
# SIMULATION
# ============================================================================

"""
    simulate_circadian_variation(param_name::Symbol, duration_hours::Float64;
                                  start_time::Float64=0.0,
                                  chronotype::Symbol=:intermediate)

Simulate circadian variation over time.
"""
function simulate_circadian_variation(param_name::Symbol, duration_hours::Float64;
                                       start_time::Float64=0.0,
                                       chronotype::Symbol=:intermediate)
    if !haskey(CIRCADIAN_PARAMETERS, param_name)
        error("Unknown parameter: $param_name")
    end

    state = CircadianState(current_time=start_time, chronotype=chronotype)

    times = Float64[]
    values = Float64[]

    dt = 0.5  # 30-minute resolution
    for t in 0:dt:duration_hours
        state.current_time = mod(start_time + t, 24.0)
        push!(times, t)
        push!(values, get_circadian_factor(state, param_name))
    end

    return Dict(
        "time" => times,
        "values" => values,
        "parameter" => param_name,
        "baseline" => CIRCADIAN_PARAMETERS[param_name][:baseline],
        "amplitude" => CIRCADIAN_PARAMETERS[param_name][:amplitude],
        "acrophase" => CIRCADIAN_PARAMETERS[param_name][:acrophase]
    )
end

# ============================================================================
# CHRONOTHERAPY
# ============================================================================

"""
    calculate_optimal_dosing_time(drug_target::Symbol;
                                   goal::Symbol=:maximize,
                                   chronotype::Symbol=:intermediate)

Calculate optimal time for dosing based on circadian rhythms.

# Arguments
- `drug_target`: Target parameter (e.g., :cyp3a4_activity, :gfr)
- `goal`: :maximize or :minimize target at Tmax
- `chronotype`: Patient chronotype

# Returns
Optimal dosing time and rationale
"""
function calculate_optimal_dosing_time(drug_target::Symbol;
                                        goal::Symbol=:maximize,
                                        chronotype::Symbol=:intermediate)
    if !haskey(CIRCADIAN_PARAMETERS, drug_target)
        return Dict(
            "optimal_time" => 8.0,
            "rationale" => "Default morning dosing (no circadian data)"
        )
    end

    param = CIRCADIAN_PARAMETERS[drug_target]
    acrophase = param[:acrophase]

    # Apply chronotype shift
    phase_shift = get(CHRONOTYPE_SHIFTS, chronotype, 0.0)
    adjusted_acrophase = mod(acrophase + phase_shift, 24.0)

    # Determine optimal time
    if goal == :maximize
        # Dose so Tmax coincides with acrophase
        # Assume ~2h Tmax for oral drugs
        optimal_time = mod(adjusted_acrophase - 2.0, 24.0)
        rationale = "Dose 2h before $(drug_target) peak for maximal effect"
    else  # :minimize
        # Dose so Tmax coincides with nadir
        nadir = mod(adjusted_acrophase + 12.0, 24.0)
        optimal_time = mod(nadir - 2.0, 24.0)
        rationale = "Dose 2h before $(drug_target) nadir for minimal effect"
    end

    return Dict(
        "optimal_time" => optimal_time,
        "target_acrophase" => adjusted_acrophase,
        "rationale" => rationale,
        "amplitude" => param[:amplitude],
        "expected_variation" => 2.0 * param[:amplitude]  # Peak-to-trough
    )
end

"""
    get_chronotype_adjustment(chronotype::Symbol)

Get PK adjustments for chronotype.
"""
function get_chronotype_adjustment(chronotype::Symbol)
    shift = get(CHRONOTYPE_SHIFTS, chronotype, 0.0)

    return Dict(
        "phase_shift" => shift,
        "morning_dose_adjustment" => 1.0 + 0.05 * shift,
        "evening_dose_adjustment" => 1.0 - 0.05 * shift,
        "cyp_activity_shift" => shift,
        "renal_clearance_shift" => shift
    )
end

# ============================================================================
# PK IMPLICATIONS
# ============================================================================

"""
    calculate_circadian_pk_effect(state::CircadianState, drug_type::Symbol)

Calculate circadian effect on drug PK.

# Arguments
- `state`: Current circadian state
- `drug_type`: :hepatic_cleared, :renal_cleared, :high_protein_bound

# Returns
Dict with PK modifiers
"""
function calculate_circadian_pk_effect(state::CircadianState, drug_type::Symbol)
    # Get current factors
    hepatic_flow = get_circadian_factor(state, :hepatic_blood_flow)
    gfr = get_circadian_factor(state, :gfr)
    albumin = get_circadian_factor(state, :albumin)
    aag = get_circadian_factor(state, :aag)
    cyp3a4 = get_circadian_factor(state, :cyp3a4_activity)

    # Normalize to baseline
    hepatic_flow_factor = hepatic_flow / CIRCADIAN_PARAMETERS[:hepatic_blood_flow][:baseline]
    gfr_factor = gfr / CIRCADIAN_PARAMETERS[:gfr][:baseline]
    albumin_factor = albumin / CIRCADIAN_PARAMETERS[:albumin][:baseline]
    aag_factor = aag / CIRCADIAN_PARAMETERS[:aag][:baseline]
    cyp_factor = cyp3a4 / CIRCADIAN_PARAMETERS[:cyp3a4_activity][:baseline]

    if drug_type == :hepatic_cleared
        cl_factor = hepatic_flow_factor * cyp_factor
        fu_factor = 1.0 / albumin_factor  # Higher albumin = lower fu
    elseif drug_type == :renal_cleared
        cl_factor = gfr_factor
        fu_factor = 1.0
    elseif drug_type == :high_protein_bound_acidic
        cl_factor = hepatic_flow_factor
        fu_factor = 1.0 / albumin_factor
    elseif drug_type == :high_protein_bound_basic
        cl_factor = hepatic_flow_factor
        fu_factor = 1.0 / aag_factor
    else
        cl_factor = 1.0
        fu_factor = 1.0
    end

    return Dict(
        "clearance_factor" => cl_factor,
        "fu_factor" => fu_factor,
        "auc_factor" => 1.0 / cl_factor,
        "cmax_factor" => 1.0 / cl_factor * fu_factor,
        "time_of_day" => state.current_time,
        "hepatic_flow" => hepatic_flow,
        "gfr" => gfr,
        "albumin" => albumin
    )
end

"""
    calculate_chronotherapy_benefit(drug_name::String)

Estimate benefit of chronotherapy vs standard dosing.
"""
function calculate_chronotherapy_benefit(drug_name::String)
    # Drug-specific recommendations based on literature
    recommendations = Dict(
        "statins" => Dict(
            "optimal_time" => 21.0,      # Evening
            "rationale" => "HMG-CoA reductase peaks at night",
            "benefit" => 0.30            # 30% better LDL reduction
        ),
        "antihypertensives" => Dict(
            "optimal_time" => 22.0,      # Bedtime
            "rationale" => "Control nocturnal dipping",
            "benefit" => 0.20
        ),
        "aspirin" => Dict(
            "optimal_time" => 22.0,
            "rationale" => "Peak platelet aggregability in morning",
            "benefit" => 0.15
        ),
        "corticosteroids" => Dict(
            "optimal_time" => 8.0,       # Morning
            "rationale" => "Mimic natural cortisol rhythm",
            "benefit" => 0.25
        ),
        "methotrexate" => Dict(
            "optimal_time" => 22.0,
            "rationale" => "Reduced toxicity with evening dosing",
            "benefit" => 0.20
        ),
        "5fu" => Dict(
            "optimal_time" => 4.0,       # Night infusion
            "rationale" => "DPD activity lowest, better tolerance",
            "benefit" => 0.40
        )
    )

    drug_lower = lowercase(drug_name)

    if haskey(recommendations, drug_lower)
        return recommendations[drug_lower]
    else
        return Dict(
            "optimal_time" => 8.0,
            "rationale" => "No specific chronotherapy data",
            "benefit" => 0.0
        )
    end
end

end # module CircadianEffects
