"""
Acute Phase Response Module

Models dynamic changes in plasma protein composition during
inflammation, infection, trauma, and surgery.

Key Features:
- Cytokine-driven protein synthesis changes
- IL-6 → AAG upregulation (5-10× in 24h)
- Albumin downregulation
- CRP/SAA rapid increase
- Time-dependent kinetics

Clinical Relevance:
- Explains PK changes in ICU/sepsis
- Important for surgery/trauma patients
- Affects highly protein-bound drugs

References:
- Gabay C (2006) Acute-phase proteins and other systemic responses
- Roberts JA (2014) PK in critically ill patients
- Urien S (2003) Albumin binding in disease states

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module AcutePhaseResponse

using Statistics

export AcutePhaseState, CytokineProfile, AcutePhaseProtein
export create_acute_phase_state, simulate_acute_phase!, calculate_protein_changes
export apply_acute_phase_binding, get_time_course, get_cytokine_profile
export predict_pk_changes, get_dosing_recommendation
export ACUTE_PHASE_PROTEINS, CYTOKINE_EFFECTS

# ============================================================================
# CONSTANTS
# ============================================================================

# Normal plasma protein concentrations
const NORMAL_ALBUMIN = 40.0          # g/L
const NORMAL_AAG = 0.8               # g/L (80 mg/dL)
const NORMAL_CRP = 0.005             # g/L (5 mg/L)
const NORMAL_SAA = 0.003             # g/L (3 mg/L)
const NORMAL_FIBRINOGEN = 3.0        # g/L
const NORMAL_HAPTOGLOBIN = 1.0       # g/L
const NORMAL_CERULOPLASMIN = 0.3     # g/L
const NORMAL_FERRITIN = 0.0001       # g/L (100 ng/mL)

# Time constants (hours)
const CRP_HALF_RISE = 6.0            # Time to 50% of max
const SAA_HALF_RISE = 4.0            # Faster than CRP
const AAG_HALF_RISE = 24.0           # Slower
const ALBUMIN_HALF_FALL = 72.0       # Days to see effect
const FIBRINOGEN_HALF_RISE = 24.0

# Maximum fold changes
const MAX_CRP_FOLD = 1000.0          # Can increase 1000-fold
const MAX_SAA_FOLD = 1000.0
const MAX_AAG_FOLD = 5.0             # 5-fold increase
const MIN_ALBUMIN_FOLD = 0.5         # 50% reduction possible
const MAX_FIBRINOGEN_FOLD = 3.0
const MAX_FERRITIN_FOLD = 10.0

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    CytokineProfile

Inflammatory cytokine concentrations.

# Fields
- `il6::Float64`: IL-6 (pg/mL, normal <5)
- `il1b::Float64`: IL-1β (pg/mL, normal <5)
- `tnfa::Float64`: TNF-α (pg/mL, normal <10)
- `il10::Float64`: IL-10 (pg/mL, anti-inflammatory)
- `ifng::Float64`: IFN-γ (pg/mL)
"""
struct CytokineProfile
    il6::Float64
    il1b::Float64
    tnfa::Float64
    il10::Float64
    ifng::Float64

    function CytokineProfile(;
        il6 = 5.0,
        il1b = 5.0,
        tnfa = 10.0,
        il10 = 5.0,
        ifng = 5.0
    )
        new(il6, il1b, tnfa, il10, ifng)
    end
end

"""
    AcutePhaseProtein

Single acute phase protein state.

# Fields
- `name::Symbol`: Protein name
- `concentration::Float64`: Current concentration (g/L)
- `baseline::Float64`: Normal concentration (g/L)
- `fold_change::Float64`: Current fold change vs baseline
- `direction::Symbol`: :positive (increases) or :negative (decreases)
- `half_time::Float64`: Half-time for change (hours)
- `max_fold::Float64`: Maximum fold change
"""
mutable struct AcutePhaseProtein
    name::Symbol
    concentration::Float64
    baseline::Float64
    fold_change::Float64
    direction::Symbol
    half_time::Float64
    max_fold::Float64
end

"""
    AcutePhaseState

Complete acute phase response state.

# Fields
- `proteins::Dict{Symbol, AcutePhaseProtein}`: All APP states
- `cytokines::CytokineProfile`: Driving cytokines
- `time_since_onset::Float64`: Hours since inflammation onset
- `severity::Symbol`: :mild, :moderate, :severe, :critical
- `phase::Symbol`: :rising, :peak, :resolving, :resolved
- `trigger::Symbol`: :sepsis, :surgery, :trauma, :infection, :burns
"""
mutable struct AcutePhaseState
    proteins::Dict{Symbol, AcutePhaseProtein}
    cytokines::CytokineProfile
    time_since_onset::Float64
    severity::Symbol
    phase::Symbol
    trigger::Symbol
end

# ============================================================================
# ACUTE PHASE PROTEIN DATABASE
# ============================================================================

"""
Acute phase protein characteristics.
"""
const ACUTE_PHASE_PROTEINS = Dict{Symbol, Dict{Symbol, Any}}(
    :albumin => Dict(
        :baseline => NORMAL_ALBUMIN,
        :direction => :negative,
        :half_time => ALBUMIN_HALF_FALL,
        :max_fold => MIN_ALBUMIN_FOLD,
        :cytokine_driver => :il6,
        :drug_binding => :acidic
    ),

    :aag => Dict(
        :baseline => NORMAL_AAG,
        :direction => :positive,
        :half_time => AAG_HALF_RISE,
        :max_fold => MAX_AAG_FOLD,
        :cytokine_driver => :il6,
        :drug_binding => :basic
    ),

    :crp => Dict(
        :baseline => NORMAL_CRP,
        :direction => :positive,
        :half_time => CRP_HALF_RISE,
        :max_fold => MAX_CRP_FOLD,
        :cytokine_driver => :il6,
        :drug_binding => :none
    ),

    :saa => Dict(
        :baseline => NORMAL_SAA,
        :direction => :positive,
        :half_time => SAA_HALF_RISE,
        :max_fold => MAX_SAA_FOLD,
        :cytokine_driver => :il6,
        :drug_binding => :none
    ),

    :fibrinogen => Dict(
        :baseline => NORMAL_FIBRINOGEN,
        :direction => :positive,
        :half_time => FIBRINOGEN_HALF_RISE,
        :max_fold => MAX_FIBRINOGEN_FOLD,
        :cytokine_driver => :il6,
        :drug_binding => :minor
    ),

    :haptoglobin => Dict(
        :baseline => NORMAL_HAPTOGLOBIN,
        :direction => :positive,
        :half_time => 36.0,
        :max_fold => 3.0,
        :cytokine_driver => :il6,
        :drug_binding => :none
    ),

    :ceruloplasmin => Dict(
        :baseline => NORMAL_CERULOPLASMIN,
        :direction => :positive,
        :half_time => 48.0,
        :max_fold => 2.0,
        :cytokine_driver => :il6,
        :drug_binding => :minor
    ),

    :ferritin => Dict(
        :baseline => NORMAL_FERRITIN,
        :direction => :positive,
        :half_time => 24.0,
        :max_fold => MAX_FERRITIN_FOLD,
        :cytokine_driver => :il6,
        :drug_binding => :none
    ),

    :transferrin => Dict(
        :baseline => 2.5,  # g/L
        :direction => :negative,
        :half_time => 48.0,
        :max_fold => 0.6,
        :cytokine_driver => :il6,
        :drug_binding => :minor
    ),

    :prealbumin => Dict(
        :baseline => 0.25,  # g/L (transthyretin)
        :direction => :negative,
        :half_time => 24.0,
        :max_fold => 0.4,
        :cytokine_driver => :il6,
        :drug_binding => :thyroid_hormones
    )
)

# ============================================================================
# CYTOKINE EFFECTS
# ============================================================================

"""
Cytokine effects on acute phase protein synthesis.
"""
const CYTOKINE_EFFECTS = Dict{Symbol, Dict{Symbol, Float64}}(
    :il6 => Dict(
        :albumin => -0.3,      # Downregulates
        :aag => 1.0,           # Strong upregulation
        :crp => 1.0,
        :saa => 1.0,
        :fibrinogen => 0.8,
        :haptoglobin => 0.6,
        :ferritin => 0.7
    ),

    :il1b => Dict(
        :albumin => -0.2,
        :aag => 0.5,
        :crp => 0.3,
        :saa => 0.5,
        :fibrinogen => 0.4
    ),

    :tnfa => Dict(
        :albumin => -0.4,      # Strong negative effect
        :aag => 0.3,
        :crp => 0.2,
        :saa => 0.3
    ),

    :il10 => Dict(  # Anti-inflammatory - dampens response
        :albumin => 0.1,       # Slight protective
        :aag => -0.2,
        :crp => -0.3,
        :saa => -0.3
    )
)

# ============================================================================
# INITIALIZATION
# ============================================================================

"""
    create_acute_phase_state(trigger::Symbol; severity::Symbol=:moderate)

Create initial acute phase response state.

Triggers:
- :sepsis - Severe systemic inflammation
- :surgery - Post-operative response
- :trauma - Injury response
- :infection - Localized infection
- :burns - Burn injury
- :mi - Myocardial infarction
- :pancreatitis - Acute pancreatitis
"""
function create_acute_phase_state(trigger::Symbol; severity::Symbol=:moderate)
    # Initialize proteins at baseline
    proteins = Dict{Symbol, AcutePhaseProtein}()

    for (name, params) in ACUTE_PHASE_PROTEINS
        proteins[name] = AcutePhaseProtein(
            name,
            params[:baseline],
            params[:baseline],
            1.0,
            params[:direction],
            params[:half_time],
            params[:max_fold]
        )
    end

    # Set cytokines based on trigger and severity
    cytokines = get_cytokine_profile(trigger, severity)

    return AcutePhaseState(
        proteins,
        cytokines,
        0.0,
        severity,
        :rising,
        trigger
    )
end

"""
    get_cytokine_profile(trigger::Symbol, severity::Symbol)

Get cytokine concentrations for trigger/severity.
"""
function get_cytokine_profile(trigger::Symbol, severity::Symbol)
    # Base multiplier for severity
    mult = if severity == :mild
        1.0
    elseif severity == :moderate
        5.0
    elseif severity == :severe
        20.0
    else  # :critical
        50.0
    end

    # Trigger-specific patterns
    if trigger == :sepsis
        return CytokineProfile(
            il6 = 5.0 * mult * 10,   # Very high in sepsis
            il1b = 5.0 * mult * 2,
            tnfa = 10.0 * mult * 5,
            il10 = 5.0 * mult,        # Also elevated
            ifng = 5.0 * mult
        )
    elseif trigger == :surgery
        return CytokineProfile(
            il6 = 5.0 * mult * 5,
            il1b = 5.0 * mult,
            tnfa = 10.0 * mult * 2,
            il10 = 5.0 * mult * 2,    # Protective response
            ifng = 5.0
        )
    elseif trigger == :trauma
        return CytokineProfile(
            il6 = 5.0 * mult * 8,
            il1b = 5.0 * mult * 2,
            tnfa = 10.0 * mult * 3,
            il10 = 5.0 * mult,
            ifng = 5.0 * mult
        )
    elseif trigger == :burns
        return CytokineProfile(
            il6 = 5.0 * mult * 15,    # Very high in burns
            il1b = 5.0 * mult * 3,
            tnfa = 10.0 * mult * 4,
            il10 = 5.0 * mult,
            ifng = 5.0 * mult * 2
        )
    elseif trigger == :infection
        return CytokineProfile(
            il6 = 5.0 * mult * 3,
            il1b = 5.0 * mult * 2,
            tnfa = 10.0 * mult * 2,
            il10 = 5.0,
            ifng = 5.0 * mult * 2
        )
    else
        return CytokineProfile(
            il6 = 5.0 * mult,
            il1b = 5.0 * mult,
            tnfa = 10.0 * mult,
            il10 = 5.0,
            ifng = 5.0
        )
    end
end

# ============================================================================
# TIME-DEPENDENT SIMULATION
# ============================================================================

"""
    simulate_acute_phase!(state::AcutePhaseState, duration_hours::Float64;
                          resolution_start::Float64=72.0)

Simulate acute phase response over time.

# Arguments
- `state`: Current state (modified in place)
- `duration_hours`: Time to simulate
- `resolution_start`: Hours until resolution begins
"""
function simulate_acute_phase!(state::AcutePhaseState, duration_hours::Float64;
                                resolution_start::Float64=72.0)
    dt = 1.0  # 1 hour timestep

    for t in 0:dt:duration_hours
        state.time_since_onset += dt

        # Update phase
        if state.time_since_onset < resolution_start
            state.phase = :rising
        elseif state.time_since_onset < resolution_start + 48.0
            state.phase = :peak
        elseif state.time_since_onset < resolution_start + 168.0  # 1 week
            state.phase = :resolving
        else
            state.phase = :resolved
        end

        # Calculate cytokine-driven synthesis rate for each protein
        for (name, protein) in state.proteins
            target_fold = calculate_target_fold_change(protein, state)

            # Exponential approach to target
            k = log(2) / protein.half_time

            if state.phase == :resolving || state.phase == :resolved
                # Return toward baseline
                target_fold = 1.0 + (target_fold - 1.0) * exp(-0.01 * (state.time_since_onset - resolution_start))
            end

            # Update concentration
            if protein.direction == :positive
                new_fold = protein.fold_change + k * dt * (target_fold - protein.fold_change)
                new_fold = min(new_fold, protein.max_fold)
            else  # :negative
                new_fold = protein.fold_change + k * dt * (target_fold - protein.fold_change)
                new_fold = max(new_fold, protein.max_fold)  # max_fold < 1 for negative
            end

            protein.fold_change = new_fold
            protein.concentration = protein.baseline * new_fold
        end
    end

    return state
end

"""
    calculate_target_fold_change(protein::AcutePhaseProtein, state::AcutePhaseState)

Calculate target fold change based on cytokines.
"""
function calculate_target_fold_change(protein::AcutePhaseProtein, state::AcutePhaseState)
    cytokines = state.cytokines

    # Sum cytokine effects
    effect = 0.0

    if haskey(CYTOKINE_EFFECTS, :il6) && haskey(CYTOKINE_EFFECTS[:il6], protein.name)
        # Normalized IL-6 effect (log scale)
        il6_norm = log10(max(cytokines.il6, 1.0)) / log10(1000.0)
        effect += CYTOKINE_EFFECTS[:il6][protein.name] * il6_norm
    end

    if haskey(CYTOKINE_EFFECTS, :il1b) && haskey(CYTOKINE_EFFECTS[:il1b], protein.name)
        il1b_norm = log10(max(cytokines.il1b, 1.0)) / log10(100.0)
        effect += CYTOKINE_EFFECTS[:il1b][protein.name] * il1b_norm
    end

    if haskey(CYTOKINE_EFFECTS, :tnfa) && haskey(CYTOKINE_EFFECTS[:tnfa], protein.name)
        tnfa_norm = log10(max(cytokines.tnfa, 1.0)) / log10(500.0)
        effect += CYTOKINE_EFFECTS[:tnfa][protein.name] * tnfa_norm
    end

    if haskey(CYTOKINE_EFFECTS, :il10) && haskey(CYTOKINE_EFFECTS[:il10], protein.name)
        il10_norm = log10(max(cytokines.il10, 1.0)) / log10(100.0)
        effect += CYTOKINE_EFFECTS[:il10][protein.name] * il10_norm
    end

    # Convert effect to fold change
    if protein.direction == :positive
        # Positive APPs increase
        target = 1.0 + effect * (protein.max_fold - 1.0)
        target = clamp(target, 1.0, protein.max_fold)
    else
        # Negative APPs decrease
        target = 1.0 + effect * (1.0 - protein.max_fold)
        target = clamp(target, protein.max_fold, 1.0)
    end

    return target
end

# ============================================================================
# PROTEIN STATE QUERIES
# ============================================================================

"""
    calculate_protein_changes(state::AcutePhaseState)

Get current protein concentrations and changes.
"""
function calculate_protein_changes(state::AcutePhaseState)
    result = Dict{Symbol, Dict{String, Float64}}()

    for (name, protein) in state.proteins
        result[name] = Dict(
            "concentration" => protein.concentration,
            "baseline" => protein.baseline,
            "fold_change" => protein.fold_change,
            "percent_change" => (protein.fold_change - 1.0) * 100.0
        )
    end

    return result
end

"""
    get_time_course(trigger::Symbol, severity::Symbol, duration_hours::Float64;
                    proteins::Vector{Symbol}=[:albumin, :aag, :crp])

Get time course of protein changes.
"""
function get_time_course(trigger::Symbol, severity::Symbol, duration_hours::Float64;
                          proteins::Vector{Symbol}=[:albumin, :aag, :crp])
    state = create_acute_phase_state(trigger; severity=severity)

    times = Float64[]
    concentrations = Dict{Symbol, Vector{Float64}}()
    for p in proteins
        concentrations[p] = Float64[]
    end

    dt = 1.0  # 1 hour
    for t in 0:dt:duration_hours
        push!(times, t)
        for p in proteins
            push!(concentrations[p], state.proteins[p].concentration)
        end
        simulate_acute_phase!(state, dt)
    end

    return Dict(
        "time" => times,
        "concentrations" => concentrations,
        "trigger" => trigger,
        "severity" => severity
    )
end

# ============================================================================
# DRUG BINDING EFFECTS
# ============================================================================

"""
    apply_acute_phase_binding(fu_normal::Float64, drug_type::Symbol,
                               state::AcutePhaseState)

Calculate adjusted fu during acute phase response.

# Arguments
- `fu_normal`: Normal fraction unbound
- `drug_type`: :acidic (albumin-bound), :basic (AAG-bound), :neutral
- `state`: Current acute phase state

# Returns
Dict with adjusted fu and binding details
"""
function apply_acute_phase_binding(fu_normal::Float64, drug_type::Symbol,
                                    state::AcutePhaseState)
    albumin = state.proteins[:albumin]
    aag = state.proteins[:aag]

    # Calculate binding adjustments
    if drug_type == :acidic
        # Albumin-bound drugs: fu ↑ when albumin ↓
        # fu_new = fu_old * (albumin_normal / albumin_current)
        albumin_ratio = albumin.baseline / albumin.concentration
        fu_adjusted = fu_normal * albumin_ratio

        # Saturability correction
        fu_adjusted = min(fu_adjusted, 1.0)
        fu_adjusted = min(fu_adjusted, fu_normal * 3.0)  # Max 3-fold increase

        binding_protein = :albumin
        protein_change = albumin.fold_change

    elseif drug_type == :basic
        # AAG-bound drugs: fu ↓ when AAG ↑
        # fu_new = fu_old / (aag_current / aag_normal)
        aag_ratio = aag.concentration / aag.baseline
        fu_adjusted = fu_normal / aag_ratio

        # Minimum fu
        fu_adjusted = max(fu_adjusted, 0.001)
        fu_adjusted = max(fu_adjusted, fu_normal * 0.2)  # Min 80% decrease

        binding_protein = :aag
        protein_change = aag.fold_change

    else  # :neutral
        # Affected by both, take average effect
        albumin_effect = albumin.baseline / albumin.concentration
        aag_effect = aag.baseline / aag.concentration
        combined = (albumin_effect + aag_effect) / 2.0
        fu_adjusted = fu_normal * combined
        fu_adjusted = clamp(fu_adjusted, fu_normal * 0.5, fu_normal * 2.0)

        binding_protein = :mixed
        protein_change = (albumin.fold_change + aag.fold_change) / 2.0
    end

    return Dict(
        "fu_adjusted" => fu_adjusted,
        "fu_normal" => fu_normal,
        "fu_ratio" => fu_adjusted / fu_normal,
        "binding_protein" => binding_protein,
        "protein_fold_change" => protein_change,
        "phase" => state.phase,
        "time_since_onset" => state.time_since_onset,
        "albumin_concentration" => albumin.concentration,
        "aag_concentration" => aag.concentration
    )
end

# ============================================================================
# CLINICAL UTILITIES
# ============================================================================

"""
    estimate_time_since_onset(crp::Float64, aag::Float64)

Estimate hours since inflammation onset from protein levels.
"""
function estimate_time_since_onset(crp::Float64, aag::Float64)
    # CRP rises fast, AAG slower
    # Use ratio to estimate timing

    crp_fold = crp / NORMAL_CRP
    aag_fold = aag / NORMAL_AAG

    if crp_fold < 2.0 && aag_fold < 1.2
        return 0.0  # No significant inflammation
    end

    # CRP peaks around 48h, AAG around 72-96h
    # If CRP >> AAG change, early phase
    # If AAG catching up, later phase

    if crp_fold > 10.0 && aag_fold < 2.0
        # Early phase (6-24h)
        estimated = 6.0 + 18.0 * (log10(crp_fold) / 3.0)
    elseif crp_fold > 5.0 && aag_fold >= 2.0
        # Peak phase (24-72h)
        estimated = 24.0 + 48.0 * ((aag_fold - 1.0) / 4.0)
    else
        # Resolving or chronic
        estimated = 72.0 + 48.0 * (aag_fold / 5.0)
    end

    return estimated
end

"""
    predict_pk_changes(fu_normal::Float64, drug_type::Symbol,
                       crp::Float64, aag::Float64)

Predict PK changes from clinical markers.
"""
function predict_pk_changes(fu_normal::Float64, drug_type::Symbol,
                             crp::Float64, aag::Float64)
    # Estimate albumin from negative APP relationship
    # In acute phase, albumin typically drops as CRP rises
    albumin_fold = 1.0 - 0.1 * log10(max(crp / NORMAL_CRP, 1.0))
    albumin_fold = max(albumin_fold, 0.5)
    albumin = NORMAL_ALBUMIN * albumin_fold

    aag_current = aag

    # Calculate fu change
    if drug_type == :acidic
        fu_ratio = NORMAL_ALBUMIN / albumin
        fu_adjusted = min(fu_normal * fu_ratio, 1.0)
    elseif drug_type == :basic
        fu_ratio = NORMAL_AAG / aag_current
        fu_adjusted = max(fu_normal * fu_ratio, 0.001)
    else
        fu_adjusted = fu_normal
    end

    # Predict Vd change (for restrictively cleared drugs)
    # Vd ∝ fu for high extraction drugs
    vd_ratio = fu_adjusted / fu_normal

    # Predict CL change
    # For low ER drugs: CL ∝ fu
    # For high ER drugs: CL unchanged
    cl_ratio_low_er = fu_adjusted / fu_normal
    cl_ratio_high_er = 1.0

    return Dict(
        "fu_adjusted" => fu_adjusted,
        "fu_ratio" => fu_adjusted / fu_normal,
        "vd_ratio" => vd_ratio,
        "cl_ratio_low_er" => cl_ratio_low_er,
        "cl_ratio_high_er" => cl_ratio_high_er,
        "estimated_albumin" => albumin,
        "aag" => aag_current,
        "recommendation" => get_dosing_recommendation(drug_type, fu_adjusted / fu_normal)
    )
end

"""
    get_dosing_recommendation(drug_type::Symbol, fu_ratio::Float64)

Get dosing recommendation based on binding changes.
"""
function get_dosing_recommendation(drug_type::Symbol, fu_ratio::Float64)
    if drug_type == :acidic
        if fu_ratio > 2.0
            return "Consider 50% dose reduction for narrow TI drugs"
        elseif fu_ratio > 1.5
            return "Monitor for toxicity; consider dose adjustment"
        else
            return "Standard dosing likely appropriate"
        end
    elseif drug_type == :basic
        if fu_ratio < 0.5
            return "Consider dose increase or more frequent dosing"
        elseif fu_ratio < 0.7
            return "Monitor for subtherapeutic levels"
        else
            return "Standard dosing likely appropriate"
        end
    else
        return "Monitor drug levels if available"
    end
end

end # module AcutePhaseResponse
