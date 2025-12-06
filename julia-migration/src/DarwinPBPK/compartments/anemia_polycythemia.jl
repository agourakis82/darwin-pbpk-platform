"""
Anemia/Polycythemia Adaptation Module

Comprehensive hematocrit-dependent pharmacokinetic adjustments for anemia
and polycythemia conditions. Integrates RBC partitioning, EPO effects,
and disease-specific PK alterations.

Key Features:
- Hematocrit-dependent blood-plasma ratio corrections
- Disease-specific PK profiles (iron deficiency, SCD, thalassemia, etc.)
- Polycythemia vera and secondary polycythemia modeling
- EPO therapy pharmacodynamic effects on PK
- Reticulocyte fraction adjustments
- RBC indices (MCV, MCH, MCHC) effects on transport

Equations:
- Blood-Plasma Ratio: Rb = 1 - Hct + (Hct × Ke_p)
- Vd Correction: Vd_corrected = Vd_reference × (Hct_patient/Hct_reference)^α
- Clearance: CL_adjusted = CL_reference × (Hct_reference/Hct_patient)^β

References:
- Størset E et al. Clin Pharmacokinet 2019 (Tacrolimus hematocrit effects)
- Darbari DS et al. Blood 2009 (Morphine PK in SCD)
- Mehta P. J Clin Apher 2016 (Hyperviscosity syndrome)
- FDA Guidance: PBPK Analysis (2018)

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module AnemiaPolycythemia

using Statistics

export HematologicalState, AnemiaProfile, PolycythemiaProfile
export RBCIndices, ReticulocyteState, EPOState
export create_normal_hematology, create_anemia_state, create_polycythemia_state
export calculate_hematocrit_correction, calculate_blood_plasma_ratio
export calculate_vd_correction, calculate_clearance_correction
export apply_anemia_pk_adjustments, apply_polycythemia_pk_adjustments
export calculate_rbc_partitioning, estimate_reticulocyte_effect
export simulate_epo_therapy, calculate_transfusion_effect
export ANEMIA_PROFILES, POLYCYTHEMIA_PROFILES, RBC_PARTITION_DATABASE
export NORMAL_HEMATOLOGY, EPO_PARAMETERS

# ============================================================================
# CONSTANTS
# ============================================================================

"""Normal hematological reference values."""
const NORMAL_HEMATOLOGY = Dict{Symbol, Float64}(
    :hematocrit => 0.42,           # L/L (range 0.37-0.47 female, 0.42-0.52 male)
    :hemoglobin => 14.0,           # g/dL (range 12-16 female, 14-18 male)
    :rbc_count => 4.7e12,          # cells/L
    :mcv => 90.0,                  # fL (mean corpuscular volume)
    :mch => 30.0,                  # pg (mean corpuscular hemoglobin)
    :mchc => 34.0,                 # g/dL (mean corpuscular Hb concentration)
    :rdw => 12.5,                  # % (red cell distribution width)
    :reticulocyte_fraction => 0.01, # 1% of RBCs
    :plasma_volume => 3.0,         # L
    :blood_volume => 5.0           # L
)

"""EPO pharmacokinetic/pharmacodynamic parameters."""
const EPO_PARAMETERS = Dict{Symbol, Any}(
    :half_life_iv => 6.0,          # hours
    :half_life_sc => 24.0,         # hours
    :bioavailability_sc => 0.36,   # 36%
    :vd => 0.05,                   # L/kg (plasma volume)
    :clearance => 0.003,           # L/h/kg
    :reticulocyte_peak_day => 10,  # days after start
    :hematocrit_effect_week => 4,  # weeks to see Hct change
    :target_hb_increase => 1.0     # g/dL per 4 weeks at standard dose
)

"""
RBC-to-plasma partition coefficients (Ke_p) for drugs.
Ke_p = Crbc / Cplasma
Higher values = more RBC accumulation.
"""
const RBC_PARTITION_DATABASE = Dict{String, Dict{Symbol, Any}}(
    # High RBC partitioning (Rb >> 1)
    "tacrolimus" => Dict(
        :ke_p => 35.0,             # Rb = 15-35 depending on hematocrit
        :binding_saturable => true,
        :km_rbc => 50.0,           # ng/mL (saturation constant)
        :hct_sensitivity => :high,
        :clinical_note => "TDM critical, standardize to Hct=0.45"
    ),
    "cyclosporine" => Dict(
        :ke_p => 2.5,
        :binding_saturable => false,
        :hct_sensitivity => :high,
        :clinical_note => "Blood concentration monitoring required"
    ),
    "chloroquine" => Dict(
        :ke_p => 4.0,              # Rb = 3-5
        :binding_saturable => false,
        :hct_sensitivity => :moderate,
        :clinical_note => "High RBC concentration for antimalarial effect"
    ),
    "sirolimus" => Dict(
        :ke_p => 12.0,
        :binding_saturable => true,
        :km_rbc => 30.0,
        :hct_sensitivity => :high
    ),

    # Moderate RBC partitioning (Rb ~ 1)
    "phenytoin" => Dict(
        :ke_p => 1.1,
        :binding_saturable => false,
        :hct_sensitivity => :low
    ),
    "metformin" => Dict(
        :ke_p => 1.0,              # Rb ≈ 1
        :binding_saturable => false,
        :hct_sensitivity => :minimal,
        :clinical_note => "Equal RBC/plasma distribution"
    ),

    # Low RBC partitioning (Rb < 1)
    "warfarin" => Dict(
        :ke_p => 0.55,             # Rb = 0.55-0.65
        :binding_saturable => false,
        :hct_sensitivity => :low,
        :clinical_note => "Albumin binding more important than Hct"
    ),
    "gentamicin" => Dict(
        :ke_p => 0.1,              # Minimal RBC uptake
        :binding_saturable => false,
        :hct_sensitivity => :minimal
    ),
    "vancomycin" => Dict(
        :ke_p => 0.15,
        :binding_saturable => false,
        :hct_sensitivity => :minimal
    )
)

"""Anemia profiles with PK adjustments."""
const ANEMIA_PROFILES = Dict{Symbol, Dict{Symbol, Any}}(
    :iron_deficiency => Dict(
        :name => "Iron Deficiency Anemia",
        :typical_hct => 0.28,
        :typical_hb => 9.0,
        :mcv => 70.0,              # Microcytic
        :mchc => 30.0,
        :absorption_factor => 0.6, # Reduced GI absorption
        :cyp_activity => 1.0,      # Normal
        :renal_function => 1.0,
        :protein_binding => 1.0,
        :special_considerations => [
            "Iron chelation by tetracyclines, fluoroquinolones",
            "Reduced oral absorption of iron-binding drugs",
            "Monitor for pica-related ingestions"
        ]
    ),
    :chronic_disease => Dict(
        :name => "Anemia of Chronic Disease/Inflammation",
        :typical_hct => 0.30,
        :typical_hb => 10.0,
        :mcv => 85.0,              # Normocytic
        :mchc => 33.0,
        :absorption_factor => 0.9,
        :cyp_activity => 0.7,      # IL-6 downregulates CYPs
        :renal_function => 0.9,
        :protein_binding => 0.8,   # Hypoalbuminemia
        :aag_elevation => 1.5,     # Acute phase
        :special_considerations => [
            "CYP1A2, CYP3A4 downregulated by IL-6",
            "Hepcidin elevation reduces iron absorption",
            "Protein binding changes"
        ]
    ),
    :hemolytic => Dict(
        :name => "Hemolytic Anemia",
        :typical_hct => 0.25,
        :typical_hb => 8.0,
        :mcv => 100.0,             # Macrocytic (reticulocytes)
        :reticulocyte_fraction => 0.10,  # 10%
        :absorption_factor => 1.0,
        :cyp_activity => 1.0,
        :renal_function => 0.9,    # Hemoglobin nephrotoxicity
        :bilirubin_elevation => 3.0,
        :special_considerations => [
            "High reticulocyte count affects drug transport",
            "Unconjugated hyperbilirubinemia",
            "Splenomegaly increases RES clearance"
        ]
    ),
    :sickle_cell => Dict(
        :name => "Sickle Cell Disease",
        :typical_hct => 0.25,
        :typical_hb => 8.0,
        :mcv => 90.0,
        :reticulocyte_fraction => 0.15,
        :absorption_factor => 0.9,
        :cyp_activity => 1.1,      # Slightly induced
        :renal_function => 0.8,    # Sickle nephropathy
        :vd_adjustment => 1.2,     # Increased Vd
        :clearance_adjustment => 1.5,  # 50% higher morphine CL
        :special_considerations => [
            "Morphine CL 55% higher (1.89 vs 1.22 L/h/kg)",
            "Hydroxyurea PK: 5-fold variability",
            "Reticulocytes have 3× higher glutamine transport",
            "Frequent transfusions alter drug distribution"
        ]
    ),
    :thalassemia => Dict(
        :name => "Thalassemia Major",
        :typical_hct => 0.22,
        :typical_hb => 7.0,
        :mcv => 65.0,              # Microcytic
        :reticulocyte_fraction => 0.08,
        :absorption_factor => 0.8,
        :cyp_activity => 0.9,
        :renal_function => 0.9,
        :iron_overload => true,
        :special_considerations => [
            "Chelation therapy required (deferasirox, deferoxamine)",
            "Deferasirox bioavailability saturates at high doses",
            "Splenomegaly common",
            "Regular transfusions alter PK acutely"
        ]
    ),
    :aplastic => Dict(
        :name => "Aplastic Anemia",
        :typical_hct => 0.20,
        :typical_hb => 6.5,
        :mcv => 100.0,             # Macrocytic
        :reticulocyte_fraction => 0.001, # Very low
        :pancytopenia => true,
        :absorption_factor => 1.0,
        :cyp_activity => 0.9,
        :renal_function => 0.95,
        :special_considerations => [
            "Pancytopenia affects drug distribution",
            "Cyclosporine/ATG therapy common",
            "Monitor for infections affecting PK"
        ]
    ),
    :renal => Dict(
        :name => "Anemia of Chronic Kidney Disease",
        :typical_hct => 0.28,
        :typical_hb => 9.0,
        :mcv => 90.0,
        :epo_deficiency => true,
        :absorption_factor => 0.85,
        :cyp_activity => 0.8,
        :renal_function => 0.3,    # GFR 15-30 mL/min
        :protein_binding => 0.7,
        :uremic_toxins => true,
        :special_considerations => [
            "EPO therapy common",
            "Uremic toxins displace albumin binding",
            "GFR-dependent dose adjustments needed",
            "Dialysis removes some drugs"
        ]
    )
)

"""Polycythemia profiles."""
const POLYCYTHEMIA_PROFILES = Dict{Symbol, Dict{Symbol, Any}}(
    :vera => Dict(
        :name => "Polycythemia Vera",
        :typical_hct => 0.55,
        :typical_hb => 18.0,
        :mcv => 85.0,
        :thrombosis_risk => :high,
        :viscosity_factor => 2.5,  # vs normal
        :hepatic_flow_reduction => 0.4,
        :special_considerations => [
            "Phlebotomy target: Hct < 0.45",
            "Hydroxyurea or interferon therapy",
            "High thrombosis risk affects drug delivery",
            "Splenomegaly common"
        ]
    ),
    :secondary_hypoxia => Dict(
        :name => "Secondary Polycythemia (Hypoxia)",
        :typical_hct => 0.52,
        :typical_hb => 17.0,
        :cause => "Chronic hypoxia (COPD, high altitude, OSA)",
        :viscosity_factor => 2.0,
        :hepatic_flow_reduction => 0.2,
        :special_considerations => [
            "EPO appropriately elevated",
            "Treat underlying cause",
            "Less aggressive Hct targets"
        ]
    ),
    :secondary_epo => Dict(
        :name => "Secondary Polycythemia (EPO-secreting)",
        :typical_hct => 0.58,
        :typical_hb => 19.0,
        :cause => "EPO-secreting tumor or exogenous EPO",
        :viscosity_factor => 3.0,
        :hepatic_flow_reduction => 0.5,
        :special_considerations => [
            "Investigate for tumor",
            "May need phlebotomy",
            "Severe hyperviscosity risk"
        ]
    )
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    RBCIndices

Red blood cell indices affecting drug transport.
"""
struct RBCIndices
    mcv::Float64          # Mean corpuscular volume (fL)
    mch::Float64          # Mean corpuscular hemoglobin (pg)
    mchc::Float64         # Mean corpuscular Hb concentration (g/dL)
    rdw::Float64          # Red cell distribution width (%)

    function RBCIndices(;
        mcv::Float64 = 90.0,
        mch::Float64 = 30.0,
        mchc::Float64 = 34.0,
        rdw::Float64 = 12.5
    )
        new(mcv, mch, mchc, rdw)
    end
end

"""
    ReticulocyteState

Reticulocyte parameters for PK modeling.
"""
struct ReticulocyteState
    fraction::Float64           # Fraction of total RBCs
    absolute_count::Float64     # cells/L
    transport_enhancement::Float64  # vs mature RBCs

    function ReticulocyteState(;
        fraction::Float64 = 0.01,
        absolute_count::Float64 = 50e9,
        transport_enhancement::Float64 = 1.5
    )
        new(fraction, absolute_count, transport_enhancement)
    end
end

"""
    EPOState

Erythropoietin therapy state.
"""
struct EPOState
    on_therapy::Bool
    dose_units_per_week::Float64
    weeks_on_therapy::Int
    baseline_hct::Float64
    current_hct::Float64
    target_hct::Float64

    function EPOState(;
        on_therapy::Bool = false,
        dose_units_per_week::Float64 = 0.0,
        weeks_on_therapy::Int = 0,
        baseline_hct::Float64 = 0.28,
        current_hct::Float64 = 0.28,
        target_hct::Float64 = 0.35
    )
        new(on_therapy, dose_units_per_week, weeks_on_therapy,
            baseline_hct, current_hct, target_hct)
    end
end

"""
    HematologicalState

Complete hematological state for PK modeling.
"""
struct HematologicalState
    hematocrit::Float64
    hemoglobin::Float64
    rbc_count::Float64
    indices::RBCIndices
    reticulocytes::ReticulocyteState
    epo_state::EPOState
    condition::Symbol           # :normal, :anemia_xxx, :polycythemia_xxx
    severity::Symbol            # :mild, :moderate, :severe

    function HematologicalState(;
        hematocrit::Float64 = 0.42,
        hemoglobin::Float64 = 14.0,
        rbc_count::Float64 = 4.7e12,
        indices::RBCIndices = RBCIndices(),
        reticulocytes::ReticulocyteState = ReticulocyteState(),
        epo_state::EPOState = EPOState(),
        condition::Symbol = :normal,
        severity::Symbol = :none
    )
        new(hematocrit, hemoglobin, rbc_count, indices,
            reticulocytes, epo_state, condition, severity)
    end
end

"""
    AnemiaProfile

Anemia-specific PK adjustments.
"""
struct AnemiaProfile
    state::HematologicalState
    absorption_factor::Float64
    cyp_activity::Float64
    renal_function::Float64
    protein_binding_factor::Float64
    vd_adjustment::Float64
    clearance_adjustment::Float64
    special_considerations::Vector{String}
end

"""
    PolycythemiaProfile

Polycythemia-specific PK adjustments.
"""
struct PolycythemiaProfile
    state::HematologicalState
    viscosity_factor::Float64
    hepatic_flow_reduction::Float64
    tissue_perfusion_factor::Float64
    thrombosis_risk::Symbol
    special_considerations::Vector{String}
end

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

"""
    create_normal_hematology()

Create normal hematological state.
"""
function create_normal_hematology()
    return HematologicalState(
        hematocrit = 0.42,
        hemoglobin = 14.0,
        rbc_count = 4.7e12,
        indices = RBCIndices(mcv=90.0, mch=30.0, mchc=34.0, rdw=12.5),
        reticulocytes = ReticulocyteState(fraction=0.01),
        condition = :normal,
        severity = :none
    )
end

"""
    create_anemia_state(anemia_type::Symbol; severity::Symbol=:moderate, kwargs...)

Create anemia state with appropriate parameters.

# Arguments
- `anemia_type`: :iron_deficiency, :chronic_disease, :hemolytic, :sickle_cell, :thalassemia, :aplastic, :renal
- `severity`: :mild, :moderate, :severe

# Example
```julia
state = create_anemia_state(:sickle_cell; severity=:severe)
```
"""
function create_anemia_state(anemia_type::Symbol; severity::Symbol=:moderate, kwargs...)
    if !haskey(ANEMIA_PROFILES, anemia_type)
        error("Unknown anemia type: $anemia_type. Valid types: $(keys(ANEMIA_PROFILES))")
    end

    profile = ANEMIA_PROFILES[anemia_type]

    # Adjust hematocrit by severity
    severity_factors = Dict(
        :mild => 1.15,
        :moderate => 1.0,
        :severe => 0.75
    )
    severity_factor = get(severity_factors, severity, 1.0)

    hct = get(kwargs, :hematocrit, profile[:typical_hct] * severity_factor)
    hb = get(kwargs, :hemoglobin, profile[:typical_hb] * severity_factor)

    reticulocyte_fraction = get(profile, :reticulocyte_fraction, 0.01)
    if severity == :severe
        reticulocyte_fraction *= 1.5  # Compensatory increase
    end

    indices = RBCIndices(
        mcv = get(profile, :mcv, 90.0),
        mch = get(profile, :mch, 30.0),
        mchc = get(profile, :mchc, 33.0)
    )

    reticulocytes = ReticulocyteState(
        fraction = reticulocyte_fraction,
        transport_enhancement = reticulocyte_fraction > 0.05 ? 2.0 : 1.5
    )

    state = HematologicalState(
        hematocrit = hct,
        hemoglobin = hb,
        indices = indices,
        reticulocytes = reticulocytes,
        condition = Symbol("anemia_", anemia_type),
        severity = severity
    )

    return AnemiaProfile(
        state,
        get(profile, :absorption_factor, 1.0),
        get(profile, :cyp_activity, 1.0),
        get(profile, :renal_function, 1.0),
        get(profile, :protein_binding, 1.0),
        get(profile, :vd_adjustment, 1.0),
        get(profile, :clearance_adjustment, 1.0),
        get(profile, :special_considerations, String[])
    )
end

"""
    create_polycythemia_state(poly_type::Symbol; hematocrit::Float64=0.0)

Create polycythemia state.

# Arguments
- `poly_type`: :vera, :secondary_hypoxia, :secondary_epo
"""
function create_polycythemia_state(poly_type::Symbol; hematocrit::Float64=0.0)
    if !haskey(POLYCYTHEMIA_PROFILES, poly_type)
        error("Unknown polycythemia type: $poly_type")
    end

    profile = POLYCYTHEMIA_PROFILES[poly_type]

    hct = hematocrit > 0 ? hematocrit : profile[:typical_hct]
    hb = hct / 0.03  # Approximate Hb from Hct

    state = HematologicalState(
        hematocrit = hct,
        hemoglobin = hb,
        condition = Symbol("polycythemia_", poly_type),
        severity = hct > 0.55 ? :severe : (hct > 0.50 ? :moderate : :mild)
    )

    viscosity_factor = profile[:viscosity_factor]
    hepatic_reduction = profile[:hepatic_flow_reduction]

    return PolycythemiaProfile(
        state,
        viscosity_factor,
        hepatic_reduction,
        1.0 - hepatic_reduction,
        get(profile, :thrombosis_risk, :moderate),
        get(profile, :special_considerations, String[])
    )
end

# ============================================================================
# BLOOD-PLASMA RATIO CALCULATIONS
# ============================================================================

"""
    calculate_blood_plasma_ratio(hematocrit::Float64, ke_p::Float64)

Calculate blood-to-plasma concentration ratio (Rb).

Rb = 1 - Hct + (Hct × Ke_p)

Where:
- Hct = hematocrit
- Ke_p = RBC-to-plasma partition coefficient

# Example
```julia
rb = calculate_blood_plasma_ratio(0.42, 35.0)  # Tacrolimus
# Returns ~15.7
```
"""
function calculate_blood_plasma_ratio(hematocrit::Float64, ke_p::Float64)
    return 1.0 - hematocrit + (hematocrit * ke_p)
end

"""
    calculate_blood_plasma_ratio(state::HematologicalState, drug::String)

Calculate Rb using drug database.
"""
function calculate_blood_plasma_ratio(state::HematologicalState, drug::String)
    drug_lower = lowercase(drug)
    if !haskey(RBC_PARTITION_DATABASE, drug_lower)
        @warn "Drug $drug not in RBC partition database, using Ke_p=1.0"
        ke_p = 1.0
    else
        ke_p = RBC_PARTITION_DATABASE[drug_lower][:ke_p]
    end

    return calculate_blood_plasma_ratio(state.hematocrit, ke_p)
end

"""
    calculate_rbc_partitioning(drug::String, concentration::Float64, hematocrit::Float64)

Calculate RBC and plasma concentrations from blood concentration.

Returns Dict with :c_blood, :c_plasma, :c_rbc, :rb
"""
function calculate_rbc_partitioning(drug::String, c_blood::Float64, hematocrit::Float64)
    drug_lower = lowercase(drug)

    if !haskey(RBC_PARTITION_DATABASE, drug_lower)
        ke_p = 1.0
        saturable = false
    else
        drug_data = RBC_PARTITION_DATABASE[drug_lower]
        ke_p = drug_data[:ke_p]
        saturable = get(drug_data, :binding_saturable, false)
    end

    rb = calculate_blood_plasma_ratio(hematocrit, ke_p)
    c_plasma = c_blood / rb
    c_rbc = c_plasma * ke_p

    return Dict(
        :c_blood => c_blood,
        :c_plasma => c_plasma,
        :c_rbc => c_rbc,
        :rb => rb,
        :ke_p => ke_p,
        :hematocrit => hematocrit
    )
end

# ============================================================================
# HEMATOCRIT CORRECTIONS
# ============================================================================

"""
    calculate_hematocrit_correction(measured_conc::Float64,
                                     patient_hct::Float64,
                                     reference_hct::Float64,
                                     ke_p::Float64)

Standardize drug concentration to reference hematocrit.

Used for TDM of tacrolimus, cyclosporine, sirolimus.

C_standardized = C_measured × (Rb_reference / Rb_patient)

# Example (Tacrolimus TDM)
```julia
c_std = calculate_hematocrit_correction(8.0, 0.30, 0.45, 35.0)
# Standardizes to Hct=0.45 for comparison
```
"""
function calculate_hematocrit_correction(measured_conc::Float64,
                                         patient_hct::Float64,
                                         reference_hct::Float64,
                                         ke_p::Float64)
    rb_patient = calculate_blood_plasma_ratio(patient_hct, ke_p)
    rb_reference = calculate_blood_plasma_ratio(reference_hct, ke_p)

    return measured_conc * (rb_reference / rb_patient)
end

"""
    calculate_vd_correction(vd_reference::Float64,
                            patient_hct::Float64,
                            reference_hct::Float64,
                            ke_p::Float64)

Correct volume of distribution for hematocrit differences.

For drugs with high RBC partitioning, Vd changes with hematocrit.
"""
function calculate_vd_correction(vd_reference::Float64,
                                 patient_hct::Float64,
                                 reference_hct::Float64,
                                 ke_p::Float64)
    if ke_p < 2.0
        # Low RBC partitioning - minimal Vd effect
        return vd_reference
    end

    # Vd correction based on blood volume distribution
    rb_patient = calculate_blood_plasma_ratio(patient_hct, ke_p)
    rb_reference = calculate_blood_plasma_ratio(reference_hct, ke_p)

    # Empirical correction factor
    alpha = 0.3  # Sensitivity coefficient
    correction = (patient_hct / reference_hct) ^ alpha * (rb_patient / rb_reference)

    return vd_reference * correction
end

"""
    calculate_clearance_correction(cl_reference::Float64,
                                   patient_hct::Float64,
                                   reference_hct::Float64;
                                   extraction_ratio::Float64=0.5)

Correct clearance for hematocrit-dependent changes.

Accounts for:
- Blood flow changes (viscosity)
- Protein binding in plasma
- RBC-bound drug unavailability
"""
function calculate_clearance_correction(cl_reference::Float64,
                                        patient_hct::Float64,
                                        reference_hct::Float64;
                                        extraction_ratio::Float64=0.5)
    # High extraction drugs: flow-limited (viscosity effect)
    # Low extraction drugs: binding/intrinsic CL limited

    if extraction_ratio > 0.7
        # High extraction - flow limited
        # Viscosity increases with hematocrit (exponential)
        viscosity_ratio = exp(2.5 * (patient_hct - reference_hct))
        flow_correction = 1.0 / viscosity_ratio
        return cl_reference * flow_correction
    else
        # Low extraction - less affected
        beta = 0.5 * (1.0 - extraction_ratio)
        correction = (reference_hct / patient_hct) ^ beta
        return cl_reference * correction
    end
end

# ============================================================================
# ANEMIA PK ADJUSTMENTS
# ============================================================================

"""
    apply_anemia_pk_adjustments(profile::AnemiaProfile, drug_params::Dict)

Apply comprehensive anemia PK adjustments to drug parameters.

# Arguments
- `profile`: AnemiaProfile from create_anemia_state
- `drug_params`: Dict with :vd, :clearance, :fu, :bioavailability, :ke_p

# Returns
Dict with adjusted parameters and rationale.
"""
function apply_anemia_pk_adjustments(profile::AnemiaProfile, drug_params::Dict)
    hct = profile.state.hematocrit
    reference_hct = NORMAL_HEMATOLOGY[:hematocrit]

    # Get drug properties
    vd = get(drug_params, :vd, 1.0)
    clearance = get(drug_params, :clearance, 1.0)
    fu = get(drug_params, :fu, 0.5)
    bioavailability = get(drug_params, :bioavailability, 1.0)
    ke_p = get(drug_params, :ke_p, 1.0)
    extraction_ratio = get(drug_params, :extraction_ratio, 0.5)

    # Calculate Rb ratio
    rb_patient = calculate_blood_plasma_ratio(hct, ke_p)
    rb_reference = calculate_blood_plasma_ratio(reference_hct, ke_p)

    # Adjust Vd
    vd_adjusted = vd * profile.vd_adjustment
    if ke_p > 2.0
        # Additional Hct correction for high RBC partitioning
        vd_adjusted *= (rb_patient / rb_reference)
    end

    # Adjust clearance
    cl_adjusted = clearance * profile.clearance_adjustment * profile.cyp_activity
    if profile.renal_function < 1.0
        # Weight hepatic vs renal contribution
        renal_fraction = get(drug_params, :renal_fraction, 0.3)
        cl_adjusted *= (1.0 - renal_fraction) + (renal_fraction * profile.renal_function)
    end

    # Adjust fu (protein binding)
    fu_adjusted = fu / profile.protein_binding_factor
    fu_adjusted = min(fu_adjusted, 1.0)  # Cap at 100%

    # Adjust bioavailability
    f_adjusted = bioavailability * profile.absorption_factor

    # Reticulocyte effect on transport
    retic_effect = 1.0
    if profile.state.reticulocytes.fraction > 0.05
        # High reticulocyte count enhances some drug transport
        retic_effect = 1.0 + (profile.state.reticulocytes.fraction - 0.01) *
                       profile.state.reticulocytes.transport_enhancement
    end

    return Dict(
        :vd_adjusted => vd_adjusted,
        :clearance_adjusted => cl_adjusted,
        :fu_adjusted => fu_adjusted,
        :bioavailability_adjusted => f_adjusted,
        :rb_patient => rb_patient,
        :rb_reference => rb_reference,
        :reticulocyte_effect => retic_effect,
        :anemia_type => profile.state.condition,
        :severity => profile.state.severity,
        :hematocrit => hct,
        :considerations => profile.special_considerations,
        :adjustment_summary => Dict(
            :vd_fold => vd_adjusted / vd,
            :cl_fold => cl_adjusted / clearance,
            :fu_fold => fu_adjusted / fu,
            :f_fold => f_adjusted / bioavailability
        )
    )
end

# ============================================================================
# POLYCYTHEMIA PK ADJUSTMENTS
# ============================================================================

"""
    apply_polycythemia_pk_adjustments(profile::PolycythemiaProfile, drug_params::Dict)

Apply polycythemia PK adjustments focusing on viscosity and flow effects.
"""
function apply_polycythemia_pk_adjustments(profile::PolycythemiaProfile, drug_params::Dict)
    hct = profile.state.hematocrit
    reference_hct = NORMAL_HEMATOLOGY[:hematocrit]

    vd = get(drug_params, :vd, 1.0)
    clearance = get(drug_params, :clearance, 1.0)
    ke_p = get(drug_params, :ke_p, 1.0)
    extraction_ratio = get(drug_params, :extraction_ratio, 0.5)

    # Calculate Rb
    rb_patient = calculate_blood_plasma_ratio(hct, ke_p)
    rb_reference = calculate_blood_plasma_ratio(reference_hct, ke_p)

    # Viscosity-adjusted clearance
    # High viscosity reduces hepatic blood flow
    hepatic_flow_factor = profile.tissue_perfusion_factor

    # For high extraction drugs, clearance is flow-limited
    if extraction_ratio > 0.7
        cl_adjusted = clearance * hepatic_flow_factor
    else
        # Low extraction - less affected by flow
        intrinsic_cl_factor = 1.0 - (profile.viscosity_factor - 1.0) * 0.1
        cl_adjusted = clearance * intrinsic_cl_factor
    end

    # Vd may decrease due to reduced tissue perfusion
    vd_adjusted = vd * (1.0 - (1.0 - profile.tissue_perfusion_factor) * 0.3)

    # RBC partitioning effect
    if ke_p > 2.0
        # More drug in blood compartment
        vd_adjusted *= (rb_patient / rb_reference)
    end

    return Dict(
        :vd_adjusted => vd_adjusted,
        :clearance_adjusted => cl_adjusted,
        :rb_patient => rb_patient,
        :rb_reference => rb_reference,
        :viscosity_factor => profile.viscosity_factor,
        :hepatic_flow_factor => hepatic_flow_factor,
        :polycythemia_type => profile.state.condition,
        :severity => profile.state.severity,
        :hematocrit => hct,
        :thrombosis_risk => profile.thrombosis_risk,
        :considerations => profile.special_considerations,
        :adjustment_summary => Dict(
            :vd_fold => vd_adjusted / vd,
            :cl_fold => cl_adjusted / clearance
        )
    )
end

# ============================================================================
# EPO THERAPY SIMULATION
# ============================================================================

"""
    simulate_epo_therapy(baseline_hct::Float64,
                         dose_units_per_week::Float64,
                         weeks::Int;
                         target_hct::Float64=0.35)

Simulate hematocrit response to EPO therapy.

# Returns
Dict with weekly hematocrit values and PK implications.
"""
function simulate_epo_therapy(baseline_hct::Float64,
                              dose_units_per_week::Float64,
                              weeks::Int;
                              target_hct::Float64=0.35)
    # EPO dose-response: ~1 g/dL Hb increase per 4 weeks at 100 U/kg/week
    # Hct ≈ 3 × Hb

    standard_dose = 10000.0  # units/week reference
    dose_factor = dose_units_per_week / standard_dose

    # Weekly hematocrit increase (logistic approach to target)
    weekly_increase = 0.01 * dose_factor  # ~1% per week at standard dose

    hct_values = Float64[]
    current_hct = baseline_hct

    for week in 1:weeks
        # Logistic slowing as approaching target
        distance_to_target = target_hct - current_hct
        if distance_to_target > 0
            increase = weekly_increase * (distance_to_target / (target_hct - baseline_hct))
            current_hct += increase
            current_hct = min(current_hct, target_hct)
        end
        push!(hct_values, current_hct)
    end

    return Dict(
        :baseline_hct => baseline_hct,
        :target_hct => target_hct,
        :dose_units_per_week => dose_units_per_week,
        :weekly_hct => hct_values,
        :final_hct => hct_values[end],
        :weeks_to_target => findfirst(h -> h >= target_hct * 0.95, hct_values),
        :pk_implications => Dict(
            :rb_change_expected => "Increases as Hct rises for drugs with Ke_p > 1",
            :dose_adjustment => "May need reduction as Hct normalizes",
            :monitoring => "Weekly CBC, adjust drug doses accordingly"
        )
    )
end

# ============================================================================
# TRANSFUSION EFFECTS
# ============================================================================

"""
    calculate_transfusion_effect(pre_hct::Float64,
                                  units_prbc::Int;
                                  patient_blood_volume::Float64=5.0)

Calculate post-transfusion hematocrit and PK implications.

Rule of thumb: 1 unit PRBC raises Hct by 3% in 70kg adult.

# Returns
Dict with post-transfusion state and drug adjustments needed.
"""
function calculate_transfusion_effect(pre_hct::Float64,
                                      units_prbc::Int;
                                      patient_blood_volume::Float64=5.0)
    # Each unit is ~300mL with Hct ~0.70
    prbc_volume = 0.3  # L per unit
    prbc_hct = 0.70

    # Mass balance
    pre_rbc_volume = pre_hct * patient_blood_volume
    added_rbc_volume = units_prbc * prbc_volume * prbc_hct

    new_blood_volume = patient_blood_volume + (units_prbc * prbc_volume * 0.3)  # ~30% stays
    new_rbc_volume = pre_rbc_volume + added_rbc_volume

    post_hct = new_rbc_volume / new_blood_volume
    post_hct = min(post_hct, 0.55)  # Cap

    delta_hct = post_hct - pre_hct

    return Dict(
        :pre_hct => pre_hct,
        :post_hct => post_hct,
        :delta_hct => delta_hct,
        :units_transfused => units_prbc,
        :pk_implications => Dict(
            :vd_change => "May decrease for high Ke_p drugs",
            :rb_change => "Increases for drugs with RBC binding",
            :timing => "Recheck drug levels 24h post-transfusion",
            :tacrolimus => "Level may appear lower if measured as blood conc"
        ),
        :monitoring_recommendations => [
            "Recheck drug levels 24h post-transfusion",
            "For tacrolimus/cyclosporine: use standardized Hct correction",
            "Consider dose reduction if high Ke_p drug levels increase"
        ]
    )
end

# ============================================================================
# RETICULOCYTE EFFECTS
# ============================================================================

"""
    estimate_reticulocyte_effect(reticulocyte_fraction::Float64,
                                  transporter::Symbol)

Estimate enhanced drug transport due to high reticulocyte count.

Young RBCs have more active transporters:
- GLUT1: 2-3× higher in reticulocytes
- Amino acid transporters: 3× higher in SCD reticulocytes
"""
function estimate_reticulocyte_effect(reticulocyte_fraction::Float64,
                                      transporter::Symbol)
    normal_fraction = 0.01

    if reticulocyte_fraction <= normal_fraction
        return 1.0
    end

    # Enhancement factors by transporter
    enhancements = Dict(
        :GLUT1 => 2.5,
        :ENT1 => 2.0,
        :amino_acid => 3.0,
        :general => 1.5
    )

    enhancement = get(enhancements, transporter, 1.5)

    # Weighted average based on reticulocyte fraction
    excess_fraction = reticulocyte_fraction - normal_fraction
    effect = 1.0 + (excess_fraction * enhancement)

    return effect
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
    get_anemia_severity(hematocrit::Float64, hemoglobin::Float64)

Classify anemia severity based on WHO criteria.
"""
function get_anemia_severity(hematocrit::Float64, hemoglobin::Float64)
    # WHO criteria (Hb in g/dL)
    if hemoglobin >= 12.0
        return :none
    elseif hemoglobin >= 11.0
        return :mild
    elseif hemoglobin >= 8.0
        return :moderate
    else
        return :severe
    end
end

"""
    list_supported_conditions()

List all supported anemia and polycythemia types.
"""
function list_supported_conditions()
    return Dict(
        :anemia_types => collect(keys(ANEMIA_PROFILES)),
        :polycythemia_types => collect(keys(POLYCYTHEMIA_PROFILES)),
        :drugs_with_rbc_data => collect(keys(RBC_PARTITION_DATABASE))
    )
end

end # module AnemiaPolycythemia
