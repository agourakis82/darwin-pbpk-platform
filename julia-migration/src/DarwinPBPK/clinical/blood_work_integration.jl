"""
Blood Work Integration for Personalized PBPK Modeling

This module translates clinical laboratory values into PBPK model parameters,
enabling personalized drug dosing based on individual patient data.

CLINICAL PARAMETERS INTEGRATED:
===============================

1. PROTEIN BINDING PARAMETERS
   - Albumin (g/dL) → fup adjustment for acidic drugs
   - α1-acid glycoprotein (mg/dL) → fup adjustment for basic drugs
   - Total protein (g/dL) → overall binding capacity

2. HEPATIC FUNCTION MARKERS
   - ALT/AST (U/L) → functional hepatocyte mass
   - Bilirubin (mg/dL) → OATP function, cholestasis
   - Alkaline phosphatase (U/L) → biliary function
   - GGT (U/L) → enzyme induction status
   - INR/PT → synthetic function, CYP2C9 status

3. RENAL FUNCTION MARKERS
   - Creatinine (mg/dL) → GFR estimation
   - eGFR (mL/min/1.73m²) → renal clearance adjustment
   - BUN (mg/dL) → nitrogen balance

4. HEMATOLOGICAL PARAMETERS
   - Hematocrit (%) → blood:plasma ratio
   - Platelets (K/μL) → hepatic synthetic function
   - RBC count → drug distribution to red cells

5. LIPID PANEL
   - Total cholesterol → lipoprotein binding capacity
   - LDL/HDL → specific lipoprotein fractions
   - Triglycerides → VLDL capacity

6. INFLAMMATORY MARKERS
   - CRP (mg/L) → acute phase response (↓ albumin, ↑ AGP)
   - ESR → chronic inflammation

CLINICAL RELEVANCE:
==================

1. HYPOALBUMINEMIA (cirrhosis, nephrotic syndrome, malnutrition)
   - ↑ free drug fraction for albumin-bound drugs
   - ↑ Vdss, ↑ clearance of unbound drug
   - May need dose REDUCTION despite lower total levels

2. ELEVATED AGP (inflammation, cancer, post-surgery)
   - ↓ free drug fraction for basic drugs
   - ↓ Vdss, ↓ clearance
   - May need dose INCREASE

3. HEPATIC IMPAIRMENT (Child-Pugh, MELD)
   - ↓ CYP450 activity, ↓ transporter function
   - ↓ albumin synthesis, ↑ bilirubin
   - Complex effects on drug disposition

4. RENAL IMPAIRMENT
   - ↓ renal clearance (obvious)
   - Also: ↑ free fraction (uremic toxins displace drugs)
   - ↓ hepatic CYP3A4 in severe CKD

References:
- Rowland & Tozer: Clinical Pharmacokinetics
- FDA Guidance: Pharmacokinetics in Patients with Impaired Hepatic Function
- FDA Guidance: Pharmacokinetics in Patients with Impaired Renal Function

Author: Darwin PBPK Platform
Date: November 2025
"""

module BloodWorkIntegration

using Statistics

export PatientBloodWork, calculate_personalized_parameters
export adjust_fup_for_albumin, adjust_fup_for_agp
export estimate_hepatic_function, estimate_renal_function
export ChildPughScore, MELDScore
export PersonalizedPBPKParameters, personalize_fup, clinical_summary
export PersonalizedPBPKParameters

# Reference values for normal adults
const REFERENCE_VALUES = Dict{Symbol,Tuple{Float64,Float64,Float64}}(
    # Parameter => (low_normal, reference, high_normal)
    :albumin_gdL => (3.5, 4.0, 5.0),
    :AGP_mgdL => (50.0, 80.0, 120.0),
    :total_protein_gdL => (6.0, 7.0, 8.0),
    :ALT_UL => (7.0, 25.0, 56.0),
    :AST_UL => (10.0, 25.0, 40.0),
    :ALP_UL => (44.0, 100.0, 147.0),
    :GGT_UL => (9.0, 30.0, 48.0),
    :bilirubin_total_mgdL => (0.1, 0.7, 1.2),
    :bilirubin_direct_mgdL => (0.0, 0.1, 0.3),
    :INR => (0.8, 1.0, 1.2),
    :creatinine_mgdL => (0.7, 1.0, 1.3),
    :BUN_mgdL => (7.0, 15.0, 20.0),
    :eGFR => (90.0, 120.0, 150.0),
    :hematocrit_pct => (36.0, 42.0, 50.0),
    :platelets_K => (150.0, 250.0, 400.0),
    :WBC_K => (4.5, 7.5, 11.0),
    :cholesterol_mgdL => (0.0, 180.0, 200.0),
    :LDL_mgdL => (0.0, 100.0, 130.0),
    :HDL_mgdL => (40.0, 55.0, 80.0),
    :triglycerides_mgdL => (0.0, 100.0, 150.0),
    :CRP_mgL => (0.0, 1.0, 3.0),
)

"""
Patient blood work results.

All values are optional - missing values use population defaults.
"""
Base.@kwdef struct PatientBloodWork
    # Demographics
    age_years::Union{Float64,Nothing} = nothing
    weight_kg::Union{Float64,Nothing} = nothing
    height_cm::Union{Float64,Nothing} = nothing
    sex::Union{Symbol,Nothing} = nothing  # :male, :female

    # Protein binding
    albumin_gdL::Union{Float64,Nothing} = nothing
    AGP_mgdL::Union{Float64,Nothing} = nothing
    total_protein_gdL::Union{Float64,Nothing} = nothing

    # Liver function
    ALT_UL::Union{Float64,Nothing} = nothing
    AST_UL::Union{Float64,Nothing} = nothing
    ALP_UL::Union{Float64,Nothing} = nothing
    GGT_UL::Union{Float64,Nothing} = nothing
    bilirubin_total_mgdL::Union{Float64,Nothing} = nothing
    bilirubin_direct_mgdL::Union{Float64,Nothing} = nothing
    INR::Union{Float64,Nothing} = nothing

    # Renal function
    creatinine_mgdL::Union{Float64,Nothing} = nothing
    BUN_mgdL::Union{Float64,Nothing} = nothing
    eGFR::Union{Float64,Nothing} = nothing

    # Hematology
    hematocrit_pct::Union{Float64,Nothing} = nothing
    hemoglobin_gdL::Union{Float64,Nothing} = nothing
    platelets_K::Union{Float64,Nothing} = nothing
    WBC_K::Union{Float64,Nothing} = nothing
    RBC_M::Union{Float64,Nothing} = nothing

    # Lipids
    cholesterol_mgdL::Union{Float64,Nothing} = nothing
    LDL_mgdL::Union{Float64,Nothing} = nothing
    HDL_mgdL::Union{Float64,Nothing} = nothing
    triglycerides_mgdL::Union{Float64,Nothing} = nothing

    # Inflammation
    CRP_mgL::Union{Float64,Nothing} = nothing
    ESR_mmhr::Union{Float64,Nothing} = nothing

    # Special tests
    prealbumin_mgdL::Union{Float64,Nothing} = nothing
    transferrin_mgdL::Union{Float64,Nothing} = nothing

    # Genetic markers (if available)
    CYP2D6_phenotype::Union{Symbol,Nothing} = nothing  # :PM, :IM, :EM, :UM
    CYP2C19_phenotype::Union{Symbol,Nothing} = nothing
    CYP2C9_genotype::Union{Symbol,Nothing} = nothing
    SLCO1B1_genotype::Union{Symbol,Nothing} = nothing  # :WT, :*5, :*15
    UGT1A1_genotype::Union{Symbol,Nothing} = nothing   # :WT, :*28
end

"""
Personalized PBPK parameters derived from blood work.
"""
struct PersonalizedPBPKParameters
    # Adjusted binding
    fup_acid_multiplier::Float64      # Multiply reference fup for acids
    fup_base_multiplier::Float64      # Multiply reference fup for bases
    fup_neutral_multiplier::Float64   # Multiply reference fup for neutrals

    # Hepatic function
    hepatic_function_fraction::Float64  # 0-1, relative to normal
    CYP_activity_multiplier::Float64    # Overall CYP activity
    transporter_function::Float64       # OATP, P-gp function (0-1)

    # Renal function
    renal_function_fraction::Float64    # Based on eGFR

    # Volume adjustments
    plasma_volume_L::Float64
    blood_plasma_ratio::Float64

    # Lipoprotein binding
    lipoprotein_capacity::Float64       # Relative to normal

    # Inflammation effects
    acute_phase_response::Float64       # 0 = none, 1 = severe

    # Genetic effects
    CYP2D6_activity::Float64
    CYP2C19_activity::Float64
    CYP2C9_activity::Float64
    OATP1B1_activity::Float64
    UGT1A1_activity::Float64

    # Clinical scores
    child_pugh_score::Int
    meld_score::Float64
end

"""
Adjust fraction unbound in plasma (fup) for albumin-bound drugs (acids).

HYPOALBUMINEMIA:
- Cirrhosis, nephrotic syndrome, malnutrition
- fup increases proportionally to albumin decrease
- Clinical impact: may need LOWER doses

HYPERALBUMINEMIA (rare):
- Dehydration
- fup decreases

Equation (linear binding model):
fup_adjusted = 1 - (1 - fup_ref) × (albumin_patient / albumin_ref)
"""
function adjust_fup_for_albumin(;
    fup_reference::Float64,
    albumin_gdL::Float64,
    albumin_reference::Float64 = 4.0
)::Float64
    # Bound fraction scales with albumin
    bound_ref = 1 - fup_reference
    albumin_ratio = albumin_gdL / albumin_reference

    # Adjust bound fraction
    bound_adjusted = bound_ref * albumin_ratio

    # Calculate adjusted fup
    fup_adjusted = 1 - bound_adjusted

    # Ensure valid range
    return clamp(fup_adjusted, 0.001, 0.999)
end

"""
Adjust fup for α1-acid glycoprotein bound drugs (bases).

AGP is an acute phase protein:
- ELEVATED in: inflammation, infection, cancer, surgery, MI
- DECREASED in: hepatic failure, nephrotic syndrome

Basic drugs (lidocaine, propranolol, imipramine) bind primarily to AGP.

Clinical impact:
- High AGP → lower fup → may need HIGHER doses for effect
- Low AGP → higher fup → risk of toxicity
"""
function adjust_fup_for_agp(;
    fup_reference::Float64,
    AGP_mgdL::Float64,
    AGP_reference::Float64 = 80.0
)::Float64
    bound_ref = 1 - fup_reference
    agp_ratio = AGP_mgdL / AGP_reference

    # AGP binding is saturable at high concentrations
    # Use Langmuir-type adjustment
    bound_adjusted = bound_ref * agp_ratio / (1 + 0.2 * (agp_ratio - 1))

    fup_adjusted = 1 - bound_adjusted

    return clamp(fup_adjusted, 0.001, 0.999)
end

"""
Estimate hepatic function from liver panel.

Returns fraction of normal hepatic function (0-1).

Components:
1. Synthetic function: albumin, INR, platelets
2. Hepatocellular injury: ALT, AST
3. Cholestasis: bilirubin, ALP, GGT
"""
function estimate_hepatic_function(bw::PatientBloodWork)::Float64
    scores = Float64[]

    # Synthetic function markers
    if bw.albumin_gdL !== nothing
        albumin_score = clamp(bw.albumin_gdL / 3.5, 0.0, 1.2)
        push!(scores, albumin_score)
    end

    if bw.INR !== nothing
        # INR > 2.5 indicates severe impairment
        inr_score = bw.INR <= 1.2 ? 1.0 : max(0.2, 1.5 - 0.3 * bw.INR)
        push!(scores, inr_score)
    end

    if bw.platelets_K !== nothing
        # Thrombocytopenia in liver disease
        plt_score = clamp(bw.platelets_K / 150.0, 0.3, 1.0)
        push!(scores, plt_score)
    end

    # Hepatocellular injury
    if bw.ALT_UL !== nothing
        # Mild elevation (< 3x ULN): minimal impact
        # Moderate (3-10x): some impact
        # Severe (> 10x): significant impact
        alt_ratio = bw.ALT_UL / 56.0  # ULN
        alt_score = if alt_ratio <= 1
            1.0
        elseif alt_ratio <= 3
            0.95
        elseif alt_ratio <= 10
            0.85 - 0.05 * (alt_ratio - 3) / 7
        else
            0.5
        end
        push!(scores, alt_score)
    end

    # Cholestasis markers
    if bw.bilirubin_total_mgdL !== nothing
        # High bilirubin indicates impaired conjugation/excretion
        # Also competes for OATP transport
        bili_score = if bw.bilirubin_total_mgdL <= 1.2
            1.0
        elseif bw.bilirubin_total_mgdL <= 3.0
            0.9 - 0.1 * (bw.bilirubin_total_mgdL - 1.2) / 1.8
        elseif bw.bilirubin_total_mgdL <= 10.0
            0.7 - 0.2 * (bw.bilirubin_total_mgdL - 3.0) / 7.0
        else
            0.4
        end
        push!(scores, bili_score)
    end

    # Average all available scores
    if isempty(scores)
        return 1.0  # Assume normal if no data
    end

    return mean(scores)
end

"""
Calculate Child-Pugh score for hepatic impairment.

Components (each 1-3 points):
- Bilirubin
- Albumin
- INR (PT prolongation)
- Ascites (not in blood work)
- Encephalopathy (not in blood work)

Score interpretation:
- 5-6: Class A (mild)
- 7-9: Class B (moderate)
- 10-15: Class C (severe)

Returns score (5-15) with assumptions for missing clinical data.
"""
function ChildPughScore(bw::PatientBloodWork;
                        ascites::Symbol = :none,      # :none, :mild, :moderate
                        encephalopathy::Symbol = :none # :none, :mild, :severe
                       )::Int
    score = 0

    # Bilirubin (mg/dL)
    bili = something(bw.bilirubin_total_mgdL, 1.0)
    score += bili < 2 ? 1 : (bili < 3 ? 2 : 3)

    # Albumin (g/dL)
    alb = something(bw.albumin_gdL, 4.0)
    score += alb > 3.5 ? 1 : (alb > 2.8 ? 2 : 3)

    # INR
    inr = something(bw.INR, 1.0)
    score += inr < 1.7 ? 1 : (inr < 2.3 ? 2 : 3)

    # Ascites
    score += ascites == :none ? 1 : (ascites == :mild ? 2 : 3)

    # Encephalopathy
    score += encephalopathy == :none ? 1 : (encephalopathy == :mild ? 2 : 3)

    return score
end

"""
Calculate MELD score (Model for End-Stage Liver Disease).

MELD = 3.78 × ln(bilirubin) + 11.2 × ln(INR) + 9.57 × ln(creatinine) + 6.43

Used for liver transplant prioritization and mortality prediction.
"""
function MELDScore(bw::PatientBloodWork)::Float64
    bili = max(1.0, something(bw.bilirubin_total_mgdL, 1.0))
    inr = max(1.0, something(bw.INR, 1.0))
    cr = max(1.0, min(4.0, something(bw.creatinine_mgdL, 1.0)))

    meld = 3.78 * log(bili) + 11.2 * log(inr) + 9.57 * log(cr) + 6.43

    return clamp(meld, 6.0, 40.0)
end

"""
Estimate renal function from creatinine/eGFR.

Returns fraction of normal renal function (0-1).
"""
function estimate_renal_function(bw::PatientBloodWork)::Float64
    # Use eGFR if available
    if bw.eGFR !== nothing
        # Normal eGFR ~120 mL/min/1.73m²
        return clamp(bw.eGFR / 120.0, 0.0, 1.2)
    end

    # Estimate from creatinine using Cockcroft-Gault approximation
    if bw.creatinine_mgdL !== nothing
        age = something(bw.age_years, 50.0)
        weight = something(bw.weight_kg, 70.0)
        is_female = bw.sex == :female

        # Cockcroft-Gault
        crcl = ((140 - age) * weight) / (72 * bw.creatinine_mgdL)
        if is_female
            crcl *= 0.85
        end

        # Normalize to fraction
        return clamp(crcl / 120.0, 0.0, 1.2)
    end

    return 1.0  # Assume normal
end

"""
Estimate transporter function from blood work.

OATP1B1/1B3 function is affected by:
- Bilirubin (competes for transport)
- Genetic polymorphisms
- Drug interactions
"""
function estimate_transporter_function(bw::PatientBloodWork)::Float64
    transporter = 1.0

    # Bilirubin competition
    if bw.bilirubin_total_mgdL !== nothing
        # High bilirubin reduces OATP function
        bili_effect = 1.0 / (1.0 + bw.bilirubin_total_mgdL / 2.0)
        transporter *= bili_effect
    end

    # Genetic effects
    if bw.SLCO1B1_genotype !== nothing
        transporter *= if bw.SLCO1B1_genotype == :WT
            1.0
        elseif bw.SLCO1B1_genotype == Symbol("*5")
            0.5  # Heterozygous
        elseif bw.SLCO1B1_genotype == Symbol("*15")
            0.3  # Homozygous
        else
            1.0
        end
    end

    return transporter
end

"""
Get CYP activity multiplier from pharmacogenomics.
"""
function get_cyp_activity(phenotype::Union{Symbol,Nothing}, default::Float64 = 1.0)::Float64
    if phenotype === nothing
        return default
    end

    return if phenotype == :PM  # Poor metabolizer
        0.1
    elseif phenotype == :IM  # Intermediate
        0.5
    elseif phenotype == :EM  # Extensive (normal)
        1.0
    elseif phenotype == :UM  # Ultra-rapid
        2.0
    else
        default
    end
end

"""
Calculate all personalized PBPK parameters from blood work.

This is the main entry point for integrating blood work into PBPK models.
"""
function calculate_personalized_parameters(bw::PatientBloodWork)::PersonalizedPBPKParameters
    # Binding adjustments
    albumin = something(bw.albumin_gdL, 4.0)
    agp = something(bw.AGP_mgdL, 80.0)

    fup_acid_mult = albumin / 4.0
    fup_base_mult = 80.0 / agp  # Inverse relationship
    fup_neutral_mult = (albumin / 4.0 + 80.0 / agp) / 2  # Mixed binding

    # Hepatic function
    hepatic_frac = estimate_hepatic_function(bw)
    cyp_mult = hepatic_frac * get_cyp_activity(bw.CYP2D6_phenotype)
    transporter = estimate_transporter_function(bw)

    # Renal function
    renal_frac = estimate_renal_function(bw)

    # Volume adjustments
    weight = something(bw.weight_kg, 70.0)
    plasma_volume = 0.04 * weight  # ~4% of body weight

    hct = something(bw.hematocrit_pct, 42.0)
    bp_ratio = 1.0 / (1.0 - hct / 100)

    # Lipoprotein capacity
    chol = something(bw.cholesterol_mgdL, 180.0)
    lipo_capacity = chol / 180.0

    # Inflammation
    crp = something(bw.CRP_mgL, 1.0)
    acute_phase = clamp((crp - 1.0) / 10.0, 0.0, 1.0)  # Scale 0-1

    # Individual CYP activities
    cyp2d6 = get_cyp_activity(bw.CYP2D6_phenotype)
    cyp2c19 = get_cyp_activity(bw.CYP2C19_phenotype)
    cyp2c9 = get_cyp_activity(bw.CYP2C9_genotype)

    # OATP activity
    oatp = if bw.SLCO1B1_genotype === nothing
        1.0
    else
        bw.SLCO1B1_genotype == :WT ? 1.0 :
        bw.SLCO1B1_genotype == Symbol("*5") ? 0.5 : 0.3
    end

    # UGT activity
    ugt = if bw.UGT1A1_genotype === nothing
        1.0
    else
        bw.UGT1A1_genotype == :WT ? 1.0 : 0.5  # *28 reduces activity
    end

    # Clinical scores
    cp_score = ChildPughScore(bw)
    meld = MELDScore(bw)

    return PersonalizedPBPKParameters(
        fup_acid_mult,
        fup_base_mult,
        fup_neutral_mult,
        hepatic_frac,
        cyp_mult,
        transporter,
        renal_frac,
        plasma_volume,
        bp_ratio,
        lipo_capacity,
        acute_phase,
        cyp2d6,
        cyp2c19,
        cyp2c9,
        oatp,
        ugt,
        cp_score,
        meld
    )
end

"""
Apply personalized parameters to adjust drug fup.

Takes reference fup (from population PK) and adjusts for individual patient.
"""
function personalize_fup(;
    fup_reference::Float64,
    drug_type::Symbol,  # :acid, :base, :neutral
    params::PersonalizedPBPKParameters
)::Float64
    multiplier = if drug_type == :acid
        params.fup_acid_multiplier
    elseif drug_type == :base
        params.fup_base_multiplier
    else
        params.fup_neutral_multiplier
    end

    # Adjust fup
    bound_ref = 1 - fup_reference
    bound_adj = bound_ref / multiplier  # Lower albumin = less bound = higher fup

    return clamp(1 - bound_adj, 0.001, 0.999)
end

"""
Adjust hepatic clearance for individual patient.
"""
function personalize_hepatic_clearance(;
    cl_reference::Float64,  # Reference hepatic clearance (L/h)
    extraction_ratio::Float64,
    params::PersonalizedPBPKParameters
)::Float64
    if extraction_ratio > 0.7
        # High extraction: flow-limited
        # Clearance proportional to blood flow (assume unchanged in most patients)
        return cl_reference * params.hepatic_function_fraction
    else
        # Low extraction: capacity-limited
        # Clearance proportional to enzyme activity and free fraction
        return cl_reference * params.CYP_activity_multiplier * params.hepatic_function_fraction
    end
end

"""
Generate clinical recommendation summary.
"""
function clinical_summary(params::PersonalizedPBPKParameters)::String
    lines = String[]

    push!(lines, "═══════════════════════════════════════════════════")
    push!(lines, "       PERSONALIZED PBPK PARAMETER SUMMARY          ")
    push!(lines, "═══════════════════════════════════════════════════")

    # Hepatic function
    hep_status = if params.hepatic_function_fraction > 0.9
        "Normal"
    elseif params.hepatic_function_fraction > 0.7
        "Mild impairment"
    elseif params.hepatic_function_fraction > 0.5
        "Moderate impairment"
    else
        "Severe impairment"
    end
    push!(lines, "\n📊 HEPATIC FUNCTION")
    push!(lines, "   Status: $hep_status ($(round(params.hepatic_function_fraction*100, digits=0))%)")
    push!(lines, "   Child-Pugh: $(params.child_pugh_score) ($(params.child_pugh_score <= 6 ? "A" : params.child_pugh_score <= 9 ? "B" : "C"))")
    push!(lines, "   MELD: $(round(params.meld_score, digits=1))")

    # Renal function
    renal_status = if params.renal_function_fraction > 0.9
        "Normal (Stage 1)"
    elseif params.renal_function_fraction > 0.6
        "Mild (Stage 2)"
    elseif params.renal_function_fraction > 0.3
        "Moderate (Stage 3)"
    elseif params.renal_function_fraction > 0.15
        "Severe (Stage 4)"
    else
        "Kidney failure (Stage 5)"
    end
    push!(lines, "\n🔬 RENAL FUNCTION")
    push!(lines, "   Status: $renal_status ($(round(params.renal_function_fraction*100, digits=0))%)")

    # Protein binding
    push!(lines, "\n💊 PROTEIN BINDING ADJUSTMENTS")
    push!(lines, "   Acidic drugs (albumin): $(round(params.fup_acid_multiplier, digits=2))×")
    push!(lines, "   Basic drugs (AGP): $(round(params.fup_base_multiplier, digits=2))×")

    # Pharmacogenomics
    if params.CYP2D6_activity != 1.0 || params.OATP1B1_activity != 1.0
        push!(lines, "\n🧬 PHARMACOGENOMICS")
        if params.CYP2D6_activity != 1.0
            status = params.CYP2D6_activity < 0.5 ? "Poor/Intermediate" : "Ultra-rapid"
            push!(lines, "   CYP2D6: $status ($(round(params.CYP2D6_activity, digits=1))×)")
        end
        if params.OATP1B1_activity != 1.0
            push!(lines, "   OATP1B1: Reduced ($(round(params.OATP1B1_activity, digits=1))×)")
            push!(lines, "   ⚠️  Consider lower statin doses")
        end
    end

    # Warnings
    warnings = String[]
    if params.hepatic_function_fraction < 0.7
        push!(warnings, "⚠️  Reduce doses of hepatically cleared drugs")
    end
    if params.renal_function_fraction < 0.5
        push!(warnings, "⚠️  Reduce doses of renally cleared drugs")
    end
    if params.fup_acid_multiplier > 1.3
        push!(warnings, "⚠️  Low albumin: monitor for toxicity of acidic drugs")
    end
    if params.acute_phase_response > 0.5
        push!(warnings, "⚠️  Acute inflammation: AGP elevated, basic drug binding ↑")
    end

    if !isempty(warnings)
        push!(lines, "\n⚠️  CLINICAL ALERTS")
        for w in warnings
            push!(lines, "   $w")
        end
    end

    push!(lines, "\n═══════════════════════════════════════════════════")

    return join(lines, "\n")
end

end # module
