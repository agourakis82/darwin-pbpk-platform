# INTEGRATED COMPARTMENT-BASED PBPK MODEL
# ========================================
#
# This module integrates all tissue-specific compartment models
# to predict Vdss with maximum physiological accuracy.
#
# KEY INSIGHT: Each compartment is UNIQUE
# - Adipose: olive oil partitioning, 85% triglycerides
# - Muscle: largest volume, pH 7.0, ion trapping
# - Liver: high transporters, first-pass metabolism
# - Brain: BBB, P-gp efflux, restricted access
# - Kidney: highest acidic PL, renal elimination
#
# By modeling each compartment with its unique physiology,
# we can capture drug-specific distribution patterns that
# "one-size-fits-all" models miss.

module IntegratedPBPK

using Statistics

# Include all compartment modules
include("adipose.jl")
include("muscle.jl")
include("liver.jl")
include("brain.jl")
include("kidney.jl")

using .AdiposeCompartment
using .MuscleCompartment
using .LiverCompartment
using .BrainCompartment
using .KidneyCompartment

export DrugProfile, PatientProfile
export calculate_vdss_integrated, calculate_all_kp
export predict_distribution_profile

"""
Drug physicochemical and pharmacological profile

This captures EVERYTHING needed for compartment-specific modeling.
"""
struct DrugProfile
    name::String
    SMILES::String
    MW::Float64
    logP::Float64
    logD::Float64
    TPSA::Float64
    HBA::Int
    HBD::Int
    pKa_acid::Union{Float64, Nothing}
    pKa_base::Union{Float64, Nothing}
    fup::Float64                    # Measured or predicted
    BP::Float64                     # Blood:plasma ratio
    # Transporter substrates
    is_pgp_substrate::Bool
    is_oatp_substrate::Bool
    is_oct_substrate::Bool
    is_oat_substrate::Bool
end

# Constructor with defaults
function DrugProfile(;
    name::String = "Unknown",
    SMILES::String = "",
    MW::Float64 = 400.0,
    logP::Float64 = 2.0,
    logD::Float64 = logP,
    TPSA::Float64 = 60.0,
    HBA::Int = 4,
    HBD::Int = 2,
    pKa_acid::Union{Float64, Nothing} = nothing,
    pKa_base::Union{Float64, Nothing} = nothing,
    fup::Float64 = 0.5,
    BP::Float64 = 1.0,
    is_pgp_substrate::Bool = false,
    is_oatp_substrate::Bool = false,
    is_oct_substrate::Bool = false,
    is_oat_substrate::Bool = false
)
    DrugProfile(name, SMILES, MW, logP, logD, TPSA, HBA, HBD,
                pKa_acid, pKa_base, fup, BP,
                is_pgp_substrate, is_oatp_substrate, is_oct_substrate, is_oat_substrate)
end

"""
Patient-specific parameters

This allows for individual variability in compartment volumes.
Crucial for: obesity, cachexia, elderly, pediatric, etc.
"""
struct PatientProfile
    weight_kg::Float64
    height_cm::Float64
    age_years::Float64
    sex::Symbol  # :male or :female
    # Derived volumes (can be overridden)
    adipose_volume_L::Float64
    muscle_volume_L::Float64
    liver_volume_L::Float64
    brain_volume_L::Float64
    kidney_volume_L::Float64
    plasma_volume_L::Float64
    # Organ function
    GFR_mL_min::Float64
    hepatic_function::Float64  # 0-1 scale
end

# Default 70kg adult
function PatientProfile(;
    weight_kg::Float64 = 70.0,
    height_cm::Float64 = 175.0,
    age_years::Float64 = 35.0,
    sex::Symbol = :male
)
    # Calculate body composition based on demographics
    bmi = weight_kg / (height_cm/100)^2

    # Estimate adipose volume from BMI
    # Lean: 10-15L, Normal: 15-20L, Overweight: 20-30L, Obese: 30-50L
    if bmi < 20
        adipose_L = 10.0 + (bmi - 18) * 2.5
    elseif bmi < 25
        adipose_L = 15.0 + (bmi - 20) * 1.0
    elseif bmi < 30
        adipose_L = 20.0 + (bmi - 25) * 2.0
    else
        adipose_L = 30.0 + (bmi - 30) * 1.5
    end
    adipose_L = clamp(adipose_L, 5.0, 50.0)

    # Muscle volume (scales with lean mass)
    lean_mass_kg = weight_kg - (adipose_L * 0.9)  # Adipose density ~0.9 kg/L
    muscle_L = 0.4 * lean_mass_kg  # ~40% of lean mass

    # Liver (relatively constant)
    liver_L = 1.8 * (weight_kg / 70)^0.7

    # Brain (constant)
    brain_L = 1.4

    # Kidney
    kidney_L = 0.31 * (weight_kg / 70)^0.7

    # Plasma
    plasma_L = 3.0 * (weight_kg / 70)

    # GFR (age-adjusted)
    # GFR declines ~1 mL/min/year after age 40
    base_gfr = 120.0
    if age_years > 40
        base_gfr -= (age_years - 40) * 1.0
    end
    base_gfr = max(base_gfr, 30.0)

    PatientProfile(weight_kg, height_cm, age_years, sex,
                   adipose_L, muscle_L, liver_L, brain_L, kidney_L, plasma_L,
                   base_gfr, 1.0)
end

"""
Calculate Kp for all compartments

Returns a Dict with Kp values and contribution (Kp × Volume) for each tissue.
"""
function calculate_all_kp(drug::DrugProfile, patient::PatientProfile = PatientProfile())

    is_acid = !isnothing(drug.pKa_acid)
    is_base = !isnothing(drug.pKa_base)
    pKa = is_base ? drug.pKa_base : drug.pKa_acid

    results = Dict{String, NamedTuple}()

    # ADIPOSE - uses olive oil partitioning
    kp_adipose = AdiposeCompartment.calculate_kp_adipose(
        logP = drug.logP,
        logD = drug.logD,
        fup = drug.fup,
        pKa = pKa,
        is_base = is_base,
        is_acid = is_acid
    )
    results["adipose"] = (
        Kp = kp_adipose,
        volume = patient.adipose_volume_L,
        contribution = kp_adipose * patient.adipose_volume_L,
        note = "Olive oil partitioning, high variability"
    )

    # MUSCLE - pH gradient, ion trapping for bases
    kp_muscle = MuscleCompartment.calculate_kp_muscle(
        logP = drug.logP,
        logD = drug.logD,
        fup = drug.fup,
        pKa = pKa,
        is_base = is_base,
        is_acid = is_acid
    )
    results["muscle"] = (
        Kp = kp_muscle,
        volume = patient.muscle_volume_L,
        contribution = kp_muscle * patient.muscle_volume_L,
        note = "Largest volume, pH 7.0 ion trapping"
    )

    # LIVER - high transporters
    transporter_effect = 1.0
    if drug.is_oatp_substrate
        transporter_effect *= 3.0
    end
    if drug.is_oct_substrate
        transporter_effect *= 2.0
    end

    kp_liver = LiverCompartment.calculate_kp_liver(
        logP = drug.logP,
        logD = drug.logD,
        fup = drug.fup,
        pKa = pKa,
        is_base = is_base,
        is_acid = is_acid,
        transporter_effect = transporter_effect
    )
    results["liver"] = (
        Kp = kp_liver,
        volume = patient.liver_volume_L,
        contribution = kp_liver * patient.liver_volume_L,
        note = transporter_effect > 1 ? "Transporter-mediated uptake" : "Passive distribution"
    )

    # BRAIN - BBB, P-gp
    kp_brain = BrainCompartment.calculate_kp_brain(
        logP = drug.logP,
        logD = drug.logD,
        fup = drug.fup,
        pKa = pKa,
        is_base = is_base,
        is_acid = is_acid,
        is_pgp_substrate = drug.is_pgp_substrate
    )

    # Check BBB permeability
    bbb_result = BrainCompartment.estimate_bbb_permeability(
        MW = drug.MW,
        logP = drug.logP,
        TPSA = drug.TPSA,
        HBD = drug.HBD,
        is_pgp_substrate = drug.is_pgp_substrate
    )

    if !bbb_result.permeable
        kp_brain *= 0.1  # Reduce if BBB not permeable
    end

    results["brain"] = (
        Kp = kp_brain,
        volume = patient.brain_volume_L,
        contribution = kp_brain * patient.brain_volume_L,
        note = bbb_result.permeable ? "BBB permeable" : "BBB restricted",
        bbb_score = bbb_result.score
    )

    # KIDNEY - highest acidic PL
    transporter_effect_kidney = 1.0
    if drug.is_oat_substrate
        transporter_effect_kidney *= 2.5
    end
    if drug.is_oct_substrate
        transporter_effect_kidney *= 2.0
    end

    kp_kidney = KidneyCompartment.calculate_kp_kidney(
        logP = drug.logP,
        logD = drug.logD,
        fup = drug.fup,
        pKa = pKa,
        is_base = is_base,
        is_acid = is_acid,
        transporter_effect = transporter_effect_kidney
    )
    results["kidney"] = (
        Kp = kp_kidney,
        volume = patient.kidney_volume_L,
        contribution = kp_kidney * patient.kidney_volume_L,
        note = "Highest acidic PL" * (is_base ? " - base accumulation" : "")
    )

    # OTHER TISSUES (simplified - use muscle-like partitioning)
    other_tissues = [
        ("gut", 1.2),
        ("heart", 0.33),
        ("lung", 1.0),
        ("skin", 3.0),
        ("spleen", 0.18),
        ("bone", 4.0),
    ]

    for (name, volume) in other_tissues
        # Use a fraction of muscle Kp (reasonable approximation)
        kp_other = kp_muscle * 0.8
        results[name] = (
            Kp = kp_other,
            volume = volume * (patient.weight_kg / 70),
            contribution = kp_other * volume * (patient.weight_kg / 70),
            note = "Approximated from muscle"
        )
    end

    return results
end

"""
Calculate Vdss using integrated compartment model

Vdss = Vp + Σ(Kp_i × V_i)

Returns detailed breakdown by compartment.
"""
function calculate_vdss_integrated(drug::DrugProfile, patient::PatientProfile = PatientProfile())

    kp_results = calculate_all_kp(drug, patient)

    # Sum up contributions
    Vp = patient.plasma_volume_L
    tissue_contribution = sum(r.contribution for r in values(kp_results))

    Vdss_L = Vp + tissue_contribution
    Vdss_L_kg = Vdss_L / patient.weight_kg

    # Identify dominant compartments
    contributions = [(name, r.contribution) for (name, r) in kp_results]
    sort!(contributions, by=x->x[2], rev=true)

    # Calculate effective fut (volume-weighted)
    total_volume = sum(r.volume for r in values(kp_results))
    weighted_kpu = sum(r.Kp / drug.fup * r.volume for r in values(kp_results))
    fut_effective = total_volume / weighted_kpu

    return (
        Vdss_L = Vdss_L,
        Vdss_L_kg = Vdss_L_kg,
        plasma_L = Vp,
        tissue_contribution_L = tissue_contribution,
        fut_effective = fut_effective,
        kp_results = kp_results,
        dominant_compartments = contributions[1:min(5, length(contributions))],
        patient = patient
    )
end

"""
Predict full distribution profile

Returns comprehensive analysis of drug distribution.
"""
function predict_distribution_profile(drug::DrugProfile, patient::PatientProfile = PatientProfile())

    vdss_result = calculate_vdss_integrated(drug, patient)

    # Determine drug class based on properties
    is_base = !isnothing(drug.pKa_base) && drug.pKa_base > 7
    is_acid = !isnothing(drug.pKa_acid) && drug.pKa_acid < 7
    is_lipophilic = drug.logP > 3
    is_hydrophilic = drug.logP < 0

    # Predict distribution characteristics
    distribution_type = if vdss_result.Vdss_L_kg < 0.2
        "Plasma-confined (highly bound or hydrophilic)"
    elseif vdss_result.Vdss_L_kg < 0.7
        "Extracellular distribution"
    elseif vdss_result.Vdss_L_kg < 2.0
        "Moderate tissue distribution"
    elseif vdss_result.Vdss_L_kg < 10.0
        "Extensive tissue distribution"
    else
        "Very high tissue accumulation"
    end

    # Key insights
    insights = String[]

    if is_base
        push!(insights, "Base (pKa $(drug.pKa_base)): Ion trapping in acidic tissues")
        if drug.pKa_base > 8
            push!(insights, "Strong base: High kidney accumulation risk")
        end
    end

    if is_lipophilic
        push!(insights, "Lipophilic (logP $(drug.logP)): Adipose accumulation, slow release")
    end

    if drug.is_pgp_substrate
        push!(insights, "P-gp substrate: Limited brain penetration, biliary excretion")
    end

    if drug.is_oatp_substrate
        push!(insights, "OATP substrate: Enhanced liver uptake (statins-like)")
    end

    # Calculate fraction in each compartment at steady state
    total = vdss_result.Vdss_L
    fraction_by_tissue = Dict{String, Float64}()
    fraction_by_tissue["plasma"] = vdss_result.plasma_L / total
    for (name, r) in vdss_result.kp_results
        fraction_by_tissue[name] = r.contribution / total
    end

    return (
        drug = drug,
        patient = patient,
        vdss = vdss_result,
        distribution_type = distribution_type,
        insights = insights,
        fraction_by_tissue = fraction_by_tissue
    )
end

"""
Compare prediction methods

Useful for validation and understanding model differences.
"""
function compare_methods(drug::DrugProfile, patient::PatientProfile = PatientProfile())

    # Method 1: Simple Øie-Tozer with empirical fut
    P = 10^drug.logP
    fut_simple = 1 / (1 + 0.05 * clamp(P, 0.001, 1e6))
    fut_simple = clamp(fut_simple, 0.01, 0.99)

    Vp_kg = 0.043
    Ve_kg = 0.15
    Vr_kg = 0.45
    Vdss_simple = Vp_kg + (Ve_kg + Vr_kg) * (drug.fup / fut_simple)

    # Method 2: Integrated compartment model
    result = calculate_vdss_integrated(drug, patient)

    # Method 3: Standard R-R (from tissue_partition.jl)
    # Would need to import and call

    return (
        simple_Vdss = Vdss_simple,
        simple_fut = fut_simple,
        integrated_Vdss = result.Vdss_L_kg,
        integrated_fut = result.fut_effective,
        difference_pct = abs(result.Vdss_L_kg - Vdss_simple) / Vdss_simple * 100,
        dominant = result.dominant_compartments
    )
end

end # module
