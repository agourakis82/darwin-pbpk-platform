# MUSCLE TISSUE COMPARTMENT MODEL
# ================================
#
# Muscle is the LARGEST compartment by volume (~30L in 70kg adult)
# Therefore, even small Kp values have HUGE impact on Vdss!
#
# Unique features:
# - High intracellular water (63%) - largest aqueous reservoir
# - Low lipid content (1% neutral, 0.7% phospholipid)
# - Contains MYOGLOBIN - binds O2 and certain drugs
# - Moderate acidic phospholipids (0.15%) - base binding
# - pH gradient: cytosol 7.0, vs plasma 7.4
#
# Muscle types differ:
# - Skeletal muscle: 40% body weight, voluntary
# - Cardiac muscle: continuous activity, high mitochondria
# - Smooth muscle: organs, blood vessels

module MuscleCompartment

export MuscleProperties, calculate_kp_muscle, calculate_muscle_contribution

"""
Muscle tissue physiological properties

Reference values for 70kg adult:
- Skeletal muscle: ~28-30 L (40% body weight!)
- Blood flow: 0.025-0.05 L/min/kg (increases 20x during exercise)
- pH: 7.0 intracellular (slightly acidic vs plasma 7.4)
"""
struct MuscleProperties
    volume_L::Float64           # Total muscle volume
    blood_flow_L_min::Float64   # Blood flow at rest
    f_neutral_lipid::Float64    # Fraction neutral lipids
    f_phospholipid::Float64     # Fraction neutral phospholipids
    f_acidic_pl::Float64        # Fraction acidic phospholipids
    f_water_iw::Float64         # Fraction intracellular water
    f_water_ew::Float64         # Fraction extracellular water
    albumin_ratio::Float64      # Tissue albumin / plasma albumin
    lipoprotein_ratio::Float64  # Tissue LP / plasma LP
    pH_iw::Float64              # Intracellular pH
end

# Default for 70kg adult
const DEFAULT_MUSCLE = MuscleProperties(
    30.0,    # volume (L) - THE LARGEST COMPARTMENT!
    0.75,    # blood flow at rest (L/min) = 0.025 × 30
    0.010,   # neutral lipids (low!)
    0.0072,  # neutral phospholipids
    0.00153, # acidic phospholipids
    0.630,   # intracellular water (HIGH!)
    0.118,   # extracellular water
    0.064,   # albumin ratio (low tissue albumin)
    0.059,   # lipoprotein ratio
    7.0      # intracellular pH (slightly acidic)
)

"""
Calculate muscle:plasma partition coefficient

Key considerations:
1. Large aqueous volume (63% IW + 12% EW = 75% water)
2. Low lipid content (1%) - lipophilic drugs go elsewhere
3. Acidic pH (7.0) - bases accumulate vs plasma (7.4)
4. Myoglobin binding - affects O2-binding drugs
"""
function calculate_kp_muscle(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    muscle::MuscleProperties = DEFAULT_MUSCLE
)
    P = 10^logP
    D = 10^logD

    # Calculate ionization factors
    pH_p = 7.4   # Plasma pH
    pH_iw = muscle.pH_iw  # Muscle intracellular pH (7.0)

    # X = ionization factor in tissue, Y = in plasma
    X = 0.0
    Y = 0.0

    if !isnothing(pKa)
        if is_base
            X = 10^(pKa - pH_iw)  # Higher in acidic muscle!
            Y = 10^(pKa - pH_p)
        elseif is_acid
            X = 10^(pH_iw - pKa)
            Y = 10^(pH_p - pKa)
        end
    end

    # For bases: muscle (pH 7.0) causes MORE ionization than plasma (pH 7.4)
    # This leads to ION TRAPPING - bases accumulate in muscle!

    # Components of Kp
    f_ew = muscle.f_water_ew
    f_iw = muscle.f_water_iw
    f_nl = muscle.f_neutral_lipid
    f_npl = muscle.f_phospholipid
    f_apl = muscle.f_acidic_pl
    AR = muscle.albumin_ratio
    LR = muscle.lipoprotein_ratio

    # Rodgers-Rowland equation for muscle
    denom = max(1 + Y, 1e-10)

    # Plasma binding constant (simplified, capped)
    Ka_PR = max(0, min((1/fup - 1), 1000))

    # Acidic phospholipid association constant for bases
    # Muscle has moderate acidic PL (0.15%)
    if is_base && !isnothing(pKa) && pKa > 7.0
        Ka_AP = 40.0 * (1 + 0.2 * (pKa - 7.0))
        Ka_AP = min(Ka_AP, 150.0)
    else
        Ka_AP = 10.0
    end

    # Kp calculation depends on drug type
    if is_base && !isnothing(pKa) && pKa > 7.0
        # Strong base: acidic phospholipid binding + ion trapping
        Kp = (f_ew +
              ((1 + X) / denom) * f_iw +
              (P * f_nl + (0.3*P + 0.7) * f_npl) / denom +
              (Ka_AP * f_apl * X) / denom) * fup
    elseif is_acid
        # Acid: albumin binding dominates
        Kp = (f_ew +
              ((1 + X) / denom) * f_iw +
              (P * f_nl + (0.3*P + 0.7) * f_npl) / denom +
              (Ka_PR * AR * X) / denom) * fup
    else
        # Neutral or weak base: lipoprotein binding
        Kp = (f_ew +
              ((1 + X) / denom) * f_iw +
              (P * f_nl + (0.3*P + 0.7) * f_npl) / denom +
              (Ka_PR * LR) / denom) * fup
    end

    return max(Kp, 0.01)
end

"""
Calculate muscle contribution to Vdss

CRITICAL: Muscle is ~50% of Vdss for many drugs due to sheer volume!
"""
function calculate_muscle_contribution(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    muscle_volume::Float64 = 30.0  # L
)
    muscle = MuscleProperties(
        muscle_volume,
        0.025 * muscle_volume,
        0.010, 0.0072, 0.00153, 0.630, 0.118,
        0.064, 0.059, 7.0
    )

    Kp = calculate_kp_muscle(
        logP=logP, logD=logD, fup=fup,
        pKa=pKa, is_base=is_base, is_acid=is_acid,
        muscle=muscle
    )

    contribution = Kp * muscle_volume

    return (Kp=Kp, contribution_L=contribution, volume=muscle_volume)
end

"""
Calculate fraction unbound in muscle (fut_muscle)

Muscle has high water content, so fut is relatively high for most drugs.
Exception: bases with pKa > 7 bind to acidic phospholipids.
"""
function calculate_fut_muscle(logP::Float64;
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false)

    P = 10^logP

    # Simplified fut for muscle
    f_iw = 0.630
    f_nl = 0.010
    f_apl = 0.00153

    # Base case: partition into lipids and water
    lipid_binding = P * f_nl

    # For bases: add acidic phospholipid binding
    apl_binding = 0.0
    if is_base && !isnothing(pKa) && pKa > 7.0
        ionization = 10^(pKa - 7.0)  # At muscle pH
        apl_binding = ionization * f_apl * 10  # Amplified binding
    end

    fut = f_iw / (f_iw + lipid_binding + apl_binding)

    return clamp(fut, 0.001, 0.99)
end

"""
Special considerations for muscle tissue:

1. LARGEST VOLUME
   - 30L in average adult
   - Even Kp = 0.1 contributes 3L to Vdss!
   - Dominates total body distribution

2. pH GRADIENT (Ion Trapping)
   - Muscle pH 7.0 vs plasma pH 7.4
   - Bases (pKa > 7): accumulate in muscle
   - 10^(7.4-7.0) = 2.5x more ionized in muscle
   - Ionized drugs are "trapped"

3. EXERCISE EFFECTS
   - Blood flow increases 20x during exercise
   - Faster drug distribution during activity
   - Important for performance drugs

4. MUSCLE WASTING (Sarcopenia)
   - Elderly: 20-40% muscle loss
   - Cancer cachexia: severe loss
   - Reduces Vdss for muscle-distributed drugs

5. MYOGLOBIN
   - Binds O2 and some drugs
   - May affect local drug concentrations
   - Important for: inhaled anesthetics

6. BODY COMPOSITION
   - Athletes: higher muscle, lower adipose
   - Obese: lower muscle fraction
   - Age: muscle decreases, adipose increases
"""

# Example drugs with significant muscle distribution:
const MUSCLE_DRUGS = Dict(
    "digoxin" => (logP=1.3, Kp_muscle=0.5, note="Cardiac glycoside, muscle reservoir"),
    "aminoglycosides" => (logP=-3.0, Kp_muscle=0.2, note="Hydrophilic, EW distribution"),
    "propranolol" => (logP=3.5, Kp_muscle=2.0, note="Basic, ion trapping"),
    "lidocaine" => (logP=2.4, Kp_muscle=1.5, note="Local anesthetic, base"),
)

end # module
