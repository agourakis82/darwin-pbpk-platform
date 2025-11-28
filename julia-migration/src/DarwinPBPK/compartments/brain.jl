# BRAIN COMPARTMENT MODEL
# =======================
#
# The brain is UNIQUE due to the BLOOD-BRAIN BARRIER (BBB).
# This is the most selective tissue barrier in the body.
#
# BBB characteristics:
# - Tight junctions between endothelial cells (no paracellular transport)
# - P-glycoprotein (P-gp) efflux - pumps drugs OUT
# - BCRP, MRPs - additional efflux
# - Limited transcytosis
# - Specific influx transporters (LAT1, GLUT1, etc.)
#
# For CNS drugs, BBB penetration is CRITICAL.
# For non-CNS drugs, BBB limits brain exposure (safety!).
#
# Key insight: Brain Kp can be MUCH LOWER than simple
# partitioning predicts due to efflux transporters.

module BrainCompartment

export BrainProperties, calculate_kp_brain, calculate_kpuu_brain
export estimate_bbb_permeability, is_bbb_permeable

"""
Brain physiological properties

Reference values for 70kg adult:
- Volume: 1.3-1.5 L
- Blood flow: 0.7-0.75 L/min (15% cardiac output!)
- Very high lipid content (especially phospholipids)
- Protected by BBB
"""
struct BrainProperties
    volume_L::Float64           # Brain volume
    blood_flow_L_min::Float64   # Cerebral blood flow
    f_neutral_lipid::Float64    # Neutral lipids
    f_phospholipid::Float64     # Neutral phospholipids
    f_acidic_pl::Float64        # Acidic phospholipids
    f_water_iw::Float64         # Intracellular water
    f_water_ew::Float64         # Extracellular/CSF
    albumin_ratio::Float64      # Very low (BBB!)
    pH_iw::Float64              # Intracellular pH
    # BBB transporter expression
    P_gp_relative::Float64      # P-gp expression (relative)
    BCRP_relative::Float64      # BCRP expression (relative)
    LAT1_relative::Float64      # LAT1 for amino acid-like drugs
end

# Default for 70kg adult
const DEFAULT_BRAIN = BrainProperties(
    1.4,     # volume (L)
    0.75,    # blood flow (L/min)
    0.039,   # neutral lipids (higher than most tissues!)
    0.0015,  # neutral phospholipids (low)
    0.0004,  # acidic phospholipids (low)
    0.620,   # intracellular water
    0.162,   # extracellular/CSF water
    0.048,   # albumin ratio (VERY LOW - BBB blocks!)
    7.0,     # intracellular pH
    # BBB transporters (relative to liver)
    3.0,     # P-gp (HIGH expression at BBB!)
    2.5,     # BCRP (HIGH expression)
    1.0      # LAT1
)

"""
Estimate BBB permeability based on physicochemical properties

Rules of thumb for CNS penetration:
- MW < 450 Da (ideally < 400)
- logP: 1-3 (sweet spot)
- TPSA < 90 Å² (ideally < 70)
- HBD ≤ 3
- pKa: weak bases (7-10) better than strong bases

Returns: (permeable::Bool, score::Float64)
"""
function estimate_bbb_permeability(;
    MW::Float64,
    logP::Float64,
    TPSA::Float64,
    HBD::Int = 0,
    HBA::Int = 0,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_pgp_substrate::Bool = false
)
    score = 0.0

    # MW score
    if MW < 400
        score += 2.0
    elseif MW < 450
        score += 1.0
    elseif MW < 500
        score += 0.0
    else
        score -= 2.0  # Very unlikely to cross BBB
    end

    # logP score (optimal: 1.5-2.5)
    if 1.0 <= logP <= 3.0
        score += 2.0
    elseif 0.5 <= logP <= 4.0
        score += 1.0
    elseif logP < 0
        score -= 1.0  # Too polar
    else
        score -= 1.0  # Too lipophilic (P-gp substrate?)
    end

    # TPSA score
    if TPSA < 70
        score += 2.0
    elseif TPSA < 90
        score += 1.0
    elseif TPSA < 120
        score += 0.0
    else
        score -= 2.0  # Too polar for BBB
    end

    # Hydrogen bond donors
    if HBD <= 1
        score += 1.0
    elseif HBD <= 3
        score += 0.0
    else
        score -= 1.0
    end

    # P-gp substrate penalty
    if is_pgp_substrate
        score -= 2.0
    end

    # Ionization (weak bases are better)
    if is_base && !isnothing(pKa)
        if 7.0 <= pKa <= 10.0
            score += 1.0  # Weak base - good
        elseif pKa > 10.0
            score -= 1.0  # Strong base - charged, poor permeability
        end
    end

    # Threshold for BBB permeability
    permeable = score >= 3.0

    return (permeable=permeable, score=score)
end

"""
Simple BBB permeability check using Lipinski-like rules
"""
function is_bbb_permeable(; MW::Float64, logP::Float64, TPSA::Float64, HBD::Int=0)
    return MW < 450 && 0.5 < logP < 4.0 && TPSA < 90 && HBD <= 3
end

"""
Calculate brain:plasma partition coefficient (Kp,brain)

IMPORTANT: This is total Kp including non-specific binding.
For efficacy, you need Kp,uu (unbound brain/unbound plasma).
"""
function calculate_kp_brain(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    is_pgp_substrate::Bool = false,
    brain::BrainProperties = DEFAULT_BRAIN
)
    P = 10^logP
    D = 10^logD

    # Ionization factors
    pH_p = 7.4
    pH_iw = brain.pH_iw

    X = 0.0
    Y = 0.0

    if !isnothing(pKa)
        if is_base
            X = 10^(pKa - pH_iw)
            Y = 10^(pKa - pH_p)
        elseif is_acid
            X = 10^(pH_iw - pKa)
            Y = 10^(pH_p - pKa)
        end
    end

    # Tissue composition
    f_ew = brain.f_water_ew
    f_iw = brain.f_water_iw
    f_nl = brain.f_neutral_lipid
    f_npl = brain.f_phospholipid
    f_apl = brain.f_acidic_pl
    LR = brain.albumin_ratio  # Very low for brain

    denom = max(1 + Y, 1e-10)

    # Brain has high neutral lipids - lipophilic drugs bind
    # But low protein content (BBB blocks albumin)

    # Simplified Kp calculation for brain
    # Note: This assumes drug CAN cross BBB
    Kp = (f_ew +
          ((1 + X) / denom) * f_iw +
          (P * f_nl + (0.3*P + 0.7) * f_npl) / denom) * fup

    # P-gp effect: reduces brain Kp significantly
    if is_pgp_substrate
        # P-gp can reduce brain Kp by 2-50x
        Kp *= 0.2  # Conservative 5x reduction
    end

    # For non-BBB-permeable drugs, Kp is very low
    # (This should be checked separately with estimate_bbb_permeability)

    return max(Kp, 0.001)
end

"""
Calculate unbound brain:plasma ratio (Kp,uu)

This is the PHARMACOLOGICALLY RELEVANT ratio!
Kp,uu = (Cb,u / Cp,u) = Kp × (fup / fub)

Where:
- Cb,u = unbound concentration in brain
- Cp,u = unbound concentration in plasma
- fub = fraction unbound in brain tissue

For CNS drugs:
- Kp,uu > 1: active uptake or trapping
- Kp,uu = 1: passive equilibrium
- Kp,uu < 1: efflux (P-gp) or poor permeability
"""
function calculate_kpuu_brain(;
    logP::Float64,
    fup::Float64,
    fub::Float64 = nothing,  # Fraction unbound in brain
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_pgp_substrate::Bool = false
)
    # Estimate brain unbound fraction if not provided
    if isnothing(fub)
        fub = estimate_fub(logP, pKa=pKa, is_base=is_base)
    end

    Kp = calculate_kp_brain(
        logP=logP, fup=fup, pKa=pKa,
        is_base=is_base, is_pgp_substrate=is_pgp_substrate
    )

    # Kp,uu = Kp × (fup / fub)
    Kpuu = Kp * (fup / max(fub, 0.001))

    # P-gp effect on Kp,uu
    if is_pgp_substrate
        Kpuu *= 0.3  # Efflux reduces unbound brain
    end

    return (Kp=Kp, Kpuu=Kpuu, fub=fub)
end

"""
Estimate fraction unbound in brain tissue

Brain tissue binding is primarily to phospholipids and lipids.
"""
function estimate_fub(logP::Float64;
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false)

    P = 10^logP

    # Brain has high lipid content
    f_lipid = 0.039 + 0.0015  # Neutral + phospholipid
    f_iw = 0.620

    # Non-specific binding to lipids
    lipid_binding = P * f_lipid

    # Bases may bind to acidic groups (low in brain)
    if is_base && !isnothing(pKa) && pKa > 7.0
        lipid_binding *= 1.2  # Slight increase
    end

    fub = f_iw / (f_iw + lipid_binding)

    return clamp(fub, 0.001, 0.99)
end

"""
Calculate brain contribution to Vdss
"""
function calculate_brain_contribution(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    is_pgp_substrate::Bool = false,
    brain_volume::Float64 = 1.4
)
    Kp = calculate_kp_brain(
        logP=logP, logD=logD, fup=fup,
        pKa=pKa, is_base=is_base, is_acid=is_acid,
        is_pgp_substrate=is_pgp_substrate
    )

    contribution = Kp * brain_volume

    return (Kp=Kp, contribution_L=contribution, volume=brain_volume)
end

"""
Special considerations for brain:

1. BLOOD-BRAIN BARRIER (BBB)
   - Tight junctions: no paracellular transport
   - P-gp, BCRP: active efflux
   - Limited pinocytosis
   - Only lipophilic, small molecules cross easily

2. P-GLYCOPROTEIN (P-gp)
   - Major determinant of brain exposure
   - Substrates: many drugs, especially lipophilic
   - Inhibitors: cyclosporine, verapamil, ritonavir
   - Genetic variants affect CNS drug response

3. CEREBROSPINAL FLUID (CSF)
   - Different from brain tissue
   - Often used as surrogate for brain
   - CSF sampling is easier (lumbar puncture)
   - But CSF ≠ brain tissue concentrations!

4. REGIONAL DIFFERENCES
   - Cortex vs brainstem vs cerebellum
   - White matter vs gray matter
   - Different lipid compositions

5. DISEASE EFFECTS
   - Brain tumors: disrupted BBB
   - Alzheimer's: altered P-gp
   - Epilepsy: enhanced P-gp
   - Stroke: transient BBB opening

6. AGE EFFECTS
   - Neonates: immature BBB
   - Elderly: may have some BBB breakdown
   - Important for pediatric/geriatric dosing
"""

# Example CNS drugs
const CNS_DRUGS = Dict(
    "diazepam" => (logP=2.8, Kp_brain=0.9, Kpuu=0.8, note="Benzodiazepine, good BBB"),
    "haloperidol" => (logP=4.3, Kp_brain=15.0, Kpuu=3.0, note="Antipsychotic, brain accumulator"),
    "risperidone" => (logP=3.5, Kp_brain=10.0, Kpuu=0.3, note="P-gp substrate!"),
    "morphine" => (logP=0.9, Kp_brain=0.3, Kpuu=0.4, note="P-gp substrate, poor BBB"),
    "caffeine" => (logP=-0.1, Kp_brain=0.8, Kpuu=1.0, note="Equilibrates freely"),
    "loperamide" => (logP=4.8, Kp_brain=0.05, Kpuu=0.02, note="P-gp, no CNS at normal dose"),
)

end # module
