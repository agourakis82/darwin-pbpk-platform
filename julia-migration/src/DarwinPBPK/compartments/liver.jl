# LIVER COMPARTMENT MODEL
# =======================
#
# The liver is THE metabolizing organ - but also a distribution compartment.
# Unique features that make it special:
#
# 1. DUAL BLOOD SUPPLY
#    - Portal vein (75%): from gut, first-pass metabolism
#    - Hepatic artery (25%): oxygenated blood
#
# 2. HIGH ENZYME CONTENT
#    - CYP450s: 1A2, 2C9, 2C19, 2D6, 3A4 (major drug metabolizers)
#    - UGTs: glucuronidation
#    - Esterases, amidases
#
# 3. TRANSPORTERS
#    - Uptake: OATPs (1B1, 1B3, 2B1), OCT1, NTCP
#    - Efflux: P-gp, BCRP, MRPs, BSEP
#    - These can cause Kp >> simple partitioning predicts!
#
# 4. HIGH PHOSPHOLIPID CONTENT
#    - 2.4% phospholipids (one of highest)
#    - 0.46% acidic phospholipids
#    - Important for base binding
#
# 5. SPECIALIZED CELLS
#    - Hepatocytes (80%): metabolism, transporters
#    - Kupffer cells: immune function
#    - Stellate cells: vitamin A storage
#    - Sinusoidal endothelium: fenestrated (no barrier!)

module LiverCompartment

export LiverProperties, calculate_kp_liver, calculate_liver_contribution
export estimate_transporter_effect

"""
Liver physiological properties

Reference values for 70kg adult:
- Volume: 1.5-1.8 L
- Blood flow: 1.5 L/min (25% of cardiac output!)
- Highest blood flow per gram of any organ
"""
struct LiverProperties
    volume_L::Float64           # Liver volume
    blood_flow_L_min::Float64   # Total hepatic blood flow
    f_neutral_lipid::Float64    # Neutral lipid fraction
    f_phospholipid::Float64     # Neutral phospholipids
    f_acidic_pl::Float64        # Acidic phospholipids (HIGH!)
    f_water_iw::Float64         # Intracellular water
    f_water_ew::Float64         # Extracellular water (sinusoids)
    albumin_ratio::Float64      # Albumin tissue/plasma
    lipoprotein_ratio::Float64  # LP tissue/plasma
    pH_iw::Float64              # Intracellular pH
    # Transporter abundances (pmol/mg protein)
    OATP1B1::Float64
    OATP1B3::Float64
    OCT1::Float64
    P_gp::Float64
end

# Default for 70kg adult
const DEFAULT_LIVER = LiverProperties(
    1.8,     # volume (L)
    1.5,     # blood flow (L/min) - VERY HIGH!
    0.014,   # neutral lipids
    0.024,   # neutral phospholipids (HIGH)
    0.00456, # acidic phospholipids (HIGH - base binding!)
    0.573,   # intracellular water
    0.161,   # extracellular water (fenestrated sinusoids)
    0.086,   # albumin ratio
    0.161,   # lipoprotein ratio
    7.0,     # intracellular pH
    # Typical transporter abundances
    4.0,     # OATP1B1 pmol/mg
    1.0,     # OATP1B3 pmol/mg
    3.5,     # OCT1 pmol/mg
    0.5      # P-gp pmol/mg
)

"""
Estimate transporter effect on liver Kp

CRITICAL INSIGHT: Transporters can cause Kp to be MUCH higher than
passive partitioning alone!

Examples:
- Statins (OATP1B1 substrates): liver Kp 10-100x higher than predicted
- Metformin (OCT1 substrate): accumulates in liver
- Rifampicin (OATP inhibitor): drug-drug interactions

Returns a multiplier for Kp (1.0 = no effect, >1 = uptake, <1 = efflux)
"""
function estimate_transporter_effect(;
    is_oatp_substrate::Bool = false,
    is_oct_substrate::Bool = false,
    is_pgp_substrate::Bool = false,
    is_anion::Bool = false,
    is_cation::Bool = false,
    MW::Float64 = 400.0
)
    effect = 1.0

    # OATP substrates: organic anions, MW > 400, amphiphilic
    if is_oatp_substrate || (is_anion && MW > 400)
        effect *= 3.0  # Statins can be 10-100x, use conservative estimate
    end

    # OCT1 substrates: organic cations
    if is_oct_substrate || is_cation
        effect *= 2.0  # Metformin-like accumulation
    end

    # P-gp substrates: efflux pumps reduce Kp
    if is_pgp_substrate
        effect *= 0.7  # Some drug is pumped out
    end

    return effect
end

"""
Calculate liver:plasma partition coefficient

Accounts for:
1. Passive partitioning (lipids, phospholipids, water)
2. pH-dependent ionization
3. Transporter-mediated uptake (optional)
"""
function calculate_kp_liver(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    liver::LiverProperties = DEFAULT_LIVER,
    transporter_effect::Float64 = 1.0  # Multiplier from transporters
)
    P = 10^logP
    D = 10^logD

    # Ionization factors
    pH_p = 7.4
    pH_iw = liver.pH_iw

    X = 0.0  # Tissue ionization
    Y = 0.0  # Plasma ionization

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
    f_ew = liver.f_water_ew
    f_iw = liver.f_water_iw
    f_nl = liver.f_neutral_lipid
    f_npl = liver.f_phospholipid
    f_apl = liver.f_acidic_pl
    AR = liver.albumin_ratio
    LR = liver.lipoprotein_ratio

    denom = max(1 + Y, 1e-10)

    # Plasma binding constant (simplified)
    # For highly bound drugs, Ka_PR can be very high
    Ka_PR = max(0, min((1/fup - 1), 1000))  # Cap at reasonable value

    # Acidic phospholipid association constant
    # Based on empirical observations: Ka_AP typically 10-100 for strong bases
    # The original R-R calculation requires experimental blood cell data
    # Here we use a simplified empirical estimate based on base strength
    if is_base && !isnothing(pKa) && pKa > 7.0
        # Stronger bases have higher affinity for acidic phospholipids
        Ka_AP = 50.0 * (1 + 0.3 * (pKa - 7.0))  # Increases with pKa
        Ka_AP = min(Ka_AP, 200.0)  # Cap at reasonable value
    else
        Ka_AP = 10.0  # Default for weak bases/neutrals
    end

    # Calculate Kp based on drug type
    if is_base && !isnothing(pKa) && pKa > 7.0
        # Strong base: acidic phospholipid binding is KEY
        # Liver has HIGH acidic PL (0.46%) - important for bases!
        Kp = (f_ew +
              ((1 + X) / denom) * f_iw +
              (P * f_nl + (0.3*P + 0.7) * f_npl) / denom +
              (Ka_AP * f_apl * X) / denom) * fup
    elseif is_acid
        # Acids: albumin binding
        Kp = (f_ew +
              ((1 + X) / denom) * f_iw +
              (P * f_nl + (0.3*P + 0.7) * f_npl) / denom +
              (Ka_PR * AR * X) / denom) * fup
    else
        # Neutrals: lipoprotein binding
        Kp = (f_ew +
              ((1 + X) / denom) * f_iw +
              (P * f_nl + (0.3*P + 0.7) * f_npl) / denom +
              (Ka_PR * LR) / denom) * fup
    end

    # Apply transporter effect
    Kp *= transporter_effect

    return max(Kp, 0.01)
end

"""
Calculate liver contribution to Vdss
"""
function calculate_liver_contribution(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    liver_volume::Float64 = 1.8,
    transporter_effect::Float64 = 1.0
)
    Kp = calculate_kp_liver(
        logP=logP, logD=logD, fup=fup,
        pKa=pKa, is_base=is_base, is_acid=is_acid,
        transporter_effect=transporter_effect
    )

    contribution = Kp * liver_volume

    return (Kp=Kp, contribution_L=contribution, volume=liver_volume)
end

"""
Special considerations for liver:

1. FIRST-PASS METABOLISM
   - Portal vein delivers gut-absorbed drugs directly
   - High extraction drugs: >70% metabolized first pass
   - Low bioavailability despite good absorption

2. TRANSPORTER-ENZYME INTERPLAY
   - OATPs bring drugs into hepatocytes
   - CYP450s metabolize them
   - Efflux pumps export metabolites
   - This "conveyor belt" determines hepatic clearance

3. PROTEIN BINDING EFFECTS
   - Highly bound drugs: restricted uptake
   - Albumin-bound acids: OATP-mediated dissociation
   - This is the "albumin-facilitated uptake" phenomenon

4. DISEASE EFFECTS
   - Cirrhosis: reduced metabolic capacity, portal shunting
   - Hepatitis: enzyme induction/inhibition
   - NAFLD: altered transporters

5. GENETIC POLYMORPHISMS
   - OATP1B1 (SLCO1B1): statin toxicity variants
   - CYP2D6: poor/extensive metabolizers
   - UGT1A1: Gilbert's syndrome

6. SINUSOIDAL PERMEABILITY
   - NO barrier (unlike BBB)
   - Fenestrated endothelium
   - Drugs freely access hepatocytes
"""

# Example liver-targeting drugs
const LIVER_DRUGS = Dict(
    "atorvastatin" => (logP=4.5, Kp_liver=25.0, note="OATP1B1 substrate, 10-50x Kp"),
    "rosuvastatin" => (logP=-0.3, Kp_liver=10.0, note="Hydrophilic statin, OATP"),
    "metformin" => (logP=-1.5, Kp_liver=5.0, note="OCT1 substrate"),
    "rifampicin" => (logP=3.7, Kp_liver=8.0, note="OATP inhibitor"),
    "digoxin" => (logP=1.3, Kp_liver=0.5, note="P-gp substrate"),
)

end # module
