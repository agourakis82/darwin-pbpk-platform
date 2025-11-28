# KIDNEY COMPARTMENT MODEL
# ========================
#
# The kidney is THE elimination organ for many drugs.
# But it's also a distribution compartment with unique features:
#
# 1. HIGHEST BLOOD FLOW PER GRAM
#    - Receives 20-25% of cardiac output (1.2 L/min!)
#    - But weighs only 0.3 kg
#    - This is for filtration, not distribution
#
# 2. ELIMINATION PROCESSES
#    - Glomerular filtration (GFR ~120 mL/min)
#    - Tubular secretion (OATs, OCTs, MATE)
#    - Tubular reabsorption (lipophilic drugs)
#
# 3. HIGHEST PHOSPHOLIPID CONTENT
#    - 2.4% phospholipids
#    - 0.5% acidic phospholipids (highest of all tissues!)
#    - This means BASES ACCUMULATE in kidney
#
# 4. PROXIMAL TUBULE
#    - Primary site of secretion
#    - OAT1, OAT3: organic anion transport
#    - OCT2: organic cation transport
#    - MATE1/2: cation efflux
#
# Key insight: Kidney Kp can be VERY HIGH for transporter
# substrates, but this doesn't mean drug is "distributed"
# there - it's being eliminated!

module KidneyCompartment

export KidneyProperties, calculate_kp_kidney, calculate_kidney_contribution
export estimate_renal_clearance_contribution

"""
Kidney physiological properties

Reference values for 70kg adult:
- Volume: 0.3 L (both kidneys)
- Blood flow: 1.2 L/min (20-25% cardiac output!)
- GFR: ~120 mL/min
"""
struct KidneyProperties
    volume_L::Float64           # Kidney volume
    blood_flow_L_min::Float64   # Renal blood flow
    f_neutral_lipid::Float64    # Neutral lipids
    f_phospholipid::Float64     # Neutral phospholipids (HIGH)
    f_acidic_pl::Float64        # Acidic phospholipids (HIGHEST!)
    f_water_iw::Float64         # Intracellular water
    f_water_ew::Float64         # Extracellular water
    albumin_ratio::Float64      # Albumin tissue/plasma
    lipoprotein_ratio::Float64  # LP tissue/plasma
    pH_iw::Float64              # Intracellular pH
    pH_tubular::Float64         # Tubular urine pH (variable!)
    GFR_mL_min::Float64         # Glomerular filtration rate
    # Transporter abundances
    OAT1::Float64               # Organic anion transporter 1
    OAT3::Float64               # Organic anion transporter 3
    OCT2::Float64               # Organic cation transporter 2
    MATE1::Float64              # Multidrug and toxin extrusion
end

# Default for 70kg adult with normal renal function
const DEFAULT_KIDNEY = KidneyProperties(
    0.31,    # volume (L)
    1.2,     # blood flow (L/min) - VERY HIGH per gram!
    0.012,   # neutral lipids
    0.0242,  # neutral phospholipids (HIGH)
    0.00503, # acidic phospholipids (HIGHEST of all tissues!)
    0.483,   # intracellular water
    0.273,   # extracellular water (high due to filtration)
    0.130,   # albumin ratio
    0.137,   # lipoprotein ratio
    7.0,     # intracellular pH
    6.0,     # tubular urine pH (can range 4.5-8.0!)
    120.0,   # GFR (mL/min)
    # Transporter abundances (pmol/mg protein)
    4.0,     # OAT1
    2.0,     # OAT3
    5.0,     # OCT2 (HIGH!)
    2.0      # MATE1
)

"""
Estimate transporter effect on kidney Kp and clearance

Kidney has BOTH uptake and efflux transporters:
- Uptake (basolateral): OAT1, OAT3 (anions), OCT2 (cations)
- Efflux (apical): MATE1, MATE2-K, OAT4, P-gp

The interplay determines:
1. Kidney Kp (tissue accumulation)
2. Tubular secretion (elimination)
"""
function estimate_transporter_effect(;
    is_oat_substrate::Bool = false,
    is_oct_substrate::Bool = false,
    is_mate_substrate::Bool = false,
    is_anion::Bool = false,
    is_cation::Bool = false,
    MW::Float64 = 400.0
)
    kp_multiplier = 1.0
    clearance_multiplier = 1.0

    # OAT substrates: organic anions accumulate
    if is_oat_substrate || (is_anion && MW < 500)
        kp_multiplier *= 2.5
        clearance_multiplier *= 3.0  # Secretion increases CL
    end

    # OCT2 substrates: cations accumulate
    if is_oct_substrate || is_cation
        kp_multiplier *= 3.0
        clearance_multiplier *= 2.5
    end

    # MATE substrates: efflux helps secretion
    if is_mate_substrate
        # MATE works WITH OCT2 for secretion
        clearance_multiplier *= 1.5
    end

    return (kp_effect=kp_multiplier, cl_effect=clearance_multiplier)
end

"""
Calculate kidney:plasma partition coefficient

KEY: Kidney has the HIGHEST acidic phospholipid content!
Strong bases (pKa > 7) can have Kp_kidney >> other tissues.

This is why aminoglycosides accumulate in kidney (nephrotoxicity)!
"""
function calculate_kp_kidney(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    kidney::KidneyProperties = DEFAULT_KIDNEY,
    transporter_effect::Float64 = 1.0
)
    P = 10^logP
    D = 10^logD

    # Ionization factors
    pH_p = 7.4
    pH_iw = kidney.pH_iw

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

    # Tissue composition - note HIGH phospholipid content!
    f_ew = kidney.f_water_ew
    f_iw = kidney.f_water_iw
    f_nl = kidney.f_neutral_lipid
    f_npl = kidney.f_phospholipid
    f_apl = kidney.f_acidic_pl  # HIGHEST of all tissues!
    AR = kidney.albumin_ratio
    LR = kidney.lipoprotein_ratio

    denom = max(1 + Y, 1e-10)

    # Plasma binding constant (simplified, capped)
    Ka_PR = max(0, min((1/fup - 1), 1000))

    # Acidic phospholipid association constant
    # Kidney has HIGHEST acidic PL (0.5%) - 3x more than muscle
    # This causes significant base accumulation (aminoglycoside nephrotoxicity!)
    if is_base && !isnothing(pKa) && pKa > 7.0
        # Stronger bases bind more to acidic phospholipids
        Ka_AP = 60.0 * (1 + 0.4 * (pKa - 7.0))  # Higher than liver due to more APL
        Ka_AP = min(Ka_AP, 250.0)  # Cap at reasonable value
    else
        Ka_AP = 15.0  # Default
    end

    # Calculate Kp based on drug type
    if is_base && !isnothing(pKa) && pKa > 7.0
        # STRONG BASE: Acidic phospholipid binding DOMINATES!
        # This is why aminoglycosides (strong bases) accumulate in kidney
        Kp = (f_ew +
              ((1 + X) / denom) * f_iw +
              (P * f_nl + (0.3*P + 0.7) * f_npl) / denom +
              (Ka_AP * f_apl * X) / denom) * fup

        # Extra binding due to VERY high acidic PL in kidney
        # f_apl is 5×10⁻³ in kidney vs 1.5×10⁻³ in muscle!
        # This ~3.3x difference is significant for bases

    elseif is_acid
        # Acids: albumin binding + OAT-mediated uptake
        Kp = (f_ew +
              ((1 + X) / denom) * f_iw +
              (P * f_nl + (0.3*P + 0.7) * f_npl) / denom +
              (Ka_PR * AR * X) / denom) * fup
    else
        # Neutrals
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
Calculate kidney contribution to Vdss
"""
function calculate_kidney_contribution(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    kidney_volume::Float64 = 0.31,
    transporter_effect::Float64 = 1.0
)
    Kp = calculate_kp_kidney(
        logP=logP, logD=logD, fup=fup,
        pKa=pKa, is_base=is_base, is_acid=is_acid,
        transporter_effect=transporter_effect
    )

    contribution = Kp * kidney_volume

    return (Kp=Kp, contribution_L=contribution, volume=kidney_volume)
end

"""
Estimate fraction of renal clearance due to tubular secretion

Total renal CL = Filtration + Secretion - Reabsorption

CLrenal = fup × GFR × (1 + secretion_ratio - reabsorption)

Where:
- Filtration = fup × GFR
- Secretion depends on OAT/OCT transporters
- Reabsorption depends on lipophilicity and urine pH
"""
function estimate_renal_clearance_contribution(;
    fup::Float64,
    GFR_mL_min::Float64 = 120.0,
    logP::Float64 = 0.0,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    is_oat_substrate::Bool = false,
    is_oct_substrate::Bool = false,
    urine_pH::Float64 = 6.0
)
    # Filtration clearance (unbound drug)
    CL_filtration = fup * GFR_mL_min

    # Secretion (transporter-mediated)
    secretion_ratio = 0.0
    if is_oat_substrate || is_oct_substrate
        secretion_ratio = 2.0  # Doubles filtration CL
    end

    # Reabsorption (passive, depends on lipophilicity and ionization)
    reabsorption = 0.0
    if logP > 0
        # Lipophilic drugs are reabsorbed
        P = 10^logP
        base_reabsorption = P / (1 + P)  # Fraction reabsorbed

        # Ionization reduces reabsorption
        if !isnothing(pKa)
            if is_base
                # Bases: ionized at acidic urine pH → less reabsorbed
                ionization = 10^(pKa - urine_pH)
                base_reabsorption /= (1 + ionization)
            elseif is_acid
                # Acids: ionized at alkaline urine pH
                ionization = 10^(urine_pH - pKa)
                base_reabsorption /= (1 + ionization)
            end
        end

        reabsorption = base_reabsorption * 0.8  # Scale factor
    end

    # Total renal clearance
    CL_renal = CL_filtration * (1 + secretion_ratio) * (1 - reabsorption)

    return (CL_renal=CL_renal, CL_filtration=CL_filtration,
            secretion_ratio=secretion_ratio, reabsorption=reabsorption)
end

"""
Special considerations for kidney:

1. NEPHROTOXICITY RISK
   - Aminoglycosides: accumulate due to strong base binding
   - NSAIDs: inhibit prostaglandins, reduce GFR
   - Cisplatin: proximal tubule damage
   - Contrast agents: acute kidney injury
   - High Kp_kidney + toxicity = danger!

2. RENAL IMPAIRMENT EFFECTS
   - GFR reduction: CKD stages 1-5
   - Transporter expression changes
   - Protein binding changes (uremia)
   - Drug accumulation risk

3. DRUG-DRUG INTERACTIONS
   - OAT1/3 inhibition: probenecid, NSAIDs
   - OCT2 inhibition: cimetidine
   - MATE inhibition: pyrimethamine
   - Reduces secretion → higher systemic exposure

4. URINE pH MANIPULATION
   - Alkalinize: increases weak acid excretion
   - Acidify: increases weak base excretion
   - Used in poisoning management

5. AGE EFFECTS
   - Neonates: immature tubular function
   - Elderly: reduced GFR (30-40% by age 80)
   - Critical for renally cleared drugs

6. CONCENTRATION IN TUBULAR FLUID
   - Proximal tubule: 3x plasma concentration (water reabsorption)
   - Collecting duct: 100x plasma (depending on drug)
   - This is why kidney has high drug concentrations
"""

# Example renally-eliminated drugs
const RENAL_DRUGS = Dict(
    "gentamicin" => (logP=-3.1, pKa=8.2, Kp_kidney=10.0, note="Aminoglycoside, nephrotoxic accumulation"),
    "metformin" => (logP=-1.5, Kp_kidney=5.0, note="OCT2/MATE substrate, renal CL > GFR"),
    "penicillin" => (logP=1.8, Kp_kidney=3.0, note="OAT substrate, tubular secretion"),
    "furosemide" => (logP=2.0, pKa=3.9, Kp_kidney=2.0, note="OAT substrate, loop diuretic"),
    "digoxin" => (logP=1.3, Kp_kidney=0.5, note="P-gp substrate, mainly filtration"),
    "vancomycin" => (logP=-3.0, Kp_kidney=8.0, note="Large molecule, proximal tubule toxicity"),
)

end # module
