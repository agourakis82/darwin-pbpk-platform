# LIVER COMPARTMENT MODEL (ENHANCED)
# ===================================
#
# The liver is THE central organ for drug disposition:
# - Metabolism (Phase I/II)
# - First-pass extraction
# - Active transporter-mediated uptake and efflux
# - Significant drug reservoir for basic drugs
#
# UNIQUE FEATURES FROM DEEP DIVE:
# ================================
# 1. DUAL BLOOD SUPPLY
#    - Portal vein (75%): from gut, first-pass metabolism
#    - Hepatic artery (25%): oxygenated blood
#    - Flow: ~1500 mL/min (25% cardiac output!)
#
# 2. SINUSOIDAL ARCHITECTURE
#    - Fenestrated endothelium (100-150nm pores)
#    - NO basement membrane - direct hepatocyte access
#    - Slow flow (~400-800 μm/s) maximizes extraction
#    - Space of Disse allows free drug access
#
# 3. ZONATION
#    - Zone 1 (periportal): high O2, gluconeogenesis, Phase II
#    - Zone 2 (mid): transitional
#    - Zone 3 (centrilobular): low O2, CYP450 HIGH, lipogenesis
#
# 4. HIGH LYSOSOMAL CONTENT
#    - ~2.5% volume fraction (5x more than muscle!)
#    - pH 4.8 creates massive trapping of basic drugs
#    - Critical for imipramine, chloroquine, amiodarone
#
# 5. HIGH ACIDIC PHOSPHOLIPID CONTENT
#    - 0.49% acidic phospholipids (3x more than muscle!)
#    - Strong binding of protonated bases
#    - Combined with lysosomes = significant base reservoir
#
# 6. TRANSPORTERS
#    - Uptake: OATP1B1/1B3 (anions), OCT1 (cations), NTCP (bile acids)
#    - Efflux to bile: P-gp, MRP2, BCRP, BSEP
#    - Safety valves: MRP3/4 (back to blood)
#    - Can increase Kp 10-100x for transporter substrates!
#
# References:
# - Schmitt et al. 2021: Extension of R-R by lysosomal trapping
# - Rodgers & Rowland 2006: Mechanistic tissue Kp prediction
# - Nagar & Korzekwa 2012: Hepatic transporter modeling
# - Liver Deep Dive documentation (6 parts)

module LiverCompartment

export LiverProperties, calculate_kp_liver, calculate_liver_contribution
export estimate_transporter_effect, calculate_lysosomal_trapping_liver
export calculate_hepatic_extraction, calculate_first_pass_bioavailability
export calculate_effective_K_tissue_liver
# SOTA 2024 Q1+ exports - Hepatic Clearance Models
export HepaticClearanceModel, WellStirredModel, ParallelTubeModel, DispersionModel
export calculate_hepatic_clearance, calculate_biliary_clearance
export BiliaryClearanceParams, EnterohepatiCirculation, EHCState
export simulate_ehc_ode!, calculate_ehc_auc_ratio
# Transporter Saturation (Michaelis-Menten)
export TransporterKinetics, calculate_saturable_uptake, calculate_saturable_efflux
export OATP1B1_KINETICS, OATP1B3_KINETICS, OCT1_KINETICS, PGP_KINETICS
# CYP Zonation
export HepaticZonation, calculate_zonal_metabolism, ZonalCYPExpression
export PERIPORTAL_ZONE, CENTRILOBULAR_ZONE, calculate_zone_weighted_clearance

"""
Liver physiological properties

Reference values for 70kg adult:
- Volume: 1.5-1.8 L (2% body weight, but 25% cardiac output!)
- Blood flow: 1.5 L/min total (1.05 portal, 0.45 arterial)
- Highest blood flow per gram of any organ

Lysosomal data from Schmitt et al. 2021:
- Liver lysosomal volume fraction: 2.5% (HIGH!)
- Lysosomal pH: 4.8
"""
struct LiverProperties
    volume_L::Float64           # Liver volume
    blood_flow_L_min::Float64   # Total hepatic blood flow
    portal_fraction::Float64    # Fraction from portal vein
    f_neutral_lipid::Float64    # Neutral lipid fraction
    f_phospholipid::Float64     # Neutral phospholipids
    f_acidic_pl::Float64        # Acidic phospholipids (HIGH!)
    f_water_iw::Float64         # Intracellular water
    f_water_ew::Float64         # Extracellular water (sinusoids)
    albumin_ratio::Float64      # Albumin tissue/plasma
    lipoprotein_ratio::Float64  # LP tissue/plasma
    pH_iw::Float64              # Intracellular pH
    f_lysosome::Float64         # Lysosomal volume fraction (HIGH: 2.5%)
    pH_lysosome::Float64        # Lysosomal pH (very acidic)
    # Transporter abundances (pmol/mg protein) - for advanced models
    OATP1B1::Float64
    OATP1B3::Float64
    OCT1::Float64
    P_gp::Float64
    BCRP::Float64
    MRP2::Float64
end

# Default for 70kg adult
# Values from Rodgers & Rowland 2006, updated with Schmitt 2021 lysosomal data
const DEFAULT_LIVER = LiverProperties(
    1.8,      # volume (L)
    1.5,      # total blood flow (L/min) - 25% cardiac output!
    0.75,     # portal fraction (75% from gut)
    0.014,    # neutral lipids
    0.024,    # neutral phospholipids (HIGH)
    0.00490,  # acidic phospholipids (HIGH - 3x muscle!)
    0.573,    # intracellular water
    0.161,    # extracellular water (fenestrated sinusoids)
    0.086,    # albumin ratio
    0.161,    # lipoprotein ratio (high due to VLDL synthesis)
    7.0,      # intracellular pH
    0.025,    # lysosomal volume fraction (2.5% - 5x muscle!)
    4.8,      # lysosomal pH (very acidic!)
    # Typical transporter abundances (pmol/mg protein)
    4.0,      # OATP1B1 - statins, methotrexate
    1.0,      # OATP1B3 - taxanes, bilirubin
    3.5,      # OCT1 - metformin
    0.5,      # P-gp - lipophilic drugs
    2.0,      # BCRP - rosuvastatin, sulfates
    1.5       # MRP2 - glucuronides, bilirubin
)

"""
Calculate effective tissue binding constant K_tissue for LIVER

Liver has 3x the acidic phospholipid content of muscle (0.49% vs 0.15%)
This means stronger binding of basic drugs to membrane-associated PS.

Like muscle, we use a lipophilicity-gated approach since PS is
concentrated in membranes and requires membrane partitioning for access.

Returns higher values than muscle due to higher APL content.
"""
function calculate_effective_K_tissue_liver(logP::Float64)
    # Liver has ~3x the acidic phospholipid content of muscle
    # Scale K_tissue accordingly, but with saturation

    if logP < 1.0
        return 0.0
    elseif logP < 2.0
        # Transition zone - slightly higher than muscle
        return 0.8 * (logP - 1.0)  # 0 to 0.8
    elseif logP < 3.0
        return 0.8 + 3.0 * (logP - 2.0)  # 0.8 to 3.8
    elseif logP < 4.0
        return 3.8 + 8.0 * (logP - 3.0)  # 3.8 to 11.8
    elseif logP < 5.0
        return 11.8 + 10.0 * (logP - 4.0)  # 11.8 to 21.8
    else
        # Plateau - liver has higher max due to more APL
        return 22.0
    end
end

"""
Calculate lysosomal trapping factor for LIVER

CRITICAL: Liver has 2.5% lysosomal volume vs 0.5% in muscle!
This 5x higher lysosome content means MASSIVE trapping of basic drugs.

Examples:
- Chloroquine: accumulates 100-1000x in lysosomes
- Imipramine: significant hepatic reservoir
- Amiodarone: long half-life partly due to lysosomal trapping

From Schmitt et al. 2021:
- Lysosomal concentration can exceed cytosolic by 160,000x
- Effect strongest for lipophilic bases (pKa > 7, logP 2-4)
"""
function calculate_lysosomal_trapping_liver(;
    pKa::Float64,
    logP::Float64,
    f_lysosome::Float64 = 0.025,  # 2.5% for liver
    pH_lysosome::Float64 = 4.8,
    pH_cytosol::Float64 = 7.0
)
    # Non-bases don't trap
    if pKa < 6.0
        return 0.0
    end

    # Ionization ratio: lysosome vs cytosol
    # At pH 4.8, a base with pKa 8 is 99.98% ionized
    # At pH 7.0, same base is 90.9% ionized
    ionized_lyso = 10^(pKa - pH_lysosome)
    ionized_cyto = 10^(pKa - pH_cytosol)

    # Accumulation ratio (can be very high!)
    # For pKa 8: (1 + 10^3.2) / (1 + 10^1) = 1585/11 = 144x
    accumulation = (1 + ionized_lyso) / (1 + ionized_cyto)

    # Lysosomal membrane permeability factor
    # Need adequate lipophilicity to cross lysosomal membrane
    permeability_factor = if logP < 1.0
        0.05  # Very hydrophilic: poor lysosomal entry
    elseif logP < 2.0
        0.05 + 0.3 * (logP - 1.0)  # 0.05 to 0.35
    elseif logP < 3.0
        0.35 + 0.4 * (logP - 2.0)  # 0.35 to 0.75
    elseif logP < 4.0
        0.75 + 0.15 * (logP - 3.0)  # 0.75 to 0.90 (optimal range)
    else
        0.85  # Very lipophilic: can escape lysosome more easily
    end

    # Final lysosomal contribution
    # Liver: 2.5% lysosomes × accumulation × permeability
    lyso_contribution = f_lysosome * accumulation * permeability_factor

    # Special case: Very strong bases (pKa > 9.5) like chloroquine
    # can show even stronger trapping due to:
    # 1. Essentially complete ionization at lysosomal pH
    # 2. Multiple basic nitrogens (di-cations)
    # 3. Lysosomal membrane binding
    # Chloroquine Kp_liver ~100, needs major correction
    if pKa > 9.5
        strong_base_factor = 1.0 + 2.0 * (pKa - 9.5)  # Up to 3x for pKa 10.5
        lyso_contribution *= strong_base_factor
    end

    return lyso_contribution
end

"""
Estimate transporter effect on liver Kp

Transporters are CRITICAL for liver distribution!
Examples where Kp >> passive prediction:
- Statins (OATP1B1/1B3): 10-100x higher liver Kp
- Metformin (OCT1): 3-10x hepatic accumulation
- Where Kp < passive prediction:
- Digoxin (P-gp): pumped out of hepatocytes

Returns a multiplier for Kp:
- 1.0 = no transporter effect
- >1.0 = net uptake (OATPs, OCT1)
- <1.0 = net efflux (P-gp, BCRP dominant)

CLINICAL RELEVANCE:
- SLCO1B1*5/*15 polymorphisms reduce statin uptake → higher plasma levels → myopathy
- OCT1 polymorphisms reduce metformin efficacy (less hepatic uptake)
- Cyclosporine inhibits OATPs → statin DDI (↑ AUC 8-15x!)
"""
function estimate_transporter_effect(;
    is_oatp_substrate::Bool = false,
    is_oct_substrate::Bool = false,
    is_pgp_substrate::Bool = false,
    is_bcrp_substrate::Bool = false,
    is_anion::Bool = false,
    is_cation::Bool = false,
    MW::Float64 = 400.0,
    PSA::Float64 = 100.0,  # Polar surface area
    logP::Float64 = 2.0
)
    effect = 1.0

    # =============================================
    # UPTAKE TRANSPORTERS (increase Kp)
    # =============================================

    # OATP1B1/1B3 substrates
    # Characteristics: organic anions, MW > 350, amphiphilic
    # Statins, methotrexate, rifampin, bilirubin
    #
    # KEY INSIGHT: OATP effect is inversely related to lipophilicity!
    # - Hydrophilic statins (rosuvastatin, pravastatin): need strong OATP for uptake
    # - Lipophilic statins (atorvastatin): some passive uptake, less OATP-dependent
    #
    # Literature Kp values:
    # - Rosuvastatin (logP -0.3): Kp ~10, passive ~0.1 → need ~100x
    # - Pravastatin (logP -0.8): Kp ~15, passive ~0.5 → need ~30x
    # - Atorvastatin (logP 4.5): Kp ~25, passive ~5 → need ~5x
    if is_oatp_substrate
        # Lipophilicity-adjusted OATP effect
        if logP < 0
            effect *= 100.0  # Very hydrophilic: totally OATP-dependent
        elseif logP < 1.0
            effect *= 50.0   # Hydrophilic
        elseif logP < 2.0
            effect *= 25.0   # Moderate
        elseif logP < 3.0
            effect *= 10.0   # Some passive contribution
        else
            effect *= 5.0    # Lipophilic: significant passive uptake
        end
    elseif is_anion && MW > 350 && PSA > 75
        # Predict OATP substrate likelihood
        effect *= 3.0
    end

    # OCT1 substrates
    # Characteristics: organic cations, MW < 500, hydrophilic
    # Metformin is the classic example
    if is_oct_substrate
        effect *= 3.0  # Metformin shows ~3-5x hepatic uptake
    elseif is_cation && MW < 500 && logP < 2.0
        # Predict OCT1 substrate likelihood
        effect *= 1.5
    end

    # =============================================
    # EFFLUX TRANSPORTERS (decrease Kp)
    # =============================================

    # P-gp substrates - pump drugs into bile
    # Generally lipophilic, MW > 400
    if is_pgp_substrate
        effect *= 0.6  # Reduces hepatic accumulation
    elseif logP > 3.0 && MW > 400
        # High lipophilicity suggests possible P-gp substrate
        effect *= 0.85
    end

    # BCRP substrates
    # Rosuvastatin is both OATP substrate (uptake) and BCRP substrate (efflux)
    if is_bcrp_substrate
        effect *= 0.8
    end

    return effect
end

"""
Calculate liver:plasma partition coefficient (ENHANCED MODEL)

This enhanced model includes:
1. Standard Rodgers-Rowland terms (water, lipids, protein)
2. Effective tissue binding for bases (K_tissue approach)
3. Lysosomal trapping (CRITICAL for liver - 2.5% lysosomes!)
4. Transporter effects (OATPs, OCT1, P-gp, BCRP)

VALIDATION TARGETS (from literature):
- Propranolol: Kp_liver ~3-5 (base, lysosomal + APL binding)
- Imipramine: Kp_liver ~8-15 (lipophilic base, strong lysosomal)
- Atorvastatin: Kp_liver ~25 (OATP substrate)
- Metformin: Kp_liver ~5 (OCT1 substrate)
- Diazepam: Kp_liver ~3-4 (lipophilic neutral)
"""
function calculate_kp_liver(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    liver::LiverProperties = DEFAULT_LIVER,
    # Transporter parameters
    transporter_effect::Float64 = 1.0,
    is_oatp_substrate::Bool = false,
    is_oct_substrate::Bool = false,
    is_pgp_substrate::Bool = false,
    is_bcrp_substrate::Bool = false
)
    P = 10^logP

    # Calculate ionization factors
    pH_p = 7.4   # Plasma pH
    pH_iw = liver.pH_iw  # Liver intracellular pH (7.0)

    X = 0.0  # Tissue ionization factor
    Y = 0.0  # Plasma ionization factor

    if !isnothing(pKa)
        if is_base
            X = 10^(pKa - pH_iw)  # Higher in acidic tissue
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
    AR = liver.albumin_ratio
    LR = liver.lipoprotein_ratio

    denom = max(1 + Y, 1e-10)

    # Plasma binding constant
    Ka_PR = max(0, min((1/fup - 1), 1000))

    # ============================================
    # WATER TERM (with ion trapping for bases)
    # ============================================
    water_term = f_ew + ((1 + X) / denom) * f_iw

    # ============================================
    # LIPID TERM (standard Rodgers-Rowland)
    # ============================================
    lipid_term = (P * f_nl + (0.3*P + 0.7) * f_npl) / denom

    # ============================================
    # TISSUE BINDING TERM (effective K_tissue approach)
    # Liver has 3x more APL than muscle (0.49% vs 0.15%)
    # ============================================
    tissue_term = 0.0
    if is_base && !isnothing(pKa) && pKa > 6.5
        K_tissue = calculate_effective_K_tissue_liver(logP)
        ion_factor = X / (1 + X)  # Fraction protonated
        tissue_term = K_tissue * ion_factor * (1 + X) / denom
    end

    # ============================================
    # LYSOSOMAL TRAPPING (CRITICAL FOR LIVER!)
    # Liver has 2.5% lysosomes vs 0.5% in muscle
    # This is a 5x multiplier on lysosomal contribution
    # ============================================
    lyso_term = 0.0
    if is_base && !isnothing(pKa) && pKa > 6.0
        lyso_term = calculate_lysosomal_trapping_liver(
            pKa=pKa,
            logP=logP,
            f_lysosome=liver.f_lysosome,
            pH_lysosome=liver.pH_lysosome,
            pH_cytosol=pH_iw
        )
    end

    # ============================================
    # TRANSPORTER EFFECT
    # ============================================
    if transporter_effect == 1.0 && (is_oatp_substrate || is_oct_substrate || is_pgp_substrate || is_bcrp_substrate)
        transporter_effect = estimate_transporter_effect(
            is_oatp_substrate=is_oatp_substrate,
            is_oct_substrate=is_oct_substrate,
            is_pgp_substrate=is_pgp_substrate,
            is_bcrp_substrate=is_bcrp_substrate,
            logP=logP
        )
    end

    # ============================================
    # CALCULATE TOTAL Kpu and Kp
    # ============================================
    if is_base && !isnothing(pKa) && pKa > 6.5
        # Strong base: water + lipid + tissue binding + lysosomal
        Kpu = water_term + lipid_term + tissue_term + lyso_term
        Kp = Kpu * fup * transporter_effect
    elseif is_acid
        # Acid: albumin binding dominates
        # Many acidic drugs are also OATP substrates!
        #
        # KEY INSIGHT: For highly bound acids, the sinusoidal (EW) space
        # provides a minimum Kp, and tissue albumin binding adds on top
        #
        # Warfarin example: logP 2.6, pKa 5.1, fup 0.01, observed Kp ~0.6
        # At plasma pH 7.4: ionized (anion)
        # Binds extensively to albumin both in plasma and tissue

        Kpu_acid = (f_ew +
                    ((1 + X) / denom) * f_iw +
                    (P * f_nl + (0.3*P + 0.7) * f_npl) / denom +
                    (Ka_PR * AR * (1 + X)) / denom)
        Kp = Kpu_acid * fup * transporter_effect

        # For highly bound acids (fup < 0.05), liver extracellular space
        # contains albumin that binds the drug. This creates a minimum Kp
        # based on the EW albumin content.
        #
        # Warfarin: fup=0.01, observed Kp=0.6
        # The EW space (16.1%) with albumin provides a reservoir
        # Kp_min ~ f_ew + (albumin-bound fraction in tissue)
        #        ~ 0.161 + 0.086 * Ka_PR * fup
        # For warfarin: Ka_PR = 99, so ~0.161 + 0.086*99*0.01 = 0.25
        # Still underpredicts. The issue is sinusoidal albumin is ~8.6% of plasma
        # but highly bound drugs will still bind to this tissue albumin
        if fup < 0.05
            # Highly bound acid - tissue albumin acts as reservoir
            # Tissue albumin concentration relative to plasma × bound fraction
            tissue_albumin_binding = AR * (1 - fup) * 0.5  # Fraction bound to tissue albumin
            ew_albumin_Kp = f_ew + tissue_albumin_binding
            Kp = max(Kp, ew_albumin_Kp) * transporter_effect
        end

        # For lipophilic acids, ensure minimum based on lipid partitioning
        if logP > 2.0
            lipid_min = 0.5 * P^0.5 * fup
            Kp = max(Kp, lipid_min) * transporter_effect
        end
    else
        # Neutral or weak base: lipoprotein binding + lipid partitioning
        #
        # KEY INSIGHT: Lipophilic neutrals partition strongly into liver
        # Liver has 2.4% phospholipids - significant binding capacity
        # The sinusoidal architecture (fenestrated, no basement membrane)
        # allows free access to hepatocyte membranes
        Kpu_neutral = (f_ew +
                       f_iw +
                       P * f_nl + (0.3*P + 0.7) * f_npl +
                       Ka_PR * LR)
        Kp = Kpu_neutral * fup * transporter_effect

        # For lipophilic neutrals (diazepam logP 2.8), the high membrane
        # surface area in liver creates significant partitioning
        # Empirical correction based on liver Kp literature:
        # - Diazepam (logP 2.8, fup 0.02): Kp ~3-4
        # - Standard R-R gives ~0.4, so need ~10x correction
        #
        # The issue: fup is so low that standard R-R underestimates
        # because it assumes all binding is to plasma proteins.
        # In reality, hepatic lipids compete for binding.
        if logP > 1.5
            # Hepatic lipid partitioning - liver-specific
            # Accounts for high phospholipid content (2.4%)
            hepatic_lipid_Kp = 0.1 * P^0.7  # Empirical
            Kp = max(Kp, hepatic_lipid_Kp)
        end
    end

    return max(Kp, 0.01)
end

"""
Calculate hepatic extraction ratio

The extraction ratio determines:
- First-pass metabolism
- Flow-limited vs capacity-limited clearance

Well-Stirred Model:
E = (fu × CLint) / (Q + fu × CLint)

Where:
- fu = fraction unbound in blood
- CLint = intrinsic clearance (L/min)
- Q = hepatic blood flow (L/min)

Returns: E (0 to 1)
- E > 0.7: High extraction (flow-limited)
- E 0.3-0.7: Intermediate
- E < 0.3: Low extraction (capacity-limited)
"""
function calculate_hepatic_extraction(;
    fub::Float64,           # Fraction unbound in blood
    CLint_L_min::Float64,   # Intrinsic clearance (L/min)
    Q_hepatic::Float64 = 1.5  # Hepatic blood flow (L/min)
)
    numerator = fub * CLint_L_min
    denominator = Q_hepatic + numerator

    E = numerator / denominator

    return clamp(E, 0.0, 0.99)
end

"""
Calculate first-pass bioavailability

F_hepatic = 1 - E

For oral drugs:
F_total = F_absorption × F_gut × F_hepatic

High extraction drugs (E > 0.7):
- Propranolol: F ~25-35%
- Morphine: F ~20-30%
- Lidocaine: F ~30-35% (given IV only)
- Nitroglycerin: F ~1-10% (given sublingual)

Low extraction drugs (E < 0.3):
- Warfarin: F ~99%
- Diazepam: F ~95%
- Theophylline: F ~95%
"""
function calculate_first_pass_bioavailability(;
    fub::Float64,
    CLint_L_min::Float64,
    Q_hepatic::Float64 = 1.5
)
    E = calculate_hepatic_extraction(
        fub=fub,
        CLint_L_min=CLint_L_min,
        Q_hepatic=Q_hepatic
    )

    F_hepatic = 1.0 - E

    return (extraction_ratio=E, F_hepatic=F_hepatic)
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
    transporter_effect::Float64 = 1.0,
    is_oatp_substrate::Bool = false,
    is_oct_substrate::Bool = false,
    is_pgp_substrate::Bool = false,
    is_bcrp_substrate::Bool = false
)
    Kp = calculate_kp_liver(
        logP=logP, logD=logD, fup=fup,
        pKa=pKa, is_base=is_base, is_acid=is_acid,
        transporter_effect=transporter_effect,
        is_oatp_substrate=is_oatp_substrate,
        is_oct_substrate=is_oct_substrate,
        is_pgp_substrate=is_pgp_substrate,
        is_bcrp_substrate=is_bcrp_substrate
    )

    contribution = Kp * liver_volume

    return (Kp=Kp, contribution_L=contribution, volume=liver_volume)
end

"""
SPECIAL CONSIDERATIONS FOR LIVER COMPARTMENT:

1. FIRST-PASS METABOLISM
   - Portal vein delivers gut-absorbed drugs directly to liver
   - High extraction drugs: >70% metabolized first pass
   - Oral bioavailability F = (1 - E)
   - Cirrhosis: portosystemic shunting → ↑ F

2. ZONATION EFFECTS
   - Zone 1 (periportal): Phase II conjugation dominant
   - Zone 3 (centrilobular): CYP450 highest, hypoxia sensitivity
   - Acetaminophen toxicity: Zone 3 (CYP2E1 + low glutathione)

3. TRANSPORTER-ENZYME INTERPLAY
   - OATP brings drugs into hepatocytes
   - CYP450s metabolize them
   - P-gp/MRP2/BCRP export metabolites to bile
   - "Conveyor belt" determines hepatic clearance

4. ENTEROHEPATIC CIRCULATION (EHC)
   - Glucuronide metabolites excreted in bile
   - Gut bacteria deconjugate → free drug reabsorbed
   - Secondary plasma peaks 4-8 hours post-dose
   - Examples: Morphine (M3G/M6G), Mycophenolate, Estradiol

5. GENETIC POLYMORPHISMS
   - SLCO1B1 (OATP1B1): *5, *15, *17 → statin toxicity risk
   - OCT1: Poor transporters → ↓ metformin efficacy
   - CYP2D6: Poor metabolizers (5-10% Caucasians)
   - UGT1A1: Gilbert's syndrome (unconjugated hyperbilirubinemia)

6. DISEASE EFFECTS
   - Cirrhosis: ↓ CYP activity, ↓ albumin, ↑ shunting
   - NAFLD/NASH: Altered transporter expression
   - Hepatitis: Enzyme induction/inhibition
   - Heart failure: Congestion → ↓ flow → ↓ clearance

7. DRUG-DRUG INTERACTIONS
   - Cyclosporine + Statin: OATP inhibition → ↑ statin AUC 8-15x
   - Rifampin: OATP inhibitor (acute) AND inducer (chronic)
   - Gemfibrozil: Glucuronide inhibits OATP1B1

8. LYSOSOMAL STORAGE DISEASES
   - Lysosomes trap cationic amphiphilic drugs
   - Can cause phospholipidosis (amiodarone, chloroquine)
   - Drug-induced lysosomal storage disease (DILSD)
"""

# Example drugs with documented liver Kp values
const LIVER_DRUG_EXAMPLES = Dict(
    # OATP substrates - high liver uptake
    "atorvastatin" => (logP=4.5, pKa=4.5, is_acid=true, is_oatp=true,
                       Kp_observed=25.0, note="OATP1B1/1B3 substrate"),
    "rosuvastatin" => (logP=-0.3, pKa=4.6, is_acid=true, is_oatp=true,
                       Kp_observed=10.0, note="Hydrophilic statin, OATP + BCRP"),
    "pravastatin"  => (logP=-0.8, pKa=4.2, is_acid=true, is_oatp=true,
                       Kp_observed=15.0, note="OATP substrate, not metabolized"),

    # OCT1 substrates
    "metformin"    => (logP=-1.5, pKa=11.5, is_base=true, is_oct=true,
                       Kp_observed=5.0, note="OCT1 substrate, biguanide"),

    # Lipophilic bases (lysosomal trapping)
    "imipramine"   => (logP=4.8, pKa=9.4, is_base=true,
                       Kp_observed=10.0, note="TCA, strong lysosomal trapping"),
    "chloroquine"  => (logP=4.6, pKa=10.1, is_base=true,
                       Kp_observed=100.0, note="Massive lysosomal accumulation"),
    "amiodarone"   => (logP=7.6, pKa=6.6, is_base=true,
                       Kp_observed=50.0, note="Very lipophilic, phospholipidosis"),

    # Beta-blockers
    "propranolol"  => (logP=3.5, pKa=9.5, is_base=true,
                       Kp_observed=4.0, note="High extraction, lysosomal"),

    # P-gp substrates (reduced liver Kp)
    "digoxin"      => (logP=1.3, pKa=nothing, is_neutral=true, is_pgp=true,
                       Kp_observed=0.5, note="P-gp substrate, effluxed"),

    # Neutral drugs
    "diazepam"     => (logP=2.8, pKa=3.4, is_base=false,
                       Kp_observed=3.5, note="Lipophilic neutral, lipoprotein binding"),
)

# ══════════════════════════════════════════════════════════════════════════════
# SOTA 2024 Q1+ ENHANCEMENTS
# ══════════════════════════════════════════════════════════════════════════════
#
# Based on Socratic Discussion Method - achieving Brain-level rigor for Liver
#
# New implementations:
# 1. Parallel-Tube and Dispersion hepatic clearance models
# 2. Biliary clearance (CLbiliary)
# 3. Michaelis-Menten transporter saturation kinetics
# 4. Enterohepatic recirculation as coupled ODE system
# 5. Quantitative CYP zonation (periportal vs centrilobular)
#
# References:
# - Pang KS, Rowland M (1977) J Pharmacokinet Biopharm - Parallel-tube model
# - Roberts MS, Rowland M (1986) J Pharmacokinet Biopharm - Dispersion model
# - Watanabe T et al. (2009) Drug Metab Dispos - OATP Km/Vmax
# - Jungermann K, Kietzmann T (1996) Hepatology - Liver zonation
# - Yang J et al. (2007) Curr Drug Metab - EHC modeling
# ══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 1. HEPATIC CLEARANCE MODELS - Beyond Well-Stirred
# ═══════════════════════════════════════════════════════════════════════════════

"""
Abstract type for hepatic clearance models.

Three classical models with different assumptions:
1. Well-Stirred: Instant equilibration (current implementation)
2. Parallel-Tube: Plug flow, concentration gradient
3. Dispersion: Intermediate, accounts for axial mixing
"""
abstract type HepaticClearanceModel end

struct WellStirredModel <: HepaticClearanceModel end
struct ParallelTubeModel <: HepaticClearanceModel end
struct DispersionModel <: HepaticClearanceModel
    dispersion_number::Float64  # DN = D/(v×L), typically 0.2-0.5
end

# Default dispersion number for liver (intermediate mixing)
DispersionModel() = DispersionModel(0.3)

"""
    calculate_hepatic_clearance(model, fub, CLint, Q)

Calculate hepatic clearance using specified model.

# Well-Stirred Model (current)
CLH = Q × E = Q × (fub × CLint) / (Q + fub × CLint)

# Parallel-Tube Model
E = 1 - exp(-fub × CLint / Q)
CLH = Q × E

# Dispersion Model
E = 1 - 4a / [(1+a)² × exp(a/DN) - (1-a)² × exp(-a/DN)]
where a = √(1 + 4×Rn×DN), Rn = fub×CLint/Q

KEY INSIGHT: The three models give DIFFERENT predictions for:
- High extraction drugs: Parallel-tube > Well-stirred > Dispersion
- Low extraction drugs: All models converge
- Intermediate: Dispersion is most physiologically realistic

Clinical Relevance:
- Well-stirred: Overestimates CLH for high-extraction drugs
- Parallel-tube: Overestimates extraction ratio
- Dispersion: Best matches in vivo data for most drugs

# Arguments
- `model`: HepaticClearanceModel type
- `fub`: Fraction unbound in blood
- `CLint`: Intrinsic clearance (L/min)
- `Q`: Hepatic blood flow (L/min), default 1.5

# Returns
NamedTuple with:
- `CLH`: Hepatic clearance (L/min)
- `E`: Extraction ratio
- `model_name`: String identifier
"""
function calculate_hepatic_clearance(
    model::WellStirredModel,
    fub::Float64,
    CLint::Float64,
    Q::Float64 = 1.5
)
    # E = (fub × CLint) / (Q + fub × CLint)
    numerator = fub * CLint
    E = numerator / (Q + numerator)
    CLH = Q * E

    return (CLH=CLH, E=E, model_name="Well-Stirred")
end

function calculate_hepatic_clearance(
    model::ParallelTubeModel,
    fub::Float64,
    CLint::Float64,
    Q::Float64 = 1.5
)
    # E = 1 - exp(-fub × CLint / Q)
    # This assumes drug concentration decreases exponentially along sinusoid

    exponent = -fub * CLint / Q
    E = 1.0 - exp(exponent)
    CLH = Q * E

    # For very high CLint, E approaches 1.0 (complete extraction)
    E = clamp(E, 0.0, 0.999)

    return (CLH=CLH, E=E, model_name="Parallel-Tube")
end

function calculate_hepatic_clearance(
    model::DispersionModel,
    fub::Float64,
    CLint::Float64,
    Q::Float64 = 1.5
)
    # Dispersion model (Roberts & Rowland 1986)
    # E = 1 - 4a / [(1+a)² × exp(a/DN) - (1-a)² × exp(-a/DN)]
    # where a = √(1 + 4×Rn×DN), Rn = fub×CLint/Q

    DN = model.dispersion_number
    Rn = fub * CLint / Q  # Efficiency number

    # Calculate 'a' parameter
    a = sqrt(1.0 + 4.0 * Rn * DN)

    # Calculate extraction ratio
    exp_pos = exp(a / DN)
    exp_neg = exp(-a / DN)

    numerator = 4.0 * a
    denominator = (1.0 + a)^2 * exp_pos - (1.0 - a)^2 * exp_neg

    E = 1.0 - numerator / denominator
    E = clamp(E, 0.0, 0.999)

    CLH = Q * E

    return (CLH=CLH, E=E, model_name="Dispersion(DN=$(DN))")
end

"""
Compare all three hepatic clearance models.

Useful for sensitivity analysis and understanding model impact.
"""
function compare_hepatic_models(fub::Float64, CLint::Float64, Q::Float64 = 1.5)
    ws = calculate_hepatic_clearance(WellStirredModel(), fub, CLint, Q)
    pt = calculate_hepatic_clearance(ParallelTubeModel(), fub, CLint, Q)
    dp = calculate_hepatic_clearance(DispersionModel(), fub, CLint, Q)

    return (
        well_stirred = ws,
        parallel_tube = pt,
        dispersion = dp,
        max_difference_pct = 100 * (maximum([ws.E, pt.E, dp.E]) - minimum([ws.E, pt.E, dp.E]))
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BILIARY CLEARANCE
# ═══════════════════════════════════════════════════════════════════════════════

"""
Parameters for biliary clearance calculation.

Biliary excretion is the primary route for:
- Glucuronide conjugates (MW > 400)
- Glutathione conjugates
- Drugs with active efflux (MRP2, P-gp, BCRP, BSEP)

Rule of 5 for Biliary Excretion (Watanabe et al.):
- MW > 400-500 (glucuronides add ~176 Da)
- Amphiphilic (logP 2-5)
- Carboxylic acid or conjugate
- MRP2/BCRP substrate
"""
struct BiliaryClearanceParams
    # Efflux transporter kinetics (Vmax in pmol/min/mg, Km in μM)
    MRP2_Vmax::Float64
    MRP2_Km::Float64
    BCRP_Vmax::Float64
    BCRP_Km::Float64
    Pgp_Vmax::Float64
    Pgp_Km::Float64
    BSEP_Vmax::Float64   # Bile salt export pump
    BSEP_Km::Float64

    # Bile flow parameters
    bile_flow_mL_min::Float64  # ~0.5-1.0 mL/min in adults

    # Drug-specific
    is_glucuronide::Bool
    is_glutathione_conj::Bool
    MW::Float64
end

# Default biliary parameters (healthy adult)
function BiliaryClearanceParams(;
    MRP2_Vmax::Float64 = 100.0,
    MRP2_Km::Float64 = 50.0,
    BCRP_Vmax::Float64 = 80.0,
    BCRP_Km::Float64 = 30.0,
    Pgp_Vmax::Float64 = 50.0,
    Pgp_Km::Float64 = 20.0,
    BSEP_Vmax::Float64 = 200.0,
    BSEP_Km::Float64 = 10.0,
    bile_flow_mL_min::Float64 = 0.7,
    is_glucuronide::Bool = false,
    is_glutathione_conj::Bool = false,
    MW::Float64 = 400.0
)
    return BiliaryClearanceParams(
        MRP2_Vmax, MRP2_Km, BCRP_Vmax, BCRP_Km,
        Pgp_Vmax, Pgp_Km, BSEP_Vmax, BSEP_Km,
        bile_flow_mL_min, is_glucuronide, is_glutathione_conj, MW
    )
end

"""
    calculate_biliary_clearance(C_hepatocyte, params; transporters)

Calculate biliary clearance using mechanistic transporter model.

CLbiliary = Σ (Vmax × C) / (Km + C) × scaling_factors

For conjugates (glucuronides, GSH):
- MRP2 is primary efflux pump
- MW > 400 required for significant biliary excretion

For parent drugs:
- P-gp and BCRP for lipophilic compounds
- Often minor compared to metabolism

# Arguments
- `C_hepatocyte`: Intracellular drug concentration (μM)
- `params`: BiliaryClearanceParams
- `is_mrp2_substrate`, etc.: Transporter substrate flags

# Returns
NamedTuple with:
- `CLbiliary`: Biliary clearance (mL/min)
- `fraction_biliary`: Estimated fraction of total hepatic CL
- `rate_limiting`: Which transporter is rate-limiting
"""
function calculate_biliary_clearance(
    C_hepatocyte::Float64,
    params::BiliaryClearanceParams;
    is_mrp2_substrate::Bool = false,
    is_bcrp_substrate::Bool = false,
    is_pgp_substrate::Bool = false,
    is_bsep_substrate::Bool = false,
    hepatocyte_protein_mg::Float64 = 40.0  # mg protein per g liver × 1800g
)
    # Michaelis-Menten for each transporter
    clearances = Float64[]
    transporters = String[]

    if is_mrp2_substrate || params.is_glucuronide || params.is_glutathione_conj
        # MRP2 - primary for conjugates
        v_mrp2 = params.MRP2_Vmax * C_hepatocyte / (params.MRP2_Km + C_hepatocyte)
        cl_mrp2 = v_mrp2 / C_hepatocyte * hepatocyte_protein_mg / 1000  # mL/min
        push!(clearances, cl_mrp2)
        push!(transporters, "MRP2")
    end

    if is_bcrp_substrate
        v_bcrp = params.BCRP_Vmax * C_hepatocyte / (params.BCRP_Km + C_hepatocyte)
        cl_bcrp = v_bcrp / C_hepatocyte * hepatocyte_protein_mg / 1000
        push!(clearances, cl_bcrp)
        push!(transporters, "BCRP")
    end

    if is_pgp_substrate
        v_pgp = params.Pgp_Vmax * C_hepatocyte / (params.Pgp_Km + C_hepatocyte)
        cl_pgp = v_pgp / C_hepatocyte * hepatocyte_protein_mg / 1000
        push!(clearances, cl_pgp)
        push!(transporters, "P-gp")
    end

    if is_bsep_substrate
        v_bsep = params.BSEP_Vmax * C_hepatocyte / (params.BSEP_Km + C_hepatocyte)
        cl_bsep = v_bsep / C_hepatocyte * hepatocyte_protein_mg / 1000
        push!(clearances, cl_bsep)
        push!(transporters, "BSEP")
    end

    # Total biliary clearance (sum of transporters)
    CLbiliary = sum(clearances)

    # Bile flow limitation
    # CLbiliary cannot exceed bile flow × unbound fraction in hepatocyte
    CLbiliary = min(CLbiliary, params.bile_flow_mL_min * 10)  # 10× safety margin

    # Determine rate-limiting transporter
    if isempty(clearances)
        rate_limiting = "None (not a biliary substrate)"
    else
        max_idx = argmax(clearances)
        rate_limiting = transporters[max_idx]
    end

    # MW-based biliary excretion likelihood
    # Glucuronides add ~176 Da, GSH conjugates add ~307 Da
    effective_MW = params.MW
    if params.is_glucuronide
        effective_MW += 176
    end
    if params.is_glutathione_conj
        effective_MW += 307
    end

    mw_factor = if effective_MW < 350
        0.1  # Minimal biliary
    elseif effective_MW < 500
        0.3 + 0.4 * (effective_MW - 350) / 150
    else
        0.7 + 0.3 * min((effective_MW - 500) / 200, 1.0)
    end

    CLbiliary *= mw_factor

    return (
        CLbiliary = CLbiliary,
        mw_factor = mw_factor,
        rate_limiting = rate_limiting,
        individual_CL = Dict(zip(transporters, clearances))
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MICHAELIS-MENTEN TRANSPORTER SATURATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Transporter kinetic parameters for saturable transport.

At low concentrations: CLuptake ≈ Vmax/Km (linear)
At high concentrations: CLuptake → 0 (saturation)

Clinical relevance:
- Statins: OATP1B1 saturation at high doses → nonlinear PK
- Rifampin: Saturates its own uptake → autoinhibition
- Metformin: OCT1 saturation affects hepatic accumulation
"""
struct TransporterKinetics
    name::String
    Vmax::Float64    # pmol/min/mg protein
    Km::Float64      # μM
    Pdiff::Float64   # Passive diffusion clearance (μL/min/mg)

    # Population variability
    CV_Vmax::Float64  # Coefficient of variation
    CV_Km::Float64

    # Genetic polymorphism effects
    polymorphism_scaling::Dict{String, Float64}
end

# Literature-based transporter kinetics
# References: Watanabe 2009, Hirano 2006, Noe 2007

const OATP1B1_KINETICS = TransporterKinetics(
    "OATP1B1",
    200.0,    # Vmax (pmol/min/mg) - typical for statins
    5.0,      # Km (μM) - varies by substrate: rosuvastatin ~5, atorvastatin ~10
    1.0,      # Pdiff - minimal passive for hydrophilic statins
    0.35,     # CV Vmax
    0.40,     # CV Km
    Dict(
        "wildtype" => 1.0,
        "*5/*5" => 0.3,      # SLCO1B1 c.521T>C - 70% reduced function
        "*5/*1" => 0.65,     # Heterozygous
        "*15/*15" => 0.25,   # *15 = *1b + *5 haplotype
        "*1b/*1b" => 1.2     # Slightly increased function
    )
)

const OATP1B3_KINETICS = TransporterKinetics(
    "OATP1B3",
    150.0,    # Vmax
    8.0,      # Km - generally higher than OATP1B1
    1.0,      # Pdiff
    0.40,
    0.45,
    Dict("wildtype" => 1.0, "reduced" => 0.5)
)

const OCT1_KINETICS = TransporterKinetics(
    "OCT1",
    500.0,    # Vmax - high for metformin
    200.0,    # Km - metformin Km ~200-500 μM
    0.5,      # Some passive for lipophilic cations
    0.45,
    0.50,
    Dict(
        "wildtype" => 1.0,
        "*2/*2" => 0.4,      # M420del - common reduced function
        "*3/*3" => 0.3,      # R61C
        "*4/*4" => 0.5,      # G401S
        "*5/*5" => 0.2       # G465R - severely reduced
    )
)

const PGP_KINETICS = TransporterKinetics(
    "P-gp",
    100.0,    # Vmax (efflux)
    15.0,     # Km
    5.0,      # Significant passive for P-gp substrates (lipophilic)
    0.50,
    0.55,
    Dict(
        "wildtype" => 1.0,
        "3435C>T" => 0.8,    # Common variant, modest effect
        "2677G>T/A" => 0.85
    )
)

"""
    calculate_saturable_uptake(C_plasma, kinetics; fu, polymorphism)

Calculate saturable transporter-mediated uptake clearance.

CLuptake = fu × (Vmax / (Km + C_unbound) + Pdiff)

At C << Km: CLuptake ≈ fu × (Vmax/Km + Pdiff) [linear]
At C >> Km: CLuptake ≈ fu × Pdiff [saturated, only passive]

# Arguments
- `C_plasma`: Total plasma concentration (μM)
- `kinetics`: TransporterKinetics struct
- `fu`: Fraction unbound in plasma
- `polymorphism`: Genetic variant key (e.g., "*5/*5")

# Returns
NamedTuple with clearance values and saturation fraction
"""
function calculate_saturable_uptake(
    C_plasma::Float64,
    kinetics::TransporterKinetics;
    fu::Float64 = 0.1,
    polymorphism::String = "wildtype",
    hepatocyte_scaling::Float64 = 40.0  # mg protein per g liver × scaling
)
    C_unbound = C_plasma * fu

    # Apply polymorphism scaling
    poly_factor = get(kinetics.polymorphism_scaling, polymorphism, 1.0)
    Vmax_adj = kinetics.Vmax * poly_factor

    # Michaelis-Menten + passive diffusion
    active_CL = Vmax_adj * C_unbound / (kinetics.Km + C_unbound)
    passive_CL = kinetics.Pdiff * C_unbound

    total_uptake_rate = active_CL + passive_CL  # pmol/min/mg

    # Convert to intrinsic clearance (μL/min/mg)
    CLint_uptake = (active_CL / C_unbound + kinetics.Pdiff)  # μL/min/mg

    # Scale to whole liver
    CLint_liver = CLint_uptake * hepatocyte_scaling / 1000  # mL/min

    # Saturation fraction (how close to Vmax)
    saturation_fraction = C_unbound / (kinetics.Km + C_unbound)

    return (
        CLint_uptake = CLint_uptake,
        CLint_liver = CLint_liver,
        saturation_fraction = saturation_fraction,
        active_fraction = active_CL / (active_CL + passive_CL + 1e-10),
        is_saturated = saturation_fraction > 0.5,
        polymorphism_effect = poly_factor
    )
end

"""
    calculate_saturable_efflux(C_hepatocyte, kinetics)

Calculate saturable efflux clearance (P-gp, MRP2, BCRP to bile).
"""
function calculate_saturable_efflux(
    C_hepatocyte::Float64,
    kinetics::TransporterKinetics;
    fu_hepatocyte::Float64 = 0.3,
    polymorphism::String = "wildtype"
)
    C_unbound = C_hepatocyte * fu_hepatocyte

    poly_factor = get(kinetics.polymorphism_scaling, polymorphism, 1.0)
    Vmax_adj = kinetics.Vmax * poly_factor

    # Efflux rate (Michaelis-Menten)
    efflux_rate = Vmax_adj * C_unbound / (kinetics.Km + C_unbound)

    # Intrinsic efflux clearance
    CLint_efflux = efflux_rate / C_unbound  # μL/min/mg

    saturation_fraction = C_unbound / (kinetics.Km + C_unbound)

    return (
        CLint_efflux = CLint_efflux,
        efflux_rate = efflux_rate,
        saturation_fraction = saturation_fraction,
        is_saturated = saturation_fraction > 0.5
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENTEROHEPATIC RECIRCULATION (EHC) ODE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

"""
Enterohepatic circulation state for ODE system.

EHC creates secondary plasma peaks and prolongs half-life for:
- Glucuronide conjugates (morphine, mycophenolate)
- Drugs excreted unchanged in bile
- Estrogens, bile acids

Mechanism:
1. Drug/metabolite excreted in bile
2. Enters intestine
3. Bacterial β-glucuronidase deconjugates glucuronides
4. Free drug reabsorbed
5. Returns to liver via portal vein
6. Cycle repeats (can recycle 2-5x)
"""
struct EHCState
    A_plasma::Float64      # Amount in plasma (mg)
    A_liver::Float64       # Amount in liver (mg)
    A_bile::Float64        # Amount in bile/gallbladder (mg)
    A_intestine::Float64   # Amount in intestinal lumen (mg)
    A_portal::Float64      # Amount in portal circulation (mg)
    A_eliminated::Float64  # Cumulative eliminated (mg)
end

"""
Parameters for EHC simulation.
"""
struct EnterohepatiCirculation
    # Rate constants (1/h)
    k_bile_secretion::Float64    # Liver → Bile
    k_gallbladder_empty::Float64 # Bile → Intestine (meal-triggered)
    k_intestinal_transit::Float64 # Movement through intestine
    k_deconjugation::Float64     # Glucuronide hydrolysis rate
    k_reabsorption::Float64      # Intestine → Portal
    k_portal_liver::Float64      # Portal → Liver
    k_elimination::Float64       # Hepatic elimination (non-biliary)
    k_renal::Float64             # Renal elimination from plasma

    # Fractions
    f_biliary::Float64           # Fraction of hepatic CL that is biliary
    f_deconjugation::Float64     # Fraction deconjugated in gut
    f_reabsorption::Float64      # Fraction reabsorbed (of deconjugated)

    # Volumes (L)
    V_plasma::Float64
    V_liver::Float64

    # Timing
    meal_times_h::Vector{Float64}  # Hours when meals trigger gallbladder emptying
end

# Default EHC parameters
function EnterohepatiCirculation(;
    k_bile_secretion::Float64 = 0.5,
    k_gallbladder_empty::Float64 = 2.0,
    k_intestinal_transit::Float64 = 0.2,
    k_deconjugation::Float64 = 1.0,
    k_reabsorption::Float64 = 0.5,
    k_portal_liver::Float64 = 5.0,
    k_elimination::Float64 = 0.1,
    k_renal::Float64 = 0.05,
    f_biliary::Float64 = 0.3,
    f_deconjugation::Float64 = 0.8,
    f_reabsorption::Float64 = 0.7,
    V_plasma::Float64 = 3.0,
    V_liver::Float64 = 1.8,
    meal_times_h::Vector{Float64} = [0.0, 6.0, 12.0]  # Breakfast, lunch, dinner
)
    return EnterohepatiCirculation(
        k_bile_secretion, k_gallbladder_empty, k_intestinal_transit,
        k_deconjugation, k_reabsorption, k_portal_liver,
        k_elimination, k_renal, f_biliary, f_deconjugation, f_reabsorption,
        V_plasma, V_liver, meal_times_h
    )
end

"""
    simulate_ehc_ode!(du, u, p, t)

ODE system for enterohepatic recirculation.

State vector u:
1. A_plasma - Amount in systemic plasma
2. A_liver - Amount in liver
3. A_bile - Amount in bile/gallbladder
4. A_intestine - Amount in intestinal lumen
5. A_portal - Amount in portal blood
6. A_eliminated - Cumulative elimination

This creates the characteristic secondary peaks seen with EHC drugs.
"""
function simulate_ehc_ode!(du, u, p, t)
    ehc = p.ehc

    A_plasma = u[1]
    A_liver = u[2]
    A_bile = u[3]
    A_intestine = u[4]
    A_portal = u[5]
    A_eliminated = u[6]

    # Meal-triggered gallbladder emptying
    # Increases emptying rate around meal times
    gb_empty_rate = ehc.k_gallbladder_empty
    for meal_time in ehc.meal_times_h
        time_from_meal = mod(t - meal_time, 24.0)
        if time_from_meal < 1.0  # Within 1 hour of meal
            gb_empty_rate *= 3.0  # Triple emptying rate
        end
    end

    # Fluxes
    flux_plasma_liver = 0.3 * A_plasma  # Distribution to liver
    flux_liver_plasma = 0.2 * A_liver   # Return from liver

    flux_liver_bile = ehc.k_bile_secretion * A_liver * ehc.f_biliary
    flux_bile_intestine = gb_empty_rate * A_bile
    flux_intestine_deconj = ehc.k_deconjugation * A_intestine * ehc.f_deconjugation
    flux_reabsorption = ehc.k_reabsorption * flux_intestine_deconj * ehc.f_reabsorption
    flux_intestine_feces = ehc.k_intestinal_transit * A_intestine * (1 - ehc.f_reabsorption)
    flux_portal_liver = ehc.k_portal_liver * A_portal

    flux_hepatic_elim = ehc.k_elimination * A_liver * (1 - ehc.f_biliary)
    flux_renal_elim = ehc.k_renal * A_plasma

    # ODEs
    du[1] = flux_liver_plasma - flux_plasma_liver - flux_renal_elim  # Plasma
    du[2] = flux_plasma_liver + flux_portal_liver - flux_liver_plasma - flux_liver_bile - flux_hepatic_elim  # Liver
    du[3] = flux_liver_bile - flux_bile_intestine  # Bile
    du[4] = flux_bile_intestine - flux_intestine_deconj - flux_intestine_feces  # Intestine
    du[5] = flux_reabsorption - flux_portal_liver  # Portal
    du[6] = flux_hepatic_elim + flux_renal_elim + flux_intestine_feces  # Eliminated

    return nothing
end

"""
Calculate AUC ratio with vs without EHC.

Drugs with significant EHC can have:
- 20-50% increase in AUC
- Secondary plasma peaks 4-8h post-dose
- Prolonged terminal half-life

Examples:
- Morphine-6-glucuronide: 30% AUC from EHC
- Mycophenolate: Secondary peak doubles exposure
- Ezetimibe glucuronide: Major EHC component
"""
function calculate_ehc_auc_ratio(ehc::EnterohepatiCirculation)
    # Simplified steady-state analysis
    # Fraction recycled = f_biliary × f_deconjugation × f_reabsorption
    f_recycled = ehc.f_biliary * ehc.f_deconjugation * ehc.f_reabsorption

    # AUC ratio = 1 / (1 - f_recycled) for infinite recycling
    # In practice, limited by intestinal transit
    n_cycles = 3  # Typical number of EHC cycles

    auc_ratio = 0.0
    for i in 0:n_cycles
        auc_ratio += f_recycled^i
    end

    secondary_peak_time = 1.0 / ehc.k_gallbladder_empty + 1.0 / ehc.k_intestinal_transit +
                          1.0 / ehc.k_reabsorption + 1.0 / ehc.k_portal_liver

    return (
        auc_ratio = auc_ratio,
        f_recycled_per_cycle = f_recycled,
        expected_secondary_peak_h = secondary_peak_time,
        clinical_significance = f_recycled > 0.3 ? "High" : (f_recycled > 0.15 ? "Moderate" : "Low")
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. HEPATIC ZONATION - CYP Expression Gradients
# ═══════════════════════════════════════════════════════════════════════════════

"""
Hepatic zonation - CYP enzyme expression by zone.

Liver lobule has THREE zones:
- Zone 1 (Periportal): High O₂, gluconeogenesis, Phase II conjugation
- Zone 2 (Mid-zonal): Transitional
- Zone 3 (Centrilobular): Low O₂, CYP450 HIGH, glycolysis, lipogenesis

KEY CLINICAL INSIGHTS:
- Acetaminophen toxicity: Zone 3 (CYP2E1 high + glutathione depleted)
- Alcohol damage: Zone 3 (CYP2E1, hypoxia)
- Viral hepatitis: Zone 1 preference (portal entry)

CYP Distribution (Zone 3 / Zone 1 ratios):
- CYP3A4: 2.0-2.5× higher in Zone 3
- CYP2E1: 3-5× higher in Zone 3 (toxicologically important!)
- CYP1A2: 1.5× higher in Zone 3
- CYP2D6: Relatively uniform
- UGT1A1: 1.5× higher in Zone 1 (conjugation)
- SULT: 2× higher in Zone 1
"""
struct ZonalCYPExpression
    zone::Symbol  # :periportal, :midzonal, :centrilobular

    # Relative expression (Zone 1 periportal = 1.0 reference)
    CYP3A4::Float64
    CYP3A5::Float64
    CYP2E1::Float64
    CYP2D6::Float64
    CYP2C9::Float64
    CYP2C19::Float64
    CYP1A2::Float64
    UGT1A1::Float64
    UGT2B7::Float64
    SULT1A1::Float64

    # Blood flow fraction
    blood_flow_fraction::Float64

    # Oxygen tension
    pO2_mmHg::Float64
end

# Zone-specific CYP expression based on Jungermann & Kietzmann (1996)
const PERIPORTAL_ZONE = ZonalCYPExpression(
    :periportal,
    1.0,   # CYP3A4 (reference)
    1.0,   # CYP3A5
    1.0,   # CYP2E1 (reference)
    1.0,   # CYP2D6 (relatively uniform)
    1.0,   # CYP2C9
    1.0,   # CYP2C19
    1.0,   # CYP1A2
    1.5,   # UGT1A1 (higher in Zone 1!)
    1.3,   # UGT2B7
    2.0,   # SULT1A1 (higher in Zone 1)
    0.33,  # Blood flow fraction (~1/3 each zone)
    65.0   # pO2 (high oxygen)
)

const MIDZONAL_ZONE = ZonalCYPExpression(
    :midzonal,
    1.5,   # CYP3A4
    1.3,   # CYP3A5
    2.0,   # CYP2E1
    1.0,   # CYP2D6
    1.2,   # CYP2C9
    1.2,   # CYP2C19
    1.25,  # CYP1A2
    1.0,   # UGT1A1
    1.0,   # UGT2B7
    1.0,   # SULT1A1
    0.33,
    45.0   # pO2 (intermediate)
)

const CENTRILOBULAR_ZONE = ZonalCYPExpression(
    :centrilobular,
    2.5,   # CYP3A4 (2.5× Zone 1!)
    2.0,   # CYP3A5
    5.0,   # CYP2E1 (5× Zone 1! - acetaminophen toxicity)
    1.0,   # CYP2D6 (uniform)
    1.5,   # CYP2C9
    1.5,   # CYP2C19
    1.5,   # CYP1A2
    0.7,   # UGT1A1 (lower - conjugation mainly Zone 1)
    0.8,   # UGT2B7
    0.5,   # SULT1A1 (lower)
    0.33,
    30.0   # pO2 (hypoxic - relevant for redox)
)

"""
Full hepatic zonation model.
"""
struct HepaticZonation
    periportal::ZonalCYPExpression
    midzonal::ZonalCYPExpression
    centrilobular::ZonalCYPExpression
end

HepaticZonation() = HepaticZonation(PERIPORTAL_ZONE, MIDZONAL_ZONE, CENTRILOBULAR_ZONE)

"""
    calculate_zonal_metabolism(zonation, enzyme, CLint_reference)

Calculate zone-weighted metabolic clearance.

For high-extraction drugs metabolized mainly by CYP3A4:
- Zone 3 (centrilobular) contributes most
- Portal blood "sees" Zone 1 first but Zone 3 has 2.5× enzyme

For CYP2E1 substrates (acetaminophen, halogenated anesthetics):
- Zone 3 is 5× more active
- Explains selective Zone 3 necrosis in overdose

# Arguments
- `zonation`: HepaticZonation struct
- `enzyme`: Symbol (:CYP3A4, :CYP2E1, etc.)
- `CLint_reference`: Intrinsic clearance at Zone 1 expression level

# Returns
Zone-weighted total CLint accounting for expression gradients
"""
function calculate_zonal_metabolism(
    zonation::HepaticZonation,
    enzyme::Symbol,
    CLint_reference::Float64
)
    # Get enzyme expression for each zone
    z1_expr = getfield(zonation.periportal, enzyme)
    z2_expr = getfield(zonation.midzonal, enzyme)
    z3_expr = getfield(zonation.centrilobular, enzyme)

    # Blood flow fractions
    f1 = zonation.periportal.blood_flow_fraction
    f2 = zonation.midzonal.blood_flow_fraction
    f3 = zonation.centrilobular.blood_flow_fraction

    # Zone-weighted CLint
    # Note: Blood flows through zones sequentially (portal → central)
    # So Zone 1 "sees" full concentration, Zone 3 sees depleted
    #
    # For simplicity, use weighted average here
    # Full model would use sequential extraction

    weighted_expression = f1 * z1_expr + f2 * z2_expr + f3 * z3_expr
    CLint_zonal = CLint_reference * weighted_expression

    return (
        CLint_total = CLint_zonal,
        zone1_contribution = f1 * z1_expr / weighted_expression,
        zone2_contribution = f2 * z2_expr / weighted_expression,
        zone3_contribution = f3 * z3_expr / weighted_expression,
        zone3_zone1_ratio = z3_expr / z1_expr
    )
end

"""
    calculate_zone_weighted_clearance(zonation, CLint_per_enzyme)

Calculate total hepatic CLint with all enzymes and zonation.

# Arguments
- `zonation`: HepaticZonation
- `CLint_per_enzyme`: Dict{Symbol, Float64} with CLint for each enzyme

# Example
```julia
CLint = Dict(:CYP3A4 => 10.0, :CYP2D6 => 2.0, :UGT1A1 => 5.0)
result = calculate_zone_weighted_clearance(HepaticZonation(), CLint)
```
"""
function calculate_zone_weighted_clearance(
    zonation::HepaticZonation,
    CLint_per_enzyme::Dict{Symbol, Float64}
)
    total_CLint = 0.0
    contributions = Dict{Symbol, Float64}()

    for (enzyme, CLint_ref) in CLint_per_enzyme
        try
            zonal = calculate_zonal_metabolism(zonation, enzyme, CLint_ref)
            total_CLint += zonal.CLint_total
            contributions[enzyme] = zonal.CLint_total
        catch
            # If enzyme not in zonation struct, use reference value
            total_CLint += CLint_ref
            contributions[enzyme] = CLint_ref
        end
    end

    return (
        CLint_total = total_CLint,
        contributions = contributions,
        fractional = Dict(k => v/total_CLint for (k,v) in contributions)
    )
end

"""
Predict acetaminophen toxicity risk based on zonation.

Zone 3 CYP2E1 converts APAP to NAPQI (toxic).
Zone 1 SULT conjugates APAP (safe).

Risk factors:
- Alcohol (induces CYP2E1)
- Fasting (depletes glutathione)
- High dose (saturates sulfation)
"""
function predict_apap_toxicity_zone(
    dose_mg::Float64,
    is_alcoholic::Bool = false,
    is_fasting::Bool = false
)
    zonation = HepaticZonation()

    # CYP2E1 (toxic pathway) - Zone 3 dominant
    cyp2e1_zone3 = zonation.centrilobular.CYP2E1  # 5×

    # SULT (safe pathway) - Zone 1 dominant
    sult_zone1 = zonation.periportal.SULT1A1  # 2×

    # Baseline toxic/safe ratio
    toxic_safe_ratio = cyp2e1_zone3 / sult_zone1

    # Risk modifiers
    if is_alcoholic
        toxic_safe_ratio *= 2.0  # CYP2E1 induction
    end
    if is_fasting
        toxic_safe_ratio *= 1.5  # Glutathione depletion, favors CYP
    end

    # Dose-dependent (SULT saturates at high doses)
    if dose_mg > 3000
        toxic_safe_ratio *= 1.5
    elseif dose_mg > 4000
        toxic_safe_ratio *= 2.5
    end

    risk_level = if toxic_safe_ratio < 3
        "Low"
    elseif toxic_safe_ratio < 5
        "Moderate"
    elseif toxic_safe_ratio < 8
        "High"
    else
        "Severe - Zone 3 necrosis likely"
    end

    return (
        toxic_safe_ratio = toxic_safe_ratio,
        risk_level = risk_level,
        zone3_cyp2e1 = cyp2e1_zone3,
        zone1_sult = sult_zone1,
        recommendation = toxic_safe_ratio > 5 ?
            "Consider N-acetylcysteine if presenting within 8h" :
            "Standard monitoring"
    )
end

end # module
