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

end # module
