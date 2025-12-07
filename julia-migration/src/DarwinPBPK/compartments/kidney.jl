# KIDNEY COMPARTMENT MODEL (ENHANCED)
# ====================================
#
# The kidney is UNIQUE among organs - both a distribution compartment AND
# the primary elimination organ for many drugs. This dual role requires
# careful modeling.
#
# ══════════════════════════════════════════════════════════════════════════
# KIDNEY PHYSIOLOGY DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════
#
# 1. HIGHEST BLOOD FLOW PER GRAM OF TISSUE
#    ─────────────────────────────────────
#    - 20-25% of cardiac output (~1200 mL/min) to 300g organ!
#    - Blood flow: 4 mL/g/min (vs liver: 0.8 mL/g/min)
#    - This extreme perfusion is for FILTRATION, not oxygenation
#    - 99% of filtered water/solutes reabsorbed!
#
# 2. UNIQUE ARCHITECTURE: THE NEPHRON
#    ─────────────────────────────────
#    Each kidney has ~1 million nephrons:
#
#    a) GLOMERULUS
#       - Capillary tuft with 3-layer filtration barrier:
#         • Fenestrated endothelium (70-100 nm pores)
#         • Basement membrane (selective for charge/size)
#         • Podocyte foot processes (slit diaphragm, 4-11 nm)
#       - GFR: 180 L/day (125 mL/min)
#       - Only UNBOUND drug filters!
#       - Size cutoff: ~65 kDa (albumin barely passes)
#       - Charge selectivity: anions rejected, cations pass easier
#
#    b) PROXIMAL TUBULE (65% of nephron length)
#       - PRIMARY site of drug transport!
#       - Brush border membrane (300x surface area increase)
#       - HIGH transporter expression:
#         • Basolateral: OAT1, OAT3, OCT2 (uptake from blood)
#         • Apical: OAT4, MATE1, MATE2-K, P-gp, MRP2/4 (efflux to urine)
#       - Active secretion can exceed GFR by 5-10x!
#       - Also site of passive reabsorption (lipophilic drugs)
#
#    c) LOOP OF HENLE
#       - Descending: water reabsorption (concentrates tubular fluid)
#       - Ascending: NaCl reabsorption (dilutes fluid)
#       - Loop diuretics (furosemide) act here
#
#    d) DISTAL TUBULE & COLLECTING DUCT
#       - Fine-tuning of Na+/K+/H+ balance
#       - Final site of passive reabsorption
#       - pH-dependent trapping of ionizable drugs
#       - Urine pH can vary 4.5-8.0 → 10,000-fold ionization change!
#
# 3. HIGHEST ACIDIC PHOSPHOLIPID CONTENT (0.5%!)
#    ────────────────────────────────────────────
#    - 3x more than muscle (0.15%)
#    - ~2x liver (0.25% effective due to distribution)
#    - Strong cationic drug binding to proximal tubule membranes
#    - Clinical relevance: Aminoglycoside nephrotoxicity!
#      • Gentamicin binds to brush border → endocytosis → lysosomal damage
#      • Kp_kidney for aminoglycosides: 10-30x (massive accumulation)
#
# 4. PROXIMAL TUBULE LYSOSOMES
#    ─────────────────────────
#    - 1.5-2% lysosomal volume fraction (between muscle and liver)
#    - pH 4.8: Strong trapping of basic drugs
#    - Aminoglycosides accumulate in lysosomes → phospholipidosis
#    - Combined with high APL = MAJOR cation reservoir
#
# 5. TRANSPORTERS: THE SECRETORY MACHINERY
#    ─────────────────────────────────────
#    ┌────────────────────────────────────────────────────────────────┐
#    │                    PROXIMAL TUBULE CELL                        │
#    │                                                                │
#    │  BLOOD                                    URINE                │
#    │  (Basolateral)                           (Apical/Luminal)      │
#    │                                                                │
#    │  ───OAT1───→                  ←───OAT4───                     │
#    │  ───OAT3───→    ORGANIC      ←───MRP2───                      │
#    │               ANIONS          ←───MRP4───                      │
#    │               (acids)         ←───BCRP───                      │
#    │                                                                │
#    │  ───OCT2───→                  ←───MATE1──                     │
#    │               ORGANIC         ←───MATE2-K                      │
#    │               CATIONS         ←───P-gp───                      │
#    │               (bases)         ←───OCT2(flip?)                  │
#    │                                                                │
#    └────────────────────────────────────────────────────────────────┘
#
#    Key substrates:
#    - OAT1: PAH, cidofovir, tenofovir, adefovir, methotrexate
#    - OAT3: Penicillins, cephalosporins, NSAIDs, furosemide, statins
#    - OCT2: Metformin, cisplatin, cimetidine, memantine
#    - MATE1/2-K: Metformin (works WITH OCT2 for secretion)
#
# 6. RENAL DRUG ELIMINATION
#    ──────────────────────
#    CLrenal = CLfiltration + CLsecretion - CLreabsorption
#
#    a) Filtration (passive, unbound drug only)
#       CLfiltration = fup × GFR
#       - Maximum ~120 mL/min if fup = 1
#       - Highly bound drugs (fup < 0.01): CLfilt < 1.2 mL/min
#
#    b) Secretion (active, transporter-mediated)
#       Can EXCEED filtration clearance!
#       - PAH secretion ratio: 5-6x (used to measure renal plasma flow)
#       - Penicillin: CLsecretion >> CLfiltration
#       - Metformin: CLrenal = 400-600 mL/min (OCT2 + MATE)
#
#    c) Reabsorption (passive, depends on lipophilicity + ionization)
#       - Lipophilic drugs: extensive reabsorption
#       - Urine pH manipulation: alkalinize → trap acids → ↑ excretion
#       - This is why furosemide (acid, logP 2.0) has CLrenal > GFR
#         despite being reabsorbed: OAT-mediated secretion wins!
#
# 7. CLINICAL IMPLICATIONS
#    ─────────────────────
#    a) Nephrotoxicity hotspots:
#       - Aminoglycosides: APL binding + lysosomal trapping → toxicity
#       - Cisplatin: OCT2 uptake into proximal tubule cells
#       - NSAIDs: COX inhibition → ↓ prostaglandin → ↓ renal blood flow
#       - Contrast agents: direct tubular toxicity + vasoconstriction
#
#    b) Renal impairment effects:
#       - ↓ GFR: dose reduction for filtration-cleared drugs
#       - ↓ secretion: transporter expression reduced in CKD
#       - ↑ drug accumulation: especially cationic drugs (OCT2)
#       - ↓ protein binding: uremic toxins displace drugs
#
#    c) Drug-drug interactions:
#       - Probenecid + Penicillin: OAT inhibition → ↑ penicillin levels
#       - Cimetidine + Metformin: OCT2/MATE inhibition → ↑ metformin
#       - Trimethoprim + Procainamide: OCT2 inhibition → ↑ procainamide
#
# References:
# - Morrissey et al. 2013: Renal transporters in drug development
# - Masereeuw & Russel 2010: Mechanisms and clinical implications
# - Launay-Vacher et al. 2006: Renal handling of drugs in elderly
# - Rodgers & Rowland 2006: Mechanistic Kp prediction
# - Schmitt et al. 2021: Lysosomal trapping extension
# ══════════════════════════════════════════════════════════════════════════

module KidneyCompartment

export KidneyProperties, calculate_kp_kidney, calculate_kidney_contribution
export estimate_renal_clearance_contribution, estimate_transporter_effect_kidney
export calculate_lysosomal_trapping_kidney, calculate_effective_K_tissue_kidney
export calculate_tubular_secretion_ratio, calculate_reabsorption_fraction
export estimate_nephrotoxicity_risk
# SOTA 2024 Q1+ exports - Michaelis-Menten Transporter Kinetics
export RenalTransporterKinetics, calculate_saturable_secretion
export calculate_saturable_reabsorption, calculate_transporter_ddi
export OAT1_RENAL_KINETICS, OAT3_RENAL_KINETICS, OCT2_RENAL_KINETICS
export MATE1_RENAL_KINETICS, MATE2K_RENAL_KINETICS, URAT1_KINETICS
export calculate_complete_renal_clearance_mm, estimate_renal_ddi_risk
export TransporterDDIResult, RenalClearanceComponents
# SOTA 2020-2024 exports
export CKDStage, CKD_G1, CKD_G2, CKD_G3a, CKD_G3b, CKD_G4, CKD_G5, CKD_G5D
export PatientKidneyStatus, TransporterPolymorphisms, WILDTYPE_TRANSPORTERS
export scale_transporter_activity_ckd, adjust_for_polymorphisms
export calculate_uremic_fup_adjustment, age_adjusted_kidney_parameters
export calculate_kp_kidney_ckd, estimate_renal_clearance_ckd
export get_ckd_stage, predict_cisplatin_nephrotoxicity_risk

"""
Kidney physiological properties

Reference values for 70kg adult:
- Volume: 0.31 L (both kidneys, 0.4% body weight)
- Blood flow: 1.2 L/min (20-25% cardiac output!)
- GFR: ~120 mL/min (125 mL/min in young adults)

Tissue composition from Rodgers & Rowland 2006:
- Highest acidic phospholipid content of any tissue (0.5%)
- High phospholipid content (2.4%) for membrane-rich tubular cells
"""
struct KidneyProperties
    volume_L::Float64           # Kidney volume (both kidneys)
    blood_flow_L_min::Float64   # Renal blood flow
    f_neutral_lipid::Float64    # Neutral lipids
    f_phospholipid::Float64     # Neutral phospholipids (HIGH)
    f_acidic_pl::Float64        # Acidic phospholipids (HIGHEST!)
    f_water_iw::Float64         # Intracellular water
    f_water_ew::Float64         # Extracellular water
    albumin_ratio::Float64      # Albumin tissue/plasma
    lipoprotein_ratio::Float64  # LP tissue/plasma
    pH_iw::Float64              # Intracellular pH
    pH_tubular::Float64         # Tubular urine pH (variable 4.5-8.0)
    f_lysosome::Float64         # Lysosomal volume fraction
    pH_lysosome::Float64        # Lysosomal pH
    GFR_mL_min::Float64         # Glomerular filtration rate
    # Transporter abundances (pmol/mg protein) - proximal tubule
    OAT1::Float64               # Organic anion transporter 1 (basolateral)
    OAT3::Float64               # Organic anion transporter 3 (basolateral)
    OCT2::Float64               # Organic cation transporter 2 (basolateral)
    MATE1::Float64              # Multidrug and toxin extrusion 1 (apical)
    MATE2K::Float64             # MATE2-K (apical, kidney-specific)
    MRP2::Float64               # MRP2 (apical)
    MRP4::Float64               # MRP4 (apical)
    P_gp::Float64               # P-gp (apical)
end

# Default for 70kg adult with normal renal function
const DEFAULT_KIDNEY = KidneyProperties(
    0.31,     # volume (L) - both kidneys
    1.2,      # blood flow (L/min) - 20-25% cardiac output!
    0.013,    # neutral lipids
    0.0244,   # neutral phospholipids (HIGH - membrane-rich tubules)
    0.00503,  # acidic phospholipids (HIGHEST of all tissues - 3x muscle!)
    0.483,    # intracellular water
    0.273,    # extracellular water (high - filtration)
    0.130,    # albumin ratio
    0.137,    # lipoprotein ratio
    7.0,      # intracellular pH
    6.0,      # tubular urine pH (can range 4.5-8.0!)
    0.018,    # lysosomal volume fraction (1.8% - between muscle and liver)
    4.8,      # lysosomal pH
    120.0,    # GFR (mL/min) - young adult value
    # Transporter abundances (pmol/mg protein)
    4.0,      # OAT1 - PAH, antivirals, methotrexate
    2.5,      # OAT3 - penicillins, NSAIDs, diuretics
    6.0,      # OCT2 (HIGH!) - metformin, cisplatin
    2.5,      # MATE1 - works with OCT2 for cation secretion
    1.5,      # MATE2-K - kidney-specific MATE
    1.5,      # MRP2 - glucuronides, anions
    2.0,      # MRP4 - cyclic nucleotides, antivirals
    0.8       # P-gp - lipophilic cations
)

"""
Calculate effective tissue binding constant K_tissue for KIDNEY

Kidney has the HIGHEST acidic phospholipid content (0.5% vs 0.15% muscle).
This 3.3x higher APL means STRONG binding of cationic drugs.

CLINICAL SIGNIFICANCE:
- Aminoglycosides (gentamicin, amikacin): Kp_kidney 10-30x
  Bind to brush border → endocytosis → lysosomal accumulation → toxicity
- Polymyxins: Similar mechanism, dose-limiting nephrotoxicity
- Vancomycin: Proximal tubule accumulation

The high K_tissue combined with lysosomal trapping creates a
"double whammy" for basic drugs in kidney.
"""
function calculate_effective_K_tissue_kidney(logP::Float64)
    # Kidney has ~3.3x the APL of muscle, highest of any tissue
    # This creates the strongest membrane binding for cations

    if logP < 0.5
        # Very hydrophilic: minimal membrane access
        # But aminoglycosides (logP -3 to -4) still bind via electrostatics!
        # Electrostatic binding to brush border doesn't require membrane partitioning
        return 0.5  # Baseline electrostatic contribution
    elseif logP < 1.0
        # Transition: some membrane access
        return 0.5 + 0.5 * (logP - 0.5)  # 0.5 to 0.75
    elseif logP < 2.0
        return 0.75 + 2.5 * (logP - 1.0)  # 0.75 to 3.25
    elseif logP < 3.0
        return 3.25 + 5.0 * (logP - 2.0)  # 3.25 to 8.25
    elseif logP < 4.0
        return 8.25 + 8.0 * (logP - 3.0)  # 8.25 to 16.25
    elseif logP < 5.0
        return 16.25 + 10.0 * (logP - 4.0)  # 16.25 to 26.25
    else
        # Very lipophilic: plateau (membrane saturation)
        return 28.0  # Higher than liver (22) due to more APL
    end
end

"""
Calculate effective K_tissue for AMINOGLYCOSIDES and POLYBASIC drugs

Aminoglycosides are unique:
- Very hydrophilic (logP -3 to -4)
- Multiple positive charges (2-5 amino groups)
- DON'T require membrane partitioning for binding!
- Bind electrostatically to anionic brush border phospholipids
- Then endocytosed via megalin receptor

This creates a special case where hydrophilic polycations
accumulate massively in kidney.

Examples:
- Gentamicin: 5 amino groups, logP -3.1, Kp_kidney ~15
- Tobramycin: 5 amino groups, logP -4.3, Kp_kidney ~12
- Amikacin: 5 amino groups, logP -4.8, Kp_kidney ~10
"""
function calculate_K_tissue_aminoglycoside(;
    n_positive_charges::Int = 4,
    MW::Float64 = 500.0
)
    # Aminoglycosides bind via ELECTROSTATIC interactions to brush border
    # Each positive charge contributes significantly
    # Gentamicin (5 charges): Kp ~15
    # Need: K_tissue × f_apl × fup = Kp
    # For gentamicin: K_tissue × 0.005 × 0.9 = 15 → K_tissue ~3300

    # Base affinity scales with charge squared (electrostatic)
    # Plus cooperative binding effects
    base_affinity = 150.0  # Calibrated for gentamicin
    charge_factor = (n_positive_charges / 4.0)^1.8  # Normalized to 4 charges

    # Molecular size: larger = more steric hindrance, less binding
    # But also more surface area for electrostatic contact
    size_factor = (MW / 500.0)^0.3  # Mild size dependence

    K_tissue = base_affinity * charge_factor * size_factor

    return min(K_tissue, 500.0)  # Higher cap for polycations
end

"""
Calculate lysosomal trapping factor for KIDNEY

Kidney has 1.8% lysosomal volume fraction (between muscle 0.5% and liver 2.5%).
However, the HIGH APL content works synergistically with lysosomal trapping:

1. Drug binds to brush border (electrostatic for polycations)
2. Endocytosed into early endosomes
3. Trafficked to lysosomes
4. Protonated at pH 4.8 → trapped!

For aminoglycosides, this pathway is well-characterized:
- Megalin-mediated endocytosis
- Lysosomal accumulation
- Eventually causes lysosomal membrane permeabilization
- Releases cathepsins → apoptosis → nephrotoxicity

The combination of APL binding + lysosomal trapping creates
Kp_kidney values that far exceed other tissues for strong bases.
"""
function calculate_lysosomal_trapping_kidney(;
    pKa::Float64,
    logP::Float64,
    f_lysosome::Float64 = 0.018,  # 1.8% for kidney
    pH_lysosome::Float64 = 4.8,
    pH_cytosol::Float64 = 7.0
)
    # Non-bases: no trapping
    if pKa < 6.0
        return 0.0
    end

    # Ionization ratio in lysosome vs cytosol
    ionized_lyso = 10^(pKa - pH_lysosome)
    ionized_cyto = 10^(pKa - pH_cytosol)

    # Accumulation ratio
    accumulation = (1 + ionized_lyso) / (1 + ionized_cyto)

    # Permeability factor
    # For kidney, even hydrophilic drugs can reach lysosomes via endocytosis!
    # This is different from liver where membrane permeation is required.
    permeability_factor = if logP < -2.0
        # Very hydrophilic polycations (aminoglycosides)
        # Can still reach lysosomes via receptor-mediated endocytosis!
        0.25  # Reduced but NOT zero
    elseif logP < 0.0
        0.25 + 0.25 * (logP + 2.0) / 2.0  # 0.25 to 0.5
    elseif logP < 1.0
        0.5 + 0.2 * logP  # 0.5 to 0.7
    elseif logP < 2.0
        0.7 + 0.15 * (logP - 1.0)  # 0.7 to 0.85
    elseif logP < 3.0
        0.85 + 0.1 * (logP - 2.0)  # 0.85 to 0.95
    else
        0.90  # Very lipophilic can escape lysosomes
    end

    # Final lysosomal contribution
    lyso_contribution = f_lysosome * accumulation * permeability_factor

    # Strong bases (pKa > 9) like aminoglycosides
    if pKa > 9.0
        strong_base_factor = 1.0 + 1.5 * (pKa - 9.0)
        lyso_contribution *= strong_base_factor
    end

    return lyso_contribution
end

"""
Estimate transporter effect on kidney Kp and clearance

Kidney transporters work in coordinated fashion:

ANION SECRETION PATHWAY:
Blood → [OAT1/OAT3] → Cell → [OAT4/MRP2/MRP4/BCRP] → Urine

CATION SECRETION PATHWAY:
Blood → [OCT2] → Cell → [MATE1/MATE2-K/P-gp] → Urine

Returns:
- kp_effect: Multiplier for tissue accumulation
- cl_effect: Multiplier for renal clearance (secretion)
"""
function estimate_transporter_effect_kidney(;
    is_oat1_substrate::Bool = false,
    is_oat3_substrate::Bool = false,
    is_oct2_substrate::Bool = false,
    is_mate_substrate::Bool = false,
    is_pgp_substrate::Bool = false,
    is_anion::Bool = false,
    is_cation::Bool = false,
    MW::Float64 = 400.0,
    logP::Float64 = 2.0,
    PSA::Float64 = 80.0
)
    kp_effect = 1.0
    cl_effect = 1.0

    # ═══════════════════════════════════════════════════════════════
    # ORGANIC ANION TRANSPORT (OAT1/OAT3)
    # ═══════════════════════════════════════════════════════════════
    # OAT substrates accumulate in proximal tubule cells
    # then are secreted into urine via apical transporters
    #
    # Classic substrates:
    # - PAH (para-aminohippurate): used to measure renal plasma flow
    # - Penicillins, cephalosporins
    # - Tenofovir, cidofovir, adefovir (antivirals - nephrotoxic!)
    # - Methotrexate
    # - Furosemide, hydrochlorothiazide
    # - NSAIDs (competitive inhibitors)

    if is_oat1_substrate
        # OAT1: high affinity for small organic anions
        # Creates cellular accumulation → potential toxicity
        # But OAT substrates are also SECRETED, so Kp effect is moderate
        # Tenofovir: Kp ~4, mainly from transporter uptake
        if logP < 0
            kp_effect *= 4.0  # Hydrophilic: totally transporter-dependent
        elseif logP < 2.0
            kp_effect *= 3.0
        else
            kp_effect *= 2.0  # Lipophilic: some passive
        end
        cl_effect *= 4.0  # Major secretory pathway
    end

    if is_oat3_substrate
        # OAT3: broader substrate specificity
        # Furosemide: Kp ~2.5, highly bound but OAT3 creates uptake
        if logP < 0
            kp_effect *= 3.5
        elseif logP < 2.0
            kp_effect *= 2.5
        else
            kp_effect *= 2.0
        end
        cl_effect *= 3.0
    end

    # Predict OAT substrate likelihood for unknown drugs
    if !is_oat1_substrate && !is_oat3_substrate && is_anion && MW > 200 && MW < 600
        # Small organic anions are likely OAT substrates
        if logP < 2.0 && PSA > 60
            kp_effect *= 2.0
            cl_effect *= 2.0
        end
    end

    # ═══════════════════════════════════════════════════════════════
    # ORGANIC CATION TRANSPORT (OCT2 + MATE)
    # ═══════════════════════════════════════════════════════════════
    # OCT2 mediates basolateral uptake
    # MATE1/MATE2-K mediate apical efflux to urine
    # Both required for efficient cation secretion!
    #
    # Classic substrates:
    # - Metformin: CLrenal 400-600 mL/min (4-5x GFR!)
    # - Cisplatin: OCT2 uptake → nephrotoxicity
    # - Cimetidine, ranitidine
    # - Memantine, amantadine
    # - Procainamide, pilsicainide

    if is_oct2_substrate
        # OCT2: major uptake transporter for cations
        # Metformin: Kp ~5, mostly water distribution with some OCT2 contribution
        # KEY INSIGHT: OCT2 + MATE creates a "flow-through" system
        # Drug is taken up by OCT2, then immediately secreted by MATE
        # Net effect on Kp is MINIMAL - effect is mainly on CLEARANCE
        if logP < 0
            kp_effect *= 1.3  # Hydrophilic: minimal Kp effect, mostly clearance
        elseif logP < 2.0
            kp_effect *= 1.2
        else
            kp_effect *= 1.15
        end

        if is_mate_substrate
            # OCT2 + MATE = efficient secretion (high clearance)
            # Drug flows THROUGH tubule cells, minimal accumulation
            cl_effect *= 4.0  # Metformin: ~4x GFR
        else
            # OCT2 only: accumulates in cell (cisplatin!)
            # Potential toxicity - much higher Kp effect
            cl_effect *= 2.0
            kp_effect *= 2.5  # Significant accumulation without efflux
        end
    elseif is_mate_substrate
        # MATE without OCT2: unusual
        cl_effect *= 1.5
    end

    # Predict OCT2 substrate likelihood
    if !is_oct2_substrate && is_cation && MW < 500
        if logP < 3.0
            kp_effect *= 1.5
            cl_effect *= 1.5
        end
    end

    # ═══════════════════════════════════════════════════════════════
    # EFFLUX TRANSPORTERS (P-gp)
    # ═══════════════════════════════════════════════════════════════
    if is_pgp_substrate
        # P-gp in kidney: apical, pumps drugs into urine
        # Reduces cellular accumulation slightly
        kp_effect *= 0.85
        cl_effect *= 1.3  # Adds to secretion
    end

    return (kp_effect=kp_effect, cl_effect=cl_effect)
end

"""
Calculate kidney:plasma partition coefficient (ENHANCED MODEL)

This model incorporates:
1. Standard Rodgers-Rowland terms (water, lipids)
2. Effective K_tissue for acidic phospholipid binding (HIGHEST in body!)
3. Lysosomal trapping (1.8% volume fraction)
4. Transporter-mediated uptake (OAT, OCT2)
5. Special handling for aminoglycosides

VALIDATION TARGETS (from literature):
- Gentamicin: Kp ~15 (polycationic aminoglycoside)
- Metformin: Kp ~5 (OCT2 substrate)
- Propranolol: Kp ~3-4 (lipophilic base)
- Furosemide: Kp ~2-3 (OAT3 substrate)
- Digoxin: Kp ~0.5 (neutral, P-gp)
- Warfarin: Kp ~0.3 (highly bound acid)
"""
function calculate_kp_kidney(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    is_aminoglycoside::Bool = false,
    n_positive_charges::Int = 1,
    kidney::KidneyProperties = DEFAULT_KIDNEY,
    # Transporter parameters
    transporter_effect::Float64 = 1.0,
    is_oat1_substrate::Bool = false,
    is_oat3_substrate::Bool = false,
    is_oct2_substrate::Bool = false,
    is_mate_substrate::Bool = false,
    is_pgp_substrate::Bool = false,
    MW::Float64 = 400.0,
    PSA::Float64 = 80.0
)
    P = 10^logP

    # Calculate ionization factors
    pH_p = 7.4
    pH_iw = kidney.pH_iw

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

    # Tissue composition - note HIGHEST APL!
    f_ew = kidney.f_water_ew
    f_iw = kidney.f_water_iw
    f_nl = kidney.f_neutral_lipid
    f_npl = kidney.f_phospholipid
    f_apl = kidney.f_acidic_pl  # 0.5% - HIGHEST of all tissues!
    AR = kidney.albumin_ratio
    LR = kidney.lipoprotein_ratio

    denom = max(1 + Y, 1e-10)

    # Plasma binding constant
    Ka_PR = max(0, min((1/fup - 1), 1000))

    # ════════════════════════════════════════════════════════════
    # SPECIAL CASE: AMINOGLYCOSIDES
    # ════════════════════════════════════════════════════════════
    if is_aminoglycoside
        # Aminoglycosides are unique:
        # - Very hydrophilic (logP -3 to -4)
        # - Multiple positive charges (4-5)
        # - Don't follow standard R-R model!
        # - Bind electrostatically to brush border
        # - Endocytosed via megalin → lysosomal accumulation
        #
        # Target: Gentamicin Kp ~15, Tobramycin ~12, Amikacin ~10

        K_tissue_AG = calculate_K_tissue_aminoglycoside(
            n_positive_charges=n_positive_charges,
            MW=MW
        )

        # Water term (freely distributed in EW and IW)
        water_term = f_ew + f_iw  # ~0.75

        # Electrostatic binding to APL (brush border membrane)
        # This is the DOMINANT term for aminoglycosides
        # K_tissue × f_apl gives binding contribution
        apl_term = K_tissue_AG * f_apl  # ~150 × 0.005 = 0.75 per charge unit

        # Lysosomal accumulation (via megalin-mediated endocytosis)
        lyso_term = 0.0
        if !isnothing(pKa) && pKa > 7.0
            lyso_term = calculate_lysosomal_trapping_kidney(
                pKa=pKa,
                logP=logP,
                f_lysosome=kidney.f_lysosome,
                pH_lysosome=kidney.pH_lysosome
            )
            # Boost lysosomal term for aminoglycosides (receptor-mediated uptake)
            lyso_term *= 3.0
        end

        # Megalin-mediated endocytosis adds significantly
        # This is unique to aminoglycosides - receptor-mediated accumulation
        megalin_term = 5.0 * (n_positive_charges / 4.0)  # Scales with charge

        Kpu = water_term + apl_term + lyso_term + megalin_term
        Kp = Kpu * fup

        # Aminoglycosides: Kp typically 10-20
        return max(Kp, 2.0)
    end

    # ════════════════════════════════════════════════════════════
    # STANDARD DRUG HANDLING
    # ════════════════════════════════════════════════════════════

    # Calculate transporter effects if not provided
    if transporter_effect == 1.0 &&
       (is_oat1_substrate || is_oat3_substrate || is_oct2_substrate || is_mate_substrate || is_pgp_substrate)
        trans_result = estimate_transporter_effect_kidney(
            is_oat1_substrate=is_oat1_substrate,
            is_oat3_substrate=is_oat3_substrate,
            is_oct2_substrate=is_oct2_substrate,
            is_mate_substrate=is_mate_substrate,
            is_pgp_substrate=is_pgp_substrate,
            is_anion=is_acid,
            is_cation=is_base,
            MW=MW,
            logP=logP,
            PSA=PSA
        )
        transporter_effect = trans_result.kp_effect
    end

    # ────────────────────────────────────────────────────────
    # WATER TERM
    # ────────────────────────────────────────────────────────
    water_term = f_ew + ((1 + X) / denom) * f_iw

    # ────────────────────────────────────────────────────────
    # LIPID TERM
    # ────────────────────────────────────────────────────────
    lipid_term = (P * f_nl + (0.3*P + 0.7) * f_npl) / denom

    # ────────────────────────────────────────────────────────
    # CALCULATE Kp BASED ON DRUG TYPE
    # ────────────────────────────────────────────────────────

    if is_base && !isnothing(pKa) && pKa > 6.5
        # CATIONIC DRUGS: APL binding + lysosomal trapping

        # Effective tissue binding (APL-mediated)
        K_tissue = calculate_effective_K_tissue_kidney(logP)
        ion_factor = X / (1 + X)
        tissue_term = K_tissue * ion_factor * (1 + X) / denom

        # Lysosomal trapping
        lyso_term = calculate_lysosomal_trapping_kidney(
            pKa=pKa,
            logP=logP,
            f_lysosome=kidney.f_lysosome,
            pH_lysosome=kidney.pH_lysosome
        )

        Kpu = water_term + lipid_term + tissue_term + lyso_term
        Kp = Kpu * fup * transporter_effect

        # OCT2 substrates: additional accumulation
        if is_oct2_substrate && !is_mate_substrate
            # Without MATE efflux, drug accumulates more
            Kp *= 1.3
        end

    elseif is_acid
        # ANIONIC DRUGS: albumin binding + OAT uptake

        # Tissue albumin binding
        albumin_term = (Ka_PR * AR * (1 + X)) / denom

        Kpu = water_term + lipid_term + albumin_term
        Kp = Kpu * fup * transporter_effect

        # Highly bound acids: minimum based on EW albumin
        if fup < 0.05
            tissue_albumin_binding = AR * (1 - fup) * 0.3
            Kp = max(Kp, (f_ew + tissue_albumin_binding) * transporter_effect)
        end

        # OAT substrates: significant cellular accumulation
        # For highly bound acids (furosemide fup=0.01), OAT creates
        # intracellular drug pool that contributes to Kp
        if is_oat1_substrate || is_oat3_substrate
            # OAT creates intracellular concentration gradient
            # Even highly bound drugs accumulate via active transport
            # Furosemide: fup=0.01, Kp=2.5 → significant OAT contribution
            oat_min_Kp = 1.0 * transporter_effect  # Minimum for OAT substrates
            if fup < 0.05
                # Highly bound: OAT effect is even more important
                oat_min_Kp = 1.5 * transporter_effect
            end
            Kp = max(Kp, oat_min_Kp)
        end

    else
        # NEUTRAL DRUGS: lipoprotein binding + lipid partitioning

        lipoprotein_term = Ka_PR * LR

        Kpu = water_term + lipid_term + lipoprotein_term
        Kp = Kpu * fup * transporter_effect

        # Lipophilic neutrals: membrane partitioning
        if logP > 2.0
            membrane_Kp = 0.2 * P^0.5
            Kp = max(Kp, membrane_Kp * transporter_effect)
        end
    end

    return max(Kp, 0.01)
end

"""
Calculate tubular secretion ratio

Secretion ratio = CLsecretion / CLfiltration

For drugs that are actively secreted:
- PAH (OAT substrate): ratio ~5-6 (used to measure renal plasma flow)
- Metformin (OCT2+MATE): ratio ~3-4
- Penicillins: ratio ~2-3

Secretion allows renal clearance to EXCEED fup × GFR!
"""
function calculate_tubular_secretion_ratio(;
    is_oat1_substrate::Bool = false,
    is_oat3_substrate::Bool = false,
    is_oct2_substrate::Bool = false,
    is_mate_substrate::Bool = false,
    is_cation::Bool = false,
    is_anion::Bool = false,
    MW::Float64 = 400.0,
    logP::Float64 = 2.0
)
    ratio = 0.0

    # OAT-mediated secretion
    if is_oat1_substrate
        ratio += 3.5  # High affinity
    end
    if is_oat3_substrate
        ratio += 2.5
    end

    # OCT2+MATE secretion
    if is_oct2_substrate && is_mate_substrate
        ratio += 3.0  # Coordinated secretion
    elseif is_oct2_substrate
        ratio += 1.5  # Uptake without efficient efflux
    end

    # Predict for unknown transporters
    if ratio == 0.0
        if is_anion && MW < 500 && logP < 2.0
            ratio += 1.0  # Likely OAT substrate
        elseif is_cation && MW < 500 && logP < 3.0
            ratio += 0.8  # Likely OCT2 substrate
        end
    end

    return ratio
end

"""
Calculate fraction reabsorbed in tubules

Reabsorption depends on:
1. Lipophilicity (passive diffusion back across tubular membrane)
2. Ionization state (ionized drugs trapped in urine)
3. Urine pH (variable 4.5-8.0)
4. Urine flow rate (higher flow = less contact time)

KEY: Urine pH manipulation is used therapeutically!
- Alkalinize urine → trap weak acids → ↑ excretion (salicylate overdose)
- Acidify urine → trap weak bases → ↑ excretion (amphetamine overdose)
"""
function calculate_reabsorption_fraction(;
    logP::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    urine_pH::Float64 = 6.0,
    urine_flow_mL_min::Float64 = 1.0
)
    # Baseline reabsorption from lipophilicity
    # High logP = crosses tubular membrane easily
    if logP < -1.0
        base_reabsorption = 0.0  # Too hydrophilic
    elseif logP < 0.0
        base_reabsorption = 0.1 * (logP + 1.0)  # 0 to 0.1
    elseif logP < 1.0
        base_reabsorption = 0.1 + 0.2 * logP  # 0.1 to 0.3
    elseif logP < 2.0
        base_reabsorption = 0.3 + 0.2 * (logP - 1.0)  # 0.3 to 0.5
    elseif logP < 3.0
        base_reabsorption = 0.5 + 0.2 * (logP - 2.0)  # 0.5 to 0.7
    elseif logP < 4.0
        base_reabsorption = 0.7 + 0.15 * (logP - 3.0)  # 0.7 to 0.85
    else
        base_reabsorption = 0.90  # Very lipophilic
    end

    # Ionization correction
    # Ionized drugs are trapped in urine (can't cross membrane)
    if !isnothing(pKa)
        if is_base
            # Bases: ionized at low pH → less reabsorption in acidic urine
            # In collecting duct (pH 5-6), basic drugs ionize and get trapped
            ionized_fraction = 1.0 / (1.0 + 10^(urine_pH - pKa))
            # Only non-ionized fraction can be reabsorbed
            reabsorption = base_reabsorption * (1.0 - ionized_fraction)
        elseif is_acid
            # Acids: ionized at high pH → less reabsorption in alkaline urine
            ionized_fraction = 1.0 / (1.0 + 10^(pKa - urine_pH))
            reabsorption = base_reabsorption * (1.0 - ionized_fraction)
        else
            reabsorption = base_reabsorption
        end
    else
        reabsorption = base_reabsorption
    end

    # Urine flow correction
    # High flow = less contact time = less reabsorption
    flow_factor = 1.0 / sqrt(urine_flow_mL_min)
    reabsorption *= min(flow_factor, 1.0)

    return clamp(reabsorption, 0.0, 0.95)
end

"""
Estimate renal clearance contribution

CLrenal = (fup × GFR) × (1 + secretion_ratio) × (1 - reabsorption)

Where:
- fup × GFR = filtration clearance
- secretion_ratio = CLsecretion / CLfiltration
- reabsorption = fraction reabsorbed

Returns breakdown of clearance components and total CLrenal.
"""
function estimate_renal_clearance_contribution(;
    fup::Float64,
    GFR_mL_min::Float64 = 120.0,
    logP::Float64 = 0.0,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    is_oat1_substrate::Bool = false,
    is_oat3_substrate::Bool = false,
    is_oct2_substrate::Bool = false,
    is_mate_substrate::Bool = false,
    urine_pH::Float64 = 6.0
)
    # Filtration clearance (unbound drug only)
    CL_filtration = fup * GFR_mL_min

    # Secretion
    secretion_ratio = calculate_tubular_secretion_ratio(
        is_oat1_substrate=is_oat1_substrate,
        is_oat3_substrate=is_oat3_substrate,
        is_oct2_substrate=is_oct2_substrate,
        is_mate_substrate=is_mate_substrate,
        is_cation=is_base,
        is_anion=is_acid,
        logP=logP
    )

    CL_secretion = CL_filtration * secretion_ratio

    # Reabsorption
    reabsorption = calculate_reabsorption_fraction(
        logP=logP,
        pKa=pKa,
        is_base=is_base,
        is_acid=is_acid,
        urine_pH=urine_pH
    )

    # Total renal clearance
    # CLrenal = (CLfilt + CLsec) × (1 - reabsorption)
    CL_renal = (CL_filtration + CL_secretion) * (1 - reabsorption)

    # Cannot exceed renal blood flow × extraction
    # Renal plasma flow ~600-700 mL/min
    max_CL = 650.0 * fup  # Upper limit
    CL_renal = min(CL_renal, max_CL)

    return (
        CL_renal=CL_renal,
        CL_filtration=CL_filtration,
        CL_secretion=CL_secretion,
        secretion_ratio=secretion_ratio,
        reabsorption=reabsorption,
        fe_predicted=CL_renal > 0.1 ? min(CL_renal / (CL_renal + 100), 1.0) : 0.0
    )
end

"""
Estimate nephrotoxicity risk

Drugs that accumulate in kidney can cause toxicity:

1. PROXIMAL TUBULE TOXICITY
   - Aminoglycosides: megalin-mediated uptake → lysosomal damage
   - Cisplatin: OCT2 uptake → mitochondrial damage, apoptosis
   - Tenofovir: OAT1 uptake → mitochondrial toxicity
   - Vancomycin: oxidative stress

2. GLOMERULAR TOXICITY
   - NSAIDs: prostaglandin inhibition → afferent vasoconstriction
   - ACE inhibitors: efferent vasodilation (beneficial unless stenosis!)
   - Contrast agents: vasoconstriction + direct toxicity

3. CRYSTALLURIA
   - High-dose sulfadiazine, acyclovir, indinavir
   - Precipitation in tubular lumen

Returns risk score (0-10) and mechanisms.
"""
function estimate_nephrotoxicity_risk(;
    Kp_kidney::Float64,
    is_aminoglycoside::Bool = false,
    is_oct2_substrate::Bool = false,
    is_oat1_substrate::Bool = false,
    is_base::Bool = false,
    pKa::Union{Float64, Nothing} = nothing,
    logP::Float64 = 2.0,
    dose_mg_kg::Float64 = 1.0
)
    risk_score = 0.0
    mechanisms = String[]

    # High Kp_kidney indicates accumulation
    if Kp_kidney > 5.0
        risk_score += min((Kp_kidney - 5.0) / 5.0 * 2.0, 4.0)
        push!(mechanisms, "High renal accumulation (Kp=$(round(Kp_kidney, digits=1)))")
    end

    # Aminoglycosides: dose-dependent nephrotoxicity
    if is_aminoglycoside
        risk_score += 3.0
        push!(mechanisms, "Aminoglycoside: megalin-mediated uptake → lysosomal toxicity")
        if dose_mg_kg > 5.0
            risk_score += 2.0
            push!(mechanisms, "High dose increases risk")
        end
    end

    # OCT2 substrates: cellular accumulation
    if is_oct2_substrate
        risk_score += 1.5
        push!(mechanisms, "OCT2 substrate: proximal tubule accumulation")
    end

    # OAT1 substrates: antiviral nephrotoxicity pattern
    if is_oat1_substrate && logP < 0
        risk_score += 1.0
        push!(mechanisms, "OAT1 substrate: potential mitochondrial toxicity")
    end

    # Strong bases with high Kp: lysosomal overload
    if is_base && !isnothing(pKa) && pKa > 8.0 && Kp_kidney > 3.0
        risk_score += 1.5
        push!(mechanisms, "Strong base: lysosomal accumulation risk")
    end

    risk_score = clamp(risk_score, 0.0, 10.0)

    risk_level = if risk_score < 2.0
        "Low"
    elseif risk_score < 5.0
        "Moderate"
    elseif risk_score < 7.0
        "High"
    else
        "Very High"
    end

    return (
        risk_score=risk_score,
        risk_level=risk_level,
        mechanisms=mechanisms
    )
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
    is_aminoglycoside::Bool = false,
    n_positive_charges::Int = 1,
    kidney_volume::Float64 = 0.31,
    transporter_effect::Float64 = 1.0,
    is_oat1_substrate::Bool = false,
    is_oat3_substrate::Bool = false,
    is_oct2_substrate::Bool = false,
    is_mate_substrate::Bool = false,
    is_pgp_substrate::Bool = false,
    MW::Float64 = 400.0,
    PSA::Float64 = 80.0
)
    Kp = calculate_kp_kidney(
        logP=logP, logD=logD, fup=fup,
        pKa=pKa, is_base=is_base, is_acid=is_acid,
        is_aminoglycoside=is_aminoglycoside,
        n_positive_charges=n_positive_charges,
        transporter_effect=transporter_effect,
        is_oat1_substrate=is_oat1_substrate,
        is_oat3_substrate=is_oat3_substrate,
        is_oct2_substrate=is_oct2_substrate,
        is_mate_substrate=is_mate_substrate,
        is_pgp_substrate=is_pgp_substrate,
        MW=MW, PSA=PSA
    )

    contribution = Kp * kidney_volume

    return (Kp=Kp, contribution_L=contribution, volume=kidney_volume)
end

# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE DRUGS WITH RENAL HANDLING DATA
# ═══════════════════════════════════════════════════════════════════════════

const RENAL_DRUG_EXAMPLES = Dict(
    # AMINOGLYCOSIDES - nephrotoxic accumulation
    "gentamicin" => (
        logP=-3.1, pKa=8.2, fup=0.9, is_base=true,
        is_aminoglycoside=true, n_charges=5,
        Kp_observed=15.0, CLrenal_observed=80.0,
        note="Aminoglycoside: megalin uptake → lysosomal damage"
    ),
    "tobramycin" => (
        logP=-4.3, pKa=7.5, fup=0.92, is_base=true,
        is_aminoglycoside=true, n_charges=5,
        Kp_observed=12.0, CLrenal_observed=75.0,
        note="Similar to gentamicin, slightly less nephrotoxic"
    ),

    # OCT2 + MATE SUBSTRATES
    "metformin" => (
        logP=-1.5, pKa=11.5, fup=0.99, is_base=true,
        is_oct2=true, is_mate=true,
        Kp_observed=5.0, CLrenal_observed=500.0,  # 4x GFR!
        note="OCT2+MATE: CLrenal >> GFR, renal impairment risk"
    ),
    "cisplatin" => (
        logP=-2.2, pKa=nothing, fup=0.05,  # Highly protein bound
        is_oct2=true, is_mate=false,  # Uptake without efflux = toxicity!
        Kp_observed=8.0, CLrenal_observed=20.0,
        note="OCT2 uptake without MATE efflux → nephrotoxicity"
    ),

    # OAT SUBSTRATES
    "tenofovir" => (
        logP=-1.6, pKa=3.8, fup=0.93, is_acid=true,
        is_oat1=true,
        Kp_observed=4.0, CLrenal_observed=300.0,
        note="OAT1 substrate: proximal tubule toxicity (Fanconi syndrome)"
    ),
    "furosemide" => (
        logP=2.0, pKa=3.9, fup=0.01, is_acid=true,
        is_oat3=true,
        Kp_observed=2.5, CLrenal_observed=120.0,  # ~GFR despite high binding
        note="OAT3: secretion compensates for high protein binding"
    ),
    "penicillin_G" => (
        logP=1.8, pKa=2.8, fup=0.45, is_acid=true,
        is_oat1=true, is_oat3=true,
        Kp_observed=3.0, CLrenal_observed=400.0,
        note="OAT1/3: classic substrate, probenecid inhibits secretion"
    ),

    # LIPOPHILIC BASES
    "propranolol" => (
        logP=3.5, pKa=9.5, fup=0.10, is_base=true,
        Kp_observed=3.5, CLrenal_observed=1.0,
        note="Lipophilic base: extensive reabsorption, hepatic clearance"
    ),
    "imipramine" => (
        logP=4.8, pKa=9.4, fup=0.10, is_base=true,
        Kp_observed=4.0, CLrenal_observed=2.0,
        note="TCA: lysosomal trapping + APL binding, renal << hepatic"
    ),

    # P-GP SUBSTRATES
    "digoxin" => (
        logP=1.3, pKa=nothing, fup=0.75, is_neutral=true,
        is_pgp=true,
        Kp_observed=0.5, CLrenal_observed=80.0,  # ~fup × GFR
        note="P-gp substrate: mostly filtration, reduced in renal failure"
    ),

    # HIGHLY BOUND ACIDS
    "warfarin" => (
        logP=2.6, pKa=5.1, fup=0.01, is_acid=true,
        Kp_observed=0.3, CLrenal_observed=0.1,
        note="Highly bound: minimal filtration, hepatic metabolism"
    ),
)

# ══════════════════════════════════════════════════════════════════════════════
# SOTA DISCOVERIES 2020-2024 - ADVANCED KIDNEY MODELING
# ══════════════════════════════════════════════════════════════════════════════
#
# Based on:
# - Lake et al., Nature 2023: Single-cell RNA-seq of human kidney
# - Muto et al., Science 2023: PT segment heterogeneity
# - Hsueh et al., J Clin Pharmacol 2023: CKD transporter scaling
# - Yonezawa et al., CPT 2023: MATE polymorphisms
# - Nolin et al., JASN 2023: Uremic protein binding
# - FDA Guidance 2024: OAT1/3 DDI assessment
# ══════════════════════════════════════════════════════════════════════════════

"""
CKD Stage classification (KDIGO 2021)
"""
@enum CKDStage begin
    CKD_G1      # GFR ≥ 90, normal or high
    CKD_G2      # GFR 60-89, mildly decreased
    CKD_G3a     # GFR 45-59, mild-moderate decrease
    CKD_G3b     # GFR 30-44, moderate-severe decrease
    CKD_G4      # GFR 15-29, severely decreased
    CKD_G5      # GFR < 15, kidney failure
    CKD_G5D     # Dialysis
end

"""
Get CKD stage from measured GFR
"""
function get_ckd_stage(GFR::Float64)::CKDStage
    if GFR >= 90
        return CKD_G1
    elseif GFR >= 60
        return CKD_G2
    elseif GFR >= 45
        return CKD_G3a
    elseif GFR >= 30
        return CKD_G3b
    elseif GFR >= 15
        return CKD_G4
    else
        return CKD_G5
    end
end

"""
Transporter polymorphism genotypes

Key polymorphisms affecting renal drug handling (SOTA 2023-2024):

OCT2 (SLC22A2):
- c.808G>T (rs316019, p.A270S): ↓ function, common in Asians
- c.596C>T: Associated with ↑ cisplatin nephrotoxicity

MATE1 (SLC47A1):
- rs2289669 (c.922-158G>A): GG normal, AA ↓ function
- Affects metformin handling significantly

MATE2-K (SLC47A2):
- rs12943590: ↓ metformin renal clearance

OAT1/OAT3:
- Generally fewer clinically significant polymorphisms
- rs4149170 (OAT1): Rare, ↓ activity
"""
struct TransporterPolymorphisms
    # OCT2 variants
    oct2_A270S::Symbol   # :wt (GG), :het (GT), :hom (TT)
    oct2_596::Symbol     # :wt (CC), :het (CT), :hom (TT)

    # MATE variants
    mate1_rs2289669::Symbol  # :wt (GG), :het (GA), :hom (AA)
    mate2k_rs12943590::Symbol # :wt, :het, :hom

    # OAT variants (rare but included for completeness)
    oat1_rs4149170::Symbol   # :wt, :het, :hom
    oat3_variant::Symbol     # placeholder
end

# Default wildtype
const WILDTYPE_TRANSPORTERS = TransporterPolymorphisms(
    :wt, :wt,  # OCT2
    :wt, :wt,  # MATE
    :wt, :wt   # OAT
)

"""
Patient-specific kidney status for individualized modeling
"""
struct PatientKidneyStatus
    age_years::Float64
    weight_kg::Float64
    sex::Symbol  # :male, :female
    GFR_measured::Float64  # mL/min/1.73m²
    ckd_stage::CKDStage
    is_diabetic::Bool
    serum_creatinine::Float64  # mg/dL
    serum_albumin::Float64  # g/dL (normal 3.5-5.0)
    polymorphisms::TransporterPolymorphisms
end

"""
Scale transporter activity based on CKD stage

SOTA 2023-2024 (Hsueh et al., J Clin Pharmacol):
In CKD, transporter expression changes NON-LINEARLY with GFR decline.
This is due to:
1. Loss of nephron mass (direct)
2. Uremic toxin accumulation → transporter downregulation
3. Compensatory upregulation in remaining nephrons (partial)

CRITICAL INSIGHT: Don't just scale CLfiltration by GFR!
You must also scale secretion by transporter activity.

Reference values (% of normal activity):
| CKD Stage | GFR (mL/min) | OAT1/3  | OCT2   | MATE   |
|-----------|--------------|---------|--------|--------|
| G1        | ≥90          | 100%    | 100%   | 100%   |
| G2        | 60-89        | 90%     | 95%    | 95%    |
| G3a       | 45-59        | 70%     | 80%    | 85%    |
| G3b       | 30-44        | 50%     | 60%    | 70%    |
| G4        | 15-29        | 30%     | 40%    | 50%    |
| G5        | <15          | 10%     | 20%    | 30%    |
"""
function scale_transporter_activity_ckd(ckd_stage::CKDStage)
    scaling = Dict{String, Float64}()

    if ckd_stage == CKD_G1
        scaling["OAT1"] = 1.00
        scaling["OAT3"] = 1.00
        scaling["OCT2"] = 1.00
        scaling["MATE1"] = 1.00
        scaling["MATE2K"] = 1.00
        scaling["P_gp"] = 1.00
    elseif ckd_stage == CKD_G2
        scaling["OAT1"] = 0.90
        scaling["OAT3"] = 0.90
        scaling["OCT2"] = 0.95
        scaling["MATE1"] = 0.95
        scaling["MATE2K"] = 0.95
        scaling["P_gp"] = 0.95
    elseif ckd_stage == CKD_G3a
        scaling["OAT1"] = 0.70
        scaling["OAT3"] = 0.70
        scaling["OCT2"] = 0.80
        scaling["MATE1"] = 0.85
        scaling["MATE2K"] = 0.85
        scaling["P_gp"] = 0.85
    elseif ckd_stage == CKD_G3b
        scaling["OAT1"] = 0.50
        scaling["OAT3"] = 0.50
        scaling["OCT2"] = 0.60
        scaling["MATE1"] = 0.70
        scaling["MATE2K"] = 0.70
        scaling["P_gp"] = 0.75
    elseif ckd_stage == CKD_G4
        scaling["OAT1"] = 0.30
        scaling["OAT3"] = 0.30
        scaling["OCT2"] = 0.40
        scaling["MATE1"] = 0.50
        scaling["MATE2K"] = 0.50
        scaling["P_gp"] = 0.55
    else  # CKD_G5 or G5D
        scaling["OAT1"] = 0.10
        scaling["OAT3"] = 0.10
        scaling["OCT2"] = 0.20
        scaling["MATE1"] = 0.30
        scaling["MATE2K"] = 0.30
        scaling["P_gp"] = 0.35
    end

    return scaling
end

"""
Adjust transporter activity for genetic polymorphisms

Reference: Yonezawa et al., CPT 2023; Filipski et al., DMD 2024

OCT2 A270S (rs316019):
- Wildtype (GG): 100% activity
- Heterozygous (GT): 80% activity
- Homozygous (TT): 50% activity
- Clinical impact: ↓ metformin CLrenal, ↑ cisplatin exposure

MATE1 rs2289669:
- Wildtype (GG): 100% activity
- Heterozygous (GA): 85% activity
- Homozygous (AA): 60% activity
- Clinical impact: ↑ intracellular metformin accumulation

Combined OCT2+MATE polymorphisms can have synergistic effects!
"""
function adjust_for_polymorphisms(
    base_activity::Dict{String, Float64},
    polymorphisms::TransporterPolymorphisms
)
    adjusted = copy(base_activity)

    # OCT2 A270S
    oct2_factor = if polymorphisms.oct2_A270S == :wt
        1.0
    elseif polymorphisms.oct2_A270S == :het
        0.80
    else  # :hom
        0.50
    end
    adjusted["OCT2"] *= oct2_factor

    # OCT2 596 variant (cisplatin specific, but affects general activity)
    if polymorphisms.oct2_596 == :het
        adjusted["OCT2"] *= 0.90
    elseif polymorphisms.oct2_596 == :hom
        adjusted["OCT2"] *= 0.75
    end

    # MATE1 rs2289669
    mate1_factor = if polymorphisms.mate1_rs2289669 == :wt
        1.0
    elseif polymorphisms.mate1_rs2289669 == :het
        0.85
    else  # :hom
        0.60
    end
    adjusted["MATE1"] *= mate1_factor

    # MATE2-K rs12943590
    mate2k_factor = if polymorphisms.mate2k_rs12943590 == :wt
        1.0
    elseif polymorphisms.mate2k_rs12943590 == :het
        0.85
    else
        0.65
    end
    adjusted["MATE2K"] *= mate2k_factor

    # OAT1 rs4149170 (rare)
    if polymorphisms.oat1_rs4149170 == :het
        adjusted["OAT1"] *= 0.85
    elseif polymorphisms.oat1_rs4149170 == :hom
        adjusted["OAT1"] *= 0.50
    end

    return adjusted
end

"""
Calculate uremic toxin effect on protein binding

SOTA 2023 (Nolin et al., JASN):
In CKD, uremic toxins accumulate and DISPLACE drugs from albumin:
- Indoxyl sulfate (IS)
- p-Cresyl sulfate (pCS)
- Hippuric acid

Effect:
- ↑ free fraction (fu) in uremia
- Partially compensates for ↓ GFR (↑ filtration clearance)
- But also ↑ tissue distribution (↑ Vd)
- Net effect: Variable, drug-specific

Scaling factors (fu_uremic / fu_normal):
| CKD Stage | Highly Bound (fu<0.1) | Mod Bound (0.1-0.5) | Low Bound (>0.5) |
|-----------|----------------------|---------------------|------------------|
| G1-G2     | 1.0                  | 1.0                 | 1.0              |
| G3        | 1.3-1.5              | 1.1-1.2             | 1.0              |
| G4        | 1.5-2.0              | 1.2-1.4             | 1.0-1.1          |
| G5        | 2.0-3.0              | 1.3-1.6             | 1.0-1.2          |

Drugs particularly affected:
- Warfarin (fu 0.01 → 0.02-0.03 in ESRD)
- Phenytoin (fu 0.1 → 0.2-0.25 in ESRD)
- Diazepam (fu 0.01 → 0.02-0.04 in ESRD)
"""
function calculate_uremic_fup_adjustment(;
    fup_normal::Float64,
    ckd_stage::CKDStage,
    serum_albumin::Float64 = 4.0  # g/dL
)
    # Albumin effect (hypoalbuminemia in CKD)
    albumin_factor = 4.0 / max(serum_albumin, 2.0)  # Normalize to 4 g/dL

    # Uremic toxin displacement effect
    displacement_factor = if ckd_stage in [CKD_G1, CKD_G2]
        1.0
    elseif ckd_stage == CKD_G3a
        fup_normal < 0.1 ? 1.3 : (fup_normal < 0.5 ? 1.1 : 1.0)
    elseif ckd_stage == CKD_G3b
        fup_normal < 0.1 ? 1.5 : (fup_normal < 0.5 ? 1.2 : 1.0)
    elseif ckd_stage == CKD_G4
        fup_normal < 0.1 ? 1.8 : (fup_normal < 0.5 ? 1.3 : 1.05)
    else  # G5
        fup_normal < 0.1 ? 2.5 : (fup_normal < 0.5 ? 1.5 : 1.1)
    end

    # Calculate adjusted fup
    fup_adjusted = fup_normal * displacement_factor * sqrt(albumin_factor)

    # Cannot exceed 1.0
    return min(fup_adjusted, 1.0)
end

"""
Age-adjusted kidney parameters

Kidney function declines with age:
- ~1% GFR loss per year after age 40
- ~10% nephron loss per decade
- Transporter expression also decreases
- But CLcr may not reflect true GFR in elderly (↓ muscle mass)

Reference values:
| Age Decade | GFR (% of 25yo) | Transporter Activity | Kidney Volume |
|------------|-----------------|---------------------|---------------|
| 20-30      | 100%            | 100%                | 100%          |
| 30-40      | 95%             | 98%                 | 99%           |
| 40-50      | 90%             | 95%                 | 98%           |
| 50-60      | 80%             | 85%                 | 95%           |
| 60-70      | 70%             | 75%                 | 90%           |
| 70-80      | 60%             | 60%                 | 85%           |
| >80        | 50%             | 50%                 | 80%           |
"""
function age_adjusted_kidney_parameters(age_years::Float64)
    # GFR decline: ~1%/year after 40
    if age_years <= 30
        gfr_factor = 1.0
    elseif age_years <= 40
        gfr_factor = 1.0 - 0.005 * (age_years - 30)  # 0.5%/year
    else
        gfr_factor = 0.95 - 0.01 * (age_years - 40)  # 1%/year after 40
    end
    gfr_factor = max(gfr_factor, 0.40)  # Floor at 40%

    # Transporter activity: slightly slower decline than GFR
    if age_years <= 40
        transporter_factor = 1.0
    elseif age_years <= 60
        transporter_factor = 1.0 - 0.0075 * (age_years - 40)  # 0.75%/year
    else
        transporter_factor = 0.85 - 0.0125 * (age_years - 60)  # 1.25%/year
    end
    transporter_factor = max(transporter_factor, 0.45)

    # Kidney volume: modest decline
    if age_years <= 40
        volume_factor = 1.0
    else
        volume_factor = 1.0 - 0.003 * (age_years - 40)  # 0.3%/year
    end
    volume_factor = max(volume_factor, 0.75)

    return (
        gfr_factor = gfr_factor,
        transporter_factor = transporter_factor,
        volume_factor = volume_factor,
        estimated_gfr = 120.0 * gfr_factor  # mL/min
    )
end

"""
Calculate Kp_kidney for CKD patients

In CKD, Kp may change due to:
1. ↓ transporter-mediated uptake (↓ transporter activity)
2. ↑ fup (uremic toxin displacement)
3. Tissue composition changes (fibrosis)
4. ↓ kidney volume (nephron loss)

Net effect is often ↓ total drug in kidney despite ↑ concentration.
"""
function calculate_kp_kidney_ckd(;
    logP::Float64,
    logD::Float64 = logP,
    fup_normal::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    patient::PatientKidneyStatus,
    is_oat1_substrate::Bool = false,
    is_oat3_substrate::Bool = false,
    is_oct2_substrate::Bool = false,
    is_mate_substrate::Bool = false,
    MW::Float64 = 400.0
)
    # Get CKD-adjusted transporter activities
    base_scaling = scale_transporter_activity_ckd(patient.ckd_stage)
    transporter_scaling = adjust_for_polymorphisms(base_scaling, patient.polymorphisms)

    # Adjust fup for uremia
    fup_adjusted = calculate_uremic_fup_adjustment(
        fup_normal = fup_normal,
        ckd_stage = patient.ckd_stage,
        serum_albumin = patient.serum_albumin
    )

    # Calculate transporter effect with CKD scaling
    trans_kp_effect = 1.0
    if is_oat1_substrate
        trans_kp_effect *= (1.0 + 3.0 * transporter_scaling["OAT1"])  # Reduced in CKD
    end
    if is_oat3_substrate
        trans_kp_effect *= (1.0 + 2.0 * transporter_scaling["OAT3"])
    end
    if is_oct2_substrate
        if is_mate_substrate
            # Both working: flow-through, minimal Kp effect
            trans_kp_effect *= (1.0 + 0.2 * transporter_scaling["OCT2"])
        else
            # OCT2 without MATE: accumulation (reduced in CKD = less toxicity)
            trans_kp_effect *= (1.0 + 1.5 * transporter_scaling["OCT2"])
        end
    end

    # Calculate base Kp using normal function
    Kp_base = calculate_kp_kidney(
        logP = logP,
        logD = logD,
        fup = fup_adjusted,
        pKa = pKa,
        is_base = is_base,
        is_acid = is_acid,
        MW = MW,
        transporter_effect = trans_kp_effect,
        is_oat1_substrate = is_oat1_substrate,
        is_oat3_substrate = is_oat3_substrate,
        is_oct2_substrate = is_oct2_substrate,
        is_mate_substrate = is_mate_substrate
    )

    # Additional CKD-specific adjustment (tissue fibrosis, altered composition)
    ckd_tissue_factor = if patient.ckd_stage in [CKD_G1, CKD_G2]
        1.0
    elseif patient.ckd_stage in [CKD_G3a, CKD_G3b]
        0.90  # Mild fibrosis
    elseif patient.ckd_stage == CKD_G4
        0.80  # Moderate fibrosis
    else
        0.70  # Severe fibrosis
    end

    return Kp_base * ckd_tissue_factor
end

"""
Estimate renal clearance in CKD patients

COMPREHENSIVE MODEL incorporating:
1. ↓ GFR (direct)
2. ↓ Transporter activity (indirect, CKD-stage dependent)
3. ↑ fup (uremic toxin displacement, partial compensation)
4. Genetic polymorphisms
5. Age effects

CLrenal_CKD = (fu_adj × GFR_adj) × (1 + Secretion_adj) × (1 - Reabsorption)

Example: Metformin
- Normal: CLrenal ≈ 500 mL/min (OCT2+MATE)
- CKD G3: CLrenal ≈ 200-250 mL/min (↓ GFR + ↓ OCT2/MATE)
- CKD G4: CLrenal ≈ 80-100 mL/min → CONTRAINDICATED at eGFR<30!
"""
function estimate_renal_clearance_ckd(;
    fup_normal::Float64,
    logP::Float64 = 0.0,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    is_oat1_substrate::Bool = false,
    is_oat3_substrate::Bool = false,
    is_oct2_substrate::Bool = false,
    is_mate_substrate::Bool = false,
    patient::PatientKidneyStatus,
    urine_pH::Float64 = 6.0
)
    # Get CKD-adjusted transporter activities
    base_scaling = scale_transporter_activity_ckd(patient.ckd_stage)
    transporter_scaling = adjust_for_polymorphisms(base_scaling, patient.polymorphisms)

    # Age adjustment
    age_adj = age_adjusted_kidney_parameters(patient.age_years)

    # Effective GFR (measured, age-adjusted if not measured)
    GFR_effective = patient.GFR_measured > 0 ? patient.GFR_measured : age_adj.estimated_gfr

    # Adjust fup for uremia
    fup_adjusted = calculate_uremic_fup_adjustment(
        fup_normal = fup_normal,
        ckd_stage = patient.ckd_stage,
        serum_albumin = patient.serum_albumin
    )

    # FILTRATION CLEARANCE
    CL_filtration = fup_adjusted * GFR_effective

    # SECRETION CLEARANCE (scaled by transporter activity)
    secretion_ratio = 0.0

    if is_oat1_substrate
        secretion_ratio += 3.5 * transporter_scaling["OAT1"]
    end
    if is_oat3_substrate
        secretion_ratio += 2.5 * transporter_scaling["OAT3"]
    end
    if is_oct2_substrate && is_mate_substrate
        # Both transporters needed for efficient secretion
        # Use geometric mean of both activities
        combined_activity = sqrt(transporter_scaling["OCT2"] *
                                  (transporter_scaling["MATE1"] + transporter_scaling["MATE2K"]) / 2)
        secretion_ratio += 3.0 * combined_activity
    elseif is_oct2_substrate
        # OCT2 only: reduced secretion
        secretion_ratio += 1.5 * transporter_scaling["OCT2"]
    end

    CL_secretion = CL_filtration * secretion_ratio

    # REABSORPTION (not significantly affected by CKD in most cases)
    reabsorption = calculate_reabsorption_fraction(
        logP = logP,
        pKa = pKa,
        is_base = is_base,
        is_acid = is_acid,
        urine_pH = urine_pH
    )

    # TOTAL RENAL CLEARANCE
    CL_renal = (CL_filtration + CL_secretion) * (1 - reabsorption)

    # Maximum = renal blood flow × extraction (reduced in CKD)
    rbf_ckd_factor = if patient.ckd_stage in [CKD_G1, CKD_G2]
        1.0
    elseif patient.ckd_stage in [CKD_G3a, CKD_G3b]
        0.85
    elseif patient.ckd_stage == CKD_G4
        0.70
    else
        0.50
    end
    max_CL = 650.0 * fup_adjusted * rbf_ckd_factor
    CL_renal = min(CL_renal, max_CL)

    # Calculate dose adjustment factor
    normal_cl = estimate_renal_clearance_contribution(
        fup = fup_normal,
        GFR_mL_min = 120.0,
        logP = logP,
        pKa = pKa,
        is_base = is_base,
        is_acid = is_acid,
        is_oat1_substrate = is_oat1_substrate,
        is_oat3_substrate = is_oat3_substrate,
        is_oct2_substrate = is_oct2_substrate,
        is_mate_substrate = is_mate_substrate
    )

    dose_adjustment = CL_renal / max(normal_cl.CL_renal, 1.0)

    return (
        CL_renal = CL_renal,
        CL_filtration = CL_filtration,
        CL_secretion = CL_secretion,
        secretion_ratio = secretion_ratio,
        reabsorption = reabsorption,
        fup_adjusted = fup_adjusted,
        dose_adjustment_factor = clamp(dose_adjustment, 0.1, 1.0),
        transporter_activities = transporter_scaling
    )
end

"""
Predict cisplatin nephrotoxicity risk based on OCT2/MATE genetics

SOTA 2023-2024:
Cisplatin is taken up by OCT2 but poorly effluxed by MATE → accumulation → toxicity

Risk factors:
1. OCT2 high activity (wildtype) → MORE uptake → MORE toxicity
2. MATE low activity (variant) → LESS efflux → MORE accumulation
3. Paradox: OCT2 ↓ function variant is PROTECTIVE!

This explains why OCT2 A270S carriers have LESS cisplatin nephrotoxicity.
"""
function predict_cisplatin_nephrotoxicity_risk(
    polymorphisms::TransporterPolymorphisms;
    dose_mg_m2::Float64 = 75.0,
    GFR::Float64 = 100.0
)
    # Base risk from dose and kidney function
    base_risk = (dose_mg_m2 / 75.0) * (100.0 / max(GFR, 30.0))

    # OCT2 effect: HIGHER activity = MORE toxicity (paradox!)
    oct2_risk = if polymorphisms.oct2_A270S == :wt
        1.0  # Wildtype = full uptake = higher risk
    elseif polymorphisms.oct2_A270S == :het
        0.70  # Reduced uptake = lower risk
    else  # :hom
        0.40  # Much less uptake = protective
    end

    # OCT2 596 variant also affects
    if polymorphisms.oct2_596 == :het
        oct2_risk *= 1.2  # Increases toxicity risk
    elseif polymorphisms.oct2_596 == :hom
        oct2_risk *= 1.5
    end

    # MATE effect: LOWER activity = MORE accumulation = MORE toxicity
    mate_risk = if polymorphisms.mate1_rs2289669 == :wt
        1.0
    elseif polymorphisms.mate1_rs2289669 == :het
        1.2
    else  # :hom
        1.5
    end

    total_risk = base_risk * oct2_risk * mate_risk

    risk_level = if total_risk < 0.7
        "Low"
    elseif total_risk < 1.0
        "Standard"
    elseif total_risk < 1.5
        "Elevated"
    else
        "High - Consider dose reduction or alternative"
    end

    return (
        risk_score = total_risk,
        risk_level = risk_level,
        oct2_contribution = oct2_risk,
        mate_contribution = mate_risk,
        recommendation = if total_risk > 1.5
            "Consider oxaliplatin (better MATE substrate) or dose reduction"
        elseif total_risk > 1.0
            "Standard monitoring, consider hydration protocol"
        else
            "Standard protocol"
        end
    )
end

# ══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE PROTEOMICS DATA FOR IVIVE
# ══════════════════════════════════════════════════════════════════════════════
# Reference: Prasad et al., DMD 2016; Wang et al., DMD 2023

"""
Transporter abundances for IVIVE scaling (pmol/mg protein)

From quantitative proteomics in human kidney cortex samples.
These values allow in vitro-in vivo extrapolation (IVIVE).
"""
const KIDNEY_TRANSPORTER_PROTEOMICS = Dict(
    # Uptake transporters (basolateral)
    "OAT1" => (abundance = 4.0, cv = 0.38, unit = "pmol/mg"),
    "OAT2" => (abundance = 0.8, cv = 0.45, unit = "pmol/mg"),
    "OAT3" => (abundance = 2.5, cv = 0.40, unit = "pmol/mg"),
    "OCT2" => (abundance = 6.0, cv = 0.33, unit = "pmol/mg"),
    "OATP4C1" => (abundance = 0.3, cv = 0.50, unit = "pmol/mg"),

    # Efflux transporters (apical)
    "MATE1" => (abundance = 2.5, cv = 0.35, unit = "pmol/mg"),
    "MATE2K" => (abundance = 1.5, cv = 0.40, unit = "pmol/mg"),
    "P_gp" => (abundance = 0.8, cv = 0.45, unit = "pmol/mg"),
    "MRP2" => (abundance = 1.5, cv = 0.42, unit = "pmol/mg"),
    "MRP4" => (abundance = 2.0, cv = 0.38, unit = "pmol/mg"),
    "BCRP" => (abundance = 1.2, cv = 0.40, unit = "pmol/mg"),

    # Reabsorption
    "OAT4" => (abundance = 1.8, cv = 0.35, unit = "pmol/mg"),
    "URAT1" => (abundance = 2.2, cv = 0.40, unit = "pmol/mg"),

    # Peptide transporters
    "PEPT1" => (abundance = 0.5, cv = 0.50, unit = "pmol/mg"),
    "PEPT2" => (abundance = 3.0, cv = 0.35, unit = "pmol/mg"),
)

"""
Scaling factors for IVIVE

To extrapolate from in vitro transporter data to in vivo CLrenal:

CLsecretion = CLint × fu × Scaling_Factor × Qkidney / (Qkidney + CLint × Scaling_Factor)

Where:
- CLint = in vitro intrinsic clearance (µL/min/pmol transporter)
- Scaling_Factor = PTCPGK × Abundance × (1 - fraction bound to cells)
- PTCPGK = proximal tubule cells per gram kidney (~60 × 10⁶)
- Qkidney = renal plasma flow (~600 mL/min)
"""
const IVIVE_SCALING = (
    PTCPGK = 60e6,           # Proximal tubule cells per gram kidney
    kidney_weight = 310.0,    # g (both kidneys)
    PT_fraction = 0.60,       # Fraction of kidney that is PT
    microsomal_protein = 40.0, # mg protein per g kidney
    renal_plasma_flow = 600.0  # mL/min
)

# ============================================================================
# MICHAELIS-MENTEN TRANSPORTER KINETICS - Q1+ SOTA 2024
# ============================================================================
#
# Scientific Foundation:
# - Zamek-Gliszczynski et al. (2013) J Pharmacol Exp Ther - Renal transporter DDI
# - Müller et al. (2017) Clin Pharmacokinet - OAT1/3 IVIVE
# - Motohashi & Inui (2013) AAPS J - OCT2/MATE interplay
# - Giacomini et al. (2010) Nat Rev Drug Discov - ITC white paper
# - Prasad & Bhatt (2023) Drug Metab Dispos - Transporter proteomics IVIVE
#
# Replaces linear transporter factors with saturable kinetics:
#   v = (Vmax × [S]) / (Km + [S])
#
# This captures:
# 1. Concentration-dependent clearance (non-linear PK)
# 2. Transporter DDI via competitive/non-competitive inhibition
# 3. Genetic polymorphism effects on Vmax (expression) and Km (affinity)
# ============================================================================

"""
    RenalTransporterKinetics

Michaelis-Menten kinetic parameters for renal transporters.

# Fields
- `name::String`: Transporter identifier (e.g., "OAT1", "OCT2")
- `gene::String`: Gene symbol (e.g., "SLC22A6", "SLC22A2")
- `location::Symbol`: Membrane location (:basolateral or :apical)
- `direction::Symbol`: Transport direction (:uptake, :efflux, or :reabsorption)
- `Km::Float64`: Michaelis constant (µM) - substrate affinity
- `Vmax::Float64`: Maximum velocity (pmol/min/mg protein)
- `Km_cv::Float64`: Coefficient of variation for Km (population variability)
- `Vmax_cv::Float64`: Coefficient of variation for Vmax
- `substrates::Vector{String}`: Representative substrates
- `inhibitors::Vector{String}`: Known inhibitors with IC50
- `polymorphisms::Dict{String,NamedTuple}`: Genetic variants affecting kinetics

# Reference Values
Kinetic parameters derived from:
- Quantitative proteomics (Prasad et al., 2016)
- In vitro transport assays in transfected cells
- Clinical DDI studies with probe substrates
"""
struct RenalTransporterKinetics
    name::String
    gene::String
    location::Symbol
    direction::Symbol
    Km::Float64              # µM
    Vmax::Float64            # pmol/min/mg protein
    Km_cv::Float64           # Coefficient of variation
    Vmax_cv::Float64
    substrates::Vector{String}
    inhibitors::Dict{String,Float64}  # Inhibitor => IC50 (µM)
    polymorphisms::Dict{String,NamedTuple}
end

# ============================================================================
# KINETIC CONSTANTS FOR MAJOR RENAL TRANSPORTERS
# ============================================================================

"""
OAT1 (SLC22A6) - Organic Anion Transporter 1

Major basolateral uptake transporter for:
- β-lactam antibiotics (penicillins, cephalosporins)
- NSAIDs (indomethacin, ibuprofen)
- Antivirals (adefovir, cidofovir, tenofovir)
- Diuretics (furosemide, bumetanide)

Km values: highly substrate-dependent (0.1-100 µM range)
Representative Km for PAH (prototypical substrate): 14-22 µM
"""
const OAT1_RENAL_KINETICS = RenalTransporterKinetics(
    "OAT1",
    "SLC22A6",
    :basolateral,
    :uptake,
    20.0,           # Km (µM) - PAH reference
    450.0,          # Vmax (pmol/min/mg protein)
    0.35,           # Km CV
    0.40,           # Vmax CV
    ["PAH", "tenofovir", "adefovir", "cidofovir", "furosemide", "methotrexate"],
    Dict(
        "probenecid" => 1.5,      # Classic OAT inhibitor
        "NSAIDs" => 5.0,          # General class
        "penicillin_G" => 45.0,
        "cimetidine" => 120.0
    ),
    Dict(
        "rs11568626" => (effect = :reduced_function, Vmax_factor = 0.7, frequency = 0.02),
        "R50H" => (effect = :reduced_function, Vmax_factor = 0.5, frequency = 0.005)
    )
)

"""
OAT3 (SLC22A8) - Organic Anion Transporter 3

Basolateral uptake transporter with broader substrate specificity than OAT1.
Key substrates:
- Statins (pravastatin, rosuvastatin)
- Antivirals (acyclovir, zidovudine)
- Antibiotics (cephalosporins, benzylpenicillin)
- Endogenous compounds (estrone sulfate, DHEAS)

Km typically higher than OAT1 (lower affinity, higher capacity).
"""
const OAT3_RENAL_KINETICS = RenalTransporterKinetics(
    "OAT3",
    "SLC22A8",
    :basolateral,
    :uptake,
    33.0,           # Km (µM) - estrone sulfate reference
    680.0,          # Vmax (pmol/min/mg protein)
    0.40,           # Km CV
    0.38,           # Vmax CV
    ["estrone_sulfate", "pravastatin", "rosuvastatin", "benzylpenicillin", "cimetidine"],
    Dict(
        "probenecid" => 0.8,
        "novobiocin" => 2.5,
        "benzbromarone" => 0.3,
        "febuxostat" => 1.2
    ),
    Dict(
        "rs4149182" => (effect = :reduced_function, Vmax_factor = 0.75, frequency = 0.08),
        "I260R" => (effect = :loss_of_function, Vmax_factor = 0.1, frequency = 0.001)
    )
)

"""
OCT2 (SLC22A2) - Organic Cation Transporter 2

Major basolateral uptake transporter for cationic drugs.
Key substrates:
- Metformin (diabetes)
- Cisplatin (oncology - nephrotoxicity!)
- Oxaliplatin
- Amiloride

Clinically important for metformin PK and cisplatin nephrotoxicity.
Works in tandem with apical MATE1/MATE2-K for net secretion.
"""
const OCT2_RENAL_KINETICS = RenalTransporterKinetics(
    "OCT2",
    "SLC22A2",
    :basolateral,
    :uptake,
    95.0,           # Km (µM) - metformin reference
    1200.0,         # Vmax (pmol/min/mg protein) - high capacity
    0.32,           # Km CV
    0.35,           # Vmax CV
    ["metformin", "cisplatin", "oxaliplatin", "amiloride", "cimetidine", "ranitidine"],
    Dict(
        "cimetidine" => 95.0,
        "dolutegravir" => 1.9,
        "trimethoprim" => 32.0,
        "vandetanib" => 2.8,
        "ondansetron" => 12.0
    ),
    Dict(
        # 808G>T (rs316019) - reduced function allele
        "808G>T" => (effect = :reduced_function, Km_factor = 1.5, Vmax_factor = 0.85, frequency = 0.13),
        "596C>T" => (effect = :reduced_function, Vmax_factor = 0.6, frequency = 0.02)
    )
)

"""
MATE1 (SLC47A1) - Multidrug and Toxin Extrusion 1

Apical efflux transporter - H⁺/cation antiporter.
Works in series with OCT2 for net tubular secretion.
Inhibition causes accumulation of cations in proximal tubule cells.

Key substrates: metformin, oxaliplatin, acyclovir, ganciclovir
"""
const MATE1_RENAL_KINETICS = RenalTransporterKinetics(
    "MATE1",
    "SLC47A1",
    :apical,
    :efflux,
    220.0,          # Km (µM) - metformin reference
    850.0,          # Vmax (pmol/min/mg protein)
    0.38,           # Km CV
    0.40,           # Vmax CV
    ["metformin", "cimetidine", "oxaliplatin", "acyclovir", "thiamine"],
    Dict(
        "cimetidine" => 3.8,
        "pyrimethamine" => 0.08,   # Very potent inhibitor
        "trimethoprim" => 5.5,
        "ondansetron" => 0.45,
        "dolutegravir" => 6.2
    ),
    Dict(
        "rs2289669" => (effect = :reduced_expression, Vmax_factor = 0.8, frequency = 0.45),
        "rs8065082" => (effect = :increased_expression, Vmax_factor = 1.15, frequency = 0.30)
    )
)

"""
MATE2-K (SLC47A2) - Multidrug and Toxin Extrusion 2-K

Kidney-specific apical efflux transporter.
Lower expression than MATE1 but important for some substrates.
Complementary to MATE1 for cation efflux.
"""
const MATE2K_RENAL_KINETICS = RenalTransporterKinetics(
    "MATE2K",
    "SLC47A2",
    :apical,
    :efflux,
    310.0,          # Km (µM) - metformin reference
    420.0,          # Vmax (pmol/min/mg protein)
    0.42,           # Km CV
    0.45,           # Vmax CV
    ["metformin", "oxaliplatin", "cimetidine", "procainamide"],
    Dict(
        "pyrimethamine" => 0.03,   # Extremely potent
        "cimetidine" => 7.5,
        "ondansetron" => 0.85
    ),
    Dict(
        "rs12943590" => (effect = :reduced_expression, Vmax_factor = 0.65, frequency = 0.35)
    )
)

"""
URAT1 (SLC22A12) - Urate Transporter 1

Apical reabsorption transporter specific for uric acid.
Target for uricosuric drugs (probenecid, benzbromarone, lesinurad).
Mutations cause renal hypouricemia (increased urate excretion).
"""
const URAT1_KINETICS = RenalTransporterKinetics(
    "URAT1",
    "SLC22A12",
    :apical,
    :reabsorption,
    370.0,          # Km (µM) - uric acid
    520.0,          # Vmax (pmol/min/mg protein)
    0.35,           # Km CV
    0.38,           # Vmax CV
    ["uric_acid", "lactate", "nicotinate", "pyrazinoate"],
    Dict(
        "benzbromarone" => 0.018,  # Very potent uricosuric
        "lesinurad" => 3.5,
        "probenecid" => 12.0,
        "losartan" => 4.2,
        "RDEA3170" => 0.8
    ),
    Dict(
        # Loss-of-function mutations cause renal hypouricemia
        "W258X" => (effect = :loss_of_function, Vmax_factor = 0.0, frequency = 0.025),  # Japanese
        "R90H" => (effect = :reduced_function, Vmax_factor = 0.3, frequency = 0.003)
    )
)

# ============================================================================
# MICHAELIS-MENTEN CALCULATION FUNCTIONS
# ============================================================================

"""
    calculate_saturable_secretion(
        substrate_conc::Float64,
        kinetics::RenalTransporterKinetics;
        inhibitor_conc::Float64 = 0.0,
        inhibitor_name::String = "",
        inhibition_type::Symbol = :competitive,
        polymorphism::String = ""
    ) -> NamedTuple

Calculate saturable secretion rate using Michaelis-Menten kinetics.

# Arguments
- `substrate_conc`: Unbound substrate concentration at transporter (µM)
- `kinetics`: Transporter kinetic parameters
- `inhibitor_conc`: Inhibitor concentration (µM), default 0
- `inhibitor_name`: Name of inhibitor (must match kinetics.inhibitors key)
- `inhibition_type`: `:competitive`, `:noncompetitive`, or `:uncompetitive`
- `polymorphism`: Genetic variant to apply (e.g., "808G>T")

# Returns
NamedTuple with:
- `velocity`: Transport rate (pmol/min/mg protein)
- `fraction_of_vmax`: How saturated is the transporter (0-1)
- `apparent_Km`: Km after inhibition adjustment
- `ddi_ratio`: Fold-change due to inhibition

# Equations

**Competitive inhibition** (inhibitor competes for binding site):
```
v = Vmax × [S] / (Km × (1 + [I]/Ki) + [S])
```

**Non-competitive inhibition** (inhibitor binds separate site):
```
v = (Vmax / (1 + [I]/Ki)) × [S] / (Km + [S])
```

**Uncompetitive inhibition** (inhibitor binds ES complex):
```
v = Vmax × [S] / (Km + [S] × (1 + [I]/Ki))
```
"""
function calculate_saturable_secretion(
    substrate_conc::Float64,
    kinetics::RenalTransporterKinetics;
    inhibitor_conc::Float64 = 0.0,
    inhibitor_name::String = "",
    inhibition_type::Symbol = :competitive,
    polymorphism::String = ""
)
    # Get base kinetic parameters
    Km = kinetics.Km
    Vmax = kinetics.Vmax

    # Apply polymorphism effects
    if polymorphism != "" && haskey(kinetics.polymorphisms, polymorphism)
        poly = kinetics.polymorphisms[polymorphism]
        if haskey(poly, :Km_factor)
            Km *= poly.Km_factor
        end
        if haskey(poly, :Vmax_factor)
            Vmax *= poly.Vmax_factor
        end
    end

    # Calculate inhibition factor
    Ki = 0.0
    inhibition_factor = 1.0
    if inhibitor_conc > 0.0 && inhibitor_name != "" && haskey(kinetics.inhibitors, inhibitor_name)
        Ki = kinetics.inhibitors[inhibitor_name]
        inhibition_factor = 1.0 + inhibitor_conc / Ki
    end

    # Calculate velocity based on inhibition type
    apparent_Km = Km
    effective_Vmax = Vmax

    if inhibition_type == :competitive
        # Increases apparent Km, Vmax unchanged
        apparent_Km = Km * inhibition_factor
        velocity = (Vmax * substrate_conc) / (apparent_Km + substrate_conc)
    elseif inhibition_type == :noncompetitive
        # Decreases effective Vmax, Km unchanged
        effective_Vmax = Vmax / inhibition_factor
        velocity = (effective_Vmax * substrate_conc) / (Km + substrate_conc)
    elseif inhibition_type == :uncompetitive
        # Decreases both apparent Km and Vmax
        apparent_Km = Km / inhibition_factor
        effective_Vmax = Vmax / inhibition_factor
        velocity = (effective_Vmax * substrate_conc) / (apparent_Km + substrate_conc)
    else
        # Default: no inhibition
        velocity = (Vmax * substrate_conc) / (Km + substrate_conc)
    end

    # Calculate uninhibited velocity for DDI ratio
    velocity_uninhibited = (Vmax * substrate_conc) / (Km + substrate_conc)
    ddi_ratio = velocity_uninhibited > 0 ? velocity / velocity_uninhibited : 1.0

    # Fraction of Vmax (saturation)
    fraction_of_vmax = velocity / effective_Vmax

    return (
        velocity = velocity,
        fraction_of_vmax = fraction_of_vmax,
        apparent_Km = apparent_Km,
        effective_Vmax = effective_Vmax,
        ddi_ratio = ddi_ratio,
        transporter = kinetics.name,
        direction = kinetics.direction
    )
end

"""
    calculate_saturable_reabsorption(
        filtrate_conc::Float64,
        kinetics::RenalTransporterKinetics;
        urine_flow::Float64 = 1.0,
        kwargs...
    ) -> NamedTuple

Calculate saturable tubular reabsorption.

For reabsorption, the driving concentration is in the tubular filtrate,
not the plasma. Flow rate affects residence time and thus efficiency.

# Arguments
- `filtrate_conc`: Concentration in tubular fluid (µM)
- `kinetics`: Transporter kinetics (must have direction = :reabsorption)
- `urine_flow`: Urine flow rate (mL/min), affects contact time

# Returns
Same as `calculate_saturable_secretion` plus:
- `reabsorption_efficiency`: Fraction reabsorbed (0-1)
"""
function calculate_saturable_reabsorption(
    filtrate_conc::Float64,
    kinetics::RenalTransporterKinetics;
    urine_flow::Float64 = 1.0,
    inhibitor_conc::Float64 = 0.0,
    inhibitor_name::String = "",
    inhibition_type::Symbol = :competitive,
    polymorphism::String = ""
)
    # Verify this is a reabsorption transporter
    if kinetics.direction != :reabsorption
        @warn "Using secretion transporter $(kinetics.name) for reabsorption calculation"
    end

    # Calculate base transport rate
    result = calculate_saturable_secretion(
        filtrate_conc,
        kinetics;
        inhibitor_conc = inhibitor_conc,
        inhibitor_name = inhibitor_name,
        inhibition_type = inhibition_type,
        polymorphism = polymorphism
    )

    # Reabsorption efficiency depends on flow rate
    # Higher flow = less contact time = less reabsorption
    # Model: efficiency = transport_rate / (transport_rate + flow_factor)
    reference_flow = 1.0  # mL/min reference
    flow_factor = urine_flow / reference_flow

    # Maximum possible reabsorption rate
    max_reabsorption = filtrate_conc * urine_flow  # µmol/min (if 100% reabsorbed)

    # Actual reabsorption limited by transporter capacity
    # Convert pmol/min/mg to scaled rate
    scaling = IVIVE_SCALING.PTCPGK * IVIVE_SCALING.kidney_weight *
              IVIVE_SCALING.PT_fraction / 1e9  # Convert to µmol/min

    reabsorption_rate = result.velocity * scaling
    reabsorption_efficiency = min(1.0, reabsorption_rate / max(max_reabsorption, 1e-10))

    # Adjust for flow
    reabsorption_efficiency *= exp(-0.1 * (flow_factor - 1.0))
    reabsorption_efficiency = clamp(reabsorption_efficiency, 0.0, 0.99)

    return (
        result...,
        reabsorption_efficiency = reabsorption_efficiency,
        reabsorption_rate = reabsorption_rate,
        urine_flow = urine_flow
    )
end

"""
    TransporterDDIResult

Result of transporter-mediated drug-drug interaction analysis.
"""
struct TransporterDDIResult
    perpetrator::String
    victim::String
    transporter::String
    inhibition_type::Symbol
    Ki::Float64
    perpetrator_conc::Float64  # Cmax or Css
    ddi_ratio::Float64         # Fold-change in CL
    clinical_significance::Symbol  # :none, :weak, :moderate, :strong
    recommendation::String
end

"""
    calculate_transporter_ddi(
        perpetrator::String,
        victim::String,
        perpetrator_conc::Float64,
        kinetics::RenalTransporterKinetics;
        victim_conc::Float64 = 10.0,
        inhibition_type::Symbol = :competitive
    ) -> TransporterDDIResult

Predict renal transporter-mediated DDI.

Uses the basic static model (R-value approach):
- R = 1 + [I]/Ki for competitive inhibition

Clinical significance thresholds (FDA 2020 guidance):
- R < 1.25: No clinical DDI
- 1.25 ≤ R < 2: Weak DDI
- 2 ≤ R < 5: Moderate DDI
- R ≥ 5: Strong DDI

# Arguments
- `perpetrator`: Inhibitor drug name
- `victim`: Substrate drug name
- `perpetrator_conc`: Unbound Cmax or Css of inhibitor (µM)
- `kinetics`: Transporter kinetics
- `victim_conc`: Victim drug concentration (µM)
- `inhibition_type`: Type of inhibition

# Returns
`TransporterDDIResult` with prediction and recommendation
"""
function calculate_transporter_ddi(
    perpetrator::String,
    victim::String,
    perpetrator_conc::Float64,
    kinetics::RenalTransporterKinetics;
    victim_conc::Float64 = 10.0,
    inhibition_type::Symbol = :competitive
)
    # Look up Ki for perpetrator
    if !haskey(kinetics.inhibitors, perpetrator)
        return TransporterDDIResult(
            perpetrator, victim, kinetics.name, inhibition_type,
            NaN, perpetrator_conc, 1.0, :unknown,
            "No inhibition data available for $perpetrator on $(kinetics.name)"
        )
    end

    Ki = kinetics.inhibitors[perpetrator]

    # Calculate DDI ratio (R-value)
    if inhibition_type == :competitive
        R = 1.0 + perpetrator_conc / Ki
    elseif inhibition_type == :noncompetitive
        # For secretion: decreased clearance = increased exposure
        R = 1.0 + perpetrator_conc / Ki
    else
        R = 1.0 + perpetrator_conc / Ki  # Simplified
    end

    # For secretory clearance, inhibition DECREASES CL, INCREASES AUC
    # DDI ratio for exposure = R (fold increase)
    ddi_ratio = R

    # Clinical significance
    significance = if R < 1.25
        :none
    elseif R < 2.0
        :weak
    elseif R < 5.0
        :moderate
    else
        :strong
    end

    # Recommendation
    recommendation = if significance == :none
        "No dose adjustment required"
    elseif significance == :weak
        "Monitor patient; dose adjustment usually not required"
    elseif significance == :moderate
        "Consider 50% dose reduction; monitor renal function"
    else
        "Avoid combination or use 75% reduced dose with close monitoring"
    end

    return TransporterDDIResult(
        perpetrator, victim, kinetics.name, inhibition_type,
        Ki, perpetrator_conc, ddi_ratio, significance, recommendation
    )
end

"""
    RenalClearanceComponents

Breakdown of renal clearance into component processes.
"""
struct RenalClearanceComponents
    CLfiltration::Float64      # GFR × fu
    CLsecretion::Float64       # Active tubular secretion
    CLreabsorption::Float64    # Tubular reabsorption (negative contribution)
    CLrenal_total::Float64     # Net renal clearance
    fraction_filtered::Float64
    fraction_secreted::Float64
    fraction_reabsorbed::Float64
    secretion_saturation::Float64  # How saturated are secretory transporters
    rate_limiting_step::Symbol     # :filtration, :secretion, or :reabsorption
end

"""
    calculate_complete_renal_clearance_mm(
        plasma_conc::Float64,
        fu::Float64,
        GFR::Float64;
        OAT1_substrate::Bool = false,
        OAT3_substrate::Bool = false,
        OCT2_substrate::Bool = false,
        reabsorption_kinetics::Union{RenalTransporterKinetics,Nothing} = nothing,
        inhibitors::Dict{String,Float64} = Dict{String,Float64}(),
        polymorphisms::Dict{String,String} = Dict{String,String}()
    ) -> RenalClearanceComponents

Calculate complete renal clearance with Michaelis-Menten kinetics.

Integrates:
1. Glomerular filtration (linear with fu × GFR)
2. Active tubular secretion (saturable, multiple transporters)
3. Tubular reabsorption (saturable or passive)

# The Renal Clearance Equation

```
CLrenal = fu × GFR + CLsecretion - CLreabsorption
```

Where:
- `CLsecretion` = Σ (Vmax,i × [S]) / (Km,i + [S]) for each transporter i
- `CLreabsorption` = saturable or fraction reabsorbed × filtered load

# Arguments
- `plasma_conc`: Total plasma concentration (µM)
- `fu`: Fraction unbound in plasma
- `GFR`: Glomerular filtration rate (mL/min)
- `OAT1_substrate`: Drug is OAT1 substrate
- `OAT3_substrate`: Drug is OAT3 substrate
- `OCT2_substrate`: Drug is OCT2 substrate (enables MATE efflux)
- `reabsorption_kinetics`: Reabsorption transporter if applicable
- `inhibitors`: Dict of inhibitor => concentration (µM)
- `polymorphisms`: Dict of transporter => polymorphism (e.g., "OCT2" => "808G>T")

# Returns
`RenalClearanceComponents` with detailed breakdown
"""
function calculate_complete_renal_clearance_mm(
    plasma_conc::Float64,
    fu::Float64,
    GFR::Float64;
    OAT1_substrate::Bool = false,
    OAT3_substrate::Bool = false,
    OCT2_substrate::Bool = false,
    reabsorption_kinetics::Union{RenalTransporterKinetics,Nothing} = nothing,
    inhibitors::Dict{String,Float64} = Dict{String,Float64}(),
    polymorphisms::Dict{String,String} = Dict{String,String}()
)
    # Unbound concentration at transporters
    Cu = plasma_conc * fu

    # 1. Filtration clearance (always linear)
    CLfiltration = fu * GFR

    # 2. Secretion clearance (saturable)
    CLsecretion = 0.0
    max_saturation = 0.0

    # IVIVE scaling factor
    scaling = IVIVE_SCALING.PTCPGK * IVIVE_SCALING.kidney_weight *
              IVIVE_SCALING.PT_fraction * IVIVE_SCALING.microsomal_protein / 1e6
    # Units: cells × g × fraction × mg/g / 1e6 → normalization factor

    # OAT1 contribution
    if OAT1_substrate
        inhibitor_name = ""
        inhibitor_conc = 0.0
        for (inh, conc) in inhibitors
            if haskey(OAT1_RENAL_KINETICS.inhibitors, inh)
                inhibitor_name = inh
                inhibitor_conc = conc
                break
            end
        end
        poly = get(polymorphisms, "OAT1", "")

        result = calculate_saturable_secretion(
            Cu, OAT1_RENAL_KINETICS;
            inhibitor_conc = inhibitor_conc,
            inhibitor_name = inhibitor_name,
            polymorphism = poly
        )

        # Convert velocity to clearance: CL = v/[S] when not saturated
        # At saturation: CL approaches Vmax/[S] which decreases with [S]
        intrinsic_cl = result.velocity / max(Cu, 0.001) * scaling * 0.001  # mL/min

        # Well-stirred model for secretion
        Qkidney = IVIVE_SCALING.renal_plasma_flow
        CLsecretion += (intrinsic_cl * Qkidney) / (Qkidney + intrinsic_cl)

        max_saturation = max(max_saturation, result.fraction_of_vmax)
    end

    # OAT3 contribution
    if OAT3_substrate
        inhibitor_name = ""
        inhibitor_conc = 0.0
        for (inh, conc) in inhibitors
            if haskey(OAT3_RENAL_KINETICS.inhibitors, inh)
                inhibitor_name = inh
                inhibitor_conc = conc
                break
            end
        end
        poly = get(polymorphisms, "OAT3", "")

        result = calculate_saturable_secretion(
            Cu, OAT3_RENAL_KINETICS;
            inhibitor_conc = inhibitor_conc,
            inhibitor_name = inhibitor_name,
            polymorphism = poly
        )

        intrinsic_cl = result.velocity / max(Cu, 0.001) * scaling * 0.001
        Qkidney = IVIVE_SCALING.renal_plasma_flow
        CLsecretion += (intrinsic_cl * Qkidney) / (Qkidney + intrinsic_cl)

        max_saturation = max(max_saturation, result.fraction_of_vmax)
    end

    # OCT2 contribution (requires MATE for efflux)
    if OCT2_substrate
        inhibitor_name = ""
        inhibitor_conc = 0.0
        for (inh, conc) in inhibitors
            if haskey(OCT2_RENAL_KINETICS.inhibitors, inh)
                inhibitor_name = inh
                inhibitor_conc = conc
                break
            end
        end
        poly = get(polymorphisms, "OCT2", "")

        # OCT2 uptake
        oct2_result = calculate_saturable_secretion(
            Cu, OCT2_RENAL_KINETICS;
            inhibitor_conc = inhibitor_conc,
            inhibitor_name = inhibitor_name,
            polymorphism = poly
        )

        # MATE1 + MATE2-K efflux (in series with OCT2)
        mate1_result = calculate_saturable_secretion(Cu * 2.0, MATE1_RENAL_KINETICS)  # Intracellular accumulation
        mate2k_result = calculate_saturable_secretion(Cu * 2.0, MATE2K_RENAL_KINETICS)

        # Rate-limiting step determines overall secretion
        oct2_velocity = oct2_result.velocity
        mate_velocity = mate1_result.velocity + mate2k_result.velocity
        effective_velocity = min(oct2_velocity, mate_velocity)

        intrinsic_cl = effective_velocity / max(Cu, 0.001) * scaling * 0.0005
        Qkidney = IVIVE_SCALING.renal_plasma_flow
        CLsecretion += (intrinsic_cl * Qkidney) / (Qkidney + intrinsic_cl)

        max_saturation = max(max_saturation, oct2_result.fraction_of_vmax)
    end

    # 3. Reabsorption (if applicable)
    CLreabsorption = 0.0
    if reabsorption_kinetics !== nothing
        # Tubular fluid concentration (after filtration and secretion)
        tubular_conc = Cu * (1.0 + CLsecretion / max(CLfiltration, 1.0))

        result = calculate_saturable_reabsorption(
            tubular_conc,
            reabsorption_kinetics;
            urine_flow = 1.0
        )

        # Reabsorption reduces effective clearance
        filtered_load = CLfiltration + CLsecretion
        CLreabsorption = result.reabsorption_efficiency * filtered_load
    end

    # Net renal clearance
    CLrenal_total = max(0.0, CLfiltration + CLsecretion - CLreabsorption)

    # Fractions
    total_handling = CLfiltration + CLsecretion
    fraction_filtered = CLfiltration / max(total_handling, 1e-10)
    fraction_secreted = CLsecretion / max(total_handling, 1e-10)
    fraction_reabsorbed = CLreabsorption / max(total_handling, 1e-10)

    # Rate-limiting step
    rate_limiting = if CLsecretion > CLfiltration * 2
        :secretion
    elseif CLreabsorption > CLfiltration * 0.5
        :reabsorption
    else
        :filtration
    end

    return RenalClearanceComponents(
        CLfiltration,
        CLsecretion,
        CLreabsorption,
        CLrenal_total,
        fraction_filtered,
        fraction_secreted,
        fraction_reabsorbed,
        max_saturation,
        rate_limiting
    )
end

"""
    estimate_renal_ddi_risk(
        perpetrator::String,
        perpetrator_cmax::Float64;
        transporters::Vector{Symbol} = [:OAT1, :OAT3, :OCT2, :MATE1]
    ) -> Dict{Symbol, TransporterDDIResult}

Screen a perpetrator drug for renal transporter DDI risk.

# Arguments
- `perpetrator`: Drug name (must match inhibitor entries)
- `perpetrator_cmax`: Unbound Cmax (µM)
- `transporters`: Which transporters to screen

# Returns
Dictionary mapping transporter to DDI result
"""
function estimate_renal_ddi_risk(
    perpetrator::String,
    perpetrator_cmax::Float64;
    transporters::Vector{Symbol} = [:OAT1, :OAT3, :OCT2, :MATE1]
)
    results = Dict{Symbol, TransporterDDIResult}()

    kinetics_map = Dict(
        :OAT1 => OAT1_RENAL_KINETICS,
        :OAT3 => OAT3_RENAL_KINETICS,
        :OCT2 => OCT2_RENAL_KINETICS,
        :MATE1 => MATE1_RENAL_KINETICS,
        :MATE2K => MATE2K_RENAL_KINETICS,
        :URAT1 => URAT1_KINETICS
    )

    for transporter in transporters
        if haskey(kinetics_map, transporter)
            kinetics = kinetics_map[transporter]
            result = calculate_transporter_ddi(
                perpetrator,
                "probe_substrate",
                perpetrator_cmax,
                kinetics
            )
            results[transporter] = result
        end
    end

    return results
end

end # module
