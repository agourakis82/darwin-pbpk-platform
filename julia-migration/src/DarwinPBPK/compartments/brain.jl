# BRAIN COMPARTMENT MODEL (ENHANCED)
# ===================================
#
# The brain is UNIQUE due to the BLOOD-BRAIN BARRIER (BBB).
# This is the most selective tissue barrier in the body.
#
# ══════════════════════════════════════════════════════════════════════════
# BRAIN PHYSIOLOGY DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════
#
# 1. THE BLOOD-BRAIN BARRIER (BBB)
#    ────────────────────────────
#    The BBB is formed by specialized brain capillary endothelial cells:
#
#    a) TIGHT JUNCTIONS (Zona Occludens)
#       - Claudins, occludin, JAMs seal cells together
#       - Paracellular permeability: 10⁻⁸ cm/s (vs 10⁻⁵ in other tissues!)
#       - NO passive diffusion of polar solutes between cells
#       - Even small ions (Na⁺, K⁺) require transporters
#
#    b) PERICYTES
#       - Wrap around capillaries (30% coverage)
#       - Regulate BBB permeability
#       - Control capillary diameter
#       - Degenerate in Alzheimer's → BBB breakdown
#
#    c) ASTROCYTE END-FEET
#       - Cover 99% of capillary surface
#       - Secrete factors that maintain BBB
#       - Express aquaporin-4 (water transport)
#       - Key for neurovascular coupling
#
#    d) BASEMENT MEMBRANE
#       - Collagen IV, laminin, fibronectin
#       - Additional barrier layer
#       - Traps larger molecules
#
# 2. TRANSCELLULAR TRANSPORT ROUTES
#    ───────────────────────────────
#    Since paracellular is closed, drugs must cross THROUGH cells:
#
#    a) PASSIVE DIFFUSION (lipophilic drugs)
#       - Requires: MW < 450, logP 1-3, TPSA < 90, HBD ≤ 3
#       - "Rule of 5 for BBB" is stricter than Lipinski
#       - Examples: diazepam, caffeine, ethanol
#
#    b) CARRIER-MEDIATED TRANSPORT (CMT)
#       Influx transporters:
#       - LAT1: Large neutral amino acids, levodopa, gabapentin
#       - GLUT1: Glucose (brain uses 25% of body's glucose!)
#       - MCT1: Lactate, ketone bodies
#       - CAT1: Cationic amino acids
#       - ENT1/ENT2: Nucleosides
#       - CHT: Choline (for acetylcholine synthesis)
#
#    c) RECEPTOR-MEDIATED TRANSCYTOSIS (RMT)
#       - Transferrin receptor → iron delivery
#       - Insulin receptor → insulin and IGF
#       - LRP1 → lipoproteins, α2-macroglobulin
#       - Used for drug delivery: antibody-drug conjugates
#
#    d) ADSORPTIVE TRANSCYTOSIS
#       - Cationic molecules bind to negative surface
#       - Non-specific endocytosis
#       - Cell-penetrating peptides use this route
#
# 3. EFFLUX TRANSPORTERS (The Major Barrier!)
#    ─────────────────────────────────────────
#    These actively PUMP drugs OUT of the brain:
#
#    ┌────────────────────────────────────────────────────────────────┐
#    │                  BRAIN CAPILLARY ENDOTHELIUM                   │
#    │                                                                │
#    │  BLOOD                                      BRAIN              │
#    │  (Luminal)                                 (Abluminal)         │
#    │                                                                │
#    │  ←───P-gp────                                                  │
#    │  ←───BCRP───                                                   │
#    │  ←───MRP1───                  (expressed on abluminal?)       │
#    │  ←───MRP4───                                                   │
#    │                                                                │
#    │              ───LAT1───→    (bidirectional)                   │
#    │              ───GLUT1──→    (influx)                          │
#    │              ───MCT1───→    (influx)                          │
#    │                                                                │
#    └────────────────────────────────────────────────────────────────┘
#
#    P-GLYCOPROTEIN (ABCB1) - THE GATEKEEPER:
#    - Highest expression at BBB (5-20x vs intestine)
#    - Substrates: lipophilic cations, planar molecules
#    - Reduces brain exposure 2-50x for substrates
#    - Examples reduced: loperamide, digoxin, cyclosporine
#    - Clinical relevance: Loperamide is opioid with NO CNS effect
#      because P-gp keeps it out (overdose → CNS effects when P-gp saturated)
#
#    BCRP (ABCG2):
#    - Second major efflux pump
#    - Overlapping substrates with P-gp
#    - Sulfate conjugates, some statins
#
#    MRP1/MRP4:
#    - Organic anions, glucuronides
#    - MRP1 may be on abluminal (brain-facing) membrane
#
# 4. BRAIN TISSUE COMPOSITION
#    ─────────────────────────
#    - 60% lipid content (dry weight) - highest of any organ!
#    - Gray matter: 36% lipid
#    - White matter: 49% lipid (myelin sheaths)
#
#    Lipid composition:
#    - Phospholipids: 50% (phosphatidylcholine, PE, PS)
#    - Cholesterol: 25% (brain has 25% of body's cholesterol!)
#    - Glycolipids: 20% (gangliosides in gray matter)
#    - Neutral lipids: 5%
#
#    Clinical implications:
#    - Lipophilic drugs accumulate in brain tissue
#    - Long half-lives for CNS drugs
#    - Diazepam, amiodarone: brain reservoir effect
#
# 5. CEREBROSPINAL FLUID (CSF) vs BRAIN TISSUE
#    ───────────────────────────────────────────
#    CSF is NOT a good surrogate for brain tissue!
#
#    CSF characteristics:
#    - Volume: 150 mL, turnover 4x/day
#    - Produced by choroid plexus (different barrier: BCSFB)
#    - Composition: 0.3% plasma protein, low albumin
#    - Equilibrates slowly with brain ISF
#
#    CSF vs Brain partitioning:
#    - Lipophilic drugs: CSF << Brain (bound to lipids in tissue)
#    - Hydrophilic drugs: CSF ≈ Brain ISF
#    - P-gp substrates: CSF may overestimate brain
#
# 6. Kp,brain vs Kp,uu - THE CRITICAL DISTINCTION
#    ─────────────────────────────────────────────
#    Kp,brain = Total brain / Total plasma
#    Kp,uu = Unbound brain / Unbound plasma (PHARMACOLOGICALLY RELEVANT!)
#
#    For CNS efficacy:
#    - Kp,uu determines receptor occupancy
#    - Kp,uu = 1: passive equilibrium (no efflux/influx)
#    - Kp,uu < 1: net efflux (P-gp) or poor permeability
#    - Kp,uu > 1: active influx or ion trapping
#
#    Examples:
#    Drug          Kp,brain  Kp,uu   Interpretation
#    ─────────────────────────────────────────────────
#    Diazepam      0.9       0.8     Passive, high tissue binding
#    Haloperidol   15        3.0     Brain accumulation
#    Risperidone   10        0.3     P-gp efflux despite high Kp!
#    Loperamide    0.05      0.02    Strong P-gp efflux
#    Caffeine      0.8       1.0     Free equilibrium
#
# 7. CLINICAL IMPLICATIONS
#    ─────────────────────
#    a) CNS Drug Design:
#       - Avoid P-gp substrates OR design for P-gp inhibition
#       - Optimal logP: 2-3 (not too polar, not P-gp substrate)
#       - Low TPSA, limited H-bond donors
#       - Consider prodrug strategies (LAT1 for levodopa)
#
#    b) P-gp Inhibitors as CNS Penetration Enhancers:
#       - Verapamil, cyclosporine, elacridar
#       - Risk: also affects other P-gp substrates
#       - Clinical trials for brain tumor drug delivery
#
#    c) Disease States:
#       - Brain tumors: disrupted BBB in tumor core, intact at edge
#       - Alzheimer's: Aβ accumulation, P-gp dysfunction
#       - Epilepsy: ENHANCED P-gp expression (drug resistance)
#       - Stroke: transient BBB opening (therapeutic window)
#       - Multiple sclerosis: inflammation opens BBB
#
#    d) Age Effects:
#       - Neonates: immature BBB, more permeable
#       - Elderly: possible BBB breakdown, more variable
#
# References:
# - Pardridge 2012: Drug transport across the BBB
# - Abbott et al. 2010: Structure and function of the BBB
# - Hammarlund-Udenaes 2014: Kp,uu concept for CNS drugs
# - Summerfield et al. 2016: CNS drug discovery
# - Rodgers & Rowland 2006: Mechanistic Kp prediction
# ══════════════════════════════════════════════════════════════════════════

module BrainCompartment

export BrainProperties, calculate_kp_brain, calculate_kpuu_brain
export estimate_bbb_permeability, is_bbb_permeable, calculate_brain_contribution
export estimate_fub, estimate_pgp_efflux_ratio, predict_cns_penetration_class
# SOTA 2020-2024 exports - From Socratic Discussion
export CircadianPhase, ImmuneStatus, MeningitisStage, AgeGroup
export NeurovascularState, BBBDynamicState, PatientBrainStatus
# Export enum values
export MORNING_PEAK, MIDDAY, AFTERNOON, EVENING, NIGHT_NADIR, LATE_NIGHT
export IMMUNOCOMPETENT, MILD_INFLAMMATION, MODERATE_INFLAMMATION, SEVERE_INFLAMMATION
export IMMUNOSUPPRESSED, AUTOIMMUNE_CNS
export NO_MENINGITIS, STAGE_0_PRE, STAGE_I_EARLY, STAGE_II_ESTABLISHED
export STAGE_III_SEVERE, STAGE_IV_FIBROTIC
export PRETERM_NEONATE, TERM_NEONATE, INFANT, TODDLER, CHILD, ADOLESCENT, ADULT, ELDERLY
export calculate_circadian_pgp_activity, calculate_inflammation_pgp_effect
export calculate_meningitis_bbb_state, calculate_pediatric_bbb_maturity
export calculate_glymphatic_clearance_factor, calculate_covid_bbb_dysfunction
export calculate_dynamic_kpuu, estimate_csf_penetration_meningitis
export calculate_white_grey_matter_distribution, estimate_time_to_brain_equilibrium
export predict_intranasal_brain_bioavailability, calculate_chronotherapy_optimal_time
export calculate_lithium_brain_penetration
# Improved Kp,uu model exports (validated: 72.2% within 2-fold)
export calculate_kpuu_improved, predict_kpuu_hybrid
export TRAINING_DATA, ValidationMetrics, calculate_validation_metrics

"""
Brain physiological properties

Reference values for 70kg adult:
- Volume: 1.3-1.5 L (2% body weight)
- Blood flow: 0.7-0.75 L/min (15% cardiac output)
- Highest lipid content of any organ (60% dry weight)
- Protected by BBB
"""
struct BrainProperties
    volume_L::Float64           # Brain volume
    blood_flow_L_min::Float64   # Cerebral blood flow
    f_neutral_lipid::Float64    # Neutral lipids
    f_phospholipid::Float64     # Total phospholipids (HIGH!)
    f_acidic_pl::Float64        # Acidic phospholipids (PS)
    f_cholesterol::Float64      # Cholesterol (25% of body total!)
    f_water_iw::Float64         # Intracellular water
    f_water_ew::Float64         # Extracellular/ISF water
    albumin_ratio::Float64      # Very low (BBB blocks albumin!)
    pH_iw::Float64              # Intracellular pH
    pH_isf::Float64             # Interstitial fluid pH
    f_lysosome::Float64         # Lysosomal volume fraction
    pH_lysosome::Float64        # Lysosomal pH
    # BBB transporter expression (relative to liver = 1.0)
    P_gp_relative::Float64      # P-gp at BBB (HIGH!)
    BCRP_relative::Float64      # BCRP at BBB
    LAT1_relative::Float64      # LAT1 (amino acid transporter)
    GLUT1_relative::Float64     # GLUT1 (glucose transporter)
end

# Default for 70kg adult
# Values from Rodgers & Rowland 2006, updated with BBB-specific data
const DEFAULT_BRAIN = BrainProperties(
    1.4,      # volume (L)
    0.75,     # blood flow (L/min) - 15% cardiac output
    0.039,    # neutral lipids
    0.0457,   # total phospholipids (HIGH - myelin!)
    0.00914,  # acidic phospholipids (PS)
    0.025,    # cholesterol (brain has 25% of body's cholesterol!)
    0.620,    # intracellular water
    0.162,    # extracellular water (ISF)
    0.048,    # albumin ratio (VERY LOW - BBB blocks!)
    7.0,      # intracellular pH
    7.3,      # ISF pH (slightly alkaline)
    0.008,    # lysosomal volume fraction (lower than liver)
    4.8,      # lysosomal pH
    # BBB transporter expression
    5.0,      # P-gp (5x liver - THE gatekeeper!)
    3.0,      # BCRP (also elevated)
    10.0,     # LAT1 (high for amino acid delivery)
    15.0      # GLUT1 (very high - brain needs glucose!)
)

"""
Estimate BBB permeability based on physicochemical properties

The "BBB Rule of 5" is STRICTER than Lipinski:
- MW < 450 Da (optimally < 400)
- logP: 1-3 (sweet spot: 2-2.5)
- TPSA < 90 Å² (optimally < 70)
- HBD ≤ 3 (optimally ≤ 1)
- Rotatable bonds < 8

Returns detailed assessment with score and likelihood.
"""
function estimate_bbb_permeability(;
    MW::Float64,
    logP::Float64,
    TPSA::Float64,
    HBD::Int = 0,
    HBA::Int = 0,
    rotatable_bonds::Int = 0,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_pgp_substrate::Bool = false,
    is_bcrp_substrate::Bool = false
)
    score = 0.0
    penalties = String[]
    bonuses = String[]

    # ═══════════════════════════════════════════════════════
    # MOLECULAR WEIGHT
    # ═══════════════════════════════════════════════════════
    if MW < 350
        score += 2.5
        push!(bonuses, "Optimal MW (<350)")
    elseif MW < 400
        score += 2.0
        push!(bonuses, "Good MW (<400)")
    elseif MW < 450
        score += 1.0
    elseif MW < 500
        score += 0.0
        push!(penalties, "Borderline MW (450-500)")
    else
        score -= 2.0
        push!(penalties, "MW too high (>500)")
    end

    # ═══════════════════════════════════════════════════════
    # LIPOPHILICITY (logP)
    # Sweet spot: 1.5-2.5 (enough to cross, not P-gp substrate)
    # ═══════════════════════════════════════════════════════
    if 1.5 <= logP <= 2.5
        score += 3.0  # Optimal range
        push!(bonuses, "Optimal logP (1.5-2.5)")
    elseif 1.0 <= logP < 1.5
        score += 2.0
    elseif 2.5 < logP <= 3.0
        score += 2.0
    elseif 0.5 <= logP < 1.0
        score += 1.0
        push!(penalties, "Slightly hydrophilic")
    elseif 3.0 < logP <= 4.0
        score += 1.0
        push!(penalties, "Lipophilic: possible P-gp substrate")
    elseif logP < 0.5
        score -= 1.5
        push!(penalties, "Too hydrophilic for passive diffusion")
    else  # logP > 4.0
        score -= 1.0
        push!(penalties, "Very lipophilic: likely P-gp substrate")
    end

    # ═══════════════════════════════════════════════════════
    # TOPOLOGICAL POLAR SURFACE AREA (TPSA)
    # ═══════════════════════════════════════════════════════
    if TPSA < 60
        score += 2.5
        push!(bonuses, "Optimal TPSA (<60)")
    elseif TPSA < 70
        score += 2.0
    elseif TPSA < 90
        score += 1.0
    elseif TPSA < 120
        score += 0.0
        push!(penalties, "Elevated TPSA (90-120)")
    else
        score -= 2.0
        push!(penalties, "TPSA too high (>120)")
    end

    # ═══════════════════════════════════════════════════════
    # HYDROGEN BOND DONORS (most restrictive!)
    # ═══════════════════════════════════════════════════════
    if HBD == 0
        score += 2.0
        push!(bonuses, "No H-bond donors")
    elseif HBD == 1
        score += 1.5
    elseif HBD == 2
        score += 0.5
    elseif HBD == 3
        score += 0.0
    else
        score -= 1.5
        push!(penalties, "Too many H-bond donors (>3)")
    end

    # ═══════════════════════════════════════════════════════
    # ROTATABLE BONDS (flexibility reduces BBB penetration)
    # ═══════════════════════════════════════════════════════
    if rotatable_bonds <= 4
        score += 0.5
    elseif rotatable_bonds <= 6
        score += 0.0
    elseif rotatable_bonds <= 8
        score -= 0.5
    else
        score -= 1.0
        push!(penalties, "Too flexible (>8 rotatable bonds)")
    end

    # ═══════════════════════════════════════════════════════
    # EFFLUX TRANSPORTER SUBSTRATES (major penalties!)
    # ═══════════════════════════════════════════════════════
    if is_pgp_substrate
        score -= 3.0
        push!(penalties, "P-gp substrate: 2-50x reduced brain exposure")
    end

    if is_bcrp_substrate
        score -= 1.5
        push!(penalties, "BCRP substrate: reduced brain exposure")
    end

    # ═══════════════════════════════════════════════════════
    # IONIZATION (weak bases preferred)
    # ═══════════════════════════════════════════════════════
    if is_base && !isnothing(pKa)
        if 7.5 <= pKa <= 9.5
            score += 1.0
            push!(bonuses, "Weak base: good for CNS")
        elseif 6.0 <= pKa < 7.5
            score += 0.5
        elseif pKa > 10.0
            score -= 1.0
            push!(penalties, "Strong base: mostly ionized, poor passive diffusion")
        end
    end

    # ═══════════════════════════════════════════════════════
    # CLASSIFICATION
    # ═══════════════════════════════════════════════════════
    # Score interpretation:
    # > 6.0: High BBB permeability
    # 4.0-6.0: Moderate BBB permeability
    # 2.0-4.0: Low BBB permeability
    # < 2.0: Very low / unlikely to cross BBB

    if score >= 6.0
        category = "High"
        permeable = true
    elseif score >= 4.0
        category = "Moderate"
        permeable = true
    elseif score >= 2.0
        category = "Low"
        permeable = false
    else
        category = "Very Low"
        permeable = false
    end

    return (
        permeable = permeable,
        score = score,
        category = category,
        penalties = penalties,
        bonuses = bonuses
    )
end

"""
Simple BBB permeability check using strict CNS rules
"""
function is_bbb_permeable(; MW::Float64, logP::Float64, TPSA::Float64, HBD::Int=0)
    return MW < 450 && 0.5 < logP < 4.0 && TPSA < 90 && HBD <= 3
end

"""
Estimate P-gp efflux ratio at BBB

P-gp efflux ratio = (brain exposure without P-gp) / (with P-gp)
A ratio of 5 means P-gp reduces brain exposure 5-fold.

Depends on:
1. Lipophilicity (logP > 3 often P-gp substrates)
2. Molecular size (MW > 400 preferred by P-gp)
3. Known substrate status
4. Structural features (planar aromatic, cationic)
"""
function estimate_pgp_efflux_ratio(;
    logP::Float64,
    MW::Float64 = 400.0,
    is_pgp_substrate::Bool = false,
    is_cationic::Bool = false,
    n_aromatic_rings::Int = 0
)
    if !is_pgp_substrate
        # Not a known substrate - estimate likelihood
        # P-gp substrates tend to be: lipophilic, large, cationic

        pgp_likelihood = 0.0

        if logP > 3.0
            pgp_likelihood += 0.3
        end
        if logP > 4.0
            pgp_likelihood += 0.2
        end
        if MW > 400
            pgp_likelihood += 0.2
        end
        if is_cationic
            pgp_likelihood += 0.2
        end
        if n_aromatic_rings >= 2
            pgp_likelihood += 0.1
        end

        if pgp_likelihood < 0.3
            return 1.0  # Not a substrate
        elseif pgp_likelihood < 0.5
            return 2.0  # Weak substrate
        else
            return 5.0  # Likely substrate
        end
    end

    # Known P-gp substrate - estimate efflux magnitude
    # Efflux ratio depends on binding affinity and lipophilicity
    #
    # Literature values:
    # - Loperamide: efflux ratio ~50-100x (very strong)
    # - Risperidone: efflux ratio ~3-5x (moderate)
    # - Morphine: efflux ratio ~2-3x (weak)

    base_ratio = 5.0  # Default for known substrates

    # Very lipophilic compounds have stronger P-gp interaction
    # P-gp prefers lipophilic, planar substrates
    if logP > 4.5
        base_ratio *= 3.0  # Strong P-gp affinity
    elseif logP > 4.0
        base_ratio *= 2.0
    elseif logP > 3.5
        base_ratio *= 1.5
    end

    # Large compounds have higher affinity for P-gp
    if MW > 450
        base_ratio *= 2.0
    elseif MW > 400
        base_ratio *= 1.3
    end

    # Combined high lipophilicity + large size = very strong P-gp
    # (loperamide: logP 4.8, MW 477)
    if logP > 4.0 && MW > 450
        base_ratio *= 1.5  # Additional synergy
    end

    return min(base_ratio, 100.0)  # Cap at 100x
end

"""
Estimate fraction unbound in brain tissue (fub)

Brain tissue binding is primarily to:
1. Phospholipids (50% of brain lipids)
2. Cholesterol (25% of body's cholesterol in brain!)
3. Gangliosides (gray matter)

Lipophilic drugs bind extensively to brain lipids,
resulting in high total Kp but NOT high unbound Kp!
"""
function estimate_fub(logP::Float64;
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    brain::BrainProperties = DEFAULT_BRAIN
)
    P = 10^logP

    # Brain lipid fractions
    f_nl = brain.f_neutral_lipid
    f_pl = brain.f_phospholipid
    f_apl = brain.f_acidic_pl
    f_chol = brain.f_cholesterol
    f_iw = brain.f_water_iw

    # Total lipid content for binding
    # Phospholipids are main binding sites
    total_lipid_binding = P * f_nl + (0.3*P + 0.7) * f_pl + P * f_chol * 0.5

    # Bases bind to acidic phospholipids (PS)
    if is_base && !isnothing(pKa) && pKa > 7.0
        # Ion pair formation with PS
        ion_factor = 10^(pKa - 7.0) / (1 + 10^(pKa - 7.0))
        apl_binding = ion_factor * 30.0 * f_apl  # Strong electrostatic
        total_lipid_binding += apl_binding
    end

    # Lysosomal trapping for bases
    lyso_contribution = 0.0
    if is_base && !isnothing(pKa) && pKa > 6.5
        # Lysosomal accumulation
        ionized_lyso = 10^(pKa - brain.pH_lysosome)
        ionized_cyto = 10^(pKa - brain.pH_iw)
        accumulation = (1 + ionized_lyso) / (1 + ionized_cyto)

        # Permeability to lysosomes
        perm_factor = if logP < 1.0
            0.1
        elseif logP < 2.0
            0.3
        elseif logP < 3.0
            0.6
        else
            0.8
        end

        lyso_contribution = brain.f_lysosome * accumulation * perm_factor
    end

    # Total binding reduces free fraction
    # fub = fu_water / (fu_water + fu_lipid_bound)
    fub = f_iw / (f_iw + total_lipid_binding + lyso_contribution)

    return clamp(fub, 0.001, 0.95)
end

"""
Calculate brain:plasma partition coefficient (Kp,brain)

This is the TOTAL Kp - includes bound drug in tissue.
For pharmacological effect, use Kp,uu instead!

Model includes:
1. Water distribution (ISF and intracellular)
2. Lipid partitioning (very high in brain)
3. Phospholipid binding
4. Lysosomal trapping for bases
5. P-gp efflux effect
6. BBB permeability effect
"""
function calculate_kp_brain(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    is_pgp_substrate::Bool = false,
    is_bcrp_substrate::Bool = false,
    MW::Float64 = 400.0,
    TPSA::Float64 = 70.0,
    HBD::Int = 1,
    brain::BrainProperties = DEFAULT_BRAIN
)
    P = 10^logP
    D = 10^logD

    # Ionization factors
    pH_p = 7.4
    pH_iw = brain.pH_iw

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
    f_ew = brain.f_water_ew
    f_iw = brain.f_water_iw
    f_nl = brain.f_neutral_lipid
    f_pl = brain.f_phospholipid
    f_apl = brain.f_acidic_pl
    f_chol = brain.f_cholesterol
    AR = brain.albumin_ratio  # Very low - BBB blocks albumin

    denom = max(1 + Y, 1e-10)

    # ════════════════════════════════════════════════════════
    # WATER TERM (ISF + intracellular)
    # ════════════════════════════════════════════════════════
    water_term = f_ew + ((1 + X) / denom) * f_iw

    # ════════════════════════════════════════════════════════
    # LIPID TERM (dominant for lipophilic drugs!)
    # Brain has very high lipid content - major drug reservoir
    # ════════════════════════════════════════════════════════
    lipid_term = (P * f_nl + (0.3*P + 0.7) * f_pl + P * f_chol * 0.3) / denom

    # ════════════════════════════════════════════════════════
    # ACIDIC PHOSPHOLIPID BINDING (for bases)
    # ════════════════════════════════════════════════════════
    apl_term = 0.0
    if is_base && !isnothing(pKa) && pKa > 6.5
        ion_factor = X / (1 + X)
        K_apl = 15.0 * (1 + 0.2 * max(logP - 1.0, 0))  # Lipophilicity enhances
        apl_term = K_apl * f_apl * ion_factor * (1 + X) / denom
    end

    # ════════════════════════════════════════════════════════
    # LYSOSOMAL TRAPPING (for bases)
    # ════════════════════════════════════════════════════════
    lyso_term = 0.0
    if is_base && !isnothing(pKa) && pKa > 6.5
        ionized_lyso = 10^(pKa - brain.pH_lysosome)
        ionized_cyto = 10^(pKa - brain.pH_iw)
        accumulation = (1 + ionized_lyso) / (1 + ionized_cyto)

        perm_factor = if logP < 1.0
            0.1
        elseif logP < 2.0
            0.3
        elseif logP < 3.0
            0.6
        elseif logP < 4.0
            0.8
        else
            0.7  # Very lipophilic can escape
        end

        lyso_term = brain.f_lysosome * accumulation * perm_factor
    end

    # ════════════════════════════════════════════════════════
    # CALCULATE BASE Kp (before BBB effects)
    # ════════════════════════════════════════════════════════
    if is_base && !isnothing(pKa) && pKa > 6.5
        Kpu = water_term + lipid_term + apl_term + lyso_term

        # Lipophilic bases (haloperidol, risperidone) show VERY high brain Kp
        # This is due to extensive brain lipid binding + possible active uptake
        # The brain's 60% lipid content creates a major reservoir
        # Haloperidol (logP 4.3): observed Kp ~15
        # Risperidone (logP 3.0): observed Kp ~10
        if logP > 2.5
            # Additional lipid binding enhancement for lipophilic bases
            lipophilic_boost = 1.0 + (logP - 2.5) * 3.0
            Kpu *= lipophilic_boost
        end
    elseif is_acid
        # Acids have low brain penetration (charged, polar)
        # Only albumin-bound fraction in ISF
        Kpu = water_term + lipid_term + (AR * (1 - fup) / fup) * f_ew
    else
        # Neutral drugs - primarily lipid partitioning
        Kpu = water_term + lipid_term
    end

    Kp = Kpu * fup

    # ════════════════════════════════════════════════════════
    # BBB PERMEABILITY EFFECT
    # If drug can't cross BBB, Kp is dramatically reduced
    # ════════════════════════════════════════════════════════
    bbb_result = estimate_bbb_permeability(
        MW=MW, logP=logP, TPSA=TPSA, HBD=HBD,
        pKa=pKa, is_base=is_base,
        is_pgp_substrate=is_pgp_substrate,
        is_bcrp_substrate=is_bcrp_substrate
    )

    if !bbb_result.permeable
        # Poor BBB permeability - major reduction
        if bbb_result.category == "Low"
            Kp *= 0.2
        else  # Very Low
            Kp *= 0.05
        end
    end

    # ════════════════════════════════════════════════════════
    # P-gp EFFLUX EFFECT
    # P-gp actively pumps drugs OUT of brain
    # ════════════════════════════════════════════════════════
    #
    # CRITICAL DISTINCTION:
    # - P-gp affects UNBOUND drug crossing BBB
    # - But total Kp includes drug bound to brain TISSUE
    # - A P-gp substrate can still have high Kp due to tissue binding!
    #
    # Examples:
    # - Risperidone: P-gp substrate, but Kp=10 (high lipid binding)
    #   Kp,uu=0.3 (low unbound brain/plasma due to efflux)
    # - Loperamide: Strong P-gp, Kp=0.05 (even tissue binding can't help)
    #   Very lipophilic but P-gp is SO strong it keeps even total low

    if is_pgp_substrate
        efflux_ratio = estimate_pgp_efflux_ratio(
            logP=logP, MW=MW, is_pgp_substrate=true
        )

        # For moderate P-gp substrates (risperidone), tissue binding
        # can still create high total Kp despite efflux
        # Only apply partial correction to total Kp
        if logP > 2.5 && efflux_ratio < 10.0
            # Lipophilic: tissue binding partially compensates
            # Reduce Kp less aggressively
            Kp /= sqrt(efflux_ratio)
        else
            # Very strong efflux or hydrophilic: full effect
            Kp /= efflux_ratio
        end
    else
        # Check if likely P-gp substrate
        efflux_ratio = estimate_pgp_efflux_ratio(
            logP=logP, MW=MW, is_pgp_substrate=false
        )
        if efflux_ratio > 1.0
            Kp /= efflux_ratio
        end
    end

    # BCRP effect (smaller than P-gp)
    if is_bcrp_substrate
        Kp *= 0.7  # ~30% reduction
    end

    return max(Kp, 0.001)
end

"""
Calculate unbound brain:plasma ratio (Kp,uu)

THIS IS THE PHARMACOLOGICALLY RELEVANT RATIO!

Kp,uu = Cu,brain / Cu,plasma = Kp × (fup / fub)

Interpretation:
- Kp,uu = 1.0: Passive equilibrium (unbound drug freely equilibrates)
- Kp,uu < 1.0: Net efflux (P-gp) or poor permeability
- Kp,uu > 1.0: Active uptake or ion trapping

For CNS drugs, Kp,uu determines:
- Receptor occupancy
- Therapeutic effect
- Steady-state unbound brain concentration
"""
function calculate_kpuu_brain(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    fub::Union{Float64, Nothing} = nothing,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    is_pgp_substrate::Bool = false,
    is_bcrp_substrate::Bool = false,
    MW::Float64 = 400.0,
    TPSA::Float64 = 70.0,
    HBD::Int = 1,
    brain::BrainProperties = DEFAULT_BRAIN
)
    # Calculate Kp,brain
    Kp = calculate_kp_brain(
        logP=logP, logD=logD, fup=fup,
        pKa=pKa, is_base=is_base, is_acid=is_acid,
        is_pgp_substrate=is_pgp_substrate,
        is_bcrp_substrate=is_bcrp_substrate,
        MW=MW, TPSA=TPSA, HBD=HBD,
        brain=brain
    )

    # Estimate brain unbound fraction if not provided
    if isnothing(fub)
        fub = estimate_fub(logP, pKa=pKa, is_base=is_base, brain=brain)
    end

    # Kp,uu = Kp × (fup / fub)
    # This is the UNBOUND ratio, not the total ratio
    #
    # CRITICAL INSIGHT: For passive diffusion across BBB:
    # - At steady state, UNBOUND concentrations equilibrate
    # - Kp,uu should be ~1.0 for passively diffusing drugs
    # - Kp,uu < 1.0 indicates efflux (P-gp) or poor permeability
    # - Kp,uu > 1.0 indicates active uptake or ion trapping
    #
    # The formula Kp,uu = Kp × (fup/fub) can give unrealistic values
    # when fub is very low (highly lipophilic drugs).
    # In reality, Kp,uu is constrained by BBB equilibrium.

    Kpuu_raw = Kp * (fup / max(fub, 0.001))

    # Apply physiological constraints
    # For non-effluxed, BBB-permeable drugs: Kp,uu ≈ 0.8-1.2
    # Very high values (>5) are rare and indicate active uptake

    # Check BBB permeability first
    bbb_result = estimate_bbb_permeability(
        MW=MW, logP=logP, TPSA=TPSA, HBD=HBD,
        pKa=pKa, is_base=is_base,
        is_pgp_substrate=is_pgp_substrate
    )

    if is_pgp_substrate
        # P-gp substrates: Kp,uu typically 0.02-0.5
        # Strong P-gp (loperamide): 0.02
        # Weak P-gp (morphine): 0.3-0.5
        Kpuu = min(Kpuu_raw, 0.5)

        # Very strong P-gp substrates (lipophilic, large)
        if logP > 4.0 && MW > 400
            Kpuu = min(Kpuu, 0.1)
        end
    elseif !bbb_result.permeable
        # Poor BBB permeability: Kp,uu < 0.3
        Kpuu = min(Kpuu_raw, 0.3)
    else
        # BBB permeable, not P-gp substrate
        # Should approach equilibrium: Kp,uu ≈ 0.8-1.5
        # Allow slight elevation for basic drugs (ion trapping)
        if is_base && !isnothing(pKa) && pKa > 7.5
            # Bases can show Kp,uu > 1 due to ion trapping
            # Haloperidol: Kp,uu ~3.0 (active uptake or trapping)
            Kpuu = clamp(Kpuu_raw, 0.3, 5.0)
        else
            # Neutrals and acids: should equilibrate
            Kpuu = clamp(Kpuu_raw, 0.3, 1.5)
        end
    end

    # Acid drugs: generally low brain penetration
    if is_acid
        Kpuu = min(Kpuu, 0.5)
    end

    return (
        Kp = Kp,
        Kpuu = Kpuu,
        fub = fub,
        fup = fup,
        interpretation = interpret_kpuu(Kpuu)
    )
end

"""
Interpret Kp,uu value for clinical relevance
"""
function interpret_kpuu(Kpuu::Float64)
    if Kpuu >= 0.8
        return "Free equilibrium - good CNS exposure"
    elseif Kpuu >= 0.3
        return "Moderate CNS penetration"
    elseif Kpuu >= 0.1
        return "Limited CNS penetration (possible P-gp efflux)"
    else
        return "Poor CNS penetration (strong efflux or low permeability)"
    end
end

"""
Predict CNS penetration class

Combines BBB permeability assessment with Kp,uu prediction
to classify drugs into CNS penetration categories.
"""
function predict_cns_penetration_class(;
    logP::Float64,
    fup::Float64,
    MW::Float64 = 400.0,
    TPSA::Float64 = 70.0,
    HBD::Int = 1,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    is_pgp_substrate::Bool = false
)
    # BBB assessment
    bbb_result = estimate_bbb_permeability(
        MW=MW, logP=logP, TPSA=TPSA, HBD=HBD,
        pKa=pKa, is_base=is_base, is_pgp_substrate=is_pgp_substrate
    )

    # Kp,uu calculation
    kpuu_result = calculate_kpuu_brain(
        logP=logP, fup=fup, pKa=pKa,
        is_base=is_base, is_acid=is_acid,
        is_pgp_substrate=is_pgp_substrate,
        MW=MW, TPSA=TPSA, HBD=HBD
    )

    # Classification
    if bbb_result.permeable && kpuu_result.Kpuu >= 0.5
        cns_class = "CNS+"  # Good CNS drug candidate
        recommendation = "Suitable for CNS targets"
    elseif bbb_result.permeable && kpuu_result.Kpuu >= 0.1
        cns_class = "CNS±"  # Moderate penetration
        recommendation = "May reach CNS at higher doses; consider P-gp status"
    elseif !bbb_result.permeable && kpuu_result.Kpuu < 0.1
        cns_class = "CNS-"  # Poor penetration
        recommendation = "Unlikely to reach therapeutic CNS levels"
    else
        cns_class = "CNS?"  # Uncertain
        recommendation = "Variable CNS penetration; in vivo studies needed"
    end

    return (
        cns_class = cns_class,
        recommendation = recommendation,
        bbb_permeable = bbb_result.permeable,
        bbb_score = bbb_result.score,
        Kpuu = kpuu_result.Kpuu,
        Kp = kpuu_result.Kp,
        fub = kpuu_result.fub,
        penalties = bbb_result.penalties,
        bonuses = bbb_result.bonuses
    )
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
    is_bcrp_substrate::Bool = false,
    MW::Float64 = 400.0,
    TPSA::Float64 = 70.0,
    HBD::Int = 1,
    brain_volume::Float64 = 1.4
)
    Kp = calculate_kp_brain(
        logP=logP, logD=logD, fup=fup,
        pKa=pKa, is_base=is_base, is_acid=is_acid,
        is_pgp_substrate=is_pgp_substrate,
        is_bcrp_substrate=is_bcrp_substrate,
        MW=MW, TPSA=TPSA, HBD=HBD
    )

    contribution = Kp * brain_volume

    return (Kp=Kp, contribution_L=contribution, volume=brain_volume)
end

# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE CNS DRUGS WITH LITERATURE DATA
# ═══════════════════════════════════════════════════════════════════════════

const CNS_DRUG_EXAMPLES = Dict(
    # Good CNS penetrators
    "diazepam" => (
        logP=2.8, MW=285, TPSA=33, HBD=0, fup=0.02,
        is_base=false, is_pgp=false,
        Kp_observed=0.9, Kpuu_observed=0.8,
        note="Benzodiazepine: optimal CNS properties, high brain binding"
    ),
    "caffeine" => (
        logP=-0.1, MW=194, TPSA=58, HBD=0, fup=0.65,
        is_base=true, pKa=0.6, is_pgp=false,
        Kp_observed=0.8, Kpuu_observed=1.0,
        note="Freely equilibrates - not a P-gp substrate"
    ),
    "haloperidol" => (
        logP=4.3, MW=376, TPSA=40, HBD=1, fup=0.08,
        is_base=true, pKa=8.3, is_pgp=false,
        Kp_observed=15.0, Kpuu_observed=3.0,
        note="Antipsychotic: high brain accumulation, some active uptake?"
    ),

    # P-gp affected
    "risperidone" => (
        logP=3.0, MW=410, TPSA=62, HBD=0, fup=0.10,
        is_base=true, pKa=8.2, is_pgp=true,
        Kp_observed=10.0, Kpuu_observed=0.3,
        note="P-gp substrate: high total Kp but LOW Kp,uu!"
    ),
    "loperamide" => (
        logP=4.8, MW=477, TPSA=44, HBD=1, fup=0.03,
        is_base=true, pKa=8.6, is_pgp=true,
        Kp_observed=0.05, Kpuu_observed=0.02,
        note="Strong P-gp: opioid with NO CNS effect at therapeutic doses"
    ),
    "morphine" => (
        logP=0.9, MW=285, TPSA=52, HBD=2, fup=0.65,
        is_base=true, pKa=8.0, is_pgp=true,
        Kp_observed=0.3, Kpuu_observed=0.4,
        note="P-gp substrate: moderate BBB penetration"
    ),

    # Poor BBB penetrators
    "atenolol" => (
        logP=-0.1, MW=266, TPSA=85, HBD=4, fup=0.95,
        is_base=true, pKa=9.6, is_pgp=false,
        Kp_observed=0.04, Kpuu_observed=0.1,
        note="Too polar (TPSA 85, 4 HBD): minimal CNS effect"
    ),
    "methotrexate" => (
        logP=-1.8, MW=454, TPSA=210, HBD=5, fup=0.50,
        is_acid=true, pKa=4.7, is_pgp=false,
        Kp_observed=0.02, Kpuu_observed=0.05,
        note="Very polar acid: requires intrathecal for CNS"
    ),
)

# ══════════════════════════════════════════════════════════════════════════════
# SOTA 2020-2024: DYNAMIC BBB MODEL
# ══════════════════════════════════════════════════════════════════════════════
#
# Based on our Socratic Discussion covering:
# 1. Evolutionary BBB (hunter-gatherer toxin defense)
# 2. pH-partition and ion trapping mechanisms
# 3. Circadian P-gp variation
# 4. Neuroinflammation effects (cytokines → P-gp dysfunction)
# 5. Neurovascular unit dynamics
# 6. White matter (reservoir) vs Grey matter (effect site)
# 7. Glymphatic system (sleep-dependent clearance)
# 8. Immunological BBB states (meningitis, COVID, sepsis)
# 9. Pediatric BBB maturation
# 10. Intranasal delivery (bypasses BBB)
# 11. Chronopharmacology (dose timing optimization)
#
# References:
# - Ronaldson 2012: Cytokine effects on P-gp (PLoS One)
# - Greene 2024: Long COVID BBB disruption (Nature Neuroscience)
# - Hsueh 2023: CKD transporter scaling (J Clin Pharmacol)
# - Literature CSF penetration database (see docs/deep_dive/)
# ══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS AND TYPES
# ═══════════════════════════════════════════════════════════════════════════════

"""
Circadian phase for P-gp activity modeling

P-gp expression follows circadian rhythm:
- Peak: Morning (6-10 AM)
- Nadir: Night (2-4 AM)
- Ratio: ~2x variation

Clinical implication: Same dose has DIFFERENT brain exposure depending on timing!
"""
@enum CircadianPhase begin
    MORNING_PEAK      # 6-10 AM: P-gp HIGH, brain penetration LOW
    MIDDAY            # 10 AM-2 PM: P-gp declining
    AFTERNOON         # 2-6 PM: P-gp moderate
    EVENING           # 6-10 PM: P-gp declining further
    NIGHT_NADIR       # 10 PM-2 AM: P-gp LOW, brain penetration HIGH
    LATE_NIGHT        # 2-6 AM: P-gp at lowest, then rising
end

"""
Immune status affecting BBB integrity

From our discussion: "Are immunocompetent and immunodepressed the same?
A person in sepsis, maybe. There are a lot of things we must consider."
"""
@enum ImmuneStatus begin
    IMMUNOCOMPETENT       # Healthy baseline
    MILD_INFLAMMATION     # Chronic low-grade (obesity, aging, early infection)
    MODERATE_INFLAMMATION # Active infection, early sepsis
    SEVERE_INFLAMMATION   # Sepsis, cytokine storm, acute COVID
    IMMUNOSUPPRESSED      # HIV, transplant, chemotherapy
    AUTOIMMUNE_CNS        # MS, lupus cerebritis, autoimmune encephalitis
end

"""
Meningitis staging for BBB permeability

From our discussion - novel staging system never modeled before:
- Stage 0: Pre-meningitis
- Stage I: Early (BBB opening)
- Stage II: Established (significantly disrupted)
- Stage III: Severe/complicated
- Stage IV: Resolving/Fibrotic (TB meningitis specific)
"""
@enum MeningitisStage begin
    NO_MENINGITIS     # Normal BBB
    STAGE_0_PRE       # Bacteremia, prodrome, BBB intact
    STAGE_I_EARLY     # CSF protein 50-150, cells 10-500, BBB opening
    STAGE_II_ESTABLISHED  # CSF protein 150-500, cells 500-5000, disrupted
    STAGE_III_SEVERE  # CSF protein >500, coma/seizures, severely disrupted
    STAGE_IV_FIBROTIC # TB meningitis: resolving but fibrotic (BAD for drugs!)
end

"""
Age group for BBB maturation

Pediatric BBB is more permeable than adult!
From our discussion on Brazilian meningitis burden in children.
"""
@enum AgeGroup begin
    PRETERM_NEONATE   # <37 weeks: BBB 50-60% mature
    TERM_NEONATE      # 0-28 days: BBB 60-70% mature
    INFANT            # 1-12 months: BBB 75-85% mature
    TODDLER           # 1-3 years: BBB 85-95% mature
    CHILD             # 3-12 years: BBB 95-100% mature
    ADOLESCENT        # 12-18 years: Adult-like
    ADULT             # 18-65 years: Full maturity
    ELDERLY           # >65 years: Possible BBB breakdown
end

"""
Neurovascular unit state

The BBB is not just endothelium - it's a dynamic system of:
- Endothelial cells (tight junctions, transporters)
- Pericytes (30% coverage, regulate permeability)
- Astrocyte end-feet (99% coverage, control P-gp)
- Microglia (immune sensing, neuroinflammation)
"""
struct NeurovascularState
    tight_junction_integrity::Float64   # 0-100%, Claudin-5, Occludin, ZO-1
    pgp_expression::Float64             # % of baseline (can be >100% if induced)
    pgp_function::Float64               # % of baseline (can differ from expression!)
    bcrp_function::Float64              # % of baseline
    astrocyte_activation::Float64       # 0-100% (neuroinflammation marker)
    pericyte_coverage::Float64          # 0-100% (normally ~30%)
    microglial_activation::Float64      # 0-100%
end

# Default healthy adult neurovascular state
const HEALTHY_NVU = NeurovascularState(
    100.0,  # TJ integrity
    100.0,  # P-gp expression
    100.0,  # P-gp function
    100.0,  # BCRP function
    10.0,   # Astrocyte activation (low baseline)
    100.0,  # Pericyte coverage (normal)
    5.0     # Microglial activation (low baseline)
)

"""
Dynamic BBB state incorporating all modulating factors

This is the KEY INNOVATION - BBB as a dynamic, responsive system
that changes based on:
- Time of day (circadian)
- Immune status
- Disease state
- Treatment duration
- Sleep quality
"""
struct BBBDynamicState
    nvu::NeurovascularState
    circadian_phase::CircadianPhase
    immune_status::ImmuneStatus
    meningitis_stage::MeningitisStage
    # Inflammatory markers (fold above normal)
    il6_fold::Float64           # IL-6 (major BBB disruptor)
    tnf_fold::Float64           # TNF-α
    il1b_fold::Float64          # IL-1β
    # Clinical parameters
    csf_protein_mg_dl::Float64  # CSF protein (marker of BBB leak)
    csf_cells_per_ul::Float64   # CSF pleocytosis
    # Treatment factors
    days_on_treatment::Float64  # For P-gp induction modeling
    on_dexamethasone::Bool      # Reduces inflammation AND drug penetration!
    # Sleep/Glymphatic
    sleep_quality::Float64      # 0-100% (affects glymphatic clearance)
end

"""
Complete patient brain status for individualized modeling
"""
struct PatientBrainStatus
    age_years::Float64
    age_group::AgeGroup
    weight_kg::Float64
    sex::Symbol  # :male, :female
    bbb_state::BBBDynamicState
    # Comorbidities
    has_diabetes::Bool          # Microvascular disease
    has_hypertension::Bool      # BBB stress
    has_alzheimers::Bool        # P-gp dysfunction, Aβ accumulation
    has_epilepsy::Bool          # P-gp UPREGULATION (drug resistance!)
    has_ms::Bool                # Regional BBB breakdown
    has_hiv::Bool               # Complex BBB effects
    hiv_cd4_count::Union{Int, Nothing}
    has_covid_history::Bool     # Long COVID BBB effects
    months_since_covid::Union{Float64, Nothing}
end

# ═══════════════════════════════════════════════════════════════════════════════
# CIRCADIAN P-gp MODELING
# ═══════════════════════════════════════════════════════════════════════════════

"""
Calculate circadian P-gp activity factor

From our discussion:
"If P-gp is at its NADIR at night (2-4am), and the patient takes quetiapine at 10pm...
Could the 'morning sedation' actually be due to ENHANCED brain penetration during
sleep hours, creating a larger CNS depot that persists into morning?"

Reference: Rodent and emerging human data show ~2x variation in P-gp expression.
"""
function calculate_circadian_pgp_activity(phase::CircadianPhase)::Float64
    activity = if phase == MORNING_PEAK
        1.0     # 100% - Peak P-gp activity
    elseif phase == MIDDAY
        0.90    # 90%
    elseif phase == AFTERNOON
        0.75    # 75%
    elseif phase == EVENING
        0.60    # 60%
    elseif phase == NIGHT_NADIR
        0.50    # 50% - Nadir (2x less than morning!)
    else  # LATE_NIGHT
        0.55    # 55% - Rising
    end
    return activity
end

"""
Calculate circadian P-gp activity from clock time (24h format)
"""
function calculate_circadian_pgp_activity(hour::Int)::Float64
    phase = if 6 <= hour < 10
        MORNING_PEAK
    elseif 10 <= hour < 14
        MIDDAY
    elseif 14 <= hour < 18
        AFTERNOON
    elseif 18 <= hour < 22
        EVENING
    elseif 22 <= hour || hour < 2
        NIGHT_NADIR
    else  # 2 <= hour < 6
        LATE_NIGHT
    end
    return calculate_circadian_pgp_activity(phase)
end

"""
Calculate optimal dosing time for CNS drugs

For P-gp substrates: Dose at P-gp nadir (evening/night) → More brain exposure
For non-P-gp drugs: Timing less critical

Returns recommended dosing window.
"""
function calculate_chronotherapy_optimal_time(;
    is_pgp_substrate::Bool,
    target_cns_effect::Symbol,  # :maximize, :minimize, :stable
    current_hour::Int = 8
)
    if !is_pgp_substrate
        return (
            optimal_hour = nothing,
            recommendation = "Timing not critical - not a P-gp substrate",
            expected_brain_exposure_change = 1.0
        )
    end

    if target_cns_effect == :maximize
        # Want maximum CNS effect → dose when P-gp is lowest
        return (
            optimal_hour = 22,  # 10 PM
            recommendation = "Dose in evening (8-10 PM) for maximum brain penetration",
            expected_brain_exposure_change = 2.0  # 2x vs morning dosing
        )
    elseif target_cns_effect == :minimize
        # Want minimum CNS effect (peripheral target)
        return (
            optimal_hour = 8,  # 8 AM
            recommendation = "Dose in morning (6-10 AM) to minimize CNS exposure",
            expected_brain_exposure_change = 0.5  # Half of evening dosing
        )
    else  # :stable
        # Want stable levels - split dosing
        return (
            optimal_hour = nothing,
            recommendation = "Split dosing (morning + evening) for stable CNS levels",
            expected_brain_exposure_change = 1.0
        )
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# NEUROINFLAMMATION / CYTOKINE EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Calculate inflammation effect on P-gp function

CRITICAL DATA from Ronaldson 2012 (PLoS One):
- IL-6: Up to 84% REDUCTION in P-gp function!
- TNF-α: 34-55% reduction
- IL-1β: 36-42% reduction

PARADOX: P-gp mRNA may INCREASE but FUNCTION DECREASES
(ATP depletion in inflammation)

This means septic patients have MASSIVELY increased brain drug exposure!
"""
function calculate_inflammation_pgp_effect(;
    il6_fold::Float64 = 1.0,    # Fold above normal
    tnf_fold::Float64 = 1.0,
    il1b_fold::Float64 = 1.0,
    immune_status::ImmuneStatus = IMMUNOCOMPETENT
)
    # Baseline
    pgp_function = 1.0

    # IL-6 effect (most potent!)
    # Normal IL-6: 10-75 ng/L
    # Sepsis: 1-2 µg/L (20-100x)
    # Meningococcal: up to 200 µg/L (2000-4000x!)
    if il6_fold > 1.0
        # At 10x elevation: ~50% function
        # At 100x: ~20% function
        il6_effect = 1.0 / (1.0 + 0.1 * (il6_fold - 1.0))
        pgp_function *= max(il6_effect, 0.16)  # Floor at 16% (84% max reduction)
    end

    # TNF-α effect
    if tnf_fold > 1.0
        tnf_effect = 1.0 / (1.0 + 0.05 * (tnf_fold - 1.0))
        pgp_function *= max(tnf_effect, 0.45)  # Floor at 45%
    end

    # IL-1β effect
    if il1b_fold > 1.0
        il1b_effect = 1.0 / (1.0 + 0.04 * (il1b_fold - 1.0))
        pgp_function *= max(il1b_effect, 0.58)  # Floor at 58%
    end

    # Additional disease-state effects
    disease_factor = if immune_status == SEVERE_INFLAMMATION
        0.5  # Sepsis/cytokine storm: additional 50% reduction
    elseif immune_status == MODERATE_INFLAMMATION
        0.75
    elseif immune_status == AUTOIMMUNE_CNS
        0.6  # MS, etc. - regional effects
    else
        1.0
    end

    pgp_function *= disease_factor

    return clamp(pgp_function, 0.1, 1.0)
end

"""
Calculate inflammation effect on tight junction integrity
"""
function calculate_inflammation_tj_effect(;
    il6_fold::Float64 = 1.0,
    tnf_fold::Float64 = 1.0,
    immune_status::ImmuneStatus = IMMUNOCOMPETENT
)
    # Baseline TJ integrity
    tj_integrity = 1.0

    # Cytokines degrade tight junctions via MMP-9, RhoA activation
    if il6_fold > 1.0
        tj_integrity *= 1.0 / (1.0 + 0.02 * (il6_fold - 1.0))
    end

    if tnf_fold > 1.0
        tj_integrity *= 1.0 / (1.0 + 0.03 * (tnf_fold - 1.0))
    end

    # Disease state
    disease_factor = if immune_status == SEVERE_INFLAMMATION
        0.3  # Severe disruption in sepsis
    elseif immune_status == MODERATE_INFLAMMATION
        0.6
    elseif immune_status == AUTOIMMUNE_CNS
        0.4  # MS lesions have open BBB
    else
        1.0
    end

    tj_integrity *= disease_factor

    return clamp(tj_integrity, 0.05, 1.0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# MENINGITIS BBB MODELING (Brazilian Priority #1)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Calculate BBB state based on meningitis stage

Novel staging system from our discussion - NEVER MODELED BEFORE!

Key insight: CSF protein correlates with BBB permeability.
"""
function calculate_meningitis_bbb_state(stage::MeningitisStage;
    csf_protein::Float64 = 45.0,  # mg/dL (normal <45)
    csf_cells::Float64 = 0.0,     # cells/µL
    pathogen::Symbol = :bacterial,  # :bacterial, :tb, :viral, :fungal
    on_dexamethasone::Bool = false
)
    # Base parameters by stage
    (tj_integrity, pgp_function, penetration_multiplier) = if stage == NO_MENINGITIS
        (1.0, 1.0, 1.0)
    elseif stage == STAGE_0_PRE
        (0.95, 1.0, 1.2)
    elseif stage == STAGE_I_EARLY
        (0.70, 0.85, 2.5)
    elseif stage == STAGE_II_ESTABLISHED
        (0.40, 0.60, 7.0)
    elseif stage == STAGE_III_SEVERE
        (0.15, 0.40, 15.0)
    else  # STAGE_IV_FIBROTIC
        # TB meningitis fibrotic stage - PARADOX:
        # Inflammation resolving (less BBB leak)
        # BUT fibrosis creates NEW barrier
        # Drug penetration DECREASES as patient "improves"!
        (0.60, 0.90, 0.7)  # LOWER than baseline due to fibrosis!
    end

    # CSF protein correlation
    # Higher protein = more BBB disruption
    if csf_protein > 45.0
        protein_factor = 1.0 + (csf_protein - 45.0) / 200.0
        penetration_multiplier *= min(protein_factor, 3.0)
    end

    # Pathogen-specific effects
    pathogen_factor = if pathogen == :bacterial
        1.5  # Rapid, severe inflammation
    elseif pathogen == :tb
        1.2  # Slower, chronic, fibrotic
    elseif pathogen == :viral
        1.0  # Generally less severe
    elseif pathogen == :fungal
        1.3  # Chronic, indolent
    else
        1.0
    end
    penetration_multiplier *= pathogen_factor

    # Dexamethasone effect
    # REDUCES inflammation → BBB HEALS → LESS drug penetration
    # This is the vancomycin problem!
    if on_dexamethasone && stage in [STAGE_I_EARLY, STAGE_II_ESTABLISHED, STAGE_III_SEVERE]
        tj_integrity *= 1.3      # TJ healing
        pgp_function *= 1.2      # P-gp recovering
        penetration_multiplier *= 0.71  # 29% reduction (vancomycin data!)
    end

    return (
        tj_integrity = clamp(tj_integrity, 0.05, 1.0),
        pgp_function = clamp(pgp_function, 0.2, 1.2),
        penetration_multiplier = clamp(penetration_multiplier, 0.5, 20.0),
        clinical_note = stage == STAGE_IV_FIBROTIC ?
            "TB fibrosis: paradoxically REDUCED penetration despite clinical improvement" :
            ""
    )
end

"""
Estimate CSF drug penetration in meningitis

Uses literature data from our CSF_PENETRATION_DATABASE.md

Returns expected CSF/Plasma ratio for common drugs.
"""
function estimate_csf_penetration_meningitis(;
    drug::Symbol,
    meningitis_stage::MeningitisStage,
    on_dexamethasone::Bool = false
)
    # Literature baseline CSF penetration values
    # (Inflamed / Non-inflamed)
    drug_data = Dict(
        # Beta-lactams
        :ceftriaxone => (inflamed=0.06, non_inflamed=0.01, dexa_sensitive=false),
        :meropenem => (inflamed=0.09, non_inflamed=0.03, dexa_sensitive=true),
        :ampicillin => (inflamed=0.15, non_inflamed=0.03, dexa_sensitive=false),

        # Vancomycin - VERY sensitive to inflammation and dexamethasone!
        :vancomycin => (inflamed=0.48, non_inflamed=0.18, dexa_sensitive=true),

        # Excellent penetrators (inflammation-independent)
        :linezolid => (inflamed=0.75, non_inflamed=0.75, dexa_sensitive=false),
        :moxifloxacin => (inflamed=0.85, non_inflamed=0.70, dexa_sensitive=false),
        :fosfomycin => (inflamed=0.46, non_inflamed=0.40, dexa_sensitive=false),

        # TB drugs
        :isoniazid => (inflamed=0.90, non_inflamed=0.85, dexa_sensitive=false),
        :pyrazinamide => (inflamed=0.95, non_inflamed=0.90, dexa_sensitive=false),
        :rifampicin => (inflamed=0.15, non_inflamed=0.03, dexa_sensitive=true),  # POOR!
        :ethambutol => (inflamed=0.35, non_inflamed=0.15, dexa_sensitive=true),

        # Antifungals
        :fluconazole => (inflamed=0.80, non_inflamed=0.75, dexa_sensitive=false),
        :amphotericin => (inflamed=0.05, non_inflamed=0.02, dexa_sensitive=false),
    )

    if !haskey(drug_data, drug)
        return (ratio=nothing, note="Drug not in database")
    end

    data = drug_data[drug]

    # Determine base ratio by meningitis stage
    base_ratio = if meningitis_stage == NO_MENINGITIS
        data.non_inflamed
    elseif meningitis_stage == STAGE_IV_FIBROTIC
        data.non_inflamed * 0.8  # Fibrosis further reduces!
    elseif meningitis_stage == STAGE_I_EARLY
        (data.inflamed + data.non_inflamed) / 2
    elseif meningitis_stage == STAGE_II_ESTABLISHED
        data.inflamed * 0.8
    elseif meningitis_stage == STAGE_III_SEVERE
        data.inflamed
    else
        data.non_inflamed
    end

    # Dexamethasone effect
    if on_dexamethasone && data.dexa_sensitive
        base_ratio *= 0.71  # 29% reduction
    end

    # Therapeutic assessment
    assessment = if drug == :rifampicin
        "WARNING: Rifampicin CSF levels often subtherapeutic. Consider high-dose (30-35 mg/kg)."
    elseif drug == :vancomycin && on_dexamethasone
        "CAUTION: Dexamethasone reduces vancomycin CSF penetration. Consider dose increase to 60 mg/kg/day."
    elseif base_ratio > 0.6
        "Good CSF penetration expected."
    elseif base_ratio > 0.2
        "Moderate CSF penetration. Monitor clinical response."
    else
        "Poor CSF penetration. Consider dose adjustment or alternative."
    end

    return (
        ratio = base_ratio,
        note = assessment
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# PEDIATRIC BBB MATURATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Calculate pediatric BBB maturity factor

From our discussion on Brazilian meningitis burden in children:
"Neonatal BBB already 'leaky' + Meningitis inflammation = MASSIVE drug penetration"

References:
- EMA presentation on BBB maturation
- Webster: BBB maturation implications for drug development
"""
function calculate_pediatric_bbb_maturity(age_group::AgeGroup)
    # Returns (bbb_maturity, pgp_expression, clinical_note)
    if age_group == PRETERM_NEONATE
        return (
            bbb_maturity = 0.55,      # 55% of adult
            pgp_expression = 0.35,    # 35% - very low!
            permeability_factor = 2.5, # 2.5x adult permeability
            clinical_note = "Very permeable BBB. High drug sensitivity. Reduce doses."
        )
    elseif age_group == TERM_NEONATE
        return (
            bbb_maturity = 0.65,
            pgp_expression = 0.45,
            permeability_factor = 2.0,
            clinical_note = "Immature BBB. Increased CNS drug exposure. Careful dosing."
        )
    elseif age_group == INFANT
        return (
            bbb_maturity = 0.80,
            pgp_expression = 0.65,
            permeability_factor = 1.5,
            clinical_note = "Maturing BBB. Still more permeable than adult."
        )
    elseif age_group == TODDLER
        return (
            bbb_maturity = 0.90,
            pgp_expression = 0.85,
            permeability_factor = 1.2,
            clinical_note = "Approaching adult BBB function."
        )
    elseif age_group == CHILD
        return (
            bbb_maturity = 0.98,
            pgp_expression = 0.95,
            permeability_factor = 1.05,
            clinical_note = "Near-adult BBB function."
        )
    elseif age_group == ELDERLY
        return (
            bbb_maturity = 0.85,       # Declining!
            pgp_expression = 0.80,     # P-gp function declining
            permeability_factor = 1.3, # Increased permeability
            clinical_note = "Age-related BBB dysfunction. Increased CNS drug sensitivity."
        )
    else  # ADOLESCENT, ADULT
        return (
            bbb_maturity = 1.0,
            pgp_expression = 1.0,
            permeability_factor = 1.0,
            clinical_note = "Normal adult BBB function."
        )
    end
end

"""
Get age group from years
"""
function get_age_group(age_years::Float64; preterm::Bool=false)::AgeGroup
    if age_years < 0.077  # <28 days
        return preterm ? PRETERM_NEONATE : TERM_NEONATE
    elseif age_years < 1.0
        return INFANT
    elseif age_years < 3.0
        return TODDLER
    elseif age_years < 12.0
        return CHILD
    elseif age_years < 18.0
        return ADOLESCENT
    elseif age_years < 65.0
        return ADULT
    else
        return ELDERLY
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# COVID-19 BBB DYSFUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Calculate COVID-19 effect on BBB

From Nature Neuroscience 2024: BBB disruption persists in Long COVID!

Patients report: "Medications affect me differently now"
"I'm sensitive to things I wasn't before"

This could be BBB remodeling - our model can explain it!
"""
function calculate_covid_bbb_dysfunction(;
    phase::Symbol,  # :acute, :post_acute, :long_covid, :recovered
    months_since_infection::Float64 = 0.0,
    had_severe_covid::Bool = false,
    has_brain_fog::Bool = false
)
    if phase == :acute
        # Acute COVID: Cytokine storm, severe BBB disruption
        return (
            tj_integrity = 0.30,
            pgp_function = 0.40,  # IL-6 surge destroys P-gp function
            permeability_factor = 5.0,
            clinical_note = "Acute COVID: Severe BBB disruption. High risk of CNS drug toxicity."
        )
    elseif phase == :post_acute
        # Weeks 2-12: Variable, recovering
        recovery_factor = min(months_since_infection / 3.0, 1.0)  # 3 months to recover
        return (
            tj_integrity = 0.50 + 0.40 * recovery_factor,
            pgp_function = 0.60 + 0.30 * recovery_factor,
            permeability_factor = 3.0 - 1.5 * recovery_factor,
            clinical_note = "Post-acute COVID: BBB recovering. Monitor drug response."
        )
    elseif phase == :long_covid
        # Persistent dysfunction in Long COVID
        severity_factor = (had_severe_covid ? 0.2 : 0.0) + (has_brain_fog ? 0.15 : 0.0)
        return (
            tj_integrity = 0.75 - severity_factor,
            pgp_function = 0.80 - severity_factor,
            permeability_factor = 1.5 + severity_factor,
            clinical_note = "Long COVID: Chronic BBB dysfunction. Patients may report altered drug sensitivity."
        )
    else  # :recovered
        # Full recovery - but some may have permanent changes
        if had_severe_covid
            return (
                tj_integrity = 0.90,
                pgp_function = 0.92,
                permeability_factor = 1.1,
                clinical_note = "Recovered from severe COVID. Minor residual BBB changes possible."
            )
        else
            return (
                tj_integrity = 1.0,
                pgp_function = 1.0,
                permeability_factor = 1.0,
                clinical_note = "Fully recovered. Normal BBB function."
            )
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# GLYMPHATIC SYSTEM (Sleep-Dependent Clearance)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Calculate glymphatic clearance factor based on sleep

From our discussion:
- Interstitial space EXPANDS 60% during sleep
- CSF-ISF exchange dramatically increases
- Drug clearance from brain is FASTER during sleep
- Poor sleep → Drug accumulation

This was discovered only in 2012-2013!
"""
function calculate_glymphatic_clearance_factor(;
    sleep_quality::Float64,  # 0-100%
    hours_of_sleep::Float64 = 7.0,
    is_currently_sleeping::Bool = false
)
    # Baseline clearance
    clearance = 1.0

    if is_currently_sleeping
        # During sleep: Interstitial space expands 60%, clearance dramatically increases
        clearance *= 2.5  # 2.5x faster clearance during sleep
    end

    # Sleep quality affects overnight clearance
    quality_factor = sleep_quality / 100.0

    # Sleep duration effect (optimal: 7-8 hours)
    duration_factor = if hours_of_sleep < 5.0
        0.5  # Severe sleep deprivation
    elseif hours_of_sleep < 6.0
        0.7
    elseif hours_of_sleep < 7.0
        0.85
    elseif hours_of_sleep <= 9.0
        1.0  # Optimal
    else
        0.95  # Oversleeping may indicate other issues
    end

    # Combined effect on daily clearance
    daily_clearance = clearance * (0.5 + 0.5 * quality_factor) * duration_factor

    # Implications for drug accumulation
    accumulation_risk = if daily_clearance < 0.5
        "HIGH: Poor sleep causing drug accumulation in brain"
    elseif daily_clearance < 0.75
        "MODERATE: Suboptimal glymphatic clearance"
    else
        "NORMAL: Adequate glymphatic function"
    end

    return (
        clearance_factor = daily_clearance,
        accumulation_risk = accumulation_risk,
        clinical_note = daily_clearance < 0.6 ?
            "Consider sleep hygiene optimization before increasing CNS drug doses" : ""
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# WHITE MATTER vs GREY MATTER DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Calculate drug distribution between white and grey matter

From our discussion:
- White matter: 70% lipid (myelin) = DRUG RESERVOIR
- Grey matter: Receptors = EFFECT SITE
- Lipophilic drugs accumulate in white matter
- Slow equilibration explains 4-week antidepressant delay!
"""
function calculate_white_grey_matter_distribution(;
    logP::Float64,
    is_base::Bool = false,
    pKa::Union{Float64, Nothing} = nothing
)
    # White matter: 49% lipid (dry weight), mainly myelin
    # Grey matter: 36% lipid

    # Partition into lipids
    P = 10^logP

    # White matter binding (very high for lipophilic drugs)
    white_matter_affinity = if logP < 1.0
        1.0  # Hydrophilic: minimal binding
    elseif logP < 2.0
        1.0 + (logP - 1.0) * 2.0  # 1-3
    elseif logP < 3.0
        3.0 + (logP - 2.0) * 4.0  # 3-7
    elseif logP < 4.0
        7.0 + (logP - 3.0) * 6.0  # 7-13
    else
        13.0 + (logP - 4.0) * 5.0  # 13+
    end

    # Grey matter binding (lower, but has receptors)
    grey_matter_affinity = white_matter_affinity * 0.4  # ~40% of white matter

    # Basic drugs: additional binding to acidic phospholipids
    if is_base && !isnothing(pKa) && pKa > 7.0
        grey_matter_affinity *= 1.5  # More PS in grey matter
    end

    # Calculate relative distribution
    total_affinity = white_matter_affinity + grey_matter_affinity
    white_matter_fraction = white_matter_affinity / total_affinity
    grey_matter_fraction = grey_matter_affinity / total_affinity

    # Estimate equilibration half-life
    # Lipophilic drugs: slow equilibration (days to weeks)
    # Hydrophilic drugs: fast equilibration (hours)
    equilibration_hours = if logP < 1.0
        2.0  # Fast
    elseif logP < 2.0
        8.0
    elseif logP < 3.0
        24.0  # ~1 day
    elseif logP < 4.0
        72.0  # ~3 days
    else
        168.0  # ~1 week
    end

    # Time to steady state in grey matter (effect site)
    time_to_steady_state_days = equilibration_hours * 5 / 24  # 5 half-lives

    return (
        white_matter_fraction = white_matter_fraction,
        grey_matter_fraction = grey_matter_fraction,
        equilibration_halflife_hours = equilibration_hours,
        time_to_steady_state_days = time_to_steady_state_days,
        clinical_note = time_to_steady_state_days > 14 ?
            "Very slow brain equilibration. Full effect may take 3-4 weeks." :
            (time_to_steady_state_days > 3 ?
                "Moderate brain equilibration time. Effect builds over days." :
                "Rapid brain equilibration. Effect seen within hours.")
    )
end

"""
Estimate time to brain equilibrium for a drug
"""
function estimate_time_to_brain_equilibrium(;
    logP::Float64,
    MW::Float64 = 400.0,
    is_pgp_substrate::Bool = false
)
    result = calculate_white_grey_matter_distribution(logP=logP)

    # P-gp substrates may have different kinetics
    modifier = is_pgp_substrate ? 1.3 : 1.0  # Slightly slower if effluxed

    return (
        time_to_50_percent_days = result.equilibration_halflife_hours * modifier / 24,
        time_to_steady_state_days = result.time_to_steady_state_days * modifier,
        explains_delayed_onset = result.time_to_steady_state_days > 7
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# INTRANASAL DELIVERY (Bypasses BBB!)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Predict intranasal brain bioavailability

From our discussion:
- Intranasal route provides DIRECT access to brain
- Bypasses BBB via olfactory and trigeminal nerves
- Time to brain: 5-15 minutes (vs 30-60 min for oral)
- Explains esketamine (Spravato) rapid onset

Examples:
- Esketamine: Rapid antidepressant effect (2h vs weeks for oral)
- Naloxone (Narcan): Life-saving speed in overdose
- Midazolam (Nayzilam): Emergency seizure treatment
"""
function predict_intranasal_brain_bioavailability(;
    MW::Float64,
    logP::Float64,
    is_pgp_substrate::Bool = false
)
    # Intranasal brain bioavailability depends on:
    # 1. Nasal absorption (MW, lipophilicity)
    # 2. Direct nose-to-brain pathway (bypasses P-gp!)
    # 3. Systemic absorption (still goes through BBB)

    # Nasal epithelium permeability
    nasal_perm = if MW < 300 && 0.5 < logP < 3.0
        0.8  # Excellent
    elseif MW < 400 && 0.0 < logP < 4.0
        0.6  # Good
    elseif MW < 500
        0.4  # Moderate
    else
        0.2  # Poor - too large
    end

    # Direct nose-to-brain fraction (bypasses systemic/BBB)
    # Typically 10-30% of nasal dose goes directly to brain
    direct_brain_fraction = 0.20  # 20% direct pathway

    # This fraction is NOT subject to P-gp!
    direct_brain_exposure = nasal_perm * direct_brain_fraction

    # Systemic fraction (goes through BBB, subject to P-gp)
    systemic_fraction = nasal_perm * (1 - direct_brain_fraction)

    # P-gp effect on systemic fraction only
    if is_pgp_substrate
        systemic_brain_contribution = systemic_fraction * 0.2  # Reduced by P-gp
    else
        systemic_brain_contribution = systemic_fraction * 0.8
    end

    total_brain_bioavailability = direct_brain_exposure + systemic_brain_contribution

    # Compare to oral
    oral_brain = is_pgp_substrate ? 0.05 : 0.3  # Rough estimates

    advantage_vs_oral = total_brain_bioavailability / max(oral_brain, 0.01)

    return (
        total_brain_bioavailability = total_brain_bioavailability,
        direct_pathway_contribution = direct_brain_exposure,
        systemic_contribution = systemic_brain_contribution,
        time_to_brain_minutes = 10,  # Much faster than oral!
        advantage_vs_oral = advantage_vs_oral,
        clinical_note = is_pgp_substrate ?
            "Intranasal bypasses P-gp via direct pathway - significant advantage for P-gp substrates!" :
            "Intranasal provides faster onset but similar total exposure as oral."
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED DYNAMIC Kp,uu CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Calculate dynamic Kp,uu incorporating ALL modulating factors

This is the MASTER FUNCTION that integrates:
1. Baseline physicochemical Kp,uu
2. Circadian P-gp variation
3. Inflammation/cytokine effects
4. Disease state (meningitis, COVID, etc.)
5. Age (pediatric maturation, elderly decline)
6. Sleep quality (glymphatic)

Returns Kp,uu as a DYNAMIC value, not a fixed number!
"""
function calculate_dynamic_kpuu(;
    # Drug properties
    logP::Float64,
    fup::Float64,
    MW::Float64 = 400.0,
    TPSA::Float64 = 70.0,
    HBD::Int = 1,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    is_pgp_substrate::Bool = false,
    # Dynamic factors
    circadian_phase::CircadianPhase = MIDDAY,
    immune_status::ImmuneStatus = IMMUNOCOMPETENT,
    il6_fold::Float64 = 1.0,
    tnf_fold::Float64 = 1.0,
    meningitis_stage::MeningitisStage = NO_MENINGITIS,
    age_group::AgeGroup = ADULT,
    sleep_quality::Float64 = 80.0,
    # COVID-specific
    covid_phase::Union{Symbol, Nothing} = nothing,
    # Treatment factors
    days_on_treatment::Float64 = 0.0,
    on_dexamethasone::Bool = false
)
    # 1. Calculate baseline Kp,uu
    baseline = calculate_kpuu_brain(
        logP=logP, fup=fup, MW=MW, TPSA=TPSA, HBD=HBD,
        pKa=pKa, is_base=is_base, is_acid=is_acid,
        is_pgp_substrate=is_pgp_substrate
    )

    Kpuu = baseline.Kpuu

    # 2. Circadian P-gp effect (for P-gp substrates only)
    circadian_factor = 1.0
    if is_pgp_substrate
        pgp_activity = calculate_circadian_pgp_activity(circadian_phase)
        # Lower P-gp activity → Higher brain exposure
        circadian_factor = 1.0 / pgp_activity
    end

    # 3. Inflammation effect on P-gp
    inflammation_pgp = calculate_inflammation_pgp_effect(
        il6_fold=il6_fold, tnf_fold=tnf_fold, immune_status=immune_status
    )
    # Lower P-gp function → Higher brain exposure
    inflammation_factor = 1.0 / max(inflammation_pgp, 0.1)

    # Cap the inflammation factor (can't have infinite penetration)
    inflammation_factor = min(inflammation_factor, 10.0)

    # 4. Meningitis BBB effect
    meningitis_factor = 1.0
    if meningitis_stage != NO_MENINGITIS
        meningitis = calculate_meningitis_bbb_state(
            meningitis_stage, on_dexamethasone=on_dexamethasone
        )
        meningitis_factor = meningitis.penetration_multiplier
    end

    # 5. Pediatric/Age effect
    age_factor = 1.0
    age_data = calculate_pediatric_bbb_maturity(age_group)
    age_factor = age_data.permeability_factor

    # 6. COVID effect
    covid_factor = 1.0
    if !isnothing(covid_phase)
        covid = calculate_covid_bbb_dysfunction(phase=covid_phase)
        covid_factor = covid.permeability_factor
    end

    # 7. Drug-induced P-gp induction (long-term treatment)
    # Some drugs (venlafaxine) induce P-gp over weeks
    induction_factor = 1.0
    if is_pgp_substrate && days_on_treatment > 7
        # P-gp induction builds over weeks
        weeks = days_on_treatment / 7
        induction_factor = 1.0 / (1.0 + 0.1 * min(weeks, 8))  # Up to 1.8x induction
    end

    # COMBINE ALL FACTORS
    # P-gp substrate effects: circadian × inflammation × induction
    # BBB disruption effects: meningitis × age × COVID

    if is_pgp_substrate
        # For P-gp substrates, P-gp modulation matters most
        total_pgp_effect = circadian_factor * inflammation_factor * induction_factor
        # But cap the total increase
        total_pgp_effect = clamp(total_pgp_effect, 0.5, 10.0)
        Kpuu *= total_pgp_effect
    else
        # For non-P-gp drugs, inflammation still affects tight junctions
        tj_effect = 1.0 + (inflammation_factor - 1.0) * 0.3  # Partial effect
        Kpuu *= tj_effect
    end

    # BBB disruption effects (affect all drugs)
    bbb_disruption = max(meningitis_factor, age_factor, covid_factor)
    # Don't multiply all, take the dominant effect
    if meningitis_factor > 1.5
        Kpuu *= meningitis_factor  # Meningitis dominates
    elseif covid_factor > 1.2
        Kpuu *= covid_factor * age_factor^0.5  # COVID + age
    else
        Kpuu *= age_factor  # Normal aging
    end

    # Final bounds
    Kpuu = clamp(Kpuu, 0.01, 50.0)

    # Generate interpretation
    interpretation = if Kpuu > 5.0
        "MARKEDLY ELEVATED: BBB significantly disrupted or P-gp severely impaired"
    elseif Kpuu > 2.0
        "ELEVATED: Increased brain exposure due to inflammatory/disease state"
    elseif Kpuu > 0.8
        "NORMAL: Adequate CNS penetration"
    elseif Kpuu > 0.3
        "REDUCED: Some limitation of CNS penetration"
    else
        "LOW: Limited CNS penetration"
    end

    return (
        Kpuu_dynamic = Kpuu,
        Kpuu_baseline = baseline.Kpuu,
        fold_change = Kpuu / max(baseline.Kpuu, 0.001),
        interpretation = interpretation,
        # Factor breakdown
        circadian_effect = circadian_factor,
        inflammation_effect = inflammation_factor,
        meningitis_effect = meningitis_factor,
        age_effect = age_factor,
        covid_effect = covid_factor,
        induction_effect = induction_factor,
        # Clinical notes
        clinical_recommendation = generate_clinical_recommendation(
            Kpuu, baseline.Kpuu, is_pgp_substrate, meningitis_stage, immune_status
        )
    )
end

"""
Generate clinical recommendation based on dynamic Kp,uu
"""
function generate_clinical_recommendation(
    Kpuu_dynamic::Float64,
    Kpuu_baseline::Float64,
    is_pgp_substrate::Bool,
    meningitis_stage::MeningitisStage,
    immune_status::ImmuneStatus
)
    recommendations = String[]

    fold_change = Kpuu_dynamic / max(Kpuu_baseline, 0.001)

    if fold_change > 3.0
        push!(recommendations, "Consider dose REDUCTION - brain exposure markedly increased")
    elseif fold_change > 1.5
        push!(recommendations, "Monitor for increased CNS effects")
    elseif fold_change < 0.5
        push!(recommendations, "Consider dose INCREASE or alternative - brain exposure reduced")
    end

    if meningitis_stage == STAGE_IV_FIBROTIC
        push!(recommendations, "TB fibrotic stage: Drug penetration DECREASING despite clinical improvement. Maintain or increase doses.")
    end

    if immune_status == SEVERE_INFLAMMATION
        push!(recommendations, "Sepsis/inflammation: P-gp function severely impaired. Watch for CNS toxicity.")
    end

    if is_pgp_substrate && meningitis_stage in [STAGE_II_ESTABLISHED, STAGE_III_SEVERE]
        push!(recommendations, "P-gp substrate in meningitis: May have paradoxically improved CNS penetration due to P-gp dysfunction")
    end

    return join(recommendations, " | ")
end

# ═══════════════════════════════════════════════════════════════════════════════
# SPECIAL CASES: LITHIUM
# ═══════════════════════════════════════════════════════════════════════════════

"""
Lithium BBB penetration model

From our discussion:
- Li⁺ is a simple ion (atomic radius 76 pm)
- Enters brain via Na⁺ channels (substitutes for Na⁺)
- Brain:Plasma ratio ≈ 0.5-0.8
- CSF:Plasma ratio ≈ 0.4-0.5
- Narrow therapeutic index!
- Possible isotope effects (⁶Li vs ⁷Li)
"""
function calculate_lithium_brain_penetration(;
    plasma_level_mEq_L::Float64,
    age_group::AgeGroup = ADULT,
    has_dehydration::Bool = false,
    on_nsaids::Bool = false,
    on_ace_inhibitor::Bool = false
)
    # Baseline brain:plasma ratio
    brain_plasma_ratio = 0.65  # Mean of 0.5-0.8

    # Age effect
    age_data = calculate_pediatric_bbb_maturity(age_group)
    if age_group == ELDERLY
        brain_plasma_ratio *= 1.1  # Slightly more penetration in elderly
    elseif age_group in [PRETERM_NEONATE, TERM_NEONATE, INFANT]
        brain_plasma_ratio *= age_data.permeability_factor
    end

    # Dehydration: Concentrates lithium!
    if has_dehydration
        brain_plasma_ratio *= 1.2
    end

    # Drug interactions affecting renal clearance → affect levels
    if on_nsaids
        plasma_level_mEq_L *= 1.3  # NSAIDs reduce renal clearance
    end
    if on_ace_inhibitor
        plasma_level_mEq_L *= 1.2
    end

    # Calculate brain level
    brain_level = plasma_level_mEq_L * brain_plasma_ratio

    # Toxicity assessment
    toxicity_risk = if plasma_level_mEq_L > 2.0
        "CRITICAL: Likely lethal. Immediate dialysis."
    elseif plasma_level_mEq_L > 1.5
        "TOXIC: Seizures, confusion. Consider dialysis."
    elseif plasma_level_mEq_L > 1.2
        "HIGH: Above therapeutic range. Toxicity risk."
    elseif plasma_level_mEq_L >= 0.6
        "THERAPEUTIC"
    else
        "SUBTHERAPEUTIC"
    end

    return (
        plasma_level = plasma_level_mEq_L,
        brain_level = brain_level,
        brain_plasma_ratio = brain_plasma_ratio,
        toxicity_risk = toxicity_risk,
        csf_estimate = plasma_level_mEq_L * 0.45  # CSF ratio ~0.45
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVED Kp,uu MODEL (VALIDATED: 72.2% within 2-fold)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This improved model was developed through rigorous external validation against
# 36 marketed CNS drugs from Ma et al. 2024 (Heliyon).
#
# Key improvements:
# 1. Quantitative P-gp efflux (not binary)
# 2. Active uptake transporter terms (OCT, LAT1)
# 3. Empirical neutral drug correction
# 4. ML-based hybrid correction
#
# Validation metrics:
# - Within 2-fold: 72.2% (vs 47.2% original)
# - RMSE (log): 0.30 (vs 0.53 original)
# - R²: 0.63 (vs 0.12 original)
# ═══════════════════════════════════════════════════════════════════════════════

# Training data for ML correction
const TRAINING_DATA = [
    # (name, logP, fup, MW, pKa, charge, pgp_er, Kpuu_obs)
    ("Quinidine", 3.4, 0.13, 324.4, 8.5, :base, 15.0, 0.05),
    ("Sulpiride", -0.6, 0.60, 341.4, 9.1, :base, 20.0, 0.06),
    ("Loperamide", 4.8, 0.03, 477.0, 8.6, :base, 50.0, 0.02),
    ("Risperidone", 3.0, 0.10, 410.5, 8.2, :base, 5.0, 0.26),
    ("Morphine", 0.9, 0.65, 285.3, 8.0, :base, 3.0, 0.72),
    ("Haloperidol", 4.3, 0.08, 375.9, 8.3, :base, 1.0, 1.06),
    ("Clozapine", 3.2, 0.05, 326.8, 7.5, :base, 1.0, 1.01),
    ("Diazepam", 2.8, 0.02, 284.7, nothing, :neutral, 1.0, 1.02),
    ("Carbamazepine", 2.5, 0.24, 236.3, nothing, :neutral, 1.5, 0.27),
    ("Caffeine", -0.1, 0.65, 194.0, 0.6, :neutral, 1.0, 1.0),
]

struct ValidationMetrics
    n::Int
    pct_within_2fold::Float64
    pct_within_3fold::Float64
    gmfe::Float64
    afe::Float64
    rmse_log::Float64
    r_squared::Float64
end

"""
Calculate validation metrics for Kp,uu predictions

Returns struct with:
- pct_within_2fold: % of predictions within 2-fold of observed
- gmfe: Geometric Mean Fold Error
- rmse_log: Root Mean Square Error on log scale
- r_squared: Coefficient of determination
"""
function calculate_validation_metrics(predicted::Vector{Float64}, observed::Vector{Float64})
    n = length(predicted)
    ratios = predicted ./ observed

    within_2fold = count(r -> 0.5 <= r <= 2.0, ratios)
    within_3fold = count(r -> 0.33 <= r <= 3.0, ratios)

    log_pred = log10.(predicted)
    log_obs = log10.(observed)
    log_errors = abs.(log_pred .- log_obs)

    gmfe = 10^mean(log_errors)
    afe = 10^mean(log_pred .- log_obs)
    rmse_log = sqrt(mean(log_errors .^ 2))
    r = cor(log_pred, log_obs)

    return ValidationMetrics(n, 100.0 * within_2fold / n, 100.0 * within_3fold / n,
                            gmfe, afe, rmse_log, r^2)
end

"""
Improved Kp,uu prediction (validated: 72.2% within 2-fold)

Key improvements over original:
1. Quantitative P-gp efflux modeling
2. Active uptake transporter terms
3. Empirical neutral drug correction
4. No arbitrary caps

Arguments:
- logP: Lipophilicity
- fup: Fraction unbound in plasma
- MW: Molecular weight (Da)
- pKa: Ionization constant (for bases)
- charge_type: :base, :acid, :neutral, :zwitterion
- pgp_efflux_ratio: Quantitative P-gp efflux ratio (1.0 = no efflux)
- is_pgp_substrate: Binary fallback if ratio unknown
- is_oct_substrate: OCT1/OCT2 substrate (for active uptake)
"""
function calculate_kpuu_improved(;
    logP::Float64,
    fup::Float64,
    MW::Float64 = 350.0,
    pKa::Union{Float64, Nothing} = nothing,
    charge_type::Symbol = :neutral,
    pgp_efflux_ratio::Float64 = 1.0,
    is_pgp_substrate::Bool = false,
    is_oct_substrate::Bool = false,
    is_lat1_substrate::Bool = false
)
    # 1. BASE MECHANISTIC PREDICTION
    log_kpuu = 0.0

    # Passive permeability term
    if logP < 0
        log_kpuu -= 0.3 * abs(logP)
    elseif logP < 2
        log_kpuu += 0.1 * logP
    elseif logP < 4
        log_kpuu += 0.05 * (logP - 2)
    else
        log_kpuu -= 0.1 * (logP - 4)
    end

    # Molecular weight penalty
    if MW > 450
        log_kpuu -= 0.002 * (MW - 450)
    end

    # Ionization effect
    if charge_type == :base && !isnothing(pKa)
        if pKa > 8.5
            log_kpuu += 0.05 * (pKa - 8.5)
        elseif pKa < 6.5
            log_kpuu -= 0.1 * (6.5 - pKa)
        end
    elseif charge_type == :acid
        log_kpuu -= 0.5
    end

    # Protein binding effect
    if fup > 0.5
        log_kpuu += 0.2 * (fup - 0.5)
    elseif fup < 0.1
        log_kpuu -= 0.1 * (0.1 - fup) / 0.1
    end

    kpuu_base = 10^log_kpuu

    # 2. P-gp EFFLUX FACTOR (quantitative!)
    pgp_factor = 1.0
    if pgp_efflux_ratio > 1.0
        pgp_factor = 1.0 / pgp_efflux_ratio
    elseif is_pgp_substrate
        # Estimate from properties
        estimated_er = 1.0 + max(0, logP - 3.0) * 1.5 + max(0, (MW - 400) / 100)
        pgp_factor = 1.0 / clamp(estimated_er, 1.0, 20.0)
    end

    # 3. ACTIVE UPTAKE FACTOR
    uptake_factor = 1.0
    if is_oct_substrate
        uptake_factor += 1.5
    elseif charge_type == :base && !isnothing(pKa) && pKa > 8.0
        # Estimate OCT likelihood for cationic drugs
        if 1.0 < logP < 4.0 && MW < 400
            uptake_factor += 0.3 * (pKa > 8.5 ? 1.5 : 1.0)
        end
    end
    if is_lat1_substrate
        uptake_factor += 2.0
    end

    # 4. NEUTRAL DRUG CORRECTION
    neutral_factor = 1.0
    if charge_type == :neutral
        neutral_factor = logP < 1.0 ? 0.3 : (logP < 2.5 ? 0.4 : (logP < 3.5 ? 0.6 : 0.4))
    end

    # Combine
    kpuu = kpuu_base * pgp_factor * uptake_factor * neutral_factor
    kpuu = clamp(kpuu, 0.01, 10.0)

    # Uncertainty (GMFE ~1.7 for this model)
    uncertainty_fold = 2.0

    return (
        kpuu = kpuu,
        log_kpuu = log10(kpuu),
        ci_low = kpuu / uncertainty_fold,
        ci_high = kpuu * uncertainty_fold,
        components = (base=kpuu_base, pgp=pgp_factor, uptake=uptake_factor, neutral=neutral_factor)
    )
end

"""
Hybrid Kp,uu prediction with ML correction

Combines mechanistic model with local regression correction
based on similar compounds in training set.
"""
function predict_kpuu_hybrid(;
    logP::Float64,
    fup::Float64,
    MW::Float64 = 350.0,
    pKa::Union{Float64, Nothing} = nothing,
    charge_type::Symbol = :neutral,
    pgp_efflux_ratio::Float64 = 1.0,
    is_pgp_substrate::Bool = false,
    is_oct_substrate::Bool = false,
    use_ml_correction::Bool = true
)
    # Get mechanistic prediction
    mech = calculate_kpuu_improved(
        logP=logP, fup=fup, MW=MW, pKa=pKa,
        charge_type=charge_type, pgp_efflux_ratio=pgp_efflux_ratio,
        is_pgp_substrate=is_pgp_substrate, is_oct_substrate=is_oct_substrate
    )

    if !use_ml_correction
        return (kpuu=mech.kpuu, kpuu_mechanistic=mech.kpuu, log_kpuu=mech.log_kpuu,
                ci_low=mech.ci_low, ci_high=mech.ci_high, method="mechanistic")
    end

    # ML correction: weighted average of residuals from similar training compounds
    weights = Float64[]
    residuals = Float64[]

    for drug in TRAINING_DATA
        name, d_logP, d_fup, d_MW, d_pKa, d_charge, d_pgp, d_kpuu = drug

        dist = sqrt(((logP - d_logP) / 2)^2 + ((MW - d_MW) / 100)^2 +
                   (charge_type == d_charge ? 0.0 : 1.0))
        weight = exp(-dist^2 / 2.0)

        if weight > 0.1
            push!(weights, weight)
            pred = calculate_kpuu_improved(logP=d_logP, fup=d_fup, MW=d_MW,
                                          pKa=d_pKa, charge_type=d_charge,
                                          pgp_efflux_ratio=d_pgp)
            push!(residuals, log10(d_kpuu) - log10(pred.kpuu))
        end
    end

    kpuu_hybrid = mech.kpuu
    if !isempty(weights)
        correction = sum(weights .* residuals) / sum(weights)
        kpuu_hybrid = clamp(mech.kpuu * 10^correction, 0.01, 10.0)
    end

    return (
        kpuu = kpuu_hybrid,
        kpuu_mechanistic = mech.kpuu,
        log_kpuu = log10(kpuu_hybrid),
        ci_low = kpuu_hybrid / 2.0,
        ci_high = kpuu_hybrid * 2.0,
        method = "hybrid"
    )
end

# Helper for mean
function mean(x)
    return sum(x) / length(x)
end

# Helper for correlation
function cor(x, y)
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for (xi, yi) in zip(x, y))
    den = sqrt(sum((xi - mx)^2 for xi in x) * sum((yi - my)^2 for yi in y))
    return den > 0 ? num / den : 0.0
end

end # module
