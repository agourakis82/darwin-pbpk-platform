# ===========================================================================
# MEDLANG RENAL ELIMINATION MODEL
# ===========================================================================
# Mechanistic model of renal drug elimination with:
#
# PROCESSES:
# 1. Glomerular filtration (GFR × fu,p)
# 2. Tubular secretion (OAT1/OAT3, OCT2, MATE1/MATE2-K)
# 3. Tubular reabsorption (pH-dependent, passive)
# 4. CKD adaptation (tubular flow rate increase per nephron)
# 5. Fanconi/mTORC1 dysfunction (transporter expression scaling)
#
# COMPARTMENTS:
# - Plasma
# - Glomerular filtrate
# - Proximal tubule (S1, S2, S3 segments)
# - Loop of Henle
# - Distal tubule
# - Collecting duct
# - Urine
#
# TRANSPORTERS (with membrane localization):
# Basolateral (blood side):
#   - OAT1, OAT3: organic anion uptake
#   - OCT2: organic cation uptake
# Apical (urine side):
#   - MATE1, MATE2-K: organic cation efflux
#   - MRP2, MRP4: organic anion efflux
#   - OAT4: organic anion reabsorption
#
# DISEASE STATES:
# - CKD stages 1-5 (GFR-based with tubular adaptation)
# - Fanconi syndrome (mTORC1-mediated transporter dysfunction)
# - Cystinosis (CTNS mutation → lysosomal cystine → mTORC1 hyperactivation)
#
# References:
# - Granda et al. 2024 (Clin Transl Sci) - Biomarker-informed kidney PBPK
# - PMC7577018 - Mechanistic PBPK for CKD with tubular adaptation
# - PMC7250368 - Urine pH effect on drug disposition
# - Nature Reviews Nephrology 2016 - Cystinosis pathogenesis
# - PMID 37621073 - CTNS-mTORC1 axis in cystinosis (2023)
#
# Author: Dr. Sounio Agourakis
# Date: November 2025
# ===========================================================================

module RenalEliminationModel

using ..MedLang

export RenalParams, RenalTransporters, TubularSegment
export generate_renal_medlang, simulate_renal_elimination
export calculate_clr, calculate_fraction_reabsorbed
export CKDStage, FanconiSyndrome, Cystinosis
export drug_renal_preset, estimate_renal_clearance
export henderson_hasselbalch_ionized_fraction

# ===========================================================================
# PHYSIOLOGICAL PARAMETERS
# ===========================================================================

"""
Human kidney physiological parameters from literature.

References:
- Rodgers & Rowland 2007 - Kidney physiology
- PMC7577018 - Tubular flow rates
- Deranged Physiology - Renal clearance
"""
const KIDNEY_PHYSIOLOGY = (
    # Blood flow
    Q_renal = 1200.0,               # Renal blood flow (mL/min) ~25% CO
    Q_renal_plasma = 660.0,         # Renal plasma flow (mL/min)

    # Filtration
    GFR_healthy = 120.0,            # Glomerular filtration rate (mL/min)
    filtration_fraction = 0.20,     # GFR/RPF

    # Tubular volumes (mL)
    V_proximal_tubule = 15.0,       # Proximal tubule volume
    V_loop_henle = 5.0,             # Loop of Henle
    V_distal_tubule = 5.0,          # Distal tubule
    V_collecting_duct = 10.0,       # Collecting duct

    # Tubular surface areas (cm²)
    SA_proximal_S1 = 2000.0,        # S1 segment (early proximal)
    SA_proximal_S2 = 2500.0,        # S2 segment (mid proximal)
    SA_proximal_S3 = 1500.0,        # S3 segment (late proximal)
    SA_distal = 500.0,              # Distal tubule
    SA_collecting = 300.0,          # Collecting duct

    # Tubular flow rates (mL/min) - healthy baseline
    TFR_glomerular = 120.0,         # = GFR
    TFR_proximal_S1 = 100.0,        # After early reabsorption
    TFR_proximal_S2 = 50.0,         # After mid reabsorption
    TFR_proximal_S3 = 25.0,         # After late reabsorption
    TFR_loop_henle = 15.0,          # Countercurrent multiplication
    TFR_distal = 10.0,              # After loop concentration
    TFR_collecting = 5.0,           # Final concentration
    TFR_urine = 1.0,                # Urine output (~1.5 L/day)

    # Water reabsorption fractions by segment
    water_reab_proximal = 0.65,     # 65% in proximal tubule
    water_reab_loop = 0.15,         # 15% in loop
    water_reab_distal = 0.10,       # 10% in distal
    water_reab_collecting = 0.09,   # 9% in collecting duct

    # pH values
    pH_plasma = 7.4,
    pH_proximal = 6.8,              # Proximal tubular fluid
    pH_distal = 6.0,                # Distal tubular fluid (can vary 4.5-8.0)
    pH_urine_normal = 6.0,          # Normal urine pH (range 4.5-8.0)

    # Protein binding in tubular fluid
    fu_tubular = 1.0,               # No protein in ultrafiltrate
)

# ===========================================================================
# CKD STAGING
# ===========================================================================

"""
CKD stages based on GFR (mL/min/1.73m²).

Incorporates tubular adaptation factor from PMC7577018.
"""
struct CKDStage
    stage::Int                      # 1-5
    gfr::Float64                    # mL/min
    description::String

    # Tubular adaptation - remaining nephrons work harder
    tubular_flow_adaptation::Float64  # Scalar for TFR increase per nephron

    # Transporter changes in CKD
    oat_expression::Float64         # OAT1/OAT3 expression (uremic toxins inhibit)
    oct2_expression::Float64        # OCT2 expression
    mate_expression::Float64        # MATE1/MATE2-K expression
end

"""
Create CKD stage with physiological adaptations.

The key insight from literature: in CKD, remaining nephrons adapt by
reducing water reabsorption per nephron, which INCREASES tubular flow rate
relative to GFR. This causes NONLINEAR reduction in reabsorption.
"""
function ckd_stage(stage::Int; gfr::Union{Float64,Nothing}=nothing)::CKDStage
    stages = Dict(
        1 => (gfr=100.0, desc="Normal/High GFR", adapt=1.0, oat=1.0, oct=1.0, mate=1.0),
        2 => (gfr=75.0,  desc="Mild decrease", adapt=1.1, oat=0.95, oct=0.95, mate=0.95),
        3 => (gfr=45.0,  desc="Moderate decrease", adapt=1.3, oat=0.80, oct=0.85, mate=0.85),
        4 => (gfr=22.0,  desc="Severe decrease", adapt=1.5, oat=0.60, oct=0.70, mate=0.70),
        5 => (gfr=10.0,  desc="Kidney failure", adapt=1.8, oat=0.30, oct=0.50, mate=0.50),
    )

    s = get(stages, stage, stages[1])
    actual_gfr = gfr !== nothing ? gfr : s.gfr

    return CKDStage(
        stage, actual_gfr, s.desc,
        s.adapt, s.oat, s.oct, s.mate
    )
end

export ckd_stage

# ===========================================================================
# FANCONI SYNDROME / CYSTINOSIS
# ===========================================================================

"""
Fanconi syndrome disease state.

The key mechanistic insight: mTORC1 hyperactivation is the central
driver of proximal tubule dysfunction, affecting transporter expression
and membrane trafficking.

References:
- PMID 37621073 (2023) - CTNS-mTORC1 axis
- Nature Reviews Nephrology 2016 - Cystinosis pathogenesis
"""
struct FanconiSyndrome
    # Disease identification
    etiology::Symbol                # :cystinosis, :drug_induced, :genetic_other
    severity::Float64               # 0.0 (none) to 1.0 (severe)

    # mTORC1 activity (central mechanism)
    # 0 = normal regulation, 1 = maximally hyperactive
    mtorc1_activity::Float64

    # Downstream effects (calculated from mTORC1)
    transporter_expression::Float64 # Scaling factor for all transporters
    atp_availability::Float64       # ATP for active transport
    autophagy_flux::Float64         # Mitophagy/autophagy function
    lysosomal_ph::Float64           # Lysosomal acidification

    # Specific transporter effects
    oat_function::Float64           # OAT1/OAT3
    oct_function::Float64           # OCT2
    napi2a_function::Float64        # Phosphate transporter
    sglt2_function::Float64         # Glucose transporter
    megalin_function::Float64       # Protein reabsorption

    # Clinical markers
    phosphaturia::Bool              # Phosphate wasting
    glucosuria::Bool                # Glucose in urine (normoglycemic)
    aminoaciduria::Bool             # Amino acid wasting
    proteinuria::Bool               # Low MW proteinuria
    metabolic_acidosis::Bool        # Type 2 RTA
end

"""
Create Fanconi syndrome state from mTORC1 activity level.

This is the KEY innovation: modeling Fanconi as "sick nephrons"
rather than "fewer nephrons" (which is CKD).
"""
function fanconi_syndrome(;
    etiology::Symbol = :cystinosis,
    mtorc1_activity::Float64 = 0.7,  # Default: moderately hyperactive
    severity::Union{Float64,Nothing} = nothing
)::FanconiSyndrome

    # If severity given, derive mTORC1 activity
    mtorc1 = severity !== nothing ? severity : mtorc1_activity
    mtorc1 = clamp(mtorc1, 0.0, 1.0)

    # Calculate downstream effects from mTORC1 hyperactivation
    # Based on literature: mTORC1 hyperactivation →
    #   - Autophagy inhibition
    #   - Mitochondrial dysfunction (damaged mito accumulate)
    #   - ATP depletion
    #   - Transporter expression/trafficking impaired

    # Transporter expression: 70% reduction at max mTORC1
    transporter_expr = 1.0 - (mtorc1 * 0.70)

    # ATP availability: 52% reduction reported in cystinosis
    atp = 1.0 - (mtorc1 * 0.52)

    # Autophagy flux: severely impaired
    autophagy = 1.0 - (mtorc1 * 0.80)

    # Lysosomal pH: V-ATPase (ATP6V0A1) downregulated
    # Normal lysosomal pH ~4.5, dysfunction → pH 5.5-6.0
    lyso_ph = 4.5 + (mtorc1 * 1.5)

    # Individual transporter sensitivity to mTORC1
    # NaPi2a and Megalin most affected in cystinosis
    oat_func = transporter_expr * atp
    oct_func = transporter_expr * atp * 1.1  # OCT2 slightly more resilient
    napi2a_func = transporter_expr * atp * 0.7  # Most affected
    sglt2_func = transporter_expr * atp * 0.9
    megalin_func = transporter_expr * atp * 0.6  # Severely affected

    # Clinical features emerge at different thresholds
    phosphaturia = mtorc1 > 0.3
    glucosuria = mtorc1 > 0.5
    aminoaciduria = mtorc1 > 0.4
    proteinuria = mtorc1 > 0.3
    metabolic_acidosis = mtorc1 > 0.6

    return FanconiSyndrome(
        etiology, mtorc1,
        mtorc1,
        transporter_expr, atp, autophagy, lyso_ph,
        oat_func, oct_func, napi2a_func, sglt2_func, megalin_func,
        phosphaturia, glucosuria, aminoaciduria, proteinuria, metabolic_acidosis
    )
end

export fanconi_syndrome

"""
Cystinosis-specific parameters.

Cystinosis = CTNS mutation → cystinosin loss → lysosomal cystine accumulation
→ Ragulator-RRAG activation → mTORC1 hyperactivation → Fanconi syndrome
"""
struct Cystinosis
    # Genetic
    ctns_mutation::String           # e.g., "57kb_deletion", "W138X"
    residual_cystinosin::Float64    # 0-1, some mutations retain partial function

    # Biochemical
    wbc_cystine_nmol_half_cystine_mg_protein::Float64  # Diagnostic marker

    # Treatment
    on_cysteamine::Bool             # Cysteamine depletes lysosomal cystine
    cysteamine_compliance::Float64  # 0-1

    # Derived Fanconi state
    fanconi::FanconiSyndrome
end

"""
Create cystinosis state.

Cysteamine treatment reduces lysosomal cystine, which reduces
mTORC1 hyperactivation, partially rescuing transporter function.
"""
function cystinosis(;
    ctns_mutation::String = "57kb_deletion",
    residual_cystinosin::Float64 = 0.0,
    wbc_cystine::Float64 = 3.0,     # nmol half-cystine/mg protein (normal <0.2)
    on_cysteamine::Bool = true,
    cysteamine_compliance::Float64 = 0.8
)::Cystinosis

    # mTORC1 activity depends on cystine accumulation
    # Cystine >1 nmol/mg → mTORC1 activation
    # Cysteamine reduces cystine ~60-80% with good compliance

    effective_cystine = wbc_cystine
    if on_cysteamine
        reduction = 0.6 + (0.2 * cysteamine_compliance)  # 60-80% reduction
        effective_cystine = wbc_cystine * (1 - reduction * cysteamine_compliance)
    end

    # Residual cystinosin provides some protection
    effective_cystine *= (1 - residual_cystinosin * 0.5)

    # Map cystine level to mTORC1 activity
    # Normal <0.2, mild 0.2-1.0, moderate 1-3, severe >3
    mtorc1 = clamp(effective_cystine / 4.0, 0.0, 1.0)

    fanconi = fanconi_syndrome(
        etiology = :cystinosis,
        mtorc1_activity = mtorc1
    )

    return Cystinosis(
        ctns_mutation, residual_cystinosin,
        wbc_cystine,
        on_cysteamine, cysteamine_compliance,
        fanconi
    )
end

export cystinosis

# ===========================================================================
# RENAL TRANSPORTERS
# ===========================================================================

"""
Renal transporter expression and kinetics.

Basolateral (blood → cell):
- OAT1: organic anions (penicillins, methotrexate, tenofovir)
- OAT3: organic anions (broader substrate specificity)
- OCT2: organic cations (metformin, cisplatin)

Apical (cell → urine):
- MATE1: organic cations (H+/cation antiporter)
- MATE2-K: organic cations (kidney-specific)
- MRP2: organic anions (glutathione conjugates)
- MRP4: organic anions, nucleotides

Apical (urine → cell, reabsorption):
- OAT4: organic anions (urate reabsorption)
- URAT1: urate-specific
"""
struct RenalTransporters
    # Basolateral uptake (blood → cell)
    oat1_expression::Float64
    oat1_km_uM::Float64
    oat1_vmax_pmol_min_mg::Float64

    oat3_expression::Float64
    oat3_km_uM::Float64
    oat3_vmax_pmol_min_mg::Float64

    oct2_expression::Float64
    oct2_km_uM::Float64
    oct2_vmax_pmol_min_mg::Float64

    # Apical efflux (cell → urine)
    mate1_expression::Float64
    mate1_km_uM::Float64
    mate1_vmax_pmol_min_mg::Float64

    mate2k_expression::Float64
    mate2k_km_uM::Float64
    mate2k_vmax_pmol_min_mg::Float64

    mrp2_expression::Float64
    mrp2_km_uM::Float64
    mrp2_vmax_pmol_min_mg::Float64

    mrp4_expression::Float64
    mrp4_km_uM::Float64
    mrp4_vmax_pmol_min_mg::Float64

    # Apical reabsorption
    oat4_expression::Float64
    oat4_km_uM::Float64
end

"""
Default renal transporter parameters for healthy kidney.
"""
function default_renal_transporters()::RenalTransporters
    return RenalTransporters(
        # OAT1 (high capacity anion uptake)
        1.0, 20.0, 500.0,
        # OAT3 (moderate capacity)
        1.0, 10.0, 300.0,
        # OCT2 (cation uptake)
        1.0, 100.0, 400.0,
        # MATE1 (cation efflux)
        1.0, 50.0, 300.0,
        # MATE2-K
        0.8, 30.0, 200.0,
        # MRP2 (anion efflux)
        0.6, 100.0, 150.0,
        # MRP4
        0.8, 50.0, 200.0,
        # OAT4 (reabsorption)
        0.5, 15.0
    )
end

"""
Apply disease state to transporter function.
"""
function apply_disease_to_transporters(
    transporters::RenalTransporters,
    ckd::Union{CKDStage,Nothing} = nothing,
    fanconi::Union{FanconiSyndrome,Nothing} = nothing
)::RenalTransporters

    # Start with baseline
    oat1_expr = transporters.oat1_expression
    oat3_expr = transporters.oat3_expression
    oct2_expr = transporters.oct2_expression
    mate1_expr = transporters.mate1_expression
    mate2k_expr = transporters.mate2k_expression
    mrp2_expr = transporters.mrp2_expression
    mrp4_expr = transporters.mrp4_expression
    oat4_expr = transporters.oat4_expression

    # Apply CKD effects (uremic toxins inhibit transporters)
    if ckd !== nothing
        oat1_expr *= ckd.oat_expression
        oat3_expr *= ckd.oat_expression
        oct2_expr *= ckd.oct2_expression
        mate1_expr *= ckd.mate_expression
        mate2k_expr *= ckd.mate_expression
        mrp2_expr *= ckd.oat_expression
        mrp4_expr *= ckd.oat_expression
    end

    # Apply Fanconi effects (mTORC1 → expression loss)
    if fanconi !== nothing
        oat1_expr *= fanconi.oat_function
        oat3_expr *= fanconi.oat_function
        oct2_expr *= fanconi.oct_function
        mate1_expr *= fanconi.transporter_expression * fanconi.atp_availability
        mate2k_expr *= fanconi.transporter_expression * fanconi.atp_availability
        mrp2_expr *= fanconi.oat_function
        mrp4_expr *= fanconi.oat_function
        oat4_expr *= fanconi.oat_function
    end

    return RenalTransporters(
        oat1_expr, transporters.oat1_km_uM, transporters.oat1_vmax_pmol_min_mg,
        oat3_expr, transporters.oat3_km_uM, transporters.oat3_vmax_pmol_min_mg,
        oct2_expr, transporters.oct2_km_uM, transporters.oct2_vmax_pmol_min_mg,
        mate1_expr, transporters.mate1_km_uM, transporters.mate1_vmax_pmol_min_mg,
        mate2k_expr, transporters.mate2k_km_uM, transporters.mate2k_vmax_pmol_min_mg,
        mrp2_expr, transporters.mrp2_km_uM, transporters.mrp2_vmax_pmol_min_mg,
        mrp4_expr, transporters.mrp4_km_uM, transporters.mrp4_vmax_pmol_min_mg,
        oat4_expr, transporters.oat4_km_uM
    )
end

export default_renal_transporters, apply_disease_to_transporters

# ===========================================================================
# DRUG PARAMETERS FOR RENAL ELIMINATION
# ===========================================================================

"""
Complete renal elimination parameters for a drug.
"""
struct RenalParams
    # Drug identification
    drug_name::String

    # Physicochemistry
    MW::Float64
    logP::Float64
    pKa::Float64
    charge_type::Symbol             # :neutral, :acid, :base, :zwitterion
    fu_plasma::Float64              # Unbound fraction in plasma

    # Permeability (for passive reabsorption)
    Papp_cm_s::Float64              # Apparent permeability

    # Transporter substrates
    is_oat1_substrate::Bool
    oat1_km_uM::Float64
    is_oat3_substrate::Bool
    oat3_km_uM::Float64
    is_oct2_substrate::Bool
    oct2_km_uM::Float64
    is_mate_substrate::Bool
    mate_km_uM::Float64
    is_mrp_substrate::Bool
    mrp_km_uM::Float64

    # Active reabsorption
    is_oat4_substrate::Bool         # Active reabsorption
    is_urat1_substrate::Bool        # Urate transporter

    # Metabolism in kidney
    renal_metabolism::Float64       # Fraction metabolized in kidney
end

# ===========================================================================
# pH-DEPENDENT IONIZATION
# ===========================================================================

"""
Calculate ionized fraction using Henderson-Hasselbalch equation.

For weak acids: fraction_ionized = 1 / (1 + 10^(pKa - pH))
For weak bases: fraction_ionized = 1 / (1 + 10^(pH - pKa))

Only un-ionized drug can passively diffuse across membranes.
"""
function henderson_hasselbalch_ionized_fraction(
    pKa::Float64,
    pH::Float64,
    charge_type::Symbol
)::Float64
    if charge_type == :neutral
        return 0.0  # No ionization
    elseif charge_type == :acid
        # Weak acid: HA ⇌ H+ + A-
        # At pH > pKa, more ionized (A-)
        return 1.0 / (1.0 + 10.0^(pKa - pH))
    elseif charge_type == :base
        # Weak base: B + H+ ⇌ BH+
        # At pH < pKa, more ionized (BH+)
        return 1.0 / (1.0 + 10.0^(pH - pKa))
    elseif charge_type == :zwitterion
        # Simplified: treat as neutral for reabsorption
        return 0.5
    else
        return 0.0
    end
end

"""
Calculate un-ionized fraction available for passive reabsorption.
"""
function fraction_unionized(pKa::Float64, pH::Float64, charge_type::Symbol)::Float64
    return 1.0 - henderson_hasselbalch_ionized_fraction(pKa, pH, charge_type)
end

export henderson_hasselbalch_ionized_fraction, fraction_unionized

# ===========================================================================
# RENAL CLEARANCE CALCULATIONS
# ===========================================================================

"""
Calculate renal clearance components.

CLr = (CLfiltration + CLsecretion) × (1 - Freab)

Where:
- CLfiltration = GFR × fu,p
- CLsecretion = f(transporter activity, substrate affinity)
- Freab = f(permeability, pH, tubular flow rate, transit time)
"""
function calculate_clr(
    params::RenalParams,
    transporters::RenalTransporters;
    gfr::Float64 = KIDNEY_PHYSIOLOGY.GFR_healthy,
    urine_ph::Float64 = KIDNEY_PHYSIOLOGY.pH_urine_normal,
    ckd::Union{CKDStage,Nothing} = nothing,
    fanconi::Union{FanconiSyndrome,Nothing} = nothing,
    C_plasma_uM::Float64 = 1.0
)::Dict{String, Float64}

    # Apply disease states to transporters
    effective_transporters = apply_disease_to_transporters(transporters, ckd, fanconi)

    # Effective GFR
    effective_gfr = ckd !== nothing ? ckd.gfr : gfr

    # 1. Filtration clearance
    CL_filtration = effective_gfr * params.fu_plasma

    # 2. Secretion clearance (sum of transporter contributions)
    CL_secretion = 0.0

    # OAT1-mediated secretion
    if params.is_oat1_substrate
        vmax_oat1 = effective_transporters.oat1_vmax_pmol_min_mg * effective_transporters.oat1_expression
        km_oat1 = params.oat1_km_uM
        # Scale by kidney mass (~300g) and convert units
        CL_oat1 = (vmax_oat1 * 300.0) / (km_oat1 + C_plasma_uM * params.fu_plasma) / 1e6  # mL/min
        CL_secretion += CL_oat1
    end

    # OAT3-mediated secretion
    if params.is_oat3_substrate
        vmax_oat3 = effective_transporters.oat3_vmax_pmol_min_mg * effective_transporters.oat3_expression
        km_oat3 = params.oat3_km_uM
        CL_oat3 = (vmax_oat3 * 300.0) / (km_oat3 + C_plasma_uM * params.fu_plasma) / 1e6
        CL_secretion += CL_oat3
    end

    # OCT2/MATE-mediated secretion (cations)
    if params.is_oct2_substrate && params.is_mate_substrate
        # Vectorial transport: OCT2 uptake → MATE efflux
        vmax_oct2 = effective_transporters.oct2_vmax_pmol_min_mg * effective_transporters.oct2_expression
        vmax_mate = effective_transporters.mate1_vmax_pmol_min_mg * effective_transporters.mate1_expression
        km_oct2 = params.oct2_km_uM

        # Rate-limiting step determines overall clearance
        CL_oct2 = (vmax_oct2 * 300.0) / (km_oct2 + C_plasma_uM * params.fu_plasma) / 1e6
        CL_mate = (vmax_mate * 300.0) / (params.mate_km_uM + C_plasma_uM * params.fu_plasma) / 1e6
        CL_cation = min(CL_oct2, CL_mate)  # Rate-limited by slower step
        CL_secretion += CL_cation
    end

    # 3. Fraction reabsorbed (pH-dependent passive diffusion)
    F_reab = calculate_fraction_reabsorbed(
        params,
        urine_ph,
        ckd = ckd,
        fanconi = fanconi
    )

    # Active reabsorption (OAT4)
    if params.is_oat4_substrate
        oat4_factor = effective_transporters.oat4_expression * 0.2  # Additional 20% at full expression
        F_reab = min(F_reab + oat4_factor, 0.99)
    end

    # 4. Total renal clearance
    CL_renal = (CL_filtration + CL_secretion) * (1.0 - F_reab)

    # 5. Renal metabolism contribution
    if params.renal_metabolism > 0
        CL_renal_metabolism = effective_gfr * params.renal_metabolism * params.fu_plasma
        CL_renal += CL_renal_metabolism
    end

    return Dict{String, Float64}(
        "CL_filtration" => CL_filtration,
        "CL_secretion" => CL_secretion,
        "CL_reabsorption" => (CL_filtration + CL_secretion) * F_reab,
        "F_reabsorbed" => F_reab,
        "CL_renal" => CL_renal,
        "CL_renal_metabolism" => params.renal_metabolism > 0 ? effective_gfr * params.renal_metabolism * params.fu_plasma : 0.0,
        "effective_GFR" => effective_gfr,
        "fe" => CL_renal / (CL_renal + 50.0)  # Approximate fe (assuming hepatic CL ~50 mL/min)
    )
end

"""
Calculate fraction of drug reabsorbed in renal tubules.

Based on PMC7577018 mechanistic model incorporating:
- Drug permeability
- Ionization state (pH-dependent)
- Tubular flow rate
- Transit time
- CKD adaptation (increased TFR per nephron)
"""
function calculate_fraction_reabsorbed(
    params::RenalParams,
    urine_ph::Float64;
    ckd::Union{CKDStage,Nothing} = nothing,
    fanconi::Union{FanconiSyndrome,Nothing} = nothing
)::Float64

    # Base permeability
    Papp = params.Papp_cm_s

    # pH-dependent ionization reduces effective permeability
    # Use average pH along tubule (proximal → collecting)
    avg_pH = (KIDNEY_PHYSIOLOGY.pH_proximal + urine_ph) / 2.0
    f_unionized = fraction_unionized(params.pKa, avg_pH, params.charge_type)

    # Effective permeability (only un-ionized drug reabsorbed)
    Peff = Papp * f_unionized

    # Tubular flow rate adaptation in CKD
    # Key insight: remaining nephrons have HIGHER flow per nephron
    tfr_scalar = 1.0
    if ckd !== nothing
        tfr_scalar = ckd.tubular_flow_adaptation
    end

    # In Fanconi, tubular dysfunction reduces reabsorption capacity
    reab_capacity = 1.0
    if fanconi !== nothing
        # mTORC1 hyperactivation disrupts epithelial integrity
        reab_capacity = fanconi.transporter_expression * fanconi.atp_availability
    end

    # Surface area for reabsorption (mainly proximal tubule)
    SA_total = KIDNEY_PHYSIOLOGY.SA_proximal_S1 +
               KIDNEY_PHYSIOLOGY.SA_proximal_S2 +
               KIDNEY_PHYSIOLOGY.SA_proximal_S3

    # Average tubular flow rate
    TFR_avg = (KIDNEY_PHYSIOLOGY.TFR_proximal_S1 +
               KIDNEY_PHYSIOLOGY.TFR_proximal_S3) / 2.0
    TFR_avg *= tfr_scalar  # CKD adaptation

    # Fraction reabsorbed using membrane permeability model
    # F_reab = 1 - exp(-Peff × SA / TFR)
    # Higher flow = less time for reabsorption = lower F_reab

    exponent = -(Peff * SA_total * 60.0 * reab_capacity) / TFR_avg  # Convert cm/s to cm/min
    F_reab = 1.0 - exp(exponent)

    # Clamp to reasonable range
    return clamp(F_reab, 0.0, 0.99)
end

export calculate_clr, calculate_fraction_reabsorbed

# ===========================================================================
# MEDLANG CODE GENERATION
# ===========================================================================

"""
Generate MedLang DSL code for complete renal elimination model.
"""
function generate_renal_medlang(
    params::RenalParams;
    ckd::Union{CKDStage,Nothing} = nothing,
    fanconi::Union{FanconiSyndrome,Nothing} = nothing,
    urine_ph::Float64 = 6.0
)::String

    buf = IOBuffer()

    # Calculate clearance components
    transporters = default_renal_transporters()
    cl_results = calculate_clr(params, transporters;
                               urine_ph=urine_ph, ckd=ckd, fanconi=fanconi)

    # Ionization
    f_ionized_plasma = henderson_hasselbalch_ionized_fraction(
        params.pKa, KIDNEY_PHYSIOLOGY.pH_plasma, params.charge_type
    )
    f_ionized_urine = henderson_hasselbalch_ionized_fraction(
        params.pKa, urine_ph, params.charge_type
    )

    disease_section = ""
    if ckd !== nothing
        disease_section *= """
    // CKD Stage $(ckd.stage): $(ckd.description)
    disease_state CKD {
        stage: $(ckd.stage)
        gfr: $(ckd.gfr)_mL/min
        tubular_flow_adaptation: $(ckd.tubular_flow_adaptation)
        oat_expression: $(round(ckd.oat_expression, digits=2))
        oct2_expression: $(round(ckd.oct2_expression, digits=2))
        mate_expression: $(round(ckd.mate_expression, digits=2))
    }
"""
    end

    if fanconi !== nothing
        disease_section *= """
    // Fanconi Syndrome ($(fanconi.etiology))
    // Central mechanism: mTORC1 hyperactivation
    disease_state Fanconi {
        etiology: $(fanconi.etiology)
        mtorc1_activity: $(round(fanconi.mtorc1_activity, digits=2))

        // Downstream effects
        transporter_expression: $(round(fanconi.transporter_expression, digits=2))
        atp_availability: $(round(fanconi.atp_availability, digits=2))
        autophagy_flux: $(round(fanconi.autophagy_flux, digits=2))
        lysosomal_pH: $(round(fanconi.lysosomal_ph, digits=1))

        // Transporter-specific function
        oat_function: $(round(fanconi.oat_function, digits=2))
        oct_function: $(round(fanconi.oct_function, digits=2))
        napi2a_function: $(round(fanconi.napi2a_function, digits=2))
        megalin_function: $(round(fanconi.megalin_function, digits=2))

        // Clinical manifestations
        phosphaturia: $(fanconi.phosphaturia)
        glucosuria: $(fanconi.glucosuria)
        aminoaciduria: $(fanconi.aminoaciduria)
        proteinuria: $(fanconi.proteinuria)
        metabolic_acidosis: $(fanconi.metabolic_acidosis)
    }
"""
    end

    println(buf, """
model $(params.drug_name)_Renal_PBPK {
    // ================================================================
    // RENAL ELIMINATION MODEL
    // Generated by Darwin PBPK Platform - MedLang DSL
    // ================================================================
    // Drug: $(params.drug_name)
    // MW: $(params.MW) Da
    // logP: $(params.logP)
    // pKa: $(params.pKa) ($(params.charge_type))
    // fu,plasma: $(params.fu_plasma)
    //
    // Ionization at pH 7.4 (plasma): $(round(f_ionized_plasma * 100, digits=1))% ionized
    // Ionization at pH $(urine_ph) (urine): $(round(f_ionized_urine * 100, digits=1))% ionized
    //
    // Transporter substrates:
    //   OAT1: $(params.is_oat1_substrate) (Km: $(params.oat1_km_uM) µM)
    //   OAT3: $(params.is_oat3_substrate) (Km: $(params.oat3_km_uM) µM)
    //   OCT2: $(params.is_oct2_substrate) (Km: $(params.oct2_km_uM) µM)
    //   MATE1/2-K: $(params.is_mate_substrate) (Km: $(params.mate_km_uM) µM)
    //
    // Calculated renal clearance:
    //   CL_filtration: $(round(cl_results["CL_filtration"], digits=1)) mL/min
    //   CL_secretion: $(round(cl_results["CL_secretion"], digits=1)) mL/min
    //   F_reabsorbed: $(round(cl_results["F_reabsorbed"] * 100, digits=1))%
    //   CL_renal: $(round(cl_results["CL_renal"], digits=1)) mL/min
    // ================================================================

$disease_section
    // ================================================================
    // KIDNEY PHYSIOLOGY
    // ================================================================
    organ kidney {
        blood_flow: $(KIDNEY_PHYSIOLOGY.Q_renal)_mL/min
        plasma_flow: $(KIDNEY_PHYSIOLOGY.Q_renal_plasma)_mL/min
        GFR: $(cl_results["effective_GFR"])_mL/min

        // Tubular segments
        segment proximal_S1 {
            surface_area: $(KIDNEY_PHYSIOLOGY.SA_proximal_S1)_cm2
            flow_rate: $(KIDNEY_PHYSIOLOGY.TFR_proximal_S1)_mL/min
            pH: $(KIDNEY_PHYSIOLOGY.pH_proximal)
        }
        segment proximal_S2 {
            surface_area: $(KIDNEY_PHYSIOLOGY.SA_proximal_S2)_cm2
            flow_rate: $(KIDNEY_PHYSIOLOGY.TFR_proximal_S2)_mL/min
        }
        segment proximal_S3 {
            surface_area: $(KIDNEY_PHYSIOLOGY.SA_proximal_S3)_cm2
            flow_rate: $(KIDNEY_PHYSIOLOGY.TFR_proximal_S3)_mL/min
        }
        segment distal {
            surface_area: $(KIDNEY_PHYSIOLOGY.SA_distal)_cm2
            flow_rate: $(KIDNEY_PHYSIOLOGY.TFR_distal)_mL/min
            pH: $(KIDNEY_PHYSIOLOGY.pH_distal)
        }
        segment collecting_duct {
            surface_area: $(KIDNEY_PHYSIOLOGY.SA_collecting)_cm2
            flow_rate: $(KIDNEY_PHYSIOLOGY.TFR_urine)_mL/min
            pH: $(urine_ph)
        }
    }

    // ================================================================
    // TRANSPORTERS (with membrane localization)
    // ================================================================

    // Basolateral membrane (blood → proximal tubule cell)
    transporters_basolateral {
        OAT1: {
            substrate: $(params.is_oat1_substrate),
            Km: $(params.oat1_km_uM)_uM,
            direction: blood_to_cell,
            substrates: [penicillins, methotrexate, tenofovir, NSAIDs]
        }
        OAT3: {
            substrate: $(params.is_oat3_substrate),
            Km: $(params.oat3_km_uM)_uM,
            direction: blood_to_cell,
            substrates: [cimetidine, pravastatin, benzylpenicillin]
        }
        OCT2: {
            substrate: $(params.is_oct2_substrate),
            Km: $(params.oct2_km_uM)_uM,
            direction: blood_to_cell,
            substrates: [metformin, cisplatin, creatinine]
        }
    }

    // Apical membrane (proximal tubule cell → urine)
    transporters_apical {
        MATE1: {
            substrate: $(params.is_mate_substrate),
            Km: $(params.mate_km_uM)_uM,
            direction: cell_to_urine,
            mechanism: H+/cation_antiporter
        }
        MATE2_K: {
            substrate: $(params.is_mate_substrate),
            Km: $(params.mate_km_uM)_uM,
            direction: cell_to_urine,
            kidney_specific: true
        }
        MRP2: {
            substrate: $(params.is_mrp_substrate),
            Km: $(params.mrp_km_uM)_uM,
            direction: cell_to_urine,
            substrates: [glutathione_conjugates, methotrexate]
        }
        MRP4: {
            substrate: $(params.is_mrp_substrate),
            direction: cell_to_urine,
            substrates: [nucleotides, adefovir, tenofovir]
        }
    }

    // Reabsorption transporters (apical, urine → cell)
    transporters_reabsorption {
        OAT4: {
            substrate: $(params.is_oat4_substrate),
            direction: urine_to_cell,
            substrates: [urate, diuretics]
        }
    }

    // ================================================================
    // pH-DEPENDENT IONIZATION (Henderson-Hasselbalch)
    // ================================================================
    ionization {
        pKa: $(params.pKa)
        charge_type: $(params.charge_type)

        // Weak $(params.charge_type): $(params.charge_type == :acid ? "HA ⇌ H+ + A-" : "B + H+ ⇌ BH+")
        // At pH > pKa: $(params.charge_type == :acid ? "more ionized (trapped)" : "less ionized (reabsorbed)")

        fraction_ionized_plasma: $(round(f_ionized_plasma, digits=3))
        fraction_ionized_urine: $(round(f_ionized_urine, digits=3))
        fraction_unionized_urine: $(round(1 - f_ionized_urine, digits=3))

        // Clinical implication:
        // $(params.charge_type == :acid ?
            "Alkalinize urine (pH↑) → more ionized → trapped → enhanced excretion" :
            "Acidify urine (pH↓) → more ionized → trapped → enhanced excretion")
    }

    // ================================================================
    // RENAL CLEARANCE EQUATIONS
    // ================================================================

    // CLr = (CL_filtration + CL_secretion) × (1 - F_reabsorbed)

    clearance filtration {
        equation: GFR × fu_plasma
        value: $(round(cl_results["CL_filtration"], digits=2))_mL/min
    }

    clearance secretion {
        mechanism: active_transport
        transporters: [OAT1, OAT3, OCT2, MATE1, MATE2_K, MRP2, MRP4]
        value: $(round(cl_results["CL_secretion"], digits=2))_mL/min
    }

    reabsorption passive {
        mechanism: pH_dependent_diffusion
        permeability: $(params.Papp_cm_s)_cm/s
        fraction_reabsorbed: $(round(cl_results["F_reabsorbed"], digits=3))

        // Factors affecting reabsorption:
        // 1. Lipophilicity (logP = $(params.logP))
        // 2. Ionization state at urine pH
        // 3. Tubular flow rate (residence time)
        // 4. Surface area available
    }

    clearance renal_total {
        equation: (CL_filt + CL_sec) × (1 - F_reab)
        value: $(round(cl_results["CL_renal"], digits=2))_mL/min
        fe: $(round(cl_results["fe"], digits=2))
    }

    // ================================================================
    // STATE VARIABLES
    // ================================================================
    state C_plasma: Concentration = 0.0_uM
    state C_proximal: Concentration = 0.0_uM
    state C_tubular_fluid: Concentration = 0.0_uM
    state A_urine: Amount = 0.0_mg

    // ================================================================
    // PARAMETERS
    // ================================================================
    param GFR: Real = $(cl_results["effective_GFR"])_mL/min
    param fu_plasma: Real = $(params.fu_plasma)
    param CL_secretion: Real = $(round(cl_results["CL_secretion"], digits=2))_mL/min
    param F_reabsorbed: Real = $(round(cl_results["F_reabsorbed"], digits=3))
    param urine_flow: Real = $(KIDNEY_PHYSIOLOGY.TFR_urine)_mL/min
    param urine_pH: Real = $(urine_ph)

    // ================================================================
    // ODE EQUATIONS
    // ================================================================

    // Glomerular filtration
    ode dC_tubular_fluid/dt = (
        GFR * fu_plasma * C_plasma    // Filtration input
        + CL_secretion * C_plasma     // Secretion input
        - (1 - F_reabsorbed) * (GFR * fu_plasma + CL_secretion) * C_tubular_fluid / V_tubule
    ) / V_tubule

    // Urinary excretion
    ode dA_urine/dt = urine_flow * C_tubular_fluid * (1 - F_reabsorbed)

    // ================================================================
    // OBSERVABLES
    // ================================================================
    observable CLr = (GFR * fu_plasma + CL_secretion) * (1 - F_reabsorbed)
    observable urinary_excretion_rate = urine_flow * C_tubular_fluid
    observable cumulative_urine_amount = A_urine
    observable fe = CLr / (CLr + CL_hepatic)  // Fraction excreted unchanged
}
""")

    return String(take!(buf))
end

export generate_renal_medlang

# ===========================================================================
# SIMULATION
# ===========================================================================

"""
Simulate renal elimination and urinary excretion.
"""
function simulate_renal_elimination(
    params::RenalParams,
    dose_mg::Float64;
    t_max_h::Float64 = 24.0,
    dt_min::Float64 = 1.0,
    urine_ph::Float64 = 6.0,
    ckd::Union{CKDStage,Nothing} = nothing,
    fanconi::Union{FanconiSyndrome,Nothing} = nothing,
    CL_hepatic_mL_min::Float64 = 50.0,
    Vd_L::Float64 = 50.0
)::Dict{String, Any}

    # Calculate clearance components
    transporters = default_renal_transporters()
    cl_results = calculate_clr(params, transporters;
                               urine_ph=urine_ph, ckd=ckd, fanconi=fanconi)

    CL_renal = cl_results["CL_renal"]
    CL_total = CL_renal + CL_hepatic_mL_min
    ke = CL_total / (Vd_L * 1000)  # min⁻¹

    # Initial concentration
    C0 = (dose_mg * 1000 / params.MW) / (Vd_L * 1000)  # µM

    # Time series
    n_steps = Int(ceil(t_max_h * 60 / dt_min))
    times = Float64[]
    plasma_conc = Float64[]
    urine_amount = Float64[]
    urine_rate = Float64[]

    C_plasma = C0
    A_urine = 0.0

    for step in 1:n_steps
        t_min = step * dt_min

        # Plasma decay
        dC = -ke * C_plasma * dt_min
        C_plasma += dC
        C_plasma = max(C_plasma, 0.0)

        # Urinary excretion
        excretion_rate = CL_renal * C_plasma * params.MW / 1000  # mg/min
        A_urine += excretion_rate * dt_min

        push!(times, t_min / 60.0)
        push!(plasma_conc, C_plasma)
        push!(urine_amount, A_urine)
        push!(urine_rate, excretion_rate * 60.0)  # mg/h
    end

    # Calculate fe
    fe = CL_renal / CL_total

    return Dict{String, Any}(
        "time_h" => times,
        "C_plasma_uM" => plasma_conc,
        "A_urine_mg" => urine_amount,
        "excretion_rate_mg_h" => urine_rate,
        "CL_renal_mL_min" => CL_renal,
        "CL_total_mL_min" => CL_total,
        "fe" => fe,
        "F_reabsorbed" => cl_results["F_reabsorbed"],
        "half_life_h" => 0.693 / ke / 60,
        "clearance_components" => cl_results,
        "params" => params
    )
end

export simulate_renal_elimination

# ===========================================================================
# DRUG PRESETS
# ===========================================================================

"""
Create RenalParams for known drugs.
"""
function drug_renal_preset(name::Symbol)::RenalParams
    presets = Dict(
        # Metformin: OCT2/MATE substrate, no metabolism, high renal clearance
        :metformin => RenalParams(
            "Metformin",
            129.2, -1.4, 11.5, :base, 0.99,
            0.1e-5,                          # Low permeability (hydrophilic)
            false, 0.0, false, 0.0,          # Not OAT substrate
            true, 250.0,                     # OCT2 substrate
            true, 100.0,                     # MATE substrate
            false, 0.0,                      # Not MRP substrate
            false, false,                    # No active reabsorption
            0.0                              # No renal metabolism
        ),

        # Tenofovir: OAT1/OAT3 substrate, MRP4 efflux
        :tenofovir => RenalParams(
            "Tenofovir",
            287.2, -1.6, 3.8, :acid, 0.99,
            0.5e-5,                          # Low permeability
            true, 30.0,                      # OAT1 substrate
            true, 20.0,                      # OAT3 substrate
            false, 0.0,                      # Not OCT2 substrate
            false, 0.0,                      # Not MATE substrate
            true, 50.0,                      # MRP4 substrate
            false, false,
            0.0
        ),

        # Penicillin G: OAT1/OAT3 substrate, classic secretion example
        :penicillin_g => RenalParams(
            "Penicillin G",
            334.4, 1.8, 2.7, :acid, 0.45,
            1.0e-5,
            true, 60.0,                      # OAT1 substrate
            true, 15.0,                      # OAT3 substrate
            false, 0.0,
            false, 0.0,
            true, 100.0,
            false, false,
            0.0
        ),

        # Methotrexate: OAT1/OAT3/MRP2 substrate
        :methotrexate => RenalParams(
            "Methotrexate",
            454.4, -1.8, 4.8, :acid, 0.50,
            0.2e-5,
            true, 15.0,                      # OAT1 substrate (high affinity)
            true, 10.0,                      # OAT3 substrate
            false, 0.0,
            false, 0.0,
            true, 30.0,                      # MRP2 substrate
            false, false,
            0.0
        ),

        # Amphetamine: weak base, pH-dependent reabsorption
        :amphetamine => RenalParams(
            "Amphetamine",
            135.2, 1.8, 9.9, :base, 0.80,
            5.0e-5,                          # Moderate permeability (lipophilic base)
            false, 0.0, false, 0.0,          # Not OAT substrate
            true, 500.0,                     # OCT2 substrate (weak)
            true, 200.0,                     # MATE substrate (weak)
            false, 0.0,
            false, false,
            0.0
        ),

        # Aspirin (salicylic acid): weak acid, pH-dependent reabsorption
        :aspirin => RenalParams(
            "Aspirin (Salicylic acid)",
            180.2, 1.2, 3.0, :acid, 0.20,
            3.0e-5,                          # Moderate permeability
            true, 100.0,                     # OAT substrate (weak)
            true, 80.0,
            false, 0.0,
            false, 0.0,
            false, 0.0,
            false, false,
            0.0
        ),

        # Gabapentin: zwitterion, low permeability, minimal reabsorption
        :gabapentin => RenalParams(
            "Gabapentin",
            171.2, -1.1, 3.7, :zwitterion, 0.97,
            0.5e-5,                          # Very low permeability
            false, 0.0, false, 0.0,
            false, 0.0,
            false, 0.0,
            false, 0.0,
            false, false,
            0.0
        ),
    )

    return get(presets, name, presets[:metformin])
end

export drug_renal_preset

# ===========================================================================
# CLINICAL SCENARIOS
# ===========================================================================

"""
Estimate renal clearance adjustment for clinical scenario.

Useful for dose adjustment in CKD or Fanconi syndrome.
"""
function estimate_renal_clearance(
    drug::Symbol;
    ckd_stage::Int = 1,
    has_fanconi::Bool = false,
    fanconi_severity::Float64 = 0.5,
    urine_ph::Float64 = 6.0
)::Dict{String, Any}

    params = drug_renal_preset(drug)

    ckd = ckd_stage > 1 ? ckd_stage(ckd_stage) : nothing
    fanconi = has_fanconi ? fanconi_syndrome(mtorc1_activity=fanconi_severity) : nothing

    transporters = default_renal_transporters()

    # Baseline (healthy)
    cl_healthy = calculate_clr(params, transporters; urine_ph=urine_ph)

    # Disease state
    cl_disease = calculate_clr(params, transporters;
                               urine_ph=urine_ph, ckd=ckd, fanconi=fanconi)

    # Dose adjustment ratio
    dose_adjustment = cl_disease["CL_renal"] / cl_healthy["CL_renal"]

    return Dict{String, Any}(
        "drug" => String(drug),
        "healthy_CLr" => cl_healthy["CL_renal"],
        "disease_CLr" => cl_disease["CL_renal"],
        "dose_adjustment_ratio" => dose_adjustment,
        "recommended_dose_fraction" => dose_adjustment,
        "ckd_stage" => ckd !== nothing ? ckd.stage : 1,
        "fanconi_severity" => has_fanconi ? fanconi_severity : 0.0,
        "clearance_components_healthy" => cl_healthy,
        "clearance_components_disease" => cl_disease
    )
end

export estimate_renal_clearance

end # module
