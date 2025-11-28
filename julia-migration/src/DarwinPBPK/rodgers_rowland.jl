"""
Rodgers-Rowland Mechanistic Tissue Partition Coefficient Prediction

Implements the mechanistic equations from:
- Rodgers & Rowland (2006) J Pharm Sci 95:1238-1257
- Rodgers, Leahy & Rowland (2005) J Pharm Sci 94:1259-1276

Predicts tissue:plasma partition coefficients (Kp) from physicochemical
properties, then uses Øie-Tozer to predict Vdss.
"""
module RodgersRowland

export predict_kp_all_tissues, predict_vdss_mechanistic, OieTozerVdss,
       PhysiologicalParams, MoleculeParams, compute_fut

using Statistics

# ============================================================================
# PHYSIOLOGICAL PARAMETERS
# ============================================================================

"""
Tissue composition data for humans (Rodgers & Rowland 2006, Table 1)
All values are fractions of tissue volume
"""
struct TissueComposition
    fw::Float64      # Fraction water (total)
    few::Float64     # Fraction extracellular water
    fiw::Float64     # Fraction intracellular water
    fnl::Float64     # Fraction neutral lipids
    fph::Float64     # Fraction phospholipids
    fap::Float64     # Fraction acidic phospholipids
    fnp::Float64     # Fraction neutral phospholipids
    pH_iw::Float64   # Intracellular pH
end

# Human tissue compositions (Rodgers & Rowland 2006)
const TISSUE_COMPOSITION = Dict{Symbol, TissueComposition}(
    :adipose => TissueComposition(0.18, 0.135, 0.017, 0.79, 0.002, 0.0004, 0.0016, 7.0),
    :bone => TissueComposition(0.44, 0.10, 0.34, 0.017, 0.0017, 0.00034, 0.00136, 7.0),
    :brain => TissueComposition(0.77, 0.16, 0.62, 0.039, 0.0457, 0.00914, 0.03656, 7.0),
    :gut => TissueComposition(0.72, 0.28, 0.45, 0.038, 0.0125, 0.0025, 0.01, 7.0),
    :heart => TissueComposition(0.76, 0.32, 0.46, 0.014, 0.0166, 0.00332, 0.01328, 7.0),
    :kidney => TissueComposition(0.78, 0.27, 0.48, 0.013, 0.0244, 0.00488, 0.01952, 7.0),
    :liver => TissueComposition(0.71, 0.16, 0.57, 0.014, 0.0243, 0.00486, 0.01944, 7.0),
    :lung => TissueComposition(0.81, 0.34, 0.45, 0.022, 0.0128, 0.00256, 0.01024, 7.0),
    :muscle => TissueComposition(0.76, 0.12, 0.63, 0.010, 0.0072, 0.00144, 0.00576, 7.0),
    :skin => TissueComposition(0.65, 0.38, 0.29, 0.060, 0.0044, 0.00088, 0.00352, 7.0),
    :spleen => TissueComposition(0.77, 0.21, 0.58, 0.010, 0.0113, 0.00226, 0.00904, 7.0),
    :pancreas => TissueComposition(0.66, 0.12, 0.54, 0.041, 0.0093, 0.00186, 0.00744, 7.0)
)

# Tissue volumes (L) for 70 kg human
const TISSUE_VOLUMES = Dict{Symbol, Float64}(
    :adipose => 14.0,
    :bone => 4.9,
    :brain => 1.4,
    :gut => 2.1,
    :heart => 0.33,
    :kidney => 0.31,
    :liver => 1.8,
    :lung => 0.53,
    :muscle => 28.0,
    :skin => 7.0,
    :spleen => 0.18,
    :pancreas => 0.10,
    :plasma => 3.0,
    :rbc => 2.0
)

# Plasma composition
const PLASMA_ALBUMIN = 45.0  # g/L
const PLASMA_AAG = 0.7  # α1-acid glycoprotein, g/L
const PLASMA_PH = 7.4
const RBC_PH = 7.22

# ============================================================================
# MOLECULE PARAMETERS
# ============================================================================

"""
Input parameters for a drug molecule
"""
struct MoleculeParams
    logP::Float64       # Octanol-water partition coefficient
    logD74::Float64     # Distribution coefficient at pH 7.4
    pKa::Float64        # Acid dissociation constant (for bases, use conjugate acid)
    is_base::Bool       # True if moderate-strong base (pKa > 7)
    is_acid::Bool       # True if acid
    is_neutral::Bool    # True if neutral or zwitterion
    fup::Float64        # Fraction unbound in plasma
    MW::Float64         # Molecular weight
    BP::Float64         # Blood:plasma ratio (if known, else estimate)
end

"""
Create MoleculeParams from basic inputs with automatic classification
"""
function MoleculeParams(; logP::Float64, logD74::Float64, pKa::Float64=7.0,
                         fup::Float64, MW::Float64=400.0, BP::Float64=-1.0)
    # Classify based on pKa and logP/logD difference
    logP_logD_diff = logP - logD74

    # Strong base: pKa > 7 and significant ionization at pH 7.4
    is_base = pKa > 7.0 && logP_logD_diff > 0.5

    # Acid: pKa < 7 and logD < logP (ionized at physiological pH)
    is_acid = pKa < 5.5 && logP_logD_diff > 0.5

    # Neutral or weak base/acid
    is_neutral = !is_base && !is_acid

    # Estimate BP if not provided
    bp_est = BP > 0 ? BP : estimate_blood_plasma_ratio(logP, logD74, pKa, fup)

    return MoleculeParams(logP, logD74, pKa, is_base, is_acid, is_neutral, fup, MW, bp_est)
end

"""
Estimate blood:plasma ratio from physicochemical properties
"""
function estimate_blood_plasma_ratio(logP, logD74, pKa, fup)
    # Simplified model based on RBC partitioning
    # BP = 1 + HCT * (Krbc - 1)
    # where Krbc depends on lipophilicity and ionization

    HCT = 0.45  # Hematocrit

    # RBC partitioning (simplified)
    P = 10^logD74  # Apparent partition coefficient
    Krbc = 1 + 0.5 * P / (1 + P)  # Sigmoid relationship

    # Correction for binding
    BP = fup + HCT * (Krbc * fup + (1 - fup))
    BP = clamp(BP, 0.5, 3.0)

    return BP
end

# ============================================================================
# PARTITION COEFFICIENT PREDICTION
# ============================================================================

"""
Predict Kpu (unbound tissue:unbound plasma partition) using Rodgers & Rowland

For moderate-to-strong bases (pKa > 7):
Uses equation accounting for electrostatic interactions with acidic phospholipids

For acids, neutrals, weak bases (pKa < 7):
Uses standard partition model
"""
function predict_kpu(mol::MoleculeParams, tissue::Symbol)
    comp = get(TISSUE_COMPOSITION, tissue, TISSUE_COMPOSITION[:muscle])

    # Apparent partition coefficient
    P = 10^mol.logD74

    # Ionization fractions at physiological pH
    fi_plasma = ionized_fraction(mol.pKa, PLASMA_PH, mol.is_base)
    fi_tissue = ionized_fraction(mol.pKa, comp.pH_iw, mol.is_base)

    if mol.is_base && mol.pKa > 7.0
        # Rodgers et al. 2005 - Strong bases
        # Kpu = (few + (1 + 10^(pKa - pH_iw)) / (1 + 10^(pKa - pH_p)) ×
        #        (fiw + Ka_AP × fap + P × fnl + (0.3P + 0.7) × fnp))

        # Association constant with acidic phospholipids
        # Estimated from RBC partitioning: Ka_AP ≈ 10^(logP + 0.5)
        Ka_AP = 10^(mol.logP + 0.5) * 125.0  # Scaling factor from Rodgers

        pH_ratio = (1 + 10^(mol.pKa - comp.pH_iw)) / (1 + 10^(mol.pKa - PLASMA_PH))

        Kpu = comp.few + pH_ratio * (
            comp.fiw +
            Ka_AP * comp.fap / mol.fup +  # AP binding term
            P * comp.fnl +
            (0.3 * P + 0.7) * comp.fnp
        )
    else
        # Rodgers & Rowland 2006 - Acids, neutrals, weak bases
        # Kpu = few + fiw × fu_iw/fup + P × fnl + (0.3P + 0.7) × fph

        # fu_iw ≈ fup for neutrals, adjusted for ionization for acids
        fu_iw = mol.fup * (1 - fi_tissue) / (1 - fi_plasma + 1e-10)
        fu_iw = clamp(fu_iw, 0.001, 1.0)

        Kpu = comp.few + comp.fiw * fu_iw / mol.fup +
              P * comp.fnl +
              (0.3 * P + 0.7) * comp.fph
    end

    # Ensure physiological bounds
    Kpu = clamp(Kpu, 0.1, 500.0)

    return Kpu
end

"""
Calculate ionized fraction at given pH
"""
function ionized_fraction(pKa::Float64, pH::Float64, is_base::Bool)
    if is_base
        # For bases: ionized = protonated form
        return 1 / (1 + 10^(pH - pKa))
    else
        # For acids: ionized = deprotonated form
        return 1 / (1 + 10^(pKa - pH))
    end
end

"""
Predict Kp (total tissue:total plasma partition) from Kpu and binding
"""
function predict_kp(mol::MoleculeParams, tissue::Symbol)
    Kpu = predict_kpu(mol, tissue)

    # Kp = Kpu × fup / fut
    # For simplicity, we assume fut ≈ fup for non-binding tissues
    # More accurate: fut from tissue protein content

    # Tissue unbound fraction (simplified)
    comp = get(TISSUE_COMPOSITION, tissue, TISSUE_COMPOSITION[:muscle])

    # Estimate fut from tissue albumin content (scaled from plasma)
    tissue_albumin_factor = comp.fw / 0.94  # Relative to plasma water
    fut = 1 - (1 - mol.fup) * tissue_albumin_factor * 0.5  # Reduced binding in tissues
    fut = clamp(fut, 0.01, 1.0)

    Kp = Kpu * mol.fup / fut

    return Kp
end

"""
Predict Kp for all major tissues
"""
function predict_kp_all_tissues(mol::MoleculeParams)
    tissues = [:adipose, :bone, :brain, :gut, :heart, :kidney,
               :liver, :lung, :muscle, :skin, :spleen, :pancreas]

    return Dict(tissue => predict_kp(mol, tissue) for tissue in tissues)
end

# ============================================================================
# VOLUME OF DISTRIBUTION PREDICTION
# ============================================================================

"""
Predict Vdss using Øie-Tozer equation with predicted Kp values

Vdss = Vp + Ve × fup + Σ(Vt × Kp_t)

Where:
- Vp = plasma volume
- Ve = extracellular fluid volume (excluding plasma)
- Vt = tissue volume
- Kp_t = tissue:plasma partition coefficient
"""
function predict_vdss_mechanistic(mol::MoleculeParams; body_weight::Float64=70.0)
    # Scale volumes to body weight
    scale = body_weight / 70.0

    # Plasma and extracellular volumes
    Vp = TISSUE_VOLUMES[:plasma] * scale
    Ve = 15.0 * scale - Vp  # Total extracellular ~ 15L for 70kg

    # Predict Kp for all tissues
    kp_values = predict_kp_all_tissues(mol)

    # Sum tissue contributions
    tissue_sum = 0.0
    for (tissue, kp) in kp_values
        Vt = get(TISSUE_VOLUMES, tissue, 1.0) * scale
        tissue_sum += Vt * kp
    end

    # Øie-Tozer
    Vdss = Vp + Ve * mol.fup + tissue_sum

    # Convert to L/kg
    Vdss_per_kg = Vdss / body_weight

    return Vdss_per_kg
end

"""
Compute fraction unbound in tissue (fut) - average across tissues
"""
function compute_fut(mol::MoleculeParams)
    # Weight by tissue volume
    total_vol = 0.0
    weighted_fut = 0.0

    for (tissue, vol) in TISSUE_VOLUMES
        if tissue in [:plasma, :rbc]
            continue
        end

        comp = get(TISSUE_COMPOSITION, tissue, nothing)
        if comp === nothing
            continue
        end

        # Estimate fut for this tissue
        tissue_albumin_factor = comp.fw / 0.94
        fut = 1 - (1 - mol.fup) * tissue_albumin_factor * 0.5
        fut = clamp(fut, 0.01, 1.0)

        weighted_fut += vol * fut
        total_vol += vol
    end

    return weighted_fut / total_vol
end

"""
Simplified Øie-Tozer prediction using fup and estimated fut
"""
function OieTozerVdss(fup::Float64, logD74::Float64; body_weight::Float64=70.0)
    # Estimate fut from lipophilicity
    # Higher logD → more tissue binding → lower fut
    fut = 1 / (1 + 0.5 * 10^(logD74 - 1))
    fut = clamp(fut, 0.01, 0.9)

    # Standard volumes for 70kg human
    Vp = 3.0   # Plasma volume (L)
    Ve = 12.0  # Extracellular volume (L)
    Vr = 27.0  # Remaining tissue volume (L)

    # Øie-Tozer equation
    # Vdss = Vp + Ve × (fup/fut_e) + Vr × (fup/fut_r)
    # Simplified: assume fut_e ≈ fut_r ≈ fut
    Vdss = Vp + Ve * (fup / fut) + Vr * (fup / fut)

    # Scale to body weight and convert to L/kg
    scale = body_weight / 70.0
    Vdss_per_kg = (Vdss * scale) / body_weight

    return Vdss_per_kg
end

end # module
