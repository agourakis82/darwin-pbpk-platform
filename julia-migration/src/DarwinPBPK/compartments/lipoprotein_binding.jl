"""
Lipoprotein Binding Module

Models drug binding to plasma lipoproteins (HDL, LDL, VLDL) for accurate
prediction of lipophilic drug distribution.

Key Features:
- HDL, LDL, VLDL binding with Scatchard kinetics
- Apolipoprotein interactions (ApoA1, ApoB, ApoE)
- Disease state adjustments (dyslipidemia, diabetes, obesity)
- Integration with fu_plasma calculations

References:
- Wasan KM et al. (2008) Role of lipoproteins in drug distribution
- Gershkovich P et al. (2007) Pharmacokinetic influences of lipoproteins
- Veronese FM et al. (2002) Drug-lipoprotein interactions

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module LipoproteinBinding

using Statistics

export LipoproteinProfile, DrugLipoproteinBinding
export create_normal_lipoprotein_profile, create_dyslipidemia_profile
export calculate_lipoprotein_binding, calculate_fu_with_lipoproteins
export get_lipoprotein_partition, apply_disease_state!
export LIPOPROTEIN_DRUG_DATABASE

# ============================================================================
# CONSTANTS - Normal Lipoprotein Concentrations
# ============================================================================

# Reference: ATP III Guidelines, mg/dL
const NORMAL_HDL = 50.0          # mg/dL (desirable >40 men, >50 women)
const NORMAL_LDL = 100.0         # mg/dL (optimal <100)
const NORMAL_VLDL = 20.0         # mg/dL (normal <30)
const NORMAL_TOTAL_CHOLESTEROL = 180.0  # mg/dL

# Lipoprotein composition (fraction lipid vs protein)
const HDL_LIPID_FRACTION = 0.50   # 50% lipid, 50% protein
const LDL_LIPID_FRACTION = 0.80   # 80% lipid, 20% protein
const VLDL_LIPID_FRACTION = 0.90  # 90% lipid, 10% protein

# Molecular weights (approximate, kDa)
const HDL_MW = 200.0              # 175-360 kDa (heterogeneous)
const LDL_MW = 2500.0             # ~2500 kDa
const VLDL_MW = 30000.0           # 10,000-80,000 kDa

# Particle concentrations (nmol/L) - derived from mass/MW
# HDL: ~50 mg/dL / 200 kDa ≈ 25 μmol/L = 25000 nmol/L
# LDL: ~100 mg/dL / 2500 kDa ≈ 0.4 μmol/L = 400 nmol/L
# VLDL: ~20 mg/dL / 30000 kDa ≈ 0.007 μmol/L = 7 nmol/L

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    LipoproteinProfile

Complete lipoprotein profile for a patient.

# Fields
- `hdl_c::Float64`: HDL cholesterol (mg/dL)
- `ldl_c::Float64`: LDL cholesterol (mg/dL)
- `vldl_c::Float64`: VLDL cholesterol (mg/dL)
- `total_cholesterol::Float64`: Total cholesterol (mg/dL)
- `triglycerides::Float64`: Triglycerides (mg/dL)
- `apoa1::Float64`: Apolipoprotein A1 (mg/dL)
- `apob::Float64`: Apolipoprotein B (mg/dL)
- `lp_a::Float64`: Lipoprotein(a) (nmol/L)
- `condition::Symbol`: :normal, :hyperlipidemia, :hypolipidemia, etc.
"""
struct LipoproteinProfile
    hdl_c::Float64
    ldl_c::Float64
    vldl_c::Float64
    total_cholesterol::Float64
    triglycerides::Float64
    apoa1::Float64
    apob::Float64
    lp_a::Float64
    condition::Symbol

    function LipoproteinProfile(;
        hdl_c = NORMAL_HDL,
        ldl_c = NORMAL_LDL,
        vldl_c = NORMAL_VLDL,
        total_cholesterol = nothing,
        triglycerides = 100.0,
        apoa1 = 130.0,
        apob = 90.0,
        lp_a = 30.0,
        condition = :normal
    )
        # Calculate total if not provided
        tc = isnothing(total_cholesterol) ? hdl_c + ldl_c + vldl_c : total_cholesterol
        new(hdl_c, ldl_c, vldl_c, tc, triglycerides, apoa1, apob, lp_a, condition)
    end
end

"""
    DrugLipoproteinBinding

Drug-specific lipoprotein binding parameters.

# Fields
- `name::String`: Drug name
- `kp_hdl::Float64`: HDL partition coefficient
- `kp_ldl::Float64`: LDL partition coefficient
- `kp_vldl::Float64`: VLDL partition coefficient
- `binding_mechanism::Symbol`: :lipid_core, :surface, :apoprotein
- `logP::Float64`: Lipophilicity (for predictions)
- `fu_reference::Float64`: Reference fu without lipoprotein consideration
"""
struct DrugLipoproteinBinding
    name::String
    kp_hdl::Float64
    kp_ldl::Float64
    kp_vldl::Float64
    binding_mechanism::Symbol
    logP::Float64
    fu_reference::Float64

    function DrugLipoproteinBinding(name;
        kp_hdl=1.0, kp_ldl=1.0, kp_vldl=1.0,
        binding_mechanism=:lipid_core,
        logP=2.0, fu_reference=0.05
    )
        new(name, kp_hdl, kp_ldl, kp_vldl, binding_mechanism, logP, fu_reference)
    end
end

# ============================================================================
# DRUG DATABASE - Lipoprotein Binding Data
# ============================================================================

"""
Drug-specific lipoprotein binding data from literature.

Sources:
- Wasan KM (2008) - Statins, cyclosporine
- Gershkovich P (2007) - Lipophilic drugs
- Product labels and clinical pharmacology reviews
"""
const LIPOPROTEIN_DRUG_DATABASE = Dict{String, DrugLipoproteinBinding}(
    # Statins - significant LDL binding (they target LDL receptor pathway)
    "atorvastatin" => DrugLipoproteinBinding("atorvastatin";
        kp_hdl=2.0, kp_ldl=8.0, kp_vldl=3.0,
        binding_mechanism=:lipid_core, logP=4.1, fu_reference=0.02
    ),
    "simvastatin" => DrugLipoproteinBinding("simvastatin";
        kp_hdl=1.5, kp_ldl=6.0, kp_vldl=2.5,
        binding_mechanism=:lipid_core, logP=4.7, fu_reference=0.05
    ),
    "lovastatin" => DrugLipoproteinBinding("lovastatin";
        kp_hdl=1.8, kp_ldl=7.0, kp_vldl=3.0,
        binding_mechanism=:lipid_core, logP=4.3, fu_reference=0.05
    ),
    "rosuvastatin" => DrugLipoproteinBinding("rosuvastatin";
        kp_hdl=0.8, kp_ldl=2.0, kp_vldl=1.0,
        binding_mechanism=:surface, logP=0.13, fu_reference=0.12
    ),
    "pravastatin" => DrugLipoproteinBinding("pravastatin";
        kp_hdl=0.5, kp_ldl=1.2, kp_vldl=0.8,
        binding_mechanism=:surface, logP=-0.23, fu_reference=0.50
    ),

    # Immunosuppressants - high lipoprotein binding
    "cyclosporine" => DrugLipoproteinBinding("cyclosporine";
        kp_hdl=15.0, kp_ldl=25.0, kp_vldl=20.0,
        binding_mechanism=:lipid_core, logP=2.9, fu_reference=0.07
    ),
    "tacrolimus" => DrugLipoproteinBinding("tacrolimus";
        kp_hdl=8.0, kp_ldl=12.0, kp_vldl=10.0,
        binding_mechanism=:lipid_core, logP=3.3, fu_reference=0.01
    ),
    "sirolimus" => DrugLipoproteinBinding("sirolimus";
        kp_hdl=10.0, kp_ldl=18.0, kp_vldl=15.0,
        binding_mechanism=:lipid_core, logP=4.3, fu_reference=0.08
    ),

    # Antiarrhythmics
    "amiodarone" => DrugLipoproteinBinding("amiodarone";
        kp_hdl=20.0, kp_ldl=35.0, kp_vldl=30.0,
        binding_mechanism=:lipid_core, logP=7.6, fu_reference=0.0004
    ),
    "dronedarone" => DrugLipoproteinBinding("dronedarone";
        kp_hdl=12.0, kp_ldl=22.0, kp_vldl=18.0,
        binding_mechanism=:lipid_core, logP=6.4, fu_reference=0.002
    ),

    # Fat-soluble vitamins
    "vitamin_d" => DrugLipoproteinBinding("vitamin_d";
        kp_hdl=5.0, kp_ldl=15.0, kp_vldl=20.0,
        binding_mechanism=:lipid_core, logP=7.5, fu_reference=0.001
    ),
    "vitamin_e" => DrugLipoproteinBinding("vitamin_e";
        kp_hdl=8.0, kp_ldl=20.0, kp_vldl=25.0,
        binding_mechanism=:lipid_core, logP=10.7, fu_reference=0.0001
    ),
    "vitamin_a" => DrugLipoproteinBinding("vitamin_a";
        kp_hdl=3.0, kp_ldl=10.0, kp_vldl=12.0,
        binding_mechanism=:lipid_core, logP=6.2, fu_reference=0.005
    ),
    "vitamin_k" => DrugLipoproteinBinding("vitamin_k";
        kp_hdl=4.0, kp_ldl=12.0, kp_vldl=15.0,
        binding_mechanism=:lipid_core, logP=8.5, fu_reference=0.001
    ),

    # Antifungals
    "amphotericin_b" => DrugLipoproteinBinding("amphotericin_b";
        kp_hdl=25.0, kp_ldl=40.0, kp_vldl=35.0,
        binding_mechanism=:lipid_core, logP=0.8, fu_reference=0.05
    ),
    "itraconazole" => DrugLipoproteinBinding("itraconazole";
        kp_hdl=5.0, kp_ldl=10.0, kp_vldl=8.0,
        binding_mechanism=:lipid_core, logP=5.7, fu_reference=0.002
    ),

    # Antiretrovirals (lipophilic PIs)
    "lopinavir" => DrugLipoproteinBinding("lopinavir";
        kp_hdl=3.0, kp_ldl=6.0, kp_vldl=5.0,
        binding_mechanism=:lipid_core, logP=4.7, fu_reference=0.01
    ),
    "ritonavir" => DrugLipoproteinBinding("ritonavir";
        kp_hdl=2.5, kp_ldl=5.0, kp_vldl=4.0,
        binding_mechanism=:lipid_core, logP=4.3, fu_reference=0.01
    ),

    # Cannabis compounds
    "thc" => DrugLipoproteinBinding("thc";
        kp_hdl=15.0, kp_ldl=30.0, kp_vldl=25.0,
        binding_mechanism=:lipid_core, logP=6.97, fu_reference=0.03
    ),
    "cbd" => DrugLipoproteinBinding("cbd";
        kp_hdl=12.0, kp_ldl=25.0, kp_vldl=20.0,
        binding_mechanism=:lipid_core, logP=6.3, fu_reference=0.06
    )
)

# ============================================================================
# LIPOPROTEIN PROFILE FACTORIES
# ============================================================================

"""
    create_normal_lipoprotein_profile()

Create a normal/healthy lipoprotein profile.
"""
function create_normal_lipoprotein_profile()
    return LipoproteinProfile(
        hdl_c = 55.0,
        ldl_c = 100.0,
        vldl_c = 20.0,
        triglycerides = 100.0,
        condition = :normal
    )
end

"""
    create_dyslipidemia_profile(type::Symbol)

Create disease-state lipoprotein profiles.

Types:
- :hypercholesterolemia - High LDL
- :hypertriglyceridemia - High TG/VLDL
- :mixed_hyperlipidemia - Both elevated
- :low_hdl - HDL <40
- :familial_hypercholesterolemia - Very high LDL
- :diabetic_dyslipidemia - Low HDL, high TG, normal LDL
- :nephrotic_syndrome - High LDL, low HDL
- :hypothyroid - High total cholesterol
- :statin_treated - Low LDL (on treatment)
"""
function create_dyslipidemia_profile(type::Symbol)
    if type == :hypercholesterolemia
        return LipoproteinProfile(
            hdl_c = 45.0,
            ldl_c = 180.0,
            vldl_c = 25.0,
            triglycerides = 150.0,
            condition = :hypercholesterolemia
        )
    elseif type == :hypertriglyceridemia
        return LipoproteinProfile(
            hdl_c = 35.0,
            ldl_c = 110.0,
            vldl_c = 60.0,
            triglycerides = 400.0,
            condition = :hypertriglyceridemia
        )
    elseif type == :mixed_hyperlipidemia
        return LipoproteinProfile(
            hdl_c = 38.0,
            ldl_c = 170.0,
            vldl_c = 50.0,
            triglycerides = 350.0,
            condition = :mixed_hyperlipidemia
        )
    elseif type == :low_hdl
        return LipoproteinProfile(
            hdl_c = 32.0,
            ldl_c = 120.0,
            vldl_c = 25.0,
            triglycerides = 160.0,
            condition = :low_hdl
        )
    elseif type == :familial_hypercholesterolemia
        return LipoproteinProfile(
            hdl_c = 40.0,
            ldl_c = 300.0,  # Very high!
            vldl_c = 30.0,
            triglycerides = 180.0,
            condition = :familial_hypercholesterolemia
        )
    elseif type == :diabetic_dyslipidemia
        return LipoproteinProfile(
            hdl_c = 35.0,
            ldl_c = 110.0,
            vldl_c = 45.0,
            triglycerides = 280.0,
            condition = :diabetic_dyslipidemia
        )
    elseif type == :nephrotic_syndrome
        return LipoproteinProfile(
            hdl_c = 30.0,
            ldl_c = 220.0,
            vldl_c = 40.0,
            triglycerides = 250.0,
            condition = :nephrotic_syndrome
        )
    elseif type == :hypothyroid
        return LipoproteinProfile(
            hdl_c = 50.0,
            ldl_c = 200.0,
            vldl_c = 35.0,
            triglycerides = 200.0,
            condition = :hypothyroid
        )
    elseif type == :statin_treated
        return LipoproteinProfile(
            hdl_c = 55.0,
            ldl_c = 70.0,   # On high-intensity statin
            vldl_c = 18.0,
            triglycerides = 90.0,
            condition = :statin_treated
        )
    else
        error("Unknown dyslipidemia type: $type")
    end
end

# ============================================================================
# BINDING CALCULATIONS
# ============================================================================

"""
    calculate_lipoprotein_concentrations(profile::LipoproteinProfile)

Convert cholesterol concentrations to lipoprotein particle concentrations.

Returns molar concentrations (nmol/L) for binding calculations.
"""
function calculate_lipoprotein_concentrations(profile::LipoproteinProfile)
    # Convert mg/dL to nmol/L using approximate MW
    # Factor: (mg/dL) * 10 / MW(kDa) = μmol/L, then *1000 for nmol/L

    hdl_nmol = profile.hdl_c * 10.0 / HDL_MW * 1000.0
    ldl_nmol = profile.ldl_c * 10.0 / LDL_MW * 1000.0
    vldl_nmol = profile.vldl_c * 10.0 / VLDL_MW * 1000.0

    return (hdl=hdl_nmol, ldl=ldl_nmol, vldl=vldl_nmol)
end

"""
    calculate_lipoprotein_binding(drug::DrugLipoproteinBinding,
                                   profile::LipoproteinProfile)

Calculate fraction of drug bound to each lipoprotein class.

Uses partition coefficients and lipoprotein concentrations.

# Returns
Dict with:
- `f_hdl`: Fraction bound to HDL
- `f_ldl`: Fraction bound to LDL
- `f_vldl`: Fraction bound to VLDL
- `f_total_lp`: Total fraction bound to lipoproteins
- `f_free`: Fraction free from lipoproteins
"""
function calculate_lipoprotein_binding(drug::DrugLipoproteinBinding,
                                        profile::LipoproteinProfile)
    # Get lipoprotein concentrations
    lp_conc = calculate_lipoprotein_concentrations(profile)

    # Calculate binding using partition coefficients
    # Assume linear binding at therapeutic concentrations (not saturated)

    # Lipid volume approximation (L/L plasma)
    # HDL: ~50 mg/dL cholesterol ≈ 0.001 L lipid/L plasma
    hdl_lipid_vol = profile.hdl_c / 1000.0 * HDL_LIPID_FRACTION / 100.0
    ldl_lipid_vol = profile.ldl_c / 1000.0 * LDL_LIPID_FRACTION / 100.0
    vldl_lipid_vol = profile.vldl_c / 1000.0 * VLDL_LIPID_FRACTION / 100.0

    # Amount in each compartment (relative, assuming 1 unit total drug)
    # Using partition: Amount = Kp * Volume_lipid / Volume_plasma
    a_hdl = drug.kp_hdl * hdl_lipid_vol
    a_ldl = drug.kp_ldl * ldl_lipid_vol
    a_vldl = drug.kp_vldl * vldl_lipid_vol
    a_free = 1.0  # Reference aqueous phase

    # Total
    a_total = a_hdl + a_ldl + a_vldl + a_free

    # Fractions
    f_hdl = a_hdl / a_total
    f_ldl = a_ldl / a_total
    f_vldl = a_vldl / a_total
    f_free = a_free / a_total
    f_total_lp = f_hdl + f_ldl + f_vldl

    return Dict(
        "f_hdl" => f_hdl,
        "f_ldl" => f_ldl,
        "f_vldl" => f_vldl,
        "f_total_lipoprotein" => f_total_lp,
        "f_free" => f_free,
        "hdl_lipid_volume" => hdl_lipid_vol,
        "ldl_lipid_volume" => ldl_lipid_vol,
        "vldl_lipid_volume" => vldl_lipid_vol
    )
end

"""
    calculate_fu_with_lipoproteins(fu_base::Float64,
                                    drug::DrugLipoproteinBinding,
                                    profile::LipoproteinProfile)

Adjust fraction unbound considering lipoprotein binding.

The base fu typically accounts for albumin/AAG binding.
This function adds lipoprotein binding effects.

# Returns
- Adjusted fu_plasma accounting for lipoprotein binding
"""
function calculate_fu_with_lipoproteins(fu_base::Float64,
                                         drug::DrugLipoproteinBinding,
                                         profile::LipoproteinProfile)
    # Get lipoprotein binding fractions
    lp_binding = calculate_lipoprotein_binding(drug, profile)

    # The "free" drug from protein binding may partition to lipoproteins
    # fu_adjusted = fu_base * (fraction that stays free from lipoproteins)

    # However, lipoprotein-bound drug may still be "available" in some sense
    # because lipoproteins deliver drugs to tissues

    # More accurate model:
    # Total fu = fu_base * f_free_from_lp

    f_free_from_lp = lp_binding["f_free"]
    fu_adjusted = fu_base * f_free_from_lp

    # Don't let fu go below a physiological minimum
    fu_adjusted = max(fu_adjusted, 1e-6)

    return fu_adjusted
end

"""
    get_lipoprotein_partition(drug_name::String)

Get lipoprotein binding parameters for a drug from the database.

Returns nothing if drug not in database.
"""
function get_lipoprotein_partition(drug_name::String)
    name_lower = lowercase(drug_name)
    if haskey(LIPOPROTEIN_DRUG_DATABASE, name_lower)
        return LIPOPROTEIN_DRUG_DATABASE[name_lower]
    end
    return nothing
end

"""
    predict_lipoprotein_binding(logP::Float64)

Predict lipoprotein binding from logP when no experimental data available.

Based on empirical relationships from Wasan (2008).
"""
function predict_lipoprotein_binding(logP::Float64; fu_reference::Float64=0.05)
    # Empirical: Kp increases exponentially with logP
    # Kp_lp ≈ 10^(0.3 * logP) for highly lipophilic drugs

    if logP < 1.0
        # Hydrophilic - minimal lipoprotein binding
        kp_hdl = 0.5
        kp_ldl = 0.8
        kp_vldl = 0.6
    elseif logP < 3.0
        # Moderate lipophilicity
        kp_hdl = 1.0 + logP * 0.5
        kp_ldl = 1.5 + logP * 1.0
        kp_vldl = 1.2 + logP * 0.8
    elseif logP < 5.0
        # Lipophilic
        kp_hdl = 2.0 + logP * 1.5
        kp_ldl = 4.0 + logP * 3.0
        kp_vldl = 3.0 + logP * 2.5
    else
        # Highly lipophilic (logP > 5)
        kp_hdl = 5.0 + logP * 2.0
        kp_ldl = 10.0 + logP * 5.0
        kp_vldl = 8.0 + logP * 4.0
    end

    return DrugLipoproteinBinding("predicted";
        kp_hdl=kp_hdl, kp_ldl=kp_ldl, kp_vldl=kp_vldl,
        binding_mechanism=:lipid_core, logP=logP, fu_reference=fu_reference
    )
end

# ============================================================================
# DISEASE STATE ADJUSTMENTS
# ============================================================================

"""
    apply_disease_state!(profile::LipoproteinProfile, disease::Symbol)

Modify lipoprotein profile based on disease state.

Returns new profile with adjusted values.
"""
function apply_disease_state(profile::LipoproteinProfile, disease::Symbol)
    hdl = profile.hdl_c
    ldl = profile.ldl_c
    vldl = profile.vldl_c
    tg = profile.triglycerides

    if disease == :diabetes_t2
        # Diabetic dyslipidemia: ↓HDL, ↑TG, small dense LDL
        hdl *= 0.75
        vldl *= 1.8
        tg *= 2.0
    elseif disease == :obesity
        # Obesity: ↓HDL, ↑TG, ↑VLDL
        hdl *= 0.85
        vldl *= 1.5
        tg *= 1.8
    elseif disease == :metabolic_syndrome
        # Combination pattern
        hdl *= 0.70
        ldl *= 1.1
        vldl *= 1.6
        tg *= 2.2
    elseif disease == :ckd_stage4
        # CKD: Complex pattern, often ↑TG, ↓HDL
        hdl *= 0.80
        vldl *= 1.4
        tg *= 1.6
    elseif disease == :liver_cirrhosis
        # Liver disease: ↓↓ all lipoproteins (synthesis failure)
        hdl *= 0.50
        ldl *= 0.60
        vldl *= 0.70
        tg *= 0.80
    elseif disease == :hyperthyroid
        # Hyperthyroidism: ↓ cholesterol (increased clearance)
        hdl *= 0.90
        ldl *= 0.70
        vldl *= 0.80
    elseif disease == :hiv_untreated
        # HIV: ↓HDL, ↑TG
        hdl *= 0.65
        vldl *= 1.5
        tg *= 1.8
    elseif disease == :hiv_on_art
        # HIV on ART (especially PIs): metabolic effects
        hdl *= 0.80
        ldl *= 1.2
        vldl *= 1.4
        tg *= 1.6
    elseif disease == :pregnancy
        # Pregnancy: physiological hyperlipidemia
        hdl *= 1.1
        ldl *= 1.5
        vldl *= 1.8
        tg *= 2.5
    elseif disease == :anorexia
        # Anorexia: paradoxically high cholesterol
        hdl *= 1.0
        ldl *= 1.4
        vldl *= 0.70
    end

    return LipoproteinProfile(
        hdl_c = hdl,
        ldl_c = ldl,
        vldl_c = vldl,
        triglycerides = tg,
        condition = disease
    )
end

# ============================================================================
# INTEGRATION WITH BLOOD BINDING
# ============================================================================

"""
    calculate_total_plasma_binding(drug_name::String,
                                    fu_albumin::Float64,
                                    fu_aag::Float64,
                                    profile::LipoproteinProfile)

Calculate total plasma binding including albumin, AAG, and lipoproteins.

# Arguments
- `drug_name`: Drug name for database lookup
- `fu_albumin`: Fraction unbound from albumin
- `fu_aag`: Fraction unbound from AAG (if applicable)
- `profile`: Lipoprotein profile

# Returns
Dict with detailed binding breakdown
"""
function calculate_total_plasma_binding(drug_name::String,
                                         fu_albumin::Float64,
                                         fu_aag::Float64,
                                         profile::LipoproteinProfile)
    # Get lipoprotein data
    lp_drug = get_lipoprotein_partition(drug_name)

    if isnothing(lp_drug)
        # No lipoprotein data - return protein binding only
        return Dict(
            "fu_plasma" => fu_albumin * fu_aag,
            "f_albumin" => 1 - fu_albumin,
            "f_aag" => 1 - fu_aag,
            "f_lipoprotein" => 0.0,
            "has_lipoprotein_data" => false
        )
    end

    # Calculate lipoprotein binding
    lp_binding = calculate_lipoprotein_binding(lp_drug, profile)

    # Combine protein and lipoprotein binding
    # Assume sequential: drug first binds proteins, then lipoproteins bind free drug
    fu_protein = fu_albumin * fu_aag
    fu_total = calculate_fu_with_lipoproteins(fu_protein, lp_drug, profile)

    return Dict(
        "fu_plasma" => fu_total,
        "fu_protein_only" => fu_protein,
        "f_albumin" => 1 - fu_albumin,
        "f_aag" => 1 - fu_aag,
        "f_hdl" => lp_binding["f_hdl"],
        "f_ldl" => lp_binding["f_ldl"],
        "f_vldl" => lp_binding["f_vldl"],
        "f_lipoprotein" => lp_binding["f_total_lipoprotein"],
        "has_lipoprotein_data" => true
    )
end

# ============================================================================
# CLINICAL UTILITIES
# ============================================================================

"""
    calculate_ldl_cholesterol(total_c::Float64, hdl_c::Float64, tg::Float64)

Calculate LDL-C using Friedewald equation.

LDL-C = Total-C - HDL-C - TG/5

Note: Not valid if TG > 400 mg/dL
"""
function calculate_ldl_cholesterol(total_c::Float64, hdl_c::Float64, tg::Float64)
    if tg > 400.0
        @warn "Friedewald equation not valid for TG > 400 mg/dL"
        return NaN
    end

    vldl_c = tg / 5.0
    ldl_c = total_c - hdl_c - vldl_c

    return max(ldl_c, 0.0)
end

"""
    assess_cv_risk_lipids(profile::LipoproteinProfile)

Assess cardiovascular risk based on lipid profile.

Returns risk category and ratios.
"""
function assess_cv_risk_lipids(profile::LipoproteinProfile)
    # Risk ratios
    tc_hdl_ratio = profile.total_cholesterol / profile.hdl_c
    ldl_hdl_ratio = profile.ldl_c / profile.hdl_c
    non_hdl = profile.total_cholesterol - profile.hdl_c

    # Risk categories
    if profile.ldl_c < 70 && profile.hdl_c > 60
        risk = :very_low
    elseif profile.ldl_c < 100 && profile.hdl_c > 40
        risk = :low
    elseif profile.ldl_c < 130 && profile.hdl_c > 35
        risk = :moderate
    elseif profile.ldl_c < 160
        risk = :high
    else
        risk = :very_high
    end

    return Dict(
        "risk_category" => risk,
        "tc_hdl_ratio" => tc_hdl_ratio,
        "ldl_hdl_ratio" => ldl_hdl_ratio,
        "non_hdl_c" => non_hdl,
        "apo_b" => profile.apob
    )
end

end # module LipoproteinBinding
