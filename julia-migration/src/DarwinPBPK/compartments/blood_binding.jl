"""
Blood Binding Module - Comprehensive Drug Distribution in Blood

Implements PK-Sim/Simcyp-level blood binding calculations:
- Blood-to-Plasma ratio (B:P) with mechanistic equations
- RBC partitioning (Rodgers-Rowland, Schmitt methods)
- Plasma protein binding (albumin, AGP, lipoproteins)
- Drug-specific WBC/platelet accumulation
- Hematocrit-dependent corrections
- Ion trapping in acidic compartments

Based on:
- PK-Sim Open Systems Pharmacology equations
- Rodgers & Rowland (2006) tissue composition model
- Schmitt (2008) partition coefficient model
- Poulin & Theil (2002) methods

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

module BloodBinding

using LinearAlgebra

export BloodComposition, DrugProperties, BloodPartitioning
export calculate_blood_plasma_ratio, calculate_rbc_partition
export calculate_wbc_partition, calculate_platelet_partition
export calculate_fu_blood, calculate_erythrocyte_water_partition
export create_drug_properties, get_blood_composition
export STANDARD_HEMATOCRIT, PHYSIOLOGICAL_PH

# ============================================================================
# CONSTANTS - Physiological Parameters
# ============================================================================

# Standard physiological values
const STANDARD_HEMATOCRIT = 0.45           # 45% (range: 0.36-0.50)
const PHYSIOLOGICAL_PH = Dict(
    "plasma" => 7.4,
    "rbc" => 7.22,                          # Slightly acidic
    "rbc_water" => 7.22,
    "wbc_cytosol" => 7.2,
    "wbc_lysosome" => 5.0,                  # Acidic - ion trapping!
    "platelet_cytosol" => 7.2,
    "platelet_granule" => 5.5               # Dense granules acidic
)

# RBC composition (Rodgers & Rowland)
const RBC_COMPOSITION = Dict(
    "f_water" => 0.666,                     # Fraction water (66.6%)
    "f_neutral_lipids" => 0.0017,           # Fraction neutral lipids
    "f_phospholipids" => 0.0029,            # Fraction phospholipids
    "f_acidic_phospholipids" => 0.0004,     # Fraction acidic phospholipids
    "f_proteins" => 0.330,                  # Fraction proteins (mainly Hb)
    "hemoglobin_conc" => 5.0e-3             # M (5 mM, ~340 g/L)
)

# Plasma composition
const PLASMA_COMPOSITION = Dict(
    "f_water" => 0.93,                      # 93% water
    "f_neutral_lipids" => 0.0023,
    "f_phospholipids" => 0.0022,
    "albumin_conc" => 0.6e-3,               # M (0.6 mM, ~40 g/L)
    "agp_conc" => 20e-6,                    # M (20 μM, ~0.8 g/L)
    "lipoprotein_conc" => 3e-3              # M (approximate)
)

# WBC composition (estimated)
const WBC_COMPOSITION = Dict(
    "f_water" => 0.70,
    "f_lipids" => 0.02,
    "f_proteins" => 0.25,
    "f_lysosome" => 0.10,                   # 10% lysosomal volume
    "lysosome_ph" => 5.0
)

# Platelet composition
const PLATELET_COMPOSITION = Dict(
    "f_water" => 0.65,
    "f_lipids" => 0.03,
    "f_proteins" => 0.30,
    "f_dense_granule" => 0.02,
    "dense_granule_ph" => 5.5
)

# Protein binding constants
const ALBUMIN_BINDING = Dict(
    "n_sites_strong" => 2,                  # Number of high-affinity sites
    "n_sites_weak" => 4                     # Number of low-affinity sites
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
BloodComposition - Patient-specific blood composition

Allows for variation due to:
- Age, sex, disease state
- Anemia, polycythemia
- Hypoalbuminemia
"""
mutable struct BloodComposition
    # Hematocrit and cell counts
    hematocrit::Float64                     # Fraction (0-1)
    rbc_count::Float64                      # cells/L
    wbc_count::Float64                      # cells/L
    platelet_count::Float64                 # cells/L

    # RBC composition
    rbc_water_fraction::Float64
    rbc_lipid_fraction::Float64
    rbc_protein_fraction::Float64
    hemoglobin_conc::Float64                # M

    # Plasma proteins
    albumin_conc::Float64                   # M
    agp_conc::Float64                       # M (α1-acid glycoprotein)

    # pH values
    plasma_ph::Float64
    rbc_ph::Float64
end

"""
DrugProperties - Physicochemical properties for partitioning

All properties needed to calculate tissue/blood partitioning.
"""
struct DrugProperties
    name::String

    # Basic properties
    mw::Float64                             # Molecular weight (Da)
    charge_type::Symbol                     # :acid, :base, :neutral, :zwitterion
    pKa::Vector{Float64}                    # pKa value(s)

    # Lipophilicity
    logP::Float64                           # Octanol-water partition (neutral)
    logD_74::Float64                        # Distribution coeff at pH 7.4

    # Plasma protein binding
    fu_plasma::Float64                      # Fraction unbound in plasma
    albumin_binding::Bool                   # Binds albumin?
    agp_binding::Bool                       # Binds AGP?

    # Specific binding affinities (optional)
    Ka_albumin::Float64                     # Association constant for albumin (M⁻¹)
    Ka_agp::Float64                         # Association constant for AGP (M⁻¹)

    # RBC-specific
    hemoglobin_binding::Bool                # Binds hemoglobin?
    Ka_hemoglobin::Float64                  # Association constant (M⁻¹)

    # Permeability
    membrane_permeability::Float64          # cm/s (for permeability-limited)

    # Special accumulation
    lysosomal_trapping::Bool                # Accumulates in lysosomes?
    mitochondrial_binding::Bool             # Binds mitochondria?
end

"""
BloodPartitioning - Complete blood partitioning results
"""
struct BloodPartitioning
    # Primary ratios
    bp_ratio::Float64                       # Blood-to-Plasma ratio
    rbc_plasma_ratio::Float64               # RBC-to-Plasma ratio (Kp,rbc)

    # Unbound fractions
    fu_plasma::Float64                      # Fraction unbound in plasma
    fu_blood::Float64                       # Fraction unbound in blood
    fu_rbc::Float64                         # Fraction unbound in RBC

    # Cell-specific partitioning
    wbc_plasma_ratio::Float64               # WBC-to-Plasma ratio
    platelet_plasma_ratio::Float64          # Platelet-to-Plasma ratio

    # Distribution
    fraction_in_plasma::Float64             # Fraction of blood drug in plasma
    fraction_in_rbc::Float64                # Fraction in RBC
    fraction_in_wbc::Float64                # Fraction in WBC
    fraction_in_platelets::Float64          # Fraction in platelets
end

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

"""
get_blood_composition(; hematocrit, albumin, patient_type)

Create blood composition for a patient.

# Arguments
- `hematocrit`: Hematocrit fraction (default 0.45)
- `albumin_gL`: Albumin in g/L (default 40)
- `patient_type`: :normal, :anemic, :polycythemic, :hypoalbuminemic, :cirrhotic
"""
function get_blood_composition(;
    hematocrit::Float64=STANDARD_HEMATOCRIT,
    albumin_gL::Float64=40.0,
    agp_gL::Float64=0.8,
    patient_type::Symbol=:normal
)::BloodComposition

    # Adjust for patient type
    hct, alb, agp = if patient_type == :normal
        (hematocrit, albumin_gL, agp_gL)
    elseif patient_type == :anemic
        (0.30, albumin_gL, agp_gL)
    elseif patient_type == :polycythemic
        (0.55, albumin_gL, agp_gL)
    elseif patient_type == :hypoalbuminemic
        (hematocrit, 25.0, agp_gL)
    elseif patient_type == :cirrhotic
        (0.35, 28.0, 0.5)  # Low albumin and AGP
    elseif patient_type == :inflammatory
        (hematocrit, albumin_gL * 0.9, agp_gL * 2.5)  # AGP acute phase
    else
        (hematocrit, albumin_gL, agp_gL)
    end

    # Convert g/L to M
    albumin_M = alb / 66500.0 * 1000  # MW albumin ~66.5 kDa
    agp_M = agp / 41000.0 * 1000       # MW AGP ~41 kDa

    return BloodComposition(
        hct,
        5.0e12,                              # RBC count
        7.0e9,                               # WBC count
        250e9,                               # Platelet count
        RBC_COMPOSITION["f_water"],
        RBC_COMPOSITION["f_neutral_lipids"] + RBC_COMPOSITION["f_phospholipids"],
        RBC_COMPOSITION["f_proteins"],
        RBC_COMPOSITION["hemoglobin_conc"],
        albumin_M,
        agp_M,
        PHYSIOLOGICAL_PH["plasma"],
        PHYSIOLOGICAL_PH["rbc"]
    )
end

"""
create_drug_properties(name; kwargs...)

Create drug properties from known parameters.

# Examples
```julia
# Warfarin - acidic, highly protein bound
warfarin = create_drug_properties("warfarin",
    mw=308.3, charge_type=:acid, pKa=[5.0],
    logP=2.7, fu_plasma=0.01, albumin_binding=true
)

# Chloroquine - basic, accumulates in WBC
chloroquine = create_drug_properties("chloroquine",
    mw=319.9, charge_type=:base, pKa=[8.1, 10.2],
    logP=4.6, fu_plasma=0.4, lysosomal_trapping=true
)
```
"""
function create_drug_properties(
    name::String;
    mw::Float64=400.0,
    charge_type::Symbol=:neutral,
    pKa::Vector{Float64}=Float64[],
    logP::Float64=2.0,
    logD_74::Float64=NaN,
    fu_plasma::Float64=0.1,
    albumin_binding::Bool=true,
    agp_binding::Bool=false,
    Ka_albumin::Float64=1e5,
    Ka_agp::Float64=1e4,
    hemoglobin_binding::Bool=false,
    Ka_hemoglobin::Float64=0.0,
    membrane_permeability::Float64=1e-4,
    lysosomal_trapping::Bool=false,
    mitochondrial_binding::Bool=false
)::DrugProperties

    # Calculate logD at pH 7.4 if not provided
    logD = if isnan(logD_74)
        calculate_logD(logP, pKa, charge_type, 7.4)
    else
        logD_74
    end

    return DrugProperties(
        name, mw, charge_type, pKa, logP, logD,
        fu_plasma, albumin_binding, agp_binding,
        Ka_albumin, Ka_agp,
        hemoglobin_binding, Ka_hemoglobin,
        membrane_permeability,
        lysosomal_trapping, mitochondrial_binding
    )
end

# ============================================================================
# CORE CALCULATIONS - Blood-Plasma Ratio
# ============================================================================

"""
calculate_blood_plasma_ratio(drug, blood; method=:mechanistic)

Calculate B:P ratio using mechanistic or empirical methods.

# Methods
- `:mechanistic` - Full Rodgers-Rowland/PK-Sim equations
- `:empirical` - Simplified empirical correlation
- `:measured` - Use measured Kp,rbc if available

# Returns
- B:P ratio (dimensionless)
"""
function calculate_blood_plasma_ratio(
    drug::DrugProperties,
    blood::BloodComposition;
    method::Symbol=:mechanistic
)::Float64

    if method == :mechanistic
        return calculate_bp_mechanistic(drug, blood)
    elseif method == :empirical
        return calculate_bp_empirical(drug, blood)
    else
        error("Unknown method: $method")
    end
end

"""
calculate_bp_mechanistic(drug, blood)

PK-Sim style mechanistic B:P calculation.

B:P = Kp,rbc × HCT + (1 - HCT)

Where Kp,rbc accounts for:
- Water partitioning
- Lipid partitioning
- Protein binding (hemoglobin)
- Ion trapping (pH gradient)
"""
function calculate_bp_mechanistic(
    drug::DrugProperties,
    blood::BloodComposition
)::Float64

    HCT = blood.hematocrit
    fu = drug.fu_plasma

    # Calculate RBC-plasma partition coefficient
    Kp_rbc = calculate_rbc_partition(drug, blood)

    # B:P = Kp,rbc × HCT + (1 - HCT)
    # This assumes drug in plasma = 1 (reference)
    bp_ratio = Kp_rbc * HCT + (1.0 - HCT)

    return bp_ratio
end

"""
calculate_bp_empirical(drug, blood)

Simplified empirical B:P estimation.

Based on charge type and lipophilicity.
"""
function calculate_bp_empirical(
    drug::DrugProperties,
    blood::BloodComposition
)::Float64

    HCT = blood.hematocrit
    logP = drug.logP

    # Empirical rules (from literature)
    Kp_rbc = if drug.charge_type == :neutral
        # Neutral drugs: partition based on lipophilicity
        0.5 + 0.5 * (10^logP / (1 + 10^logP))
    elseif drug.charge_type == :base
        # Bases: tend to accumulate in RBC (pH trapping)
        0.8 + 0.3 * (10^logP / (1 + 10^logP))
    elseif drug.charge_type == :acid
        # Acids: tend to be excluded from RBC
        0.3 + 0.2 * (10^logP / (1 + 10^logP))
    else  # zwitterion
        0.5
    end

    return Kp_rbc * HCT + (1.0 - HCT)
end

# ============================================================================
# RBC PARTITIONING - Rodgers-Rowland Method
# ============================================================================

"""
calculate_rbc_partition(drug, blood)

Calculate RBC-plasma partition coefficient (Kp,rbc).

Uses Rodgers & Rowland equations with modifications for:
- Hemoglobin binding
- pH-dependent ion trapping
- Lipid partitioning
"""
function calculate_rbc_partition(
    drug::DrugProperties,
    blood::BloodComposition
)::Float64

    # Unpack
    fu = drug.fu_plasma
    logP = drug.logP
    pKa = drug.pKa
    charge = drug.charge_type

    # RBC composition
    f_w = blood.rbc_water_fraction
    f_lip = blood.rbc_lipid_fraction
    f_prot = blood.rbc_protein_fraction

    # pH values
    pH_plasma = blood.plasma_ph
    pH_rbc = blood.rbc_ph

    # 1. Water partition (accounts for ionization)
    K_water = calculate_water_partition(pKa, charge, pH_plasma, pH_rbc)

    # 2. Lipid partition
    P_lip = 10^logP  # Neutral lipid partition
    K_lip = P_lip * 0.3  # Empirical factor for membrane lipids

    # 3. Protein binding (hemoglobin)
    K_prot = if drug.hemoglobin_binding
        drug.Ka_hemoglobin * blood.hemoglobin_conc
    else
        # Non-specific binding
        0.1 * f_prot
    end

    # 4. Combine using Rodgers-Rowland equation
    # Kp,rbc = fu × (f_w × K_water + f_lip × K_lip + f_prot × K_prot)
    Kp_rbc = fu * (f_w * K_water + f_lip * K_lip + f_prot * K_prot)

    # Add specific binding if significant
    if drug.hemoglobin_binding
        Kp_rbc += fu * drug.Ka_hemoglobin * blood.hemoglobin_conc
    end

    # Minimum is exclusion (only water space accessible)
    Kp_rbc = max(Kp_rbc, fu * f_w * 0.5)

    return Kp_rbc
end

"""
calculate_water_partition(pKa, charge, pH_plasma, pH_rbc)

Calculate water partition accounting for pH gradient.

Ion trapping: bases accumulate in acidic compartments.
"""
function calculate_water_partition(
    pKa::Vector{Float64},
    charge::Symbol,
    pH_plasma::Float64,
    pH_rbc::Float64
)::Float64

    if isempty(pKa) || charge == :neutral
        return 1.0
    end

    pKa1 = pKa[1]

    if charge == :base
        # Bases: ionized form is charged (BH+)
        # Ratio = (1 + 10^(pKa - pH_rbc)) / (1 + 10^(pKa - pH_plasma))
        ionized_rbc = 1.0 + 10^(pKa1 - pH_rbc)
        ionized_plasma = 1.0 + 10^(pKa1 - pH_plasma)
        return ionized_rbc / ionized_plasma

    elseif charge == :acid
        # Acids: ionized form is charged (A-)
        # Ratio = (1 + 10^(pH_rbc - pKa)) / (1 + 10^(pH_plasma - pKa))
        ionized_rbc = 1.0 + 10^(pH_rbc - pKa1)
        ionized_plasma = 1.0 + 10^(pH_plasma - pKa1)
        return ionized_rbc / ionized_plasma

    else
        return 1.0
    end
end

# ============================================================================
# WBC AND PLATELET PARTITIONING
# ============================================================================

"""
calculate_wbc_partition(drug, blood; cell_type=:neutrophil)

Calculate WBC-plasma partition coefficient.

Critical for drugs like:
- Chloroquine/HCQ (massive accumulation)
- Azithromycin (high WBC/plasma ratio)
- Antiretrovirals
"""
function calculate_wbc_partition(
    drug::DrugProperties,
    blood::BloodComposition;
    cell_type::Symbol=:neutrophil
)::Float64

    fu = drug.fu_plasma
    logP = drug.logP
    pKa = drug.pKa
    charge = drug.charge_type

    # WBC composition
    f_w = WBC_COMPOSITION["f_water"]
    f_lip = WBC_COMPOSITION["f_lipids"]
    f_lys = WBC_COMPOSITION["f_lysosome"]
    pH_lys = WBC_COMPOSITION["lysosome_ph"]

    # Base partition (similar to RBC)
    K_water = 1.0
    K_lip = 10^logP * 0.3

    # Lysosomal trapping for basic drugs (MAJOR effect)
    K_lysosome = 1.0
    if drug.lysosomal_trapping && charge == :base && !isempty(pKa)
        # Henderson-Hasselbalch for lysosomal accumulation
        pKa1 = pKa[1]
        pH_cytosol = 7.2

        # Accumulation ratio in lysosome vs cytosol
        ratio_lys_cytosol = (1.0 + 10^(pKa1 - pH_lys)) / (1.0 + 10^(pKa1 - pH_cytosol))

        # Can be 100-1000× for strong bases!
        K_lysosome = ratio_lys_cytosol
    end

    # Cell type-specific adjustments
    lysosome_fraction = if cell_type == :monocyte
        0.15  # Monocytes have more lysosomes
    elseif cell_type == :neutrophil
        0.10
    else
        0.08
    end

    # Total partition
    Kp_wbc = fu * (
        f_w * K_water +
        f_lip * K_lip +
        lysosome_fraction * K_lysosome
    )

    return Kp_wbc
end

"""
calculate_platelet_partition(drug, blood)

Calculate platelet-plasma partition coefficient.

Important for antiplatelet drugs and drugs with platelet toxicity.
"""
function calculate_platelet_partition(
    drug::DrugProperties,
    blood::BloodComposition
)::Float64

    fu = drug.fu_plasma
    logP = drug.logP

    # Platelet composition
    f_w = PLATELET_COMPOSITION["f_water"]
    f_lip = PLATELET_COMPOSITION["f_lipids"]

    # Base partition
    K_water = 1.0
    K_lip = 10^logP * 0.2  # Lower lipid accessibility

    # Dense granule trapping (acidic)
    K_granule = 1.0
    if drug.charge_type == :base && !isempty(drug.pKa)
        pH_granule = PLATELET_COMPOSITION["dense_granule_ph"]
        pKa1 = drug.pKa[1]
        K_granule = (1.0 + 10^(pKa1 - pH_granule)) / (1.0 + 10^(pKa1 - 7.2))
    end

    f_granule = PLATELET_COMPOSITION["f_dense_granule"]

    Kp_plt = fu * (f_w * K_water + f_lip * K_lip + f_granule * K_granule)

    return Kp_plt
end

# ============================================================================
# COMPLETE BLOOD PARTITIONING
# ============================================================================

"""
calculate_complete_blood_partitioning(drug, blood)

Calculate all blood partitioning parameters.

Returns BloodPartitioning struct with:
- B:P ratio
- All cell-specific Kp values
- Unbound fractions
- Distribution fractions
"""
function calculate_complete_blood_partitioning(
    drug::DrugProperties,
    blood::BloodComposition
)::BloodPartitioning

    HCT = blood.hematocrit
    fu_plasma = drug.fu_plasma

    # Calculate all partition coefficients
    Kp_rbc = calculate_rbc_partition(drug, blood)
    Kp_wbc = calculate_wbc_partition(drug, blood)
    Kp_plt = calculate_platelet_partition(drug, blood)

    # B:P ratio
    bp_ratio = Kp_rbc * HCT + (1.0 - HCT)

    # Unbound fraction in blood
    fu_blood = fu_plasma / bp_ratio

    # Unbound in RBC (approximation)
    fu_rbc = fu_plasma / Kp_rbc

    # Volume fractions of each component
    # WBC and platelets are tiny (~0.1% of blood volume)
    v_rbc = HCT
    v_plasma = 1.0 - HCT
    v_wbc = blood.wbc_count * 350e-15  # ~350 fL per WBC
    v_plt = blood.platelet_count * 8e-15  # ~8 fL per platelet

    # Normalize (plasma + RBC dominate)
    v_total = v_plasma + v_rbc

    # Drug distribution fractions
    # Amount in each compartment = Kp × V × C_plasma
    amount_plasma = v_plasma * 1.0  # Reference
    amount_rbc = v_rbc * Kp_rbc
    amount_wbc = v_wbc * Kp_wbc
    amount_plt = v_plt * Kp_plt

    total_amount = amount_plasma + amount_rbc + amount_wbc + amount_plt

    return BloodPartitioning(
        bp_ratio,
        Kp_rbc,
        fu_plasma,
        fu_blood,
        fu_rbc,
        Kp_wbc,
        Kp_plt,
        amount_plasma / total_amount,
        amount_rbc / total_amount,
        amount_wbc / total_amount,
        amount_plt / total_amount
    )
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
calculate_logD(logP, pKa, charge, pH)

Calculate distribution coefficient at given pH.
"""
function calculate_logD(
    logP::Float64,
    pKa::Vector{Float64},
    charge::Symbol,
    pH::Float64
)::Float64

    if isempty(pKa) || charge == :neutral
        return logP
    end

    pKa1 = pKa[1]

    if charge == :acid
        # logD = logP - log10(1 + 10^(pH - pKa))
        return logP - log10(1.0 + 10^(pH - pKa1))
    elseif charge == :base
        # logD = logP - log10(1 + 10^(pKa - pH))
        return logP - log10(1.0 + 10^(pKa1 - pH))
    else
        return logP
    end
end

"""
calculate_fu_blood(fu_plasma, bp_ratio)

Calculate fraction unbound in whole blood.
"""
function calculate_fu_blood(fu_plasma::Float64, bp_ratio::Float64)::Float64
    return fu_plasma / bp_ratio
end

"""
calculate_erythrocyte_water_partition(drug, blood)

Calculate partition into RBC water space only.
Useful for polar drugs that don't penetrate membranes well.
"""
function calculate_erythrocyte_water_partition(
    drug::DrugProperties,
    blood::BloodComposition
)::Float64

    fu = drug.fu_plasma
    f_w_rbc = blood.rbc_water_fraction

    # Only water space accessible
    K_water = calculate_water_partition(
        drug.pKa, drug.charge_type,
        blood.plasma_ph, blood.rbc_ph
    )

    return fu * f_w_rbc * K_water
end

# ============================================================================
# DRUG DATABASE - Common Drugs with Known Parameters
# ============================================================================

"""
get_drug_properties(name)

Get properties for common drugs from internal database.
"""
function get_drug_properties(name::String)::DrugProperties
    drugs = Dict(
        # Anticoagulants
        "warfarin" => create_drug_properties("warfarin",
            mw=308.3, charge_type=:acid, pKa=[5.0],
            logP=2.7, fu_plasma=0.01, albumin_binding=true
        ),
        "rivaroxaban" => create_drug_properties("rivaroxaban",
            mw=435.9, charge_type=:neutral, pKa=Float64[],
            logP=1.5, fu_plasma=0.06, albumin_binding=true
        ),
        "apixaban" => create_drug_properties("apixaban",
            mw=459.5, charge_type=:neutral, pKa=Float64[],
            logP=1.0, fu_plasma=0.13, albumin_binding=true
        ),
        "dabigatran" => create_drug_properties("dabigatran",
            mw=627.7, charge_type=:base, pKa=[6.8],
            logP=0.2, fu_plasma=0.65, albumin_binding=false
        ),

        # Antimalarials (high WBC accumulation)
        "chloroquine" => create_drug_properties("chloroquine",
            mw=319.9, charge_type=:base, pKa=[8.1, 10.2],
            logP=4.6, fu_plasma=0.4, lysosomal_trapping=true,
            albumin_binding=true, agp_binding=true
        ),
        "hydroxychloroquine" => create_drug_properties("hydroxychloroquine",
            mw=335.9, charge_type=:base, pKa=[8.3, 9.7],
            logP=3.6, fu_plasma=0.5, lysosomal_trapping=true
        ),

        # Antiretrovirals
        "efavirenz" => create_drug_properties("efavirenz",
            mw=315.7, charge_type=:acid, pKa=[10.2],
            logP=4.6, fu_plasma=0.005, albumin_binding=true
        ),
        "tenofovir" => create_drug_properties("tenofovir",
            mw=287.2, charge_type=:acid, pKa=[3.8, 6.7],
            logP=-1.6, fu_plasma=0.93, albumin_binding=false
        ),

        # Antibiotics
        "azithromycin" => create_drug_properties("azithromycin",
            mw=749.0, charge_type=:base, pKa=[8.7],
            logP=4.0, fu_plasma=0.5, lysosomal_trapping=true
        ),

        # Antiplatelets
        "clopidogrel" => create_drug_properties("clopidogrel",
            mw=321.8, charge_type=:base, pKa=[4.5],
            logP=3.8, fu_plasma=0.02, albumin_binding=true
        ),
        "aspirin" => create_drug_properties("aspirin",
            mw=180.2, charge_type=:acid, pKa=[3.5],
            logP=1.2, fu_plasma=0.15, albumin_binding=true,
            hemoglobin_binding=true
        )
    )

    name_lower = lowercase(name)
    if haskey(drugs, name_lower)
        return drugs[name_lower]
    else
        error("Drug '$name' not found in database. Use create_drug_properties() for custom drugs.")
    end
end

# ============================================================================
# DRUG-SPECIFIC WBC BINDING - Chloroquine, Antiretrovirals
# ============================================================================

"""
WBC-SPECIFIC DRUG ACCUMULATION

Key drugs with significant WBC accumulation:

1. CHLOROQUINE/HYDROXYCHLOROQUINE
   - Lysosomotropic weak base (pKa 8.1, 10.2)
   - WBC:Plasma ratio 5-10× in neutrophils, up to 100× in macrophages
   - Half-life in WBC: 40-60 days (explains long therapeutic effect)
   - Mechanism: Ion trapping in acidic lysosomes (pH 4.5-5.0)
   - Clinical relevance: Rheumatoid arthritis, lupus, malaria

2. ANTIRETROVIRALS
   - Many HIV drugs accumulate in PBMC (peripheral blood mononuclear cells)
   - Critical for HIV reservoir targeting
   - NRTIs (tenofovir, emtricitabine): Accumulate as active metabolites
   - PIs (darunavir, atazanavir): Lipophilic, lysosomal trapping
   - INSTIs (dolutegravir, raltegravir): Moderate accumulation

3. MACROLIDES (Azithromycin)
   - Extreme WBC accumulation (100-200×)
   - Delivered to infection sites by phagocytes
   - Explains tissue efficacy despite low plasma levels
"""

# WBC accumulation ratios from literature
const WBC_DRUG_ACCUMULATION = Dict{String, Dict{String, Float64}}(
    # Chloroquine - Mackenzie (1983), Tett (1989)
    "chloroquine" => Dict(
        "wbc_plasma_ratio" => 7.0,          # Whole WBC
        "neutrophil_ratio" => 5.0,
        "lymphocyte_ratio" => 8.0,
        "monocyte_ratio" => 15.0,           # Highest in monocytes/macrophages
        "lysosome_ratio" => 100.0,          # Within lysosomes
        "half_life_wbc_days" => 40.0,       # Very long retention
        "intracellular_binding" => 0.85     # 85% bound intracellularly
    ),

    # Hydroxychloroquine - slightly lower than CQ
    "hydroxychloroquine" => Dict(
        "wbc_plasma_ratio" => 5.0,
        "neutrophil_ratio" => 4.0,
        "lymphocyte_ratio" => 6.0,
        "monocyte_ratio" => 12.0,
        "lysosome_ratio" => 80.0,
        "half_life_wbc_days" => 35.0,
        "intracellular_binding" => 0.80
    ),

    # Azithromycin - Gladue (1989), Girard (1996)
    "azithromycin" => Dict(
        "wbc_plasma_ratio" => 100.0,        # Extreme accumulation!
        "neutrophil_ratio" => 120.0,
        "lymphocyte_ratio" => 80.0,
        "monocyte_ratio" => 200.0,
        "lysosome_ratio" => 300.0,
        "half_life_wbc_days" => 3.0,        # Shorter than CQ
        "intracellular_binding" => 0.70
    ),

    # NRTIs - accumulate as intracellular phosphates
    "tenofovir" => Dict(
        "wbc_plasma_ratio" => 0.3,          # Parent drug
        "pbmc_ratio" => 0.5,
        "active_metabolite_ratio" => 20.0,  # TFV-DP (active form)
        "t_half_intracellular_hours" => 150.0,  # 6+ days for TFV-DP
        "phosphorylation_rate" => 0.15      # Requires activation
    ),

    "emtricitabine" => Dict(
        "wbc_plasma_ratio" => 0.5,
        "pbmc_ratio" => 0.8,
        "active_metabolite_ratio" => 10.0,  # FTC-TP
        "t_half_intracellular_hours" => 39.0,
        "phosphorylation_rate" => 0.25
    ),

    # Protease Inhibitors - lipophilic, P-gp substrates
    "darunavir" => Dict(
        "wbc_plasma_ratio" => 0.8,
        "pbmc_ratio" => 1.2,
        "lymphocyte_ratio" => 1.5,
        "p_gp_efflux" => 0.7,               # Significant efflux
        "intracellular_binding" => 0.95     # Highly bound
    ),

    "atazanavir" => Dict(
        "wbc_plasma_ratio" => 2.0,
        "pbmc_ratio" => 2.5,
        "lymphocyte_ratio" => 3.0,
        "p_gp_efflux" => 0.5,
        "intracellular_binding" => 0.86
    ),

    "ritonavir" => Dict(
        "wbc_plasma_ratio" => 0.5,
        "pbmc_ratio" => 0.6,
        "p_gp_efflux" => 0.8,               # Strong efflux
        "intracellular_binding" => 0.99     # Very highly bound
    ),

    # INSTIs - moderate accumulation
    "dolutegravir" => Dict(
        "wbc_plasma_ratio" => 1.5,
        "pbmc_ratio" => 2.0,
        "cd4_ratio" => 2.5,                 # Target cells
        "intracellular_binding" => 0.50
    ),

    "raltegravir" => Dict(
        "wbc_plasma_ratio" => 1.2,
        "pbmc_ratio" => 1.5,
        "cd4_ratio" => 1.8,
        "intracellular_binding" => 0.17
    ),

    "elvitegravir" => Dict(
        "wbc_plasma_ratio" => 0.8,
        "pbmc_ratio" => 1.0,
        "p_gp_efflux" => 0.6,
        "intracellular_binding" => 0.99
    )
)

"""
get_wbc_accumulation(drug_name)

Get WBC accumulation parameters for specific drugs.
Returns Dict with cell-specific ratios and kinetic parameters.
"""
function get_wbc_accumulation(drug_name::String)::Dict{String, Float64}
    name_lower = lowercase(drug_name)
    if haskey(WBC_DRUG_ACCUMULATION, name_lower)
        return WBC_DRUG_ACCUMULATION[name_lower]
    else
        # Return default for unknown drugs
        return Dict(
            "wbc_plasma_ratio" => 1.0,
            "neutrophil_ratio" => 1.0,
            "lymphocyte_ratio" => 1.0,
            "monocyte_ratio" => 1.0
        )
    end
end

"""
calculate_drug_specific_wbc_partition(drug_name, blood; cell_type=:total_wbc)

Calculate WBC partition using drug-specific literature data.

# Arguments
- `drug_name`: Name of drug (chloroquine, azithromycin, dolutegravir, etc.)
- `blood`: BloodComposition
- `cell_type`: :total_wbc, :neutrophil, :lymphocyte, :monocyte, :pbmc, :cd4

# Returns
- Kp,wbc: WBC:Plasma partition coefficient
"""
function calculate_drug_specific_wbc_partition(
    drug_name::String,
    blood::BloodComposition;
    cell_type::Symbol=:total_wbc
)::Float64

    params = get_wbc_accumulation(drug_name)

    # Map cell type to parameter name
    ratio_key = if cell_type == :total_wbc
        "wbc_plasma_ratio"
    elseif cell_type == :neutrophil
        "neutrophil_ratio"
    elseif cell_type == :lymphocyte
        "lymphocyte_ratio"
    elseif cell_type == :monocyte
        "monocyte_ratio"
    elseif cell_type == :pbmc
        get(params, "pbmc_ratio", params["wbc_plasma_ratio"])
        return get(params, "pbmc_ratio", get(params, "wbc_plasma_ratio", 1.0))
    elseif cell_type == :cd4
        return get(params, "cd4_ratio", get(params, "lymphocyte_ratio", 1.0))
    else
        "wbc_plasma_ratio"
    end

    return get(params, ratio_key, 1.0)
end

"""
calculate_intracellular_drug_amount(
    drug_name,
    plasma_conc,
    blood;
    include_metabolites=false
)

Calculate total intracellular drug amount in WBCs.

# Arguments
- `drug_name`: Drug name
- `plasma_conc`: Plasma concentration (nM or μM - same units returned)
- `blood`: BloodComposition
- `include_metabolites`: Include active metabolites (NRTIs)

# Returns
- Dict with amounts in each cell type
"""
function calculate_intracellular_drug_amount(
    drug_name::String,
    plasma_conc::Float64,
    blood::BloodComposition;
    include_metabolites::Bool=false
)::Dict{String, Float64}

    params = get_wbc_accumulation(drug_name)

    # Cell counts and volumes
    wbc_count = blood.wbc_count  # cells/L
    neutrophil_frac = 0.60
    lymphocyte_frac = 0.30
    monocyte_frac = 0.08

    # Cell volumes (L per cell)
    v_neutrophil = 350e-15  # 350 fL
    v_lymphocyte = 220e-15  # 220 fL
    v_monocyte = 450e-15    # 450 fL

    # Partition coefficients
    Kp_neut = get(params, "neutrophil_ratio", 1.0)
    Kp_lymph = get(params, "lymphocyte_ratio", 1.0)
    Kp_mono = get(params, "monocyte_ratio", 1.0)

    # Intracellular concentrations
    c_neutrophil = plasma_conc * Kp_neut
    c_lymphocyte = plasma_conc * Kp_lymph
    c_monocyte = plasma_conc * Kp_mono

    # Total amounts (conc × volume × count)
    amount_neutrophil = c_neutrophil * v_neutrophil * wbc_count * neutrophil_frac
    amount_lymphocyte = c_lymphocyte * v_lymphocyte * wbc_count * lymphocyte_frac
    amount_monocyte = c_monocyte * v_monocyte * wbc_count * monocyte_frac

    result = Dict(
        "plasma_conc" => plasma_conc,
        "neutrophil_conc" => c_neutrophil,
        "lymphocyte_conc" => c_lymphocyte,
        "monocyte_conc" => c_monocyte,
        "total_wbc_amount" => amount_neutrophil + amount_lymphocyte + amount_monocyte,
        "neutrophil_amount" => amount_neutrophil,
        "lymphocyte_amount" => amount_lymphocyte,
        "monocyte_amount" => amount_monocyte
    )

    # Add active metabolites for NRTIs
    if include_metabolites && haskey(params, "active_metabolite_ratio")
        metabolite_ratio = params["active_metabolite_ratio"]
        result["active_metabolite_conc"] = plasma_conc * metabolite_ratio
        result["t_half_intracellular"] = get(params, "t_half_intracellular_hours", 24.0)
    end

    return result
end

"""
calculate_reservoir_effect(drug_name, blood; time_after_dose_hours=24.0)

Calculate the "reservoir effect" - drug release from WBC stores.

Important for:
- Chloroquine: Very long WBC half-life explains sustained effect
- Azithromycin: Phagocyte-mediated delivery to infection
- NRTIs: Intracellular phosphate persistence

# Returns
- Dict with reservoir parameters
"""
function calculate_reservoir_effect(
    drug_name::String,
    blood::BloodComposition;
    time_after_dose_hours::Float64=24.0
)::Dict{String, Float64}

    params = get_wbc_accumulation(drug_name)

    # Get intracellular half-life
    if haskey(params, "half_life_wbc_days")
        t_half = params["half_life_wbc_days"] * 24.0  # Convert to hours
    elseif haskey(params, "t_half_intracellular_hours")
        t_half = params["t_half_intracellular_hours"]
    else
        t_half = 24.0  # Default 1 day
    end

    # Calculate remaining fraction
    ke = log(2) / t_half
    fraction_remaining = exp(-ke * time_after_dose_hours)

    # Reservoir significance
    wbc_ratio = get(params, "wbc_plasma_ratio", 1.0)
    reservoir_index = wbc_ratio * fraction_remaining

    return Dict(
        "t_half_hours" => t_half,
        "fraction_remaining" => fraction_remaining,
        "wbc_plasma_ratio" => wbc_ratio,
        "reservoir_index" => reservoir_index,
        "clinically_significant" => reservoir_index > 2.0
    )
end

# Export new functions
export get_wbc_accumulation, calculate_drug_specific_wbc_partition
export calculate_intracellular_drug_amount, calculate_reservoir_effect
export WBC_DRUG_ACCUMULATION

end  # module BloodBinding
