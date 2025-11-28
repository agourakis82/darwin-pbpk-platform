# Rodgers-Rowland Tissue Partition Coefficient Prediction
# Based on: Rodgers & Rowland (2005, 2006) J Pharm Sci
#
# This module implements mechanistic tissue:plasma partition coefficient (Kp)
# prediction based on tissue composition and drug physicochemical properties.

module TissuePartition

export TissueComposition, DrugProperties, IonizationType
export calculate_kp, calculate_all_kp, calculate_fut, calculate_vdss_mechanistic
export HUMAN_TISSUE_COMPOSITION
export NEUTRAL, MONOPROTIC_ACID, MONOPROTIC_BASE, DIPROTIC_ACID, DIPROTIC_BASE
export ZWITTERION, TRIPROTIC_ACID, TRIPROTIC_BASE
export DIPROTIC_ACID_MONOPROTIC_BASE, DIPROTIC_BASE_MONOPROTIC_ACID

"""
Ionization types for drugs
"""
@enum IonizationType begin
    NEUTRAL = 1
    MONOPROTIC_ACID = 2
    MONOPROTIC_BASE = 3
    DIPROTIC_ACID = 4
    DIPROTIC_BASE = 5
    ZWITTERION = 6  # monoprotic acid/base
    TRIPROTIC_ACID = 7
    TRIPROTIC_BASE = 8
    DIPROTIC_ACID_MONOPROTIC_BASE = 9
    DIPROTIC_BASE_MONOPROTIC_ACID = 10
end

"""
Tissue composition data structure
Based on Rodgers & Rowland (2005, 2006)
"""
struct TissueComposition
    name::String
    f_n_l::Float64    # Neutral lipid fraction
    f_n_pl::Float64   # Neutral phospholipid fraction
    f_a_pl::Float64   # Acidic phospholipid fraction (×10⁻³ for storage, actual value)
    f_ew::Float64     # Extracellular water fraction
    f_iw::Float64     # Intracellular water fraction
    AR::Float64       # Albumin ratio (tissue/plasma)
    LR::Float64       # Lipoprotein ratio (tissue/plasma)
    volume::Float64   # Tissue volume (L) for 70kg human
end

"""
Drug physicochemical properties
"""
struct DrugProperties
    logP::Float64           # Octanol:water partition coefficient
    pKa::Vector{Float64}    # Ionization constants
    fup::Float64            # Fraction unbound in plasma
    BP::Float64             # Blood:plasma ratio (default 1.0)
    ion_type::IonizationType
end

# Constructor with default BP
DrugProperties(logP, pKa, fup, ion_type) = DrugProperties(logP, pKa, fup, 1.0, ion_type)

# Human tissue composition data (Rodgers & Rowland 2005, 2006)
# Volumes from ICRP Reference Man (70kg)
const HUMAN_TISSUE_COMPOSITION = Dict{String, TissueComposition}(
    # Tissue          name        f_n_l   f_n_pl  f_a_pl   f_ew    f_iw    AR      LR      Vol(L)
    "adipose"  => TissueComposition("adipose",  0.853,  0.0016, 0.0004,  0.135,  0.017,  0.049,  0.068,  12.0),
    "bone"     => TissueComposition("bone",     0.017,  0.0017, 0.00067, 0.100,  0.346,  0.100,  0.050,  4.0),
    "brain"    => TissueComposition("brain",    0.039,  0.0015, 0.0004,  0.162,  0.620,  0.048,  0.041,  1.4),
    "gut"      => TissueComposition("gut",      0.038,  0.0125, 0.00241, 0.282,  0.475,  0.158,  0.141,  1.2),
    "heart"    => TissueComposition("heart",    0.014,  0.0111, 0.00225, 0.320,  0.456,  0.157,  0.160,  0.33),
    "kidney"   => TissueComposition("kidney",   0.012,  0.0242, 0.00503, 0.273,  0.483,  0.130,  0.137,  0.31),
    "liver"    => TissueComposition("liver",    0.014,  0.0240, 0.00456, 0.161,  0.573,  0.086,  0.161,  1.8),
    "lung"     => TissueComposition("lung",     0.022,  0.0128, 0.00391, 0.336,  0.446,  0.212,  0.168,  1.0),
    "muscle"   => TissueComposition("muscle",   0.010,  0.0072, 0.00153, 0.118,  0.630,  0.064,  0.059,  30.0),
    "pancreas" => TissueComposition("pancreas", 0.041,  0.0093, 0.00167, 0.120,  0.664,  0.060,  0.060,  0.1),
    "skin"     => TissueComposition("skin",     0.060,  0.0044, 0.00132, 0.382,  0.291,  0.277,  0.096,  3.0),
    "spleen"   => TissueComposition("spleen",   0.0077, 0.0113, 0.00318, 0.207,  0.579,  0.097,  0.207,  0.18),
    "thymus"   => TissueComposition("thymus",   0.017,  0.0092, 0.00230, 0.150,  0.626,  0.075,  0.075,  0.02),
    # Blood components
    "rbc"      => TissueComposition("rbc",      0.0017, 0.0029, 0.0005,  0.0,    0.603,  0.0,    0.0,    2.5),
    "plasma"   => TissueComposition("plasma",   0.0023, 0.0013, 0.0,     0.0,    0.0,    0.0,    0.0,    3.0),
)

# Constants
const pH_IW = 7.0       # Intracellular water pH
const pH_P = 7.4        # Plasma pH
const pH_RBC = 7.22     # Red blood cell pH
const HCT = 0.45        # Hematocrit

"""
Calculate ionization factor X (intracellular water)
"""
function ionization_factor_X(pKa::Vector{Float64}, ion_type::IonizationType)
    if isempty(pKa) || ion_type == NEUTRAL
        return 0.0
    end

    pH = pH_IW

    return if ion_type == MONOPROTIC_ACID
        10^(pH - pKa[1])
    elseif ion_type == MONOPROTIC_BASE
        10^(pKa[1] - pH)
    elseif ion_type == DIPROTIC_ACID
        10^(pH - pKa[1]) + 10^(2*pH - pKa[1] - pKa[2])
    elseif ion_type == DIPROTIC_BASE
        10^(pKa[2] - pH) + 10^(pKa[1] + pKa[2] - 2*pH)
    elseif ion_type == ZWITTERION
        10^(pKa[2] - pH) + 10^(pH - pKa[1])
    elseif ion_type == TRIPROTIC_ACID
        10^(pH - pKa[1]) + 10^(2*pH - pKa[1] - pKa[2]) + 10^(3*pH - pKa[1] - pKa[2] - pKa[3])
    elseif ion_type == TRIPROTIC_BASE
        10^(pKa[3] - pH) + 10^(pKa[3] + pKa[2] - 2*pH) + 10^(pKa[1] + pKa[2] + pKa[3] - 3*pH)
    elseif ion_type == DIPROTIC_ACID_MONOPROTIC_BASE
        10^(pKa[3] - pH) + 10^(pH - pKa[1]) + 10^(2*pH - pKa[1] - pKa[2])
    elseif ion_type == DIPROTIC_BASE_MONOPROTIC_ACID
        10^(pH - pKa[1]) + 10^(pKa[3] - pH) + 10^(pKa[2] + pKa[3] - 2*pH)
    else
        0.0
    end
end

"""
Calculate ionization factor Y (plasma)
"""
function ionization_factor_Y(pKa::Vector{Float64}, ion_type::IonizationType)
    if isempty(pKa) || ion_type == NEUTRAL
        return 0.0
    end

    pH = pH_P

    return if ion_type == MONOPROTIC_ACID
        10^(pH - pKa[1])
    elseif ion_type == MONOPROTIC_BASE
        10^(pKa[1] - pH)
    elseif ion_type == DIPROTIC_ACID
        10^(pH - pKa[1]) + 10^(2*pH - pKa[1] - pKa[2])
    elseif ion_type == DIPROTIC_BASE
        10^(pKa[2] - pH) + 10^(pKa[1] + pKa[2] - 2*pH)
    elseif ion_type == ZWITTERION
        10^(pKa[2] - pH) + 10^(pH - pKa[1])
    elseif ion_type == TRIPROTIC_ACID
        10^(pH - pKa[1]) + 10^(2*pH - pKa[1] - pKa[2]) + 10^(3*pH - pKa[1] - pKa[2] - pKa[3])
    elseif ion_type == TRIPROTIC_BASE
        10^(pKa[3] - pH) + 10^(pKa[3] + pKa[2] - 2*pH) + 10^(pKa[1] + pKa[2] + pKa[3] - 3*pH)
    elseif ion_type == DIPROTIC_ACID_MONOPROTIC_BASE
        10^(pKa[3] - pH) + 10^(pH - pKa[1]) + 10^(2*pH - pKa[1] - pKa[2])
    elseif ion_type == DIPROTIC_BASE_MONOPROTIC_ACID
        10^(pH - pKa[1]) + 10^(pKa[3] - pH) + 10^(pKa[2] + pKa[3] - 2*pH)
    else
        0.0
    end
end

"""
Calculate ionization factor Z (red blood cells)
"""
function ionization_factor_Z(pKa::Vector{Float64}, ion_type::IonizationType)
    if isempty(pKa)
        return 1.0
    end

    pH = pH_RBC

    return if ion_type == NEUTRAL
        1.0
    elseif ion_type == MONOPROTIC_ACID
        1.0
    elseif ion_type == MONOPROTIC_BASE
        10^(pKa[1] - pH)
    elseif ion_type == DIPROTIC_ACID
        1.0
    elseif ion_type == DIPROTIC_BASE
        10^(pKa[2] - pH) + 10^(pKa[1] + pKa[2] - 2*pH)
    elseif ion_type == ZWITTERION
        10^(pKa[2] - pH) + 10^(pH - pKa[1])
    elseif ion_type == TRIPROTIC_ACID
        1.0
    elseif ion_type == TRIPROTIC_BASE
        10^(pKa[3] - pH) + 10^(pKa[3] + pKa[2] - 2*pH) + 10^(pKa[1] + pKa[2] + pKa[3] - 3*pH)
    elseif ion_type == DIPROTIC_ACID_MONOPROTIC_BASE
        10^(pKa[3] - pH) + 10^(pH - pKa[1]) + 10^(2*pH - pKa[1] - pKa[2])
    elseif ion_type == DIPROTIC_BASE_MONOPROTIC_ACID
        10^(pH - pKa[1]) + 10^(pKa[3] - pH) + 10^(pKa[2] + pKa[3] - 2*pH)
    else
        1.0
    end
end

"""
Determine calculation type for Kp
Returns 1 for moderate-to-strong bases, 2 for acids/zwitterions, 3 for neutrals
"""
function get_calculation_type(pKa::Vector{Float64}, ion_type::IonizationType)
    if ion_type == NEUTRAL
        return 3
    end

    # Check if it's a moderate-to-strong base (pKa > 7)
    is_strong_base = false
    if ion_type == MONOPROTIC_BASE && !isempty(pKa) && pKa[1] > 7
        is_strong_base = true
    elseif ion_type == DIPROTIC_BASE && !isempty(pKa) && pKa[1] > 7
        is_strong_base = true
    elseif ion_type == ZWITTERION && length(pKa) >= 2 && pKa[2] > 7
        is_strong_base = true
    elseif ion_type == TRIPROTIC_BASE && !isempty(pKa) && pKa[1] > 7
        is_strong_base = true
    elseif ion_type == DIPROTIC_ACID_MONOPROTIC_BASE && length(pKa) >= 3 && pKa[3] > 7
        is_strong_base = true
    elseif ion_type == DIPROTIC_BASE_MONOPROTIC_ACID && length(pKa) >= 2 && pKa[2] > 7
        is_strong_base = true
    end

    return is_strong_base ? 1 : 2
end

"""
Calculate tissue:plasma partition coefficient (Kp) for a single tissue
using Rodgers-Rowland equations.

# Arguments
- `drug`: DrugProperties struct with drug physicochemical properties
- `tissue`: TissueComposition struct with tissue composition data

# Returns
- Kp: tissue:plasma partition coefficient
"""
function calculate_kp(drug::DrugProperties, tissue::TissueComposition)
    # Get partition coefficients
    P = 10^drug.logP
    logP_OW = 1.115 * drug.logP - 1.35  # Oil:water from octanol:water
    P_OW = 10^logP_OW

    # Ionization factors
    X = ionization_factor_X(drug.pKa, drug.ion_type)
    Y = ionization_factor_Y(drug.pKa, drug.ion_type)
    Z = ionization_factor_Z(drug.pKa, drug.ion_type)

    # Blood cell partition coefficient
    rbc = HUMAN_TISSUE_COMPOSITION["rbc"]
    plasma = HUMAN_TISSUE_COMPOSITION["plasma"]
    Kpu_bc = (HCT - 1 + drug.BP) / (HCT * drug.fup)

    # Affinity constants
    Ka_PR = (1/drug.fup - 1 - (P*plasma.f_n_l + (0.3*P + 0.7)*plasma.f_n_pl)/(1+Y))
    Ka_AP = (Kpu_bc - (1 + Z)/(1 + Y)*rbc.f_iw - (P*rbc.f_n_l + (0.3*P + 0.7)*rbc.f_n_pl)/(1 + Y)) * (1 + Y)/(rbc.f_a_pl * max(Z, 1e-10))

    # Get calculation type
    calc_type = get_calculation_type(drug.pKa, drug.ion_type)

    # Use appropriate partition coefficient for adipose
    P_tissue = tissue.name == "adipose" ? P_OW : P

    # Calculate Kp based on drug type
    if calc_type == 1  # Moderate to strong bases - use acidic phospholipid binding
        Kp = (tissue.f_ew +
              ((1+X)/(1+Y)) * tissue.f_iw +
              (P_tissue*tissue.f_n_l + (0.3*P_tissue + 0.7)*tissue.f_n_pl) / (1+Y) +
              (Ka_AP * tissue.f_a_pl * X) / (1+Y)) * drug.fup
    elseif calc_type == 2  # Acids and zwitterions - use albumin binding
        Kp = (tissue.f_ew +
              ((1+X)/(1+Y)) * tissue.f_iw +
              (P_tissue*tissue.f_n_l + (0.3*P_tissue + 0.7)*tissue.f_n_pl) / (1+Y) +
              (Ka_PR * tissue.AR * X) / (1+Y)) * drug.fup
    else  # Neutrals - use lipoprotein binding
        Kp = (tissue.f_ew +
              ((1+X)/(1+Y)) * tissue.f_iw +
              (P_tissue*tissue.f_n_l + (0.3*P_tissue + 0.7)*tissue.f_n_pl) / (1+Y) +
              (Ka_PR * tissue.LR * X) / (1+Y)) * drug.fup
    end

    return max(Kp, 0.01)  # Ensure positive Kp
end

"""
Calculate all tissue Kp values for a drug
Returns a Dict of tissue_name => Kp
"""
function calculate_all_kp(drug::DrugProperties)
    kp_values = Dict{String, Float64}()

    for (name, tissue) in HUMAN_TISSUE_COMPOSITION
        if name ∉ ["rbc", "plasma"]  # Skip blood components
            kp_values[name] = calculate_kp(drug, tissue)
        end
    end

    return kp_values
end

"""
Calculate fraction unbound in tissue (fut) from Kp values
Using the relationship: Kp = fup/fut for non-lipophilic distribution
Or more accurately: Kpu = Kp/fup, then fut = 1/Kpu for simple cases

For a more rigorous approach, we use the volume-weighted average across tissues.
"""
function calculate_fut(drug::DrugProperties; method::Symbol=:volume_weighted)
    kp_values = calculate_all_kp(drug)

    if method == :volume_weighted
        # Volume-weighted average Kpu across all tissues
        total_volume = 0.0
        weighted_kpu_sum = 0.0

        for (name, tissue) in HUMAN_TISSUE_COMPOSITION
            if name ∉ ["rbc", "plasma"]
                kp = kp_values[name]
                kpu = kp / drug.fup  # Unbound partition coefficient
                weighted_kpu_sum += kpu * tissue.volume
                total_volume += tissue.volume
            end
        end

        avg_kpu = weighted_kpu_sum / total_volume
        fut = 1.0 / avg_kpu

    elseif method == :muscle
        # Use muscle as representative (largest tissue, well-perfused)
        kp_muscle = kp_values["muscle"]
        kpu_muscle = kp_muscle / drug.fup
        fut = 1.0 / kpu_muscle

    else  # :simple
        # Simple average across major tissues
        major_tissues = ["muscle", "liver", "kidney", "adipose", "gut"]
        kpu_sum = sum(kp_values[t] / drug.fup for t in major_tissues)
        avg_kpu = kpu_sum / length(major_tissues)
        fut = 1.0 / avg_kpu
    end

    return clamp(fut, 0.001, 0.99)
end

"""
Calculate Vdss mechanistically using Øie-Tozer equation with tissue Kp values

Vdss = Vp + Σ(Kp_tissue × V_tissue)

Where Vp is plasma volume and the sum is over all tissues.
"""
function calculate_vdss_mechanistic(drug::DrugProperties)
    kp_values = calculate_all_kp(drug)

    # Plasma volume (L)
    Vp = HUMAN_TISSUE_COMPOSITION["plasma"].volume

    # Sum of Kp × Vtissue for all tissues
    tissue_contribution = 0.0
    for (name, tissue) in HUMAN_TISSUE_COMPOSITION
        if name ∉ ["rbc", "plasma"]
            tissue_contribution += kp_values[name] * tissue.volume
        end
    end

    # Add RBC contribution
    rbc = HUMAN_TISSUE_COMPOSITION["rbc"]
    # RBC Kp based on BP ratio
    Kp_rbc = drug.BP * (1 - HCT) / HCT + 1
    tissue_contribution += Kp_rbc * rbc.volume

    Vdss = Vp + tissue_contribution

    return Vdss
end

"""
Helper function to classify drug ionization type from pKa values
"""
function classify_ionization(pKa_acid::Vector{Float64}, pKa_base::Vector{Float64})
    n_acid = length(pKa_acid)
    n_base = length(pKa_base)

    if n_acid == 0 && n_base == 0
        return NEUTRAL, Float64[]
    elseif n_acid == 1 && n_base == 0
        return MONOPROTIC_ACID, pKa_acid
    elseif n_acid == 0 && n_base == 1
        return MONOPROTIC_BASE, pKa_base
    elseif n_acid == 2 && n_base == 0
        return DIPROTIC_ACID, sort(pKa_acid)
    elseif n_acid == 0 && n_base == 2
        return DIPROTIC_BASE, sort(pKa_base, rev=true)
    elseif n_acid == 1 && n_base == 1
        return ZWITTERION, [pKa_acid[1], pKa_base[1]]
    else
        # Complex cases - simplify to dominant ionization
        if n_acid > n_base
            return DIPROTIC_ACID, sort(pKa_acid)[1:min(2, n_acid)]
        else
            return DIPROTIC_BASE, sort(pKa_base, rev=true)[1:min(2, n_base)]
        end
    end
end

end # module
