"""
White Blood Cells (WBC) Compartment - Detailed Modeling with Subpopulations

Models all white blood cell subpopulations with:
- Separate compartments for each type (neutrophils, lymphocytes T/B/NK, monocytes, eosinophils, basophils)
- Detailed binding and internalization parameters
- Intracellular compartments (cytosol, lysosomes)
- Pathology-dependent dynamics (leukemia, sepsis, leukopenia)
- Fractal morphology analysis integration

Author: Darwin PBPK Platform
Date: 2025-12-01
"""

module WhiteBloodCells

using ..PatientProfile
using ..FractalBlood
using LinearAlgebra
using Statistics

export WhiteBloodCellSubpopulation, WhiteBloodCellCompartment
export create_WBC_compartment, calculate_WBC_volume_fraction
export adjust_WBC_for_pathology, create_WBC_phases_for_fractal_blood
export calculate_partition_coefficient, calculate_internalization_rate
export get_fractal_corrected_parameters

# ============================================================================
# CONSTANTS
# ============================================================================

# Normal cell counts (cells/μL → converted to cells/L in code)
const NORMAL_NEUTROPHILS = 3000.0  # cells/μL (50-70% of WBC)
const NORMAL_LYMPHOCYTES_T = 1200.0  # cells/μL (T cells)
const NORMAL_LYMPHOCYTES_B = 400.0   # cells/μL (B cells)
const NORMAL_LYMPHOCYTES_NK = 200.0  # cells/μL (NK cells)
const NORMAL_MONOCYTES = 500.0      # cells/μL (2-8%)
const NORMAL_EOSINOPHILS = 200.0    # cells/μL (1-4%)
const NORMAL_BASOPHILS = 30.0       # cells/μL (<1%)

# Cell volumes (fL per cell)
const VOLUME_NEUTROPHIL = 330.0     # fL
const VOLUME_LYMPHOCYTE_SMALL = 200.0  # fL (small lymphocyte)
const VOLUME_LYMPHOCYTE_LARGE = 500.0  # fL (large lymphocyte)
const VOLUME_MONOCYTE = 400.0       # fL
const VOLUME_EOSINOPHIL = 400.0     # fL
const VOLUME_BASOPHIL = 300.0       # fL

# Intracellular pH values
const PH_CYTOSOL = 7.2
const PH_LYSOSOME = 5.0
const PH_AZUROPHILIC_GRANULE = 5.5  # Neutrophil-specific

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
WhiteBloodCellSubpopulation - Detailed model for a single WBC subpopulation

Each subpopulation is modeled with:
- Physical parameters (volume, count, size)
- Transport parameters (velocity, partition)
- Binding parameters (capacity, affinity)
- Internalization parameters (rates, compartments)
- Intracellular compartments (cytosol, lysosomes)
"""
mutable struct WhiteBloodCellSubpopulation
    name::String
    
    # Physical parameters
    cell_count::Float64              # cells/L blood
    cell_volume_fL::Float64          # fL per cell
    volume_fraction::Float64         # Fraction of blood volume
    
    # Transport parameters (similar to RBC)
    velocity_factor::Float64         # Relative to plasma (0.7-1.0)
    partition_coefficient::Float64   # Plasma → WBC partition
    
    # Binding parameters (drug-specific, stored as Dict)
    binding_capacity::Dict{String, Float64}      # Bmax (mol/cell) per drug
    binding_affinity::Dict{String, Float64}      # Kd (mol/L) per drug
    
    # Internalization parameters
    internalization_rate::Dict{String, Float64}  # k_internalization (1/h) per drug
    efflux_rate::Dict{String, Float64}           # k_efflux (1/h) per drug
    
    # Intracellular compartments
    lysosome_volume_fraction::Float64  # Fraction of cell volume (0.05-0.15)
    pH_lysosome::Float64              # pH of lysosomes
    
    # Special compartments (subpopulation-specific)
    azurophilic_granule_fraction::Float64  # Neutrophils only
    pH_azurophilic::Float64
    
    # Exchange rates
    exchange_rate_plasma::Float64     # Rate of exchange with plasma (1/h)
    
    # Fractal morphology parameters (for analysis integration)
    fractal_dimension_edge::Float64   # df_edge from image analysis
    fractal_dimension_distribution::Float64  # df_distribution
    
    # Pathology modifiers
    pathology_multiplier::Float64     # Multiplier for cell count (1.0 = normal)
end

"""
WhiteBloodCellCompartment - Complete WBC compartment with all subpopulations
"""
mutable struct WhiteBloodCellCompartment
    # All subpopulations (separated as requested)
    neutrophils::WhiteBloodCellSubpopulation
    lymphocytes_T::WhiteBloodCellSubpopulation
    lymphocytes_B::WhiteBloodCellSubpopulation
    lymphocytes_NK::WhiteBloodCellSubpopulation
    monocytes::WhiteBloodCellSubpopulation
    eosinophils::WhiteBloodCellSubpopulation
    basophils::WhiteBloodCellSubpopulation
    
    # Pathology state
    pathology::String  # "normal", "leukocytosis", "leukopenia", "leukemia", "sepsis"
    pathology_severity::Float64  # 0.0-1.0 scale
    
    # Total WBC parameters
    total_WBC_count::Float64  # cells/L
    total_volume_fraction::Float64
end

# ============================================================================
# FACTORY FUNCTIONS - CREATE SUBPOPULATIONS
# ============================================================================

"""
create_neutrophil_subpopulation(cell_count, pathology_multiplier=1.0)

Create neutrophil subpopulation with default parameters.
"""
function create_neutrophil_subpopulation(
    cell_count::Float64;
    pathology_multiplier::Float64=1.0,
    fractal_df_edge::Float64=1.7,
    fractal_df_dist::Float64=1.5
)::WhiteBloodCellSubpopulation
    
    adjusted_count = cell_count * pathology_multiplier
    volume_fraction = (adjusted_count * VOLUME_NEUTROPHIL) / 1e15  # fL → L per L blood
    
    return WhiteBloodCellSubpopulation(
        "neutrophil",
        adjusted_count,
        VOLUME_NEUTROPHIL,
        volume_fraction,
        0.7,  # Velocity factor (WBCs slower than plasma)
        1.0,  # Partition coefficient (default, drug-specific)
        
        # Binding parameters (drug-specific, empty dicts to be filled)
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        
        # Internalization (drug-specific)
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        
        # Intracellular compartments
        0.10,  # Lysosome fraction (10% of cell volume)
        PH_LYSOSOME,
        
        # Neutrophil-specific: azurophilic granules
        0.05,  # 5% of cell volume
        PH_AZUROPHILIC_GRANULE,
        
        # Exchange
        0.5,  # Exchange rate (1/h) - slower than RBC
        
        # Fractal parameters
        fractal_df_edge,
        fractal_df_dist,
        
        # Pathology
        pathology_multiplier
    )
end

"""
create_lymphocyte_T_subpopulation(cell_count, pathology_multiplier=1.0)

Create T lymphocyte subpopulation.
"""
function create_lymphocyte_T_subpopulation(
    cell_count::Float64;
    pathology_multiplier::Float64=1.0,
    fractal_df_edge::Float64=1.75,
    fractal_df_dist::Float64=1.6
)::WhiteBloodCellSubpopulation
    
    adjusted_count = cell_count * pathology_multiplier
    volume_fraction = (adjusted_count * VOLUME_LYMPHOCYTE_SMALL) / 1e15
    
    return WhiteBloodCellSubpopulation(
        "lymphocyte_T",
        adjusted_count,
        VOLUME_LYMPHOCYTE_SMALL,
        volume_fraction,
        0.75,  # Velocity factor
        1.0,   # Partition (default)
        
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        
        0.08,  # Lysosome fraction (smaller than neutrophils)
        PH_LYSOSOME,
        
        0.0,   # No azurophilic granules
        0.0,
        
        0.3,   # Exchange rate (slower - more stable)
        
        fractal_df_edge,
        fractal_df_dist,
        
        pathology_multiplier
    )
end

"""
create_lymphocyte_B_subpopulation(cell_count, pathology_multiplier=1.0)

Create B lymphocyte subpopulation (important for rituximab, etc.)
"""
function create_lymphocyte_B_subpopulation(
    cell_count::Float64;
    pathology_multiplier::Float64=1.0,
    fractal_df_edge::Float64=1.75,
    fractal_df_dist::Float64=1.6
)::WhiteBloodCellSubpopulation
    
    adjusted_count = cell_count * pathology_multiplier
    volume_fraction = (adjusted_count * VOLUME_LYMPHOCYTE_SMALL) / 1e15
    
    return WhiteBloodCellSubpopulation(
        "lymphocyte_B",
        adjusted_count,
        VOLUME_LYMPHOCYTE_SMALL,
        volume_fraction,
        0.75,
        1.0,
        
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        
        0.08,
        PH_LYSOSOME,
        
        0.0,
        0.0,
        
        0.3,
        
        fractal_df_edge,
        fractal_df_dist,
        
        pathology_multiplier
    )
end

"""
create_lymphocyte_NK_subpopulation(cell_count, pathology_multiplier=1.0)

Create Natural Killer (NK) cell subpopulation.
"""
function create_lymphocyte_NK_subpopulation(
    cell_count::Float64;
    pathology_multiplier::Float64=1.0,
    fractal_df_edge::Float64=1.75,
    fractal_df_dist::Float64=1.6
)::WhiteBloodCellSubpopulation
    
    adjusted_count = cell_count * pathology_multiplier
    volume_fraction = (adjusted_count * VOLUME_LYMPHOCYTE_LARGE) / 1e15  # NK cells are larger
    
    return WhiteBloodCellSubpopulation(
        "lymphocyte_NK",
        adjusted_count,
        VOLUME_LYMPHOCYTE_LARGE,
        volume_fraction,
        0.75,
        1.0,
        
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        
        0.10,  # More lysosomes (cytotoxic function)
        PH_LYSOSOME,
        
        0.0,
        0.0,
        
        0.3,
        
        fractal_df_edge,
        fractal_df_dist,
        
        pathology_multiplier
    )
end

"""
create_monocyte_subpopulation(cell_count, pathology_multiplier=1.0)

Create monocyte subpopulation (important for antimalarials, nanoparticles).
"""
function create_monocyte_subpopulation(
    cell_count::Float64;
    pathology_multiplier::Float64=1.0,
    fractal_df_edge::Float64=1.65,
    fractal_df_dist::Float64=1.5
)::WhiteBloodCellSubpopulation
    
    adjusted_count = cell_count * pathology_multiplier
    volume_fraction = (adjusted_count * VOLUME_MONOCYTE) / 1e15
    
    return WhiteBloodCellSubpopulation(
        "monocyte",
        adjusted_count,
        VOLUME_MONOCYTE,
        volume_fraction,
        0.7,
        1.0,
        
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        
        0.15,  # More lysosomes (phagocytic function)
        PH_LYSOSOME,
        
        0.0,
        0.0,
        
        0.4,
        
        fractal_df_edge,
        fractal_df_dist,
        
        pathology_multiplier
    )
end

"""
create_eosinophil_subpopulation(cell_count, pathology_multiplier=1.0)

Create eosinophil subpopulation.
"""
function create_eosinophil_subpopulation(
    cell_count::Float64;
    pathology_multiplier::Float64=1.0,
    fractal_df_edge::Float64=1.7,
    fractal_df_dist::Float64=1.55
)::WhiteBloodCellSubpopulation
    
    adjusted_count = cell_count * pathology_multiplier
    volume_fraction = (adjusted_count * VOLUME_EOSINOPHIL) / 1e15
    
    return WhiteBloodCellSubpopulation(
        "eosinophil",
        adjusted_count,
        VOLUME_EOSINOPHIL,
        volume_fraction,
        0.7,
        1.0,
        
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        
        0.12,  # Eosinophilic granules
        PH_LYSOSOME,
        
        0.0,
        0.0,
        
        0.35,
        
        fractal_df_edge,
        fractal_df_dist,
        
        pathology_multiplier
    )
end

"""
create_basophil_subpopulation(cell_count, pathology_multiplier=1.0)

Create basophil subpopulation (smallest population, least PK relevance).
"""
function create_basophil_subpopulation(
    cell_count::Float64;
    pathology_multiplier::Float64=1.0,
    fractal_df_edge::Float64=1.7,
    fractal_df_dist::Float64=1.55
)::WhiteBloodCellSubpopulation
    
    adjusted_count = cell_count * pathology_multiplier
    volume_fraction = (adjusted_count * VOLUME_BASOPHIL) / 1e15
    
    return WhiteBloodCellSubpopulation(
        "basophil",
        adjusted_count,
        VOLUME_BASOPHIL,
        volume_fraction,
        0.7,
        1.0,
        
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        
        0.08,
        PH_LYSOSOME,
        
        0.0,
        0.0,
        
        0.3,
        
        fractal_df_edge,
        fractal_df_dist,
        
        pathology_multiplier
    )
end

# ============================================================================
# CREATE COMPLETE WBC COMPARTMENT
# ============================================================================

"""
create_WBC_compartment(patient; pathology="normal", pathology_severity=0.0, fractal_params=Dict())

Create complete WBC compartment with all subpopulations.

Parameters:
- patient: PatientProfile.PatientData
- pathology: "normal", "leukocytosis", "leukopenia", "leukemia", "sepsis"
- pathology_severity: 0.0-1.0 scale
- fractal_params: Dict with fractal dimensions per subpopulation
"""
function create_WBC_compartment(
    patient::PatientProfile.PatientData;
    pathology::String="normal",
    pathology_severity::Float64=0.0,
    fractal_params::Dict{String, Dict{String, Float64}}=Dict()
)::WhiteBloodCellCompartment
    
    # Get pathology multipliers
    multipliers = get_pathology_multipliers(pathology, pathology_severity)
    
    # Extract fractal parameters (with defaults)
    get_fractal = (subpop_name, param) -> 
        haskey(fractal_params, subpop_name) && 
        haskey(fractal_params[subpop_name], param) ?
        fractal_params[subpop_name][param] : 
        (param == "df_edge" ? 1.7 : 1.5)
    
    # Create all subpopulations
    neutrophils = create_neutrophil_subpopulation(
        NORMAL_NEUTROPHILS * 1e6,  # Convert /μL to /L
        pathology_multiplier=multipliers["neutrophil"],
        fractal_df_edge=get_fractal("neutrophil", "df_edge"),
        fractal_df_dist=get_fractal("neutrophil", "df_distribution")
    )
    
    lymphocytes_T = create_lymphocyte_T_subpopulation(
        NORMAL_LYMPHOCYTES_T * 1e6,
        pathology_multiplier=multipliers["lymphocyte_T"],
        fractal_df_edge=get_fractal("lymphocyte_T", "df_edge"),
        fractal_df_dist=get_fractal("lymphocyte_T", "df_distribution")
    )
    
    lymphocytes_B = create_lymphocyte_B_subpopulation(
        NORMAL_LYMPHOCYTES_B * 1e6,
        pathology_multiplier=multipliers["lymphocyte_B"],
        fractal_df_edge=get_fractal("lymphocyte_B", "df_edge"),
        fractal_df_dist=get_fractal("lymphocyte_B", "df_distribution")
    )
    
    lymphocytes_NK = create_lymphocyte_NK_subpopulation(
        NORMAL_LYMPHOCYTES_NK * 1e6,
        pathology_multiplier=multipliers["lymphocyte_NK"],
        fractal_df_edge=get_fractal("lymphocyte_NK", "df_edge"),
        fractal_df_dist=get_fractal("lymphocyte_NK", "df_distribution")
    )
    
    monocytes = create_monocyte_subpopulation(
        NORMAL_MONOCYTES * 1e6,
        pathology_multiplier=multipliers["monocyte"],
        fractal_df_edge=get_fractal("monocyte", "df_edge"),
        fractal_df_dist=get_fractal("monocyte", "df_distribution")
    )
    
    eosinophils = create_eosinophil_subpopulation(
        NORMAL_EOSINOPHILS * 1e6,
        pathology_multiplier=multipliers["eosinophil"],
        fractal_df_edge=get_fractal("eosinophil", "df_edge"),
        fractal_df_dist=get_fractal("eosinophil", "df_distribution")
    )
    
    basophils = create_basophil_subpopulation(
        NORMAL_BASOPHILS * 1e6,
        pathology_multiplier=multipliers["basophil"],
        fractal_df_edge=get_fractal("basophil", "df_edge"),
        fractal_df_dist=get_fractal("basophil", "df_distribution")
    )
    
    # Calculate totals
    total_count = (
        neutrophils.cell_count +
        lymphocytes_T.cell_count +
        lymphocytes_B.cell_count +
        lymphocytes_NK.cell_count +
        monocytes.cell_count +
        eosinophils.cell_count +
        basophils.cell_count
    )
    
    total_volume_fraction = (
        neutrophils.volume_fraction +
        lymphocytes_T.volume_fraction +
        lymphocytes_B.volume_fraction +
        lymphocytes_NK.volume_fraction +
        monocytes.volume_fraction +
        eosinophils.volume_fraction +
        basophils.volume_fraction
    )
    
    return WhiteBloodCellCompartment(
        neutrophils,
        lymphocytes_T,
        lymphocytes_B,
        lymphocytes_NK,
        monocytes,
        eosinophils,
        basophils,
        pathology,
        pathology_severity,
        total_count,
        total_volume_fraction
    )
end

# ============================================================================
# PATHOLOGY MODELS
# ============================================================================

"""
get_pathology_multipliers(pathology, severity)

Get cell count multipliers for different pathologies.

Severity scale: 0.0 = normal, 1.0 = maximum severity
"""
function get_pathology_multipliers(pathology::String, severity::Float64)::Dict{String, Float64}
    
    multipliers = Dict{String, Float64}()
    
    if pathology == "normal"
        for subpop in ["neutrophil", "lymphocyte_T", "lymphocyte_B", "lymphocyte_NK",
                       "monocyte", "eosinophil", "basophil"]
            multipliers[subpop] = 1.0
        end
        
    elseif pathology == "leukocytosis"  # Infection
        # Neutrophils increase dramatically
        multipliers["neutrophil"] = 1.0 + severity * 19.0  # Up to 20×
        multipliers["lymphocyte_T"] = 1.0 + severity * 0.5
        multipliers["lymphocyte_B"] = 1.0
        multipliers["lymphocyte_NK"] = 1.0 + severity * 0.3
        multipliers["monocyte"] = 1.0 + severity * 2.0
        multipliers["eosinophil"] = 1.0
        multipliers["basophil"] = 1.0
        
    elseif pathology == "leukopenia"  # Chemotherapy
        # All decrease
        multipliers["neutrophil"] = 1.0 - severity * 0.9  # Down to 10%
        multipliers["lymphocyte_T"] = 1.0 - severity * 0.8
        multipliers["lymphocyte_B"] = 1.0 - severity * 0.9
        multipliers["lymphocyte_NK"] = 1.0 - severity * 0.7
        multipliers["monocyte"] = 1.0 - severity * 0.8
        multipliers["eosinophil"] = 1.0 - severity * 0.8
        multipliers["basophil"] = 1.0 - severity * 0.7
        
    elseif pathology == "leukemia"
        # Massive increase in blasts (modeled as increased lymphocytes)
        multipliers["neutrophil"] = 1.0 - severity * 0.5  # Suppressed
        multipliers["lymphocyte_T"] = 1.0 + severity * 99.0  # Up to 100×
        multipliers["lymphocyte_B"] = 1.0 + severity * 99.0
        multipliers["lymphocyte_NK"] = 1.0
        multipliers["monocyte"] = 1.0
        multipliers["eosinophil"] = 1.0
        multipliers["basophil"] = 1.0
        
    elseif pathology == "sepsis"
        # Neutrophils increase, lymphocytes decrease (lymphopenia)
        multipliers["neutrophil"] = 1.0 + severity * 9.0  # Up to 10×
        multipliers["lymphocyte_T"] = 1.0 - severity * 0.7  # Lymphopenia
        multipliers["lymphocyte_B"] = 1.0 - severity * 0.7
        multipliers["lymphocyte_NK"] = 1.0 - severity * 0.5
        multipliers["monocyte"] = 1.0 + severity * 1.0
        multipliers["eosinophil"] = 1.0 - severity * 0.5  # Eosinopenia
        multipliers["basophil"] = 1.0 - severity * 0.5
        
    else
        error("Unknown pathology: $(pathology)")
    end
    
    return multipliers
end

# ============================================================================
# INTEGRATION WITH FRACTAL BLOOD
# ============================================================================

"""
create_WBC_phases_for_fractal_blood(wbc_compartment)

Create BloodPhase objects for each WBC subpopulation to integrate with FractalBlood.
"""
function create_WBC_phases_for_fractal_blood(
    wbc_compartment::WhiteBloodCellCompartment
)::Vector{BloodPhase}
    
    phases = BloodPhase[]
    
    # Helper to create phase
    add_phase(subpop) = push!(phases, BloodPhase(
        "wbc_$(subpop.name)",
        subpop.volume_fraction,
        subpop.velocity_factor,
        subpop.partition_coefficient,
        subpop.exchange_rate_plasma
    ))
    
    # Add all subpopulations
    add_phase(wbc_compartment.neutrophils)
    add_phase(wbc_compartment.lymphocytes_T)
    add_phase(wbc_compartment.lymphocytes_B)
    add_phase(wbc_compartment.lymphocytes_NK)
    add_phase(wbc_compartment.monocytes)
    add_phase(wbc_compartment.eosinophils)
    add_phase(wbc_compartment.basophils)
    
    return phases
end

# ============================================================================
# FRACTAL MORPHOLOGY INTEGRATION
# ============================================================================

"""
calculate_partition_coefficient(subpop, drug_name, drug_pKa, drug_logP; 
                                use_fractal_correction=true)

Calculate partition coefficient considering fractal morphology.

Hypothesis: Lower df_edge → simpler membrane → higher permeability → higher partition
"""
function calculate_partition_coefficient(
    subpop::WhiteBloodCellSubpopulation,
    drug_name::String,
    drug_pKa::Float64,
    drug_logP::Float64;
    use_fractal_correction::Bool=true
)::Float64
    
    # Base partition (drug-specific, simplified model)
    base_partition = 1.0 + 0.5 * (10.0^drug_logP) / (1.0 + 10.0^drug_logP)
    
    # Fractal correction
    if use_fractal_correction
        # Hypothesis: df_edge < 1.7 → higher permeability
        df_factor = 1.0 + (1.7 - subpop.fractal_dimension_edge) * 0.3
        base_partition *= df_factor
    end
    
    # Ion trapping for basic drugs (lysosomes)
    if drug_pKa > 7.0  # Basic drug
        ion_trapping = (1.0 + 10.0^(drug_pKa - subpop.pH_lysosome)) / 
                       (1.0 + 10.0^(drug_pKa - PH_CYTOSOL))
        base_partition *= (1.0 + subpop.lysosome_volume_fraction * (ion_trapping - 1.0))
    end
    
    return base_partition
end

"""
calculate_internalization_rate(subpop, drug_name; use_fractal_correction=true)

Calculate internalization rate based on fractal morphology.

Hypothesis: Lower df_edge → simpler membrane → easier internalization
"""
function calculate_internalization_rate(
    subpop::WhiteBloodCellSubpopulation,
    drug_name::String;
    use_fractal_correction::Bool=true
)::Float64
    
    # Base rate (drug-specific, would be loaded from database)
    base_rate = 0.5  # 1/h (default)
    
    if use_fractal_correction
        # Lower df_edge → higher internalization
        df_correction = 1.0 + (1.7 - subpop.fractal_dimension_edge) * 0.4
        base_rate *= df_correction
    end
    
    return base_rate
end

"""
get_fractal_corrected_parameters(wbc_compartment, drug_name, drug_pKa, drug_logP)

Get all fractal-corrected parameters for a drug.
"""
function get_fractal_corrected_parameters(
    wbc_compartment::WhiteBloodCellCompartment,
    drug_name::String,
    drug_pKa::Float64,
    drug_logP::Float64
)::Dict{String, Dict{String, Float64}}
    
    results = Dict{String, Dict{String, Float64}}()
    
    for (name, subpop) in [
        ("neutrophil", wbc_compartment.neutrophils),
        ("lymphocyte_T", wbc_compartment.lymphocytes_T),
        ("lymphocyte_B", wbc_compartment.lymphocytes_B),
        ("lymphocyte_NK", wbc_compartment.lymphocytes_NK),
        ("monocyte", wbc_compartment.monocytes),
        ("eosinophil", wbc_compartment.eosinophils),
        ("basophil", wbc_compartment.basophils)
    ]
        results[name] = Dict(
            "partition_coefficient" => calculate_partition_coefficient(subpop, drug_name, drug_pKa, drug_logP),
            "internalization_rate" => calculate_internalization_rate(subpop, drug_name),
            "df_edge" => subpop.fractal_dimension_edge,
            "df_distribution" => subpop.fractal_dimension_distribution
        )
    end
    
    return results
end

end  # module WhiteBloodCells

