"""
Spleen & RES Clearance Module

Models drug clearance via the reticuloendothelial system (RES),
distinct from hepatic clearance.

Key Features:
- Splenic macrophage uptake
- Fc receptor-mediated clearance
- Complement-opsonized particle removal
- Kupffer cell contribution
- Splenectomy effects

Clinical Relevance:
- mAb clearance is 80% RES-mediated
- Splenectomy increases drug exposure 2-3×
- Important for opsonized nanoparticles
- Affects IgG-coated cell clearance

References:
- Davies B (2002) Splenic blood flow in health and disease
- Moghimi SM (2012) RES clearance of nanoparticles
- Bowdler AJ (2002) The Complete Spleen

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module SpleenRESClearance

using Statistics

export SpleenState, RESCapacity, MacrophagePool
export create_normal_spleen, create_disease_spleen
export calculate_res_clearance, calculate_splenic_uptake
export apply_splenectomy, calculate_fcr_mediated_clearance
export SPLEEN_PARAMETERS, RES_TISSUE_WEIGHTS

# ============================================================================
# CONSTANTS
# ============================================================================

# Spleen physiology
const NORMAL_SPLEEN_WEIGHT = 150.0        # grams
const NORMAL_SPLEEN_BLOOD_FLOW = 250.0    # mL/min (5% of CO)
const SPLENIC_TRANSIT_TIME = 30.0         # seconds
const RED_PULP_FRACTION = 0.75            # 75% red pulp
const WHITE_PULP_FRACTION = 0.25          # 25% white pulp

# RES tissue distribution
const LIVER_RES_FRACTION = 0.80           # Kupffer cells - major
const SPLEEN_RES_FRACTION = 0.10          # Splenic macrophages
const BONE_MARROW_RES_FRACTION = 0.05     # Bone marrow macrophages
const LUNG_RES_FRACTION = 0.03            # Pulmonary intravascular macrophages
const OTHER_RES_FRACTION = 0.02           # Lymph nodes, etc.

# Macrophage parameters
const KUPFFER_CELL_COUNT = 2.0e10         # Total in liver
const SPLENIC_MACROPHAGE_COUNT = 1.0e9    # Total in spleen
const MACROPHAGE_UPTAKE_RATE = 0.1        # 1/min per cell (particles)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    MacrophagePool

State of tissue macrophage population.

# Fields
- `cell_count::Float64`: Number of macrophages
- `activation_state::Float64`: 0-1 (1 = fully activated)
- `fcr_expression::Dict{Symbol, Float64}`: Fc receptor expression
- `saturation::Float64`: Current saturation (0-1)
- `uptake_capacity::Float64`: Max uptake rate (particles/min)
"""
mutable struct MacrophagePool
    cell_count::Float64
    activation_state::Float64
    fcr_expression::Dict{Symbol, Float64}
    saturation::Float64
    uptake_capacity::Float64

    function MacrophagePool(;
        cell_count = 1.0e9,
        activation_state = 0.5,
        fcr_expression = Dict(:FcγRI => 1.0, :FcγRIIa => 1.0, :FcγRIIIa => 1.0),
        saturation = 0.0,
        uptake_capacity = 1.0e12
    )
        new(cell_count, activation_state, fcr_expression, saturation, uptake_capacity)
    end
end

"""
    SpleenState

Splenic physiology state.

# Fields
- `weight::Float64`: Spleen weight (g)
- `blood_flow::Float64`: Blood flow (mL/min)
- `macrophages::MacrophagePool`: Splenic macrophages
- `filtration_efficiency::Float64`: 0-1
- `present::Bool`: False if splenectomy
- `condition::Symbol`: :normal, :splenomegaly, :hyposplenism, :asplenic
"""
struct SpleenState
    weight::Float64
    blood_flow::Float64
    macrophages::MacrophagePool
    filtration_efficiency::Float64
    present::Bool
    condition::Symbol

    function SpleenState(;
        weight = NORMAL_SPLEEN_WEIGHT,
        blood_flow = NORMAL_SPLEEN_BLOOD_FLOW,
        macrophages = MacrophagePool(cell_count=SPLENIC_MACROPHAGE_COUNT),
        filtration_efficiency = 0.3,
        present = true,
        condition = :normal
    )
        new(weight, blood_flow, macrophages, filtration_efficiency, present, condition)
    end
end

"""
    RESCapacity

Total RES capacity across tissues.

# Fields
- `liver::MacrophagePool`: Kupffer cells
- `spleen::MacrophagePool`: Splenic macrophages
- `bone_marrow::MacrophagePool`: BM macrophages
- `lung::MacrophagePool`: Pulmonary intravascular macrophages
- `total_capacity::Float64`: Combined capacity
"""
struct RESCapacity
    liver::MacrophagePool
    spleen::MacrophagePool
    bone_marrow::MacrophagePool
    lung::MacrophagePool
    total_capacity::Float64

    function RESCapacity(;
        liver = MacrophagePool(cell_count=KUPFFER_CELL_COUNT),
        spleen = MacrophagePool(cell_count=SPLENIC_MACROPHAGE_COUNT),
        bone_marrow = MacrophagePool(cell_count=5.0e8),
        lung = MacrophagePool(cell_count=1.0e8)
    )
        total = liver.uptake_capacity * LIVER_RES_FRACTION +
                spleen.uptake_capacity * SPLEEN_RES_FRACTION +
                bone_marrow.uptake_capacity * BONE_MARROW_RES_FRACTION +
                lung.uptake_capacity * LUNG_RES_FRACTION
        new(liver, spleen, bone_marrow, lung, total)
    end
end

# ============================================================================
# PARAMETERS
# ============================================================================

const SPLEEN_PARAMETERS = Dict{Symbol, Any}(
    :normal_weight => NORMAL_SPLEEN_WEIGHT,
    :normal_blood_flow => NORMAL_SPLEEN_BLOOD_FLOW,
    :transit_time => SPLENIC_TRANSIT_TIME,
    :extraction_ratio => 0.05,       # 5% first-pass extraction
    :macrophage_density => 1.0e7,    # cells/g tissue
    :filtration_pore_size => 3.0     # μm (filters rigid RBCs)
)

const RES_TISSUE_WEIGHTS = Dict{Symbol, Float64}(
    :liver => LIVER_RES_FRACTION,
    :spleen => SPLEEN_RES_FRACTION,
    :bone_marrow => BONE_MARROW_RES_FRACTION,
    :lung => LUNG_RES_FRACTION,
    :other => OTHER_RES_FRACTION
)

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

"""
    create_normal_spleen()

Create normal splenic state.
"""
function create_normal_spleen()
    return SpleenState()
end

"""
    create_disease_spleen(disease::Symbol)

Create disease-specific splenic state.

Diseases:
- :splenomegaly - Enlarged spleen
- :hypersplenism - Overactive spleen
- :functional_asplenia - Sickle cell, celiac
- :splenectomy - Surgical removal
- :cirrhosis - Portal hypertension, congestion
- :malaria - Tropical splenomegaly
- :lymphoma - Infiltrative
"""
function create_disease_spleen(disease::Symbol)
    if disease == :splenomegaly
        return SpleenState(
            weight = 500.0,  # 3× normal
            blood_flow = 500.0,
            macrophages = MacrophagePool(cell_count=3.0e9),
            filtration_efficiency = 0.5,
            present = true,
            condition = :splenomegaly
        )

    elseif disease == :hypersplenism
        return SpleenState(
            weight = 400.0,
            blood_flow = 600.0,
            macrophages = MacrophagePool(
                cell_count = 4.0e9,
                activation_state = 0.8
            ),
            filtration_efficiency = 0.7,
            present = true,
            condition = :hypersplenism
        )

    elseif disease == :functional_asplenia
        return SpleenState(
            weight = 100.0,
            blood_flow = 100.0,
            macrophages = MacrophagePool(cell_count=1.0e8),
            filtration_efficiency = 0.05,
            present = true,
            condition = :functional_asplenia
        )

    elseif disease == :splenectomy
        return SpleenState(
            weight = 0.0,
            blood_flow = 0.0,
            macrophages = MacrophagePool(cell_count=0.0),
            filtration_efficiency = 0.0,
            present = false,
            condition = :asplenic
        )

    elseif disease == :cirrhosis
        # Portal hypertension → congestion
        return SpleenState(
            weight = 350.0,
            blood_flow = 400.0,
            macrophages = MacrophagePool(cell_count=2.5e9),
            filtration_efficiency = 0.25,  # Reduced due to congestion
            present = true,
            condition = :cirrhosis
        )

    elseif disease == :malaria
        return SpleenState(
            weight = 800.0,  # Tropical splenomegaly
            blood_flow = 600.0,
            macrophages = MacrophagePool(
                cell_count = 5.0e9,
                activation_state = 0.9
            ),
            filtration_efficiency = 0.6,
            present = true,
            condition = :malaria
        )

    else
        return create_normal_spleen()
    end
end

# ============================================================================
# RES CLEARANCE CALCULATIONS
# ============================================================================

"""
    calculate_res_clearance(particle_size::Float64,
                            opsonization::Float64,
                            res::RESCapacity;
                            particle_concentration::Float64=1.0)

Calculate RES clearance rate for particles/complexes.

# Arguments
- `particle_size`: Hydrodynamic diameter (nm)
- `opsonization`: Degree of complement/IgG coating (0-1)
- `res`: RES capacity state
- `particle_concentration`: Particles/mL

# Returns
Dict with clearance rates and tissue distribution
"""
function calculate_res_clearance(particle_size::Float64,
                                  opsonization::Float64,
                                  res::RESCapacity;
                                  particle_concentration::Float64=1.0)
    # Size-dependent uptake (optimal ~100-500 nm for phagocytosis)
    size_factor = if particle_size < 50
        0.3  # Small particles less efficiently taken up
    elseif particle_size < 200
        1.0  # Optimal range
    elseif particle_size < 1000
        0.8
    else
        0.5  # Very large, less efficient
    end

    # Opsonization dramatically increases uptake
    opsonic_factor = 1.0 + 10.0 * opsonization  # Up to 11× with full opsonization

    # Calculate clearance by each tissue
    liver_cl = res.liver.uptake_capacity * LIVER_RES_FRACTION *
               size_factor * opsonic_factor * (1.0 - res.liver.saturation)

    spleen_cl = res.spleen.uptake_capacity * SPLEEN_RES_FRACTION *
                size_factor * opsonic_factor * (1.0 - res.spleen.saturation)

    bm_cl = res.bone_marrow.uptake_capacity * BONE_MARROW_RES_FRACTION *
            size_factor * opsonic_factor

    lung_cl = res.lung.uptake_capacity * LUNG_RES_FRACTION *
              size_factor * opsonic_factor

    total_cl = liver_cl + spleen_cl + bm_cl + lung_cl

    # Convert to clearance (L/h)
    # Assume uptake_capacity in particles/min, convert to mL/min equivalent
    cl_ml_min = total_cl / max(particle_concentration, 1.0)
    cl_l_h = cl_ml_min * 60.0 / 1000.0

    # Half-life
    vd_plasma = 3.0  # L (plasma volume)
    t_half = log(2) * vd_plasma / cl_l_h

    return Dict(
        "cl_total" => cl_l_h,
        "cl_liver" => liver_cl / total_cl * cl_l_h,
        "cl_spleen" => spleen_cl / total_cl * cl_l_h,
        "cl_bone_marrow" => bm_cl / total_cl * cl_l_h,
        "cl_lung" => lung_cl / total_cl * cl_l_h,
        "liver_fraction" => liver_cl / total_cl,
        "spleen_fraction" => spleen_cl / total_cl,
        "t_half_hours" => t_half,
        "size_factor" => size_factor,
        "opsonic_enhancement" => opsonic_factor
    )
end

"""
    calculate_splenic_uptake(drug_concentration::Float64,
                              spleen::SpleenState;
                              bound_to_cells::Bool=false,
                              opsonized::Bool=false)

Calculate drug uptake specifically by spleen.
"""
function calculate_splenic_uptake(drug_concentration::Float64,
                                   spleen::SpleenState;
                                   bound_to_cells::Bool=false,
                                   opsonized::Bool=false)
    if !spleen.present
        return Dict(
            "uptake_rate" => 0.0,
            "extraction_ratio" => 0.0,
            "clearance" => 0.0,
            "status" => :asplenic
        )
    end

    # Base extraction
    extraction = spleen.filtration_efficiency

    # Cell-bound drugs (e.g., bound to RBCs, platelets) filtered more
    if bound_to_cells
        extraction *= 2.0
    end

    # Opsonized particles avidly cleared
    if opsonized
        extraction *= 5.0
    end

    extraction = min(extraction, 0.95)  # Max 95% extraction

    # Calculate clearance
    blood_flow_l_h = spleen.blood_flow * 60.0 / 1000.0  # mL/min → L/h
    clearance = blood_flow_l_h * extraction

    # Uptake rate
    uptake_rate = drug_concentration * clearance  # amount/h

    return Dict(
        "uptake_rate" => uptake_rate,
        "extraction_ratio" => extraction,
        "clearance" => clearance,
        "blood_flow" => blood_flow_l_h,
        "macrophage_saturation" => spleen.macrophages.saturation,
        "status" => spleen.condition
    )
end

"""
    calculate_fcr_mediated_clearance(antibody_concentration::Float64,
                                      isotype::Symbol,
                                      res::RESCapacity)

Calculate Fc receptor-mediated clearance of antibodies.
"""
function calculate_fcr_mediated_clearance(antibody_concentration::Float64,
                                           isotype::Symbol,
                                           res::RESCapacity)
    # Fc receptor affinity by isotype
    fcr_affinity = if isotype == :igg1
        Dict(:FcγRI => 1.0, :FcγRIIa => 0.5, :FcγRIIIa => 0.8)
    elseif isotype == :igg2
        Dict(:FcγRI => 0.1, :FcγRIIa => 0.3, :FcγRIIIa => 0.1)
    elseif isotype == :igg3
        Dict(:FcγRI => 1.0, :FcγRIIa => 0.5, :FcγRIIIa => 1.0)
    elseif isotype == :igg4
        Dict(:FcγRI => 0.2, :FcγRIIa => 0.2, :FcγRIIIa => 0.05)
    else
        Dict(:FcγRI => 0.0, :FcγRIIa => 0.0, :FcγRIIIa => 0.0)
    end

    # Calculate uptake via each receptor
    fcr_cl = 0.0
    for (receptor, affinity) in fcr_affinity
        # Receptor expression on macrophages
        liver_expr = get(res.liver.fcr_expression, receptor, 0.0)
        spleen_expr = get(res.spleen.fcr_expression, receptor, 0.0)

        # Contribution
        fcr_cl += affinity * (liver_expr * LIVER_RES_FRACTION +
                              spleen_expr * SPLEEN_RES_FRACTION)
    end

    # Convert to clearance (L/h)
    # Base clearance for IgG ~ 0.2-0.3 L/day, FcR adds to this
    base_cl = 0.01  # L/h
    fcr_contribution = fcr_cl * 0.005  # Scale factor

    total_cl = base_cl + fcr_contribution

    return Dict(
        "cl_total" => total_cl,
        "cl_base" => base_cl,
        "cl_fcr" => fcr_contribution,
        "fcr_enhancement" => fcr_contribution / base_cl,
        "isotype" => isotype
    )
end

# ============================================================================
# SPLENECTOMY EFFECTS
# ============================================================================

"""
    apply_splenectomy(res::RESCapacity)

Model effects of splenectomy on RES capacity.
"""
function apply_splenectomy(res::RESCapacity)
    # Zero splenic contribution
    new_spleen = MacrophagePool(cell_count=0.0, uptake_capacity=0.0)

    # Liver compensates partially (hypertrophy)
    compensation = 1.3  # 30% increase
    new_liver = MacrophagePool(
        cell_count = res.liver.cell_count * compensation,
        activation_state = res.liver.activation_state,
        fcr_expression = res.liver.fcr_expression,
        saturation = res.liver.saturation,
        uptake_capacity = res.liver.uptake_capacity * compensation
    )

    return RESCapacity(
        liver = new_liver,
        spleen = new_spleen,
        bone_marrow = res.bone_marrow,
        lung = res.lung
    )
end

"""
    calculate_splenectomy_pk_effect()

Calculate PK changes after splenectomy.
"""
function calculate_splenectomy_pk_effect()
    # Based on clinical data
    # mAbs: 20-30% reduced clearance
    # Opsonized particles: 50-70% reduced clearance
    # Immune complexes: 40-60% reduced clearance

    return Dict(
        "mab_cl_reduction" => 0.25,       # 25% reduced CL
        "mab_exposure_increase" => 1.33,   # 33% higher AUC
        "mab_t_half_increase" => 1.25,     # 25% longer half-life
        "opsonized_cl_reduction" => 0.60,  # 60% reduced
        "immune_complex_cl_reduction" => 0.50,
        "rbc_bound_drug_effect" => 1.2,    # 20% higher RBC-bound drug
        "platelet_count_increase" => 1.3   # Post-splenectomy thrombocytosis
    )
end

# ============================================================================
# DISEASE STATE EFFECTS
# ============================================================================

"""
    calculate_disease_res_effect(disease::Symbol)

Calculate RES function changes in disease states.
"""
function calculate_disease_res_effect(disease::Symbol)
    if disease == :cirrhosis
        return Dict(
            "kupffer_function" => 0.5,     # Reduced
            "portal_shunting" => 0.3,       # 30% bypasses liver
            "splenic_sequestration" => 1.5, # Increased
            "overall_res" => 0.6
        )
    elseif disease == :sepsis
        return Dict(
            "kupffer_function" => 0.3,     # Overwhelmed
            "macrophage_activation" => 2.0,
            "cytokine_effect" => 1.5,
            "overall_res" => 0.5            # Impaired
        )
    elseif disease == :sickle_cell
        return Dict(
            "kupffer_function" => 1.2,     # Compensatory
            "functional_asplenia" => true,
            "splenic_function" => 0.1,
            "overall_res" => 0.7
        )
    elseif disease == :hiv
        return Dict(
            "kupffer_function" => 0.8,
            "splenic_function" => 0.7,
            "macrophage_infection" => true,
            "overall_res" => 0.6
        )
    else
        return Dict(
            "kupffer_function" => 1.0,
            "splenic_function" => 1.0,
            "overall_res" => 1.0
        )
    end
end

end # module SpleenRESClearance
