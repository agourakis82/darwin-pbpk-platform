"""
Immunoglobulin Isotype Effects Module

Models isotype-specific binding and clearance for IgG, IgM, IgA, IgE, IgD
and complement activation effects on drug disposition.

Key Features:
- IgG subclass differences (IgG1-4) - extends mAb PBPK
- IgM pentameric structure (900 kDa)
- IgA dimers and secretory component
- IgE high-affinity FcεRI binding
- Complement activation (C1q, C3b, C4b)
- Immune complex formation kinetics
- RES (reticuloendothelial) clearance

References:
- Vidarsson G et al. (2014) IgG subclasses and allotypes
- Woof JM (2013) Structure and function of IgA
- Schroeder HW (2010) Structure and function of immunoglobulins
- Ricklin D et al. (2010) Complement: a key system for immune surveillance

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module ImmunoglobulinIsotypes

using Statistics

export ImmunoglobulinProperties, ComplementSystem, ImmuneComplex
export create_igg_subclass, create_igm, create_iga, create_ige
export calculate_complement_activation, calculate_immune_complex_clearance
export calculate_isotype_clearance, calculate_fc_receptor_binding
export IMMUNOGLOBULIN_DATABASE, COMPLEMENT_PARAMETERS, FC_RECEPTOR_DATABASE

# ============================================================================
# CONSTANTS
# ============================================================================

# Molecular weights (kDa)
const MW_IGG = 150.0           # Monomeric IgG
const MW_IGM_MONOMER = 180.0   # IgM monomer
const MW_IGM_PENTAMER = 970.0  # IgM pentamer (5 monomers + J chain)
const MW_IGA_MONOMER = 160.0   # IgA monomer
const MW_IGA_DIMER = 385.0     # IgA dimer + J chain + secretory component
const MW_IGE = 188.0           # IgE
const MW_IGD = 184.0           # IgD

# Normal serum concentrations (mg/dL)
const NORMAL_IGG_TOTAL = 1200.0   # 700-1600 mg/dL
const NORMAL_IGG1 = 800.0         # 60-70% of total IgG
const NORMAL_IGG2 = 300.0         # 20-25%
const NORMAL_IGG3 = 60.0          # 4-8%
const NORMAL_IGG4 = 40.0          # 2-6%
const NORMAL_IGM = 120.0          # 40-230 mg/dL
const NORMAL_IGA = 200.0          # 70-400 mg/dL
const NORMAL_IGE = 0.02           # 0.004-0.08 mg/dL (very low)
const NORMAL_IGD = 3.0            # 0.3-40 mg/dL

# Half-lives (days)
const HALFLIFE_IGG1 = 21.0
const HALFLIFE_IGG2 = 21.0
const HALFLIFE_IGG3 = 7.0         # Shorter due to hinge flexibility
const HALFLIFE_IGG4 = 21.0
const HALFLIFE_IGM = 5.0          # Shorter, no FcRn recycling
const HALFLIFE_IGA = 5.0          # No FcRn recycling
const HALFLIFE_IGE = 2.0          # Very short in serum
const HALFLIFE_IGD = 3.0

# Complement activation efficiency (relative to IgG1 = 1.0)
const C1Q_BINDING_IGG1 = 1.0
const C1Q_BINDING_IGG2 = 0.3      # Poor C1q binding
const C1Q_BINDING_IGG3 = 1.1      # Highest
const C1Q_BINDING_IGG4 = 0.0      # No C1q binding
const C1Q_BINDING_IGM = 1.5       # Excellent (pentameric)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    ImmunoglobulinProperties

Complete characterization of an immunoglobulin.

# Fields
- `name::String`: Name/identifier
- `isotype::Symbol`: :igg1, :igg2, :igg3, :igg4, :igm, :iga, :ige, :igd
- `molecular_weight::Float64`: kDa
- `valency::Int`: Number of antigen binding sites (2 for IgG, 10 for IgM)
- `half_life::Float64`: Serum half-life (days)
- `c1q_binding::Float64`: Complement C1q binding (relative)
- `fcrn_binding::Float64`: FcRn binding affinity (relative to IgG1)
- `fcr_binding::Dict{Symbol, Float64}`: Fc receptor binding affinities
- `j_chain::Bool`: Contains J chain (IgM, dimeric IgA)
- `secretory_component::Bool`: Has SC (secretory IgA)
- `glycosylation_sites::Int`: Number of N-glycosylation sites
"""
struct ImmunoglobulinProperties
    name::String
    isotype::Symbol
    molecular_weight::Float64
    valency::Int
    half_life::Float64
    c1q_binding::Float64
    fcrn_binding::Float64
    fcr_binding::Dict{Symbol, Float64}
    j_chain::Bool
    secretory_component::Bool
    glycosylation_sites::Int

    function ImmunoglobulinProperties(name::String;
        isotype::Symbol = :igg1,
        molecular_weight::Float64 = MW_IGG,
        valency::Int = 2,
        half_life::Float64 = 21.0,
        c1q_binding::Float64 = 1.0,
        fcrn_binding::Float64 = 1.0,
        fcr_binding::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
        j_chain::Bool = false,
        secretory_component::Bool = false,
        glycosylation_sites::Int = 1
    )
        new(name, isotype, molecular_weight, valency, half_life,
            c1q_binding, fcrn_binding, fcr_binding, j_chain,
            secretory_component, glycosylation_sites)
    end
end

"""
    ComplementSystem

Complement cascade parameters.

# Fields
- `c1q_concentration::Float64`: C1q (μg/mL, normal ~70)
- `c3_concentration::Float64`: C3 (mg/dL, normal ~100-200)
- `c4_concentration::Float64`: C4 (mg/dL, normal ~20-50)
- `classical_pathway_activity::Float64`: 0-1
- `alternative_pathway_activity::Float64`: 0-1
- `mannose_binding_lectin::Float64`: MBL (μg/mL)
- `c1_inhibitor::Float64`: C1-INH (mg/dL, normal ~20-30)
"""
struct ComplementSystem
    c1q_concentration::Float64
    c3_concentration::Float64
    c4_concentration::Float64
    classical_pathway_activity::Float64
    alternative_pathway_activity::Float64
    mannose_binding_lectin::Float64
    c1_inhibitor::Float64

    function ComplementSystem(;
        c1q_concentration = 70.0,
        c3_concentration = 150.0,
        c4_concentration = 35.0,
        classical_pathway_activity = 1.0,
        alternative_pathway_activity = 1.0,
        mannose_binding_lectin = 2.0,
        c1_inhibitor = 25.0
    )
        new(c1q_concentration, c3_concentration, c4_concentration,
            classical_pathway_activity, alternative_pathway_activity,
            mannose_binding_lectin, c1_inhibitor)
    end
end

"""
    ImmuneComplex

Immune complex (antigen-antibody) properties.

# Fields
- `antibody::ImmunoglobulinProperties`: Antibody component
- `antigen_name::String`: Antigen identifier
- `antigen_mw::Float64`: Antigen molecular weight (kDa)
- `stoichiometry::Float64`: Ag:Ab ratio
- `complex_size::Float64`: Approximate complex size (kDa)
- `complement_coated::Bool`: C3b/C4b opsonized
- `formation_constant::Float64`: Kd for complex formation (nM)
"""
struct ImmuneComplex
    antibody::ImmunoglobulinProperties
    antigen_name::String
    antigen_mw::Float64
    stoichiometry::Float64
    complex_size::Float64
    complement_coated::Bool
    formation_constant::Float64

    function ImmuneComplex(antibody::ImmunoglobulinProperties, antigen_name::String;
        antigen_mw::Float64 = 50.0,
        stoichiometry::Float64 = 1.0,
        complement_coated::Bool = false,
        formation_constant::Float64 = 1.0
    )
        # Calculate complex size
        complex_size = antibody.molecular_weight + stoichiometry * antigen_mw
        new(antibody, antigen_name, antigen_mw, stoichiometry,
            complex_size, complement_coated, formation_constant)
    end
end

# ============================================================================
# FC RECEPTOR DATABASE
# ============================================================================

"""
Fc receptor binding affinities and cellular expression.
"""
const FC_RECEPTOR_DATABASE = Dict{Symbol, Dict{Symbol, Any}}(
    # High-affinity IgG receptor
    :FcγRI => Dict(
        :affinity_igg1 => 1e-9,   # Kd (M)
        :affinity_igg2 => 1e-7,
        :affinity_igg3 => 1e-9,
        :affinity_igg4 => 1e-8,
        :cells => [:monocytes, :macrophages, :dendritic_cells],
        :function => :phagocytosis
    ),

    # Low-affinity IgG receptors
    :FcγRIIa => Dict(
        :affinity_igg1 => 5e-7,
        :affinity_igg2 => 1e-6,
        :affinity_igg3 => 1e-7,
        :affinity_igg4 => 5e-7,
        :cells => [:monocytes, :macrophages, :neutrophils, :platelets],
        :function => :phagocytosis
    ),

    :FcγRIIb => Dict(
        :affinity_igg1 => 1e-6,
        :affinity_igg2 => 2e-6,
        :affinity_igg3 => 5e-7,
        :affinity_igg4 => 1e-6,
        :cells => [:b_cells, :mast_cells, :macrophages],
        :function => :inhibition
    ),

    :FcγRIIIa => Dict(
        :affinity_igg1 => 2e-7,
        :affinity_igg2 => 1e-5,
        :affinity_igg3 => 1e-7,
        :affinity_igg4 => 2e-6,
        :cells => [:nk_cells, :macrophages, :monocytes],
        :function => :adcc
    ),

    # IgA receptor
    :FcαRI => Dict(
        :affinity_iga => 1e-7,
        :cells => [:neutrophils, :monocytes, :macrophages],
        :function => :phagocytosis
    ),

    # High-affinity IgE receptor
    :FcεRI => Dict(
        :affinity_ige => 1e-10,  # Very high affinity
        :cells => [:mast_cells, :basophils],
        :function => :degranulation
    ),

    # Low-affinity IgE receptor
    :FcεRII => Dict(
        :affinity_ige => 1e-6,
        :cells => [:b_cells, :monocytes, :eosinophils],
        :function => :regulation
    )
)

# ============================================================================
# COMPLEMENT PARAMETERS
# ============================================================================

const COMPLEMENT_PARAMETERS = Dict{Symbol, Any}(
    :c1q_kd_igg => 50.0,        # nM (for IgG hexamer)
    :c1q_kd_igm => 10.0,        # nM (high affinity)
    :c3_convertase_rate => 0.1,  # 1/s
    :c3b_deposition_rate => 1.0, # molecules/s per C3 convertase
    :mac_formation_rate => 0.01, # 1/s
    :decay_accelerating_factor => 0.5,  # DAF effect
    :factor_h_regulation => 0.3  # Factor H effect
)

# ============================================================================
# IMMUNOGLOBULIN DATABASE
# ============================================================================

const IMMUNOGLOBULIN_DATABASE = Dict{String, ImmunoglobulinProperties}(
    # IgG subclasses
    "igg1_reference" => ImmunoglobulinProperties("igg1_reference";
        isotype = :igg1,
        molecular_weight = MW_IGG,
        valency = 2,
        half_life = HALFLIFE_IGG1,
        c1q_binding = C1Q_BINDING_IGG1,
        fcrn_binding = 1.0,
        fcr_binding = Dict(:FcγRI => 1.0, :FcγRIIa => 1.0, :FcγRIIIa => 1.0),
        glycosylation_sites = 1
    ),

    "igg2_reference" => ImmunoglobulinProperties("igg2_reference";
        isotype = :igg2,
        molecular_weight = MW_IGG,
        valency = 2,
        half_life = HALFLIFE_IGG2,
        c1q_binding = C1Q_BINDING_IGG2,
        fcrn_binding = 0.9,
        fcr_binding = Dict(:FcγRI => 0.1, :FcγRIIa => 0.5, :FcγRIIIa => 0.1),
        glycosylation_sites = 1
    ),

    "igg3_reference" => ImmunoglobulinProperties("igg3_reference";
        isotype = :igg3,
        molecular_weight = MW_IGG,
        valency = 2,
        half_life = HALFLIFE_IGG3,
        c1q_binding = C1Q_BINDING_IGG3,
        fcrn_binding = 0.8,  # Reduced due to hinge region
        fcr_binding = Dict(:FcγRI => 1.1, :FcγRIIa => 1.0, :FcγRIIIa => 1.2),
        glycosylation_sites = 1
    ),

    "igg4_reference" => ImmunoglobulinProperties("igg4_reference";
        isotype = :igg4,
        molecular_weight = MW_IGG,
        valency = 2,
        half_life = HALFLIFE_IGG4,
        c1q_binding = C1Q_BINDING_IGG4,
        fcrn_binding = 1.0,
        fcr_binding = Dict(:FcγRI => 0.3, :FcγRIIa => 0.3, :FcγRIIIa => 0.1),
        glycosylation_sites = 1
    ),

    # IgM
    "igm_pentamer" => ImmunoglobulinProperties("igm_pentamer";
        isotype = :igm,
        molecular_weight = MW_IGM_PENTAMER,
        valency = 10,  # Pentameric, but steric constraints limit to ~5 effective
        half_life = HALFLIFE_IGM,
        c1q_binding = C1Q_BINDING_IGM,
        fcrn_binding = 0.0,  # No FcRn binding
        fcr_binding = Dict{Symbol, Float64}(),  # IgM has separate receptor
        j_chain = true,
        glycosylation_sites = 5
    ),

    # IgA
    "iga_monomer" => ImmunoglobulinProperties("iga_monomer";
        isotype = :iga,
        molecular_weight = MW_IGA_MONOMER,
        valency = 2,
        half_life = HALFLIFE_IGA,
        c1q_binding = 0.0,  # No complement activation
        fcrn_binding = 0.0,
        fcr_binding = Dict(:FcαRI => 1.0),
        glycosylation_sites = 2
    ),

    "iga_dimer" => ImmunoglobulinProperties("iga_dimer";
        isotype = :iga,
        molecular_weight = MW_IGA_DIMER,
        valency = 4,
        half_life = HALFLIFE_IGA,
        c1q_binding = 0.0,
        fcrn_binding = 0.0,
        fcr_binding = Dict(:FcαRI => 1.5),
        j_chain = true,
        secretory_component = true,
        glycosylation_sites = 4
    ),

    # IgE
    "ige_reference" => ImmunoglobulinProperties("ige_reference";
        isotype = :ige,
        molecular_weight = MW_IGE,
        valency = 2,
        half_life = HALFLIFE_IGE,
        c1q_binding = 0.0,
        fcrn_binding = 0.0,
        fcr_binding = Dict(:FcεRI => 1.0, :FcεRII => 1.0),
        glycosylation_sites = 6
    ),

    # IgD
    "igd_reference" => ImmunoglobulinProperties("igd_reference";
        isotype = :igd,
        molecular_weight = MW_IGD,
        valency = 2,
        half_life = HALFLIFE_IGD,
        c1q_binding = 0.0,
        fcrn_binding = 0.0,
        fcr_binding = Dict{Symbol, Float64}(),
        glycosylation_sites = 3
    )
)

# ============================================================================
# ISOTYPE FACTORY FUNCTIONS
# ============================================================================

"""
    create_igg_subclass(name::String, subclass::Int; kwargs...)

Create IgG of specific subclass (1-4).
"""
function create_igg_subclass(name::String, subclass::Int; kwargs...)
    if subclass == 1
        return ImmunoglobulinProperties(name;
            isotype = :igg1,
            molecular_weight = MW_IGG,
            valency = 2,
            half_life = HALFLIFE_IGG1,
            c1q_binding = C1Q_BINDING_IGG1,
            fcrn_binding = 1.0,
            kwargs...
        )
    elseif subclass == 2
        return ImmunoglobulinProperties(name;
            isotype = :igg2,
            molecular_weight = MW_IGG,
            valency = 2,
            half_life = HALFLIFE_IGG2,
            c1q_binding = C1Q_BINDING_IGG2,
            fcrn_binding = 0.9,
            kwargs...
        )
    elseif subclass == 3
        return ImmunoglobulinProperties(name;
            isotype = :igg3,
            molecular_weight = MW_IGG,
            valency = 2,
            half_life = HALFLIFE_IGG3,
            c1q_binding = C1Q_BINDING_IGG3,
            fcrn_binding = 0.8,
            kwargs...
        )
    elseif subclass == 4
        return ImmunoglobulinProperties(name;
            isotype = :igg4,
            molecular_weight = MW_IGG,
            valency = 2,
            half_life = HALFLIFE_IGG4,
            c1q_binding = C1Q_BINDING_IGG4,
            fcrn_binding = 1.0,
            kwargs...
        )
    else
        error("Invalid IgG subclass: $subclass. Must be 1-4.")
    end
end

"""
    create_igm(name::String; pentameric::Bool=true, kwargs...)

Create IgM (pentameric by default).
"""
function create_igm(name::String; pentameric::Bool=true, kwargs...)
    mw = pentameric ? MW_IGM_PENTAMER : MW_IGM_MONOMER
    valency = pentameric ? 10 : 2

    return ImmunoglobulinProperties(name;
        isotype = :igm,
        molecular_weight = mw,
        valency = valency,
        half_life = HALFLIFE_IGM,
        c1q_binding = C1Q_BINDING_IGM,
        fcrn_binding = 0.0,
        j_chain = pentameric,
        glycosylation_sites = pentameric ? 5 : 1,
        kwargs...
    )
end

"""
    create_iga(name::String; dimeric::Bool=false, secretory::Bool=false, kwargs...)

Create IgA (monomeric, dimeric, or secretory).
"""
function create_iga(name::String; dimeric::Bool=false, secretory::Bool=false, kwargs...)
    if secretory
        dimeric = true  # Secretory IgA is always dimeric
    end

    mw = dimeric ? MW_IGA_DIMER : MW_IGA_MONOMER
    valency = dimeric ? 4 : 2

    return ImmunoglobulinProperties(name;
        isotype = :iga,
        molecular_weight = mw,
        valency = valency,
        half_life = HALFLIFE_IGA,
        c1q_binding = 0.0,
        fcrn_binding = 0.0,
        j_chain = dimeric,
        secretory_component = secretory,
        glycosylation_sites = dimeric ? 4 : 2,
        kwargs...
    )
end

"""
    create_ige(name::String; kwargs...)

Create IgE.
"""
function create_ige(name::String; kwargs...)
    return ImmunoglobulinProperties(name;
        isotype = :ige,
        molecular_weight = MW_IGE,
        valency = 2,
        half_life = HALFLIFE_IGE,
        c1q_binding = 0.0,
        fcrn_binding = 0.0,
        glycosylation_sites = 6,
        kwargs...
    )
end

# ============================================================================
# COMPLEMENT ACTIVATION
# ============================================================================

"""
    calculate_complement_activation(ig::ImmunoglobulinProperties,
                                     ig_concentration::Float64,
                                     complement::ComplementSystem;
                                     antigen_bound::Bool=true)

Calculate complement activation by immunoglobulin.

# Arguments
- `ig`: Immunoglobulin properties
- `ig_concentration`: Antibody concentration (nM)
- `complement`: Complement system state
- `antigen_bound`: Whether antibody is bound to antigen (required for activation)

# Returns
Dict with complement activation metrics
"""
function calculate_complement_activation(ig::ImmunoglobulinProperties,
                                          ig_concentration::Float64,
                                          complement::ComplementSystem;
                                          antigen_bound::Bool=true)
    # No activation without antigen binding (for most isotypes)
    if !antigen_bound && ig.isotype != :igm
        return Dict(
            "classical_pathway" => 0.0,
            "c3_convertase_formed" => 0.0,
            "c3b_deposited" => 0.0,
            "mac_formation" => 0.0,
            "activation_status" => :none
        )
    end

    # C1q binding efficiency
    c1q_binding = ig.c1q_binding

    # No activation for non-complement-fixing isotypes
    if c1q_binding ≤ 0
        return Dict(
            "classical_pathway" => 0.0,
            "c3_convertase_formed" => 0.0,
            "c3b_deposited" => 0.0,
            "mac_formation" => 0.0,
            "activation_status" => :none
        )
    end

    # Classical pathway activation
    # Requires hexameric IgG or pentameric IgM for efficient C1q binding
    c1q_saturation = ig_concentration / (COMPLEMENT_PARAMETERS[:c1q_kd_igg] + ig_concentration)

    # IgM is much more efficient
    if ig.isotype == :igm
        c1q_saturation = ig_concentration / (COMPLEMENT_PARAMETERS[:c1q_kd_igm] + ig_concentration)
    end

    # C1 inhibitor effect
    c1_inh_effect = complement.c1_inhibitor / 25.0  # Normalized to normal
    c1q_effective = c1q_saturation * c1q_binding / c1_inh_effect

    # C3 convertase formation
    c3_convertase = c1q_effective * complement.classical_pathway_activity

    # C3b deposition
    c3_available = complement.c3_concentration / 150.0  # Normalized
    c3b_deposited = c3_convertase * c3_available * COMPLEMENT_PARAMETERS[:c3b_deposition_rate]

    # MAC formation
    c4_available = complement.c4_concentration / 35.0
    mac = c3b_deposited * c4_available * COMPLEMENT_PARAMETERS[:mac_formation_rate]

    # Determine activation status
    activation_status = if mac > 0.5
        :strong
    elseif mac > 0.1
        :moderate
    elseif c3b_deposited > 0.1
        :weak
    else
        :minimal
    end

    return Dict(
        "classical_pathway" => c1q_effective,
        "c3_convertase_formed" => c3_convertase,
        "c3b_deposited" => c3b_deposited,
        "mac_formation" => mac,
        "activation_status" => activation_status,
        "c1_inhibitor_effect" => c1_inh_effect
    )
end

# ============================================================================
# FC RECEPTOR BINDING
# ============================================================================

"""
    calculate_fc_receptor_binding(ig::ImmunoglobulinProperties,
                                   ig_concentration::Float64)

Calculate Fc receptor occupancy for immunoglobulin.

# Returns
Dict with receptor-specific binding fractions
"""
function calculate_fc_receptor_binding(ig::ImmunoglobulinProperties,
                                        ig_concentration::Float64)
    results = Dict{Symbol, Float64}()

    for (receptor, params) in FC_RECEPTOR_DATABASE
        # Get affinity for this isotype
        affinity_key = Symbol("affinity_", lowercase(string(ig.isotype)[1:3]))

        if haskey(params, affinity_key)
            kd = params[affinity_key]
            # Calculate occupancy
            occupancy = ig_concentration * 1e-9 / (kd + ig_concentration * 1e-9)

            # Apply Fc binding modifier from immunoglobulin
            if haskey(ig.fcr_binding, receptor)
                occupancy *= ig.fcr_binding[receptor]
            end

            results[receptor] = occupancy
        end
    end

    return results
end

# ============================================================================
# ISOTYPE-SPECIFIC CLEARANCE
# ============================================================================

"""
    calculate_isotype_clearance(ig::ImmunoglobulinProperties,
                                 ig_concentration::Float64;
                                 fcrn_saturation::Float64=0.0,
                                 complement_active::Bool=false,
                                 immune_complexed::Bool=false)

Calculate clearance rate based on isotype-specific mechanisms.

# Returns
Dict with clearance components (1/day)
"""
function calculate_isotype_clearance(ig::ImmunoglobulinProperties,
                                      ig_concentration::Float64;
                                      fcrn_saturation::Float64=0.0,
                                      complement_active::Bool=false,
                                      immune_complexed::Bool=false)
    # Base catabolism rate
    ke_base = log(2) / ig.half_life  # 1/day

    # FcRn-mediated protection
    fcrn_protection = 0.0
    if ig.fcrn_binding > 0
        # FcRn protects from degradation
        fcrn_available = 1.0 - fcrn_saturation
        fcrn_protection = ig.fcrn_binding * fcrn_available * 0.7  # Up to 70% protection
    end

    # Adjusted catabolism
    ke_catabolism = ke_base * (1.0 - fcrn_protection)

    # Fc receptor-mediated clearance
    fc_binding = calculate_fc_receptor_binding(ig, ig_concentration)
    ke_fcr = 0.0
    for (receptor, occupancy) in fc_binding
        # Different receptors have different clearance rates
        if receptor == :FcγRI
            ke_fcr += occupancy * 0.5  # High-affinity, moderate internalization
        elseif receptor == :FcγRIIa
            ke_fcr += occupancy * 0.3  # Phagocytosis
        elseif receptor == :FcγRIIIa
            ke_fcr += occupancy * 0.2  # ADCC (doesn't clear antibody directly)
        elseif receptor == :FcεRI
            ke_fcr += occupancy * 0.1  # IgE stays bound for weeks
        end
    end

    # Complement-mediated clearance
    ke_complement = 0.0
    if complement_active && ig.c1q_binding > 0
        ke_complement = ig.c1q_binding * 0.5  # Opsonization increases clearance
    end

    # Immune complex clearance (much faster)
    ke_complex = 0.0
    if immune_complexed
        ke_complex = 2.0  # t1/2 of ~8 hours for IC
    end

    # Total clearance
    ke_total = ke_catabolism + ke_fcr + ke_complement + ke_complex

    # Calculate effective half-life
    t_half_effective = log(2) / ke_total

    return Dict(
        "ke_total" => ke_total,
        "ke_catabolism" => ke_catabolism,
        "ke_fcr_mediated" => ke_fcr,
        "ke_complement" => ke_complement,
        "ke_immune_complex" => ke_complex,
        "t_half_effective" => t_half_effective,
        "fcrn_protection" => fcrn_protection
    )
end

# ============================================================================
# IMMUNE COMPLEX CLEARANCE
# ============================================================================

"""
    calculate_immune_complex_clearance(complex::ImmuneComplex,
                                        complex_concentration::Float64,
                                        complement::ComplementSystem)

Calculate immune complex clearance via complement and RES.

# Returns
Dict with clearance mechanisms and rates
"""
function calculate_immune_complex_clearance(complex::ImmuneComplex,
                                             complex_concentration::Float64,
                                             complement::ComplementSystem)
    ig = complex.antibody

    # Size-dependent filtration (very large complexes cleared faster by RES)
    size_factor = complex.complex_size / 150.0  # Normalized to IgG

    # C3b opsonization effect
    c3b_effect = 1.0
    if complex.complement_coated
        c3b_effect = 3.0  # Opsonization increases clearance 3-fold
    else
        # Calculate C3b deposition
        comp_act = calculate_complement_activation(ig, complex_concentration, complement;
                                                    antigen_bound=true)
        c3b_effect = 1.0 + 2.0 * comp_act["c3b_deposited"]
    end

    # Fc receptor-mediated uptake
    fc_uptake = 0.0
    fc_binding = calculate_fc_receptor_binding(ig, complex_concentration)
    for (receptor, occupancy) in fc_binding
        if receptor in [:FcγRI, :FcγRIIa]  # Phagocytic receptors
            fc_uptake += occupancy
        end
    end

    # RES clearance (liver, spleen)
    # Large complexes preferentially cleared by spleen
    liver_fraction = 0.7 - 0.2 * min(size_factor - 1.0, 1.0)
    spleen_fraction = 1.0 - liver_fraction

    # Base clearance rate for IC
    ke_base = 2.0  # 1/day (t1/2 ~8 hours)

    # Total clearance
    ke_total = ke_base * size_factor * c3b_effect * (1.0 + fc_uptake)

    return Dict(
        "ke_total" => ke_total,
        "t_half_hours" => log(2) / ke_total * 24.0,
        "c3b_enhancement" => c3b_effect,
        "fc_uptake" => fc_uptake,
        "liver_clearance_fraction" => liver_fraction,
        "spleen_clearance_fraction" => spleen_fraction,
        "size_factor" => size_factor
    )
end

# ============================================================================
# DISEASE STATE EFFECTS
# ============================================================================

"""
    apply_disease_state_ig(ig::ImmunoglobulinProperties, disease::Symbol)

Adjust immunoglobulin parameters for disease state.
"""
function apply_disease_state_ig(ig::ImmunoglobulinProperties, disease::Symbol)
    halflife_mod = 1.0
    fcrn_mod = 1.0

    if disease == :hypergammaglobulinemia
        # FcRn saturation reduces all IgG half-lives
        fcrn_mod = 0.7
        halflife_mod = 0.75
    elseif disease == :hypogammaglobulinemia
        # Less competition for FcRn
        halflife_mod = 1.2
    elseif disease == :c3_deficiency
        # Complement-dependent clearance reduced
        # (would need to modify complement system)
        halflife_mod = 1.1
    elseif disease == :fcrn_deficiency
        # Rare, causes hypercatabolism
        fcrn_mod = 0.0
        halflife_mod = 0.3
    elseif disease == :cirrhosis
        # Reduced hepatic RES function
        halflife_mod = 1.3
    elseif disease == :splenectomy
        # Reduced splenic clearance
        halflife_mod = 1.2
    elseif disease == :inflammation
        # Increased Fc receptor expression
        halflife_mod = 0.8
    end

    return ImmunoglobulinProperties(ig.name;
        isotype = ig.isotype,
        molecular_weight = ig.molecular_weight,
        valency = ig.valency,
        half_life = ig.half_life * halflife_mod,
        c1q_binding = ig.c1q_binding,
        fcrn_binding = ig.fcrn_binding * fcrn_mod,
        fcr_binding = ig.fcr_binding,
        j_chain = ig.j_chain,
        secretory_component = ig.secretory_component,
        glycosylation_sites = ig.glycosylation_sites
    )
end

end # module ImmunoglobulinIsotypes
