"""
Monoclonal Antibody (mAb) PBPK Module

Comprehensive PBPK model for therapeutic antibodies including:
- IgG subtype-specific behavior
- FcRn-mediated recycling
- Target-Mediated Drug Disposition (TMDD)
- Antigen binding kinetics
- Immunogenicity effects

Supported Therapeutics:
- Full-length IgG1, IgG2, IgG4
- Fab fragments
- BiTE/bispecific antibodies
- ADCs (Antibody-Drug Conjugates)
- Fc-fusion proteins

References:
- Dirks NL (2010) - Population PK of mAbs
- Shah DK (2012) - mAb PBPK modeling
- Dua P (2015) - FcRn and mAb PK
- FDA Guidance (2021) - PBPK for biologics

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module mAbPBPK

using Statistics

export mAbProperties, TargetProperties, TMDDParameters
export FcRnParameters, ImmunogenicityState
export create_igg1, create_igg2, create_igg4, create_fab
export calculate_tmdd_clearance, calculate_fcrn_recycling
export simulate_mab_pk, calculate_target_occupancy
export MAB_DATABASE, TARGET_DATABASE

# ============================================================================
# CONSTANTS
# ============================================================================

# IgG molecular weights (kDa)
const MW_IGG_FULL = 150.0           # Full-length IgG
const MW_FAB = 50.0                 # Fab fragment
const MW_FC = 50.0                  # Fc region
const MW_SCFV = 27.0                # Single-chain variable fragment

# Normal plasma IgG levels
const NORMAL_IGG_TOTAL = 12.0       # g/L (range 7-16)
const NORMAL_IGG1 = 8.0             # g/L (60-70% of total)
const NORMAL_IGG2 = 3.0             # g/L
const NORMAL_IGG3 = 0.5             # g/L
const NORMAL_IGG4 = 0.5             # g/L

# FcRn parameters
const FCRN_KD_IGG1 = 0.5            # μM (pH 6.0)
const FCRN_KD_IGG2 = 0.8            # μM
const FCRN_KD_IGG4 = 0.6            # μM
const FCRN_SATURATION_CONC = 50.0   # μg/mL (approximate)

# Physiological parameters
const VASCULAR_REFLECTION = 0.95    # Endothelial reflection coefficient
const LYMPH_FLOW_FRACTION = 0.002   # L/h per L tissue

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    mAbProperties

Complete characterization of a monoclonal antibody.

# Fields
- `name::String`: Drug name
- `igg_subclass::Symbol`: :igg1, :igg2, :igg3, :igg4, :fab, :fc_fusion
- `molecular_weight::Float64`: kDa
- `target::String`: Target antigen name
- `kon::Float64`: Association rate (1/M/s)
- `koff::Float64`: Dissociation rate (1/s)
- `kd::Float64`: Dissociation constant (nM)
- `fcrn_affinity::Float64`: Relative FcRn affinity (1.0 = wild-type IgG1)
- `effector_function::Bool`: Has ADCC/CDC activity
- `half_life_days::Float64`: Expected half-life (days)
- `glycosylation::Symbol`: :normal, :afucosylated, :aglycosylated
- `is_adc::Bool`: Is antibody-drug conjugate
- `dar::Float64`: Drug-to-antibody ratio (for ADCs)
"""
struct mAbProperties
    name::String
    igg_subclass::Symbol
    molecular_weight::Float64
    target::String
    kon::Float64
    koff::Float64
    kd::Float64
    fcrn_affinity::Float64
    effector_function::Bool
    half_life_days::Float64
    glycosylation::Symbol
    is_adc::Bool
    dar::Float64

    function mAbProperties(name;
        igg_subclass = :igg1,
        molecular_weight = MW_IGG_FULL,
        target = "unknown",
        kon = 1.0e5,
        koff = 1.0e-4,
        kd = nothing,
        fcrn_affinity = 1.0,
        effector_function = true,
        half_life_days = 21.0,
        glycosylation = :normal,
        is_adc = false,
        dar = 0.0
    )
        # Calculate Kd if not provided
        kd_calc = isnothing(kd) ? koff / kon * 1e9 : kd  # Convert to nM
        new(name, igg_subclass, molecular_weight, target,
            kon, koff, kd_calc, fcrn_affinity, effector_function,
            half_life_days, glycosylation, is_adc, dar)
    end
end

"""
    TargetProperties

Properties of the target antigen.

# Fields
- `name::String`: Target name (e.g., "CD20", "HER2", "PD-1")
- `expression_level::Float64`: Receptors/cell (typical range 10³-10⁶)
- `target_cells::Float64`: Cells/L expressing target
- `baseline_concentration::Float64`: Soluble target (nM) if applicable
- `turnover_rate::Float64`: Target turnover (1/day)
- `internalization_rate::Float64`: Receptor internalization (1/h)
- `shedding_rate::Float64`: Receptor shedding (1/h)
- `location::Symbol`: :membrane, :soluble, :both
- `tissue_distribution::Dict{Symbol, Float64}`: Tissue weights
"""
struct TargetProperties
    name::String
    expression_level::Float64
    target_cells::Float64
    baseline_concentration::Float64
    turnover_rate::Float64
    internalization_rate::Float64
    shedding_rate::Float64
    location::Symbol
    tissue_distribution::Dict{Symbol, Float64}

    function TargetProperties(name;
        expression_level = 1.0e5,
        target_cells = 1.0e9,
        baseline_concentration = 0.0,
        turnover_rate = 0.1,
        internalization_rate = 0.05,
        shedding_rate = 0.01,
        location = :membrane,
        tissue_distribution = Dict(:blood => 0.5, :tumor => 0.3, :lymph => 0.2)
    )
        new(name, expression_level, target_cells, baseline_concentration,
            turnover_rate, internalization_rate, shedding_rate,
            location, tissue_distribution)
    end
end

"""
    TMDDParameters

Parameters for Target-Mediated Drug Disposition.

# Fields
- `kint::Float64`: Internalization rate of drug-target complex (1/h)
- `kdeg::Float64`: Target degradation rate (1/h)
- `ksyn::Float64`: Target synthesis rate (nM/h)
- `target_baseline::Float64`: Baseline target concentration (nM)
- `nonlinear_range::Tuple{Float64, Float64}`: Concentration range for TMDD (nM)
"""
struct TMDDParameters
    kint::Float64
    kdeg::Float64
    ksyn::Float64
    target_baseline::Float64
    nonlinear_range::Tuple{Float64, Float64}

    function TMDDParameters(;
        kint = 0.05,
        kdeg = 0.01,
        ksyn = nothing,
        target_baseline = 1.0,
        nonlinear_range = (0.1, 100.0)
    )
        # ksyn = kdeg * target_baseline at steady state
        ksyn_calc = isnothing(ksyn) ? kdeg * target_baseline : ksyn
        new(kint, kdeg, ksyn_calc, target_baseline, nonlinear_range)
    end
end

"""
    FcRnParameters

FcRn-mediated recycling parameters.

# Fields
- `expression_level::Float64`: Relative FcRn expression (1.0 = normal)
- `recycling_efficiency::Float64`: Fraction recycled (0-1)
- `endosomal_ph::Float64`: Endosomal pH for binding
- `kd_ph6::Float64`: Kd at pH 6.0 (μM)
- `kd_ph7::Float64`: Kd at pH 7.4 (μM)
"""
struct FcRnParameters
    expression_level::Float64
    recycling_efficiency::Float64
    endosomal_ph::Float64
    kd_ph6::Float64
    kd_ph7::Float64

    function FcRnParameters(;
        expression_level = 1.0,
        recycling_efficiency = 0.7,
        endosomal_ph = 6.0,
        kd_ph6 = 0.5,
        kd_ph7 = 10.0
    )
        new(expression_level, recycling_efficiency, endosomal_ph, kd_ph6, kd_ph7)
    end
end

"""
    ImmunogenicityState

Immunogenicity (ADA) status.

# Fields
- `ada_positive::Bool`: Has developed ADA
- `ada_titer::Float64`: ADA titer (relative units)
- `ada_type::Symbol`: :binding, :neutralizing, :both
- `time_to_ada::Float64`: Time to ADA development (days)
- `clearance_multiplier::Float64`: CL increase due to ADA
"""
struct ImmunogenicityState
    ada_positive::Bool
    ada_titer::Float64
    ada_type::Symbol
    time_to_ada::Float64
    clearance_multiplier::Float64

    function ImmunogenicityState(;
        ada_positive = false,
        ada_titer = 0.0,
        ada_type = :none,
        time_to_ada = Inf,
        clearance_multiplier = 1.0
    )
        cm = ada_positive ? max(clearance_multiplier, 1.5) : 1.0
        new(ada_positive, ada_titer, ada_type, time_to_ada, cm)
    end
end

# ============================================================================
# MAB DATABASE
# ============================================================================

"""
Database of approved/common therapeutic antibodies.
"""
const MAB_DATABASE = Dict{String, mAbProperties}(
    # Anti-CD20 (B cells)
    "rituximab" => mAbProperties("rituximab";
        igg_subclass = :igg1,
        target = "CD20",
        kon = 1.5e5, koff = 3.0e-5, kd = 0.2,
        effector_function = true,
        half_life_days = 21.0
    ),
    "obinutuzumab" => mAbProperties("obinutuzumab";
        igg_subclass = :igg1,
        target = "CD20",
        kon = 2.0e5, koff = 2.0e-5, kd = 0.1,
        effector_function = true,
        glycosylation = :afucosylated,
        half_life_days = 28.0
    ),
    "ofatumumab" => mAbProperties("ofatumumab";
        igg_subclass = :igg1,
        target = "CD20",
        kon = 1.8e5, koff = 1.5e-5, kd = 0.08,
        half_life_days = 14.0
    ),

    # Anti-HER2
    "trastuzumab" => mAbProperties("trastuzumab";
        igg_subclass = :igg1,
        target = "HER2",
        kon = 2.0e5, koff = 1.8e-4, kd = 0.9,
        effector_function = true,
        half_life_days = 28.0
    ),
    "pertuzumab" => mAbProperties("pertuzumab";
        igg_subclass = :igg1,
        target = "HER2",
        kon = 1.5e5, koff = 2.5e-4, kd = 1.7,
        half_life_days = 18.0
    ),
    "trastuzumab_emtansine" => mAbProperties("trastuzumab_emtansine";
        igg_subclass = :igg1,
        target = "HER2",
        kon = 1.8e5, koff = 2.0e-4, kd = 1.1,
        is_adc = true, dar = 3.5,
        half_life_days = 4.0  # Shorter due to payload
    ),

    # Anti-PD-1/PD-L1 (Checkpoint inhibitors)
    "pembrolizumab" => mAbProperties("pembrolizumab";
        igg_subclass = :igg4,
        target = "PD-1",
        kon = 1.0e6, koff = 1.0e-4, kd = 0.1,
        effector_function = false,
        half_life_days = 25.0
    ),
    "nivolumab" => mAbProperties("nivolumab";
        igg_subclass = :igg4,
        target = "PD-1",
        kon = 8.0e5, koff = 2.0e-4, kd = 0.25,
        effector_function = false,
        half_life_days = 26.0
    ),
    "atezolizumab" => mAbProperties("atezolizumab";
        igg_subclass = :igg1,
        target = "PD-L1",
        kon = 5.0e5, koff = 3.0e-4, kd = 0.6,
        effector_function = false,  # Engineered to remove ADCC
        half_life_days = 27.0
    ),
    "durvalumab" => mAbProperties("durvalumab";
        igg_subclass = :igg1,
        target = "PD-L1",
        kon = 4.0e5, koff = 2.5e-4, kd = 0.6,
        half_life_days = 18.0
    ),

    # Anti-CTLA-4
    "ipilimumab" => mAbProperties("ipilimumab";
        igg_subclass = :igg1,
        target = "CTLA-4",
        kon = 3.0e5, koff = 4.0e-4, kd = 1.3,
        effector_function = true,
        half_life_days = 15.0
    ),

    # Anti-TNF
    "infliximab" => mAbProperties("infliximab";
        igg_subclass = :igg1,
        target = "TNF-alpha",
        kon = 1.5e5, koff = 1.0e-4, kd = 0.7,
        half_life_days = 9.0
    ),
    "adalimumab" => mAbProperties("adalimumab";
        igg_subclass = :igg1,
        target = "TNF-alpha",
        kon = 2.0e5, koff = 8.0e-5, kd = 0.4,
        half_life_days = 14.0
    ),
    "golimumab" => mAbProperties("golimumab";
        igg_subclass = :igg1,
        target = "TNF-alpha",
        kon = 1.8e5, koff = 1.2e-4, kd = 0.7,
        half_life_days = 14.0
    ),
    "certolizumab" => mAbProperties("certolizumab";
        igg_subclass = :fab,
        molecular_weight = MW_FAB + 40.0,  # PEGylated Fab
        target = "TNF-alpha",
        kon = 2.5e5, koff = 1.5e-4, kd = 0.6,
        fcrn_affinity = 0.0,  # No Fc region
        half_life_days = 14.0  # Extended by PEGylation
    ),

    # Anti-IL-6/IL-6R
    "tocilizumab" => mAbProperties("tocilizumab";
        igg_subclass = :igg1,
        target = "IL-6R",
        kon = 1.0e5, koff = 2.0e-4, kd = 2.0,
        half_life_days = 11.0
    ),
    "sarilumab" => mAbProperties("sarilumab";
        igg_subclass = :igg1,
        target = "IL-6R",
        kon = 1.5e5, koff = 1.0e-4, kd = 0.7,
        half_life_days = 21.0
    ),

    # Anti-VEGF
    "bevacizumab" => mAbProperties("bevacizumab";
        igg_subclass = :igg1,
        target = "VEGF-A",
        kon = 5.0e5, koff = 1.0e-4, kd = 0.2,
        half_life_days = 20.0
    ),
    "ranibizumab" => mAbProperties("ranibizumab";
        igg_subclass = :fab,
        molecular_weight = MW_FAB,
        target = "VEGF-A",
        kon = 6.0e5, koff = 8.0e-5, kd = 0.13,
        fcrn_affinity = 0.0,
        half_life_days = 0.4  # ~9 hours (no FcRn recycling)
    ),

    # Anti-EGFR
    "cetuximab" => mAbProperties("cetuximab";
        igg_subclass = :igg1,
        target = "EGFR",
        kon = 2.0e5, koff = 2.5e-4, kd = 1.3,
        half_life_days = 7.0
    ),
    "panitumumab" => mAbProperties("panitumumab";
        igg_subclass = :igg2,
        target = "EGFR",
        kon = 1.5e5, koff = 5.0e-5, kd = 0.3,
        effector_function = false,  # IgG2 has minimal ADCC
        half_life_days = 7.5
    ),

    # Anti-integrin
    "vedolizumab" => mAbProperties("vedolizumab";
        igg_subclass = :igg1,
        target = "alpha4beta7",
        kon = 1.0e5, koff = 1.5e-4, kd = 1.5,
        half_life_days = 25.0
    ),
    "natalizumab" => mAbProperties("natalizumab";
        igg_subclass = :igg4,
        target = "alpha4",
        kon = 8.0e4, koff = 2.0e-4, kd = 2.5,
        half_life_days = 16.0
    )
)

# ============================================================================
# TARGET DATABASE
# ============================================================================

const TARGET_DATABASE = Dict{String, TargetProperties}(
    "CD20" => TargetProperties("CD20";
        expression_level = 1.0e5,
        target_cells = 1.0e9,  # B cells in circulation
        location = :membrane,
        internalization_rate = 0.02,
        tissue_distribution = Dict(:blood => 0.05, :lymph => 0.7, :spleen => 0.2, :bone_marrow => 0.05)
    ),
    "HER2" => TargetProperties("HER2";
        expression_level = 1.0e6,  # High in HER2+ tumors
        target_cells = 1.0e10,
        location = :membrane,
        internalization_rate = 0.1,
        shedding_rate = 0.05,
        tissue_distribution = Dict(:tumor => 0.9, :other => 0.1)
    ),
    "PD-1" => TargetProperties("PD-1";
        expression_level = 5.0e4,
        target_cells = 1.0e9,  # Activated T cells
        location = :membrane,
        internalization_rate = 0.03,
        tissue_distribution = Dict(:blood => 0.1, :lymph => 0.5, :tumor => 0.4)
    ),
    "PD-L1" => TargetProperties("PD-L1";
        expression_level = 2.0e5,
        target_cells = 1.0e10,
        baseline_concentration = 0.1,  # Soluble PD-L1
        location = :both,
        tissue_distribution = Dict(:tumor => 0.6, :lymph => 0.3, :other => 0.1)
    ),
    "TNF-alpha" => TargetProperties("TNF-alpha";
        expression_level = 0.0,  # Soluble target
        baseline_concentration = 0.005,  # ~0.1 pg/mL normal, elevated in RA
        location = :soluble,
        turnover_rate = 0.5,  # Fast turnover
        tissue_distribution = Dict(:blood => 1.0)
    ),
    "VEGF-A" => TargetProperties("VEGF-A";
        expression_level = 0.0,
        baseline_concentration = 0.002,
        location = :soluble,
        turnover_rate = 0.3,
        tissue_distribution = Dict(:blood => 0.3, :tumor => 0.7)
    ),
    "IL-6R" => TargetProperties("IL-6R";
        expression_level = 1.0e4,
        target_cells = 5.0e9,
        baseline_concentration = 0.05,  # Soluble IL-6R
        location = :both,
        tissue_distribution = Dict(:blood => 0.3, :liver => 0.4, :other => 0.3)
    ),
    "EGFR" => TargetProperties("EGFR";
        expression_level = 5.0e5,
        target_cells = 1.0e10,
        location = :membrane,
        internalization_rate = 0.15,
        shedding_rate = 0.03,
        tissue_distribution = Dict(:tumor => 0.8, :skin => 0.15, :other => 0.05)
    )
)

# ============================================================================
# MAB FACTORIES
# ============================================================================

"""
    create_igg1(name::String; kwargs...)

Create an IgG1 antibody with default IgG1 properties.
"""
function create_igg1(name::String; target::String="unknown", kd::Float64=1.0, kwargs...)
    return mAbProperties(name;
        igg_subclass = :igg1,
        target = target,
        kd = kd,
        fcrn_affinity = 1.0,
        effector_function = true,
        half_life_days = 21.0,
        kwargs...
    )
end

"""
    create_igg2(name::String; kwargs...)

Create an IgG2 antibody (reduced effector function).
"""
function create_igg2(name::String; target::String="unknown", kd::Float64=1.0, kwargs...)
    return mAbProperties(name;
        igg_subclass = :igg2,
        target = target,
        kd = kd,
        fcrn_affinity = 0.9,
        effector_function = false,
        half_life_days = 21.0,
        kwargs...
    )
end

"""
    create_igg4(name::String; kwargs...)

Create an IgG4 antibody (minimal effector function, used for blocking).
"""
function create_igg4(name::String; target::String="unknown", kd::Float64=1.0, kwargs...)
    return mAbProperties(name;
        igg_subclass = :igg4,
        target = target,
        kd = kd,
        fcrn_affinity = 0.85,
        effector_function = false,
        half_life_days = 21.0,
        kwargs...
    )
end

"""
    create_fab(name::String; kwargs...)

Create a Fab fragment (no Fc, short half-life).
"""
function create_fab(name::String; target::String="unknown", kd::Float64=1.0, kwargs...)
    return mAbProperties(name;
        igg_subclass = :fab,
        molecular_weight = MW_FAB,
        target = target,
        kd = kd,
        fcrn_affinity = 0.0,  # No FcRn binding
        effector_function = false,
        half_life_days = 0.5,  # ~12 hours
        kwargs...
    )
end

# ============================================================================
# TMDD CALCULATIONS
# ============================================================================

"""
    calculate_tmdd_clearance(mab::mAbProperties,
                              target::TargetProperties,
                              mab_conc::Float64,
                              tmdd::TMDDParameters)

Calculate target-mediated drug disposition clearance.

# Arguments
- `mab`: Antibody properties
- `target`: Target properties
- `mab_conc`: Current mAb concentration (nM)
- `tmdd`: TMDD parameters

# Returns
Dict with clearance components
"""
function calculate_tmdd_clearance(mab::mAbProperties,
                                   target::TargetProperties,
                                   mab_conc::Float64,
                                   tmdd::TMDDParameters)
    # Free target at current mAb concentration
    # Using quasi-equilibrium approximation
    kd = mab.kd  # nM

    # Total target
    target_total = tmdd.target_baseline

    # Calculate free target and complex using Kd
    # L + R ⇌ LR
    # At equilibrium: [L][R]/[LR] = Kd

    # Quadratic solution for free target
    a = 1.0
    b = -(mab_conc + target_total + kd)
    c = mab_conc * target_total

    discriminant = b^2 - 4*a*c
    if discriminant < 0
        discriminant = 0
    end

    # Free target (smaller root)
    target_free = (-b - sqrt(discriminant)) / (2*a)
    target_free = max(target_free, 0.0)

    # Bound complex
    complex = target_total - target_free

    # Target-mediated clearance
    # CL_TMDD = kint × [LR] × Vd
    cl_tmdd = tmdd.kint * complex  # Relative units

    # Linear clearance (constant)
    cl_linear = 0.01  # L/h/kg baseline

    # Total clearance
    cl_total = cl_linear + cl_tmdd

    # Target occupancy
    occupancy = complex / target_total

    return Dict(
        "cl_linear" => cl_linear,
        "cl_tmdd" => cl_tmdd,
        "cl_total" => cl_total,
        "target_free" => target_free,
        "target_bound" => complex,
        "target_occupancy" => occupancy,
        "is_nonlinear" => mab_conc > tmdd.nonlinear_range[1] && mab_conc < tmdd.nonlinear_range[2]
    )
end

"""
    calculate_target_occupancy(mab::mAbProperties,
                                mab_conc::Float64,
                                target_conc::Float64)

Calculate fraction of target occupied by mAb.

Uses equilibrium binding equation.
"""
function calculate_target_occupancy(mab::mAbProperties,
                                     mab_conc::Float64,
                                     target_conc::Float64)
    kd = mab.kd

    # Occupancy = [L] / (Kd + [L])
    # This assumes excess mAb over target (reasonable for most therapeutics)
    occupancy = mab_conc / (kd + mab_conc)

    return occupancy
end

# ============================================================================
# FCRN RECYCLING
# ============================================================================

"""
    calculate_fcrn_recycling(mab::mAbProperties,
                              fcrn::FcRnParameters,
                              mab_conc::Float64)

Calculate FcRn-mediated protection from catabolism.

# Returns
Dict with:
- `recycling_fraction`: Fraction recycled
- `catabolism_fraction`: Fraction catabolized
- `half_life_effect`: Effect on half-life
"""
function calculate_fcrn_recycling(mab::mAbProperties,
                                   fcrn::FcRnParameters,
                                   mab_conc::Float64)
    # No FcRn binding for Fab fragments
    if mab.fcrn_affinity == 0.0
        return Dict(
            "recycling_fraction" => 0.0,
            "catabolism_fraction" => 1.0,
            "half_life_effect" => 0.05  # Very short half-life
        )
    end

    # FcRn binding at endosomal pH
    # Kd_apparent = Kd_pH6 / fcrn_affinity
    kd_app = fcrn.kd_ph6 / mab.fcrn_affinity

    # Fraction bound to FcRn in endosome
    # This is what gets recycled
    mab_conc_um = mab_conc / 1000.0  # Convert nM to μM
    fcrn_bound = mab_conc_um / (kd_app + mab_conc_um)

    # Saturation effect at high concentrations
    saturation_factor = 1.0
    if mab_conc > FCRN_SATURATION_CONC * 1000.0  # nM
        saturation_factor = FCRN_SATURATION_CONC * 1000.0 / mab_conc
    end

    # Recycling fraction
    recycling = fcrn.recycling_efficiency * fcrn_bound * saturation_factor * fcrn.expression_level
    recycling = min(recycling, 0.95)  # Cap at 95%

    # Catabolism (what's not recycled)
    catabolism = 1.0 - recycling

    # Half-life effect (relative to normal IgG)
    # Higher recycling = longer half-life
    hl_effect = recycling / fcrn.recycling_efficiency

    return Dict(
        "recycling_fraction" => recycling,
        "catabolism_fraction" => catabolism,
        "half_life_effect" => hl_effect,
        "fcrn_saturation" => 1.0 - saturation_factor
    )
end

# ============================================================================
# PK SIMULATION
# ============================================================================

"""
    simulate_mab_pk(mab::mAbProperties,
                     dose::Float64,
                     time_points::Vector{Float64};
                     target::Union{TargetProperties, Nothing} = nothing,
                     tmdd::Union{TMDDParameters, Nothing} = nothing,
                     fcrn::FcRnParameters = FcRnParameters(),
                     immunogenicity::ImmunogenicityState = ImmunogenicityState(),
                     body_weight::Float64 = 70.0)

Simulate mAb pharmacokinetics.

# Arguments
- `mab`: Antibody properties
- `dose`: Dose in mg
- `time_points`: Time points for output (days)
- `target`: Target properties (for TMDD)
- `tmdd`: TMDD parameters
- `fcrn`: FcRn parameters
- `immunogenicity`: ADA status
- `body_weight`: Patient weight (kg)

# Returns
Dict with concentration-time data
"""
function simulate_mab_pk(mab::mAbProperties,
                          dose::Float64,
                          time_points::Vector{Float64};
                          target::Union{TargetProperties, Nothing} = nothing,
                          tmdd::Union{TMDDParameters, Nothing} = nothing,
                          fcrn::FcRnParameters = FcRnParameters(),
                          immunogenicity::ImmunogenicityState = ImmunogenicityState(),
                          body_weight::Float64 = 70.0)

    # PK parameters
    vd_central = 0.05 * body_weight  # L (plasma volume)
    vd_peripheral = 0.1 * body_weight  # L

    # Base clearance from half-life
    ke = log(2) / (mab.half_life_days * 24.0)  # 1/h
    cl_base = ke * vd_central  # L/h

    # Apply ADA effect
    cl_base *= immunogenicity.clearance_multiplier

    # Initial concentration
    c0 = dose * 1000.0 / mab.molecular_weight / vd_central  # nM

    # Simulate
    concentrations = Float64[]
    target_occupancies = Float64[]

    dt = 0.01  # days
    t = 0.0
    c = c0

    for t_out in time_points
        while t < t_out
            # FcRn recycling effect
            fcrn_result = calculate_fcrn_recycling(mab, fcrn, c)
            cl_effective = cl_base / fcrn_result["half_life_effect"]

            # TMDD effect
            if !isnothing(target) && !isnothing(tmdd)
                tmdd_result = calculate_tmdd_clearance(mab, target, c, tmdd)
                cl_effective += tmdd_result["cl_tmdd"] * vd_central
            end

            # Update concentration
            dc = -cl_effective / vd_central * c * dt * 24.0
            c += dc
            c = max(c, 0.0)

            t += dt
        end

        push!(concentrations, c)

        # Target occupancy
        if !isnothing(target)
            occ = calculate_target_occupancy(mab, c, tmdd.target_baseline)
            push!(target_occupancies, occ)
        else
            push!(target_occupancies, NaN)
        end
    end

    return Dict(
        "time" => time_points,
        "concentration" => concentrations,
        "target_occupancy" => target_occupancies,
        "dose" => dose,
        "mab" => mab.name,
        "half_life_apparent" => mab.half_life_days * fcrn.recycling_efficiency / 0.7,
        "ada_status" => immunogenicity.ada_positive
    )
end

# ============================================================================
# CLINICAL UTILITIES
# ============================================================================

"""
    calculate_loading_dose(mab::mAbProperties,
                            target_conc::Float64,
                            body_weight::Float64)

Calculate loading dose to achieve target concentration.

# Arguments
- `mab`: Antibody properties
- `target_conc`: Target trough concentration (μg/mL)
- `body_weight`: Patient weight (kg)

# Returns
Recommended loading dose (mg)
"""
function calculate_loading_dose(mab::mAbProperties,
                                 target_conc::Float64,
                                 body_weight::Float64)
    vd = 0.05 * body_weight  # L

    # Account for distribution phase
    target_nM = target_conc * 1000.0 / mab.molecular_weight

    # Loading dose
    dose_mg = target_nM * mab.molecular_weight * vd / 1000.0

    # Round to practical dose
    dose_mg = round(dose_mg / 10.0) * 10.0

    return dose_mg
end

"""
    predict_immunogenicity_risk(mab::mAbProperties)

Estimate immunogenicity risk based on mAb properties.

# Returns
Dict with risk assessment
"""
function predict_immunogenicity_risk(mab::mAbProperties)
    # Base risk
    risk_score = 0.0
    factors = String[]

    # IgG1 generally less immunogenic than IgG2/IgG4
    if mab.igg_subclass == :igg1
        risk_score += 1.0
    elseif mab.igg_subclass == :igg4
        risk_score += 1.5
        push!(factors, "IgG4 subclass (higher ADA risk)")
    end

    # Humanization level (assumed from name patterns)
    if occursin("mab", lowercase(mab.name)) && !occursin("ximab", lowercase(mab.name))
        risk_score += 0.5
        push!(factors, "Fully human sequence")
    elseif occursin("zumab", lowercase(mab.name))
        risk_score += 1.0
        push!(factors, "Humanized")
    elseif occursin("ximab", lowercase(mab.name))
        risk_score += 2.0
        push!(factors, "Chimeric (higher risk)")
    end

    # ADCs have higher risk
    if mab.is_adc
        risk_score += 1.5
        push!(factors, "ADC (payload-related immunogenicity)")
    end

    # Target location
    if mab.target in ["CD20", "CD19", "CD22"]  # B cell targets
        risk_score -= 1.0
        push!(factors, "B cell depletion reduces ADA")
    end

    # Risk category
    if risk_score < 1.5
        category = :low
    elseif risk_score < 3.0
        category = :moderate
    else
        category = :high
    end

    return Dict(
        "risk_score" => risk_score,
        "risk_category" => category,
        "contributing_factors" => factors,
        "recommendation" => category == :high ? "Consider ADA monitoring" : "Standard monitoring"
    )
end

"""
    get_mab(name::String)

Get mAb properties from database.
"""
function get_mab(name::String)
    name_lower = lowercase(name)
    if haskey(MAB_DATABASE, name_lower)
        return MAB_DATABASE[name_lower]
    end
    return nothing
end

"""
    get_target(name::String)

Get target properties from database.
"""
function get_target(name::String)
    name_upper = uppercase(name)
    for (key, target) in TARGET_DATABASE
        if uppercase(key) == name_upper
            return target
        end
    end
    return nothing
end

end # module mAbPBPK
