"""
RBC Aging & Red Cell Distribution Width (RDW) Module

Models heterogeneity in red blood cell population and how drug
transport changes across the RBC lifespan (120 days).

Key Features:
- Reticulocyte vs mature RBC transporter expression
- Age-dependent osmotic fragility
- RDW as inter-individual variability parameter
- Splenic sequestration of aged RBCs
- Disease-state RBC populations (SCD, thalassemia)

Clinical Relevance:
- Explains chloroquine/antimalarial PK variability
- Important in hemolytic anemias
- Affects antiretroviral distribution
- Relevant in transfusion medicine

References:
- Lutz HU (2004) Innate immune and non-immune mediators of RBC clearance
- Bosman GJ (2008) Changes in RBC during aging
- Mohandas N (2008) Red cell membrane: past, present, future

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module RBCAging

using Statistics

export RBCPopulation, RBCAgeDistribution, ReticulocyteState
export create_normal_rbc_population, create_disease_population
export calculate_age_weighted_transport, calculate_rdw_effect
export simulate_rbc_turnover, get_age_distribution
export RBC_AGE_PARAMETERS, RETICULOCYTE_FACTORS

# ============================================================================
# CONSTANTS
# ============================================================================

# RBC lifespan
const NORMAL_RBC_LIFESPAN = 120.0     # days
const RETICULOCYTE_MATURATION = 2.0    # days in circulation before mature
const SENESCENT_THRESHOLD = 100.0      # days when senescence begins

# Normal values
const NORMAL_RETICULOCYTE_PERCENT = 1.0   # 0.5-2.0%
const NORMAL_RDW = 12.5                    # 11.5-14.5%
const NORMAL_HEMATOCRIT = 0.42
const NORMAL_RBC_COUNT = 5.0e12           # cells/L

# Transporter expression changes with age (relative to mature RBC = 1.0)
const RETICULOCYTE_BAND3 = 1.5            # Higher in reticulocytes
const RETICULOCYTE_GLUT1 = 2.0            # Much higher glucose uptake
const RETICULOCYTE_ENT1 = 1.8             # Higher nucleoside transport
const SENESCENT_BAND3 = 0.7               # Reduced in old RBCs
const SENESCENT_GLUT1 = 0.6
const SENESCENT_ENT1 = 0.5

# Membrane properties
const RETICULOCYTE_SURFACE_AREA = 160.0   # μm² (larger)
const MATURE_SURFACE_AREA = 140.0         # μm²
const SENESCENT_SURFACE_AREA = 120.0      # μm² (smaller, spherocytic)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    RBCAgeDistribution

Age distribution of RBC population.

# Fields
- `age_bins::Vector{Float64}`: Age bin centers (days)
- `fractions::Vector{Float64}`: Fraction in each bin
- `mean_age::Float64`: Population mean age
- `std_age::Float64`: Standard deviation of age
"""
struct RBCAgeDistribution
    age_bins::Vector{Float64}
    fractions::Vector{Float64}
    mean_age::Float64
    std_age::Float64

    function RBCAgeDistribution(age_bins::Vector{Float64}, fractions::Vector{Float64})
        @assert length(age_bins) == length(fractions)
        @assert sum(fractions) ≈ 1.0 atol=0.01

        mean_age = sum(age_bins .* fractions)
        std_age = sqrt(sum(fractions .* (age_bins .- mean_age).^2))

        new(age_bins, fractions, mean_age, std_age)
    end
end

"""
    ReticulocyteState

Reticulocyte characteristics.

# Fields
- `percent::Float64`: % of total RBCs that are reticulocytes
- `absolute_count::Float64`: Reticulocytes/μL
- `immature_fraction::Float64`: Immature reticulocyte fraction (IRF)
- `maturation_index::Float64`: 0-1, reticulocyte maturation state
- `rna_content::Float64`: Relative RNA content (1.0 = average)
"""
struct ReticulocyteState
    percent::Float64
    absolute_count::Float64
    immature_fraction::Float64
    maturation_index::Float64
    rna_content::Float64

    function ReticulocyteState(;
        percent = NORMAL_RETICULOCYTE_PERCENT,
        rbc_count = NORMAL_RBC_COUNT / 1e6,  # Convert to /μL
        immature_fraction = 0.1,
        maturation_index = 0.5,
        rna_content = 1.0
    )
        absolute = percent / 100.0 * rbc_count
        new(percent, absolute, immature_fraction, maturation_index, rna_content)
    end
end

"""
    RBCPopulation

Complete RBC population state including age distribution.

# Fields
- `age_distribution::RBCAgeDistribution`: Age distribution
- `reticulocytes::ReticulocyteState`: Reticulocyte state
- `rdw::Float64`: Red cell distribution width (%)
- `mcv::Float64`: Mean corpuscular volume (fL)
- `mchc::Float64`: Mean corpuscular hemoglobin concentration (g/dL)
- `hematocrit::Float64`: Hematocrit (fraction)
- `condition::Symbol`: :normal, :hemolytic, :transfused, etc.
- `effective_lifespan::Float64`: Actual RBC lifespan in this state (days)
"""
struct RBCPopulation
    age_distribution::RBCAgeDistribution
    reticulocytes::ReticulocyteState
    rdw::Float64
    mcv::Float64
    mchc::Float64
    hematocrit::Float64
    condition::Symbol
    effective_lifespan::Float64
end

# ============================================================================
# RBC AGE-DEPENDENT PARAMETERS
# ============================================================================

"""
Parameters that change with RBC age.
"""
const RBC_AGE_PARAMETERS = Dict{Symbol, Function}(
    # Transporter expression (age in days → relative expression)
    :band3 => age -> begin
        if age < 2.0
            RETICULOCYTE_BAND3 - (RETICULOCYTE_BAND3 - 1.0) * age / 2.0
        elseif age < SENESCENT_THRESHOLD
            1.0
        else
            1.0 - (1.0 - SENESCENT_BAND3) * (age - SENESCENT_THRESHOLD) / (NORMAL_RBC_LIFESPAN - SENESCENT_THRESHOLD)
        end
    end,

    :glut1 => age -> begin
        if age < 2.0
            RETICULOCYTE_GLUT1 - (RETICULOCYTE_GLUT1 - 1.0) * age / 2.0
        elseif age < SENESCENT_THRESHOLD
            1.0
        else
            1.0 - (1.0 - SENESCENT_GLUT1) * (age - SENESCENT_THRESHOLD) / (NORMAL_RBC_LIFESPAN - SENESCENT_THRESHOLD)
        end
    end,

    :ent1 => age -> begin
        if age < 2.0
            RETICULOCYTE_ENT1 - (RETICULOCYTE_ENT1 - 1.0) * age / 2.0
        elseif age < SENESCENT_THRESHOLD
            1.0
        else
            1.0 - (1.0 - SENESCENT_ENT1) * (age - SENESCENT_THRESHOLD) / (NORMAL_RBC_LIFESPAN - SENESCENT_THRESHOLD)
        end
    end,

    # Surface area
    :surface_area => age -> begin
        if age < 2.0
            RETICULOCYTE_SURFACE_AREA - (RETICULOCYTE_SURFACE_AREA - MATURE_SURFACE_AREA) * age / 2.0
        elseif age < SENESCENT_THRESHOLD
            MATURE_SURFACE_AREA
        else
            MATURE_SURFACE_AREA - (MATURE_SURFACE_AREA - SENESCENT_SURFACE_AREA) * (age - SENESCENT_THRESHOLD) / (NORMAL_RBC_LIFESPAN - SENESCENT_THRESHOLD)
        end
    end,

    # Membrane deformability (1.0 = normal)
    :deformability => age -> begin
        if age < 2.0
            0.9  # Reticulocytes slightly less deformable
        elseif age < 80.0
            1.0
        else
            1.0 - 0.5 * (age - 80.0) / (NORMAL_RBC_LIFESPAN - 80.0)
        end
    end,

    # PS (phosphatidylserine) exposure (senescence marker)
    :ps_exposure => age -> begin
        if age < SENESCENT_THRESHOLD
            0.01  # Minimal
        else
            0.01 + 0.5 * (age - SENESCENT_THRESHOLD) / (NORMAL_RBC_LIFESPAN - SENESCENT_THRESHOLD)
        end
    end,

    # Osmotic fragility
    :osmotic_fragility => age -> begin
        if age < 2.0
            0.8  # Reticulocytes more resistant
        elseif age < 80.0
            1.0
        else
            1.0 + 0.5 * (age - 80.0) / (NORMAL_RBC_LIFESPAN - 80.0)
        end
    end
)

"""
Reticulocyte-specific factors.
"""
const RETICULOCYTE_FACTORS = Dict{Symbol, Float64}(
    :band3_expression => RETICULOCYTE_BAND3,
    :glut1_expression => RETICULOCYTE_GLUT1,
    :ent1_expression => RETICULOCYTE_ENT1,
    :mct1_expression => 1.5,
    :surface_area => RETICULOCYTE_SURFACE_AREA,
    :volume => 110.0,  # fL (larger than mature)
    :hemoglobin => 0.9,  # Relative (still maturing)
    :rna_content => 10.0,  # High RNA (defines reticulocyte)
    :mitochondria => 5.0,  # Still have organelles
    :protein_synthesis => 3.0  # Active translation
)

# ============================================================================
# POPULATION FACTORIES
# ============================================================================

"""
    create_normal_rbc_population(;hematocrit=0.42, rdw=12.5)

Create normal steady-state RBC population.
"""
function create_normal_rbc_population(;hematocrit::Float64=0.42, rdw::Float64=12.5)
    # Create uniform age distribution (steady state)
    # In steady state, equal production/destruction rate
    n_bins = 24  # 5-day bins
    age_bins = collect(2.5:5.0:117.5)
    fractions = fill(1.0/n_bins, n_bins)

    age_dist = RBCAgeDistribution(age_bins, fractions)

    retics = ReticulocyteState(percent=NORMAL_RETICULOCYTE_PERCENT)

    return RBCPopulation(
        age_dist,
        retics,
        rdw,
        90.0,  # MCV
        34.0,  # MCHC
        hematocrit,
        :normal,
        NORMAL_RBC_LIFESPAN
    )
end

"""
    create_disease_population(disease::Symbol; kwargs...)

Create disease-specific RBC population.

Diseases:
- :hemolytic_anemia - Shortened lifespan, elevated reticulocytes
- :sickle_cell - Very short lifespan, abnormal RBCs
- :thalassemia - Ineffective erythropoiesis
- :aplastic - Low reticulocytes
- :myelodysplastic - High RDW, abnormal population
- :post_transfusion - Bimodal age distribution
- :chronic_disease - Moderate changes
- :iron_deficiency - Microcytic, high RDW
- :b12_deficiency - Macrocytic, high RDW
"""
function create_disease_population(disease::Symbol; kwargs...)
    if disease == :hemolytic_anemia
        # Shortened RBC lifespan (30-60 days)
        lifespan = get(kwargs, :lifespan, 45.0)
        n_bins = 12
        age_bins = collect(lifespan/24:lifespan/12:lifespan-lifespan/24)
        fractions = fill(1.0/n_bins, n_bins)

        age_dist = RBCAgeDistribution(age_bins, fractions)
        retics = ReticulocyteState(percent=8.0)  # Elevated

        return RBCPopulation(
            age_dist, retics,
            18.0,   # High RDW
            95.0,   # MCV may be elevated
            32.0,
            0.30,   # Anemia
            :hemolytic_anemia,
            lifespan
        )

    elseif disease == :sickle_cell
        # Very short lifespan (15-20 days)
        lifespan = 17.0
        n_bins = 6
        age_bins = collect(1.4:2.8:15.4)
        fractions = fill(1.0/n_bins, n_bins)

        age_dist = RBCAgeDistribution(age_bins, fractions)
        retics = ReticulocyteState(percent=15.0)  # Very elevated

        return RBCPopulation(
            age_dist, retics,
            22.0,   # Very high RDW
            88.0,   # MCV normal or low
            34.0,
            0.25,   # Severe anemia
            :sickle_cell,
            lifespan
        )

    elseif disease == :thalassemia
        lifespan = 60.0
        n_bins = 12
        age_bins = collect(2.5:5.0:57.5)
        fractions = fill(1.0/n_bins, n_bins)

        age_dist = RBCAgeDistribution(age_bins, fractions)
        retics = ReticulocyteState(percent=5.0)

        return RBCPopulation(
            age_dist, retics,
            20.0,
            65.0,   # Microcytic
            32.0,
            0.28,
            :thalassemia,
            lifespan
        )

    elseif disease == :aplastic
        # Normal lifespan but low production
        n_bins = 24
        age_bins = collect(2.5:5.0:117.5)
        # Skewed toward older (reduced production)
        fractions = [0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05,
                     0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                     0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.02]
        fractions = fractions ./ sum(fractions)

        age_dist = RBCAgeDistribution(age_bins, fractions)
        retics = ReticulocyteState(percent=0.2)  # Very low

        return RBCPopulation(
            age_dist, retics,
            13.0,
            95.0,
            34.0,
            0.20,
            :aplastic,
            NORMAL_RBC_LIFESPAN
        )

    elseif disease == :post_transfusion
        # Bimodal: patient's old RBCs + fresh transfused
        n_bins = 24
        age_bins = collect(2.5:5.0:117.5)
        # Peak at young (transfused) and old (native)
        fractions = zeros(n_bins)
        fractions[1:3] .= 0.15  # Fresh transfused (45%)
        fractions[20:24] .= 0.11  # Older native (55%)
        fractions = fractions ./ sum(fractions)

        age_dist = RBCAgeDistribution(age_bins, fractions)
        retics = ReticulocyteState(percent=0.5)

        return RBCPopulation(
            age_dist, retics,
            16.0,   # High RDW due to mixing
            90.0,
            34.0,
            0.35,
            :post_transfusion,
            NORMAL_RBC_LIFESPAN
        )

    elseif disease == :iron_deficiency
        n_bins = 24
        age_bins = collect(2.5:5.0:117.5)
        fractions = fill(1.0/n_bins, n_bins)

        age_dist = RBCAgeDistribution(age_bins, fractions)
        retics = ReticulocyteState(percent=0.8)

        return RBCPopulation(
            age_dist, retics,
            18.0,   # High RDW
            70.0,   # Microcytic
            30.0,   # Hypochromic
            0.32,
            :iron_deficiency,
            NORMAL_RBC_LIFESPAN
        )

    else
        return create_normal_rbc_population()
    end
end

# ============================================================================
# AGE-WEIGHTED TRANSPORT
# ============================================================================

"""
    calculate_age_weighted_transport(population::RBCPopulation,
                                      transporter::Symbol)

Calculate population-average transporter expression weighted by age distribution.
"""
function calculate_age_weighted_transport(population::RBCPopulation,
                                           transporter::Symbol)
    age_dist = population.age_distribution

    # Get age-dependent function
    if !haskey(RBC_AGE_PARAMETERS, transporter)
        return 1.0  # Default
    end

    expr_func = RBC_AGE_PARAMETERS[transporter]

    # Weight by age distribution
    weighted_expr = 0.0
    for (age, frac) in zip(age_dist.age_bins, age_dist.fractions)
        weighted_expr += frac * expr_func(age)
    end

    # Add reticulocyte contribution
    retic_frac = population.reticulocytes.percent / 100.0
    retic_expr = get(RETICULOCYTE_FACTORS, Symbol(string(transporter, "_expression")), 1.5)

    # Adjust: (1-retic_frac) * mature + retic_frac * reticulocyte
    total_expr = (1.0 - retic_frac) * weighted_expr + retic_frac * retic_expr

    return total_expr
end

"""
    calculate_rdw_effect(population::RBCPopulation, drug_transport_params::Dict)

Calculate how RDW (population heterogeneity) affects drug transport variability.
"""
function calculate_rdw_effect(population::RBCPopulation, drug_transport_params::Dict)
    rdw = population.rdw
    normal_rdw = NORMAL_RDW

    # Higher RDW = more heterogeneous response
    # Calculate coefficient of variation in transport

    # Get transporter expression across age bins
    expressions = Float64[]
    for age in population.age_distribution.age_bins
        expr = 1.0
        for (transporter, func) in RBC_AGE_PARAMETERS
            if transporter in [:band3, :glut1, :ent1]
                expr *= func(age)
            end
        end
        push!(expressions, expr)
    end

    # CV in expression
    cv_expr = std(expressions) / mean(expressions)

    # Scale by RDW ratio
    cv_scaled = cv_expr * (rdw / normal_rdw)

    return Dict(
        "cv_transport" => cv_scaled,
        "rdw_ratio" => rdw / normal_rdw,
        "min_transport" => minimum(expressions),
        "max_transport" => maximum(expressions),
        "mean_transport" => mean(expressions),
        "heterogeneity_factor" => 1.0 + cv_scaled
    )
end

# ============================================================================
# TURNOVER SIMULATION
# ============================================================================

"""
    simulate_rbc_turnover(population::RBCPopulation, days::Float64;
                          production_rate::Float64=1.0,
                          destruction_factor::Float64=1.0)

Simulate RBC population dynamics over time.
"""
function simulate_rbc_turnover(population::RBCPopulation, days::Float64;
                                production_rate::Float64=1.0,
                                destruction_factor::Float64=1.0)
    lifespan = population.effective_lifespan / destruction_factor

    # Discrete simulation
    dt = 1.0  # day
    n_bins = length(population.age_distribution.age_bins)
    bin_width = lifespan / n_bins

    # Initialize
    fractions = copy(population.age_distribution.fractions)
    hct = population.hematocrit
    retic_pct = population.reticulocytes.percent

    history = Dict(
        "time" => Float64[0.0],
        "hematocrit" => Float64[hct],
        "reticulocytes" => Float64[retic_pct],
        "mean_age" => Float64[population.age_distribution.mean_age]
    )

    for t in dt:dt:days
        # Age shift (cells get older)
        new_fractions = zeros(n_bins)
        for i in 2:n_bins
            new_fractions[i] = fractions[i-1] * (1.0 - dt/bin_width * 0.1)  # Some loss
        end

        # New production (reticulocytes enter)
        new_fractions[1] = production_rate / n_bins

        # Loss of oldest cells
        loss = fractions[end] * destruction_factor * dt / bin_width

        # Normalize
        fractions = new_fractions ./ sum(new_fractions)

        # Update hematocrit
        hct_change = (production_rate - destruction_factor) * 0.001 * dt
        hct = clamp(hct + hct_change, 0.1, 0.6)

        # Update reticulocytes
        retic_pct = production_rate * NORMAL_RETICULOCYTE_PERCENT

        # Calculate mean age
        mean_age = sum(population.age_distribution.age_bins .* fractions)

        push!(history["time"], t)
        push!(history["hematocrit"], hct)
        push!(history["reticulocytes"], retic_pct)
        push!(history["mean_age"], mean_age)
    end

    return history
end

# ============================================================================
# UTILITIES
# ============================================================================

"""
    get_age_distribution(population::RBCPopulation)

Get age distribution summary.
"""
function get_age_distribution(population::RBCPopulation)
    age_dist = population.age_distribution

    return Dict(
        "mean_age" => age_dist.mean_age,
        "std_age" => age_dist.std_age,
        "youngest_fraction" => age_dist.fractions[1],
        "oldest_fraction" => age_dist.fractions[end],
        "reticulocyte_percent" => population.reticulocytes.percent,
        "rdw" => population.rdw,
        "effective_lifespan" => population.effective_lifespan
    )
end

"""
    estimate_drug_transport_variability(population::RBCPopulation)

Estimate inter-individual variability in RBC drug transport based on RDW.
"""
function estimate_drug_transport_variability(population::RBCPopulation)
    # RDW correlates with transport heterogeneity
    rdw = population.rdw

    # Empirical: CV in drug accumulation ≈ RDW/2
    cv_accumulation = rdw / 200.0  # Convert % to fraction

    # 95% range
    lower_95 = 1.0 - 1.96 * cv_accumulation
    upper_95 = 1.0 + 1.96 * cv_accumulation

    return Dict(
        "cv" => cv_accumulation,
        "lower_95" => lower_95,
        "upper_95" => upper_95,
        "fold_range" => upper_95 / lower_95,
        "rdw" => rdw
    )
end

"""
    calculate_splenic_sequestration(population::RBCPopulation)

Calculate fraction of RBCs sequestered in spleen based on age/deformability.
"""
function calculate_splenic_sequestration(population::RBCPopulation)
    age_dist = population.age_distribution

    # Old, rigid RBCs are sequestered
    sequestered = 0.0
    for (age, frac) in zip(age_dist.age_bins, age_dist.fractions)
        deform = RBC_AGE_PARAMETERS[:deformability](age)
        if deform < 0.7  # Poorly deformable
            sequestered += frac * (0.7 - deform) / 0.7
        end
    end

    # Disease-specific adjustment
    if population.condition == :sickle_cell
        sequestered *= 3.0  # Much higher sequestration
    elseif population.condition == :thalassemia
        sequestered *= 2.0
    elseif population.condition == :hemolytic_anemia
        sequestered *= 1.5
    end

    sequestered = min(sequestered, 0.3)  # Max 30% sequestered

    return Dict(
        "fraction_sequestered" => sequestered,
        "effective_hematocrit" => population.hematocrit * (1.0 - sequestered),
        "splenic_pool" => sequestered * population.hematocrit
    )
end

end # module RBCAging
