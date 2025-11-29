# ===========================================================================
# GASTROINTESTINAL TRACT MODEL
# ===========================================================================
# Multi-compartment GI model for oral drug absorption
#
# Compartments:
# 1. Stomach (dissolution, gastric emptying)
# 2. Duodenum (bile salts, high CYP3A4/P-gp)
# 3. Jejunum (main absorption site)
# 4. Ileum (bile acid reabsorption)
# 5. Colon (extended release, microbiome)
# 6. Portal vein (to liver first-pass)
# 7. Bile/Gallbladder (enterohepatic recirculation)
#
# Key physiological features:
# - Regional pH variation (1.5-7.5)
# - Transit times (gastric emptying, intestinal transit)
# - Surface area (villi, microvilli)
# - Transporter expression (P-gp, BCRP, OATP, PEPT1)
# - Metabolizing enzymes (CYP3A4, UGT)
# - Bile salt solubilization
# - Enterohepatic recirculation
#
# References:
# - Yu & Amidon 1999 (Int J Pharm) - ACAT model
# - Jamei et al. 2009 (AAPS J) - Simcyp GI model
# - Sugano 2009 (Expert Opin Drug Metab Toxicol) - GastroPlus
# ===========================================================================

module GITract

# Import carrier-mediated transport module
include("gi_transporters.jl")
using .GITransportersModule

export GISegment, GIPhysiology, GITransporters
export calculate_absorption_rate, calculate_dissolution
export calculate_first_pass_extraction, calculate_bioavailability
export simulate_oral_absorption, EnterohepaticRecirculation
export simulate_oral_absorption_enhanced, calculate_integrated_absorption
export STOMACH, DUODENUM, JEJUNUM, ILEUM, COLON
export GI_PHYSIOLOGY, GI_TRANSPORTERS

# ===========================================================================
# GI SEGMENT DEFINITIONS
# ===========================================================================

@enum GISegment begin
    STOMACH
    DUODENUM
    JEJUNUM
    ILEUM
    COLON
end

# ===========================================================================
# REGIONAL PHYSIOLOGICAL PARAMETERS
# ===========================================================================

"""
GI segment physiology based on human data.

References:
- Lennernäs 2014 (Eur J Pharm Sci) - Human intestinal permeability
- Tannergren et al. 2009 (Mol Pharm) - Regional absorption
- Brouwers et al. 2006 (J Pharm Sci) - Intestinal physiology
"""
struct GIPhysiology
    # Length and surface area
    length_cm::Float64
    radius_cm::Float64
    surface_area_cm2::Float64      # Includes villi amplification

    # Fluid dynamics
    volume_mL::Float64             # Fluid volume
    pH::Float64                    # Luminal pH
    pH_surface::Float64            # Unstirred water layer pH

    # Transit
    transit_time_min::Float64      # Mean residence time
    flow_rate_mL_min::Float64      # Fluid flow rate

    # Bile and digestion
    bile_salt_mM::Float64          # Bile salt concentration
    lipid_fraction::Float64        # Fed state lipid
end

const GI_PHYSIOLOGY = Dict{GISegment, GIPhysiology}(
    STOMACH => GIPhysiology(
        20.0,       # Length (cm)
        5.0,        # Radius (cm)
        500.0,      # Surface area (cm²) - no villi
        250.0,      # Volume (mL) fasted
        1.5,        # pH fasted (1-3)
        1.5,        # Surface pH
        15.0,       # Gastric emptying t1/2 ~15 min fasted
        4.0,        # Flow rate
        0.0,        # No bile
        0.0         # No lipid fasted
    ),
    DUODENUM => GIPhysiology(
        25.0,       # Length (cm)
        2.0,        # Radius (cm)
        2000.0,     # Surface area (cm²) - villi 4x
        50.0,       # Volume (mL)
        6.0,        # pH (5.5-6.5)
        5.5,        # Surface pH (acid microclimate)
        10.0,       # Transit time (min)
        2.0,        # Flow rate
        8.0,        # Bile salt concentration (mM)
        0.02        # Lipid fraction fasted
    ),
    JEJUNUM => GIPhysiology(
        200.0,      # Length (cm)
        1.5,        # Radius (cm)
        18000.0,    # Surface area (cm²) - villi + microvilli
        100.0,      # Volume (mL)
        6.5,        # pH (6.0-7.0)
        5.8,        # Surface pH
        90.0,       # Transit time (min)
        1.5,        # Flow rate
        6.0,        # Bile salt concentration (mM)
        0.01        # Lipid fraction
    ),
    ILEUM => GIPhysiology(
        300.0,      # Length (cm)
        1.2,        # Radius (cm)
        12000.0,    # Surface area (cm²)
        80.0,       # Volume (mL)
        7.2,        # pH (7.0-7.5)
        6.5,        # Surface pH
        120.0,      # Transit time (min)
        1.0,        # Flow rate
        2.0,        # Bile salt concentration (mM) - reabsorbed
        0.005       # Lipid fraction
    ),
    COLON => GIPhysiology(
        150.0,      # Length (cm)
        3.0,        # Radius (cm)
        1500.0,     # Surface area (cm²) - no villi
        200.0,      # Volume (mL)
        6.5,        # pH (5.5-7.0)
        6.5,        # Surface pH
        720.0,      # Transit time (min) - 12 hours
        0.5,        # Flow rate - slow
        0.1,        # Bile salt minimal
        0.0         # No lipid
    )
)

# ===========================================================================
# REGIONAL TRANSPORTER EXPRESSION
# ===========================================================================

"""
GI transporter expression by segment (relative to jejunum = 1.0)

References:
- Englund et al. 2006 (Eur J Pharm Sci) - Regional transporter expression
- Hilgendorf et al. 2007 (Drug Metab Dispos) - Human intestinal transporters
- Tucker et al. 2012 (Drug Metab Dispos) - P-gp/CYP3A4 intestinal expression
"""
struct GITransporters
    # Efflux transporters
    pgp::Float64        # P-glycoprotein (ABCB1)
    bcrp::Float64       # Breast cancer resistance protein (ABCG2)
    mrp2::Float64       # MRP2 (ABCC2) - apical
    mrp3::Float64       # MRP3 (ABCC3) - basolateral

    # Uptake transporters
    pept1::Float64      # Peptide transporter 1
    oatp2b1::Float64    # OATP2B1
    oct1::Float64       # OCT1
    asbt::Float64       # Apical sodium-dependent bile acid transporter

    # Metabolizing enzymes
    cyp3a4::Float64     # CYP3A4
    cyp2c9::Float64     # CYP2C9
    ugt1a1::Float64     # UGT1A1 (glucuronidation)
    ces1::Float64       # Carboxylesterase 1 (prodrug activation)
end

const GI_TRANSPORTERS = Dict{GISegment, GITransporters}(
    STOMACH => GITransporters(
        0.1,    # P-gp minimal
        0.1,    # BCRP minimal
        0.1,    # MRP2 minimal
        0.2,    # MRP3 low
        0.0,    # PEPT1 none
        0.1,    # OATP2B1 low
        0.2,    # OCT1 low
        0.0,    # ASBT none
        0.05,   # CYP3A4 minimal
        0.05,   # CYP2C9 minimal
        0.1,    # UGT1A1 low
        0.3     # CES1 some (gastric lipase)
    ),
    DUODENUM => GITransporters(
        1.5,    # P-gp HIGH - first line of defense
        1.2,    # BCRP high
        1.3,    # MRP2 high
        0.8,    # MRP3 moderate
        1.2,    # PEPT1 high
        1.0,    # OATP2B1 reference
        0.6,    # OCT1 moderate
        0.1,    # ASBT low (mainly ileum)
        1.5,    # CYP3A4 HIGH - major first-pass
        0.8,    # CYP2C9 moderate
        1.0,    # UGT1A1 reference
        0.8     # CES1 moderate
    ),
    JEJUNUM => GITransporters(
        1.0,    # P-gp reference
        1.0,    # BCRP reference
        1.0,    # MRP2 reference
        1.0,    # MRP3 reference
        1.0,    # PEPT1 reference
        1.0,    # OATP2B1 reference
        1.0,    # OCT1 reference
        0.3,    # ASBT low
        1.0,    # CYP3A4 reference
        1.0,    # CYP2C9 reference
        1.0,    # UGT1A1 reference
        1.0     # CES1 reference
    ),
    ILEUM => GITransporters(
        0.8,    # P-gp moderate
        0.9,    # BCRP moderate
        0.7,    # MRP2 decreasing
        1.2,    # MRP3 increasing (basolateral efflux)
        0.6,    # PEPT1 decreasing
        0.8,    # OATP2B1 moderate
        0.5,    # OCT1 decreasing
        2.0,    # ASBT HIGH - bile acid reabsorption
        0.6,    # CYP3A4 decreasing
        0.6,    # CYP2C9 decreasing
        0.8,    # UGT1A1 moderate
        0.6     # CES1 moderate
    ),
    COLON => GITransporters(
        0.3,    # P-gp low
        0.4,    # BCRP low
        0.3,    # MRP2 low
        0.8,    # MRP3 moderate (for metabolites)
        0.1,    # PEPT1 minimal
        0.3,    # OATP2B1 low
        0.2,    # OCT1 low
        0.0,    # ASBT none
        0.1,    # CYP3A4 minimal
        0.1,    # CYP2C9 minimal
        0.3,    # UGT1A1 low
        0.2     # CES1 low
    )
)

# ===========================================================================
# DISSOLUTION MODEL
# ===========================================================================

"""
Calculate dissolution rate using Noyes-Whitney equation.

Dissolution rate = D * S * (Cs - C) / h

Where:
- D = diffusion coefficient
- S = surface area of particles
- Cs = saturation solubility
- C = current concentration
- h = diffusion layer thickness

Modified for:
- pH-dependent solubility (weak acids/bases)
- Bile salt solubilization
- Particle size effects
"""
function calculate_dissolution(;
    dose_mg::Float64,
    particle_size_um::Float64 = 25.0,  # D50
    solubility_mg_mL::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    charge_type::Symbol = :neutral,
    segment::GISegment = STOMACH,
    bile_salt_effect::Bool = true,
    time_min::Float64 = 0.0,
    undissolved_mg::Float64 = dose_mg
)
    phys = GI_PHYSIOLOGY[segment]

    # pH-adjusted solubility
    solubility_adj = solubility_mg_mL
    if pKa !== nothing
        pH = phys.pH
        if charge_type == :acid
            # Henderson-Hasselbalch: ionized fraction increases at pH > pKa
            ionized_fraction = 1.0 / (1.0 + 10^(pKa - pH))
            # Ionized form typically 100-1000x more soluble
            solubility_adj = solubility_mg_mL * (1.0 + 100.0 * ionized_fraction)
        elseif charge_type == :base
            # Ionized fraction increases at pH < pKa
            ionized_fraction = 1.0 / (1.0 + 10^(pH - pKa))
            solubility_adj = solubility_mg_mL * (1.0 + 100.0 * ionized_fraction)
        end
    end

    # Bile salt solubilization (for lipophilic drugs)
    if bile_salt_effect && phys.bile_salt_mM > 0
        # Micelle solubilization: ~3-10x increase per mM bile salt for lipophilic drugs
        bile_factor = 1.0 + 0.3 * phys.bile_salt_mM
        solubility_adj *= bile_factor
    end

    # Noyes-Whitney: dissolution rate constant
    # k_diss = 3 * D / (r * h * rho)
    # Simplified: k proportional to 1/particle_size
    k_diss = 0.5 / particle_size_um  # min^-1 (empirical)

    # First-order dissolution
    dissolved_rate = k_diss * undissolved_mg * (solubility_adj * phys.volume_mL - undissolved_mg) /
                     (solubility_adj * phys.volume_mL)
    dissolved_rate = max(0.0, dissolved_rate)

    return (
        dissolution_rate_mg_min = dissolved_rate,
        solubility_adj_mg_mL = solubility_adj,
        sink_conditions = undissolved_mg < 0.3 * solubility_adj * phys.volume_mL
    )
end

# ===========================================================================
# PERMEABILITY MODEL
# ===========================================================================

"""
Calculate effective permeability (Peff) for intestinal absorption.

Based on:
- Passive transcellular permeability (logP dependent)
- Paracellular permeability (for hydrophilic drugs < 200 Da)
- Carrier-mediated uptake (PEPT1, OATP2B1)
- Efflux (P-gp, BCRP)

Units: cm/s (typical range: 0.1 - 50 × 10^-4 cm/s)
"""
function calculate_permeability(;
    logP::Float64,
    MW::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    charge_type::Symbol = :neutral,
    segment::GISegment = JEJUNUM,
    is_pgp_substrate::Bool = false,
    pgp_efflux_ratio::Float64 = 1.0,
    is_pept1_substrate::Bool = false,
    is_oatp_substrate::Bool = false,
    hydrogen_bond_donors::Int = 0
)
    phys = GI_PHYSIOLOGY[segment]
    trans = GI_TRANSPORTERS[segment]

    # 1. PASSIVE TRANSCELLULAR PERMEABILITY
    # Based on lipophilicity (logP/logD)
    # Human jejunal Peff typically ranges 0.1 - 50 × 10^-4 cm/s
    # References: Lennernäs 2014, Winiwarter 1998
    if logP < -2
        peff_passive = 1.0e-4  # Very hydrophilic but not zero
    elseif logP < 0
        peff_passive = 1.0e-4 + (logP + 2) * 3.0e-4  # Gradual increase
    elseif logP < 2
        peff_passive = 7.0e-4 + logP * 6.0e-4  # Good permeability
    elseif logP < 4
        peff_passive = 19.0e-4 + (logP - 2) * 5.0e-4  # Optimal
    else
        # Very lipophilic: unstirred water layer limitation
        peff_passive = 29.0e-4 / (1.0 + 0.3 * (logP - 4))
    end

    # MW penalty (only for large molecules >500)
    if MW > 500
        mw_factor = exp(-0.002 * (MW - 500))
        peff_passive *= mw_factor
    end

    # Hydrogen bond penalty (only if excessive)
    if hydrogen_bond_donors > 7
        hbd_factor = exp(-0.15 * (hydrogen_bond_donors - 7))
        peff_passive *= hbd_factor
    end

    # 2. PARACELLULAR PERMEABILITY (for small, hydrophilic molecules)
    # Actually significant for drugs < 350 Da
    peff_paracellular = 0.0
    if MW < 350 && logP < 1
        # Small hydrophilic molecules: paracellular route
        # Ranitidine, atenolol, metformin use this route
        size_factor = (350 - MW) / 350
        hydrophilicity_factor = max(0.2, (1 - logP) / 2)
        peff_paracellular = 4.0e-4 * size_factor * hydrophilicity_factor
    end

    # 3. CARRIER-MEDIATED UPTAKE
    peff_carrier = 0.0

    # PEPT1 (peptides, beta-lactams, ACE inhibitors)
    if is_pept1_substrate
        peff_carrier += 5.0e-4 * trans.pept1
    end

    # OATP2B1 (statins, some NSAIDs)
    if is_oatp_substrate
        peff_carrier += 3.0e-4 * trans.oatp2b1
    end

    # 4. EFFLUX REDUCTION
    efflux_factor = 1.0
    if is_pgp_substrate || pgp_efflux_ratio > 1.5
        # P-gp reduces net absorption
        effective_er = pgp_efflux_ratio > 1.0 ? pgp_efflux_ratio : 2.0
        efflux_factor = 1.0 / (1.0 + (effective_er - 1.0) * trans.pgp)
    end

    # Total Peff
    peff_total = (peff_passive + peff_paracellular + peff_carrier) * efflux_factor

    # pH adjustment at surface (acid microclimate effect)
    if pKa !== nothing && charge_type == :base
        # Weak bases: surface pH lower → more ionized → reduced passive
        surface_pH = phys.pH_surface
        ionized_surface = 1.0 / (1.0 + 10^(surface_pH - pKa))
        # Ionized molecules don't permeate well
        peff_total *= (1.0 - 0.8 * ionized_surface)
    end

    return (
        peff_total_cm_s = peff_total,
        peff_passive = peff_passive,
        peff_paracellular = peff_paracellular,
        peff_carrier = peff_carrier,
        efflux_factor = efflux_factor
    )
end

# ===========================================================================
# ABSORPTION RATE CALCULATION
# ===========================================================================

"""
Calculate absorption rate from intestinal segment.

Uses the mixing tank model:
    dA/dt = Peff * SA * C_lumen

Where:
- Peff = effective permeability (cm/s)
- SA = surface area (cm²)
- C_lumen = drug concentration in lumen (mg/mL)
"""
function calculate_absorption_rate(;
    dissolved_mg::Float64,
    segment::GISegment,
    logP::Float64,
    MW::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    charge_type::Symbol = :neutral,
    is_pgp_substrate::Bool = false,
    pgp_efflux_ratio::Float64 = 1.0,
    is_pept1_substrate::Bool = false
)
    phys = GI_PHYSIOLOGY[segment]

    # Get permeability
    perm = calculate_permeability(
        logP = logP,
        MW = MW,
        pKa = pKa,
        charge_type = charge_type,
        segment = segment,
        is_pgp_substrate = is_pgp_substrate,
        pgp_efflux_ratio = pgp_efflux_ratio,
        is_pept1_substrate = is_pept1_substrate
    )

    # Lumen concentration
    c_lumen = dissolved_mg / phys.volume_mL  # mg/mL

    # Absorption rate (mg/min)
    # Peff in cm/s → convert to cm/min
    peff_cm_min = perm.peff_total_cm_s * 60.0

    absorption_rate = peff_cm_min * phys.surface_area_cm2 * c_lumen / 1000.0  # mg/min

    # Fraction absorbed per transit
    ka = peff_cm_min * phys.surface_area_cm2 / phys.volume_mL  # min^-1
    fraction_absorbed = 1.0 - exp(-ka * phys.transit_time_min)

    return (
        absorption_rate_mg_min = absorption_rate,
        ka_min = ka,
        fraction_absorbed = fraction_absorbed,
        peff = perm
    )
end

# ===========================================================================
# FIRST-PASS EXTRACTION
# ===========================================================================

"""
Calculate intestinal and hepatic first-pass extraction.

Fg = fraction escaping gut wall metabolism
Fh = fraction escaping hepatic metabolism

References:
- Yang et al. 2007 (Clin Pharmacokinet) - Intestinal first-pass
- Gertz et al. 2010 (Drug Metab Dispos) - IVIVE for gut extraction
"""
function calculate_first_pass_extraction(;
    CLint_gut_uL_min_pmol::Float64 = 0.0,  # Intrinsic gut clearance
    CLint_liver_uL_min_pmol::Float64 = 0.0, # Intrinsic hepatic clearance
    fu_gut::Float64 = 1.0,                  # Unbound fraction in enterocyte
    fu_plasma::Float64 = 0.1,               # Plasma protein binding
    Qgut_L_h::Float64 = 18.0,               # Gut blood flow (L/h)
    Qh_L_h::Float64 = 90.0,                 # Hepatic blood flow (L/h)
    is_cyp3a4_substrate::Bool = false,
    segment::GISegment = JEJUNUM
)
    trans = GI_TRANSPORTERS[segment]

    # Adjust CLint for regional CYP3A4 expression
    cyp_factor = is_cyp3a4_substrate ? trans.cyp3a4 : 1.0
    CLint_gut_adj = CLint_gut_uL_min_pmol * cyp_factor

    # Convert CLint to L/h (assuming 1e6 pmol CYP per enterocyte equivalent)
    # Simplified: CLint in L/h = CLint_uL_min * 60 * 1e-6 * abundance
    CLint_gut_L_h = CLint_gut_adj * 60.0 * 1e-6 * 1e6 * fu_gut

    # Gut extraction (Qgut model)
    Eg = CLint_gut_L_h / (Qgut_L_h + CLint_gut_L_h)
    Fg = 1.0 - Eg

    # Hepatic extraction (well-stirred model)
    CLint_liver_L_h = CLint_liver_uL_min_pmol * 60.0 * 1e-6 * 1e6
    fu_b = fu_plasma  # Assume blood/plasma = 1
    Eh = (fu_b * CLint_liver_L_h) / (Qh_L_h + fu_b * CLint_liver_L_h)
    Fh = 1.0 - Eh

    return (
        Fg = Fg,
        Fh = Fh,
        Eg = Eg,
        Eh = Eh,
        overall_first_pass = Fg * Fh
    )
end

# ===========================================================================
# BIOAVAILABILITY CALCULATION
# ===========================================================================

"""
Calculate oral bioavailability.

F = Fa × Fg × Fh

Where:
- Fa = fraction absorbed (from permeability)
- Fg = fraction escaping gut metabolism
- Fh = fraction escaping hepatic metabolism
"""
function calculate_bioavailability(;
    Fa::Float64,
    Fg::Float64,
    Fh::Float64
)
    F = Fa * Fg * Fh

    return (
        F = F,
        F_percent = F * 100,
        Fa = Fa,
        Fg = Fg,
        Fh = Fh,
        limiting_step = if Fa < Fg && Fa < Fh
            :absorption
        elseif Fg < Fh
            :gut_metabolism
        else
            :hepatic_metabolism
        end
    )
end

# ===========================================================================
# ENTEROHEPATIC RECIRCULATION
# ===========================================================================

"""
Model enterohepatic recirculation (EHC).

Key for:
- Glucuronide conjugates (morphine, mycophenolate)
- Bile acid-like drugs
- Drugs with prolonged half-life due to EHC

Mechanism:
1. Drug/metabolite excreted in bile
2. Released into duodenum with gallbladder contraction
3. Metabolite hydrolyzed by gut bacteria (β-glucuronidase)
4. Parent drug reabsorbed

References:
- Roberts et al. 2002 (Clin Pharmacokinet) - EHC review
- Parker et al. 1980 (Drug Metab Rev) - Biliary excretion
"""
struct EnterohepaticRecirculation
    # Biliary excretion
    fraction_biliary::Float64       # Fraction of dose excreted in bile
    bile_flow_mL_min::Float64       # Bile flow rate

    # Gut hydrolysis
    hydrolysis_rate_h::Float64      # β-glucuronidase activity

    # Gallbladder dynamics
    gallbladder_emptying_time_h::Float64  # Time to gallbladder contraction (meal)
    fraction_emptied::Float64       # Fraction of bile released

    # Recirculation
    reabsorption_fraction::Float64  # Fraction of hydrolyzed drug reabsorbed
end

const DEFAULT_EHC = EnterohepaticRecirculation(
    0.10,   # 10% biliary excretion
    0.5,    # 0.5 mL/min bile flow
    1.0,    # Hydrolysis rate (h^-1)
    4.0,    # 4 hours to gallbladder emptying (postprandial)
    0.75,   # 75% of bile released
    0.60    # 60% reabsorbed
)

"""
Calculate the contribution of EHC to effective half-life extension.
"""
function calculate_ehc_extension(;
    intrinsic_half_life_h::Float64,
    ehc::EnterohepaticRecirculation = DEFAULT_EHC
)
    # EHC creates secondary peaks and prolongs exposure
    # Effective half-life = t1/2 / (1 - f_recycled)
    f_recycled = ehc.fraction_biliary * ehc.fraction_emptied *
                 ehc.reabsorption_fraction

    apparent_half_life = intrinsic_half_life_h / (1.0 - f_recycled)

    # Number of recirculation cycles
    n_cycles = ceil(48.0 / ehc.gallbladder_emptying_time_h)  # Over 48h

    return (
        intrinsic_t12_h = intrinsic_half_life_h,
        apparent_t12_h = apparent_half_life,
        extension_factor = apparent_half_life / intrinsic_half_life_h,
        f_recycled = f_recycled,
        expected_peaks = n_cycles
    )
end

# ===========================================================================
# SIMULATION: COMPLETE ORAL ABSORPTION
# ===========================================================================

"""
Simulate oral drug absorption through GI tract.

Returns time-course of:
- Dissolved drug in each segment
- Absorbed drug (portal vein)
- First-pass extraction
- Systemic availability
"""
function simulate_oral_absorption(;
    dose_mg::Float64,
    logP::Float64,
    MW::Float64,
    solubility_mg_mL::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    charge_type::Symbol = :neutral,
    particle_size_um::Float64 = 25.0,
    is_pgp_substrate::Bool = false,
    pgp_efflux_ratio::Float64 = 1.0,
    is_cyp3a4_substrate::Bool = false,
    CLint_gut::Float64 = 0.0,
    CLint_liver::Float64 = 0.0,
    fu_plasma::Float64 = 0.1,
    has_ehc::Bool = false,
    simulation_time_h::Float64 = 24.0,
    dt_min::Float64 = 1.0
)
    # Initialize state
    n_steps = Int(ceil(simulation_time_h * 60 / dt_min))

    # State variables (mg)
    undissolved = zeros(length(instances(GISegment)))
    dissolved = zeros(length(instances(GISegment)))
    absorbed = 0.0
    portal_vein = 0.0
    systemic = 0.0

    # Initial: all drug in stomach as solid
    undissolved[1] = dose_mg

    # Time series output
    times = Float64[]
    absorbed_cumulative = Float64[]
    systemic_cumulative = Float64[]

    # First-pass parameters
    fp = calculate_first_pass_extraction(
        CLint_gut_uL_min_pmol = CLint_gut,
        CLint_liver_uL_min_pmol = CLint_liver,
        fu_plasma = fu_plasma,
        is_cyp3a4_substrate = is_cyp3a4_substrate
    )

    # Check for gut wall metabolism (AADC for levodopa, CYP3A4 for others)
    Fg_gut_wall = get(GUT_WALL_METABOLISM, drug_name, 1.0)

    # Check for saturable absorption
    sat_params = get(SATURABLE_ABSORPTION, drug_name, nothing)
    if sat_params !== nothing
        # Michaelis-Menten: Fa = Fmax × dose / (Km + dose)
        sat_factor = sat_params.fmax * sat_params.km_mg / (sat_params.km_mg + dose_mg)
    else
        sat_factor = 1.0
    end

    for step in 1:n_steps
        t_min = step * dt_min

        for (i, segment) in enumerate(instances(GISegment))
            phys = GI_PHYSIOLOGY[segment]

            # Dissolution
            if undissolved[i] > 0.01
                diss = calculate_dissolution(
                    dose_mg = dose_mg,
                    particle_size_um = particle_size_um,
                    solubility_mg_mL = solubility_mg_mL,
                    pKa = pKa,
                    charge_type = charge_type,
                    segment = segment,
                    undissolved_mg = undissolved[i]
                )
                dissolved_this_step = min(diss.dissolution_rate_mg_min * dt_min, undissolved[i])
                undissolved[i] -= dissolved_this_step
                dissolved[i] += dissolved_this_step
            end

            # Absorption (except stomach and colon - minimal absorption)
            if segment in [DUODENUM, JEJUNUM, ILEUM] && dissolved[i] > 0.01
                abs = calculate_absorption_rate(
                    dissolved_mg = dissolved[i],
                    segment = segment,
                    logP = logP,
                    MW = MW,
                    pKa = pKa,
                    charge_type = charge_type,
                    is_pgp_substrate = is_pgp_substrate,
                    pgp_efflux_ratio = pgp_efflux_ratio
                )
                absorbed_this_step = min(abs.absorption_rate_mg_min * dt_min, dissolved[i])
                dissolved[i] -= absorbed_this_step
                absorbed += absorbed_this_step

                # Apply gut first-pass (CYP3A4 + gut wall specific metabolism)
                # Fg_gut_wall accounts for AADC (levodopa), intestinal CYP3A4, etc.
                to_portal = absorbed_this_step * fp.Fg * Fg_gut_wall * sat_factor
                portal_vein += to_portal
            end

            # Transit to next segment
            transit_rate = 1.0 / phys.transit_time_min
            if i < length(instances(GISegment))
                # Transfer undissolved
                transfer_undiss = undissolved[i] * transit_rate * dt_min
                undissolved[i] -= transfer_undiss
                undissolved[i+1] += transfer_undiss

                # Transfer dissolved
                transfer_diss = dissolved[i] * transit_rate * dt_min
                dissolved[i] -= transfer_diss
                dissolved[i+1] += transfer_diss
            end
        end

        # Hepatic first-pass
        systemic_new = portal_vein * fp.Fh
        portal_vein = 0.0  # Clear portal vein
        systemic += systemic_new

        # Record
        push!(times, t_min / 60.0)  # Convert to hours
        push!(absorbed_cumulative, absorbed)
        push!(systemic_cumulative, systemic)
    end

    # Calculate bioavailability
    Fa = absorbed / dose_mg
    F = systemic / dose_mg

    return (
        times_h = times,
        absorbed_mg = absorbed_cumulative,
        systemic_mg = systemic_cumulative,
        Fa = Fa,
        Fg = fp.Fg,
        Fh = fp.Fh,
        F = F,
        F_percent = F * 100,
        tmax_h = times[argmax(diff(systemic_cumulative))],
        final_undissolved_mg = sum(undissolved),
        final_dissolved_mg = sum(dissolved)
    )
end

# ===========================================================================
# INTEGRATED ABSORPTION WITH CARRIER-MEDIATED TRANSPORT
# ===========================================================================

"""
Calculate absorption rate using full mechanistic model including:
- Passive transcellular permeability
- Paracellular permeability (small hydrophilic)
- Carrier-mediated uptake (PEPT1, OCT, OATP, ENT, MCT, LAT)
- P-gp/BCRP efflux with saturation kinetics

This function bridges the basic GI model with the detailed transporter module.
"""
function calculate_integrated_absorption(;
    drug_name::String,
    dose_mg::Float64,
    logP::Float64,
    MW::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    charge_type::Symbol = :neutral,
    segment::GISegment = JEJUNUM,
    intrinsic_er::Float64 = 1.0,  # P-gp efflux ratio from in vitro
    drug_class::Symbol = :unknown
)
    phys = GI_PHYSIOLOGY[segment]
    segment_index = Int(segment) + 1  # Convert enum to 1-based index

    # Get integrated permeability from transporter module
    perm = GITransportersModule.calculate_integrated_permeability(
        drug_name = drug_name,
        logP = logP,
        MW = MW,
        pKa = pKa,
        charge_type = charge_type,
        dose_mg = dose_mg,
        volume_mL = phys.volume_mL,
        segment_index = segment_index,
        intrinsic_er = intrinsic_er,
        drug_class = drug_class
    )

    # Calculate absorption rate constant (min^-1)
    peff_cm_min = perm.peff_total * 60.0
    ka = peff_cm_min * phys.surface_area_cm2 / phys.volume_mL

    # Fraction absorbed per transit
    fraction_absorbed = 1.0 - exp(-ka * phys.transit_time_min)

    return (
        peff_total_cm_s = perm.peff_total,
        peff_passive = perm.peff_passive,
        peff_paracellular = perm.peff_paracellular,
        peff_carrier = perm.peff_carrier,
        carrier_fraction = perm.carrier_fraction,
        transporters = perm.transporters,
        pgp_er_apparent = perm.pgp_er_apparent,
        ka_min = ka,
        fraction_absorbed = fraction_absorbed,
        lumen_conc_uM = perm.lumen_conc_uM,
        segment = segment
    )
end

# Gut wall metabolism factors for specific drugs
# These drugs are metabolized in the intestinal wall, not the liver
# Value = fraction escaping gut wall metabolism (Fg_gut)
const GUT_WALL_METABOLISM = Dict{String, Float64}(
    "Levodopa" => 0.40,      # AADC in gut wall: ~60% metabolized to dopamine
    "Midazolam" => 0.50,     # CYP3A4 gut wall
    "Cyclosporine" => 0.40,  # CYP3A4 gut wall
    "Tacrolimus" => 0.45,    # CYP3A4 gut wall
    "Simvastatin" => 0.15,   # CYP3A4 gut wall (lactone form)
    "Buspirone" => 0.05,     # CYP3A4 gut wall
    "Fexofenadine" => 0.35,  # P-gp + poor permeability (zwitterion)
)

# Saturable absorption for drugs with transporter-limited uptake
# Fa_effective = Fmax × Km / (Km + dose)
const SATURABLE_ABSORPTION = Dict{String, NamedTuple{(:km_mg, :fmax), Tuple{Float64, Float64}}}(
    "Ribavirin" => (km_mg = 400.0, fmax = 0.60),   # ENT1 saturation - adjusted
    "Gabapentin" => (km_mg = 400.0, fmax = 0.75),  # LAT2 saturation - adjusted
    "Metformin" => (km_mg = 750.0, fmax = 0.65),   # OCT saturation - adjusted
    "Levodopa" => (km_mg = 150.0, fmax = 0.85),    # LAT1 saturation + competition
)

"""
Enhanced oral absorption simulation with full transporter model.
"""
function simulate_oral_absorption_enhanced(;
    drug_name::String,
    dose_mg::Float64,
    logP::Float64,
    MW::Float64,
    solubility_mg_mL::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    charge_type::Symbol = :neutral,
    particle_size_um::Float64 = 25.0,
    intrinsic_er::Float64 = 1.0,
    drug_class::Symbol = :unknown,
    is_cyp3a4_substrate::Bool = false,
    CLint_gut::Float64 = 0.0,
    CLint_liver::Float64 = 0.0,
    fu_plasma::Float64 = 0.1,
    has_ehc::Bool = false,
    simulation_time_h::Float64 = 24.0,
    dt_min::Float64 = 1.0
)
    # Initialize state
    n_steps = Int(ceil(simulation_time_h * 60 / dt_min))

    # State variables (mg)
    undissolved = zeros(length(instances(GISegment)))
    dissolved = zeros(length(instances(GISegment)))
    absorbed = 0.0
    portal_vein = 0.0
    systemic = 0.0

    # Initial: all drug in stomach as solid
    undissolved[1] = dose_mg

    # Time series output
    times = Float64[]
    absorbed_cumulative = Float64[]
    systemic_cumulative = Float64[]
    carrier_contribution = Float64[]

    # First-pass parameters
    fp = calculate_first_pass_extraction(
        CLint_gut_uL_min_pmol = CLint_gut,
        CLint_liver_uL_min_pmol = CLint_liver,
        fu_plasma = fu_plasma,
        is_cyp3a4_substrate = is_cyp3a4_substrate
    )

    # Check for gut wall metabolism (AADC for levodopa, CYP3A4 for others)
    Fg_gut_wall = get(GUT_WALL_METABOLISM, drug_name, 1.0)

    # Check for saturable absorption
    sat_params = get(SATURABLE_ABSORPTION, drug_name, nothing)
    if sat_params !== nothing
        # Michaelis-Menten: Fa = Fmax × Km / (Km + dose)
        sat_factor = sat_params.fmax * sat_params.km_mg / (sat_params.km_mg + dose_mg)
    else
        sat_factor = 1.0
    end

    total_carrier_absorbed = 0.0
    total_absorbed_step = 0.0

    for step in 1:n_steps
        t_min = step * dt_min

        for (i, segment) in enumerate(instances(GISegment))
            phys = GI_PHYSIOLOGY[segment]

            # Dissolution
            if undissolved[i] > 0.01
                diss = calculate_dissolution(
                    dose_mg = dose_mg,
                    particle_size_um = particle_size_um,
                    solubility_mg_mL = solubility_mg_mL,
                    pKa = pKa,
                    charge_type = charge_type,
                    segment = segment,
                    undissolved_mg = undissolved[i]
                )
                dissolved_this_step = min(diss.dissolution_rate_mg_min * dt_min, undissolved[i])
                undissolved[i] -= dissolved_this_step
                dissolved[i] += dissolved_this_step
            end

            # Absorption using integrated model (except stomach and colon)
            if segment in [DUODENUM, JEJUNUM, ILEUM] && dissolved[i] > 0.01
                # Use enhanced transporter model
                abs_result = calculate_integrated_absorption(
                    drug_name = drug_name,
                    dose_mg = dissolved[i],  # Current dissolved amount
                    logP = logP,
                    MW = MW,
                    pKa = pKa,
                    charge_type = charge_type,
                    segment = segment,
                    intrinsic_er = intrinsic_er,
                    drug_class = drug_class
                )

                absorbed_this_step = min(
                    abs_result.ka_min * dissolved[i] * dt_min,
                    dissolved[i]
                )
                dissolved[i] -= absorbed_this_step
                absorbed += absorbed_this_step
                total_absorbed_step += absorbed_this_step

                # Track carrier contribution
                if abs_result.carrier_fraction > 0
                    total_carrier_absorbed += absorbed_this_step * abs_result.carrier_fraction
                end

                # Apply gut first-pass (CYP3A4 + gut wall specific metabolism)
                # Fg_gut_wall accounts for AADC (levodopa), intestinal CYP3A4, etc.
                to_portal = absorbed_this_step * fp.Fg * Fg_gut_wall * sat_factor
                portal_vein += to_portal
            end

            # Transit to next segment
            transit_rate = 1.0 / phys.transit_time_min
            if i < length(instances(GISegment))
                # Transfer undissolved
                transfer_undiss = undissolved[i] * transit_rate * dt_min
                undissolved[i] -= transfer_undiss
                undissolved[i+1] += transfer_undiss

                # Transfer dissolved
                transfer_diss = dissolved[i] * transit_rate * dt_min
                dissolved[i] -= transfer_diss
                dissolved[i+1] += transfer_diss
            end
        end

        # Hepatic first-pass
        systemic_new = portal_vein * fp.Fh
        portal_vein = 0.0
        systemic += systemic_new

        # Record
        push!(times, t_min / 60.0)
        push!(absorbed_cumulative, absorbed)
        push!(systemic_cumulative, systemic)
        push!(carrier_contribution, total_absorbed_step > 0 ?
              total_carrier_absorbed / max(absorbed, 0.01) : 0.0)
    end

    # Calculate bioavailability
    Fa = absorbed / dose_mg
    F = systemic / dose_mg
    carrier_fraction_total = total_carrier_absorbed / max(absorbed, 0.01)

    # Effective Fg includes gut wall metabolism and saturation
    Fg_effective = fp.Fg * Fg_gut_wall * sat_factor

    return (
        times_h = times,
        absorbed_mg = absorbed_cumulative,
        systemic_mg = systemic_cumulative,
        Fa = Fa,
        Fg = Fg_effective,  # Now includes gut wall metabolism
        Fh = fp.Fh,
        F = F,
        F_percent = F * 100,
        carrier_fraction = carrier_fraction_total,
        carrier_absorbed_mg = total_carrier_absorbed,
        tmax_h = times[argmax(diff(systemic_cumulative))],
        final_undissolved_mg = sum(undissolved),
        final_dissolved_mg = sum(dissolved)
    )
end

end # module
