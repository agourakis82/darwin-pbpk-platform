# ===========================================================================
# BRAIN SUB-COMPARTMENT MODEL
# ===========================================================================
# Multi-compartment brain model for regional drug distribution
#
# Compartments:
# 1. Grey Matter (cortex, basal ganglia, hippocampus, thalamus)
# 2. White Matter (corpus callosum, internal capsule, myelin-rich)
# 3. Deep Structures (substantia nigra, VTA, hypothalamus)
# 4. Glial Compartment (astrocytes, microglia, oligodendrocytes)
# 5. Ventricular CSF
# 6. Subarachnoid CSF
#
# Key physiological features:
# - Regional blood flow differences (grey >> white)
# - Lipid composition differences (white > grey)
# - Transporter expression heterogeneity
# - Receptor binding effects (D2, 5-HT2A, etc.)
# ===========================================================================

module BrainSubcompartments

export BrainRegion, BrainSubcompartmentParams, BrainRegionalState
export calculate_regional_distribution, calculate_receptor_occupancy
export GREY_MATTER, WHITE_MATTER, DEEP_STRUCTURES, GLIAL
export VENTRICULAR_CSF, SUBARACHNOID_CSF
export RegionalTransporters, calculate_regional_kpuu

# ===========================================================================
# BRAIN REGION DEFINITIONS
# ===========================================================================

@enum BrainRegion begin
    GREY_MATTER         # Cortex, hippocampus - receptor-rich
    WHITE_MATTER        # Myelin-rich, lipid reservoir
    DEEP_STRUCTURES     # Basal ganglia, substantia nigra, VTA
    GLIAL               # Astrocytes, microglia
    VENTRICULAR_CSF     # Lateral, 3rd, 4th ventricles
    SUBARACHNOID_CSF    # Cisterns, spinal
end

# ===========================================================================
# REGIONAL PHYSIOLOGICAL PARAMETERS
# ===========================================================================

"""
Regional brain physiology parameters based on literature.

References:
- Syková & Nicholson 2008 (Physiol Rev) - Brain extracellular space
- Hladky & Barrand 2014 (Fluids Barriers CNS) - BBB physiology
- de Lange 2013 (CNS Drug Del) - Brain PK modeling
"""
struct RegionalPhysiology
    # Volumes (fraction of total brain)
    volume_fraction::Float64

    # Blood flow (mL/min/100g tissue)
    blood_flow::Float64

    # Lipid content (fraction)
    lipid_fraction::Float64

    # Extracellular fluid fraction
    ecf_fraction::Float64

    # Protein content (g/100g tissue)
    protein_content::Float64

    # pH
    pH::Float64
end

# Literature values for regional physiology
const REGIONAL_PHYSIOLOGY = Dict{BrainRegion, RegionalPhysiology}(
    GREY_MATTER => RegionalPhysiology(
        0.45,      # 45% of brain volume
        80.0,      # High blood flow (80 mL/min/100g)
        0.36,      # Lower lipid (36%)
        0.20,      # 20% ECF
        10.5,      # Higher protein
        7.0        # pH 7.0
    ),
    WHITE_MATTER => RegionalPhysiology(
        0.40,      # 40% of brain volume
        25.0,      # Lower blood flow (25 mL/min/100g)
        0.70,      # High lipid (70% - myelin)
        0.15,      # 15% ECF
        8.0,       # Lower protein
        7.0        # pH 7.0
    ),
    DEEP_STRUCTURES => RegionalPhysiology(
        0.08,      # 8% of brain volume
        60.0,      # Moderate-high blood flow
        0.45,      # Moderate lipid
        0.18,      # 18% ECF
        11.0,      # High protein (receptors)
        7.0        # pH 7.0
    ),
    GLIAL => RegionalPhysiology(
        0.07,      # 7% of brain volume (distributed)
        50.0,      # Variable blood flow
        0.40,      # Moderate lipid
        0.25,      # Higher ECF (astrocyte endfeet)
        9.0,       # Moderate protein
        6.8        # Slightly acidic (lactate)
    ),
    VENTRICULAR_CSF => RegionalPhysiology(
        0.03,      # 3% (ventricles ~30mL)
        0.0,       # No direct blood flow
        0.01,      # Minimal lipid
        1.0,       # All fluid
        0.03,      # Very low protein (0.3 g/L)
        7.33       # CSF pH
    ),
    SUBARACHNOID_CSF => RegionalPhysiology(
        0.10,      # 10% (~100mL)
        0.0,       # No direct blood flow
        0.01,      # Minimal lipid
        1.0,       # All fluid
        0.03,      # Very low protein
        7.33       # CSF pH
    )
)

# ===========================================================================
# REGIONAL TRANSPORTER EXPRESSION
# ===========================================================================

"""
Regional transporter expression levels (relative to average BBB)

Based on:
- Uchida et al. 2011 (J Neurochem) - Quantitative proteomics
- Hoshi et al. 2013 (J Cereb Blood Flow Metab) - Regional P-gp
- Shawahna et al. 2011 (Mol Pharm) - Human BBB transporters
"""
struct RegionalTransporters
    # Efflux transporters (relative expression, 1.0 = average)
    pgp::Float64        # P-glycoprotein (ABCB1)
    bcrp::Float64       # Breast cancer resistance protein (ABCG2)
    mrp1::Float64       # Multidrug resistance protein 1 (ABCC1)
    mrp4::Float64       # MRP4 (ABCC4)

    # Uptake transporters
    oatp1a2::Float64    # Organic anion transporting polypeptide
    oct3::Float64       # Organic cation transporter 3
    lat1::Float64       # Large amino acid transporter 1
    glut1::Float64      # Glucose transporter 1
    ent1::Float64       # Equilibrative nucleoside transporter
end

const REGIONAL_TRANSPORTERS = Dict{BrainRegion, RegionalTransporters}(
    # Grey matter - high transporter expression
    GREY_MATTER => RegionalTransporters(
        1.2,    # P-gp high (cortical capillaries)
        1.1,    # BCRP moderate-high
        0.8,    # MRP1 moderate
        0.9,    # MRP4 moderate
        1.0,    # OATP normal
        1.3,    # OCT3 high (neuronal)
        1.2,    # LAT1 high (neurons need amino acids)
        1.5,    # GLUT1 high (glucose demand)
        1.0     # ENT1 normal
    ),
    # White matter - lower transporter expression
    WHITE_MATTER => RegionalTransporters(
        0.6,    # P-gp low (fewer capillaries)
        0.7,    # BCRP low
        1.2,    # MRP1 higher (oligodendrocytes)
        0.7,    # MRP4 low
        0.5,    # OATP low
        0.4,    # OCT3 low
        0.8,    # LAT1 moderate
        0.8,    # GLUT1 lower
        0.6     # ENT1 low
    ),
    # Deep structures - variable, D2-receptor rich
    DEEP_STRUCTURES => RegionalTransporters(
        1.0,    # P-gp normal
        0.9,    # BCRP normal
        1.0,    # MRP1 normal
        1.0,    # MRP4 normal
        1.5,    # OATP high! (explains antipsychotic uptake)
        1.8,    # OCT3 very high (dopaminergic regions)
        1.3,    # LAT1 high
        1.2,    # GLUT1 high
        1.2     # ENT1 moderate
    ),
    # Glial - astrocyte transporters
    GLIAL => RegionalTransporters(
        0.3,    # P-gp very low (astrocytes)
        0.4,    # BCRP low
        2.0,    # MRP1 very high! (astrocyte marker)
        1.5,    # MRP4 high
        0.8,    # OATP moderate
        1.5,    # OCT3 high
        0.7,    # LAT1 moderate
        2.0,    # GLUT1 very high (astrocyte endfeet)
        1.0     # ENT1 normal
    ),
    # Ventricular CSF - choroid plexus transporters (BCSFB)
    VENTRICULAR_CSF => RegionalTransporters(
        0.5,    # P-gp low at choroid plexus (apical)
        2.0,    # BCRP very high at choroid plexus!
        3.0,    # MRP1 very high (basolateral)
        2.5,    # MRP4 high
        2.0,    # OATP high (CSF clearance)
        0.3,    # OCT3 low
        1.5,    # LAT1 high
        1.0,    # GLUT1 normal
        1.5     # ENT1 moderate
    ),
    # Subarachnoid CSF - minimal transporters
    SUBARACHNOID_CSF => RegionalTransporters(
        0.1,    # P-gp minimal
        0.1,    # BCRP minimal
        0.2,    # MRP1 low
        0.2,    # MRP4 low
        0.1,    # OATP minimal
        0.1,    # OCT3 minimal
        0.1,    # LAT1 minimal
        0.3,    # GLUT1 low
        0.2     # ENT1 low
    )
)

# ===========================================================================
# RECEPTOR DENSITY BY REGION
# ===========================================================================

"""
Receptor density for PET-relevant targets (relative, 1.0 = reference region)

Based on:
- Farde et al. 1986 (Science) - D2 receptor PET
- Pike et al. 1996 (Eur J Pharmacol) - 5-HT2A distribution
- Innis et al. 2007 (J Cereb Blood Flow Metab) - PET standards
"""
struct ReceptorDensity
    D1::Float64         # Dopamine D1
    D2::Float64         # Dopamine D2/D3
    D4::Float64         # Dopamine D4
    HT2A::Float64       # Serotonin 5-HT2A
    HT1A::Float64       # Serotonin 5-HT1A
    M1::Float64         # Muscarinic M1
    H1::Float64         # Histamine H1
    GABA_A::Float64     # GABA-A
    NMDA::Float64       # NMDA glutamate
end

const RECEPTOR_DENSITY = Dict{BrainRegion, ReceptorDensity}(
    # Grey matter (cortex) - 5-HT2A, NMDA rich
    GREY_MATTER => ReceptorDensity(
        0.5,    # D1 moderate
        0.3,    # D2 low (cortical)
        0.4,    # D4 moderate
        1.5,    # 5-HT2A high (frontal cortex)
        1.2,    # 5-HT1A high
        1.0,    # M1 normal
        0.8,    # H1 moderate
        1.5,    # GABA-A high
        1.5     # NMDA high
    ),
    # White matter - minimal receptors
    WHITE_MATTER => ReceptorDensity(
        0.1,    # D1 minimal
        0.05,   # D2 minimal
        0.1,    # D4 minimal
        0.1,    # 5-HT2A minimal
        0.1,    # 5-HT1A minimal
        0.2,    # M1 low
        0.2,    # H1 low
        0.3,    # GABA-A low
        0.2     # NMDA low
    ),
    # Deep structures (basal ganglia, striatum) - D2 HOTSPOT
    DEEP_STRUCTURES => ReceptorDensity(
        2.5,    # D1 very high (striatum)
        5.0,    # D2 VERY HIGH (striatum, SN) - TARGET!
        1.5,    # D4 high
        0.5,    # 5-HT2A moderate
        0.8,    # 5-HT1A moderate
        0.6,    # M1 moderate
        0.5,    # H1 moderate
        2.0,    # GABA-A high (striatopallidal)
        1.0     # NMDA normal
    ),
    # Glial - mainly glutamate/GABA
    GLIAL => ReceptorDensity(
        0.2,    # D1 low
        0.3,    # D2 low (some on microglia)
        0.2,    # D4 low
        0.3,    # 5-HT2A low
        0.2,    # 5-HT1A low
        0.3,    # M1 low
        0.8,    # H1 moderate (microglia)
        0.5,    # GABA-A moderate
        1.2     # NMDA moderate (astrocyte NMDA)
    ),
    # CSF - no receptors
    VENTRICULAR_CSF => ReceptorDensity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    SUBARACHNOID_CSF => ReceptorDensity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
)

# ===========================================================================
# REGIONAL Kp,uu CALCULATION
# ===========================================================================

"""
Calculate region-specific Kp,uu based on local transporter expression
and tissue composition.

This explains why:
- Antipsychotics concentrate in basal ganglia (high OATP, OCT3)
- Lipophilic drugs accumulate in white matter (lipid partitioning)
- Hydrophilic drugs stay in CSF (no tissue binding)
"""
function calculate_regional_kpuu(;
    base_kpuu::Float64,
    region::BrainRegion,
    logP::Float64,
    charge_type::Symbol,
    drug_class::Symbol = :unknown,
    is_oatp_substrate::Bool = false,
    is_oct_substrate::Bool = false,
    is_lat1_substrate::Bool = false
)
    phys = REGIONAL_PHYSIOLOGY[region]
    trans = REGIONAL_TRANSPORTERS[region]

    regional_kpuu = base_kpuu

    # 1. LIPID PARTITIONING
    # White matter accumulates lipophilic drugs
    if logP > 3.0
        lipid_factor = 1.0 + (phys.lipid_fraction - 0.4) * (logP - 3.0) * 0.3
        regional_kpuu *= lipid_factor
    end

    # 2. TRANSPORTER EFFECTS

    # P-gp efflux (reduces Kp,uu in high-expression regions)
    if drug_class in [:antipsychotic, :opioid, :cardiac] || base_kpuu < 0.5
        pgp_effect = 1.0 / (1.0 + 0.3 * (trans.pgp - 1.0))
        regional_kpuu *= pgp_effect
    end

    # OATP uptake (critical for antipsychotics!)
    if is_oatp_substrate || drug_class == :antipsychotic
        oatp_effect = 1.0 + 0.5 * (trans.oatp1a2 - 1.0)
        regional_kpuu *= oatp_effect
    end

    # OCT3 uptake (cationic drugs)
    if is_oct_substrate || (charge_type == :base && logP >= 1.0)
        oct_effect = 1.0 + 0.4 * (trans.oct3 - 1.0)
        regional_kpuu *= oct_effect
    end

    # LAT1 uptake (amino acid analogs) - CRITICAL for gabapentinoids
    if is_lat1_substrate
        # LAT1 provides active uptake that can overcome poor passive permeability
        # Gabapentin, pregabalin, vigabatrin are LAT1 substrates
        # Use saturation kinetics: very hydrophilic drugs benefit more
        lat1_boost = if logP < -1.5
            1.5  # Very hydrophilic (pregabalin, vigabatrin)
        elseif logP < -0.5
            1.0  # Moderately hydrophilic (gabapentin)
        else
            0.5  # Lipophilic drugs don't need LAT1 as much
        end
        lat1_effect = 1.0 + lat1_boost * trans.lat1
        regional_kpuu *= lat1_effect
    elseif drug_class == :anticonvulsant
        # Non-LAT1 anticonvulsants get moderate benefit
        lat1_effect = 1.0 + 0.3 * (trans.lat1 - 1.0)
        regional_kpuu *= lat1_effect
    end

    # 3. CSF EQUILIBRIUM
    # CSF behaves differently - approaches plasma free fraction
    if region in [VENTRICULAR_CSF, SUBARACHNOID_CSF]
        # CSF/plasma ratio approaches fup for unbound drugs
        # But BCSFB has strong efflux
        bcrp_effect = 1.0 / (1.0 + 0.5 * trans.bcrp)
        regional_kpuu = base_kpuu * bcrp_effect * 0.8  # CSF typically lower
    end

    # 4. GLIAL UPTAKE
    # MRP1 is the major astrocyte efflux pump
    if region == GLIAL
        mrp1_effect = 1.0 / (1.0 + 0.2 * (trans.mrp1 - 1.0))
        regional_kpuu *= mrp1_effect
    end

    # 5. LYSOSOMOTROPIC TRAPPING (for lipophilic bases)
    # Many antipsychotics are weak bases that accumulate in acidic lysosomes
    # This provides an intracellular reservoir that increases apparent Kp,uu
    # References:
    # - Daniel & Bhatt 1997 (J Pharm Exp Ther) - Lysosomotropic accumulation
    # - Kazmi et al. 2013 (Drug Metab Dispos) - Brain lysosomal trapping
    if charge_type == :base && logP >= 3.0 && drug_class == :antipsychotic
        # Lysosomotropic drugs: lipophilic weak bases
        # pH gradient: cytosol 7.2 → lysosome 4.5-5.0
        # Ion trapping ratio can be 100-1000x
        lysosome_factor = 1.0 + 0.3 * (logP - 3.0)  # Increases with lipophilicity
        regional_kpuu *= lysosome_factor
    end

    return clamp(regional_kpuu, 0.001, 20.0)
end

# ===========================================================================
# RECEPTOR OCCUPANCY CALCULATION
# ===========================================================================

"""
Calculate receptor occupancy at a given regional concentration.

Based on Hill equation: Occupancy = C^n / (C^n + EC50^n)

This is critical for:
- Antipsychotic D2 occupancy (therapeutic: 65-80%)
- Atypical antipsychotic 5-HT2A/D2 ratio
- SSRI 5-HT transporter occupancy
"""
struct DrugReceptorBinding
    D2_Ki::Float64      # D2 affinity (nM)
    HT2A_Ki::Float64    # 5-HT2A affinity (nM)
    H1_Ki::Float64      # H1 affinity (nM)
    M1_Ki::Float64      # M1 affinity (nM)
end

# Receptor binding data for key antipsychotics (Ki in nM)
const ANTIPSYCHOTIC_BINDING = Dict{String, DrugReceptorBinding}(
    "haloperidol" => DrugReceptorBinding(1.4, 45.0, 1000.0, 1000.0),
    "risperidone" => DrugReceptorBinding(3.3, 0.16, 5.2, 1000.0),
    "olanzapine" => DrugReceptorBinding(20.0, 1.5, 2.5, 4.7),
    "clozapine" => DrugReceptorBinding(130.0, 3.3, 1.8, 6.2),
    "quetiapine" => DrugReceptorBinding(180.0, 31.0, 7.0, 120.0),
    "aripiprazole" => DrugReceptorBinding(0.95, 8.7, 27.0, 1000.0),
    "ziprasidone" => DrugReceptorBinding(3.1, 0.73, 22.0, 1000.0),
    "paliperidone" => DrugReceptorBinding(2.8, 1.2, 3.4, 1000.0),
    "lurasidone" => DrugReceptorBinding(1.7, 0.47, 1000.0, 1000.0),
    "asenapine" => DrugReceptorBinding(1.3, 0.03, 1.2, 8.1),
)

"""
Calculate D2 receptor occupancy in deep structures (target for antipsychotics)
"""
function calculate_d2_occupancy(;
    drug_name::String,
    plasma_conc_nM::Float64,
    fup::Float64,
    regional_kpuu::Float64
)
    if !haskey(ANTIPSYCHOTIC_BINDING, lowercase(drug_name))
        return nothing
    end

    binding = ANTIPSYCHOTIC_BINDING[lowercase(drug_name)]

    # Free brain concentration in deep structures
    brain_free_conc = plasma_conc_nM * fup * regional_kpuu

    # D2 occupancy (Hill equation, n=1)
    # Using Ki as proxy for EC50 (approximation)
    occupancy = brain_free_conc / (brain_free_conc + binding.D2_Ki)

    return (
        brain_free_nM = brain_free_conc,
        D2_occupancy = occupancy * 100,  # percentage
        therapeutic = 65 <= occupancy * 100 <= 80,
        eps_risk = occupancy * 100 > 80  # EPS risk above 80%
    )
end

# ===========================================================================
# REGIONAL DISTRIBUTION CALCULATION
# ===========================================================================

"""
Calculate drug distribution across all brain regions.

Returns a NamedTuple with:
- Regional Kp,uu values
- Regional concentrations
- Receptor occupancy (if applicable)
"""
function calculate_regional_distribution(;
    base_kpuu::Float64,
    plasma_conc::Float64,
    fup::Float64,
    logP::Float64,
    MW::Float64,
    charge_type::Symbol,
    drug_class::Symbol = :unknown,
    drug_name::String = "",
    is_oatp_substrate::Bool = false,
    is_oct_substrate::Bool = false,
    is_lat1_substrate::Bool = false
)
    results = Dict{BrainRegion, NamedTuple}()

    for region in instances(BrainRegion)
        regional_kpuu = calculate_regional_kpuu(
            base_kpuu = base_kpuu,
            region = region,
            logP = logP,
            charge_type = charge_type,
            drug_class = drug_class,
            is_oatp_substrate = is_oatp_substrate,
            is_oct_substrate = is_oct_substrate,
            is_lat1_substrate = is_lat1_substrate
        )

        # Regional free concentration
        regional_free_conc = plasma_conc * fup * regional_kpuu

        # Total regional concentration (with tissue binding)
        phys = REGIONAL_PHYSIOLOGY[region]
        tissue_binding = 1.0 + phys.lipid_fraction * logP * 0.5  # Simplified
        regional_total_conc = regional_free_conc * tissue_binding

        results[region] = (
            kpuu = regional_kpuu,
            free_conc = regional_free_conc,
            total_conc = regional_total_conc,
            blood_flow = phys.blood_flow
        )
    end

    # Calculate D2 occupancy for antipsychotics
    d2_occupancy = nothing
    if drug_class == :antipsychotic && !isempty(drug_name)
        deep_kpuu = results[DEEP_STRUCTURES].kpuu
        d2_occupancy = calculate_d2_occupancy(
            drug_name = drug_name,
            plasma_conc_nM = plasma_conc * 1e9 / MW,  # Convert to nM
            fup = fup,
            regional_kpuu = deep_kpuu
        )
    end

    return (
        regional = results,
        average_kpuu = sum(r.kpuu * REGIONAL_PHYSIOLOGY[reg].volume_fraction
                          for (reg, r) in results),
        d2_occupancy = d2_occupancy
    )
end

# ===========================================================================
# EXPORTS FOR TESTING
# ===========================================================================

function get_regional_physiology(region::BrainRegion)
    return REGIONAL_PHYSIOLOGY[region]
end

function get_regional_transporters(region::BrainRegion)
    return REGIONAL_TRANSPORTERS[region]
end

function get_receptor_density(region::BrainRegion)
    return RECEPTOR_DENSITY[region]
end

end # module
