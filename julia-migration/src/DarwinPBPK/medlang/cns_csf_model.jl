# ===========================================================================
# MEDLANG CNS/CSF COMPARTMENT MODEL
# ===========================================================================
# Mechanistic model of drug distribution in CNS with:
#
# COMPARTMENTS:
# 1. Brain ECF (extracellular fluid) - target site for most CNS drugs
# 2. Brain ICF (intracellular fluid) - for drugs with intracellular targets
# 3. CSF_LV (lateral ventricle) - produced by choroid plexus, BCSFB
# 4. CSF_TFV (third/fourth ventricle) - transit compartment
# 5. CSF_CM (cisterna magna) - relevant for brainstem targets
# 6. CSF_SAS (subarachnoid space/lumbar) - clinical sampling site
#
# BARRIERS:
# - BBB (blood-brain barrier): P-gp efflux TO BLOOD (restricts entry)
# - BCSFB (blood-CSF barrier): P-gp efflux TO CSF (opposite orientation!)
#
# KEY PROCESSES:
# - Passive diffusion (transcellular, paracellular)
# - Active transport (P-gp, MRP, OATP, BCRP)
# - Bulk flow: ECF → CSF (glymphatic)
# - CSF circulation: LV → TFV → CM → SAS → arachnoid villi → venous
# - Tissue binding (Kp,brain, fu,brain)
#
# Based on LeiCNS-PK3.0/3.1 model structure with MedLang DSL syntax
#
# References:
# - Yamamoto et al. 2017 (CPT:PSP) - LeiCNS-PK3.0
# - Saleh et al. 2021 (JPKPD) - CSF surrogacy
# - Uchida et al. 2011 (J Cereb Blood Flow Metab) - BBB transporters
# - Wijnholds et al. 2000 (JCI) - BCSFB MRP1
#
# Author: Dr. Sounio Agourakis
# Date: November 2025
# ===========================================================================

module CNSCSFModel

using ..MedLang

export CNSParams, BBBTransporters, BCSFBTransporters
export generate_cns_medlang, simulate_cns_distribution
export calculate_kpuu_bbb, calculate_kpuu_bcsfb
export default_bbb_transporters, default_bcsfb_transporters
export drug_preset

# ===========================================================================
# PHYSIOLOGICAL PARAMETERS
# ===========================================================================

"""
Human CNS physiological parameters from literature.

References:
- Kielbasa & Bhutta 2020 - Human CSF volumes
- Hladky & Barrand 2014 - CSF production and flow
- Abbott 2004 - BBB surface area
"""
const CNS_PHYSIOLOGY = (
    # Volumes (mL)
    V_plasma = 3000.0,              # Plasma volume
    V_brain_ECF = 250.0,            # Brain ECF (~20% of 1400g brain)
    V_brain_ICF = 1000.0,           # Brain ICF (~80% of brain)
    V_CSF_LV = 22.0,                # Lateral ventricles
    V_CSF_TFV = 4.0,                # Third + fourth ventricles
    V_CSF_CM = 5.0,                 # Cisterna magna
    V_CSF_SAS = 120.0,              # Subarachnoid space (including lumbar)
    V_CSF_total = 150.0,            # Total CSF volume

    # Blood flows (mL/min)
    Q_brain = 750.0,                # Cerebral blood flow
    Q_choroid_plexus = 3.0,         # Choroid plexus blood flow

    # CSF dynamics (mL/min)
    Q_CSF_production = 0.35,        # CSF production rate (~500 mL/day)
    Q_CSF_absorption = 0.35,        # CSF absorption at arachnoid villi
    Q_ECF_bulk_flow = 0.20,         # ECF → CSF bulk flow (glymphatic)

    # Surface areas (cm²)
    SA_BBB = 200.0,                 # BBB surface area (human)
    SA_BCSFB = 200.0,               # BCSFB surface area (choroid plexus)

    # pH values (for ionization)
    pH_plasma = 7.4,
    pH_brain_ECF = 7.3,
    pH_brain_ICF = 7.0,
    pH_CSF = 7.3,

    # Turnover
    CSF_turnover_h = 6.0,           # Complete CSF turnover (~4x/day)
)

# ===========================================================================
# TRANSPORTER DEFINITIONS
# ===========================================================================

"""
BBB transporter expression and kinetics.

KEY POINT: At BBB, efflux transporters face BLOOD side
- P-gp on luminal membrane → efflux TO blood
- This RESTRICTS drug entry into brain
"""
struct BBBTransporters
    # Efflux transporters (luminal, facing blood)
    pgp_expression::Float64         # Relative P-gp expression (1.0 = reference)
    pgp_km_uM::Float64              # P-gp Km
    pgp_vmax_pmol_min_cm2::Float64  # P-gp Vmax

    bcrp_expression::Float64        # BCRP expression
    bcrp_km_uM::Float64
    bcrp_vmax_pmol_min_cm2::Float64

    mrp4_expression::Float64        # MRP4 (at BBB luminal)
    mrp4_km_uM::Float64
    mrp4_vmax_pmol_min_cm2::Float64

    # Uptake transporters (abluminal or bidirectional)
    oatp1a2_expression::Float64     # OATP1A2
    oatp2b1_expression::Float64     # OATP2B1
    lat1_expression::Float64        # LAT1 (large amino acids)
    glut1_expression::Float64       # GLUT1 (glucose)
end

"""
Default BBB transporter parameters.
"""
function default_bbb_transporters()::BBBTransporters
    return BBBTransporters(
        # P-gp (major efflux)
        1.0, 10.0, 5000.0,
        # BCRP
        1.0, 5.0, 3000.0,
        # MRP4
        0.5, 50.0, 1000.0,
        # Uptake transporters
        0.5, 0.8, 1.0, 1.0
    )
end

"""
BCSFB transporter expression and kinetics.

KEY POINT: At BCSFB, transporter orientation is OPPOSITE to BBB!
- P-gp on APICAL membrane (facing CSF) → efflux INTO CSF
- MRP1 on BASOLATERAL membrane → efflux TO blood (clears CSF)
- This means P-gp substrates may have HIGHER CSF than expected
"""
struct BCSFBTransporters
    # P-gp (APICAL - facing CSF - efflux INTO CSF)
    pgp_expression::Float64         # Relative expression
    pgp_km_uM::Float64
    pgp_vmax_pmol_min_cm2::Float64

    # MRP1 (BASOLATERAL - facing blood - clears CSF)
    mrp1_expression::Float64
    mrp1_km_uM::Float64
    mrp1_vmax_pmol_min_cm2::Float64

    # MRP4 (BASOLATERAL)
    mrp4_expression::Float64
    mrp4_km_uM::Float64
    mrp4_vmax_pmol_min_cm2::Float64

    # OATPs (bidirectional)
    oatp1_expression::Float64       # OATP1 (apical - CSF uptake)
    oatp2_expression::Float64       # OATP2 (basolateral - blood uptake)

    # BCRP (apical)
    bcrp_expression::Float64
    bcrp_km_uM::Float64
end

"""
Default BCSFB transporter parameters.
"""
function default_bcsfb_transporters()::BCSFBTransporters
    return BCSFBTransporters(
        # P-gp (apical → CSF)
        0.8, 10.0, 3000.0,
        # MRP1 (basolateral → blood)
        1.0, 20.0, 4000.0,
        # MRP4
        0.5, 50.0, 1000.0,
        # OATPs
        0.6, 0.8,
        # BCRP
        0.5, 5.0
    )
end

# ===========================================================================
# CNS DRUG PARAMETERS
# ===========================================================================

"""
Complete CNS distribution parameters for a drug.
"""
struct CNSParams
    # Drug identification
    drug_name::String

    # Physicochemistry
    MW::Float64
    logP::Float64
    pKa::Union{Float64, Nothing}
    charge_type::Symbol             # :neutral, :acid, :base, :zwitterion
    fu_plasma::Float64              # Unbound fraction in plasma

    # Brain binding
    fu_brain::Float64               # Unbound fraction in brain homogenate
    Kp_brain::Float64               # Total brain/plasma partition coefficient

    # Passive permeability
    Papp_BBB_cm_s::Float64          # BBB passive permeability
    Papp_BCSFB_cm_s::Float64        # BCSFB passive permeability

    # Transporter substrates
    is_pgp_substrate::Bool
    pgp_km_uM::Float64
    is_bcrp_substrate::Bool
    bcrp_km_uM::Float64
    is_mrp_substrate::Bool          # MRP1 or MRP4
    mrp_km_uM::Float64
    is_oatp_substrate::Bool

    # Active influx (carrier-mediated)
    is_lat1_substrate::Bool         # Large neutral amino acids
    is_glut_substrate::Bool         # Glucose transporter

    # Target location
    target_compartment::Symbol      # :brain_ecf, :brain_icf, :csf_cm, etc.
end

# ===========================================================================
# Kp,uu CALCULATIONS
# ===========================================================================

"""
Calculate Kp,uu at BBB (brain ECF / plasma unbound).

Kp,uu,BBB = CL_influx / CL_efflux

At steady state:
- Kp,uu = 1: equilibrium (passive diffusion dominates)
- Kp,uu < 1: net efflux (P-gp restricts entry)
- Kp,uu > 1: net influx (carrier-mediated uptake)
"""
function calculate_kpuu_bbb(
    params::CNSParams,
    bbb::BBBTransporters;
    C_plasma_uM::Float64 = 1.0
)::Float64
    # Passive influx clearance
    CL_passive = params.Papp_BBB_cm_s * CNS_PHYSIOLOGY.SA_BBB * 60.0  # cm³/min

    # P-gp efflux (Michaelis-Menten)
    if params.is_pgp_substrate
        # At BBB, P-gp reduces brain exposure
        CL_pgp_efflux = (bbb.pgp_vmax_pmol_min_cm2 * bbb.pgp_expression *
                         CNS_PHYSIOLOGY.SA_BBB) /
                        (bbb.pgp_km_uM + C_plasma_uM)
    else
        CL_pgp_efflux = 0.0
    end

    # BCRP efflux
    if params.is_bcrp_substrate
        CL_bcrp_efflux = (bbb.bcrp_vmax_pmol_min_cm2 * bbb.bcrp_expression *
                          CNS_PHYSIOLOGY.SA_BBB) /
                         (bbb.bcrp_km_uM + C_plasma_uM)
    else
        CL_bcrp_efflux = 0.0
    end

    # Total efflux
    CL_efflux_total = CL_passive + CL_pgp_efflux + CL_bcrp_efflux

    # Kp,uu = influx / efflux
    # For passive drug: Kp,uu ≈ 1
    # For P-gp substrate: Kp,uu < 1 (efflux dominates)
    kpuu = CL_passive / CL_efflux_total

    return clamp(kpuu, 0.01, 10.0)
end

"""
Calculate Kp,uu at BCSFB (CSF / plasma unbound).

KEY DIFFERENCE: P-gp at BCSFB pumps INTO CSF!
- P-gp substrate → higher CSF than expected
- MRP1 clears CSF → lower CSF
"""
function calculate_kpuu_bcsfb(
    params::CNSParams,
    bcsfb::BCSFBTransporters;
    C_plasma_uM::Float64 = 1.0
)::Float64
    # Passive influx clearance (blood → CSF)
    CL_passive = params.Papp_BCSFB_cm_s * CNS_PHYSIOLOGY.SA_BCSFB * 60.0

    # P-gp at BCSFB: INCREASES CSF concentration!
    # P-gp on apical (CSF) side pumps INTO CSF from choroid plexus cells
    if params.is_pgp_substrate
        CL_pgp_to_csf = (bcsfb.pgp_vmax_pmol_min_cm2 * bcsfb.pgp_expression *
                         CNS_PHYSIOLOGY.SA_BCSFB) /
                        (bcsfb.pgp_km_uM + C_plasma_uM)
    else
        CL_pgp_to_csf = 0.0
    end

    # MRP1 at BCSFB: DECREASES CSF concentration
    # MRP1 on basolateral side pumps back to blood
    if params.is_mrp_substrate
        CL_mrp1_to_blood = (bcsfb.mrp1_vmax_pmol_min_cm2 * bcsfb.mrp1_expression *
                            CNS_PHYSIOLOGY.SA_BCSFB) /
                           (bcsfb.mrp1_km_uM + C_plasma_uM)
    else
        CL_mrp1_to_blood = 0.0
    end

    # Net influx to CSF
    CL_influx = CL_passive + CL_pgp_to_csf
    CL_efflux = CL_mrp1_to_blood + CNS_PHYSIOLOGY.Q_CSF_absorption  # CSF turnover

    # Kp,uu,BCSFB
    # For P-gp substrate: may be > 1 (P-gp pumps INTO CSF)
    kpuu = CL_influx / (CL_influx + CL_efflux)

    return clamp(kpuu, 0.01, 10.0)
end

# ===========================================================================
# MEDLANG CODE GENERATION
# ===========================================================================

"""
Generate MedLang DSL code for complete CNS/CSF model.
"""
function generate_cns_medlang(params::CNSParams; include_systemic::Bool = true)::String
    buf = IOBuffer()

    # Calculate Kp,uu values
    bbb = default_bbb_transporters()
    bcsfb = default_bcsfb_transporters()
    kpuu_bbb = calculate_kpuu_bbb(params, bbb)
    kpuu_bcsfb = calculate_kpuu_bcsfb(params, bcsfb)

    println(buf, """
model $(params.drug_name)_CNS_PBPK {
    // ================================================================
    // CNS/CSF DISTRIBUTION MODEL
    // Generated by Darwin PBPK Platform - MedLang DSL
    // ================================================================
    // Drug: $(params.drug_name)
    // MW: $(params.MW) Da
    // logP: $(params.logP)
    // fu,plasma: $(params.fu_plasma)
    // fu,brain: $(params.fu_brain)
    // Kp,brain: $(params.Kp_brain)
    //
    // Transporter substrates:
    //   P-gp: $(params.is_pgp_substrate)
    //   BCRP: $(params.is_bcrp_substrate)
    //   MRP: $(params.is_mrp_substrate)
    //   OATP: $(params.is_oatp_substrate)
    //
    // Calculated Kp,uu values:
    //   Kp,uu,BBB: $(round(kpuu_bbb, digits=3))
    //   Kp,uu,BCSFB: $(round(kpuu_bcsfb, digits=3))
    // ================================================================

    route: iv  // CNS models typically use IV for PK analysis

    // ================================================================
    // BLOOD-BRAIN BARRIER (BBB)
    // Transporters face BLOOD side: efflux RESTRICTS brain entry
    // ================================================================
    barrier BBB {
        surface_area: $(CNS_PHYSIOLOGY.SA_BBB)_cm2
        blood_flow: $(CNS_PHYSIOLOGY.Q_brain)_mL/min

        passive_permeability: $(params.Papp_BBB_cm_s)_cm/s

        // Efflux transporters (luminal - facing blood)
        transporters_luminal {
            PGP: {
                substrate: $(params.is_pgp_substrate),
                Km: $(params.pgp_km_uM)_uM,
                direction: blood,  // efflux TO blood
                effect: restricts_brain_entry
            }
            BCRP: {
                substrate: $(params.is_bcrp_substrate),
                Km: $(params.bcrp_km_uM)_uM,
                direction: blood
            }
        }

        // Uptake transporters (abluminal or bidirectional)
        transporters_abluminal {
            OATP: { substrate: $(params.is_oatp_substrate) }
            LAT1: { substrate: $(params.is_lat1_substrate) }
        }

        Kpuu: $(round(kpuu_bbb, digits=3))
    }

    // ================================================================
    // BLOOD-CSF BARRIER (BCSFB) - Choroid Plexus
    // CRITICAL: P-gp faces CSF side - efflux INTO CSF!
    // ================================================================
    barrier BCSFB {
        surface_area: $(CNS_PHYSIOLOGY.SA_BCSFB)_cm2
        blood_flow: $(CNS_PHYSIOLOGY.Q_choroid_plexus)_mL/min

        passive_permeability: $(params.Papp_BCSFB_cm_s)_cm/s

        // P-gp on APICAL (CSF-facing) membrane
        // This INCREASES CSF concentration for P-gp substrates!
        transporters_apical {
            PGP: {
                substrate: $(params.is_pgp_substrate),
                Km: $(params.pgp_km_uM)_uM,
                direction: csf,  // efflux INTO CSF
                effect: increases_csf_concentration
            }
            BCRP: {
                substrate: $(params.is_bcrp_substrate),
                direction: csf
            }
            OATP1: {
                substrate: $(params.is_oatp_substrate),
                direction: uptake_from_csf
            }
        }

        // MRP1 on BASOLATERAL (blood-facing) membrane
        // This CLEARS drug from CSF/choroid plexus
        transporters_basolateral {
            MRP1: {
                substrate: $(params.is_mrp_substrate),
                Km: $(params.mrp_km_uM)_uM,
                direction: blood,  // efflux TO blood
                effect: clears_csf
            }
            OATP2: {
                substrate: $(params.is_oatp_substrate),
                direction: uptake_from_blood
            }
        }

        Kpuu: $(round(kpuu_bcsfb, digits=3))
    }

    // ================================================================
    // BRAIN COMPARTMENTS
    // ================================================================
    compartment brain_ECF {
        volume: $(CNS_PHYSIOLOGY.V_brain_ECF)_mL
        pH: $(CNS_PHYSIOLOGY.pH_brain_ECF)

        // ECF is target site for most CNS drugs
        // Connected to: plasma (via BBB), CSF_LV (via bulk flow)

        input_from: BBB
        output_to: CSF_LV  // bulk flow (glymphatic)
        bulk_flow: $(CNS_PHYSIOLOGY.Q_ECF_bulk_flow)_mL/min

        fu: $(params.fu_brain)  // unbound fraction
    }

    compartment brain_ICF {
        volume: $(CNS_PHYSIOLOGY.V_brain_ICF)_mL
        pH: $(CNS_PHYSIOLOGY.pH_brain_ICF)

        // Intracellular - for drugs with intracellular targets
        // Passive diffusion from ECF, pH-dependent trapping

        input_from: brain_ECF
        diffusion_clearance: $(params.Papp_BBB_cm_s * 1000)_mL/min
    }

    // Tissue binding creates the "sink" that maintains ECF > CSF
    tissue_binding {
        Kp_brain: $(params.Kp_brain)
        fu_brain: $(params.fu_brain)
        // High Kp + low fu = strong tissue sink = high ECF/CSF ratio
    }

    // ================================================================
    // CSF COMPARTMENTS (in series)
    // Flow: LV → TFV → CM → SAS → arachnoid villi → venous
    // ================================================================
    compartment CSF_LV {
        // Lateral ventricles - CSF produced here by choroid plexus
        volume: $(CNS_PHYSIOLOGY.V_CSF_LV)_mL
        pH: $(CNS_PHYSIOLOGY.pH_CSF)

        production_rate: $(CNS_PHYSIOLOGY.Q_CSF_production)_mL/min
        input_from: BCSFB, brain_ECF  // BCSFB + glymphatic bulk flow
        output_to: CSF_TFV
    }

    compartment CSF_TFV {
        // Third and fourth ventricles - transit compartment
        volume: $(CNS_PHYSIOLOGY.V_CSF_TFV)_mL
        pH: $(CNS_PHYSIOLOGY.pH_CSF)

        input_from: CSF_LV
        output_to: CSF_CM
    }

    compartment CSF_CM {
        // Cisterna magna - relevant for brainstem targets
        // Antipsychotics: D2 receptors in VTA, substantia nigra
        // Anatomically close to basilar cisterns
        volume: $(CNS_PHYSIOLOGY.V_CSF_CM)_mL
        pH: $(CNS_PHYSIOLOGY.pH_CSF)

        input_from: CSF_TFV
        output_to: CSF_SAS

        // Brainstem access
        brainstem_proximity: true
        target_nuclei: [VTA, substantia_nigra, raphe, locus_coeruleus]
    }

    compartment CSF_SAS {
        // Subarachnoid space including lumbar region
        // This is where clinical lumbar puncture samples
        // Reflects drug that ESCAPED brain uptake
        volume: $(CNS_PHYSIOLOGY.V_CSF_SAS)_mL
        pH: $(CNS_PHYSIOLOGY.pH_CSF)

        input_from: CSF_CM
        output_to: arachnoid_villi

        clinical_sampling: lumbar_puncture
        interpretation: "reflects escaped drug, not brain exposure"
    }

    // CSF reabsorption
    sink arachnoid_villi {
        absorption_rate: $(CNS_PHYSIOLOGY.Q_CSF_absorption)_mL/min
        output_to: venous_blood
    }

    // ================================================================
    // STATE VARIABLES
    // ================================================================
    state C_plasma: Concentration = 0.0_uM
    state C_brain_ECF: Concentration = 0.0_uM
    state C_brain_ICF: Concentration = 0.0_uM
    state C_CSF_LV: Concentration = 0.0_uM
    state C_CSF_TFV: Concentration = 0.0_uM
    state C_CSF_CM: Concentration = 0.0_uM
    state C_CSF_SAS: Concentration = 0.0_uM

    // ================================================================
    // PARAMETERS
    // ================================================================
    param fu_plasma: Real = $(params.fu_plasma)
    param fu_brain: Real = $(params.fu_brain)
    param Kpuu_BBB: Real = $(round(kpuu_bbb, digits=3))
    param Kpuu_BCSFB: Real = $(round(kpuu_bcsfb, digits=3))

    // Clearances (mL/min)
    param CL_BBB_passive: Real = $(params.Papp_BBB_cm_s * CNS_PHYSIOLOGY.SA_BBB * 60.0)
    param CL_BCSFB_passive: Real = $(params.Papp_BCSFB_cm_s * CNS_PHYSIOLOGY.SA_BCSFB * 60.0)
    param Q_bulk_flow: Real = $(CNS_PHYSIOLOGY.Q_ECF_bulk_flow)
    param Q_CSF_flow: Real = $(CNS_PHYSIOLOGY.Q_CSF_production)

    // ================================================================
    // ODE EQUATIONS
    // ================================================================

    // Brain ECF: BBB exchange + bulk flow out to CSF
    ode dC_brain_ECF/dt = (
        Kpuu_BBB * CL_BBB_passive * fu_plasma * C_plasma
        - CL_BBB_passive * fu_brain * C_brain_ECF
        - Q_bulk_flow * C_brain_ECF
    ) / V_brain_ECF

    // Brain ICF: diffusion from ECF (pH-dependent)
    ode dC_brain_ICF/dt = (
        CL_ECF_ICF * (C_brain_ECF - C_brain_ICF * fu_brain / fu_ICF)
    ) / V_brain_ICF

    // CSF_LV: BCSFB input + bulk flow from ECF
    ode dC_CSF_LV/dt = (
        Kpuu_BCSFB * CL_BCSFB_passive * fu_plasma * C_plasma
        + Q_bulk_flow * C_brain_ECF
        - Q_CSF_flow * C_CSF_LV
    ) / V_CSF_LV

    // CSF_TFV: transit
    ode dC_CSF_TFV/dt = (
        Q_CSF_flow * (C_CSF_LV - C_CSF_TFV)
    ) / V_CSF_TFV

    // CSF_CM: transit (brainstem relevant)
    ode dC_CSF_CM/dt = (
        Q_CSF_flow * (C_CSF_TFV - C_CSF_CM)
    ) / V_CSF_CM

    // CSF_SAS: transit + absorption
    ode dC_CSF_SAS/dt = (
        Q_CSF_flow * C_CSF_CM
        - Q_CSF_absorption * C_CSF_SAS
    ) / V_CSF_SAS

    // ================================================================
    // OBSERVABLES
    // ================================================================

    // Unbound concentrations (therapeutically relevant)
    observable Cu_plasma = fu_plasma * C_plasma
    observable Cu_brain_ECF = fu_brain * C_brain_ECF
    observable Cu_CSF_LV = C_CSF_LV  // CSF has no protein binding
    observable Cu_CSF_CM = C_CSF_CM
    observable Cu_CSF_SAS = C_CSF_SAS  // Clinical sample

    // Partition coefficients
    observable Kpuu_brain_observed = Cu_brain_ECF / Cu_plasma
    observable Kpuu_CSF_observed = Cu_CSF_SAS / Cu_plasma
    observable ECF_to_CSF_ratio = Cu_brain_ECF / Cu_CSF_LV

    // Clinical interpretation
    observable brain_exposure = Cu_brain_ECF
    observable clinical_csf_sample = Cu_CSF_SAS
    observable cisternal_concentration = Cu_CSF_CM  // Antipsychotic relevant
}
""")

    return String(take!(buf))
end

# ===========================================================================
# SIMULATION
# ===========================================================================

"""
Simulate CNS drug distribution.

Returns time-course of drug in all CNS compartments.
"""
function simulate_cns_distribution(
    params::CNSParams,
    dose_mg::Float64;
    t_max_h::Float64 = 24.0,
    dt_min::Float64 = 1.0,
    plasma_model::Symbol = :one_compartment,
    CL_plasma_mL_min::Float64 = 500.0,
    Vd_L::Float64 = 70.0
)
    # Calculate derived parameters
    bbb = default_bbb_transporters()
    bcsfb = default_bcsfb_transporters()
    kpuu_bbb = calculate_kpuu_bbb(params, bbb)
    kpuu_bcsfb = calculate_kpuu_bcsfb(params, bcsfb)

    # Clearances
    CL_BBB = params.Papp_BBB_cm_s * CNS_PHYSIOLOGY.SA_BBB * 60.0  # mL/min
    CL_BCSFB = params.Papp_BCSFB_cm_s * CNS_PHYSIOLOGY.SA_BCSFB * 60.0
    Q_bulk = CNS_PHYSIOLOGY.Q_ECF_bulk_flow
    Q_csf = CNS_PHYSIOLOGY.Q_CSF_production

    # Initialize state
    n_steps = Int(ceil(t_max_h * 60 / dt_min))

    # Concentrations (µM)
    MW = params.MW
    C_plasma = (dose_mg * 1000 / MW) / (Vd_L * 1000)  # Initial plasma µM
    C_ECF = 0.0
    C_ICF = 0.0
    C_LV = 0.0
    C_TFV = 0.0
    C_CM = 0.0
    C_SAS = 0.0

    # Time series
    times = Float64[]
    plasma_profile = Float64[]
    ecf_profile = Float64[]
    csf_lv_profile = Float64[]
    csf_cm_profile = Float64[]
    csf_sas_profile = Float64[]

    ke = CL_plasma_mL_min / (Vd_L * 1000)  # Elimination rate constant

    for step in 1:n_steps
        t_min = step * dt_min

        # Plasma decay (one-compartment)
        dC_plasma = -ke * C_plasma * dt_min
        C_plasma += dC_plasma
        C_plasma = max(C_plasma, 0.0)

        # Brain ECF
        influx_ECF = kpuu_bbb * CL_BBB * params.fu_plasma * C_plasma
        efflux_ECF = CL_BBB * params.fu_brain * C_ECF + Q_bulk * C_ECF
        dC_ECF = (influx_ECF - efflux_ECF) / CNS_PHYSIOLOGY.V_brain_ECF * dt_min
        C_ECF += dC_ECF
        C_ECF = max(C_ECF, 0.0)

        # Brain ICF (simplified)
        CL_ECF_ICF = 0.1 * CL_BBB  # Assume 10% of BBB clearance
        dC_ICF = CL_ECF_ICF * (C_ECF - C_ICF) / CNS_PHYSIOLOGY.V_brain_ICF * dt_min
        C_ICF += dC_ICF
        C_ICF = max(C_ICF, 0.0)

        # CSF_LV: BCSFB + bulk flow
        influx_LV = kpuu_bcsfb * CL_BCSFB * params.fu_plasma * C_plasma + Q_bulk * C_ECF
        efflux_LV = Q_csf * C_LV
        dC_LV = (influx_LV - efflux_LV) / CNS_PHYSIOLOGY.V_CSF_LV * dt_min
        C_LV += dC_LV
        C_LV = max(C_LV, 0.0)

        # CSF_TFV
        dC_TFV = Q_csf * (C_LV - C_TFV) / CNS_PHYSIOLOGY.V_CSF_TFV * dt_min
        C_TFV += dC_TFV
        C_TFV = max(C_TFV, 0.0)

        # CSF_CM (cisternal - relevant for antipsychotics)
        dC_CM = Q_csf * (C_TFV - C_CM) / CNS_PHYSIOLOGY.V_CSF_CM * dt_min
        C_CM += dC_CM
        C_CM = max(C_CM, 0.0)

        # CSF_SAS (lumbar - clinical sample)
        dC_SAS = (Q_csf * C_CM - CNS_PHYSIOLOGY.Q_CSF_absorption * C_SAS) /
                 CNS_PHYSIOLOGY.V_CSF_SAS * dt_min
        C_SAS += dC_SAS
        C_SAS = max(C_SAS, 0.0)

        # Record
        push!(times, t_min / 60.0)
        push!(plasma_profile, C_plasma * params.fu_plasma)  # Unbound
        push!(ecf_profile, C_ECF * params.fu_brain)
        push!(csf_lv_profile, C_LV)
        push!(csf_cm_profile, C_CM)
        push!(csf_sas_profile, C_SAS)
    end

    # Calculate observed Kp,uu values
    if !isempty(ecf_profile) && !isempty(plasma_profile) && maximum(plasma_profile) > 0
        kpuu_bbb_obs = maximum(ecf_profile) / maximum(plasma_profile)
        kpuu_csf_obs = maximum(csf_sas_profile) / maximum(plasma_profile)
    else
        kpuu_bbb_obs = kpuu_bbb
        kpuu_csf_obs = kpuu_bcsfb
    end

    return Dict{String, Any}(
        "time_h" => times,
        "Cu_plasma" => plasma_profile,
        "Cu_brain_ECF" => ecf_profile,
        "C_CSF_LV" => csf_lv_profile,
        "C_CSF_CM" => csf_cm_profile,
        "C_CSF_SAS" => csf_sas_profile,
        "Kpuu_BBB_predicted" => kpuu_bbb,
        "Kpuu_BCSFB_predicted" => kpuu_bcsfb,
        "Kpuu_BBB_observed" => kpuu_bbb_obs,
        "Kpuu_CSF_observed" => kpuu_csf_obs,
        "ECF_to_CSF_ratio" => maximum(ecf_profile) / max(maximum(csf_lv_profile), 1e-10),
        "params" => params
    )
end

# ===========================================================================
# DRUG PRESETS
# ===========================================================================

"""
Create CNSParams for known drugs.
"""
function drug_preset(name::Symbol)::CNSParams
    presets = Dict(
        :risperidone => CNSParams(
            "Risperidone",
            410.5, 3.0, 8.2, :base, 0.1,
            0.05, 15.0,                    # fu_brain, Kp_brain
            5.0e-5, 3.0e-5,                # Papp BBB, BCSFB
            true, 10.0,                    # P-gp substrate
            false, 0.0,                    # BCRP
            false, 0.0,                    # MRP
            false,                         # OATP
            false, false,                  # LAT1, GLUT
            :brain_ecf
        ),
        :haloperidol => CNSParams(
            "Haloperidol",
            375.9, 4.3, 8.7, :base, 0.08,
            0.03, 20.0,
            8.0e-5, 5.0e-5,
            true, 15.0,                    # P-gp substrate
            false, 0.0,
            false, 0.0,
            false,
            false, false,
            :brain_ecf
        ),
        :morphine => CNSParams(
            "Morphine",
            285.3, 0.9, 8.0, :base, 0.65,
            0.50, 1.5,                     # Low Kp (hydrophilic)
            1.0e-5, 1.0e-5,                # Low permeability
            true, 50.0,                    # P-gp substrate (weak)
            false, 0.0,
            false, 0.0,
            false,
            false, false,
            :brain_ecf
        ),
        :gabapentin => CNSParams(
            "Gabapentin",
            171.2, -1.1, 3.7, :zwitterion, 0.97,
            0.85, 0.8,
            0.5e-5, 0.5e-5,                # Very low passive
            false, 0.0,
            false, 0.0,
            false, 0.0,
            false,
            true, false,                   # LAT1 substrate!
            :brain_ecf
        ),
        :levodopa => CNSParams(
            "Levodopa",
            197.2, -2.7, 2.3, :zwitterion, 0.99,
            0.90, 0.5,
            0.2e-5, 0.2e-5,
            false, 0.0,
            false, 0.0,
            false, 0.0,
            false,
            true, false,                   # LAT1 substrate
            :brain_icf                     # Intracellular (dopamine synthesis)
        ),
    )

    return get(presets, name, presets[:risperidone])
end

export drug_preset

end # module
