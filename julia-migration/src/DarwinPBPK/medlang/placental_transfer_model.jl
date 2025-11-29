# =============================================================================
# PLACENTAL TRANSFER MODEL - MedLang v1.0
# =============================================================================
# Darwin PBPK Platform - Publication-Ready Mechanistic Model
#
# Key Mechanisms:
# 1. Transplacental passive diffusion (MW, lipophilicity dependent)
# 2. Active efflux transporters (P-gp, BCRP) - fetal protection
# 3. Uptake transporters (OATPs, OATs, OCTs)
# 4. Gestational age-dependent changes
# 5. Fetal compartments (blood, amniotic fluid, tissues)
# 6. Maternal-fetal pH gradient (ion trapping)
#
# Literature Basis:
# - Staud et al. (2012) Drug Metab Rev - placental transporters
# - Myllynen et al. (2009) Int J Dev Biol - placental drug transport
# - Hutson et al. (2011) Clin Pharmacokinet - pregnancy PBPK
# - Codaccioni & Bhatt (2020) J Clin Pharmacol - fetal PBPK
# - Zhang & Bhatt (2023) AAPS J - pregnancy PBPK advances
# =============================================================================

module PlacentalTransferModel

using DifferentialEquations
using LinearAlgebra
using Statistics: mean

export PlacentalBarrier, PlacentalTransporters, FetalCompartments
export GestationalAge, MaternalPhysiology, FetalPhysiology
export DrugPlacentalProperties, PregnancyCondition
export calculate_placental_clearance, transplacental_flux
export fetal_maternal_ratio, ion_trapping_factor
export simulate_placental_transfer, gestational_scaling
export get_drug_preset, create_pregnancy_model
export validate_placental_model

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

"""
    GestationalAge

Gestational age parameters affecting placental transfer.
Trimester-specific changes in membrane thickness, surface area, blood flow.
"""
struct GestationalAge
    weeks::Float64               # Gestational weeks (0-40)
    trimester::Int               # 1, 2, or 3
    membrane_thickness_um::Float64  # µm (25 → 2-4)
    surface_area_m2::Float64     # m² (1 → 14)
    maternal_blood_flow::Float64 # mL/min (50 → 500)
    fetal_blood_flow::Float64    # mL/min
    villous_maturation::Float64  # 0-1 maturation factor
end

"""
    PlacentalTransporters

Transporter expression at the placental barrier.
Critical for fetal protection from xenobiotics.

Localization:
- MVM (microvillous membrane): maternal-facing
- BM (basal membrane): fetal-facing
"""
struct PlacentalTransporters
    # Efflux transporters (fetal protection)
    Pgp_MVM::Float64        # P-gp on MVM (maternal efflux)
    BCRP_MVM::Float64       # BCRP on MVM (maternal efflux)
    MRP2_MVM::Float64       # MRP2 on MVM

    # Efflux on basal membrane
    MRP1_BM::Float64        # MRP1 on BM (fetal efflux)
    MRP3_BM::Float64        # MRP3 on BM

    # Uptake transporters
    OATP2B1_MVM::Float64    # OATP2B1 on MVM
    OATP4A1_BM::Float64     # OATP4A1 on BM
    OAT4_MVM::Float64       # OAT4 on MVM
    OCT3_MVM::Float64       # OCT3 on MVM

    # Nucleoside transporters
    ENT1::Float64           # Equilibrative nucleoside
    ENT2::Float64
    CNT1::Float64           # Concentrative nucleoside

    # Amino acid transporters
    LAT1::Float64           # Large neutral amino acids
    SNAT::Float64           # Sodium-coupled neutral
end

"""
    PlacentalBarrier

Structural parameters of the placental barrier.
"""
struct PlacentalBarrier
    # Membrane layers
    syncytiotrophoblast_thickness::Float64  # µm
    cytotrophoblast_present::Bool           # Diminishes after T1
    basement_membrane::Float64              # µm
    fetal_endothelium::Float64              # µm

    # Permeability
    passive_permeability::Float64           # cm/s base
    pore_radius::Float64                    # nm

    # Binding
    placental_protein_binding::Float64      # Fraction bound
    placental_tissue_volume::Float64        # mL
end

"""
    FetalCompartments

Fetal physiological compartments for drug distribution.
"""
struct FetalCompartments
    # Volumes (mL)
    fetal_blood_volume::Float64
    amniotic_fluid_volume::Float64
    fetal_body_volume::Float64

    # Fetal organ volumes
    fetal_liver_volume::Float64
    fetal_brain_volume::Float64
    fetal_heart_volume::Float64

    # Fetal tissue binding
    fetal_plasma_protein::Float64   # Lower than maternal
    fetal_albumin::Float64          # g/dL
    fetal_AAG::Float64              # g/dL (α1-acid glycoprotein)

    # Fetal metabolism
    fetal_CYP3A7::Float64           # Fetal-specific CYP
    fetal_CYP1A1::Float64
    fetal_UGT::Float64              # Glucuronidation
end

"""
    MaternalPhysiology

Maternal physiological changes during pregnancy.
"""
struct MaternalPhysiology
    # Plasma changes
    plasma_volume_increase::Float64  # Fraction increase (0.4-0.5)
    albumin_decrease::Float64        # Fraction decrease
    AAG_change::Float64              # α1-acid glycoprotein change

    # Renal
    GFR_increase::Float64            # Fraction increase (0.5)
    renal_blood_flow_increase::Float64

    # Hepatic
    hepatic_blood_flow_increase::Float64
    CYP_changes::Dict{String, Float64}  # CYP activity changes

    # Cardiovascular
    cardiac_output_increase::Float64
    uterine_blood_flow::Float64      # mL/min
end

"""
    FetalPhysiology

Fetal physiological parameters.
"""
struct FetalPhysiology
    weight_kg::Float64              # Fetal weight
    blood_pH::Float64               # Slightly more acidic than maternal
    hematocrit::Float64             # Higher than maternal
    cardiac_output::Float64         # mL/min/kg

    # Amniotic fluid dynamics
    AF_production_rate::Float64     # mL/h
    AF_swallowing_rate::Float64     # mL/h
    AF_reabsorption_rate::Float64   # mL/h
end

"""
    DrugPlacentalProperties

Drug properties relevant to placental transfer.
"""
struct DrugPlacentalProperties
    name::String
    molecular_weight::Float64       # Da
    log_P::Float64                  # Lipophilicity
    pKa::Float64                    # Ionization
    drug_type::Symbol               # :acid, :base, :neutral, :zwitterion
    fu_maternal::Float64            # Unbound fraction maternal
    fu_fetal::Float64               # Unbound fraction fetal

    # Transporter affinities (Km in µM)
    Pgp_substrate::Bool
    Pgp_Km::Float64
    BCRP_substrate::Bool
    BCRP_Km::Float64
    OATP_substrate::Bool
    OATP_Km::Float64

    # Placental metabolism
    placental_clearance::Float64    # mL/min/g placenta
end

"""
    PregnancyCondition

Pathological conditions affecting placental transfer.
"""
struct PregnancyCondition
    condition::Symbol               # :normal, :preeclampsia, :GDM, :IUGR, :twins
    severity::Float64               # 0-1
    blood_flow_change::Float64      # Multiplier
    transporter_change::Float64     # Multiplier
    barrier_change::Float64         # Permeability change
end

# =============================================================================
# GESTATIONAL AGE FUNCTIONS
# =============================================================================

"""
    gestational_age(weeks)

Create gestational age parameters from weeks.
"""
function gestational_age(weeks::Float64)
    # Determine trimester
    trimester = weeks <= 12 ? 1 : (weeks <= 27 ? 2 : 3)

    # Membrane thickness decreases (25 µm → 2-4 µm)
    thickness = 25.0 * exp(-0.05 * weeks) + 2.0

    # Surface area increases exponentially
    # 1 m² at week 10 → 14 m² at term
    surface_area = 1.0 * exp(0.065 * weeks)
    surface_area = min(surface_area, 14.0)

    # Blood flow increases
    maternal_flow = 50.0 + (450.0 * (weeks / 40.0)^2)
    fetal_flow = 20.0 + (200.0 * (weeks / 40.0)^2)

    # Villous maturation
    maturation = min(1.0, weeks / 35.0)

    return GestationalAge(
        weeks, trimester, thickness, surface_area,
        maternal_flow, fetal_flow, maturation
    )
end

"""
    gestational_scaling(parameter, GA_weeks, reference_weeks=40)

Scale physiological parameters by gestational age.
"""
function gestational_scaling(parameter::Symbol, GA_weeks::Float64, reference::Float64=40.0)
    ratio = GA_weeks / reference

    scaling = Dict(
        :fetal_weight => 0.001 + 3.4 * ratio^3,  # kg, exponential growth
        :placental_weight => 0.05 + 0.5 * ratio^2,  # kg
        :amniotic_fluid => 50.0 + 900.0 * sin(π * ratio / 2)^2,  # mL, peaks ~34w
        :fetal_blood_volume => 10.0 + 300.0 * ratio^2,  # mL
        :cardiac_output_fetal => 50.0 + 400.0 * ratio^2,  # mL/min
        :GFR_fetal => 0.1 + 2.0 * ratio^2,  # mL/min
        :surface_area => 1.0 + 13.0 * ratio^2,  # m²
        :membrane_thickness => 25.0 * exp(-0.05 * GA_weeks) + 2.0,  # µm
    )

    return get(scaling, parameter, 1.0)
end

# =============================================================================
# TRANSPORTER FUNCTIONS
# =============================================================================

"""
    default_transporters(GA_weeks)

Create default transporter expression for gestational age.
Expression changes throughout pregnancy.
"""
function default_transporters(GA_weeks::Float64)
    # P-gp expression is relatively constant
    # BCRP increases throughout pregnancy
    # Many transporters have complex expression patterns

    # Normalized to term (1.0 = term expression)
    ratio = GA_weeks / 40.0

    return PlacentalTransporters(
        # Efflux MVM
        1.0,                          # P-gp constant
        0.5 + 0.5 * ratio,            # BCRP increases
        0.7 + 0.3 * ratio,            # MRP2
        # Efflux BM
        0.8,                          # MRP1
        0.6 + 0.4 * ratio,            # MRP3
        # Uptake MVM
        0.6 + 0.4 * ratio,            # OATP2B1
        0.7 + 0.3 * ratio,            # OATP4A1
        1.0,                          # OAT4
        0.8 + 0.2 * ratio,            # OCT3
        # Nucleoside
        1.0,                          # ENT1
        1.0,                          # ENT2
        0.5 + 0.5 * ratio,            # CNT1
        # Amino acid
        1.0,                          # LAT1
        0.8 + 0.2 * ratio             # SNAT
    )
end

"""
    transporter_efflux_ratio(drug, transporters)

Calculate net efflux ratio due to P-gp and BCRP.
Returns ratio > 1 if net efflux (fetal protection).
"""
function transporter_efflux_ratio(
    drug::DrugPlacentalProperties,
    transporters::PlacentalTransporters
)
    efflux_ratio = 1.0

    # P-gp efflux
    if drug.Pgp_substrate && drug.Pgp_Km > 0
        Pgp_activity = transporters.Pgp_MVM / (1.0 + drug.Pgp_Km / 10.0)
        efflux_ratio *= (1.0 + Pgp_activity)
    end

    # BCRP efflux
    if drug.BCRP_substrate && drug.BCRP_Km > 0
        BCRP_activity = transporters.BCRP_MVM / (1.0 + drug.BCRP_Km / 10.0)
        efflux_ratio *= (1.0 + BCRP_activity)
    end

    # OATP uptake (counters efflux)
    if drug.OATP_substrate && drug.OATP_Km > 0
        OATP_activity = transporters.OATP2B1_MVM / (1.0 + drug.OATP_Km / 50.0)
        efflux_ratio /= (1.0 + 0.5 * OATP_activity)
    end

    return efflux_ratio
end

# =============================================================================
# PASSIVE DIFFUSION CALCULATIONS
# =============================================================================

"""
    passive_permeability(drug, barrier)

Calculate passive permeability coefficient.
Based on MW and lipophilicity (Staud model).

P = P0 × f(MW) × f(LogP)
"""
function passive_permeability(
    drug::DrugPlacentalProperties,
    barrier::PlacentalBarrier
)
    # Base permeability (cm/s)
    P0 = barrier.passive_permeability

    # MW effect: decreases with increasing MW
    # Cutoff around 500-600 Da for passive diffusion
    MW_factor = exp(-drug.molecular_weight / 500.0)

    # Lipophilicity effect: optimal around Log P 1-3
    # Too hydrophilic: poor membrane crossing
    # Too lipophilic: membrane retention
    logP = drug.log_P
    if logP < 0
        lipophilicity_factor = exp(logP)
    elseif logP <= 3
        lipophilicity_factor = 1.0
    else
        lipophilicity_factor = exp(-(logP - 3.0) / 2.0)
    end

    # Membrane thickness effect
    thickness_factor = 10.0 / barrier.syncytiotrophoblast_thickness

    return P0 * MW_factor * lipophilicity_factor * thickness_factor
end

"""
    ion_trapping_factor(drug, maternal_pH, fetal_pH)

Calculate ion trapping effect due to maternal-fetal pH gradient.

Fetal blood is slightly more acidic (pH ~7.35 vs maternal ~7.40).
Weak bases accumulate in fetal circulation.
Weak acids accumulate in maternal circulation.
"""
function ion_trapping_factor(
    drug::DrugPlacentalProperties,
    maternal_pH::Float64 = 7.40,
    fetal_pH::Float64 = 7.35
)
    pKa = drug.pKa

    if drug.drug_type == :neutral || pKa <= 0
        return 1.0
    end

    if drug.drug_type == :base
        # Weak base: B + H⁺ ⇌ BH⁺
        # Henderson-Hasselbalch: ratio = 10^(pKa - pH)
        ionized_maternal = 10^(pKa - maternal_pH)
        ionized_fetal = 10^(pKa - fetal_pH)

        # Fetal/Maternal ratio of total drug
        # (1 + ionized_fetal) / (1 + ionized_maternal)
        trap_ratio = (1 + ionized_fetal) / (1 + ionized_maternal)

    elseif drug.drug_type == :acid
        # Weak acid: HA ⇌ A⁻ + H⁺
        ionized_maternal = 10^(maternal_pH - pKa)
        ionized_fetal = 10^(fetal_pH - pKa)

        # Maternal accumulation
        trap_ratio = (1 + ionized_fetal) / (1 + ionized_maternal)

    else  # zwitterion
        trap_ratio = 1.0
    end

    return trap_ratio
end

# =============================================================================
# PLACENTAL CLEARANCE CALCULATIONS
# =============================================================================

"""
    calculate_placental_clearance(drug, barrier, transporters, GA)

Calculate clearance across placenta (mL/min).

CL_placenta = P × SA × fu_m / Efflux_ratio

Where:
- P = permeability coefficient
- SA = surface area
- fu_m = maternal unbound fraction
- Efflux_ratio = transporter-mediated efflux
"""
function calculate_placental_clearance(
    drug::DrugPlacentalProperties,
    barrier::PlacentalBarrier,
    transporters::PlacentalTransporters,
    GA::GestationalAge
)
    # Passive permeability
    P = passive_permeability(drug, barrier)

    # Surface area (m² → cm²)
    SA = GA.surface_area_m2 * 10000.0

    # Unbound fraction
    fu = drug.fu_maternal

    # Transporter efflux
    efflux = transporter_efflux_ratio(drug, transporters)

    # Passive clearance (mL/min)
    # P (cm/s) × SA (cm²) × 60 (s/min) × 0.001 (mL/cm³)
    CL_passive = P * SA * 60.0 * fu / efflux

    # Add any placental metabolism
    CL_metabolism = drug.placental_clearance * barrier.placental_tissue_volume / 1000.0

    # Flow limitation
    CL_flow_limited = min(CL_passive, GA.maternal_blood_flow * fu)

    return (
        CL_total = CL_flow_limited,
        CL_passive = CL_passive,
        CL_metabolism = CL_metabolism,
        efflux_ratio = efflux,
        P = P
    )
end

"""
    transplacental_flux(C_maternal, C_fetal, drug, barrier, transporters, GA)

Calculate net flux across placenta (amount/min).
Positive = maternal → fetal.
"""
function transplacental_flux(
    C_maternal::Float64,        # µg/mL
    C_fetal::Float64,           # µg/mL
    drug::DrugPlacentalProperties,
    barrier::PlacentalBarrier,
    transporters::PlacentalTransporters,
    GA::GestationalAge
)
    # Get clearance
    CL = calculate_placental_clearance(drug, barrier, transporters, GA)

    # Ion trapping
    trap = ion_trapping_factor(drug)

    # Unbound concentrations
    Cu_maternal = C_maternal * drug.fu_maternal
    Cu_fetal = C_fetal * drug.fu_fetal

    # Net flux (maternal → fetal positive)
    # Accounts for bidirectional transfer
    flux_m_to_f = CL.CL_total * Cu_maternal
    flux_f_to_m = CL.CL_total * Cu_fetal / trap

    net_flux = flux_m_to_f - flux_f_to_m

    return (
        net_flux = net_flux,
        flux_maternal_to_fetal = flux_m_to_f,
        flux_fetal_to_maternal = flux_f_to_m,
        clearance = CL
    )
end

"""
    fetal_maternal_ratio(drug, barrier, transporters, GA; time_h)

Calculate steady-state fetal/maternal concentration ratio.

F:M = (fu_m / fu_f) × (1 / Efflux_ratio) × Ion_trap × Perfusion_factor
"""
function fetal_maternal_ratio(
    drug::DrugPlacentalProperties,
    barrier::PlacentalBarrier,
    transporters::PlacentalTransporters,
    GA::GestationalAge;
    include_ion_trap::Bool = true
)
    # Unbound fraction ratio
    fu_ratio = drug.fu_maternal / drug.fu_fetal

    # Transporter effect
    efflux = transporter_efflux_ratio(drug, transporters)

    # Ion trapping
    trap = include_ion_trap ? ion_trapping_factor(drug) : 1.0

    # Perfusion ratio (fetal/maternal blood flow)
    perfusion = GA.fetal_blood_flow / GA.maternal_blood_flow

    # F:M ratio
    FM_ratio = fu_ratio * (1.0 / efflux) * trap * (1.0 + perfusion) / 2.0

    return (
        FM_ratio = FM_ratio,
        fu_ratio = fu_ratio,
        efflux_ratio = efflux,
        ion_trap = trap,
        perfusion_ratio = perfusion
    )
end

# =============================================================================
# FETAL COMPARTMENT DYNAMICS
# =============================================================================

"""
    default_fetal_compartments(GA_weeks)

Create fetal compartment parameters for gestational age.
"""
function default_fetal_compartments(GA_weeks::Float64)
    ratio = GA_weeks / 40.0

    # Fetal weight (exponential growth)
    fetal_weight = 0.001 + 3.4 * ratio^3  # kg

    return FetalCompartments(
        # Volumes scale with weight
        fetal_weight * 80.0,          # Blood volume mL (~80 mL/kg)
        50.0 + 900.0 * sin(π * ratio / 2)^2,  # AF peaks ~34w
        fetal_weight * 800.0,         # Body volume mL
        # Organ volumes
        fetal_weight * 40.0,          # Liver
        fetal_weight * 100.0 * (1.0 - 0.5 * ratio),  # Brain (proportionally larger early)
        fetal_weight * 5.0,           # Heart
        # Binding
        0.3 + 0.4 * ratio,            # Plasma protein (lower than maternal)
        2.0 + 1.5 * ratio,            # Albumin g/dL
        0.02 + 0.03 * ratio,          # AAG g/dL (very low)
        # Metabolism
        1.0,                          # CYP3A7 (fetal-specific, high)
        0.2,                          # CYP1A1 (low)
        0.3 + 0.4 * ratio             # UGT (increases)
    )
end

"""
    amniotic_fluid_dynamics!(dA, A_AF, fetal, drug, C_fetal)

Calculate amniotic fluid drug dynamics.
Drug enters via fetal urine, exits via swallowing and membrane absorption.
"""
function amniotic_fluid_dynamics!(
    fetal::FetalPhysiology,
    drug_in_AF::Float64,        # Amount in AF
    C_fetal::Float64,           # Fetal blood concentration
    AF_volume::Float64          # mL
)
    # Drug enters AF via fetal urination (renal clearance)
    # Simplified: proportional to fetal blood concentration
    k_in = 0.01  # Rate constant for entry
    drug_entry = k_in * C_fetal * 2.0  # Simplified GFR estimate

    # Drug exits via swallowing
    C_AF = drug_in_AF / AF_volume
    k_swallow = fetal.AF_swallowing_rate / AF_volume
    drug_swallowed = k_swallow * drug_in_AF

    # Drug exits via membrane reabsorption
    k_reabsorb = fetal.AF_reabsorption_rate / AF_volume * 0.1  # 10% of water
    drug_reabsorbed = k_reabsorb * drug_in_AF

    # Net rate
    dA_AF = drug_entry - drug_swallowed - drug_reabsorbed

    return (
        dA_AF = dA_AF,
        C_AF = C_AF,
        entry_rate = drug_entry,
        exit_rate = drug_swallowed + drug_reabsorbed
    )
end

# =============================================================================
# DISEASE STATE EFFECTS
# =============================================================================

"""
    pregnancy_condition(condition, severity)

Create pregnancy condition with associated changes.
"""
function pregnancy_condition(condition::Symbol, severity::Float64 = 0.5)
    conditions = Dict(
        :normal => PregnancyCondition(:normal, 0.0, 1.0, 1.0, 1.0),

        :preeclampsia => PregnancyCondition(
            :preeclampsia, severity,
            0.5 + 0.5 * (1 - severity),   # Reduced blood flow
            1.2 + 0.3 * severity,          # Increased P-gp/BCRP
            0.8 - 0.3 * severity           # Barrier dysfunction
        ),

        :gestational_diabetes => PregnancyCondition(
            :gestational_diabetes, severity,
            1.0 + 0.2 * severity,          # Slightly increased flow
            0.9 - 0.2 * severity,          # Reduced transporter function
            1.1 + 0.2 * severity           # Increased permeability
        ),

        :IUGR => PregnancyCondition(
            :IUGR, severity,
            0.6 + 0.4 * (1 - severity),   # Reduced flow
            1.0,                           # Unchanged transporters
            1.0                            # Unchanged barrier
        ),

        :twins => PregnancyCondition(
            :twins, 1.0,
            1.4,                           # Increased flow (two placentas)
            1.0,                           # Normal transporters
            1.0                            # Normal barrier
        ),

        :chorioamnionitis => PregnancyCondition(
            :chorioamnionitis, severity,
            1.2 + 0.3 * severity,          # Increased flow (inflammation)
            0.7 - 0.2 * severity,          # Reduced transporter integrity
            1.5 + 0.5 * severity           # Increased permeability
        )
    )

    return get(conditions, condition, conditions[:normal])
end

"""
    apply_condition(clearance, condition)

Apply pregnancy condition effects to clearance.
"""
function apply_condition(CL::Float64, condition::PregnancyCondition)
    if condition.condition == :normal
        return CL
    end

    # Modify clearance based on condition effects
    CL_modified = CL * condition.blood_flow_change * condition.barrier_change /
                  condition.transporter_change

    return CL_modified
end

# =============================================================================
# ODE SYSTEM
# =============================================================================

"""
    placental_ode_system!(du, u, p, t)

Differential equations for maternal-fetal drug transfer.

Compartments:
1. Maternal plasma
2. Placental tissue
3. Fetal blood
4. Amniotic fluid
5. Fetal tissues

State variables [u]:
- u[1]: Drug in maternal plasma (µg)
- u[2]: Drug in placenta (µg)
- u[3]: Drug in fetal blood (µg)
- u[4]: Drug in amniotic fluid (µg)
- u[5]: Drug in fetal tissues (µg)
"""
function placental_ode_system!(du, u, p, t)
    # Unpack parameters
    drug = p.drug
    barrier = p.barrier
    transporters = p.transporters
    GA = p.GA
    fetal = p.fetal
    maternal_V = p.maternal_plasma_volume
    condition = p.condition
    k_elim_maternal = p.k_elim_maternal

    # State variables
    A_maternal = u[1]
    A_placenta = u[2]
    A_fetal_blood = u[3]
    A_AF = u[4]
    A_fetal_tissue = u[5]

    # Volumes
    V_placenta = barrier.placental_tissue_volume
    V_fetal_blood = fetal.fetal_blood_volume
    V_AF = fetal.amniotic_fluid_volume
    V_fetal_tissue = fetal.fetal_body_volume - V_fetal_blood

    # Concentrations
    C_maternal = A_maternal / maternal_V
    C_placenta = A_placenta / V_placenta
    C_fetal = A_fetal_blood / V_fetal_blood
    C_AF = A_AF / V_AF
    C_fetal_tissue = A_fetal_tissue / V_fetal_tissue

    # Calculate clearances
    CL_result = calculate_placental_clearance(drug, barrier, transporters, GA)
    CL_placenta = apply_condition(CL_result.CL_total, condition)

    # Ion trapping
    trap = ion_trapping_factor(drug)

    # Fluxes
    # Maternal → Placenta
    flux_m_to_p = CL_placenta * C_maternal * drug.fu_maternal
    flux_p_to_m = CL_placenta * C_placenta * 0.5  # Slower backflow

    # Placenta → Fetal blood
    flux_p_to_f = CL_placenta * C_placenta * 0.8  # Forward flux
    flux_f_to_p = CL_placenta * C_fetal * drug.fu_fetal / trap * 0.5

    # Fetal blood ↔ Fetal tissue
    k_tissue = 0.5  # Tissue distribution rate
    flux_fb_to_ft = k_tissue * A_fetal_blood
    flux_ft_to_fb = k_tissue * A_fetal_tissue / 2.0

    # Fetal blood ↔ Amniotic fluid
    k_AF_in = 0.02   # Entry via urine
    k_AF_out = 0.03  # Exit via swallowing
    flux_fb_to_AF = k_AF_in * A_fetal_blood
    flux_AF_to_fb = k_AF_out * A_AF

    # Maternal elimination
    elim_maternal = k_elim_maternal * A_maternal

    # ODEs
    # Maternal
    du[1] = -flux_m_to_p + flux_p_to_m - elim_maternal

    # Placenta
    du[2] = flux_m_to_p - flux_p_to_m - flux_p_to_f + flux_f_to_p

    # Fetal blood
    du[3] = flux_p_to_f - flux_f_to_p - flux_fb_to_ft + flux_ft_to_fb -
            flux_fb_to_AF + flux_AF_to_fb

    # Amniotic fluid
    du[4] = flux_fb_to_AF - flux_AF_to_fb

    # Fetal tissue
    du[5] = flux_fb_to_ft - flux_ft_to_fb

    return nothing
end

"""
    simulate_placental_transfer(drug, GA_weeks, maternal_dose; kwargs...)

Simulate drug transfer across placenta.
"""
function simulate_placental_transfer(
    drug::DrugPlacentalProperties,
    GA_weeks::Float64,
    maternal_dose::Float64;          # µg
    tspan::Tuple{Float64, Float64} = (0.0, 24.0),
    condition::Symbol = :normal,
    maternal_plasma_volume::Float64 = 3500.0,  # mL
    k_elim_maternal::Float64 = 0.1,  # 1/h
    saveat::Float64 = 0.1
)
    # Create physiological parameters
    GA = gestational_age(GA_weeks)
    barrier = PlacentalBarrier(
        GA.membrane_thickness_um,
        GA.trimester == 1,
        0.5,
        0.2,
        1e-5,  # Base permeability cm/s
        5.0,   # Pore radius nm
        0.5,   # Placental binding
        GA.surface_area_m2 * 50.0  # Placental volume mL
    )
    transporters = default_transporters(GA_weeks)
    fetal_comp = default_fetal_compartments(GA_weeks)
    fetal_phys = FetalPhysiology(
        gestational_scaling(:fetal_weight, GA_weeks),
        7.35,
        0.50,
        200.0,
        15.0, 20.0, 10.0
    )
    cond = pregnancy_condition(condition)

    # Parameters
    p = (
        drug = drug,
        barrier = barrier,
        transporters = transporters,
        GA = GA,
        fetal = fetal_comp,
        fetal_phys = fetal_phys,
        maternal_plasma_volume = maternal_plasma_volume,
        condition = cond,
        k_elim_maternal = k_elim_maternal
    )

    # Initial conditions (all drug in maternal plasma)
    u0 = [maternal_dose, 0.0, 0.0, 0.0, 0.0]

    # Solve ODE
    prob = ODEProblem(placental_ode_system!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=saveat)

    # Extract results
    times = sol.t
    C_maternal = [s[1] / maternal_plasma_volume for s in sol.u]
    C_placenta = [s[2] / barrier.placental_tissue_volume for s in sol.u]
    V_fetal_blood = fetal_comp.fetal_blood_volume
    C_fetal = [s[3] / V_fetal_blood for s in sol.u]
    C_AF = [s[4] / fetal_comp.amniotic_fluid_volume for s in sol.u]

    # Calculate F:M ratios
    FM_ratios = [cf / cm for (cf, cm) in zip(C_fetal, C_maternal) if cm > 0]

    # Fetal AUC
    AUC_fetal = 0.0
    AUC_maternal = 0.0
    for i in 2:length(times)
        dt = times[i] - times[i-1]
        AUC_fetal += 0.5 * (C_fetal[i] + C_fetal[i-1]) * dt
        AUC_maternal += 0.5 * (C_maternal[i] + C_maternal[i-1]) * dt
    end

    return (
        times = times,
        C_maternal = C_maternal,
        C_placenta = C_placenta,
        C_fetal = C_fetal,
        C_amniotic = C_AF,
        FM_ratio = length(FM_ratios) > 0 ? mean(FM_ratios[end-min(10,length(FM_ratios)-1):end]) : 0.0,
        AUC_fetal = AUC_fetal,
        AUC_maternal = AUC_maternal,
        GA = GA,
        solution = sol
    )
end

# =============================================================================
# DRUG PRESETS
# =============================================================================

"""
Drug presets for placental transfer modeling.
"""
function get_drug_preset(drug_name::Symbol)
    presets = Dict(
        # Well-studied drugs with known placental transfer
        :metformin => DrugPlacentalProperties(
            "Metformin",
            129.2,      # MW
            -1.4,       # Log P
            11.5,       # pKa
            :base,
            1.0,        # fu_maternal (no binding)
            1.0,        # fu_fetal
            false, 0.0, # P-gp
            false, 0.0, # BCRP
            true, 200.0, # OCT substrate
            0.0         # No placental metabolism
        ),

        :glyburide => DrugPlacentalProperties(
            "Glyburide",
            494.0,      # MW
            4.8,        # Log P
            5.3,        # pKa
            :acid,
            0.01,       # fu_maternal (highly bound)
            0.02,       # fu_fetal
            false, 0.0, # P-gp
            true, 0.5,  # Strong BCRP substrate!
            false, 0.0, # OATP
            0.0
        ),

        :dolutegravir => DrugPlacentalProperties(
            "Dolutegravir",
            419.4,      # MW
            2.2,        # Log P
            8.2,        # pKa (weakly basic)
            :base,
            0.01,       # fu_maternal
            0.02,       # fu_fetal
            true, 50.0, # P-gp substrate
            false, 0.0, # BCRP
            true, 10.0, # OATP1B1
            0.0
        ),

        :zidovudine => DrugPlacentalProperties(
            "Zidovudine (AZT)",
            267.2,      # MW
            0.05,       # Log P
            9.7,        # pKa
            :neutral,
            0.75,       # fu_maternal
            0.8,        # fu_fetal
            false, 0.0, # P-gp
            true, 200.0, # BCRP substrate
            false, 0.0,
            0.5         # Some placental metabolism
        ),

        :tenofovir => DrugPlacentalProperties(
            "Tenofovir",
            287.2,      # MW
            -1.6,       # Log P (hydrophilic)
            3.8,        # pKa
            :acid,
            0.93,       # fu_maternal
            0.95,       # fu_fetal
            false, 0.0, # Not P-gp
            true, 100.0, # BCRP substrate
            false, 0.0,
            0.0
        ),

        :caffeine => DrugPlacentalProperties(
            "Caffeine",
            194.2,      # MW
            -0.07,      # Log P
            10.4,       # pKa
            :base,
            0.65,       # fu_maternal
            0.75,       # fu_fetal
            false, 0.0,
            false, 0.0,
            false, 0.0,
            0.0
        ),

        :digoxin => DrugPlacentalProperties(
            "Digoxin",
            780.9,      # MW (large)
            1.3,        # Log P
            0.0,        # Neutral
            :neutral,
            0.75,       # fu_maternal
            0.8,        # fu_fetal
            true, 10.0, # Strong P-gp substrate!
            false, 0.0,
            true, 30.0, # OATP
            0.0
        ),

        :labetalol => DrugPlacentalProperties(
            "Labetalol",
            328.4,      # MW
            3.1,        # Log P
            7.4,        # pKa
            :base,
            0.5,        # fu_maternal
            0.55,       # fu_fetal
            true, 100.0, # Weak P-gp
            false, 0.0,
            false, 0.0,
            0.1
        ),

        :nifedipine => DrugPlacentalProperties(
            "Nifedipine",
            346.3,      # MW
            2.5,        # Log P
            0.0,        # Neutral
            :neutral,
            0.04,       # fu_maternal (highly bound)
            0.06,       # fu_fetal
            true, 50.0, # P-gp substrate
            false, 0.0,
            false, 0.0,
            0.2
        ),

        :dexamethasone => DrugPlacentalProperties(
            "Dexamethasone",
            392.5,      # MW
            1.8,        # Log P
            0.0,        # Neutral
            :neutral,
            0.23,       # fu_maternal
            0.3,        # fu_fetal
            true, 30.0, # P-gp substrate (but poor)
            false, 0.0,
            false, 0.0,
            0.0
        ),

        :betamethasone => DrugPlacentalProperties(
            "Betamethasone",
            392.5,      # MW
            1.9,        # Log P
            0.0,        # Neutral
            :neutral,
            0.35,       # fu_maternal
            0.4,        # fu_fetal
            true, 100.0, # Weaker P-gp than dexamethasone
            false, 0.0,
            false, 0.0,
            0.0
        )
    )

    if !haskey(presets, drug_name)
        available = join(keys(presets), ", ")
        error("Unknown drug: $drug_name. Available: $available")
    end

    return presets[drug_name]
end

# =============================================================================
# MODEL CREATION HELPER
# =============================================================================

"""
    create_pregnancy_model(GA_weeks; condition)

Create complete pregnancy model with all parameters.
"""
function create_pregnancy_model(
    GA_weeks::Float64;
    condition::Symbol = :normal,
    severity::Float64 = 0.5
)
    GA = gestational_age(GA_weeks)

    barrier = PlacentalBarrier(
        GA.membrane_thickness_um,
        GA.trimester == 1,
        0.5,
        0.2,
        1e-5,
        5.0,
        0.5,
        GA.surface_area_m2 * 50.0
    )

    transporters = default_transporters(GA_weeks)
    fetal_comp = default_fetal_compartments(GA_weeks)
    cond = pregnancy_condition(condition, severity)

    maternal = MaternalPhysiology(
        0.4,    # Plasma volume increase
        0.15,   # Albumin decrease
        0.0,    # AAG no change
        0.5,    # GFR increase
        0.3,    # Renal blood flow
        0.3,    # Hepatic blood flow
        Dict("CYP3A4" => 1.5, "CYP2D6" => 1.0, "CYP1A2" => 0.7),
        0.5,    # Cardiac output
        GA.maternal_blood_flow
    )

    return (
        gestational_age = GA,
        barrier = barrier,
        transporters = transporters,
        fetal_compartments = fetal_comp,
        condition = cond,
        maternal_physiology = maternal
    )
end

# =============================================================================
# VALIDATION
# =============================================================================

"""
    validate_placental_model()

Validate model against literature F:M ratios.
"""
function validate_placental_model()
    results = Dict{String, Any}()

    GA_weeks = 38.0  # Near term
    GA = gestational_age(GA_weeks)
    barrier = PlacentalBarrier(
        GA.membrane_thickness_um, false, 0.5, 0.2,
        1e-5, 5.0, 0.5, GA.surface_area_m2 * 50.0
    )
    transporters = default_transporters(GA_weeks)

    # Test 1: Glyburide - very low transfer due to BCRP
    # Literature: F:M ~0.1-0.3
    glyb = get_drug_preset(:glyburide)
    fm_glyb = fetal_maternal_ratio(glyb, barrier, transporters, GA)
    results["glyburide"] = (
        calculated_FM = fm_glyb.FM_ratio,
        literature_FM = 0.2,
        efflux_ratio = fm_glyb.efflux_ratio,
        note = "BCRP substrate - low transfer"
    )

    # Test 2: Digoxin - P-gp limits transfer
    # Literature: F:M ~0.5-0.8
    dig = get_drug_preset(:digoxin)
    fm_dig = fetal_maternal_ratio(dig, barrier, transporters, GA)
    results["digoxin"] = (
        calculated_FM = fm_dig.FM_ratio,
        literature_FM = 0.65,
        efflux_ratio = fm_dig.efflux_ratio,
        note = "P-gp substrate - moderate transfer"
    )

    # Test 3: Caffeine - crosses freely
    # Literature: F:M ~0.8-1.0
    caff = get_drug_preset(:caffeine)
    fm_caff = fetal_maternal_ratio(caff, barrier, transporters, GA)
    results["caffeine"] = (
        calculated_FM = fm_caff.FM_ratio,
        literature_FM = 0.9,
        efflux_ratio = fm_caff.efflux_ratio,
        note = "No efflux - free transfer"
    )

    # Test 4: Metformin - OCT-mediated uptake, basic
    # Literature: F:M ~1.1-1.5 (ion trapping)
    met = get_drug_preset(:metformin)
    fm_met = fetal_maternal_ratio(met, barrier, transporters, GA)
    results["metformin"] = (
        calculated_FM = fm_met.FM_ratio,
        literature_FM = 1.1,
        ion_trap = fm_met.ion_trap,
        note = "Basic drug - fetal accumulation"
    )

    return results
end

end # module PlacentalTransferModel
