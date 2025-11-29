# =============================================================================
# LYMPHATIC ABSORPTION MODEL - MedLang v1.0
# =============================================================================
# Darwin PBPK Platform - Publication-Ready Mechanistic Model
#
# Key Mechanisms:
# 1. Chylomicron formation and drug association
# 2. Log P and triglyceride solubility-dependent partitioning
# 3. Lipid formulation effects (LCT vs MCT)
# 4. Lacteal uptake and lymphatic transport
# 5. First-pass bypass quantification
# 6. Disease state adaptations
#
# Literature Basis:
# - Trevaskis et al. (2008) Nat Rev Drug Discov - lymphatic transport
# - Charman & Porter (1996) Adv Drug Deliv Rev - lipophilic drug transport
# - Yáñez et al. (2011) J Pharm Sci - intestinal lymphatic delivery
# - Caliph et al. (2000) J Pharm Sci - log P and triglyceride effects
# - O'Driscoll (2002) Eur J Pharm Sci - lipid-based delivery systems
# =============================================================================

module LymphaticAbsorptionModel

using DifferentialEquations
using LinearAlgebra
using Statistics: mean

export LymphaticSystem, ChylomicronDynamics, LipidFormulation
export DrugLymphPartitioning, LymphaticFlow, DiseaseState
export calculate_lymphatic_fraction, chylomicron_association
export lacteal_uptake_rate, thoracic_duct_flow
export first_pass_bypass_fraction, lymph_node_exposure
export simulate_lymphatic_absorption, get_drug_preset
export create_default_system, create_disease_state, disease_modifier
export validate_lymphatic_model, bioavailability_enhancement
export lymphatic_partitioning_curve, CM_drug_capacity
export chylomicron_formation_rate, lymph_transit_time

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

"""
    ChylomicronDynamics

Chylomicron formation and drug association kinetics.

Fields:
- diameter_nm: Mean chylomicron diameter (75-1200 nm)
- formation_rate: Rate of chylomicron formation (nmol/min)
- triglyceride_core: Triglyceride content (mg)
- apoB48_content: ApoB48 per particle
- Ka_drug: Drug association constant (L/mg TG)
- Kd_drug: Drug dissociation constant (1/min)
- surface_phospholipid: Phospholipid shell thickness
- cholesterol_ester: Cholesterol ester content
"""
struct ChylomicronDynamics
    diameter_nm::Float64          # Mean diameter
    formation_rate::Float64       # nmol/min during lipid digestion
    triglyceride_core::Float64    # mg TG per particle
    apoB48_content::Float64       # molecules per particle
    Ka_drug::Float64              # Association constant
    Kd_drug::Float64              # Dissociation constant
    surface_phospholipid::Float64 # nm thickness
    cholesterol_ester::Float64    # fraction of core
end

"""
    LipidFormulation

Lipid-based drug delivery formulation characteristics.

Types:
- Type I: Oils without surfactant
- Type II: SEDDS (self-emulsifying)
- Type III: SMEDDS (self-microemulsifying)
- Type IV: Surfactant/cosolvent only
"""
struct LipidFormulation
    formulation_type::Symbol      # :type_I, :type_II, :type_III, :type_IV
    triglyceride_type::Symbol     # :LCT, :MCT, :mixed
    triglyceride_dose::Float64    # mg
    surfactant_HLB::Float64       # Hydrophilic-lipophilic balance
    droplet_size_nm::Float64      # After dispersion
    digestion_rate::Float64       # Lipolysis rate constant
    lymph_targeting::Float64      # 0-1 fraction favoring lymph
end

"""
    DrugLymphPartitioning

Drug physicochemical properties affecting lymphatic partitioning.

Key Parameters:
- log_P: Octanol-water partition coefficient
- TG_solubility: Solubility in triglycerides (mg/mL)
- CM_binding: Chylomicron binding affinity
"""
struct DrugLymphPartitioning
    log_P::Float64               # Octanol-water
    log_D_74::Float64            # Distribution at pH 7.4
    TG_solubility::Float64       # mg/mL in triglyceride
    molecular_weight::Float64    # Da
    melting_point::Float64       # °C
    CM_binding_Kd::Float64       # nM
    protein_binding::Float64     # Fraction bound in lymph
    lipophilicity_index::Float64 # Combined metric
end

"""
    LymphaticFlow

Intestinal lymphatic system flow parameters.
"""
struct LymphaticFlow
    lacteal_flow_basal::Float64    # mL/h basal
    lacteal_flow_fed::Float64      # mL/h postprandial
    mesenteric_flow::Float64       # mL/h
    thoracic_duct_flow::Float64    # mL/h (0.5-4 L/day)
    cisterna_chyli_volume::Float64 # mL
    lymph_node_volume::Float64     # Total mesenteric nodes mL
    interstitial_pressure::Float64 # mmHg
    lymph_protein_conc::Float64    # g/dL
end

"""
    DiseaseState

Lymphatic system pathology affecting drug transport.
"""
struct DiseaseState
    condition::Symbol             # :normal, :lymphedema, :lipodystrophy, etc.
    severity::Float64            # 0-1 scale
    flow_reduction::Float64      # Fraction of normal flow
    chylomicron_impairment::Float64  # Fraction of normal CM formation
    lacteal_dysfunction::Float64 # Fraction of normal uptake
    node_involvement::Float64    # Fraction of nodes affected
end

"""
    LymphaticSystem

Complete lymphatic absorption system combining all components.
"""
struct LymphaticSystem
    chylomicron::ChylomicronDynamics
    formulation::LipidFormulation
    drug::DrugLymphPartitioning
    flow::LymphaticFlow
    disease::DiseaseState
    # Compartment volumes
    enterocyte_volume::Float64    # mL
    lacteal_volume::Float64       # mL
    mesenteric_lymph_volume::Float64
    thoracic_duct_volume::Float64
end

# =============================================================================
# CHYLOMICRON FORMATION AND DRUG ASSOCIATION
# =============================================================================

"""
    chylomicron_formation_rate(TG_absorbed, formulation)

Calculate rate of chylomicron formation based on triglyceride absorption.

The rate follows Michaelis-Menten kinetics:
    Rate = Vmax × [TG] / (Km + [TG])

Where Vmax depends on formulation type and enterocyte capacity.
"""
function chylomicron_formation_rate(
    TG_absorbed::Float64,        # mg absorbed
    formulation::LipidFormulation,
    time_postprandial::Float64   # hours
)
    # Vmax for CM formation (particles/min)
    Vmax_base = 1e9  # Peak capacity

    # Formulation-dependent factor
    form_factor = if formulation.triglyceride_type == :LCT
        1.0  # Long-chain → full chylomicron pathway
    elseif formulation.triglyceride_type == :MCT
        0.15  # Medium-chain → mostly portal
    else  # :mixed
        0.6
    end

    # Time-dependent secretion (peaks at 2-4h postprandial)
    time_factor = exp(-((time_postprandial - 3.0)^2) / 4.0)

    # Km for TG incorporation
    Km_TG = 50.0  # mg

    # Rate calculation
    rate = (Vmax_base * form_factor * time_factor * TG_absorbed) /
           (Km_TG + TG_absorbed)

    return rate
end

"""
    chylomicron_association(drug, CM, free_drug_conc)

Calculate drug association with chylomicrons using equilibrium binding.

Association depends on:
1. Log P of drug (Log P > 5 strongly favors association)
2. Triglyceride solubility
3. Chylomicron surface area and core volume
"""
function chylomicron_association(
    drug::DrugLymphPartitioning,
    CM::ChylomicronDynamics,
    free_drug_conc::Float64,     # µM
    TG_mass::Float64             # mg triglyceride available
)
    # Log P-dependent partition coefficient into CM
    # Sigmoidal relationship with inflection at Log P ≈ 5
    log_P_factor = 1.0 / (1.0 + exp(-(drug.log_P - 5.0) * 2.0))

    # Triglyceride solubility factor
    TG_factor = min(1.0, drug.TG_solubility / 10.0)  # Normalized to 10 mg/mL

    # Combined partition coefficient
    K_CM = CM.Ka_drug * log_P_factor * TG_factor

    # Amount associated (equilibrium)
    # Drug_CM = K_CM × [Drug_free] × [TG]
    drug_associated = K_CM * free_drug_conc * TG_mass

    # Fraction associated
    total_drug = free_drug_conc * 1000.0  # Convert to total
    fraction_associated = drug_associated / (drug_associated + free_drug_conc)

    return (
        drug_in_CM = drug_associated,
        fraction_CM = fraction_associated,
        K_partition = K_CM
    )
end

"""
    CM_drug_capacity(CM, drug)

Calculate maximum drug loading capacity of chylomicrons.
"""
function CM_drug_capacity(CM::ChylomicronDynamics, drug::DrugLymphPartitioning)
    # Core volume (nm³ → mL)
    core_radius = CM.diameter_nm / 2.0 - CM.surface_phospholipid
    core_volume_nm3 = (4/3) * π * core_radius^3
    core_volume_mL = core_volume_nm3 * 1e-21

    # TG mass in core
    TG_density = 0.93  # g/mL
    TG_mass_per_CM = core_volume_mL * TG_density * (1 - CM.cholesterol_ester)

    # Drug capacity based on TG solubility
    drug_capacity = TG_mass_per_CM * drug.TG_solubility * 1000  # ng per CM

    return drug_capacity
end

# =============================================================================
# LYMPHATIC PARTITIONING CALCULATIONS
# =============================================================================

"""
    calculate_lymphatic_fraction(drug, formulation, fed_state)

Calculate fraction of absorbed drug transported via lymphatics vs portal vein.

Based on Caliph et al. model:
    F_lymph = f(Log P, TG solubility, formulation, fed state)

Key thresholds:
- Log P < 4: Negligible lymphatic transport
- Log P 4-5: Transitional (5-20% lymphatic)
- Log P > 5: Significant lymphatic (20-80%+)
"""
function calculate_lymphatic_fraction(
    drug::DrugLymphPartitioning,
    formulation::LipidFormulation,
    fed_state::Bool = true
)
    # Base lymphatic fraction from Log P
    # Sigmoidal model from Trevaskis et al.
    base_fraction = 0.8 / (1.0 + exp(-(drug.log_P - 5.0) * 1.5))

    # Triglyceride solubility modifier
    TG_modifier = min(1.0, drug.TG_solubility / 50.0)  # Plateau at 50 mg/mL

    # Formulation factor
    form_factor = if formulation.triglyceride_type == :LCT
        1.0
    elseif formulation.triglyceride_type == :MCT
        0.1  # MCT largely bypasses lymph
    else
        0.5
    end

    # Fed state dramatically increases lymphatic transport
    fed_factor = fed_state ? 3.0 : 1.0  # 3-fold increase postprandially

    # Lipid dose effect (threshold ~1g for significant lymph transport)
    dose_factor = 1.0 - exp(-formulation.triglyceride_dose / 1000.0)

    # Combined fraction
    F_lymph = base_fraction * TG_modifier * form_factor *
              min(1.0, fed_factor * dose_factor)

    # Cap at realistic maximum (rarely >80%)
    F_lymph = min(0.85, F_lymph)

    return (
        F_lymph = F_lymph,
        F_portal = 1.0 - F_lymph,
        base_fraction = base_fraction,
        modifiers = (TG = TG_modifier, form = form_factor,
                    fed = fed_factor, dose = dose_factor)
    )
end

"""
    lymphatic_partitioning_curve(log_P_range, formulation)

Generate lymphatic fraction vs Log P curve for a given formulation.
Useful for drug design decisions.
"""
function lymphatic_partitioning_curve(
    log_P_range::AbstractVector,
    formulation::LipidFormulation;
    TG_solubility::Float64 = 50.0
)
    fractions = Float64[]

    for log_P in log_P_range
        # Create temporary drug struct
        drug = DrugLymphPartitioning(
            log_P, log_P - 0.5, TG_solubility,
            400.0, 150.0, 100.0, 0.9, log_P/7.0
        )
        result = calculate_lymphatic_fraction(drug, formulation, true)
        push!(fractions, result.F_lymph)
    end

    return fractions
end

# =============================================================================
# LACTEAL UPTAKE AND LYMPHATIC TRANSPORT
# =============================================================================

"""
    lacteal_uptake_rate(CM_conc, flow, drug_in_CM)

Calculate rate of drug-laden chylomicron uptake into lacteals.

Lacteals have unique button-like junctions that allow:
- Chylomicrons (75-1200 nm)
- Large macromolecules
- Immune cells

Uptake driven by:
1. Interstitial pressure gradient
2. Smooth muscle contraction (villus motility)
3. Chylomicron concentration
"""
function lacteal_uptake_rate(
    CM_concentration::Float64,    # particles/mL in interstitium
    flow::LymphaticFlow,
    drug_in_CM::Float64,         # µg drug in CM
    villus_motility::Float64 = 1.0  # 0-1 factor
)
    # Base uptake coefficient
    k_uptake = 0.5  # 1/min, rapid for CM-sized particles

    # Pressure-driven flow
    pressure_factor = flow.interstitial_pressure / 5.0  # Normalized to 5 mmHg

    # Motility enhances uptake
    motility_factor = 0.5 + 0.5 * villus_motility

    # Uptake rate (µg/min)
    uptake_rate = k_uptake * drug_in_CM * pressure_factor * motility_factor

    # Flow-limited maximum
    max_rate = flow.lacteal_flow_fed / 60.0 * drug_in_CM  # Convert to per min
    uptake_rate = min(uptake_rate, max_rate)

    return uptake_rate
end

"""
    thoracic_duct_flow(flow, fed_state, time_postprandial)

Calculate thoracic duct lymph flow rate.

Flow varies from 0.5-4 L/day:
- Fasted: ~0.5-1 L/day
- Fed: ~2-4 L/day (peaks 3-5h postprandial)
"""
function thoracic_duct_flow(
    flow::LymphaticFlow,
    fed_state::Bool,
    time_postprandial::Float64
)
    if !fed_state
        return flow.lacteal_flow_basal  # mL/h
    end

    # Postprandial flow profile
    # Peaks at 3-4 hours, returns to baseline by 8-10 hours
    if time_postprandial < 0.5
        # Initial lag
        factor = time_postprandial / 0.5
    elseif time_postprandial < 4.0
        # Rising phase
        factor = 1.0 + 2.0 * (1.0 - exp(-(time_postprandial - 0.5)))
    else
        # Declining phase
        factor = 1.0 + 2.0 * exp(-(time_postprandial - 4.0) / 3.0)
    end

    current_flow = flow.lacteal_flow_basal * factor

    # Cap at maximum
    current_flow = min(current_flow, flow.lacteal_flow_fed)

    return current_flow
end

"""
    lymph_transit_time(flow, volumes)

Calculate transit time through lymphatic system.
"""
function lymph_transit_time(flow::LymphaticFlow, fed_state::Bool)
    current_flow = fed_state ? flow.lacteal_flow_fed : flow.lacteal_flow_basal

    # Total volume: lacteals + mesenteric + cisterna + thoracic duct
    total_volume = 50.0 + 100.0 + flow.cisterna_chyli_volume + 50.0  # mL

    # Transit time
    transit_h = total_volume / current_flow

    return (
        transit_time_h = transit_h,
        transit_time_min = transit_h * 60,
        current_flow_mL_h = current_flow
    )
end

# =============================================================================
# FIRST-PASS BYPASS AND BIOAVAILABILITY
# =============================================================================

"""
    first_pass_bypass_fraction(drug, formulation, hepatic_extraction)

Calculate extent of first-pass bypass via lymphatic transport.

For drugs with high hepatic extraction ratio (Eh > 0.7),
lymphatic transport can dramatically improve oral bioavailability.

F_oral = F_abs × [(F_lymph × 1.0) + (F_portal × (1 - Eh))]
"""
function first_pass_bypass_fraction(
    drug::DrugLymphPartitioning,
    formulation::LipidFormulation,
    hepatic_extraction::Float64,  # 0-1
    F_abs::Float64 = 1.0          # Fraction absorbed
)
    # Calculate lymphatic fraction
    lymph_result = calculate_lymphatic_fraction(drug, formulation, true)
    F_lymph = lymph_result.F_lymph
    F_portal = lymph_result.F_portal

    # Bioavailability calculation
    # Lymphatic: bypasses liver completely
    # Portal: subject to first-pass
    F_oral_lymph = F_abs * F_lymph * 1.0
    F_oral_portal = F_abs * F_portal * (1.0 - hepatic_extraction)
    F_oral_total = F_oral_lymph + F_oral_portal

    # Improvement factor vs no lymphatic transport
    F_oral_no_lymph = F_abs * (1.0 - hepatic_extraction)
    improvement = F_oral_total / max(0.01, F_oral_no_lymph)

    return (
        F_oral_total = F_oral_total,
        F_oral_lymph_contribution = F_oral_lymph,
        F_oral_portal_contribution = F_oral_portal,
        F_oral_no_lymph = F_oral_no_lymph,
        improvement_factor = improvement,
        F_lymph = F_lymph
    )
end

"""
    bioavailability_enhancement(drug, formulations, Eh)

Compare bioavailability across different formulation strategies.
"""
function bioavailability_enhancement(
    drug::DrugLymphPartitioning,
    Eh::Float64
)
    formulations = [
        ("Aqueous solution", LipidFormulation(:type_IV, :MCT, 0.0, 15.0, 100.0, 0.0, 0.0)),
        ("MCT emulsion", LipidFormulation(:type_I, :MCT, 2000.0, 12.0, 200.0, 0.5, 0.2)),
        ("LCT emulsion", LipidFormulation(:type_I, :LCT, 2000.0, 12.0, 200.0, 0.3, 0.8)),
        ("SEDDS", LipidFormulation(:type_II, :LCT, 1000.0, 10.0, 100.0, 0.4, 0.7)),
        ("SMEDDS", LipidFormulation(:type_III, :mixed, 500.0, 13.0, 50.0, 0.6, 0.5)),
    ]

    results = []
    for (name, form) in formulations
        bypass = first_pass_bypass_fraction(drug, form, Eh)
        push!(results, (
            formulation = name,
            F_oral = bypass.F_oral_total,
            F_lymph = bypass.F_lymph,
            improvement = bypass.improvement_factor
        ))
    end

    return results
end

# =============================================================================
# LYMPH NODE EXPOSURE
# =============================================================================

"""
    lymph_node_exposure(drug_in_lymph, flow, node_params)

Calculate drug exposure in mesenteric lymph nodes.

Critical for:
- HIV antiretrovirals (lymph node reservoirs)
- Cancer immunotherapy
- Vaccine adjuvants
"""
function lymph_node_exposure(
    drug_in_lymph::Float64,      # µg in lymph fluid
    flow::LymphaticFlow,
    residence_time_factor::Float64 = 1.0
)
    # Lymph node residence time (longer than transit)
    base_residence = 2.0  # hours
    actual_residence = base_residence * residence_time_factor

    # Concentration in nodes
    # Nodes concentrate lymph-borne substances
    concentration_factor = 3.0  # Nodes accumulate drug

    # Node volume
    node_volume = flow.lymph_node_volume  # mL

    # Exposure calculation (µg·h/mL)
    C_node = drug_in_lymph * concentration_factor / node_volume
    AUC_node = C_node * actual_residence

    return (
        C_node = C_node,
        AUC_node = AUC_node,
        residence_time = actual_residence,
        concentration_factor = concentration_factor
    )
end

# =============================================================================
# DISEASE STATE ADAPTATIONS
# =============================================================================

"""
    disease_modifier(disease, parameter)

Apply disease state modifications to lymphatic parameters.
"""
function disease_modifier(disease::DiseaseState, parameter::Symbol)
    if disease.condition == :normal
        return 1.0
    end

    modifiers = Dict(
        :lymphedema => Dict(
            :flow => 0.3,
            :CM_formation => 1.0,
            :lacteal_uptake => 0.4,
            :transit_time => 3.0  # Prolonged
        ),
        :lipodystrophy => Dict(
            :flow => 0.8,
            :CM_formation => 0.3,  # Impaired fat absorption
            :lacteal_uptake => 0.7,
            :transit_time => 1.0
        ),
        :chylothorax => Dict(
            :flow => 0.5,
            :CM_formation => 1.0,
            :lacteal_uptake => 0.6,
            :transit_time => 2.0
        ),
        :intestinal_lymphangiectasia => Dict(
            :flow => 0.4,
            :CM_formation => 0.5,
            :lacteal_uptake => 0.3,
            :transit_time => 2.5
        ),
        :obesity => Dict(
            :flow => 0.7,
            :CM_formation => 1.5,  # Increased postprandial CM
            :lacteal_uptake => 0.8,
            :transit_time => 1.3
        )
    )

    if !haskey(modifiers, disease.condition)
        return 1.0
    end

    disease_mods = modifiers[disease.condition]
    base_mod = get(disease_mods, parameter, 1.0)

    # Scale by severity
    if parameter == :transit_time
        # Transit time increases with severity
        return 1.0 + (base_mod - 1.0) * disease.severity
    else
        # Other parameters decrease with severity
        return 1.0 - (1.0 - base_mod) * disease.severity
    end
end

"""
    create_disease_state(condition, severity)

Create a DiseaseState with appropriate parameter modifications.
"""
function create_disease_state(condition::Symbol, severity::Float64 = 0.5)
    defaults = Dict(
        :normal => (1.0, 1.0, 1.0, 0.0),
        :lymphedema => (0.3, 1.0, 0.4, 0.3),
        :lipodystrophy => (0.8, 0.3, 0.7, 0.1),
        :chylothorax => (0.5, 1.0, 0.6, 0.2),
        :intestinal_lymphangiectasia => (0.4, 0.5, 0.3, 0.4),
        :obesity => (0.7, 1.5, 0.8, 0.5)
    )

    params = get(defaults, condition, (1.0, 1.0, 1.0, 0.0))

    return DiseaseState(
        condition,
        severity,
        params[1],  # flow_reduction
        params[2],  # chylomicron_impairment
        params[3],  # lacteal_dysfunction
        params[4]   # node_involvement
    )
end

# =============================================================================
# ODE SYSTEM FOR LYMPHATIC ABSORPTION
# =============================================================================

"""
    lymphatic_ode_system!(du, u, p, t)

Differential equations for lymphatic drug absorption.

Compartments:
1. GI lumen (drug + lipid)
2. Enterocyte (lipid processing)
3. Interstitial fluid
4. Lacteals
5. Mesenteric lymph
6. Thoracic duct
7. Systemic circulation

State variables [u]:
- u[1]: Drug in GI lumen
- u[2]: Free drug in enterocyte
- u[3]: Drug in chylomicron (enterocyte)
- u[4]: TG in enterocyte (for CM formation)
- u[5]: Drug-CM in lacteals
- u[6]: Drug-CM in mesenteric lymph
- u[7]: Drug-CM in thoracic duct
- u[8]: Drug in systemic (from lymph)
- u[9]: Drug in systemic (from portal)
"""
function lymphatic_ode_system!(du, u, p, t)
    # Unpack parameters
    system = p.system
    ka = p.ka              # Absorption rate constant
    k_CM_form = p.k_CM_form  # CM formation rate
    k_CM_assoc = p.k_CM_assoc  # Drug-CM association
    k_lacteal = p.k_lacteal   # Lacteal uptake
    k_lymph_flow = p.k_lymph_flow  # Lymph flow rate
    k_portal = p.k_portal    # Portal absorption
    k_elim = p.k_elim       # Elimination rate
    F_lymph = p.F_lymph     # Fraction via lymph

    # Disease modifiers
    disease = system.disease
    flow_mod = disease_modifier(disease, :flow)
    CM_mod = disease_modifier(disease, :CM_formation)
    lacteal_mod = disease_modifier(disease, :lacteal_uptake)

    # State variables
    drug_lumen = u[1]
    drug_ent_free = u[2]
    drug_ent_CM = u[3]
    TG_enterocyte = u[4]
    drug_lacteal = u[5]
    drug_mesen = u[6]
    drug_thoracic = u[7]
    drug_sys_lymph = u[8]
    drug_sys_portal = u[9]

    # === GI Lumen ===
    # Absorption into enterocyte
    absorption = ka * drug_lumen
    du[1] = -absorption

    # === Enterocyte ===
    # Free drug: absorption - CM association - portal efflux
    CM_association = k_CM_assoc * drug_ent_free * TG_enterocyte * CM_mod
    portal_efflux = k_portal * drug_ent_free * (1.0 - F_lymph)
    du[2] = absorption - CM_association - portal_efflux

    # Drug in CM: association - lacteal secretion
    lacteal_secretion = k_lacteal * drug_ent_CM * lacteal_mod
    du[3] = CM_association - lacteal_secretion

    # TG pool: decays as CM formed
    du[4] = -k_CM_form * TG_enterocyte * CM_mod

    # === Lacteals ===
    # Drug-CM: secretion in - flow to mesenteric
    flow_to_mesen = k_lymph_flow * drug_lacteal * flow_mod
    du[5] = lacteal_secretion - flow_to_mesen

    # === Mesenteric Lymph ===
    # Flow from lacteals - flow to thoracic
    flow_to_thoracic = k_lymph_flow * drug_mesen * flow_mod
    du[6] = flow_to_mesen - flow_to_thoracic

    # === Thoracic Duct ===
    # Flow from mesenteric - emptying to systemic
    systemic_emptying = k_lymph_flow * drug_thoracic * flow_mod * 2.0  # Faster emptying
    du[7] = flow_to_thoracic - systemic_emptying

    # === Systemic Circulation ===
    # From lymph
    du[8] = systemic_emptying - k_elim * drug_sys_lymph
    # From portal (immediate for comparison)
    du[9] = portal_efflux - k_elim * drug_sys_portal

    return nothing
end

"""
    simulate_lymphatic_absorption(system, dose, tspan; kwargs...)

Simulate lymphatic drug absorption with full ODE model.
"""
function simulate_lymphatic_absorption(
    system::LymphaticSystem,
    dose::Float64,           # mg
    tspan::Tuple{Float64, Float64} = (0.0, 24.0);  # hours
    TG_dose::Float64 = 0.0,  # mg (0 = use formulation default)
    fed_state::Bool = true,
    saveat::Float64 = 0.1
)
    # Calculate lymphatic fraction
    lymph_result = calculate_lymphatic_fraction(system.drug, system.formulation, fed_state)
    F_lymph = lymph_result.F_lymph

    # TG dose
    TG = TG_dose > 0 ? TG_dose : system.formulation.triglyceride_dose

    # Rate constants
    ka = 0.5  # 1/h
    k_CM_form = 0.3
    k_CM_assoc = 0.1 / max(1.0, TG)  # Normalize by TG
    k_lacteal = 0.4
    k_lymph_flow = thoracic_duct_flow(system.flow, fed_state, 2.0) / 100.0  # Scaled
    k_portal = 0.5
    k_elim = 0.1  # 1/h

    # Parameters
    p = (
        system = system,
        ka = ka,
        k_CM_form = k_CM_form,
        k_CM_assoc = k_CM_assoc,
        k_lacteal = k_lacteal,
        k_lymph_flow = k_lymph_flow,
        k_portal = k_portal,
        k_elim = k_elim,
        F_lymph = F_lymph
    )

    # Initial conditions
    u0 = [
        dose,    # Drug in lumen
        0.0,     # Drug in enterocyte (free)
        0.0,     # Drug in CM (enterocyte)
        TG,      # TG available
        0.0,     # Drug in lacteals
        0.0,     # Drug in mesenteric lymph
        0.0,     # Drug in thoracic duct
        0.0,     # Drug systemic (lymph)
        0.0      # Drug systemic (portal)
    ]

    # Solve ODE
    prob = ODEProblem(lymphatic_ode_system!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=saveat)

    # Extract results
    times = sol.t
    C_systemic_lymph = [s[8] for s in sol.u]
    C_systemic_portal = [s[9] for s in sol.u]
    C_systemic_total = C_systemic_lymph .+ C_systemic_portal

    # Calculate PK parameters
    Cmax = maximum(C_systemic_total)
    tmax_idx = argmax(C_systemic_total)
    tmax = times[tmax_idx]

    # AUC (trapezoidal)
    AUC = 0.0
    for i in 2:length(times)
        dt = times[i] - times[i-1]
        AUC += 0.5 * (C_systemic_total[i] + C_systemic_total[i-1]) * dt
    end

    return (
        times = times,
        C_total = C_systemic_total,
        C_lymph = C_systemic_lymph,
        C_portal = C_systemic_portal,
        F_lymph = F_lymph,
        Cmax = Cmax,
        tmax = tmax,
        AUC = AUC,
        solution = sol
    )
end

# =============================================================================
# DRUG PRESETS
# =============================================================================

"""
Drug presets for lymphatic absorption modeling.

Drugs with significant lymphatic transport:
- Halofantrine (Log P ~8.5) - antimalarial
- DDT and lipophilic pesticides
- Vitamin A, D, E, K
- Cannabis/THC (Log P ~7)
- Penclomedine (Log P ~5)
"""
function get_drug_preset(drug_name::Symbol)
    presets = Dict(
        :halofantrine => (
            drug = DrugLymphPartitioning(
                8.5,    # log_P - highly lipophilic
                7.8,    # log_D
                85.0,   # TG_solubility mg/mL
                500.5,  # MW
                82.0,   # melting point
                15.0,   # CM_binding_Kd nM
                0.98,   # protein binding
                0.95    # lipophilicity index
            ),
            formulation = LipidFormulation(
                :type_I, :LCT, 2000.0, 12.0, 300.0, 0.3, 0.85
            ),
            notes = "Antimalarial with ~80% lymphatic transport"
        ),

        :vitamin_A => (
            drug = DrugLymphPartitioning(
                6.2,    # log_P
                5.8,    # log_D
                50.0,   # TG_solubility
                286.5,  # MW (retinol)
                63.0,   # melting point
                50.0,   # CM_binding_Kd
                0.95,   # protein binding
                0.80    # lipophilicity index
            ),
            formulation = LipidFormulation(
                :type_I, :LCT, 1000.0, 12.0, 250.0, 0.4, 0.75
            ),
            notes = "Fat-soluble vitamin, ~60% lymphatic"
        ),

        :THC => (
            drug = DrugLymphPartitioning(
                7.0,    # log_P
                6.5,    # log_D
                150.0,  # Very high TG solubility
                314.5,  # MW
                66.0,   # melting point
                25.0,   # CM_binding_Kd
                0.97,   # protein binding
                0.90    # lipophilicity index
            ),
            formulation = LipidFormulation(
                :type_II, :LCT, 500.0, 10.0, 150.0, 0.5, 0.70
            ),
            notes = "Cannabis, significant lymphatic (~50-70%)"
        ),

        :lopinavir => (
            drug = DrugLymphPartitioning(
                5.9,    # log_P
                4.7,    # log_D at pH 7.4
                25.0,   # TG_solubility
                628.8,  # MW
                92.0,   # melting point
                80.0,   # CM_binding_Kd
                0.99,   # protein binding
                0.65    # lipophilicity index
            ),
            formulation = LipidFormulation(
                :type_III, :mixed, 300.0, 13.0, 50.0, 0.6, 0.40
            ),
            notes = "HIV protease inhibitor, lymph node targeting"
        ),

        :testosterone_undecanoate => (
            drug = DrugLymphPartitioning(
                7.1,    # log_P
                6.8,    # log_D
                75.0,   # TG_solubility
                456.7,  # MW
                60.0,   # melting point
                30.0,   # CM_binding_Kd
                0.99,   # protein binding
                0.85    # lipophilicity index
            ),
            formulation = LipidFormulation(
                :type_I, :LCT, 1250.0, 11.0, 400.0, 0.25, 0.90
            ),
            notes = "Oral testosterone, ~90% lymphatic bypass"
        ),

        :probucol => (
            drug = DrugLymphPartitioning(
                10.9,   # log_P - extremely lipophilic
                10.5,   # log_D
                200.0,  # TG_solubility
                516.8,  # MW
                125.0,  # melting point
                5.0,    # CM_binding_Kd (very tight)
                0.99,   # protein binding
                0.98    # lipophilicity index
            ),
            formulation = LipidFormulation(
                :type_I, :LCT, 3000.0, 10.0, 500.0, 0.2, 0.95
            ),
            notes = "Lipid-lowering, essentially 100% lymphatic"
        ),

        # Comparison: portal-route drug
        :metformin => (
            drug = DrugLymphPartitioning(
                -1.4,   # log_P - hydrophilic
                -5.6,   # log_D
                0.01,   # Essentially insoluble in TG
                129.2,  # MW
                223.0,  # melting point
                10000.0, # CM_binding_Kd (no binding)
                0.0,    # No protein binding
                0.0     # No lipophilicity
            ),
            formulation = LipidFormulation(
                :type_IV, :MCT, 0.0, 15.0, 100.0, 0.0, 0.0
            ),
            notes = "Hydrophilic, 0% lymphatic (portal only)"
        )
    )

    if !haskey(presets, drug_name)
        available = join(keys(presets), ", ")
        error("Unknown drug: $drug_name. Available: $available")
    end

    return presets[drug_name]
end

# =============================================================================
# DEFAULT SYSTEM CREATION
# =============================================================================

"""
    create_default_system(drug_preset; disease, formulation_override)

Create a complete LymphaticSystem with defaults or drug preset.
"""
function create_default_system(;
    drug_preset::Union{Symbol, Nothing} = nothing,
    disease::Symbol = :normal,
    disease_severity::Float64 = 0.5
)
    # Default chylomicron dynamics
    CM = ChylomicronDynamics(
        200.0,   # diameter_nm
        1e8,     # formation_rate
        0.01,    # triglyceride_core mg
        1.0,     # apoB48_content
        0.5,     # Ka_drug
        0.1,     # Kd_drug
        10.0,    # surface_phospholipid nm
        0.1      # cholesterol_ester fraction
    )

    # Get drug and formulation from preset or use defaults
    if drug_preset !== nothing
        preset = get_drug_preset(drug_preset)
        drug = preset.drug
        form = preset.formulation
    else
        drug = DrugLymphPartitioning(
            5.0, 4.5, 30.0, 400.0, 100.0, 100.0, 0.95, 0.6
        )
        form = LipidFormulation(
            :type_II, :LCT, 1000.0, 12.0, 150.0, 0.4, 0.6
        )
    end

    # Lymphatic flow parameters
    flow = LymphaticFlow(
        20.0,    # lacteal_flow_basal mL/h
        120.0,   # lacteal_flow_fed mL/h
        80.0,    # mesenteric_flow mL/h
        100.0,   # thoracic_duct_flow mL/h
        10.0,    # cisterna_chyli_volume mL
        25.0,    # lymph_node_volume mL
        5.0,     # interstitial_pressure mmHg
        3.0      # lymph_protein_conc g/dL
    )

    # Disease state
    disease_state = create_disease_state(disease, disease_severity)

    return LymphaticSystem(
        CM,
        form,
        drug,
        flow,
        disease_state,
        5.0,     # enterocyte_volume mL
        2.0,     # lacteal_volume mL
        50.0,    # mesenteric_lymph_volume mL
        25.0     # thoracic_duct_volume mL
    )
end

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

"""
    validate_lymphatic_model()

Run validation tests against literature data.
"""
function validate_lymphatic_model()
    results = Dict{String, Any}()

    # Test 1: Halofantrine lymphatic fraction
    # Literature: ~80% lymphatic (Caliph et al., 2000)
    preset = get_drug_preset(:halofantrine)
    lymph_frac = calculate_lymphatic_fraction(preset.drug, preset.formulation, true)
    results["halofantrine_F_lymph"] = (
        calculated = lymph_frac.F_lymph,
        literature = 0.80,
        error_pct = abs(lymph_frac.F_lymph - 0.80) / 0.80 * 100
    )

    # Test 2: Metformin (control - no lymphatic)
    preset_met = get_drug_preset(:metformin)
    lymph_frac_met = calculate_lymphatic_fraction(preset_met.drug, preset_met.formulation, true)
    results["metformin_F_lymph"] = (
        calculated = lymph_frac_met.F_lymph,
        literature = 0.0,
        error_pct = lymph_frac_met.F_lymph * 100  # Should be near zero
    )

    # Test 3: First-pass bypass enhancement for high-Eh drug
    # Testosterone undecanoate: Eh ~0.95, F_lymph ~90%
    preset_TU = get_drug_preset(:testosterone_undecanoate)
    bypass = first_pass_bypass_fraction(preset_TU.drug, preset_TU.formulation, 0.95)
    results["testosterone_bypass"] = (
        F_oral_with_lymph = bypass.F_oral_total,
        F_oral_without = bypass.F_oral_no_lymph,
        improvement = bypass.improvement_factor
    )

    # Test 4: Log P curve shape
    log_P_range = collect(1.0:0.5:10.0)
    form = LipidFormulation(:type_I, :LCT, 2000.0, 12.0, 300.0, 0.3, 0.8)
    fractions = lymphatic_partitioning_curve(log_P_range, form)
    results["logP_curve"] = (
        log_P_values = log_P_range,
        F_lymph_values = fractions,
        inflection_point = log_P_range[findfirst(f -> f > 0.4, fractions)]
    )

    return results
end

end # module LymphaticAbsorptionModel
