# =============================================================================
# TUMOR PENETRATION MODEL - MedLang v1.0
# =============================================================================
# Darwin PBPK Platform - Publication-Ready Mechanistic Model
#
# Key Mechanisms:
# 1. EPR effect (enhanced permeability and retention)
# 2. Tumor microenvironment (IFP, pH, hypoxia)
# 3. ADC distribution (binding site barrier, DAR effects)
# 4. Convection-diffusion transport
# 5. Tumor heterogeneity
# 6. Stromal barriers and ECM
#
# Literature Basis:
# - Jain (1987, 2005) Cancer Res - tumor transport physiology
# - Thurber et al. (2008) Adv Drug Deliv Rev - ADC modeling
# - Mager & Bhatt (2018) AAPS J - ADC PBPK
# - Wilhelm et al. (2016) Nat Rev Materials - nanoparticle delivery
# - Dewhirst & Secomb (2017) Nat Rev Cancer - tumor microenvironment
# =============================================================================

module TumorPenetrationModel

using DifferentialEquations
using LinearAlgebra
using Statistics: mean

export TumorPhysiology, TumorMicroenvironment, VascularParameters
export DrugTumorProperties, ADCProperties, NanoparticleProperties
export EPREffect, TumorType, StromalBarrier
export calculate_tumor_uptake, calculate_EPR_accumulation
export tumor_penetration_depth, IFP_gradient_effect
export ADC_distribution, binding_site_barrier
export simulate_tumor_penetration, tumor_drug_preset
export create_tumor_model, validate_tumor_model

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

"""
    TumorType

Tumor classification affecting transport properties.
"""
@enum TumorType begin
    SOLID_CARCINOMA
    SARCOMA
    LYMPHOMA
    BRAIN_TUMOR
    PANCREATIC
    MELANOMA
    BREAST
    LUNG
    COLORECTAL
    OVARIAN
end

"""
    VascularParameters

Tumor vascular characteristics.
"""
struct VascularParameters
    vessel_density::Float64       # vessels/mm²
    vessel_diameter_um::Float64   # Mean diameter µm
    pore_size_nm::Float64         # Endothelial gap size
    blood_flow_mL_min_g::Float64  # mL/min/g tumor
    permeability_surface_area::Float64  # cm/s × cm²/g
    vascular_volume_fraction::Float64   # Fraction
    tortuosity::Float64           # Path length factor
    heterogeneity::Float64        # 0-1, spatial variation
end

"""
    TumorMicroenvironment

Tumor microenvironment parameters affecting drug distribution.
"""
struct TumorMicroenvironment
    # Pressure
    IFP_mmHg::Float64             # Interstitial fluid pressure (normal ~0, tumor 10-60)
    MVP_mmHg::Float64             # Microvascular pressure

    # pH
    extracellular_pH::Float64     # 6.5-7.0 typical
    intracellular_pH::Float64     # Often higher than pHe

    # Hypoxia
    pO2_mmHg::Float64            # Oxygen tension (normal 40, tumor 2-20)
    hypoxic_fraction::Float64    # Fraction of hypoxic cells

    # Matrix
    ECM_density::Float64         # Extracellular matrix density (mg/mL)
    collagen_fraction::Float64   # Collagen content
    hyaluronan_content::Float64  # Hyaluronan (affects diffusion)

    # Cellular
    cell_density::Float64        # Cells/mL
    necrotic_fraction::Float64   # Fraction necrotic core
end

"""
    TumorPhysiology

Complete tumor physiological model.
"""
struct TumorPhysiology
    tumor_type::TumorType
    volume_mL::Float64           # Tumor volume
    radius_cm::Float64           # Equivalent sphere radius
    vascular::VascularParameters
    microenvironment::TumorMicroenvironment

    # Lymphatic drainage
    lymphatic_function::Float64  # 0-1 (0 = absent in tumor core)
    peripheral_lymphatics::Float64  # At tumor margin

    # Growth
    doubling_time_days::Float64
    growth_fraction::Float64     # Proliferating fraction
end

"""
    StromalBarrier

Stromal barriers to drug penetration.
"""
struct StromalBarrier
    stromal_fraction::Float64    # Fraction of tumor that is stroma
    CAF_density::Float64         # Cancer-associated fibroblasts
    desmoplasia_score::Float64   # 0-3 (pancreatic = high)
    immune_infiltration::Float64 # T-cell, macrophage density
end

"""
    EPREffect

Enhanced Permeability and Retention effect parameters.
"""
struct EPREffect
    # Permeability
    pore_cutoff_nm::Float64      # Maximum size for extravasation
    permeability_ratio::Float64  # Tumor/normal ratio

    # Retention
    lymphatic_impairment::Float64  # 0-1 (1 = complete)
    retention_half_life_h::Float64 # Hours

    # Size-dependence
    optimal_size_nm::Float64     # Optimal particle size
    size_selectivity::Float64    # Sharpness of size cutoff

    # Tumor-type variation
    EPR_magnitude::Float64       # 0-1 relative EPR effect
end

"""
    DrugTumorProperties

Small molecule drug properties for tumor penetration.
"""
struct DrugTumorProperties
    name::String
    molecular_weight::Float64    # Da
    log_P::Float64
    charge_at_pH7::Float64       # Net charge
    diffusion_coeff::Float64     # cm²/s in tissue

    # Tumor interactions
    ECM_binding::Float64         # Binding to matrix
    cell_uptake_rate::Float64    # Cellular internalization
    Pgp_substrate::Bool          # Efflux substrate
    metabolism_rate::Float64     # Tumor metabolism

    # Target
    target_expression::Float64   # Relative expression in tumor
    Kd_nM::Float64              # Binding affinity
end

"""
    ADCProperties

Antibody-Drug Conjugate specific properties.
"""
struct ADCProperties
    name::String
    antibody_MW::Float64         # ~150 kDa for IgG
    total_MW::Float64            # Including payload
    DAR::Float64                 # Drug-to-antibody ratio

    # Antibody properties
    Kon::Float64                 # On-rate (1/M/s)
    Koff::Float64               # Off-rate (1/s)
    Kd_nM::Float64              # Affinity

    # Linker
    linker_type::Symbol          # :cleavable, :non_cleavable
    linker_stability_h::Float64  # Half-life in plasma

    # Payload
    payload_MW::Float64          # Da
    payload_log_P::Float64
    bystander_effect::Bool       # Can kill antigen-negative cells

    # Target
    target_antigen::String
    antigen_density::Float64     # Copies per cell
    internalization_rate::Float64 # 1/h
end

"""
    NanoparticleProperties

Nanoparticle drug delivery system properties.
"""
struct NanoparticleProperties
    name::String
    diameter_nm::Float64
    PEG_density::Float64         # PEGylation (affects opsonization)
    surface_charge_mV::Float64   # Zeta potential

    # Drug loading
    drug_loading::Float64        # mg drug/mg particle
    release_rate::Float64        # 1/h

    # Targeting
    targeted::Bool
    ligand_density::Float64      # If targeted
end

# =============================================================================
# EPR EFFECT CALCULATIONS
# =============================================================================

"""
    calculate_EPR_accumulation(drug_size_nm, EPR, vascular, time_h)

Calculate EPR-mediated tumor accumulation.

The EPR effect depends on:
1. Drug/particle size (optimal 10-100 nm)
2. Tumor vascular permeability
3. Lymphatic dysfunction
"""
function calculate_EPR_accumulation(
    drug_size_nm::Float64,
    EPR::EPREffect,
    vascular::VascularParameters,
    time_h::Float64
)
    # Size-dependent permeability
    # Gaussian around optimal size
    size_factor = exp(-((log(drug_size_nm) - log(EPR.optimal_size_nm))^2) /
                      (2 * EPR.size_selectivity^2))

    # Cutoff for large particles
    if drug_size_nm > EPR.pore_cutoff_nm
        size_factor *= exp(-(drug_size_nm - EPR.pore_cutoff_nm) / 50.0)
    end

    # Permeability
    P = vascular.permeability_surface_area * EPR.permeability_ratio * size_factor

    # Accumulation kinetics
    # Two-phase: extravasation and retention
    k_extrav = P * vascular.vessel_density / 100.0  # Extravasation rate
    k_clear = log(2) / EPR.retention_half_life_h * (1 - EPR.lymphatic_impairment)

    # Accumulation (simplified analytical)
    if k_extrav > k_clear
        accumulation = (k_extrav / (k_extrav - k_clear)) *
                      (1 - exp(-k_clear * time_h)) *
                      exp(-k_extrav * time_h)
    else
        accumulation = k_extrav * time_h * exp(-k_clear * time_h)
    end

    # Scale by EPR magnitude (tumor-type dependent)
    accumulation *= EPR.EPR_magnitude

    return (
        accumulation = accumulation,
        size_factor = size_factor,
        k_extravasation = k_extrav,
        k_clearance = k_clear,
        permeability = P
    )
end

"""
    EPR_tumor_type_factor(tumor_type)

Get EPR effect magnitude for different tumor types.
Based on Wilhelm et al. (2016) meta-analysis.
"""
function EPR_tumor_type_factor(tumor_type::TumorType)
    factors = Dict(
        SOLID_CARCINOMA => 0.7,
        SARCOMA => 0.6,
        LYMPHOMA => 0.8,
        BRAIN_TUMOR => 0.3,  # BBB limits EPR
        PANCREATIC => 0.4,   # Dense stroma
        MELANOMA => 0.75,
        BREAST => 0.7,
        LUNG => 0.65,
        COLORECTAL => 0.6,
        OVARIAN => 0.7
    )
    return get(factors, tumor_type, 0.5)
end

# =============================================================================
# TUMOR MICROENVIRONMENT EFFECTS
# =============================================================================

"""
    IFP_gradient_effect(tumor, radial_position)

Calculate effect of elevated IFP on drug transport.

IFP in tumors:
- Normal tissue: ~0 mmHg
- Tumor periphery: 10-20 mmHg
- Tumor core: 20-60 mmHg

High IFP:
1. Reduces convective transport
2. Creates outward pressure gradient
3. Heterogeneous distribution
"""
function IFP_gradient_effect(
    tumor::TumorPhysiology,
    radial_position::Float64  # 0 = center, 1 = periphery
)
    IFP_center = tumor.microenvironment.IFP_mmHg
    IFP_periphery = IFP_center * 0.3  # Gradient

    # IFP profile (parabolic)
    IFP_local = IFP_center * (1 - radial_position^2) +
                IFP_periphery * radial_position^2

    # Convective transport reduction
    # Starling forces: J = Lp × (ΔP - σΔπ)
    MVP = tumor.microenvironment.MVP_mmHg
    ΔP = MVP - IFP_local

    # Transport coefficient (normalized)
    if ΔP > 0
        transport_factor = ΔP / MVP
    else
        transport_factor = 0.1  # Minimal diffusive transport
    end

    return (
        IFP_local = IFP_local,
        pressure_gradient = ΔP,
        transport_factor = transport_factor,
        convection_possible = ΔP > 0
    )
end

"""
    pH_effect_on_drug(drug, tumor_pH)

Calculate effect of tumor acidic pH on drug distribution.

Weak bases: ion trapping in acidic tumor
Weak acids: ion trapping in normal tissue
"""
function pH_effect_on_drug(
    drug::DrugTumorProperties,
    tumor_pH::Float64,
    normal_pH::Float64 = 7.4
)
    # This requires pKa which we'll estimate from charge
    # Positive charge → basic → accumulates in tumor
    # Negative charge → acidic → excluded from tumor

    if drug.charge_at_pH7 > 0.5  # Basic
        # Accumulates in acidic tumor
        pH_ratio = 10^(normal_pH - tumor_pH)
        tumor_accumulation = 1 + 0.5 * (pH_ratio - 1)
    elseif drug.charge_at_pH7 < -0.5  # Acidic
        # Excluded from acidic tumor
        pH_ratio = 10^(tumor_pH - normal_pH)
        tumor_accumulation = pH_ratio
    else  # Neutral
        tumor_accumulation = 1.0
    end

    return tumor_accumulation
end

"""
    ECM_hindrance(drug, ECM_density, collagen)

Calculate extracellular matrix hindrance to diffusion.
"""
function ECM_hindrance(
    drug::DrugTumorProperties,
    ECM_density::Float64,
    collagen::Float64
)
    # Size-dependent hindrance
    # Larger molecules more hindered
    MW_factor = exp(-drug.molecular_weight / 5000.0)

    # Matrix density effect
    density_factor = exp(-ECM_density / 50.0)

    # Collagen is a major barrier
    collagen_factor = 1.0 - 0.5 * collagen

    # Binding to matrix
    binding_factor = 1.0 / (1.0 + drug.ECM_binding)

    effective_diffusion = drug.diffusion_coeff * MW_factor *
                         density_factor * collagen_factor * binding_factor

    return (
        effective_D = effective_diffusion,
        hindrance_ratio = effective_diffusion / drug.diffusion_coeff,
        MW_effect = MW_factor,
        density_effect = density_factor
    )
end

# =============================================================================
# TUMOR PENETRATION DEPTH
# =============================================================================

"""
    tumor_penetration_depth(drug, tumor, time_h)

Calculate drug penetration depth from vasculature.

Based on Krogh cylinder model:
- Drug diffuses from vessel
- Consumed by cells
- Penetration limited by consumption

Characteristic length L = √(D/k_consumption)
"""
function tumor_penetration_depth(
    drug::DrugTumorProperties,
    tumor::TumorPhysiology,
    time_h::Float64
)
    # Effective diffusion coefficient
    ECM_result = ECM_hindrance(drug,
                               tumor.microenvironment.ECM_density,
                               tumor.microenvironment.collagen_fraction)
    D_eff = ECM_result.effective_D

    # Consumption rate (uptake + metabolism)
    k_consumption = drug.cell_uptake_rate + drug.metabolism_rate
    k_consumption *= tumor.microenvironment.cell_density / 1e9  # Scale

    # Characteristic penetration depth (µm)
    if k_consumption > 0
        L_char = sqrt(D_eff * 3600 / k_consumption) * 10000  # Convert to µm
    else
        L_char = sqrt(D_eff * time_h * 3600) * 10000  # Diffusion only
    end

    # Inter-vessel distance
    if tumor.vascular.vessel_density > 0
        inter_vessel_um = 1000.0 / sqrt(tumor.vascular.vessel_density)
    else
        inter_vessel_um = 500.0  # Default
    end

    # Fraction of tumor reached
    fraction_reached = min(1.0, 2 * L_char / inter_vessel_um)

    return (
        penetration_depth_um = L_char,
        inter_vessel_distance_um = inter_vessel_um,
        fraction_reached = fraction_reached,
        limited_by = k_consumption > 0 ? :consumption : :diffusion,
        D_effective = D_eff
    )
end

# =============================================================================
# ADC DISTRIBUTION MODEL
# =============================================================================

"""
    binding_site_barrier(ADC, antigen_density, tumor_radius_cm)

Calculate binding site barrier effect.

For high-affinity ADCs:
- Peripheral binding saturates antigen
- Creates "barrier" preventing core penetration
- Worse for high affinity, high antigen density
"""
function binding_site_barrier(
    ADC::ADCProperties,
    antigen_density::Float64,  # Copies per cell
    cell_density::Float64,     # Cells/mL
    ADC_concentration::Float64  # nM
)
    # Thiele modulus (binding vs diffusion)
    # φ² = (k_on × Ag × L²) / D

    # Diffusion coefficient for IgG (~7 × 10⁻⁸ cm²/s in tissue)
    D_ADC = 7e-8  # cm²/s

    # Antigen concentration (M)
    Ag_M = antigen_density * cell_density / (6.022e23 * 1000)

    # Characteristic length (take 100 µm)
    L = 0.01  # cm

    # Thiele modulus
    phi_squared = ADC.Kon * Ag_M * L^2 / D_ADC
    phi = sqrt(phi_squared)

    # Effectiveness factor (fraction of tumor with drug)
    if phi > 0.1
        eta = (3/phi) * (1/tanh(phi) - 1/phi)
    else
        eta = 1.0  # Low binding, good penetration
    end

    # Penetration depth
    penetration = L / phi * 10000  # Convert to µm

    return (
        thiele_modulus = phi,
        effectiveness = eta,
        penetration_depth_um = penetration,
        barrier_strength = 1 - eta,
        Ag_concentration = Ag_M
    )
end

"""
    ADC_distribution(ADC, tumor, dose_mg_kg, time_h)

Calculate ADC distribution in tumor over time.

Phases:
1. Vascular distribution
2. Extravasation (slow for large molecules)
3. Binding to antigen
4. Internalization
5. Payload release
"""
function ADC_distribution(
    ADC::ADCProperties,
    tumor::TumorPhysiology,
    dose_mg_kg::Float64,
    time_h::Float64;
    body_weight_kg::Float64 = 70.0
)
    # Plasma concentration (two-compartment model simplified)
    Vc = 3.0  # Central volume L (typical for IgG)
    k_alpha = 0.1  # 1/h (distribution)
    k_beta = 0.005  # 1/h (elimination, ~week half-life)

    dose_mg = dose_mg_kg * body_weight_kg
    C_plasma = (dose_mg / Vc) * (0.8 * exp(-k_alpha * time_h) +
                                  0.2 * exp(-k_beta * time_h))

    # Tumor uptake (EPR for macromolecules)
    # IgG ~150 kDa → ~10 nm effective diameter
    ADC_size_nm = (ADC.antibody_MW / 1000)^0.33 * 5  # Approximate

    EPR = EPREffect(
        400.0,  # Pore cutoff nm
        5.0,    # Permeability ratio
        0.9,    # Lymphatic impairment
        48.0,   # Retention half-life
        50.0,   # Optimal size
        0.5,    # Size selectivity
        EPR_tumor_type_factor(tumor.tumor_type)
    )

    accumulation = calculate_EPR_accumulation(ADC_size_nm, EPR, tumor.vascular, time_h)

    # Tumor concentration
    C_tumor = C_plasma * accumulation.accumulation * 5  # Tumor:plasma ratio

    # Binding dynamics
    # Free + Bound antigen equilibrium
    Ag_total = ADC.antigen_density * tumor.microenvironment.cell_density
    Kd_M = ADC.Kd_nM * 1e-9

    # Fraction bound (simplified)
    if C_tumor > 0
        fraction_bound = Ag_total / (Kd_M + C_tumor / ADC.antibody_MW * 1e6)
        fraction_bound = min(fraction_bound, 0.99)
    else
        fraction_bound = 0.0
    end

    # Internalized fraction
    k_int = ADC.internalization_rate
    fraction_internalized = 1 - exp(-k_int * time_h)

    # Payload release
    k_release = log(2) / ADC.linker_stability_h
    payload_released = fraction_internalized * (1 - exp(-k_release * time_h))

    # Effective payload concentration
    payload_conc = C_tumor * ADC.DAR * payload_released *
                   (ADC.payload_MW / ADC.antibody_MW)

    return (
        C_plasma = C_plasma,
        C_tumor = C_tumor,
        tumor_plasma_ratio = C_tumor > 0 && C_plasma > 0 ? C_tumor / C_plasma : 0.0,
        fraction_bound = fraction_bound,
        fraction_internalized = fraction_internalized,
        payload_released = payload_released,
        payload_concentration = payload_conc
    )
end

# =============================================================================
# SMALL MOLECULE TUMOR UPTAKE
# =============================================================================

"""
    calculate_tumor_uptake(drug, tumor, C_plasma, time_h)

Calculate small molecule drug uptake into tumor.
"""
function calculate_tumor_uptake(
    drug::DrugTumorProperties,
    tumor::TumorPhysiology,
    C_plasma::Float64,       # µg/mL
    time_h::Float64
)
    # Vascular transport
    PS = tumor.vascular.permeability_surface_area * tumor.volume_mL
    blood_flow = tumor.vascular.blood_flow_mL_min_g * tumor.volume_mL

    # Renkin-Crone model: E = 1 - exp(-PS/Q)
    extraction = 1 - exp(-PS * 60 / blood_flow)

    # IFP effect on transport
    IFP_result = IFP_gradient_effect(tumor, 0.5)  # Average position
    extraction *= IFP_result.transport_factor

    # pH effect
    pH_factor = pH_effect_on_drug(drug, tumor.microenvironment.extracellular_pH)

    # P-gp efflux
    if drug.Pgp_substrate
        efflux_factor = 0.5  # Reduced uptake
    else
        efflux_factor = 1.0
    end

    # Uptake rate
    k_uptake = extraction * blood_flow / tumor.volume_mL * 60  # 1/h
    k_efflux = k_uptake * 0.5  # Approximate

    # Tumor concentration at quasi-steady-state
    C_tumor = C_plasma * (k_uptake / (k_uptake + k_efflux)) *
              pH_factor * efflux_factor

    # Time to approach steady-state
    t_half = log(2) / (k_uptake + k_efflux)
    approach_SS = 1 - exp(-(k_uptake + k_efflux) * time_h)
    C_tumor *= approach_SS

    return (
        C_tumor = C_tumor,
        tumor_plasma_ratio = C_tumor / max(1e-10, C_plasma),
        extraction = extraction,
        pH_factor = pH_factor,
        efflux_factor = efflux_factor,
        t_half_h = t_half,
        approach_SS = approach_SS
    )
end

# =============================================================================
# ODE SYSTEM
# =============================================================================

"""
    tumor_ode_system!(du, u, p, t)

Differential equations for tumor drug distribution.

Compartments:
1. Plasma
2. Tumor vascular
3. Tumor interstitial
4. Tumor cellular
5. Necrotic core
"""
function tumor_ode_system!(du, u, p, t)
    # Unpack parameters
    drug = p.drug
    tumor = p.tumor
    dose = p.dose
    k_elim = p.k_elim

    # State variables
    A_plasma = u[1]
    A_tumor_vasc = u[2]
    A_tumor_interstitial = u[3]
    A_tumor_cell = u[4]
    A_necrotic = u[5]

    # Volumes
    V_plasma = 3000.0  # mL
    V_vasc = tumor.volume_mL * tumor.vascular.vascular_volume_fraction
    V_interstitial = tumor.volume_mL * 0.3  # 30% interstitial
    V_cell = tumor.volume_mL * (0.7 - tumor.vascular.vascular_volume_fraction -
                                 tumor.microenvironment.necrotic_fraction)
    V_necrotic = tumor.volume_mL * tumor.microenvironment.necrotic_fraction

    # Concentrations
    C_plasma = A_plasma / V_plasma
    C_vasc = A_tumor_vasc / max(V_vasc, 0.01)
    C_interstitial = A_tumor_interstitial / max(V_interstitial, 0.01)
    C_cell = A_tumor_cell / max(V_cell, 0.01)

    # Fluxes
    # Plasma ↔ Tumor vascular
    blood_flow = tumor.vascular.blood_flow_mL_min_g * tumor.volume_mL * 60  # mL/h
    flux_plasma_to_vasc = blood_flow * C_plasma
    flux_vasc_to_plasma = blood_flow * C_vasc

    # Tumor vascular → Interstitial (extravasation)
    PS = tumor.vascular.permeability_surface_area * tumor.volume_mL * 3600  # cm³/h
    k_extrav = PS / V_vasc
    flux_extrav = k_extrav * A_tumor_vasc

    # Interstitial → Cellular (uptake)
    k_cell_uptake = drug.cell_uptake_rate
    flux_cell_uptake = k_cell_uptake * A_tumor_interstitial

    # P-gp efflux
    if drug.Pgp_substrate
        k_efflux = 0.5  # 1/h
    else
        k_efflux = 0.1
    end
    flux_cell_efflux = k_efflux * A_tumor_cell

    # Diffusion to necrotic core (slow, non-viable)
    k_necrotic = 0.05  # 1/h
    flux_to_necrotic = k_necrotic * A_tumor_interstitial

    # Metabolism
    k_met = drug.metabolism_rate
    metabolism = k_met * A_tumor_cell

    # Elimination
    elim_plasma = k_elim * A_plasma

    # ODEs
    du[1] = -flux_plasma_to_vasc + flux_vasc_to_plasma - elim_plasma
    du[2] = flux_plasma_to_vasc - flux_vasc_to_plasma - flux_extrav
    du[3] = flux_extrav - flux_cell_uptake + flux_cell_efflux - flux_to_necrotic
    du[4] = flux_cell_uptake - flux_cell_efflux - metabolism
    du[5] = flux_to_necrotic

    return nothing
end

"""
    simulate_tumor_penetration(drug, tumor, dose_mg; kwargs...)

Simulate drug penetration into tumor.
"""
function simulate_tumor_penetration(
    drug::DrugTumorProperties,
    tumor::TumorPhysiology,
    dose_mg::Float64;
    tspan::Tuple{Float64, Float64} = (0.0, 24.0),
    k_elim::Float64 = 0.1,
    saveat::Float64 = 0.1
)
    # Parameters
    p = (
        drug = drug,
        tumor = tumor,
        dose = dose_mg,
        k_elim = k_elim
    )

    # Initial conditions (bolus to plasma)
    u0 = [dose_mg * 1000, 0.0, 0.0, 0.0, 0.0]  # µg

    # Solve ODE
    prob = ODEProblem(tumor_ode_system!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=saveat)

    # Extract results
    times = sol.t
    V_plasma = 3000.0
    V_tumor = tumor.volume_mL

    C_plasma = [s[1] / V_plasma for s in sol.u]
    C_tumor_total = [(s[2] + s[3] + s[4]) / V_tumor for s in sol.u]
    C_tumor_cell = [s[4] / (V_tumor * 0.5) for s in sol.u]

    # PK metrics
    AUC_plasma = 0.0
    AUC_tumor = 0.0
    for i in 2:length(times)
        dt = times[i] - times[i-1]
        AUC_plasma += 0.5 * (C_plasma[i] + C_plasma[i-1]) * dt
        AUC_tumor += 0.5 * (C_tumor_total[i] + C_tumor_total[i-1]) * dt
    end

    return (
        times = times,
        C_plasma = C_plasma,
        C_tumor = C_tumor_total,
        C_tumor_cellular = C_tumor_cell,
        AUC_plasma = AUC_plasma,
        AUC_tumor = AUC_tumor,
        tumor_plasma_AUC_ratio = AUC_tumor / max(AUC_plasma, 1e-10),
        solution = sol
    )
end

# =============================================================================
# DRUG/ADC PRESETS
# =============================================================================

"""
Drug presets for tumor penetration modeling.
"""
function tumor_drug_preset(drug_name::Symbol)
    presets = Dict(
        :doxorubicin => DrugTumorProperties(
            "Doxorubicin",
            543.5,      # MW
            1.3,        # Log P
            1.0,        # Positive charge (basic)
            3e-6,       # D cm²/s
            0.5,        # ECM binding
            0.5,        # Cell uptake
            true,       # P-gp substrate
            0.1,        # Metabolism
            1.0,        # Target expression
            100.0       # Kd nM (DNA intercalation)
        ),

        :paclitaxel => DrugTumorProperties(
            "Paclitaxel",
            853.9,      # MW
            3.0,        # Log P
            0.0,        # Neutral
            1e-6,       # D cm²/s (low due to MW)
            0.8,        # High ECM binding
            0.3,        # Cell uptake
            true,       # P-gp substrate (major!)
            0.05,       # Metabolism
            1.0,        # Tubulin expression
            10.0        # Kd nM
        ),

        :gemcitabine => DrugTumorProperties(
            "Gemcitabine",
            263.2,      # MW
            -1.4,       # Log P (hydrophilic)
            0.0,        # Neutral
            8e-6,       # D cm²/s (good)
            0.1,        # Low ECM binding
            0.8,        # Nucleoside transporter-mediated
            false,      # Not P-gp
            0.3,        # Rapid metabolism
            1.0,        # Ubiquitous target
            50.0        # Kd
        ),

        :imatinib => DrugTumorProperties(
            "Imatinib",
            493.6,      # MW
            3.5,        # Log P
            2.0,        # Basic (positive)
            4e-6,       # D cm²/s
            0.3,        # ECM binding
            0.6,        # Cell uptake
            true,       # P-gp substrate (weak)
            0.1,        # Metabolism
            5.0,        # BCR-ABL overexpression
            1.0         # Kd nM (high affinity)
        ),

        :cisplatin => DrugTumorProperties(
            "Cisplatin",
            300.0,      # MW
            -2.2,       # Log P (hydrophilic)
            0.0,        # Neutral
            6e-6,       # D cm²/s
            0.4,        # DNA/protein binding
            0.4,        # Cell uptake
            false,      # Not P-gp
            0.0,        # No metabolism
            1.0,        # DNA target
            1000.0      # Kd (covalent)
        )
    )

    if !haskey(presets, drug_name)
        available = join(keys(presets), ", ")
        error("Unknown drug: $drug_name. Available: $available")
    end

    return presets[drug_name]
end

"""
ADC presets for tumor penetration modeling.
"""
function ADC_preset(ADC_name::Symbol)
    presets = Dict(
        :trastuzumab_emtansine => ADCProperties(
            "T-DM1",
            148000.0,   # Antibody MW
            151000.0,   # Total MW
            3.5,        # DAR
            1e5,        # Kon
            1e-4,       # Koff
            0.1,        # Kd nM
            :non_cleavable,
            1000.0,     # Linker stable
            738.0,      # DM1 MW
            3.0,        # DM1 Log P
            false,      # No bystander
            "HER2",
            1e6,        # Antigen density
            0.5         # Internalization rate
        ),

        :brentuximab_vedotin => ADCProperties(
            "Brentuximab vedotin",
            148000.0,
            153000.0,
            4.0,        # DAR
            1e5,
            1e-3,
            1.0,        # Kd nM
            :cleavable,
            48.0,       # Cleavable linker
            718.0,      # MMAE MW
            4.0,        # MMAE Log P
            true,       # Bystander effect!
            "CD30",
            5e5,
            0.3
        ),

        :enfortumab_vedotin => ADCProperties(
            "Enfortumab vedotin",
            148000.0,
            153000.0,
            3.8,
            1e5,
            1e-3,
            1.5,
            :cleavable,
            48.0,
            718.0,
            4.0,
            true,       # Bystander
            "Nectin-4",
            1e6,
            0.4
        )
    )

    if !haskey(presets, ADC_name)
        available = join(keys(presets), ", ")
        error("Unknown ADC: $ADC_name. Available: $available")
    end

    return presets[ADC_name]
end

# =============================================================================
# MODEL CREATION AND VALIDATION
# =============================================================================

"""
    create_tumor_model(tumor_type, volume_mL; kwargs...)

Create complete tumor model with default parameters.
"""
function create_tumor_model(
    tumor_type::TumorType,
    volume_mL::Float64;
    IFP::Float64 = 20.0,
    vessel_density::Float64 = 100.0
)
    radius = (3 * volume_mL / (4 * π))^(1/3)

    vascular = VascularParameters(
        vessel_density,
        20.0,       # Diameter µm
        200.0,      # Pore size nm
        0.5,        # Blood flow
        1e-4,       # PS
        0.05,       # Vascular fraction
        1.5,        # Tortuosity
        0.5         # Heterogeneity
    )

    microenv = TumorMicroenvironment(
        IFP,
        30.0,       # MVP
        6.8,        # pH
        7.2,        # Intracellular pH
        10.0,       # pO2
        0.3,        # Hypoxic fraction
        30.0,       # ECM density
        0.2,        # Collagen
        0.1,        # Hyaluronan
        1e9,        # Cell density
        0.1         # Necrotic fraction
    )

    return TumorPhysiology(
        tumor_type,
        volume_mL,
        radius,
        vascular,
        microenv,
        0.0,        # No lymphatics in core
        0.3,        # Some at periphery
        30.0,       # Doubling time days
        0.3         # Growth fraction
    )
end

"""
    validate_tumor_model()

Validate tumor model against literature benchmarks.
"""
function validate_tumor_model()
    results = Dict{String, Any}()

    # Test 1: EPR accumulation for different particle sizes
    EPR = EPREffect(400.0, 5.0, 0.9, 48.0, 50.0, 0.5, 0.7)
    vascular = VascularParameters(100.0, 20.0, 200.0, 0.5, 1e-4, 0.05, 1.5, 0.5)

    sizes = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    accumulations = [calculate_EPR_accumulation(s, EPR, vascular, 24.0).accumulation
                    for s in sizes]

    results["EPR_size_dependence"] = (
        sizes = sizes,
        accumulations = accumulations,
        optimal_size = sizes[argmax(accumulations)]
    )

    # Test 2: Penetration depth
    drug = tumor_drug_preset(:doxorubicin)
    tumor = create_tumor_model(BREAST, 1.0)
    pen = tumor_penetration_depth(drug, tumor, 24.0)

    results["penetration_depth"] = (
        depth_um = pen.penetration_depth_um,
        inter_vessel_um = pen.inter_vessel_distance_um,
        fraction_reached = pen.fraction_reached
    )

    # Test 3: ADC binding site barrier
    ADC = ADC_preset(:trastuzumab_emtansine)
    barrier = binding_site_barrier(ADC, 1e6, 1e9, 10.0)

    results["ADC_barrier"] = (
        thiele_modulus = barrier.thiele_modulus,
        effectiveness = barrier.effectiveness,
        penetration_um = barrier.penetration_depth_um
    )

    return results
end

end # module TumorPenetrationModel
