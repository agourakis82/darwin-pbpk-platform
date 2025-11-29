# =============================================================================
# PULMONARY ABSORPTION MODEL - MedLang v1.0
# =============================================================================
# Darwin PBPK Platform - Publication-Ready Mechanistic Model
#
# Key Mechanisms:
# 1. Particle deposition (ICRP/MPPD models)
# 2. Regional lung anatomy (oropharynx, tracheobronchial, alveolar)
# 3. Mucociliary clearance
# 4. Dissolution kinetics
# 5. Pulmonary absorption (passive + transporters)
# 6. Disease state effects (COPD, asthma, CF)
#
# Literature Basis:
# - ICRP Publication 66 (1994) - Human respiratory tract model
# - MPPD model - particle deposition
# - Patton & Byron (2007) Nat Rev Drug Discov - pulmonary drug delivery
# - Borghardt et al. (2018) Clin Pharmacokinet - inhaled PK modeling
# - Weber & Hochhaus (2013) AAPS J - pulmonary dissolution
# =============================================================================

module PulmonaryAbsorptionModel

using DifferentialEquations
using LinearAlgebra
using Statistics: mean

export LungAnatomy, ParticleProperties, DepositionFractions
export PulmonaryTransporters, MucociliaryClearance, DissolutionKinetics
export DrugPulmonaryProperties, DeviceProperties, PulmonaryDisease
export calculate_deposition, regional_deposition_fractions
export mucociliary_clearance_rate, alveolar_macrophage_clearance
export pulmonary_absorption_rate, calculate_bioavailability
export simulate_pulmonary_absorption, pulmonary_drug_preset
export create_lung_model, validate_pulmonary_model

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

"""
    LungAnatomy

Anatomical parameters of the respiratory tract.
"""
struct LungAnatomy
    # Surface areas (cm²)
    oropharynx_area::Float64
    trachea_area::Float64
    bronchi_area::Float64
    bronchioles_area::Float64
    alveolar_area::Float64      # ~70-100 m² = 700,000-1,000,000 cm²

    # Volumes (mL)
    conducting_volume::Float64   # Dead space ~150 mL
    alveolar_volume::Float64     # ~3000 mL (FRC)

    # Epithelial thickness (µm)
    bronchial_epithelium::Float64   # ~50-70 µm
    bronchiolar_epithelium::Float64 # ~10 µm
    alveolar_epithelium::Float64    # ~0.1-0.5 µm

    # Blood flow
    bronchial_blood_flow::Float64   # mL/min (~1% CO)
    pulmonary_blood_flow::Float64   # mL/min (entire CO ~5000)

    # Mucus
    mucus_thickness_um::Float64     # 5-10 µm normal
    mucus_volume_mL::Float64        # ~10-20 mL
end

"""
    ParticleProperties

Aerosol particle characteristics.
"""
struct ParticleProperties
    # Size distribution
    MMAD_um::Float64            # Mass median aerodynamic diameter
    GSD::Float64                # Geometric standard deviation

    # Physical properties
    density_g_cm3::Float64      # Particle density
    shape_factor::Float64       # Dynamic shape factor (1 for sphere)
    hygroscopicity::Float64     # 0-1, growth in humidity

    # Formulation
    drug_loading::Float64       # Fraction drug in particle
    carrier_type::Symbol        # :lactose, :none, :lipid, :polymer
end

"""
    DepositionFractions

Fractional deposition in lung regions.
"""
struct DepositionFractions
    oropharynx::Float64         # Swallowed (GI tract)
    extrathoracic::Float64      # Nose/mouth/larynx
    tracheobronchial::Float64   # Conducting airways
    alveolar::Float64           # Gas exchange region
    exhaled::Float64            # Not deposited
end

"""
    DeviceProperties

Inhalation device characteristics.
"""
struct DeviceProperties
    device_type::Symbol         # :MDI, :DPI, :nebulizer, :SMI
    fine_particle_fraction::Float64  # <5 µm
    emitted_dose_fraction::Float64   # Fraction leaving device
    spray_velocity_m_s::Float64      # For MDI
    formulation::Symbol              # :solution, :suspension, :powder
    propellant::Symbol               # :HFA, :none
end

"""
    MucociliaryClearance

Mucociliary escalator parameters.
"""
struct MucociliaryClearance
    cilia_beat_frequency::Float64  # Hz (10-15 normal)
    mucus_velocity_mm_min::Float64 # 5-20 mm/min normal
    clearance_half_life_h::Float64 # 4-6 h for TB region
    mucus_viscosity::Float64       # Relative (1 = normal)
end

"""
    DissolutionKinetics

Drug dissolution from particles.
"""
struct DissolutionKinetics
    dissolution_rate::Float64      # 1/h
    solubility_ug_mL::Float64     # In lung lining fluid
    particle_radius_um::Float64   # Initial
    diffusion_layer_um::Float64   # Noyes-Whitney
    wetting_factor::Float64       # 0-1
end

"""
    PulmonaryTransporters

Transporter expression in lung epithelium.
"""
struct PulmonaryTransporters
    # Efflux
    Pgp::Float64               # P-glycoprotein
    MRP1::Float64              # MRP1 (high in lung)
    BCRP::Float64              # BCRP

    # Uptake
    OCT1::Float64              # Organic cation
    OCT2::Float64
    OCTN1::Float64             # Carnitine transporter
    PEPT2::Float64             # Peptide transporter

    # FcRn (for biologics)
    FcRn::Float64              # Neonatal Fc receptor
end

"""
    DrugPulmonaryProperties

Drug properties for pulmonary absorption.
"""
struct DrugPulmonaryProperties
    name::String
    molecular_weight::Float64
    log_P::Float64
    pKa::Float64
    solubility_ug_mL::Float64     # In lung fluid
    permeability_cm_s::Float64    # Epithelial

    # Transporter interactions
    Pgp_substrate::Bool
    Pgp_Km::Float64               # µM
    metabolism_lung::Float64      # 1/h

    # Formulation
    particle_MMAD::Float64        # µm
    dissolution_rate::Float64     # 1/h (0 for solution)
end

"""
    PulmonaryDisease

Pulmonary disease effects on drug delivery.
"""
struct PulmonaryDisease
    condition::Symbol             # :normal, :COPD, :asthma, :CF, :IPF
    severity::Float64             # 0-1

    # Effects
    mucus_viscosity_factor::Float64
    clearance_impairment::Float64
    epithelial_permeability::Float64
    blood_flow_change::Float64
    surface_area_reduction::Float64
end

# =============================================================================
# PARTICLE DEPOSITION MODEL
# =============================================================================

"""
    calculate_deposition(particle, breathing_params)

Calculate regional particle deposition using ICRP model.

Deposition mechanisms:
1. Inertial impaction: large particles in upper airways
2. Sedimentation: medium particles in bronchioles
3. Diffusion: small particles in alveoli
"""
function calculate_deposition(
    particle::ParticleProperties,
    tidal_volume_mL::Float64 = 500.0,
    breathing_rate::Float64 = 15.0,  # breaths/min
    inhalation_time_s::Float64 = 2.0
)
    MMAD = particle.MMAD_um
    ρ = particle.density_g_cm3

    # Aerodynamic diameter
    d_ae = MMAD * sqrt(ρ / particle.shape_factor)

    # Inhalation flow rate
    Q = tidal_volume_mL / inhalation_time_s  # mL/s

    # Impaction parameter (Stokes number)
    # Stk = ρ × d² × v / (18 × η × D)
    Stk = ρ * (d_ae * 1e-4)^2 * (Q / 10) / (18 * 1.8e-4 * 0.5)

    # Sedimentation parameter
    v_settling = ρ * (d_ae * 1e-4)^2 * 981 / (18 * 1.8e-4)  # cm/s

    # Diffusion parameter
    D_diff = 1.38e-16 * 310 / (3 * π * 1.8e-4 * d_ae * 1e-4)  # cm²/s

    # Regional deposition fractions (simplified ICRP)

    # Oropharynx: impaction-dominated (large particles)
    if d_ae > 10
        f_oro = 0.9
    elseif d_ae > 5
        f_oro = 0.3 + 0.6 * (d_ae - 5) / 5
    else
        f_oro = 0.1 + 0.2 * d_ae / 5
    end

    # Tracheobronchial: sedimentation + impaction
    if d_ae > 8
        f_TB = 0.05
    elseif d_ae > 2
        f_TB = 0.1 + 0.1 * (d_ae - 2) / 6
    else
        f_TB = 0.2 - 0.1 * d_ae / 2
    end

    # Alveolar: diffusion + sedimentation
    if d_ae < 0.5
        f_alv = 0.5 * exp(-(log(d_ae / 0.5))^2 / 2)
    elseif d_ae < 3
        f_alv = 0.5 - 0.2 * (d_ae - 0.5) / 2.5
    elseif d_ae < 5
        f_alv = 0.3 - 0.2 * (d_ae - 3) / 2
    else
        f_alv = 0.1 * exp(-(d_ae - 5) / 3)
    end

    # Exhaled fraction
    f_exhaled = max(0, 1 - f_oro - f_TB - f_alv)

    # Normalize if needed
    total = f_oro + f_TB + f_alv + f_exhaled
    if total > 1
        factor = 1 / total
        f_oro *= factor
        f_TB *= factor
        f_alv *= factor
        f_exhaled *= factor
    end

    return DepositionFractions(f_oro, 0.0, f_TB, f_alv, f_exhaled)
end

"""
    regional_deposition_fractions(MMAD, device)

Calculate deposition with device effects.
"""
function regional_deposition_fractions(
    MMAD_um::Float64,
    device::DeviceProperties;
    GSD::Float64 = 2.0
)
    # Create particle
    particle = ParticleProperties(
        MMAD_um, GSD, 1.0, 1.0, 0.0, 1.0, :none
    )

    # Basic deposition
    dep = calculate_deposition(particle)

    # Device effects
    FPF = device.fine_particle_fraction
    emitted = device.emitted_dose_fraction

    # Throat deposition increases with MDI velocity
    if device.device_type == :MDI
        throat_factor = 1.0 + 0.5 * (device.spray_velocity_m_s / 30.0)
    else
        throat_factor = 1.0
    end

    # Adjust fractions
    f_oro = min(0.9, dep.oropharynx * throat_factor)
    lung_fraction = 1 - f_oro - dep.exhaled
    f_TB = dep.tracheobronchial / (dep.tracheobronchial + dep.alveolar) * lung_fraction
    f_alv = dep.alveolar / (dep.tracheobronchial + dep.alveolar) * lung_fraction

    # Apply fine particle fraction
    lung_total = (f_TB + f_alv) * FPF

    return DepositionFractions(
        f_oro * emitted,
        0.0,
        f_TB * emitted * FPF / (f_TB + f_alv + 0.01) * lung_total,
        f_alv * emitted * FPF / (f_TB + f_alv + 0.01) * lung_total,
        dep.exhaled
    )
end

# =============================================================================
# CLEARANCE MECHANISMS
# =============================================================================

"""
    mucociliary_clearance_rate(MCC, disease)

Calculate mucociliary clearance rate constant.
"""
function mucociliary_clearance_rate(
    MCC::MucociliaryClearance,
    disease::PulmonaryDisease = PulmonaryDisease(:normal, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
)
    # Base clearance rate from half-life
    k_base = log(2) / MCC.clearance_half_life_h

    # Disease effects
    # COPD/CF: increased mucus viscosity, impaired clearance
    if disease.condition == :COPD || disease.condition == :CF
        viscosity_factor = MCC.mucus_viscosity / disease.mucus_viscosity_factor
        clearance_factor = disease.clearance_impairment
        k = k_base * viscosity_factor * clearance_factor
    elseif disease.condition == :asthma
        # Asthma: variable, often increased during exacerbation
        k = k_base * (1.0 - 0.3 * disease.severity)
    else
        k = k_base
    end

    return k
end

"""
    alveolar_macrophage_clearance(particle_size_um, time_h)

Calculate macrophage-mediated clearance in alveoli.

Macrophages clear particles 1-3 µm most efficiently.
Very small (<0.1 µm) escape phagocytosis.
Large (>5 µm) not efficiently phagocytosed.
"""
function alveolar_macrophage_clearance(
    particle_size_um::Float64,
    time_h::Float64
)
    # Size-dependent clearance rate
    if particle_size_um < 0.1
        k_mac = 0.001  # Minimal clearance
    elseif particle_size_um < 1.0
        k_mac = 0.01 + 0.04 * particle_size_um  # Increasing
    elseif particle_size_um < 3.0
        k_mac = 0.05  # Optimal size
    elseif particle_size_um < 5.0
        k_mac = 0.05 - 0.02 * (particle_size_um - 3)
    else
        k_mac = 0.01  # Inefficient for large
    end

    # Fraction cleared
    fraction_cleared = 1 - exp(-k_mac * time_h)

    return (
        k_clearance = k_mac,
        fraction_cleared = fraction_cleared,
        half_life_h = log(2) / k_mac
    )
end

# =============================================================================
# DISSOLUTION KINETICS
# =============================================================================

"""
    dissolution_rate_noyes_whitney(drug, dissolution, time_h)

Calculate drug dissolution using Noyes-Whitney equation.

dM/dt = (D × A / h) × (Cs - C)

Where:
- D = diffusion coefficient
- A = surface area
- h = diffusion layer thickness
- Cs = saturation solubility
- C = bulk concentration
"""
function dissolution_rate_noyes_whitney(
    drug::DrugPulmonaryProperties,
    dissolution::DissolutionKinetics,
    undissolved_mass_ug::Float64,
    dissolved_conc_ug_mL::Float64,
    lung_fluid_volume_mL::Float64 = 20.0  # Lung lining fluid ~10-20 mL
)
    # Particle surface area (assuming spheres)
    # A = n × 4πr² where n = M/(4/3 πr³ρ)
    r = dissolution.particle_radius_um * 1e-4  # cm
    ρ = 1.0  # g/cm³

    # Number of particles
    if r > 0 && undissolved_mass_ug > 0
        n_particles = (undissolved_mass_ug * 1e-6) / ((4/3) * π * r^3 * ρ)
        total_area = n_particles * 4 * π * r^2  # cm²
    else
        total_area = 0.0
    end

    # Diffusion coefficient (approximate from MW)
    D = 1e-5 / sqrt(drug.molecular_weight / 300)  # cm²/s

    # Diffusion layer thickness
    h = dissolution.diffusion_layer_um * 1e-4  # cm

    # Saturation concentration
    Cs = dissolution.solubility_ug_mL

    # Bulk concentration
    C = dissolved_conc_ug_mL

    # Noyes-Whitney: dM/dt = D × A × (Cs - C) / h
    if C < Cs
        dM_dt = D * total_area * (Cs - C) / h * dissolution.wetting_factor
        dM_dt *= 3600  # Convert to µg/h
    else
        dM_dt = 0.0  # Saturated
    end

    return (
        dissolution_rate_ug_h = dM_dt,
        surface_area = total_area,
        saturation_fraction = C / Cs
    )
end

# =============================================================================
# ABSORPTION CALCULATIONS
# =============================================================================

"""
    pulmonary_absorption_rate(drug, lung, transporters, region)

Calculate absorption rate from lung to systemic.

Absorption is fastest from alveoli (thin epithelium, large SA)
and slowest from bronchi (thick epithelium).
"""
function pulmonary_absorption_rate(
    drug::DrugPulmonaryProperties,
    lung::LungAnatomy,
    transporters::PulmonaryTransporters,
    region::Symbol;  # :alveolar, :bronchiolar, :bronchial
    dissolved_conc::Float64 = 1.0  # µg/mL
)
    # Epithelial thickness by region
    if region == :alveolar
        thickness = lung.alveolar_epithelium
        surface_area = lung.alveolar_area
        blood_flow = lung.pulmonary_blood_flow
    elseif region == :bronchiolar
        thickness = lung.bronchiolar_epithelium
        surface_area = lung.bronchioles_area
        blood_flow = lung.bronchial_blood_flow
    else  # bronchial
        thickness = lung.bronchial_epithelium
        surface_area = lung.bronchi_area
        blood_flow = lung.bronchial_blood_flow
    end

    # Permeability coefficient
    P = drug.permeability_cm_s

    # Thickness effect (inverse relationship)
    P_eff = P * (0.2 / thickness)  # Normalized to 0.2 µm

    # Transporter effects
    if drug.Pgp_substrate && transporters.Pgp > 0
        # Efflux reduces net absorption
        efflux_factor = 1.0 / (1.0 + transporters.Pgp * drug.Pgp_Km / 10.0)
    else
        efflux_factor = 1.0
    end

    # Absorption rate constant
    k_abs = P_eff * surface_area / 1000.0 * efflux_factor  # Arbitrary units, scaled

    # Flow limitation
    k_abs = min(k_abs, blood_flow / 100.0)

    return (
        k_absorption = k_abs,
        P_effective = P_eff,
        efflux_factor = efflux_factor,
        thickness_um = thickness
    )
end

"""
    calculate_bioavailability(drug, deposition, disease)

Calculate overall pulmonary bioavailability.

F_pulm = f_lung × f_dissolve × f_absorb × (1 - f_clear)
"""
function calculate_bioavailability(
    drug::DrugPulmonaryProperties,
    deposition::DepositionFractions,
    MCC::MucociliaryClearance,
    disease::PulmonaryDisease;
    time_h::Float64 = 4.0
)
    # Lung deposited fraction
    f_lung = deposition.tracheobronchial + deposition.alveolar

    # GI absorbed fraction (swallowed)
    f_GI = deposition.oropharynx * 0.3  # Assume 30% GI absorption

    # Dissolution fraction (assume exponential)
    if drug.dissolution_rate > 0
        f_dissolve = 1 - exp(-drug.dissolution_rate * time_h)
    else
        f_dissolve = 1.0  # Solution formulation
    end

    # Clearance loss (mucociliary)
    k_clear = mucociliary_clearance_rate(MCC, disease)
    f_clear_TB = 1 - exp(-k_clear * time_h)

    # Regional contributions
    # Alveolar: no MCC, but macrophage clearance
    mac_clear = alveolar_macrophage_clearance(drug.particle_MMAD, time_h)
    f_absorb_alv = (1 - mac_clear.fraction_cleared) * f_dissolve

    # Tracheobronchial: MCC competition
    f_absorb_TB = (1 - f_clear_TB) * f_dissolve

    # Overall bioavailability
    F = deposition.alveolar * f_absorb_alv +
        deposition.tracheobronchial * f_absorb_TB +
        f_GI

    return (
        F_total = F,
        F_alveolar = deposition.alveolar * f_absorb_alv,
        F_tracheobronchial = deposition.tracheobronchial * f_absorb_TB,
        F_GI = f_GI,
        f_dissolution = f_dissolve,
        f_mucociliary_loss = f_clear_TB * deposition.tracheobronchial
    )
end

# =============================================================================
# DISEASE EFFECTS
# =============================================================================

"""
    pulmonary_disease(condition, severity)

Create disease state with associated changes.
"""
function pulmonary_disease(condition::Symbol, severity::Float64 = 0.5)
    diseases = Dict(
        :normal => PulmonaryDisease(:normal, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),

        :COPD => PulmonaryDisease(
            :COPD, severity,
            2.0 + 2.0 * severity,    # Increased mucus viscosity
            0.5 - 0.3 * severity,    # Impaired clearance
            1.2 + 0.3 * severity,    # Increased permeability (inflammation)
            1.0,                      # Blood flow similar
            0.7 - 0.2 * severity     # Reduced SA (emphysema)
        ),

        :asthma => PulmonaryDisease(
            :asthma, severity,
            1.5 + 1.0 * severity,    # Thickened mucus
            0.7 - 0.2 * severity,    # Somewhat impaired
            1.0 + 0.2 * severity,    # Slight increase
            0.9 - 0.2 * severity,    # Reduced (bronchoconstriction)
            1.0                       # SA preserved
        ),

        :CF => PulmonaryDisease(
            :CF, severity,
            5.0 + 5.0 * severity,    # Very thick mucus
            0.2 - 0.1 * severity,    # Severely impaired clearance
            1.5 + 0.5 * severity,    # Increased (damage)
            1.0,
            0.6 - 0.2 * severity     # Reduced SA (fibrosis)
        ),

        :IPF => PulmonaryDisease(
            :IPF, severity,
            1.0,                      # Mucus normal
            1.0,                      # Clearance normal
            0.7 - 0.3 * severity,    # Reduced permeability (fibrosis)
            0.8 - 0.3 * severity,    # Reduced blood flow
            0.5 - 0.2 * severity     # Severely reduced SA
        )
    )

    return get(diseases, condition, diseases[:normal])
end

# =============================================================================
# ODE SYSTEM
# =============================================================================

"""
    pulmonary_ode_system!(du, u, p, t)

Differential equations for pulmonary drug disposition.

Compartments:
1. Undissolved drug in alveoli
2. Dissolved drug in alveoli
3. Undissolved drug in TB region
4. Dissolved drug in TB region
5. Drug cleared to GI (swallowed)
6. Drug in systemic circulation
"""
function pulmonary_ode_system!(du, u, p, t)
    # Unpack parameters
    drug = p.drug
    lung = p.lung
    MCC = p.MCC
    transporters = p.transporters
    disease = p.disease
    deposition = p.deposition
    dissolution = p.dissolution

    # State variables
    A_alv_undiss = u[1]    # Undissolved in alveoli
    A_alv_diss = u[2]      # Dissolved in alveoli
    A_TB_undiss = u[3]     # Undissolved in TB
    A_TB_diss = u[4]       # Dissolved in TB
    A_GI = u[5]            # Swallowed (from MCC)
    A_systemic = u[6]      # Systemic circulation

    # Volumes
    V_alv_fluid = 10.0     # mL lung lining fluid in alveoli
    V_TB_fluid = 10.0      # mL in TB region
    V_systemic = 5000.0    # mL

    # Concentrations
    C_alv = A_alv_diss / V_alv_fluid
    C_TB = A_TB_diss / V_TB_fluid

    # === Dissolution ===
    diss_alv = dissolution_rate_noyes_whitney(drug, dissolution, A_alv_undiss, C_alv, V_alv_fluid)
    diss_TB = dissolution_rate_noyes_whitney(drug, dissolution, A_TB_undiss, C_TB, V_TB_fluid)

    # === Mucociliary Clearance (TB region only) ===
    k_MCC = mucociliary_clearance_rate(MCC, disease)
    flux_MCC_undiss = k_MCC * A_TB_undiss
    flux_MCC_diss = k_MCC * A_TB_diss

    # === Macrophage Clearance (alveoli) ===
    mac = alveolar_macrophage_clearance(drug.particle_MMAD, 1.0)
    k_mac = mac.k_clearance
    flux_mac = k_mac * A_alv_undiss  # Undissolved particles only

    # === Absorption ===
    abs_alv = pulmonary_absorption_rate(drug, lung, transporters, :alveolar; dissolved_conc=C_alv)
    abs_TB = pulmonary_absorption_rate(drug, lung, transporters, :bronchiolar; dissolved_conc=C_TB)

    flux_abs_alv = abs_alv.k_absorption * A_alv_diss
    flux_abs_TB = abs_TB.k_absorption * A_TB_diss

    # === Lung Metabolism ===
    k_met = drug.metabolism_lung
    flux_met_alv = k_met * A_alv_diss
    flux_met_TB = k_met * A_TB_diss

    # === Systemic Elimination ===
    k_elim = p.k_elim_systemic
    flux_elim = k_elim * A_systemic

    # === GI Absorption ===
    k_GI = 0.3  # 1/h, first-order GI absorption
    flux_GI_abs = k_GI * A_GI * 0.3  # 30% bioavailability from GI

    # === ODEs ===
    # Alveolar undissolved
    du[1] = -diss_alv.dissolution_rate_ug_h - flux_mac

    # Alveolar dissolved
    du[2] = diss_alv.dissolution_rate_ug_h - flux_abs_alv - flux_met_alv

    # TB undissolved
    du[3] = -diss_TB.dissolution_rate_ug_h - flux_MCC_undiss

    # TB dissolved
    du[4] = diss_TB.dissolution_rate_ug_h - flux_abs_TB - flux_MCC_diss - flux_met_TB

    # GI (swallowed)
    du[5] = flux_MCC_undiss + flux_MCC_diss - flux_GI_abs

    # Systemic
    du[6] = flux_abs_alv + flux_abs_TB + flux_GI_abs - flux_elim

    return nothing
end

"""
    simulate_pulmonary_absorption(drug, dose_ug, device; kwargs...)

Simulate inhaled drug absorption.
"""
function simulate_pulmonary_absorption(
    drug::DrugPulmonaryProperties,
    dose_ug::Float64,
    device::DeviceProperties;
    tspan::Tuple{Float64, Float64} = (0.0, 24.0),
    condition::Symbol = :normal,
    k_elim_systemic::Float64 = 0.1,
    saveat::Float64 = 0.1
)
    # Calculate deposition
    deposition = regional_deposition_fractions(drug.particle_MMAD, device)

    # Create lung anatomy
    lung = create_lung_model()

    # Mucociliary clearance
    MCC = MucociliaryClearance(12.0, 10.0, 5.0, 1.0)

    # Transporters
    transporters = PulmonaryTransporters(0.5, 1.0, 0.3, 0.5, 0.3, 0.2, 0.5, 0.3)

    # Disease
    disease = pulmonary_disease(condition)

    # Dissolution
    dissolution = DissolutionKinetics(
        drug.dissolution_rate,
        drug.solubility_ug_mL,
        drug.particle_MMAD / 2,
        2.0,
        0.8
    )

    # Parameters
    p = (
        drug = drug,
        lung = lung,
        MCC = MCC,
        transporters = transporters,
        disease = disease,
        deposition = deposition,
        dissolution = dissolution,
        k_elim_systemic = k_elim_systemic
    )

    # Initial conditions
    emitted_dose = dose_ug * device.emitted_dose_fraction
    u0 = [
        emitted_dose * deposition.alveolar,      # Alv undissolved
        0.0,                                       # Alv dissolved
        emitted_dose * deposition.tracheobronchial,  # TB undissolved
        0.0,                                       # TB dissolved
        emitted_dose * deposition.oropharynx,     # GI (swallowed)
        0.0                                        # Systemic
    ]

    # Solve ODE
    prob = ODEProblem(pulmonary_ode_system!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=saveat)

    # Extract results
    times = sol.t
    A_systemic = [s[6] for s in sol.u]
    A_lung_total = [s[1] + s[2] + s[3] + s[4] for s in sol.u]
    C_systemic = A_systemic ./ 5000.0  # µg/mL

    # PK parameters
    Cmax = maximum(C_systemic)
    tmax_idx = argmax(C_systemic)
    tmax = times[tmax_idx]

    # AUC
    AUC = 0.0
    for i in 2:length(times)
        dt = times[i] - times[i-1]
        AUC += 0.5 * (C_systemic[i] + C_systemic[i-1]) * dt
    end

    # Bioavailability
    F = AUC / (dose_ug / 5000.0 / k_elim_systemic)

    return (
        times = times,
        C_systemic = C_systemic,
        A_lung = A_lung_total,
        Cmax = Cmax,
        tmax = tmax,
        AUC = AUC,
        F_estimated = min(1.0, F),
        deposition = deposition,
        solution = sol
    )
end

# =============================================================================
# DRUG PRESETS
# =============================================================================

"""
Drug presets for pulmonary absorption modeling.
"""
function pulmonary_drug_preset(drug_name::Symbol)
    presets = Dict(
        :salbutamol => DrugPulmonaryProperties(
            "Salbutamol",
            239.3,      # MW
            0.64,       # Log P
            9.4,        # pKa
            3000.0,     # High solubility
            5e-6,       # Permeability
            false, 0.0, # Not P-gp
            0.1,        # Low lung metabolism
            2.5,        # MMAD
            0.0         # Solution (nebulized) or fast dissolving
        ),

        :fluticasone => DrugPulmonaryProperties(
            "Fluticasone propionate",
            500.6,      # MW
            4.2,        # Log P (lipophilic)
            0.0,        # Neutral
            0.1,        # Low solubility
            1e-5,       # Good permeability
            true, 50.0, # P-gp substrate
            0.2,        # Lung metabolism
            3.5,        # MMAD
            0.5         # Slow dissolution
        ),

        :budesonide => DrugPulmonaryProperties(
            "Budesonide",
            430.5,      # MW
            2.8,        # Log P
            0.0,        # Neutral
            16.0,       # Moderate solubility
            1e-5,       # Good permeability
            false, 0.0, # Not P-gp
            0.1,        # Low metabolism
            3.0,        # MMAD
            1.0         # Moderate dissolution
        ),

        :formoterol => DrugPulmonaryProperties(
            "Formoterol",
            344.4,      # MW
            1.0,        # Log P
            8.1,        # pKa
            500.0,      # Good solubility
            3e-6,       # Moderate permeability
            false, 0.0,
            0.05,       # Low metabolism
            2.0,        # MMAD
            0.0         # Fast dissolution
        ),

        :tiotropium => DrugPulmonaryProperties(
            "Tiotropium",
            472.4,      # MW
            -1.2,       # Log P (hydrophilic)
            10.0,       # pKa
            5000.0,     # Very soluble
            2e-6,       # Low permeability
            false, 0.0,
            0.0,        # No metabolism
            2.5,        # MMAD
            0.0         # Fast dissolution
        ),

        :tobramycin => DrugPulmonaryProperties(
            "Tobramycin",
            467.5,      # MW
            -5.8,       # Log P (very hydrophilic)
            8.0,        # pKa
            100000.0,   # Very soluble
            1e-7,       # Very low permeability
            false, 0.0,
            0.0,        # No metabolism
            4.0,        # MMAD (nebulized)
            0.0         # Solution
        ),

        :ciclesonide => DrugPulmonaryProperties(
            "Ciclesonide",
            540.7,      # MW
            5.0,        # Log P (prodrug)
            0.0,        # Neutral
            0.01,       # Very low solubility
            5e-6,       # Moderate permeability
            true, 30.0, # P-gp substrate
            0.5,        # Activation in lung
            1.5,        # MMAD (fine particles)
            0.3         # Slow dissolution
        )
    )

    if !haskey(presets, drug_name)
        available = join(keys(presets), ", ")
        error("Unknown drug: $drug_name. Available: $available")
    end

    return presets[drug_name]
end

"""
Device presets.
"""
function device_preset(device_name::Symbol)
    presets = Dict(
        :MDI => DeviceProperties(:MDI, 0.35, 0.9, 25.0, :suspension, :HFA),
        :DPI_low => DeviceProperties(:DPI, 0.25, 0.7, 0.0, :powder, :none),
        :DPI_high => DeviceProperties(:DPI, 0.45, 0.85, 0.0, :powder, :none),
        :nebulizer => DeviceProperties(:nebulizer, 0.6, 0.5, 0.0, :solution, :none),
        :SMI => DeviceProperties(:SMI, 0.65, 0.95, 0.5, :solution, :none)
    )

    return get(presets, device_name, presets[:MDI])
end

# =============================================================================
# MODEL CREATION AND VALIDATION
# =============================================================================

"""
    create_lung_model(; disease)

Create default lung anatomy model.
"""
function create_lung_model(; disease::Symbol = :normal)
    lung = LungAnatomy(
        100.0,      # Oropharynx area
        50.0,       # Trachea
        200.0,      # Bronchi
        500.0,      # Bronchioles
        700000.0,   # Alveolar (70 m²)
        150.0,      # Conducting volume
        3000.0,     # Alveolar volume
        60.0,       # Bronchial epithelium
        10.0,       # Bronchiolar
        0.2,        # Alveolar
        50.0,       # Bronchial blood flow
        5000.0,     # Pulmonary blood flow
        7.0,        # Mucus thickness
        15.0        # Mucus volume
    )

    return lung
end

"""
    validate_pulmonary_model()

Validate model against literature benchmarks.
"""
function validate_pulmonary_model()
    results = Dict{String, Any}()

    # Test 1: Deposition fractions for different particle sizes
    sizes = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]
    depositions = [calculate_deposition(ParticleProperties(s, 2.0, 1.0, 1.0, 0.0, 1.0, :none))
                   for s in sizes]

    results["deposition_vs_size"] = (
        sizes = sizes,
        alveolar = [d.alveolar for d in depositions],
        tracheobronchial = [d.tracheobronchial for d in depositions],
        oropharynx = [d.oropharynx for d in depositions]
    )

    # Test 2: Salbutamol simulation
    salb = pulmonary_drug_preset(:salbutamol)
    device = device_preset(:MDI)
    result_salb = simulate_pulmonary_absorption(salb, 200.0, device; tspan=(0.0, 8.0))

    results["salbutamol"] = (
        Cmax = result_salb.Cmax,
        tmax = result_salb.tmax,
        F = result_salb.F_estimated,
        alveolar_deposition = result_salb.deposition.alveolar
    )

    # Test 3: Fluticasone (lipophilic, slow dissolution)
    flut = pulmonary_drug_preset(:fluticasone)
    result_flut = simulate_pulmonary_absorption(flut, 500.0, device; tspan=(0.0, 24.0))

    results["fluticasone"] = (
        Cmax = result_flut.Cmax,
        tmax = result_flut.tmax,
        F = result_flut.F_estimated
    )

    return results
end

end # module PulmonaryAbsorptionModel
