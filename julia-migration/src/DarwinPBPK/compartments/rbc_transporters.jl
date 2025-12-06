"""
RBC Transporter Module

Models active and facilitated transport mechanisms in red blood cells
for accurate prediction of drug accumulation and RBC:plasma ratios.

Key Transporters:
- AE1 (Band 3): Anion exchanger (Cl⁻/HCO₃⁻), organic anions
- OAT: Organic anion transporters
- OCT: Organic cation transporters
- GLUT1: Glucose transporter (affects cell volume)
- MCT1: Monocarboxylate transporter (lactate, pyruvate)
- ENT1/ENT2: Equilibrative nucleoside transporters
- URAT1: Urate transporter

Clinical Relevance:
- Chloroquine: Band 3 substrate, high RBC accumulation
- Metformin: OCT substrate, minimal RBC entry
- Nucleoside analogs: ENT substrates

References:
- Hebert SC (2004) Renal and red blood cell transporters
- Ellory JC (1998) Ion transport in red blood cells
- Tse CM (2004) Organic cation transporters in RBC

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module RBCTransporters

using Statistics

export RBCTransporterProfile, DrugRBCTransport
export create_normal_rbc_transporters, calculate_rbc_transport
export calculate_rbc_accumulation, get_rbc_transport_data
export RBC_TRANSPORTER_SUBSTRATES, apply_transporter_inhibition

# ============================================================================
# CONSTANTS
# ============================================================================

# Transporter expression levels (relative to reference)
const NORMAL_AE1_EXPRESSION = 1.0e6      # copies/cell (very high - major membrane protein)
const NORMAL_GLUT1_EXPRESSION = 3.0e5    # copies/cell
const NORMAL_ENT1_EXPRESSION = 1.0e4     # copies/cell
const NORMAL_MCT1_EXPRESSION = 5.0e3     # copies/cell
const NORMAL_URAT1_EXPRESSION = 1.0e3    # copies/cell

# Kinetic parameters (Km in μM, Vmax in pmol/min/10^6 cells)
const AE1_KM_CHLORIDE = 40.0e3      # μM (40 mM) - physiological
const AE1_KM_BICARBONATE = 20.0e3   # μM (20 mM)
const GLUT1_KM_GLUCOSE = 5.0e3      # μM (5 mM)
const MCT1_KM_LACTATE = 3.5e3       # μM (3.5 mM)

# RBC physical parameters
const RBC_VOLUME = 90.0             # fL (femtoliters)
const RBC_SURFACE_AREA = 140.0      # μm²
const RBC_MEMBRANE_THICKNESS = 7.5  # nm

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    RBCTransporterProfile

Complete transporter profile for red blood cells.

# Fields
Expression levels (relative to normal):
- `ae1::Float64`: Anion exchanger 1 (Band 3)
- `glut1::Float64`: Glucose transporter 1
- `ent1::Float64`: Equilibrative nucleoside transporter 1
- `ent2::Float64`: Equilibrative nucleoside transporter 2
- `mct1::Float64`: Monocarboxylate transporter 1
- `urat1::Float64`: Urate transporter 1
- `oat_like::Float64`: OAT-like activity
- `oct_like::Float64`: OCT-like activity
- `condition::Symbol`: :normal, :sickle_cell, :thalassemia, etc.
"""
struct RBCTransporterProfile
    ae1::Float64
    glut1::Float64
    ent1::Float64
    ent2::Float64
    mct1::Float64
    urat1::Float64
    oat_like::Float64
    oct_like::Float64
    condition::Symbol

    function RBCTransporterProfile(;
        ae1 = 1.0,
        glut1 = 1.0,
        ent1 = 1.0,
        ent2 = 1.0,
        mct1 = 1.0,
        urat1 = 1.0,
        oat_like = 1.0,
        oct_like = 1.0,
        condition = :normal
    )
        new(ae1, glut1, ent1, ent2, mct1, urat1, oat_like, oct_like, condition)
    end
end

"""
    DrugRBCTransport

Drug-specific RBC transport parameters.

# Fields
- `name::String`: Drug name
- `primary_transporter::Symbol`: Main transporter (:ae1, :ent1, :passive, etc.)
- `km::Float64`: Michaelis constant (μM)
- `vmax_relative::Float64`: Relative Vmax (1.0 = reference substrate)
- `passive_permeability::Float64`: Passive diffusion (cm/s)
- `is_substrate::Bool`: Is drug a transporter substrate?
- `is_inhibitor::Bool`: Does drug inhibit transporters?
- `inhibition_ki::Float64`: Ki for inhibition (μM)
"""
struct DrugRBCTransport
    name::String
    primary_transporter::Symbol
    km::Float64
    vmax_relative::Float64
    passive_permeability::Float64
    is_substrate::Bool
    is_inhibitor::Bool
    inhibition_ki::Float64

    function DrugRBCTransport(name;
        primary_transporter = :passive,
        km = 100.0,
        vmax_relative = 1.0,
        passive_permeability = 1.0e-6,
        is_substrate = false,
        is_inhibitor = false,
        inhibition_ki = Inf
    )
        new(name, primary_transporter, km, vmax_relative,
            passive_permeability, is_substrate, is_inhibitor, inhibition_ki)
    end
end

# ============================================================================
# TRANSPORTER SUBSTRATE DATABASE
# ============================================================================

"""
Drug-specific RBC transporter data.

Sources:
- Ellory JC (1998) - Antimalarials and Band 3
- Benderitter P (2019) - Chloroquine transport
- Koepsell H (2013) - OCT substrates
"""
const RBC_TRANSPORTER_SUBSTRATES = Dict{String, DrugRBCTransport}(
    # =========================================
    # ANTIMALARIALS - AE1 (Band 3) substrates
    # =========================================
    "chloroquine" => DrugRBCTransport("chloroquine";
        primary_transporter = :ae1,
        km = 50.0,              # μM
        vmax_relative = 0.8,
        passive_permeability = 5.0e-5,  # Also significant passive
        is_substrate = true,
        is_inhibitor = true,
        inhibition_ki = 20.0    # Also inhibits AE1
    ),
    "hydroxychloroquine" => DrugRBCTransport("hydroxychloroquine";
        primary_transporter = :ae1,
        km = 80.0,
        vmax_relative = 0.6,
        passive_permeability = 3.0e-5,
        is_substrate = true,
        is_inhibitor = true,
        inhibition_ki = 35.0
    ),
    "quinine" => DrugRBCTransport("quinine";
        primary_transporter = :ae1,
        km = 100.0,
        vmax_relative = 0.5,
        passive_permeability = 2.0e-5,
        is_substrate = true,
        is_inhibitor = false
    ),
    "primaquine" => DrugRBCTransport("primaquine";
        primary_transporter = :passive,  # Mainly passive
        km = 200.0,
        vmax_relative = 0.3,
        passive_permeability = 1.0e-4,
        is_substrate = false
    ),
    "mefloquine" => DrugRBCTransport("mefloquine";
        primary_transporter = :passive,
        km = 150.0,
        vmax_relative = 0.4,
        passive_permeability = 8.0e-5,
        is_substrate = false
    ),

    # =========================================
    # NUCLEOSIDE ANALOGS - ENT substrates
    # =========================================
    "zidovudine" => DrugRBCTransport("zidovudine";
        primary_transporter = :ent1,
        km = 250.0,
        vmax_relative = 1.0,
        passive_permeability = 5.0e-7,
        is_substrate = true
    ),
    "didanosine" => DrugRBCTransport("didanosine";
        primary_transporter = :ent1,
        km = 300.0,
        vmax_relative = 0.8,
        passive_permeability = 2.0e-7,
        is_substrate = true
    ),
    "stavudine" => DrugRBCTransport("stavudine";
        primary_transporter = :ent1,
        km = 180.0,
        vmax_relative = 1.2,
        passive_permeability = 3.0e-7,
        is_substrate = true
    ),
    "lamivudine" => DrugRBCTransport("lamivudine";
        primary_transporter = :ent1,
        km = 400.0,
        vmax_relative = 0.6,
        passive_permeability = 1.0e-7,
        is_substrate = true
    ),
    "abacavir" => DrugRBCTransport("abacavir";
        primary_transporter = :ent1,
        km = 220.0,
        vmax_relative = 0.9,
        passive_permeability = 1.0e-6,
        is_substrate = true
    ),
    "ribavirin" => DrugRBCTransport("ribavirin";
        primary_transporter = :ent1,
        km = 15.0,              # Very high affinity
        vmax_relative = 2.0,
        passive_permeability = 1.0e-8,
        is_substrate = true
    ),
    "gemcitabine" => DrugRBCTransport("gemcitabine";
        primary_transporter = :ent1,
        km = 160.0,
        vmax_relative = 1.5,
        passive_permeability = 5.0e-8,
        is_substrate = true
    ),

    # =========================================
    # ORGANIC ANIONS - OAT-like substrates
    # =========================================
    "probenecid" => DrugRBCTransport("probenecid";
        primary_transporter = :oat_like,
        km = 80.0,
        vmax_relative = 0.5,
        passive_permeability = 2.0e-6,
        is_substrate = true,
        is_inhibitor = true,
        inhibition_ki = 15.0
    ),
    "furosemide" => DrugRBCTransport("furosemide";
        primary_transporter = :ae1,
        km = 200.0,
        vmax_relative = 0.3,
        passive_permeability = 5.0e-7,
        is_substrate = true,
        is_inhibitor = true,    # Loop diuretic mechanism
        inhibition_ki = 50.0
    ),
    "penicillin_g" => DrugRBCTransport("penicillin_g";
        primary_transporter = :oat_like,
        km = 500.0,
        vmax_relative = 0.2,
        passive_permeability = 1.0e-8,
        is_substrate = true
    ),
    "methotrexate" => DrugRBCTransport("methotrexate";
        primary_transporter = :oat_like,
        km = 20.0,
        vmax_relative = 0.4,
        passive_permeability = 1.0e-9,
        is_substrate = true
    ),

    # =========================================
    # ORGANIC CATIONS - OCT-like substrates
    # =========================================
    "metformin" => DrugRBCTransport("metformin";
        primary_transporter = :oct_like,
        km = 1500.0,            # Low affinity
        vmax_relative = 0.3,
        passive_permeability = 1.0e-8,  # Very low passive
        is_substrate = true
    ),
    "cimetidine" => DrugRBCTransport("cimetidine";
        primary_transporter = :oct_like,
        km = 300.0,
        vmax_relative = 0.5,
        passive_permeability = 5.0e-7,
        is_substrate = true,
        is_inhibitor = true,
        inhibition_ki = 100.0
    ),
    "ranitidine" => DrugRBCTransport("ranitidine";
        primary_transporter = :oct_like,
        km = 400.0,
        vmax_relative = 0.4,
        passive_permeability = 3.0e-7,
        is_substrate = true
    ),

    # =========================================
    # MONOCARBOXYLATES - MCT1 substrates
    # =========================================
    "valproic_acid" => DrugRBCTransport("valproic_acid";
        primary_transporter = :mct1,
        km = 1000.0,
        vmax_relative = 0.8,
        passive_permeability = 5.0e-5,  # Also high passive
        is_substrate = true
    ),
    "salicylic_acid" => DrugRBCTransport("salicylic_acid";
        primary_transporter = :mct1,
        km = 800.0,
        vmax_relative = 0.6,
        passive_permeability = 1.0e-5,
        is_substrate = true
    ),

    # =========================================
    # URATE - URAT1 substrates
    # =========================================
    "uric_acid" => DrugRBCTransport("uric_acid";
        primary_transporter = :urat1,
        km = 200.0,
        vmax_relative = 1.0,
        passive_permeability = 1.0e-8,
        is_substrate = true
    ),
    "benzbromarone" => DrugRBCTransport("benzbromarone";
        primary_transporter = :urat1,
        km = 5.0,
        vmax_relative = 0.1,
        passive_permeability = 1.0e-5,
        is_substrate = false,
        is_inhibitor = true,
        inhibition_ki = 2.0
    ),

    # =========================================
    # MAINLY PASSIVE TRANSPORT
    # =========================================
    "caffeine" => DrugRBCTransport("caffeine";
        primary_transporter = :passive,
        passive_permeability = 5.0e-4,
        is_substrate = false
    ),
    "theophylline" => DrugRBCTransport("theophylline";
        primary_transporter = :passive,
        passive_permeability = 3.0e-4,
        is_substrate = false
    ),
    "ethanol" => DrugRBCTransport("ethanol";
        primary_transporter = :passive,
        passive_permeability = 1.0e-2,  # Very high
        is_substrate = false
    )
)

# ============================================================================
# PROFILE FACTORIES
# ============================================================================

"""
    create_normal_rbc_transporters()

Create normal RBC transporter profile.
"""
function create_normal_rbc_transporters()
    return RBCTransporterProfile(condition = :normal)
end

"""
    create_disease_rbc_transporters(disease::Symbol)

Create RBC transporter profile for disease states.

Diseases:
- :sickle_cell - Altered AE1, increased passive permeability
- :thalassemia - Reduced transporter expression
- :g6pd_deficiency - Oxidative stress effects
- :hereditary_spherocytosis - Band 3 deficiency
- :malaria_infected - Parasite-induced changes
- :diabetes - Glycation effects
- :uremia - Uremic toxin effects
- :elderly - Age-related changes
"""
function create_disease_rbc_transporters(disease::Symbol)
    profile = RBCTransporterProfile(condition = disease)

    if disease == :sickle_cell
        # Sickling affects membrane transporters
        return RBCTransporterProfile(
            ae1 = 0.7,      # Reduced Band 3
            glut1 = 1.2,    # Compensatory increase
            ent1 = 0.9,
            mct1 = 0.8,
            condition = :sickle_cell
        )
    elseif disease == :thalassemia
        return RBCTransporterProfile(
            ae1 = 0.8,
            glut1 = 0.9,
            ent1 = 0.85,
            condition = :thalassemia
        )
    elseif disease == :g6pd_deficiency
        # Oxidative damage reduces function
        return RBCTransporterProfile(
            ae1 = 0.85,
            glut1 = 1.1,    # Increased glucose demand
            mct1 = 0.9,
            condition = :g6pd_deficiency
        )
    elseif disease == :hereditary_spherocytosis
        # Band 3 (AE1) is deficient
        return RBCTransporterProfile(
            ae1 = 0.3,      # Severely reduced
            glut1 = 0.9,
            condition = :hereditary_spherocytosis
        )
    elseif disease == :malaria_infected
        # Parasite modifies RBC membrane
        return RBCTransporterProfile(
            ae1 = 1.5,      # New permeation pathways
            glut1 = 2.0,    # Increased glucose uptake
            ent1 = 1.5,     # Parasite imports nucleosides
            condition = :malaria_infected
        )
    elseif disease == :diabetes
        # Glycation affects transporters
        return RBCTransporterProfile(
            ae1 = 0.9,
            glut1 = 0.7,    # Downregulated (chronic hyperglycemia)
            mct1 = 1.1,
            condition = :diabetes
        )
    elseif disease == :uremia
        # Uremic toxins inhibit transporters
        return RBCTransporterProfile(
            ae1 = 0.8,
            oat_like = 0.6,
            oct_like = 0.7,
            condition = :uremia
        )
    elseif disease == :elderly
        # Age-related decline
        return RBCTransporterProfile(
            ae1 = 0.85,
            glut1 = 0.9,
            ent1 = 0.8,
            mct1 = 0.85,
            condition = :elderly
        )
    else
        return profile
    end
end

# ============================================================================
# TRANSPORT CALCULATIONS
# ============================================================================

"""
    calculate_michaelis_menten(conc::Float64, km::Float64, vmax::Float64)

Calculate transport rate using Michaelis-Menten kinetics.

Rate = Vmax * [S] / (Km + [S])
"""
function calculate_michaelis_menten(conc::Float64, km::Float64, vmax::Float64)
    return vmax * conc / (km + conc)
end

"""
    calculate_passive_flux(conc_out::Float64, conc_in::Float64,
                           permeability::Float64, surface_area::Float64)

Calculate passive diffusion flux across RBC membrane.

Flux = P * A * (Cout - Cin)
"""
function calculate_passive_flux(conc_out::Float64, conc_in::Float64,
                                 permeability::Float64, surface_area::Float64)
    return permeability * surface_area * (conc_out - conc_in)
end

"""
    calculate_rbc_transport(drug::DrugRBCTransport,
                            plasma_conc::Float64,
                            rbc_conc::Float64,
                            transporters::RBCTransporterProfile)

Calculate net transport rate into RBC.

# Arguments
- `drug`: Drug transport parameters
- `plasma_conc`: Plasma concentration (μM)
- `rbc_conc`: Current RBC concentration (μM)
- `transporters`: RBC transporter profile

# Returns
Dict with:
- `net_flux`: Net influx rate (pmol/min/10^6 cells)
- `active_influx`: Active transport component
- `passive_flux`: Passive diffusion component
- `transporter_saturation`: Fraction of Vmax used
"""
function calculate_rbc_transport(drug::DrugRBCTransport,
                                  plasma_conc::Float64,
                                  rbc_conc::Float64,
                                  transporters::RBCTransporterProfile)
    # Get transporter expression factor
    transporter_factor = get_transporter_expression(drug.primary_transporter, transporters)

    # Calculate active transport (if substrate)
    active_influx = 0.0
    saturation = 0.0

    if drug.is_substrate
        # Vmax adjusted for expression level
        vmax_adjusted = drug.vmax_relative * transporter_factor * 100.0  # Reference Vmax = 100 pmol/min

        # Influx (extracellular → intracellular)
        active_influx = calculate_michaelis_menten(plasma_conc, drug.km, vmax_adjusted)

        # Saturation fraction
        saturation = plasma_conc / (drug.km + plasma_conc)
    end

    # Calculate passive diffusion
    # Surface area in cm² (140 μm² = 1.4e-6 cm²)
    sa_cm2 = RBC_SURFACE_AREA * 1.0e-8
    passive_flux = calculate_passive_flux(plasma_conc, rbc_conc,
                                          drug.passive_permeability, sa_cm2)

    # Net flux
    net_flux = active_influx + passive_flux

    return Dict(
        "net_flux" => net_flux,
        "active_influx" => active_influx,
        "passive_flux" => passive_flux,
        "transporter_saturation" => saturation,
        "transporter_expression" => transporter_factor
    )
end

"""
    get_transporter_expression(transporter::Symbol, profile::RBCTransporterProfile)

Get relative expression level for a specific transporter.
"""
function get_transporter_expression(transporter::Symbol, profile::RBCTransporterProfile)
    if transporter == :ae1
        return profile.ae1
    elseif transporter == :glut1
        return profile.glut1
    elseif transporter == :ent1
        return profile.ent1
    elseif transporter == :ent2
        return profile.ent2
    elseif transporter == :mct1
        return profile.mct1
    elseif transporter == :urat1
        return profile.urat1
    elseif transporter == :oat_like
        return profile.oat_like
    elseif transporter == :oct_like
        return profile.oct_like
    elseif transporter == :passive
        return 1.0  # Passive doesn't depend on expression
    else
        return 1.0
    end
end

"""
    calculate_rbc_accumulation(drug::DrugRBCTransport,
                                plasma_conc::Float64,
                                transporters::RBCTransporterProfile;
                                time_hours::Float64 = 24.0)

Calculate steady-state RBC accumulation ratio.

# Returns
Dict with:
- `rbc_plasma_ratio`: RBC:plasma concentration ratio at steady state
- `time_to_steady_state`: Approximate time to 90% steady state (hours)
- `accumulation_mechanism`: Dominant mechanism
"""
function calculate_rbc_accumulation(drug::DrugRBCTransport,
                                     plasma_conc::Float64,
                                     transporters::RBCTransporterProfile;
                                     time_hours::Float64 = 24.0)
    # Simulate to steady state
    rbc_conc = 0.0
    dt = 0.01  # hours

    for t in 0:dt:time_hours
        transport = calculate_rbc_transport(drug, plasma_conc, rbc_conc, transporters)

        # Convert flux to concentration change
        # pmol/min/10^6 cells → μM change
        # Assuming 5×10^6 RBC/μL blood, 90 fL/cell
        d_conc = transport["net_flux"] * dt * 60.0 / RBC_VOLUME * 1e-3

        rbc_conc += d_conc

        # Prevent negative
        rbc_conc = max(rbc_conc, 0.0)
    end

    # Calculate ratio
    rbc_plasma_ratio = rbc_conc / max(plasma_conc, 1e-10)

    # Determine mechanism
    if drug.is_substrate && drug.passive_permeability < 1e-6
        mechanism = :active_transport
    elseif drug.passive_permeability > 1e-4
        mechanism = :passive_diffusion
    else
        mechanism = :mixed
    end

    # Estimate time to steady state (approximate)
    # Based on transport rate at 50% accumulation
    half_conc = rbc_conc / 2.0
    transport_50 = calculate_rbc_transport(drug, plasma_conc, half_conc, transporters)

    if transport_50["net_flux"] > 0
        t_half = (half_conc * RBC_VOLUME * 1e3) / (transport_50["net_flux"] * 60.0)
        t_90 = t_half * 3.32  # t_90% ≈ 3.32 × t_half
    else
        t_90 = Inf
    end

    return Dict(
        "rbc_plasma_ratio" => rbc_plasma_ratio,
        "rbc_concentration" => rbc_conc,
        "time_to_steady_state" => min(t_90, time_hours),
        "accumulation_mechanism" => mechanism
    )
end

"""
    get_rbc_transport_data(drug_name::String)

Get RBC transport data for a drug from database.
"""
function get_rbc_transport_data(drug_name::String)
    name_lower = lowercase(drug_name)
    if haskey(RBC_TRANSPORTER_SUBSTRATES, name_lower)
        return RBC_TRANSPORTER_SUBSTRATES[name_lower]
    end
    return nothing
end

# ============================================================================
# TRANSPORTER INHIBITION
# ============================================================================

"""
    apply_transporter_inhibition(transporters::RBCTransporterProfile,
                                  inhibitor::DrugRBCTransport,
                                  inhibitor_conc::Float64)

Calculate effective transporter activity with inhibitor present.

Uses competitive inhibition model:
v = Vmax * [S] / (Km * (1 + [I]/Ki) + [S])
"""
function apply_transporter_inhibition(transporters::RBCTransporterProfile,
                                       inhibitor::DrugRBCTransport,
                                       inhibitor_conc::Float64)
    if !inhibitor.is_inhibitor
        return transporters
    end

    # Calculate inhibition factor
    inhibition_factor = 1.0 / (1.0 + inhibitor_conc / inhibitor.inhibition_ki)

    # Apply to relevant transporter
    if inhibitor.primary_transporter == :ae1
        return RBCTransporterProfile(
            ae1 = transporters.ae1 * inhibition_factor,
            glut1 = transporters.glut1,
            ent1 = transporters.ent1,
            ent2 = transporters.ent2,
            mct1 = transporters.mct1,
            urat1 = transporters.urat1,
            oat_like = transporters.oat_like,
            oct_like = transporters.oct_like,
            condition = transporters.condition
        )
    elseif inhibitor.primary_transporter == :ent1
        return RBCTransporterProfile(
            ae1 = transporters.ae1,
            glut1 = transporters.glut1,
            ent1 = transporters.ent1 * inhibition_factor,
            ent2 = transporters.ent2,
            mct1 = transporters.mct1,
            urat1 = transporters.urat1,
            oat_like = transporters.oat_like,
            oct_like = transporters.oct_like,
            condition = transporters.condition
        )
    elseif inhibitor.primary_transporter == :oat_like
        return RBCTransporterProfile(
            ae1 = transporters.ae1,
            glut1 = transporters.glut1,
            ent1 = transporters.ent1,
            ent2 = transporters.ent2,
            mct1 = transporters.mct1,
            urat1 = transporters.urat1,
            oat_like = transporters.oat_like * inhibition_factor,
            oct_like = transporters.oct_like,
            condition = transporters.condition
        )
    elseif inhibitor.primary_transporter == :oct_like
        return RBCTransporterProfile(
            ae1 = transporters.ae1,
            glut1 = transporters.glut1,
            ent1 = transporters.ent1,
            ent2 = transporters.ent2,
            mct1 = transporters.mct1,
            urat1 = transporters.urat1,
            oat_like = transporters.oat_like,
            oct_like = transporters.oct_like * inhibition_factor,
            condition = transporters.condition
        )
    elseif inhibitor.primary_transporter == :urat1
        return RBCTransporterProfile(
            ae1 = transporters.ae1,
            glut1 = transporters.glut1,
            ent1 = transporters.ent1,
            ent2 = transporters.ent2,
            mct1 = transporters.mct1,
            urat1 = transporters.urat1 * inhibition_factor,
            oat_like = transporters.oat_like,
            oct_like = transporters.oct_like,
            condition = transporters.condition
        )
    else
        return transporters
    end
end

# ============================================================================
# DRUG INTERACTION PREDICTIONS
# ============================================================================

"""
    predict_ddi_rbc_transport(victim::DrugRBCTransport,
                               perpetrator::DrugRBCTransport,
                               perpetrator_conc::Float64,
                               transporters::RBCTransporterProfile)

Predict drug-drug interaction effect on RBC transport.

# Returns
Dict with:
- `auc_ratio`: Predicted AUC change in RBC
- `interaction_magnitude`: :none, :weak, :moderate, :strong
- `mechanism`: Interaction mechanism description
"""
function predict_ddi_rbc_transport(victim::DrugRBCTransport,
                                    perpetrator::DrugRBCTransport,
                                    perpetrator_conc::Float64,
                                    transporters::RBCTransporterProfile)
    # Same transporter?
    if victim.primary_transporter != perpetrator.primary_transporter
        return Dict(
            "auc_ratio" => 1.0,
            "interaction_magnitude" => :none,
            "mechanism" => "Different transporters - no interaction expected"
        )
    end

    # Competitive inhibition
    if perpetrator.is_inhibitor
        inhibited = apply_transporter_inhibition(transporters, perpetrator, perpetrator_conc)

        # Calculate accumulation with and without inhibitor
        acc_normal = calculate_rbc_accumulation(victim, 1.0, transporters)
        acc_inhibited = calculate_rbc_accumulation(victim, 1.0, inhibited)

        ratio = acc_inhibited["rbc_plasma_ratio"] / max(acc_normal["rbc_plasma_ratio"], 0.01)

        if ratio < 0.5
            magnitude = :strong
        elseif ratio < 0.8
            magnitude = :moderate
        elseif ratio < 0.95
            magnitude = :weak
        else
            magnitude = :none
        end

        return Dict(
            "auc_ratio" => ratio,
            "interaction_magnitude" => magnitude,
            "mechanism" => "Competitive inhibition at $(perpetrator.primary_transporter)"
        )
    end

    # Substrate competition (if both are substrates)
    if victim.is_substrate && perpetrator.is_substrate
        # Simple competitive model
        km_apparent = victim.km * (1.0 + perpetrator_conc / perpetrator.km)
        ratio = victim.km / km_apparent

        magnitude = ratio < 0.5 ? :moderate : (ratio < 0.8 ? :weak : :none)

        return Dict(
            "auc_ratio" => ratio,
            "interaction_magnitude" => magnitude,
            "mechanism" => "Substrate competition at $(victim.primary_transporter)"
        )
    end

    return Dict(
        "auc_ratio" => 1.0,
        "interaction_magnitude" => :none,
        "mechanism" => "No interaction mechanism identified"
    )
end

end # module RBCTransporters
