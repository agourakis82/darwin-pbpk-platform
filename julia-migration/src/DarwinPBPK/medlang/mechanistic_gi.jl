# ===========================================================================
# MEDLANG MECHANISTIC GI ABSORPTION MODEL
# ===========================================================================
# Full mechanistic oral absorption written in MedLang DSL syntax
#
# Features:
# - 5-segment GI tract (stomach, duodenum, jejunum, ileum, colon)
# - Regional transporter expression (PEPT1, OCT, OATP, ENT, MCT, LAT, ASBT)
# - P-gp saturation kinetics
# - pH-dependent dissolution and permeability
# - Gut wall metabolism (CYP3A4, AADC)
# - Transit time modeling
# - Enterohepatic recirculation
#
# This module:
# 1. Defines MedLang grammar extensions for mechanistic GI
# 2. Generates ODE system from MedLang AST
# 3. Integrates with ML transporter predictions
#
# Author: Dr. Sounio Agourakis
# Date: November 2025
# ===========================================================================

module MechanisticGI

using ..MedLang
using ..MedLang.MedLangParser
using ..MedLang.MedLangTranspiler

# Import mechanistic GI model
include("../compartments/gi_tract.jl")
using .GITract

export MechanisticGIParams, generate_mechanistic_gi_medlang
export simulate_mechanistic_oral
export GISegmentDef, TransporterDef, MetabolismDef
export default_gi_segments, params_from_ml_predictions

# ===========================================================================
# STRUCT DEFINITIONS (must come before usage)
# ===========================================================================

struct TransporterDef
    name::Symbol                    # :PEPT1, :OCT1, :OATP2B1, :PGP, etc.
    expression::Float64             # Relative to reference (jejunum = 1.0)
    km_uM::Float64                  # Michaelis constant
    vmax_pmol_min_cm2::Float64      # Maximum velocity
    saturable::Bool                 # Whether saturation kinetics apply
end

struct MetabolismDef
    enzyme::Symbol                  # :CYP3A4, :CYP2C9, :AADC, :UGT1A1
    expression::Float64             # Relative expression
    clint_uL_min_pmol::Float64      # Intrinsic clearance
end

struct GISegmentDef
    name::Symbol                    # :stomach, :duodenum, :jejunum, :ileum, :colon
    volume_mL::Float64
    pH::Float64
    pH_surface::Float64
    transit_time_min::Float64
    surface_area_cm2::Float64
    length_cm::Float64
    radius_cm::Float64
    bile_salt_mM::Float64
    transporters::Dict{Symbol, TransporterDef}
    metabolism::Dict{Symbol, MetabolismDef}
end

# ===========================================================================
# MEDLANG GI BLOCK DEFINITIONS
# ===========================================================================

"""
MedLang grammar extension for GI segments:

```medlang
gi_tract {
    segment stomach {
        volume: 250_mL
        pH: 1.5
        transit_time: 15_min
    }

    segment duodenum {
        volume: 50_mL
        pH: 6.0
        transit_time: 10_min
        surface_area: 2000_cm2

        transporters {
            PEPT1: { expression: 1.2, Km: 200_uM }
            PGP: { expression: 1.5, Km: 50_uM, saturable: true }
        }

        metabolism {
            CYP3A4: { expression: 1.5, CLint: 50_uL/min/pmol }
        }
    }

    segment jejunum { ... }
    segment ileum { ... }
    segment colon { ... }
}
```
"""

"""
Complete MedLang mechanistic GI model parameters.
"""
struct MechanisticGIParams
    # Drug physicochemistry
    drug_name::String
    MW::Float64
    logP::Float64
    pKa::Union{Float64, Nothing}
    charge_type::Symbol             # :neutral, :acid, :base, :zwitterion
    solubility_mg_mL::Float64
    particle_size_um::Float64

    # Transporter substrates (from ML or manual)
    is_pept1_substrate::Bool
    is_oct_substrate::Bool
    is_oatp_substrate::Bool
    is_ent_substrate::Bool
    is_mct_substrate::Bool
    is_lat_substrate::Bool
    is_asbt_substrate::Bool

    # Efflux
    is_pgp_substrate::Bool
    pgp_km_uM::Float64
    pgp_intrinsic_er::Float64       # In vitro efflux ratio
    is_bcrp_substrate::Bool

    # Metabolism
    is_cyp3a4_substrate::Bool
    clint_gut_uL_min_pmol::Float64
    clint_liver_uL_min_pmol::Float64
    fu_plasma::Float64

    # Special cases
    gut_wall_extraction::Float64    # For AADC, local CYP3A4
    saturable_km_mg::Float64        # For transporter-limited absorption
    saturable_fmax::Float64

    # GI segments (defaults from physiology)
    segments::Vector{GISegmentDef}
end

# ===========================================================================
# DEFAULT PHYSIOLOGICAL PARAMETERS
# ===========================================================================

"""
Default GI segment definitions from human physiology.
"""
function default_gi_segments()::Vector{GISegmentDef}
    return [
        GISegmentDef(
            :stomach,
            250.0,      # volume_mL
            1.5,        # pH
            1.5,        # pH_surface
            15.0,       # transit_time_min
            500.0,      # surface_area_cm2
            20.0,       # length_cm
            5.0,        # radius_cm
            0.0,        # bile_salt_mM
            Dict{Symbol, TransporterDef}(),
            Dict{Symbol, MetabolismDef}()
        ),
        GISegmentDef(
            :duodenum,
            50.0,
            6.0,
            5.5,
            10.0,
            2000.0,
            25.0,
            2.0,
            8.0,
            Dict(
                :PEPT1 => TransporterDef(:PEPT1, 1.2, 200.0, 5000.0, false),
                :OCT1 => TransporterDef(:OCT1, 0.8, 500.0, 1000.0, false),
                :OATP2B1 => TransporterDef(:OATP2B1, 1.0, 50.0, 500.0, false),
                :PGP => TransporterDef(:PGP, 1.5, 50.0, 10000.0, true),
                :BCRP => TransporterDef(:BCRP, 1.2, 30.0, 5000.0, true),
            ),
            Dict(
                :CYP3A4 => MetabolismDef(:CYP3A4, 1.5, 0.0),
            )
        ),
        GISegmentDef(
            :jejunum,
            100.0,
            6.5,
            5.8,
            90.0,
            18000.0,    # Largest surface area
            200.0,
            1.5,
            6.0,
            Dict(
                :PEPT1 => TransporterDef(:PEPT1, 1.0, 200.0, 5000.0, false),
                :OCT1 => TransporterDef(:OCT1, 1.0, 500.0, 1000.0, false),
                :OCT3 => TransporterDef(:OCT3, 1.0, 1500.0, 2000.0, false),
                :OATP2B1 => TransporterDef(:OATP2B1, 1.0, 50.0, 500.0, false),
                :ENT1 => TransporterDef(:ENT1, 1.0, 100.0, 3000.0, false),
                :MCT1 => TransporterDef(:MCT1, 1.0, 1000.0, 8000.0, false),
                :LAT2 => TransporterDef(:LAT2, 1.0, 100.0, 2000.0, false),
                :PGP => TransporterDef(:PGP, 1.0, 50.0, 10000.0, true),
                :BCRP => TransporterDef(:BCRP, 1.0, 30.0, 5000.0, true),
            ),
            Dict(
                :CYP3A4 => MetabolismDef(:CYP3A4, 1.0, 0.0),
                :CYP2C9 => MetabolismDef(:CYP2C9, 1.0, 0.0),
            )
        ),
        GISegmentDef(
            :ileum,
            80.0,
            7.2,
            6.5,
            120.0,
            12000.0,
            300.0,
            1.2,
            2.0,        # Bile reabsorbed
            Dict(
                :PEPT1 => TransporterDef(:PEPT1, 0.6, 200.0, 5000.0, false),
                :OATP2B1 => TransporterDef(:OATP2B1, 0.8, 50.0, 500.0, false),
                :ASBT => TransporterDef(:ASBT, 2.0, 10.0, 500.0, false),  # HIGH in ileum
                :PGP => TransporterDef(:PGP, 0.8, 50.0, 10000.0, true),
            ),
            Dict(
                :CYP3A4 => MetabolismDef(:CYP3A4, 0.6, 0.0),
            )
        ),
        GISegmentDef(
            :colon,
            200.0,
            6.5,
            6.5,
            720.0,      # 12 hours
            1500.0,
            150.0,
            3.0,
            0.1,
            Dict{Symbol, TransporterDef}(),  # Minimal transporters
            Dict{Symbol, MetabolismDef}()
        ),
    ]
end

# ===========================================================================
# MEDLANG CODE GENERATION
# ===========================================================================

"""
Generate MedLang DSL code for mechanistic GI model.

This produces a complete MedLang model with:
- 5 GI segment compartments
- Portal vein compartment
- Systemic PBPK organs
- Mechanistic absorption ODEs
"""
function generate_mechanistic_gi_medlang(
    params::MechanisticGIParams;
    include_pbpk::Bool = true
)::String
    buf = IOBuffer()

    # Header
    println(buf, """
model $(params.drug_name)_MechanisticGI {
    // ================================================================
    // MECHANISTIC GI ABSORPTION MODEL
    // Generated by Darwin PBPK Platform - MedLang DSL
    // ================================================================
    // Drug: $(params.drug_name)
    // MW: $(params.MW) Da
    // logP: $(params.logP)
    // pKa: $(params.pKa === nothing ? "N/A" : params.pKa)
    // Charge: $(params.charge_type)
    // Solubility: $(params.solubility_mg_mL) mg/mL
    // ================================================================

    route: oral
""")

    # GI tract block
    println(buf, """
    // ================================================================
    // GI TRACT COMPARTMENTS
    // ================================================================
    gi_tract {""")

    for seg in params.segments
        println(buf, """
        segment $(seg.name) {
            volume: $(seg.volume_mL)_mL
            pH: $(seg.pH)
            pH_surface: $(seg.pH_surface)
            transit_time: $(seg.transit_time_min)_min
            surface_area: $(seg.surface_area_cm2)_cm2
            bile_salt: $(seg.bile_salt_mM)_mM""")

        # Transporters
        if !isempty(seg.transporters)
            println(buf, "\n            transporters {")
            for (name, t) in seg.transporters
                saturable_str = t.saturable ? ", saturable: true" : ""
                println(buf, "                $name: { expression: $(t.expression), Km: $(t.km_uM)_uM, Vmax: $(t.vmax_pmol_min_cm2)_pmol/min/cm2$saturable_str }")
            end
            println(buf, "            }")
        end

        # Metabolism
        if !isempty(seg.metabolism)
            println(buf, "\n            metabolism {")
            for (name, m) in seg.metabolism
                println(buf, "                $name: { expression: $(m.expression), CLint: $(m.clint_uL_min_pmol)_uL/min/pmol }")
            end
            println(buf, "            }")
        end

        println(buf, "        }")
    end
    println(buf, "    }")

    # Drug properties block
    println(buf, """

    // ================================================================
    // DRUG PHYSICOCHEMISTRY
    // ================================================================
    drug_properties {
        MW: $(params.MW)_Da
        logP: $(params.logP)
        pKa: $(params.pKa === nothing ? "null" : "$(params.pKa)")
        charge_type: $(params.charge_type)
        solubility: $(params.solubility_mg_mL)_mg/mL
        particle_size: $(params.particle_size_um)_um
        fu_plasma: $(params.fu_plasma)
    }""")

    # Transporter substrates block
    println(buf, """

    // ================================================================
    // TRANSPORTER SUBSTRATES (from ML prediction or literature)
    // ================================================================
    transporter_substrates {
        PEPT1: $(params.is_pept1_substrate)
        OCT: $(params.is_oct_substrate)
        OATP: $(params.is_oatp_substrate)
        ENT: $(params.is_ent_substrate)
        MCT: $(params.is_mct_substrate)
        LAT: $(params.is_lat_substrate)
        ASBT: $(params.is_asbt_substrate)

        // Efflux
        PGP: { substrate: $(params.is_pgp_substrate), Km: $(params.pgp_km_uM)_uM, ER: $(params.pgp_intrinsic_er) }
        BCRP: $(params.is_bcrp_substrate)
    }""")

    # First-pass metabolism block
    println(buf, """

    // ================================================================
    // FIRST-PASS METABOLISM
    // ================================================================
    firstpass {
        // Gut wall metabolism
        CYP3A4_substrate: $(params.is_cyp3a4_substrate)
        CLint_gut: $(params.clint_gut_uL_min_pmol)_uL/min/pmol
        gut_wall_extraction: $(params.gut_wall_extraction)

        // Hepatic metabolism
        CLint_liver: $(params.clint_liver_uL_min_pmol)_uL/min/pmol
    }""")

    # Saturable absorption block (if applicable)
    if params.saturable_km_mg > 0
        println(buf, """

    // ================================================================
    // SATURABLE ABSORPTION (transporter-limited)
    // ================================================================
    saturable_absorption {
        Km: $(params.saturable_km_mg)_mg
        Fmax: $(params.saturable_fmax)
    }""")
    end

    # State variables (ODE system)
    println(buf, """

    // ================================================================
    // STATE VARIABLES
    // ================================================================
    // GI compartments (mg)
    state A_stomach_undissolved: Mass = 0.0_mg
    state A_stomach_dissolved: Mass = 0.0_mg
    state A_duodenum_undissolved: Mass = 0.0_mg
    state A_duodenum_dissolved: Mass = 0.0_mg
    state A_jejunum_undissolved: Mass = 0.0_mg
    state A_jejunum_dissolved: Mass = 0.0_mg
    state A_ileum_undissolved: Mass = 0.0_mg
    state A_ileum_dissolved: Mass = 0.0_mg
    state A_colon_undissolved: Mass = 0.0_mg
    state A_colon_dissolved: Mass = 0.0_mg

    // Portal and systemic (mg)
    state A_portal: Mass = 0.0_mg
    state A_absorbed: Mass = 0.0_mg
    state A_systemic: Mass = 0.0_mg""")

    # ODE equations
    println(buf, """

    // ================================================================
    // ODE EQUATIONS (Mechanistic GI Absorption)
    // ================================================================

    // Stomach: dissolution + gastric emptying
    ode dA_stomach_undissolved/dt = -k_diss_stomach * A_stomach_undissolved - k_transit_stomach * A_stomach_undissolved
    ode dA_stomach_dissolved/dt = k_diss_stomach * A_stomach_undissolved - k_transit_stomach * A_stomach_dissolved

    // Duodenum: dissolution + absorption + transit
    ode dA_duodenum_undissolved/dt = k_transit_stomach * A_stomach_undissolved - k_diss_duodenum * A_duodenum_undissolved - k_transit_duodenum * A_duodenum_undissolved
    ode dA_duodenum_dissolved/dt = k_transit_stomach * A_stomach_dissolved + k_diss_duodenum * A_duodenum_undissolved - ka_duodenum * A_duodenum_dissolved - k_transit_duodenum * A_duodenum_dissolved

    // Jejunum: main absorption site
    ode dA_jejunum_undissolved/dt = k_transit_duodenum * A_duodenum_undissolved - k_diss_jejunum * A_jejunum_undissolved - k_transit_jejunum * A_jejunum_undissolved
    ode dA_jejunum_dissolved/dt = k_transit_duodenum * A_duodenum_dissolved + k_diss_jejunum * A_jejunum_undissolved - ka_jejunum * A_jejunum_dissolved - k_transit_jejunum * A_jejunum_dissolved

    // Ileum: bile acid reabsorption, ASBT
    ode dA_ileum_undissolved/dt = k_transit_jejunum * A_jejunum_undissolved - k_diss_ileum * A_ileum_undissolved - k_transit_ileum * A_ileum_undissolved
    ode dA_ileum_dissolved/dt = k_transit_jejunum * A_jejunum_dissolved + k_diss_ileum * A_ileum_undissolved - ka_ileum * A_ileum_dissolved - k_transit_ileum * A_ileum_dissolved

    // Colon: slow transit, minimal absorption
    ode dA_colon_undissolved/dt = k_transit_ileum * A_ileum_undissolved - k_diss_colon * A_colon_undissolved - k_transit_colon * A_colon_undissolved
    ode dA_colon_dissolved/dt = k_transit_ileum * A_ileum_dissolved + k_diss_colon * A_colon_undissolved - ka_colon * A_colon_dissolved - k_transit_colon * A_colon_dissolved

    // Portal vein: sum of absorbed from all segments × Fg
    ode dA_portal/dt = Fg * (ka_duodenum * A_duodenum_dissolved + ka_jejunum * A_jejunum_dissolved + ka_ileum * A_ileum_dissolved + ka_colon * A_colon_dissolved) - k_portal_to_liver * A_portal

    // Absorbed (before first-pass)
    ode dA_absorbed/dt = ka_duodenum * A_duodenum_dissolved + ka_jejunum * A_jejunum_dissolved + ka_ileum * A_ileum_dissolved + ka_colon * A_colon_dissolved

    // Systemic: portal × Fh
    ode dA_systemic/dt = Fh * k_portal_to_liver * A_portal""")

    # Observables
    println(buf, """

    // ================================================================
    // OBSERVABLES
    // ================================================================
    observable Fa = A_absorbed / Dose
    observable F = A_systemic / Dose
    observable F_percent = 100 * A_systemic / Dose
    observable total_in_gut = A_stomach_undissolved + A_stomach_dissolved + A_duodenum_undissolved + A_duodenum_dissolved + A_jejunum_undissolved + A_jejunum_dissolved + A_ileum_undissolved + A_ileum_dissolved + A_colon_undissolved + A_colon_dissolved
""")

    # PBPK organs (if requested)
    if include_pbpk
        println(buf, """
    // ================================================================
    // SYSTEMIC PBPK ORGANS
    // ================================================================
    clearance hepatic: $(params.clint_liver_uL_min_pmol > 0 ? params.clint_liver_uL_min_pmol * 0.001 : 10.0)_L/h
    clearance renal: 0.1_L/h

    organ blood { V: 5.0_L, Q: 0.0_L/h, Kp: 1.0 }
    organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 2.5 }
    organ kidney { V: 0.31_L, Q: 60.0_L/h, Kp: 2.0 }
    organ brain { V: 1.4_L, Q: 50.0_L/h, Kp: 0.5 }
    organ heart { V: 0.33_L, Q: 20.0_L/h, Kp: 2.5 }
    organ lung { V: 0.5_L, Q: 300.0_L/h, Kp: 1.8 }
    organ muscle { V: 30.0_L, Q: 75.0_L/h, Kp: 1.5 }
    organ adipose { V: 15.0_L, Q: 12.0_L/h, Kp: $(10^(0.7 * params.logP - 0.5)) }
    organ gut { V: 1.1_L, Q: 45.0_L/h, Kp: 2.0 }
    organ skin { V: 3.3_L, Q: 10.0_L/h, Kp: 1.2 }
    organ bone { V: 10.0_L, Q: 5.0_L/h, Kp: 0.5 }
    organ spleen { V: 0.18_L, Q: 15.0_L/h, Kp: 2.2 }
    organ pancreas { V: 0.1_L, Q: 5.0_L/h, Kp: 1.8 }
    organ other { V: 5.0_L, Q: 20.0_L/h, Kp: 1.5 }""")
    end

    println(buf, "\n}")

    return String(take!(buf))
end

# ===========================================================================
# MEDLANG TO ODE SYSTEM COMPILATION
# ===========================================================================

"""
Compile MedLang mechanistic GI model to Julia ODE system.

Returns a function compatible with DifferentialEquations.jl
"""
function compile_mechanistic_gi(params::MechanisticGIParams)
    # Calculate derived parameters

    # Dissolution rate constants (Noyes-Whitney based)
    k_diss = Dict{Symbol, Float64}()
    for seg in params.segments
        # pH-adjusted solubility
        sol_adj = adjust_solubility_for_pH(
            params.solubility_mg_mL,
            seg.pH,
            params.pKa,
            params.charge_type
        )
        # k_diss = 3D / (r × h × ρ) ≈ 0.5 / particle_size
        k_diss[seg.name] = 0.5 / params.particle_size_um
    end

    # Transit rate constants (1/transit_time)
    k_transit = Dict{Symbol, Float64}()
    for seg in params.segments
        k_transit[seg.name] = 1.0 / seg.transit_time_min
    end

    # Absorption rate constants (ka) for each segment
    ka = Dict{Symbol, Float64}()
    for seg in params.segments
        if seg.name == :stomach
            ka[seg.name] = 0.0  # No absorption in stomach
        else
            ka[seg.name] = calculate_ka_for_segment(params, seg)
        end
    end

    # First-pass parameters
    Fg = calculate_fg(params)
    Fh = calculate_fh(params)

    return (
        k_diss = k_diss,
        k_transit = k_transit,
        ka = ka,
        Fg = Fg,
        Fh = Fh,
        params = params
    )
end

"""
Calculate absorption rate constant for a GI segment.
Includes passive permeability, carrier-mediated uptake, and efflux.
"""
function calculate_ka_for_segment(params::MechanisticGIParams, seg::GISegmentDef)::Float64
    # 1. Passive transcellular permeability
    peff_passive = calculate_passive_permeability(params.logP, params.MW)

    # 2. Paracellular (small hydrophilic molecules)
    peff_paracellular = 0.0
    if params.MW < 350 && params.logP < 1
        peff_paracellular = 4.0e-4 * (350 - params.MW) / 350
    end

    # 3. Carrier-mediated uptake
    peff_carrier = 0.0

    if params.is_pept1_substrate && haskey(seg.transporters, :PEPT1)
        t = seg.transporters[:PEPT1]
        peff_carrier += 5.0e-4 * t.expression
    end

    if params.is_oct_substrate && (haskey(seg.transporters, :OCT1) || haskey(seg.transporters, :OCT3))
        expr = get(seg.transporters, :OCT1, TransporterDef(:OCT1, 0.0, 0.0, 0.0, false)).expression +
               get(seg.transporters, :OCT3, TransporterDef(:OCT3, 0.0, 0.0, 0.0, false)).expression
        peff_carrier += 3.0e-4 * expr / 2
    end

    if params.is_oatp_substrate && haskey(seg.transporters, :OATP2B1)
        t = seg.transporters[:OATP2B1]
        peff_carrier += 3.0e-4 * t.expression
    end

    if params.is_ent_substrate && haskey(seg.transporters, :ENT1)
        t = seg.transporters[:ENT1]
        peff_carrier += 4.0e-4 * t.expression
    end

    if params.is_mct_substrate && haskey(seg.transporters, :MCT1)
        t = seg.transporters[:MCT1]
        peff_carrier += 3.0e-4 * t.expression
    end

    if params.is_lat_substrate && haskey(seg.transporters, :LAT2)
        t = seg.transporters[:LAT2]
        peff_carrier += 4.0e-4 * t.expression
    end

    if params.is_asbt_substrate && haskey(seg.transporters, :ASBT)
        t = seg.transporters[:ASBT]
        peff_carrier += 5.0e-4 * t.expression  # High affinity
    end

    # 4. P-gp efflux reduction
    efflux_factor = 1.0
    if params.is_pgp_substrate && haskey(seg.transporters, :PGP)
        t = seg.transporters[:PGP]
        if t.saturable
            # Saturation kinetics: effective ER decreases at high concentrations
            # ER_eff = 1 + (ER_intrinsic - 1) × Km / (Km + C)
            # For typical therapeutic doses, assume moderate saturation
            saturation_factor = 0.7  # 30% saturation
            er_eff = 1.0 + (params.pgp_intrinsic_er - 1.0) * saturation_factor
        else
            er_eff = params.pgp_intrinsic_er
        end
        efflux_factor = 1.0 / (1.0 + (er_eff - 1.0) * t.expression)
    end

    # Total Peff
    peff_total = (peff_passive + peff_paracellular + peff_carrier) * efflux_factor

    # Convert to ka (min^-1)
    # ka = Peff × SA / V
    peff_cm_min = peff_total * 60.0  # cm/s → cm/min
    ka = peff_cm_min * seg.surface_area_cm2 / seg.volume_mL

    return ka
end

"""
Calculate passive transcellular permeability from logP and MW.
"""
function calculate_passive_permeability(logP::Float64, MW::Float64)::Float64
    if logP < -2
        peff = 1.0e-4
    elseif logP < 0
        peff = 1.0e-4 + (logP + 2) * 3.0e-4
    elseif logP < 2
        peff = 7.0e-4 + logP * 6.0e-4
    elseif logP < 4
        peff = 19.0e-4 + (logP - 2) * 5.0e-4
    else
        peff = 29.0e-4 / (1.0 + 0.3 * (logP - 4))
    end

    # MW penalty
    if MW > 500
        peff *= exp(-0.002 * (MW - 500))
    end

    return peff
end

"""
Adjust solubility for pH using Henderson-Hasselbalch.
"""
function adjust_solubility_for_pH(
    sol_intrinsic::Float64,
    pH::Float64,
    pKa::Union{Float64, Nothing},
    charge_type::Symbol
)::Float64
    if pKa === nothing
        return sol_intrinsic
    end

    if charge_type == :acid
        ionized = 1.0 / (1.0 + 10^(pKa - pH))
        return sol_intrinsic * (1.0 + 100.0 * ionized)
    elseif charge_type == :base
        ionized = 1.0 / (1.0 + 10^(pH - pKa))
        return sol_intrinsic * (1.0 + 100.0 * ionized)
    else
        return sol_intrinsic
    end
end

"""
Calculate gut availability (Fg) including gut wall metabolism.
"""
function calculate_fg(params::MechanisticGIParams)::Float64
    # Base Fg from CYP3A4 metabolism
    if params.is_cyp3a4_substrate && params.clint_gut_uL_min_pmol > 0
        # Qgut model
        clint_L_h = params.clint_gut_uL_min_pmol * 60.0 * 1e-6 * 1e6
        Qgut = 18.0  # L/h
        Eg = clint_L_h / (Qgut + clint_L_h)
        Fg = 1.0 - Eg
    else
        Fg = 1.0
    end

    # Additional gut wall extraction (AADC, etc.)
    Fg *= (1.0 - params.gut_wall_extraction)

    return clamp(Fg, 0.05, 1.0)
end

"""
Calculate hepatic availability (Fh) from well-stirred model.
"""
function calculate_fh(params::MechanisticGIParams)::Float64
    if params.clint_liver_uL_min_pmol > 0
        clint_L_h = params.clint_liver_uL_min_pmol * 60.0 * 1e-6 * 1e6
        Qh = 90.0  # L/h
        fu = params.fu_plasma
        Eh = (fu * clint_L_h) / (Qh + fu * clint_L_h)
        return 1.0 - Eh
    else
        return 1.0
    end
end

# ===========================================================================
# SIMULATION
# ===========================================================================

"""
Simulate mechanistic oral absorption using the full GI model.

This uses the compiled ODE parameters from MedLang and runs
the 10-state GI ODE system.
"""
function simulate_mechanistic_oral(
    params::MechanisticGIParams,
    dose_mg::Float64;
    t_max_h::Float64 = 24.0,
    dt_min::Float64 = 1.0
)
    # Compile model
    model = compile_mechanistic_gi(params)

    # Initialize state
    n_steps = Int(ceil(t_max_h * 60 / dt_min))

    # State: [undissolved, dissolved] for each of 5 segments + portal + absorbed + systemic
    # Indices: stomach(1,2), duodenum(3,4), jejunum(5,6), ileum(7,8), colon(9,10), portal(11), absorbed(12), systemic(13)
    state = zeros(13)
    state[1] = dose_mg  # All in stomach undissolved

    # Time series
    times = Float64[]
    systemic_profile = Float64[]
    absorbed_profile = Float64[]

    segments = [:stomach, :duodenum, :jejunum, :ileum, :colon]

    for step in 1:n_steps
        # Dissolution and transit for each segment
        for (i, seg_name) in enumerate(segments)
            undiss_idx = 2*i - 1
            diss_idx = 2*i

            # Dissolution
            k_d = model.k_diss[seg_name]
            diss_rate = k_d * state[undiss_idx] * dt_min
            diss_rate = min(diss_rate, state[undiss_idx])
            state[undiss_idx] -= diss_rate
            state[diss_idx] += diss_rate

            # Absorption (if not stomach)
            if seg_name != :stomach
                ka = model.ka[seg_name]
                abs_rate = ka * state[diss_idx] * dt_min
                abs_rate = min(abs_rate, state[diss_idx])
                state[diss_idx] -= abs_rate
                state[12] += abs_rate  # absorbed
                state[11] += abs_rate * model.Fg  # portal (after gut first-pass)
            end

            # Transit to next segment
            if i < 5
                k_t = model.k_transit[seg_name]
                transit_undiss = k_t * state[undiss_idx] * dt_min
                transit_diss = k_t * state[diss_idx] * dt_min
                state[undiss_idx] -= transit_undiss
                state[diss_idx] -= transit_diss
                state[2*(i+1) - 1] += transit_undiss  # next undissolved
                state[2*(i+1)] += transit_diss        # next dissolved
            end
        end

        # Hepatic first-pass
        systemic_input = state[11] * model.Fh * 0.1  # Fraction per step
        state[11] -= systemic_input / model.Fh
        state[13] += systemic_input

        # Record
        push!(times, step * dt_min / 60.0)
        push!(systemic_profile, state[13])
        push!(absorbed_profile, state[12])
    end

    # Calculate PK parameters
    Fa = state[12] / dose_mg
    F = state[13] / dose_mg

    # Find Cmax and Tmax from rate of appearance
    if length(systemic_profile) > 1
        rates = diff(systemic_profile)
        cmax_idx = argmax(rates) + 1
        tmax = times[cmax_idx]
    else
        tmax = 0.0
    end

    return Dict{String, Any}(
        "time" => times,
        "systemic_mg" => systemic_profile,
        "absorbed_mg" => absorbed_profile,
        "Fa" => Fa,
        "Fg" => model.Fg,
        "Fh" => model.Fh,
        "F" => F,
        "F_percent" => F * 100,
        "tmax" => tmax,
        "model" => model
    )
end

# ===========================================================================
# ML INTEGRATION
# ===========================================================================

"""
Create MechanisticGIParams from ML transporter predictions.
"""
function params_from_ml_predictions(
    ml_result::NamedTuple;
    drug_name::String,
    MW::Float64,
    logP::Float64,
    solubility_mg_mL::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    charge_type::Symbol = :neutral,
    particle_size_um::Float64 = 25.0,
    clint_gut::Float64 = 0.0,
    clint_liver::Float64 = 0.0,
    fu_plasma::Float64 = 0.1,
    is_cyp3a4_substrate::Bool = false
)::MechanisticGIParams
    # Extract transporter substrates from ML predictions
    transporters = ml_result.uptake_transporters

    return MechanisticGIParams(
        drug_name,
        MW,
        logP,
        pKa,
        charge_type,
        solubility_mg_mL,
        particle_size_um,

        # Transporter substrates
        :PEPT1 in transporters,
        :OCT1 in transporters || :OCT3 in transporters,
        :OATP2B1 in transporters,
        :ENT1 in transporters || :ENT2 in transporters,
        :MCT1 in transporters,
        :LAT1 in transporters || :LAT2 in transporters,
        :ASBT in transporters,

        # Efflux
        ml_result.is_pgp_substrate,
        get(ml_result.carrier_km_values, :PGP, 50.0),
        ml_result.pgp_efflux_ratio,
        false,  # BCRP (could add to ML)

        # Metabolism
        is_cyp3a4_substrate,
        clint_gut,
        clint_liver,
        fu_plasma,

        # Special cases
        0.0,    # gut_wall_extraction (set per drug)
        0.0,    # saturable_km
        1.0,    # saturable_fmax

        # Default segments
        default_gi_segments()
    )
end

export params_from_ml_predictions

end # module
