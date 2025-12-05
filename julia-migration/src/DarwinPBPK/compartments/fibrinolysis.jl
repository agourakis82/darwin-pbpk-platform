"""
Fibrinolysis Module - Complete Model of Clot Dissolution

Models the fibrinolytic system with:
- Plasminogen/Plasmin system
- Tissue plasminogen activator (tPA) and urokinase (uPA)
- Inhibitors: PAI-1, α2-antiplasmin, α2-macroglobulin
- Fibrin degradation products (FDPs, D-dimer)
- Antifibrinolytic drugs (tranexamic acid, EACA)
- Integration with coagulation cascade

Based on SOTA Q1 literature:
- Zhalyalov et al. - 1D reaction-diffusion model
- Nature Scientific Reports 2017 - Multiscale fibrinolysis model
- PMC8694003 - Mathematical models predicting fibrinolytic outcomes

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

module Fibrinolysis

using LinearAlgebra
using Statistics

export FibrinolyticSystem, PlasminogenState, FibrinDegradation
export create_fibrinolytic_system, simulate_fibrinolysis!
export apply_tpa_therapy!, apply_antifibrinolytic!
export calculate_d_dimer, calculate_lysis_time
export get_fibrinolysis_state, plasmin_generation_assay
export NORMAL_PLASMINOGEN, NORMAL_TPA, NORMAL_PAI1

# ============================================================================
# CONSTANTS - From SOTA Literature
# ============================================================================

# Plasma concentrations (normal values)
const NORMAL_PLASMINOGEN = 2000.0        # nM (~200 μg/mL, 2 μM)
const NORMAL_TPA = 0.07                  # nM (~5 ng/mL, 70 pM)
const NORMAL_UPA = 0.02                  # nM (~2 ng/mL)
const NORMAL_PAI1 = 0.4                  # nM (~20 ng/mL, active form)
const NORMAL_ALPHA2_ANTIPLASMIN = 1000.0 # nM (1 μM, ~70 μg/mL)
const NORMAL_ALPHA2_MACROGLOBULIN = 3000.0  # nM (3 μM)
const NORMAL_TAFI = 75.0                 # nM (Thrombin-activatable fibrinolysis inhibitor)

# Half-lives
const HALFLIFE_TPA = 4.0                 # minutes (2-6 min)
const HALFLIFE_PLASMIN = 0.3             # seconds (very short, immediately inhibited)
const HALFLIFE_PAI1_ACTIVE = 30.0        # minutes (stabilized by vitronectin: 60-90 min)
const HALFLIFE_D_DIMER = 480.0           # minutes (8 hours)

# Kinetic parameters
const KINETIC_FIBRINOLYSIS = Dict(
    # tPA activation of plasminogen
    "kcat_tPA_plg" => 0.05,              # s⁻¹ (on fibrin surface)
    "Km_tPA_plg" => 160.0,               # nM
    "kcat_tPA_plg_fibrin" => 0.5,        # s⁻¹ (1000× enhanced on fibrin)
    "Km_tPA_plg_fibrin" => 160.0,        # nM

    # uPA activation of plasminogen
    "kcat_uPA_plg" => 0.02,              # s⁻¹
    "Km_uPA_plg" => 2000.0,              # nM

    # Plasmin degradation of fibrin
    "kcat_plasmin_fibrin" => 1.0,        # s⁻¹
    "Km_plasmin_fibrin" => 500.0,        # nM

    # Inhibition rate constants (second-order, M⁻¹s⁻¹)
    "k_PAI1_tPA" => 2.7e7,               # Very fast (PAI-1 inhibits tPA)
    "k_PAI1_uPA" => 2.0e7,               # Fast
    "k_A2AP_plasmin" => 3.8e7,           # Very fast (α2-antiplasmin inhibits plasmin)
    "k_A2M_plasmin" => 3.0e5,            # Slower (α2-macroglobulin)

    # TAFI inhibition of fibrinolysis
    "k_TAFIa_fibrin" => 0.01,            # s⁻¹ (removes C-terminal lysines)

    # tPA/plasminogen binding to fibrin
    "Kd_tPA_fibrin" => 20.0,             # nM (high affinity)
    "Kd_plg_fibrin" => 30.0              # nM
)

# Antifibrinolytic drug parameters
const ANTIFIBRINOLYTIC_PARAMS = Dict(
    "tranexamic_acid" => Dict(
        "Ki_plasminogen_high" => 1.1e3,   # nM (1.1 μM, high affinity site)
        "Ki_plasminogen_low" => 750e3,    # nM (0.75 mM, medium affinity)
        "Ki_uPA" => 2e6,                  # nM (2 mM)
        "EC50_fibrinolysis" => 43e3,      # nM (~6.8 mg/L = 43 μM)
        "EC95_fibrinolysis" => 72e3       # nM (~11.3 mg/L = 72 μM)
    ),
    "aminocaproic_acid" => Dict(
        "Ki_plasminogen" => 1e6,          # nM (1 mM, ~10× less potent than TXA)
        "EC50_fibrinolysis" => 760e3      # nM (~100 mg/L = 760 μM)
    ),
    "aprotinin" => Dict(
        "Ki_plasmin" => 0.1,              # nM (very potent direct plasmin inhibitor)
        "Ki_kallikrein" => 30.0           # nM
    )
)

# Thrombolytic drug parameters
const THROMBOLYTIC_PARAMS = Dict(
    "alteplase" => Dict(                  # Recombinant tPA
        "specific_activity" => 580000.0,  # IU/mg
        "fibrin_selectivity" => 1.0,      # Reference
        "halflife_min" => 4.0
    ),
    "tenecteplase" => Dict(               # Modified tPA
        "specific_activity" => 200.0,     # MU/mg
        "fibrin_selectivity" => 14.0,     # 14× more fibrin-selective
        "halflife_min" => 20.0            # Longer half-life
    ),
    "reteplase" => Dict(
        "specific_activity" => 0.56,      # MU/mg
        "fibrin_selectivity" => 0.2,      # Less selective
        "halflife_min" => 15.0
    ),
    "streptokinase" => Dict(
        "specific_activity" => 100000.0,  # IU/mg
        "fibrin_selectivity" => 0.0,      # Non-selective
        "halflife_min" => 23.0,
        "antigenic" => true               # Can cause allergic reactions
    )
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
PlasminogenState - Plasminogen/Plasmin system state
"""
mutable struct PlasminogenState
    # Plasminogen forms
    glu_plasminogen::Float64      # Native form (nM)
    lys_plasminogen::Float64      # Partially degraded, more activatable (nM)

    # Plasmin
    plasmin_free::Float64         # Free active plasmin (nM, very low normally)
    plasmin_fibrin_bound::Float64 # Fibrin-bound plasmin (nM)

    # Plasminogen activators
    tpa_free::Float64             # Free tPA (nM)
    tpa_fibrin_bound::Float64     # Fibrin-bound tPA (nM)
    upa_free::Float64             # Free uPA (nM)

    # Inhibitors
    pai1_active::Float64          # Active PAI-1 (nM)
    pai1_latent::Float64          # Latent/inactive PAI-1 (nM)
    alpha2_antiplasmin::Float64   # α2-AP (nM)
    alpha2_macroglobulin::Float64 # α2-M (nM)
    tafi::Float64                 # TAFI (nM)
    tafi_active::Float64          # TAFIa (activated by thrombin-TM)

    # Complexes (inactive)
    pai1_tpa_complex::Float64     # PAI-1·tPA (nM)
    pai1_upa_complex::Float64     # PAI-1·uPA (nM)
    a2ap_plasmin_complex::Float64 # α2-AP·plasmin (PAP) (nM)
end

"""
FibrinDegradation - Fibrin and degradation products
"""
mutable struct FibrinDegradation
    # Fibrin states
    fibrin_intact::Float64        # Intact cross-linked fibrin (nM)
    fibrin_partially_degraded::Float64  # Partially lysed (nM)

    # Fibrin degradation products
    fdp_large::Float64            # Large FDPs (X, Y fragments) (nM)
    fdp_small::Float64            # Small FDPs (D, E fragments) (nM)
    d_dimer::Float64              # D-dimer (cross-linked) (nM)

    # Fibrin properties
    fibrin_density::Float64       # Fiber density (fibers/μm²)
    fiber_diameter::Float64       # Average fiber diameter (nm)
    clot_permeability::Float64    # Darcy permeability (μm²)
end

"""
FibrinolyticSystem - Complete fibrinolysis model
"""
mutable struct FibrinolyticSystem
    # Plasminogen/plasmin state
    plasminogen::PlasminogenState

    # Fibrin state
    fibrin::FibrinDegradation

    # Thrombolytic therapy
    thrombolytic_drug::String     # "none", "alteplase", "tenecteplase", etc.
    thrombolytic_conc::Float64    # Drug concentration (IU/mL or nM)

    # Antifibrinolytic therapy
    antifibrinolytic_drug::String # "none", "tranexamic_acid", etc.
    antifibrinolytic_conc::Float64 # Drug concentration (nM)
    fibrinolysis_inhibition::Float64 # Fraction inhibited (0-1)

    # Clinical state
    hyperfibrinolysis::Bool       # Pathological state
    dic_score::Float64            # DIC-related score

    # Calculated parameters (plasmin generation assay)
    lag_time::Float64             # Time to 6 nM plasmin (min)
    peak_plasmin::Float64         # Peak plasmin concentration (nM)
    time_to_peak::Float64         # Time to peak (min)
    epp::Float64                  # Endogenous plasmin potential (nM·min)
    lysis_time::Float64           # Time for 50% clot lysis (min)
end

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

"""
create_plasminogen_state(; pai1_elevated=false)

Create initial plasminogen/plasmin system state.
"""
function create_plasminogen_state(;
    pai1_elevated::Bool=false,
    pai1_multiplier::Float64=1.0
)::PlasminogenState

    pai1 = NORMAL_PAI1 * (pai1_elevated ? 5.0 : 1.0) * pai1_multiplier

    return PlasminogenState(
        NORMAL_PLASMINOGEN * 0.95,  # 95% Glu-plasminogen
        NORMAL_PLASMINOGEN * 0.05,  # 5% Lys-plasminogen
        0.0,                        # plasmin_free (essentially zero)
        0.0,                        # plasmin_fibrin_bound
        NORMAL_TPA,                 # tpa_free
        0.0,                        # tpa_fibrin_bound
        NORMAL_UPA,                 # upa_free
        pai1,                       # pai1_active
        pai1 * 10.0,               # pai1_latent (10× more)
        NORMAL_ALPHA2_ANTIPLASMIN,
        NORMAL_ALPHA2_MACROGLOBULIN,
        NORMAL_TAFI,
        0.0,                        # tafi_active
        0.0,                        # pai1_tpa_complex
        0.0,                        # pai1_upa_complex
        0.0                         # a2ap_plasmin_complex
    )
end

"""
create_fibrin_degradation(fibrin_amount; clot_structure=:normal)

Create initial fibrin state.
"""
function create_fibrin_degradation(
    fibrin_amount::Float64;
    clot_structure::Symbol=:normal
)::FibrinDegradation

    # Clot structure affects lysis rate
    density, diameter, permeability = if clot_structure == :dense
        (30.0, 50.0, 0.01)   # Dense clot: slower lysis
    elseif clot_structure == :loose
        (10.0, 150.0, 0.1)   # Loose clot: faster lysis
    else  # :normal
        (20.0, 100.0, 0.05)
    end

    return FibrinDegradation(
        fibrin_amount,        # fibrin_intact
        0.0,                  # fibrin_partially_degraded
        0.0,                  # fdp_large
        0.0,                  # fdp_small
        0.0,                  # d_dimer
        density,              # fibrin_density
        diameter,             # fiber_diameter
        permeability          # clot_permeability
    )
end

"""
create_fibrinolytic_system(; fibrin_amount, pai1_elevated, clot_structure)

Create complete fibrinolytic system.

# Arguments
- `fibrin_amount`: Initial fibrin concentration (nM), default 0 (no clot)
- `pai1_elevated`: Elevated PAI-1 (obesity, metabolic syndrome)
- `clot_structure`: :normal, :dense, :loose
"""
function create_fibrinolytic_system(;
    fibrin_amount::Float64=0.0,
    pai1_elevated::Bool=false,
    clot_structure::Symbol=:normal
)::FibrinolyticSystem

    return FibrinolyticSystem(
        create_plasminogen_state(pai1_elevated=pai1_elevated),
        create_fibrin_degradation(fibrin_amount, clot_structure=clot_structure),
        "none",               # thrombolytic_drug
        0.0,                  # thrombolytic_conc
        "none",               # antifibrinolytic_drug
        0.0,                  # antifibrinolytic_conc
        0.0,                  # fibrinolysis_inhibition
        false,                # hyperfibrinolysis
        0.0,                  # dic_score
        0.0,                  # lag_time
        0.0,                  # peak_plasmin
        0.0,                  # time_to_peak
        0.0,                  # epp
        Inf                   # lysis_time
    )
end

# ============================================================================
# ODE SYSTEM - Fibrinolysis Dynamics
# ============================================================================

"""
fibrinolysis_ode!(du, u, p, t)

ODE system for fibrinolysis.

State vector (16 components):
1-2: Plasminogen (Glu, Lys)
3-4: Plasmin (free, bound)
5-7: Activators (tPA free, tPA bound, uPA)
8-10: Inhibitors (PAI-1, α2-AP, TAFIa)
11-13: Complexes (PAI-tPA, PAI-uPA, α2AP-plasmin)
14-16: Fibrin (intact, degraded, D-dimer)
"""
function fibrinolysis_ode!(du, u, p, t)
    # Unpack state
    glu_plg, lys_plg = u[1:2]
    plasmin_free, plasmin_bound = u[3:4]
    tpa_free, tpa_bound, upa = u[5:7]
    pai1, a2ap, tafia = u[8:10]
    pai_tpa, pai_upa, pap = u[11:13]
    fibrin, fibrin_deg, d_dimer = u[14:16]

    # Total plasminogen
    plg_total = glu_plg + lys_plg

    # Parameters
    k = p.kinetics
    fibrin_effect = fibrin / (fibrin + 100.0)  # Fibrin enhances tPA activity

    # Drug effects
    tpa_therapy = p.tpa_therapy_rate
    inhibition = p.fibrinolysis_inhibition

    # ================================================================
    # PLASMINOGEN ACTIVATION
    # ================================================================

    # tPA-mediated activation (enhanced on fibrin surface)
    kcat_tpa = k["kcat_tPA_plg"] + (k["kcat_tPA_plg_fibrin"] - k["kcat_tPA_plg"]) * fibrin_effect
    rate_tpa_plg = kcat_tpa * (tpa_free + tpa_bound * 10.0) * plg_total / (k["Km_tPA_plg"] + plg_total)
    rate_tpa_plg *= (1.0 - inhibition)  # Antifibrinolytic effect

    # uPA-mediated activation
    rate_upa_plg = k["kcat_uPA_plg"] * upa * plg_total / (k["Km_uPA_plg"] + plg_total)
    rate_upa_plg *= (1.0 - inhibition)

    # Total plasmin generation
    plasmin_generation = rate_tpa_plg + rate_upa_plg + tpa_therapy

    # ================================================================
    # FIBRIN DEGRADATION
    # ================================================================

    # Plasmin degrades fibrin
    total_plasmin = plasmin_free + plasmin_bound
    rate_fibrin_lysis = k["kcat_plasmin_fibrin"] * total_plasmin * fibrin / (k["Km_plasmin_fibrin"] + fibrin)

    # TAFIa slows fibrinolysis by removing C-terminal lysines
    tafia_effect = 1.0 / (1.0 + tafia / 10.0)
    rate_fibrin_lysis *= tafia_effect

    # D-dimer generation from fibrin degradation
    rate_d_dimer = rate_fibrin_lysis * 0.5  # 50% of degraded fibrin becomes D-dimer

    # ================================================================
    # INHIBITION REACTIONS
    # ================================================================

    # PAI-1 inhibits tPA (very fast)
    rate_pai_tpa = k["k_PAI1_tPA"] * pai1 * tpa_free * 1e-9

    # PAI-1 inhibits uPA
    rate_pai_upa = k["k_PAI1_uPA"] * pai1 * upa * 1e-9

    # α2-antiplasmin inhibits plasmin (very fast)
    rate_a2ap_plasmin = k["k_A2AP_plasmin"] * a2ap * plasmin_free * 1e-9

    # ================================================================
    # tPA BINDING TO FIBRIN
    # ================================================================

    kd_tpa = k["Kd_tPA_fibrin"]
    rate_tpa_bind = 0.1 * tpa_free * fibrin / (kd_tpa + fibrin)
    rate_tpa_unbind = 0.01 * tpa_bound

    # ================================================================
    # ODEs
    # ================================================================

    # Plasminogen
    du[1] = -0.1 * glu_plg - plasmin_generation * (glu_plg / plg_total)  # Glu-Plg → Lys-Plg and Plasmin
    du[2] = 0.1 * glu_plg - plasmin_generation * (lys_plg / plg_total)   # Lys-Plg → Plasmin

    # Plasmin
    du[3] = plasmin_generation * 0.3 - rate_a2ap_plasmin  # Free plasmin (most is immediately inhibited)
    du[4] = plasmin_generation * 0.7 - 0.1 * plasmin_bound  # Fibrin-bound plasmin

    # Activators
    du[5] = -rate_pai_tpa - rate_tpa_bind + rate_tpa_unbind  # tPA free
    du[6] = rate_tpa_bind - rate_tpa_unbind - rate_fibrin_lysis * 0.01  # tPA bound (released during lysis)
    du[7] = -rate_pai_upa  # uPA

    # Inhibitors
    du[8] = -rate_pai_tpa - rate_pai_upa  # PAI-1 consumed
    du[9] = -rate_a2ap_plasmin  # α2-AP consumed
    du[10] = -0.001 * tafia  # TAFIa decay

    # Complexes
    du[11] = rate_pai_tpa  # PAI-1·tPA complex
    du[12] = rate_pai_upa  # PAI-1·uPA complex
    du[13] = rate_a2ap_plasmin  # PAP (plasmin-antiplasmin) complex

    # Fibrin degradation
    du[14] = -rate_fibrin_lysis  # Fibrin intact
    du[15] = rate_fibrin_lysis - rate_d_dimer  # Partially degraded
    du[16] = rate_d_dimer  # D-dimer accumulation

    return nothing
end

"""
simulate_fibrinolysis!(system, tspan, dt; thrombin_conc)

Simulate fibrinolysis over time.

# Arguments
- `system`: FibrinolyticSystem to simulate
- `tspan`: (t_start, t_end) in seconds
- `dt`: Time step in seconds
- `thrombin_conc`: Thrombin concentration for TAFI activation (nM)
"""
function simulate_fibrinolysis!(
    system::FibrinolyticSystem,
    tspan::Tuple{Float64, Float64},
    dt::Float64;
    thrombin_conc::Float64=0.0
)
    t_start, t_end = tspan
    times = collect(t_start:dt:t_end)
    n_steps = length(times)

    plg = system.plasminogen
    fib = system.fibrin

    # Initialize state vector
    u = [
        plg.glu_plasminogen, plg.lys_plasminogen,
        plg.plasmin_free, plg.plasmin_fibrin_bound,
        plg.tpa_free, plg.tpa_fibrin_bound, plg.upa_free,
        plg.pai1_active, plg.alpha2_antiplasmin, plg.tafi_active,
        plg.pai1_tpa_complex, plg.pai1_upa_complex, plg.a2ap_plasmin_complex,
        fib.fibrin_intact, fib.fibrin_partially_degraded, fib.d_dimer
    ]

    du = zeros(16)

    # Calculate tPA therapy rate if thrombolytic given
    tpa_therapy_rate = calculate_tpa_therapy_rate(system)

    # Activate TAFI if thrombin present
    if thrombin_conc > 0
        tafi_activation = thrombin_conc / (thrombin_conc + 10.0) * plg.tafi * 0.1
        u[10] = tafi_activation
    end

    params = (
        kinetics = KINETIC_FIBRINOLYSIS,
        tpa_therapy_rate = tpa_therapy_rate,
        fibrinolysis_inhibition = system.fibrinolysis_inhibition
    )

    # Store results
    results = Vector{Dict{String, Float64}}(undef, n_steps)
    initial_fibrin = u[14]

    for (i, t) in enumerate(times)
        results[i] = Dict(
            "time" => t,
            "plasminogen_nM" => u[1] + u[2],
            "plasmin_nM" => u[3] + u[4],
            "tpa_nM" => u[5] + u[6],
            "pai1_nM" => u[8],
            "fibrin_nM" => u[14],
            "fibrin_degraded_nM" => u[15],
            "d_dimer_nM" => u[16],
            "pap_complex_nM" => u[13],
            "lysis_percent" => initial_fibrin > 0 ? (1.0 - u[14] / initial_fibrin) * 100 : 0.0
        )

        fibrinolysis_ode!(du, u, params, t)
        u .= u .+ du .* dt
        u .= max.(u, 0.0)
    end

    # Update system
    update_fibrinolytic_system!(system, u)
    calculate_lysis_parameters!(system, times, results)

    return times, results
end

"""
calculate_tpa_therapy_rate(system)

Calculate tPA generation rate from thrombolytic therapy.
"""
function calculate_tpa_therapy_rate(system::FibrinolyticSystem)::Float64
    if system.thrombolytic_drug == "none"
        return 0.0
    end

    drug = system.thrombolytic_drug
    conc = system.thrombolytic_conc

    if !haskey(THROMBOLYTIC_PARAMS, drug)
        return 0.0
    end

    params = THROMBOLYTIC_PARAMS[drug]
    selectivity = params["fibrin_selectivity"]

    # Convert to plasmin generation rate
    return conc * 0.1 * (1.0 + selectivity)
end

"""
update_fibrinolytic_system!(system, u)

Update system from ODE state vector.
"""
function update_fibrinolytic_system!(system::FibrinolyticSystem, u::Vector{Float64})
    plg = system.plasminogen
    fib = system.fibrin

    plg.glu_plasminogen = u[1]
    plg.lys_plasminogen = u[2]
    plg.plasmin_free = u[3]
    plg.plasmin_fibrin_bound = u[4]
    plg.tpa_free = u[5]
    plg.tpa_fibrin_bound = u[6]
    plg.upa_free = u[7]
    plg.pai1_active = u[8]
    plg.alpha2_antiplasmin = u[9]
    plg.tafi_active = u[10]
    plg.pai1_tpa_complex = u[11]
    plg.pai1_upa_complex = u[12]
    plg.a2ap_plasmin_complex = u[13]

    fib.fibrin_intact = u[14]
    fib.fibrin_partially_degraded = u[15]
    fib.d_dimer = u[16]

    return nothing
end

"""
calculate_lysis_parameters!(system, times, results)

Calculate fibrinolysis assay parameters.
"""
function calculate_lysis_parameters!(
    system::FibrinolyticSystem,
    times::Vector{Float64},
    results::Vector{Dict{String, Float64}}
)
    plasmin = [r["plasmin_nM"] for r in results]
    lysis_pct = [r["lysis_percent"] for r in results]

    # Peak plasmin
    peak_idx = argmax(plasmin)
    system.peak_plasmin = plasmin[peak_idx]
    system.time_to_peak = times[peak_idx] / 60.0

    # Lag time (time to 6 nM plasmin)
    lag_idx = findfirst(p -> p > 6.0, plasmin)
    system.lag_time = isnothing(lag_idx) ? Inf : times[lag_idx] / 60.0

    # EPP (area under plasmin curve)
    dt = times[2] - times[1]
    system.epp = sum(plasmin) * dt / 60.0

    # Lysis time (50% clot lysis)
    lysis_50_idx = findfirst(l -> l >= 50.0, lysis_pct)
    system.lysis_time = isnothing(lysis_50_idx) ? Inf : times[lysis_50_idx] / 60.0

    return nothing
end

# ============================================================================
# DRUG EFFECTS
# ============================================================================

"""
apply_tpa_therapy!(system, drug_name, concentration)

Apply thrombolytic therapy.

# Arguments
- `system`: FibrinolyticSystem
- `drug_name`: "alteplase", "tenecteplase", "reteplase", "streptokinase"
- `concentration`: Drug concentration (IU/mL for alteplase, MU/mg for others)
"""
function apply_tpa_therapy!(
    system::FibrinolyticSystem,
    drug_name::String,
    concentration::Float64
)
    drug = lowercase(drug_name)

    if !haskey(THROMBOLYTIC_PARAMS, drug)
        @warn "Unknown thrombolytic: $drug_name"
        return nothing
    end

    system.thrombolytic_drug = drug
    system.thrombolytic_conc = concentration

    # Increase tPA levels
    params = THROMBOLYTIC_PARAMS[drug]
    if drug == "alteplase"
        # Add exogenous tPA
        system.plasminogen.tpa_free += concentration * 0.01  # Simplified conversion
    end

    return nothing
end

"""
apply_antifibrinolytic!(system, drug_name, concentration)

Apply antifibrinolytic therapy.

# Arguments
- `system`: FibrinolyticSystem
- `drug_name`: "tranexamic_acid", "aminocaproic_acid", "aprotinin"
- `concentration`: Drug concentration (nM)
"""
function apply_antifibrinolytic!(
    system::FibrinolyticSystem,
    drug_name::String,
    concentration::Float64
)
    drug = lowercase(drug_name)

    if !haskey(ANTIFIBRINOLYTIC_PARAMS, drug)
        @warn "Unknown antifibrinolytic: $drug_name"
        return nothing
    end

    params = ANTIFIBRINOLYTIC_PARAMS[drug]
    system.antifibrinolytic_drug = drug
    system.antifibrinolytic_conc = concentration

    # Calculate inhibition using Hill equation
    ec50 = get(params, "EC50_fibrinolysis", 50e3)
    system.fibrinolysis_inhibition = concentration / (ec50 + concentration)

    return nothing
end

# ============================================================================
# CLINICAL ENDPOINTS
# ============================================================================

"""
calculate_d_dimer(system)

Calculate D-dimer level in clinical units.

# Returns
- D-dimer in ng/mL FEU (fibrinogen equivalent units)
"""
function calculate_d_dimer(system::FibrinolyticSystem)::Float64
    # Convert nM to ng/mL FEU
    # D-dimer MW ≈ 180 kDa
    d_dimer_nM = system.fibrin.d_dimer
    d_dimer_ng_ml = d_dimer_nM * 180.0 / 1000.0  # nM × MW(kDa) = μg/L = ng/mL

    return d_dimer_ng_ml
end

"""
calculate_lysis_time(system)

Get time for 50% clot lysis in minutes.
"""
function calculate_lysis_time(system::FibrinolyticSystem)::Float64
    return system.lysis_time
end

"""
detect_hyperfibrinolysis(system)

Detect pathological hyperfibrinolysis based on parameters.

Returns true if hyperfibrinolysis detected.
"""
function detect_hyperfibrinolysis(system::FibrinolyticSystem)::Bool
    # Criteria for hyperfibrinolysis:
    # 1. Very short lysis time (<30 min)
    # 2. High plasmin generation
    # 3. Rapid D-dimer elevation

    short_lysis = system.lysis_time < 30.0
    high_plasmin = system.peak_plasmin > 100.0
    high_d_dimer = system.fibrin.d_dimer > 1000.0

    return short_lysis || (high_plasmin && high_d_dimer)
end

"""
plasmin_generation_assay(system; tpa_conc, duration_min)

Perform plasmin generation assay (PGA) simulation.

Similar to thrombin generation assay but for fibrinolysis.
"""
function plasmin_generation_assay(
    system::FibrinolyticSystem;
    tpa_conc::Float64=0.1,  # 0.1 nM tPA stimulus
    fibrin_amount::Float64=1000.0,  # 1 μM fibrin
    duration_min::Float64=60.0
)::Dict{String, Float64}

    # Create assay system
    assay = deepcopy(system)
    assay.fibrin.fibrin_intact = fibrin_amount
    assay.plasminogen.tpa_free += tpa_conc

    # Simulate
    tspan = (0.0, duration_min * 60.0)
    dt = 1.0

    times, results = simulate_fibrinolysis!(assay, tspan, dt)

    return Dict(
        "lag_time_min" => assay.lag_time,
        "peak_plasmin_nM" => assay.peak_plasmin,
        "time_to_peak_min" => assay.time_to_peak,
        "epp_nM_min" => assay.epp,
        "lysis_time_min" => assay.lysis_time
    )
end

# ============================================================================
# STATE REPORTING
# ============================================================================

"""
get_fibrinolysis_state(system)

Get comprehensive state report.
"""
function get_fibrinolysis_state(system::FibrinolyticSystem)::Dict{String, Any}
    plg = system.plasminogen
    fib = system.fibrin

    return Dict(
        "plasminogen_system" => Dict(
            "plasminogen_total_nM" => plg.glu_plasminogen + plg.lys_plasminogen,
            "plasmin_free_nM" => plg.plasmin_free,
            "plasmin_bound_nM" => plg.plasmin_fibrin_bound,
            "tpa_total_nM" => plg.tpa_free + plg.tpa_fibrin_bound,
            "upa_nM" => plg.upa_free
        ),

        "inhibitors" => Dict(
            "pai1_active_nM" => plg.pai1_active,
            "alpha2_antiplasmin_nM" => plg.alpha2_antiplasmin,
            "tafi_active_nM" => plg.tafi_active
        ),

        "complexes" => Dict(
            "pai1_tpa_nM" => plg.pai1_tpa_complex,
            "pap_nM" => plg.a2ap_plasmin_complex
        ),

        "fibrin_state" => Dict(
            "fibrin_intact_nM" => fib.fibrin_intact,
            "fibrin_degraded_nM" => fib.fibrin_partially_degraded,
            "d_dimer_nM" => fib.d_dimer,
            "d_dimer_ng_ml" => calculate_d_dimer(system)
        ),

        "assay_parameters" => Dict(
            "lag_time_min" => system.lag_time,
            "peak_plasmin_nM" => system.peak_plasmin,
            "time_to_peak_min" => system.time_to_peak,
            "epp_nM_min" => system.epp,
            "lysis_time_min" => system.lysis_time
        ),

        "therapy" => Dict(
            "thrombolytic" => system.thrombolytic_drug,
            "thrombolytic_conc" => system.thrombolytic_conc,
            "antifibrinolytic" => system.antifibrinolytic_drug,
            "fibrinolysis_inhibition" => system.fibrinolysis_inhibition
        ),

        "clinical" => Dict(
            "hyperfibrinolysis" => detect_hyperfibrinolysis(system)
        )
    )
end

end  # module Fibrinolysis
