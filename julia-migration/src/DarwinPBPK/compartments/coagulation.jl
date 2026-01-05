"""
Coagulation System - Comprehensive Model with ODE Dynamics

Models the coagulation cascade with:
- All coagulation factors (II, V, VII, VIII, IX, X, XI, XII, XIII)
- Activated factors and enzyme complexes
- Natural anticoagulants (ATIII, Protein C/S, TFPI)
- Vitamin K-dependent synthesis
- Drug effects (warfarin, DOACs, heparin)
- Clinical endpoints (PT/INR, aPTT, Anti-Xa)

Based on SOTA Q1 models:
- Wajima et al. (2009) - 56 ODEs coagulation network
- Hockin-Mann (2002) - 34 ODEs, 42 rate constants
- Hartmann et al. (2016) - QSP anticoagulant model

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

module Coagulation

using LinearAlgebra
using Statistics
using SparseArrays
using DifferentialEquations: ODEFunction, ODEProblem, solve, Rodas4, Tsit5, FBDF, CallbackSet, DiscreteCallback

export CoagulationFactors, CoagulationSystem, AnticoagulantState
export create_coagulation_system, simulate_coagulation!, simulate_coagulation_stiff!
export apply_warfarin!, apply_doac!, apply_heparin!
export calculate_pt_inr, calculate_aptt, calculate_anti_xa
export thrombin_generation_assay, get_coagulation_state
export NORMAL_FACTOR_CONCENTRATIONS, FACTOR_HALF_LIVES
export create_coagulation_jacobian_prototype, CoagulationSolverConfig

# ============================================================================
# CONSTANTS - From SOTA Literature (Wajima, Hockin-Mann)
# ============================================================================

# Normal plasma concentrations (nM)
const NORMAL_FACTOR_CONCENTRATIONS = Dict(
    "II" => 1400.0,      # Prothrombin (100 μg/mL)
    "V" => 22.0,         # Proaccelerin (20-25 nM)
    "VII" => 10.0,       # Proconvertin (0.5 μg/mL)
    "VIIa" => 0.1,       # Activated VII (trace, 3.6 ng/mL)
    "VIII" => 0.7,       # Anti-hemophilic A
    "IX" => 90.0,        # Christmas factor (5 μg/mL)
    "X" => 170.0,        # Stuart-Prower (10 μg/mL)
    "XI" => 45.0,        # Plasma thromboplastin (30-60 nM)
    "XII" => 375.0,      # Hageman factor
    "XIII" => 70.0,      # Fibrin-stabilizing
    "fibrinogen" => 9000.0,  # 7-12 μM → 7000-12000 nM (3 g/L)
    "ATIII" => 2400.0,   # Antithrombin III (2.4 μM)
    "TFPI" => 2.5,       # Tissue factor pathway inhibitor
    "protein_C" => 65.0, # Protein C
    "protein_S" => 350.0 # Protein S
)

# Half-lives (hours)
const FACTOR_HALF_LIVES = Dict(
    "II" => 66.0,        # 60-72 h
    "V" => 24.0,         # 12-36 h
    "VII" => 4.5,        # 3-6 h (shortest - first to decrease with warfarin)
    "VIII" => 12.0,      # 12 h
    "IX" => 21.0,        # 18-24 h
    "X" => 48.0,         # 48 h
    "XI" => 70.0,        # 60-80 h
    "XII" => 60.0,       # 50-70 h
    "XIII" => 160.0,     # 120-200 h (longest)
    "fibrinogen" => 108.0, # 96-120 h
    "protein_C" => 8.0,  # 8 h (short, important for warfarin)
    "protein_S" => 42.0  # 42 h
)

# Vitamin K-dependent factors
const VK_DEPENDENT_FACTORS = ["II", "VII", "IX", "X", "protein_C", "protein_S"]

# Kinetic parameters from Hockin-Mann model (1/s or nM)
const KINETIC_PARAMS = Dict(
    # Extrinsic pathway
    "kcat_VIIa_X" => 103.0,      # s⁻¹ (VIIa activates X)
    "Km_VIIa_X" => 450.0,        # nM
    "kcat_VIIa_IX" => 2.3,       # s⁻¹ (VIIa activates IX)
    "Km_VIIa_IX" => 300.0,       # nM

    # Tenase complex (IXa·VIIIa)
    "kcat_tenase" => 10.7,       # s⁻¹
    "Km_tenase" => 160.0,        # nM

    # Prothrombinase complex (Xa·Va)
    "kcat_prothrombinase" => 32.4,  # s⁻¹
    "Km_prothrombinase" => 300.0,   # nM

    # Thrombin feedback
    "kcat_IIa_V" => 20.0,        # s⁻¹
    "Km_IIa_V" => 100.0,         # nM
    "kcat_IIa_VIII" => 20.0,     # s⁻¹
    "Km_IIa_VIII" => 100.0,      # nM
    "kcat_IIa_XI" => 0.23,       # s⁻¹
    "Km_IIa_XI" => 2000.0,       # nM

    # Fibrin formation
    "kcat_IIa_fibrinogen" => 84.0,  # s⁻¹
    "Km_IIa_fibrinogen" => 7200.0,  # nM

    # Inhibition by ATIII (second-order rate constants, M⁻¹s⁻¹)
    "k_ATIII_IIa" => 7.1e3,      # M⁻¹s⁻¹
    "k_ATIII_Xa" => 1.5e3,       # M⁻¹s⁻¹
    "k_ATIII_IXa" => 1.0e2,      # M⁻¹s⁻¹

    # TFPI inhibition
    "k_TFPI_Xa" => 9.0e5,        # M⁻¹s⁻¹

    # Protein C activation by thrombin-thrombomodulin
    "kcat_IIa_TM_PC" => 0.31,    # s⁻¹
    "Km_IIa_TM_PC" => 2000.0,    # nM

    # APC inactivation of Va and VIIIa
    "k_APC_Va" => 2.0e5,         # M⁻¹s⁻¹
    "k_APC_VIIIa" => 2.0e5       # M⁻¹s⁻¹
)

# DOAC parameters (Ki in nM)
const DOAC_PARAMS = Dict(
    "rivaroxaban" => Dict("Ki_Xa" => 0.4, "Ki_prothrombinase" => 2.1),
    "apixaban" => Dict("Ki_Xa" => 0.08, "Ki_prothrombinase" => 0.62),
    "edoxaban" => Dict("Ki_Xa" => 0.56, "Ki_prothrombinase" => 1.6),
    "betrixaban" => Dict("Ki_Xa" => 0.12, "Ki_prothrombinase" => 0.35),
    "dabigatran" => Dict("Ki_IIa" => 4.5)  # Direct thrombin inhibitor
)

# Warfarin parameters
const WARFARIN_PARAMS = Dict(
    "IC50_VKORC1" => 0.3,        # μM (S-warfarin)
    "Emax" => 0.99,              # Maximum inhibition
    "gamma" => 1.0,              # Hill coefficient
    "MTT_rapid" => 27.2,         # hours (Factor VII)
    "MTT_slow" => 110.9          # hours (Factors II, IX, X)
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
CoagulationFactors - All coagulation factor concentrations (nM)
"""
mutable struct CoagulationFactors
    # Zymogens (inactive precursors)
    factor_II::Float64       # Prothrombin
    factor_V::Float64        # Proaccelerin
    factor_VII::Float64      # Proconvertin
    factor_VIII::Float64     # Anti-hemophilic A
    factor_IX::Float64       # Christmas factor
    factor_X::Float64        # Stuart-Prower
    factor_XI::Float64       # Plasma thromboplastin
    factor_XII::Float64      # Hageman factor
    factor_XIII::Float64     # Fibrin-stabilizing
    fibrinogen::Float64      # Fibrinogen

    # Activated factors
    factor_IIa::Float64      # Thrombin (key enzyme!)
    factor_Va::Float64
    factor_VIIa::Float64
    factor_VIIIa::Float64
    factor_IXa::Float64
    factor_Xa::Float64
    factor_XIa::Float64
    factor_XIIa::Float64
    factor_XIIIa::Float64

    # Fibrin
    fibrin::Float64          # Cross-linked fibrin

    # Natural anticoagulants
    antithrombin::Float64    # ATIII
    tfpi::Float64            # Tissue factor pathway inhibitor
    protein_C::Float64       # Inactive
    protein_C_active::Float64 # APC
    protein_S::Float64

    # Inhibitor complexes (inactive)
    ATIII_IIa::Float64       # Thrombin-antithrombin complex (TAT)
    ATIII_Xa::Float64
    TFPI_Xa::Float64
end

"""
AnticoagulantState - Drug-induced anticoagulation state
"""
mutable struct AnticoagulantState
    # Warfarin
    warfarin_S_conc::Float64      # S-warfarin concentration (μM)
    warfarin_R_conc::Float64      # R-warfarin concentration (μM)
    vkorc1_inhibition::Float64    # Fraction inhibited (0-1)
    vk_synthesis_rate::Float64    # Relative synthesis rate (0-1)

    # DOACs
    doac_type::String             # "none", "rivaroxaban", "apixaban", etc.
    doac_conc::Float64            # Concentration (nM)
    xa_inhibition::Float64        # Fraction Xa inhibited
    iia_inhibition::Float64       # Fraction IIa inhibited (dabigatran)

    # Heparin
    heparin_conc::Float64         # IU/mL (or nM equivalent)
    heparin_type::String          # "none", "UFH", "LMWH"
    atiii_potentiation::Float64   # Fold increase in ATIII activity

    # Genetic factors (for warfarin)
    vkorc1_genotype::String       # "AA", "AB", "BB"
    cyp2c9_genotype::String       # "*1/*1", "*1/*2", "*2/*2", "*1/*3", etc.
end

"""
CoagulationSystem - Complete coagulation model
"""
mutable struct CoagulationSystem
    # Factor concentrations
    factors::CoagulationFactors

    # Anticoagulant drug state
    anticoagulant::AnticoagulantState

    # Tissue factor (input/stimulus)
    tissue_factor::Float64        # nM (initiates extrinsic pathway)

    # Thrombomodulin (endothelial)
    thrombomodulin::Float64       # nM

    # Clinical endpoints (calculated)
    pt_seconds::Float64           # Prothrombin time (seconds)
    inr::Float64                  # International Normalized Ratio
    aptt_seconds::Float64         # Activated partial thromboplastin time
    anti_xa_IU::Float64           # Anti-Xa activity (IU/mL)

    # Thrombin generation parameters
    lag_time::Float64             # minutes
    peak_thrombin::Float64        # nM
    time_to_peak::Float64         # minutes
    etp::Float64                  # Endogenous thrombin potential (nM·min)
end

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

"""
create_coagulation_factors(; deficiencies=Dict())

Create normal coagulation factors with optional deficiencies.

# Arguments
- `deficiencies`: Dict mapping factor name to fraction of normal (0-1)
"""
function create_coagulation_factors(;
    deficiencies::Dict{String, Float64}=Dict{String, Float64}()
)::CoagulationFactors

    get_level = (name) -> NORMAL_FACTOR_CONCENTRATIONS[name] * get(deficiencies, name, 1.0)

    return CoagulationFactors(
        # Zymogens
        get_level("II"),
        get_level("V"),
        get_level("VII"),
        get_level("VIII"),
        get_level("IX"),
        get_level("X"),
        get_level("XI"),
        get_level("XII"),
        get_level("XIII"),
        get_level("fibrinogen"),

        # Activated factors (initially zero or trace)
        0.0,    # IIa (thrombin)
        0.0,    # Va
        get_level("VIIa"),  # VIIa has trace levels normally
        0.0,    # VIIIa
        0.0,    # IXa
        0.0,    # Xa
        0.0,    # XIa
        0.0,    # XIIa
        0.0,    # XIIIa

        # Fibrin
        0.0,

        # Anticoagulants
        get_level("ATIII"),
        get_level("TFPI"),
        get_level("protein_C"),
        0.0,    # APC (activated protein C)
        get_level("protein_S"),

        # Complexes
        0.0,    # TAT
        0.0,    # ATIII-Xa
        0.0     # TFPI-Xa
    )
end

"""
create_anticoagulant_state()

Create initial (no drugs) anticoagulant state.
"""
function create_anticoagulant_state()::AnticoagulantState
    return AnticoagulantState(
        0.0,    # warfarin_S_conc
        0.0,    # warfarin_R_conc
        0.0,    # vkorc1_inhibition
        1.0,    # vk_synthesis_rate (100%)
        "none", # doac_type
        0.0,    # doac_conc
        0.0,    # xa_inhibition
        0.0,    # iia_inhibition
        0.0,    # heparin_conc
        "none", # heparin_type
        1.0,    # atiii_potentiation
        "AB",   # vkorc1_genotype (wild-type)
        "*1/*1" # cyp2c9_genotype (wild-type)
    )
end

"""
create_coagulation_system(; tissue_factor, deficiencies)

Create complete coagulation system.

# Arguments
- `tissue_factor`: Initial TF concentration (nM), default 0 (no stimulus)
- `deficiencies`: Dict of factor deficiencies
"""
function create_coagulation_system(;
    tissue_factor::Float64=0.0,
    deficiencies::Dict{String, Float64}=Dict{String, Float64}()
)::CoagulationSystem

    return CoagulationSystem(
        create_coagulation_factors(deficiencies=deficiencies),
        create_anticoagulant_state(),
        tissue_factor,
        10.0,   # thrombomodulin (nM, endothelial)

        # Clinical endpoints (normal values)
        13.0,   # PT (11-15 s)
        1.0,    # INR
        30.0,   # aPTT (25-35 s)
        0.0,    # Anti-Xa

        # TG parameters (will be calculated)
        0.0,
        0.0,
        0.0,
        0.0
    )
end

# ============================================================================
# ODE SYSTEM - Coagulation Cascade
# ============================================================================

"""
coagulation_ode!(du, u, p, t)

ODE system for coagulation cascade based on Hockin-Mann model.

State vector u (25 components):
1-10: Zymogens (II, V, VII, VIII, IX, X, XI, XII, XIII, Fg)
11-19: Activated factors (IIa, Va, VIIa, VIIIa, IXa, Xa, XIa, XIIa, XIIIa)
20: Fibrin
21-23: Anticoagulants (ATIII, TFPI, APC)
24-25: Complexes (TAT, ATIII-Xa)

Parameters p: kinetic constants and drug effects
"""
function coagulation_ode!(du, u, p, t)
    # Unpack state
    II, V, VII, VIII, IX, X, XI, XII, XIII, Fg = u[1:10]
    IIa, Va, VIIa, VIIIa, IXa, Xa, XIa, XIIa, XIIIa = u[11:19]
    fibrin = u[20]
    ATIII, TFPI, APC = u[21:23]
    TAT, ATIII_Xa = u[24:25]

    # Parameters
    TF = p.tissue_factor
    k = p.kinetics

    # Drug effects
    xa_inhibition = p.xa_inhibition
    iia_inhibition = p.iia_inhibition
    atiii_potentiation = p.atiii_potentiation
    vk_synthesis = p.vk_synthesis_rate

    # Effective concentrations (after drug inhibition)
    Xa_eff = Xa * (1.0 - xa_inhibition)
    IIa_eff = IIa * (1.0 - iia_inhibition)

    # ================================================================
    # EXTRINSIC PATHWAY (TF-initiated)
    # ================================================================

    # TF·VIIa activates X → Xa
    TF_VIIa = TF * VIIa / (TF + 1.0)  # Simplified TF·VIIa complex
    rate_VIIa_Xa = k["kcat_VIIa_X"] * TF_VIIa * X / (k["Km_VIIa_X"] + X)

    # TF·VIIa activates IX → IXa
    rate_VIIa_IXa = k["kcat_VIIa_IX"] * TF_VIIa * IX / (k["Km_VIIa_IX"] + IX)

    # ================================================================
    # INTRINSIC PATHWAY (Amplification)
    # ================================================================

    # Tenase complex: IXa·VIIIa activates X → Xa
    tenase = IXa * VIIIa / (IXa + VIIIa + 1.0)  # Simplified complex
    rate_tenase_Xa = k["kcat_tenase"] * tenase * X / (k["Km_tenase"] + X)

    # XIa activates IX → IXa (contact pathway)
    rate_XIa_IXa = 0.1 * XIa * IX / (100.0 + IX)  # Simplified

    # ================================================================
    # COMMON PATHWAY
    # ================================================================

    # Prothrombinase complex: Xa·Va activates II → IIa (Thrombin)
    prothrombinase = Xa_eff * Va / (Xa_eff + Va + 1.0)
    rate_prothrombinase = k["kcat_prothrombinase"] * prothrombinase * II / (k["Km_prothrombinase"] + II)

    # Thrombin converts fibrinogen → fibrin
    rate_fibrin = k["kcat_IIa_fibrinogen"] * IIa_eff * Fg / (k["Km_IIa_fibrinogen"] + Fg)

    # ================================================================
    # THROMBIN FEEDBACK (Amplification)
    # ================================================================

    # Thrombin activates V → Va
    rate_IIa_Va = k["kcat_IIa_V"] * IIa_eff * V / (k["Km_IIa_V"] + V)

    # Thrombin activates VIII → VIIIa
    rate_IIa_VIIIa = k["kcat_IIa_VIII"] * IIa_eff * VIII / (k["Km_IIa_VIII"] + VIII)

    # Thrombin activates XI → XIa
    rate_IIa_XIa = k["kcat_IIa_XI"] * IIa_eff * XI / (k["Km_IIa_XI"] + XI)

    # Thrombin activates XIII → XIIIa
    rate_IIa_XIIIa = 0.5 * IIa_eff * XIII / (500.0 + XIII)

    # ================================================================
    # INHIBITION (Termination)
    # ================================================================

    # Potentiated ATIII rate constants
    k_ATIII_IIa = k["k_ATIII_IIa"] * atiii_potentiation
    k_ATIII_Xa = k["k_ATIII_Xa"] * atiii_potentiation

    # ATIII inhibits thrombin → TAT complex
    rate_ATIII_IIa = k_ATIII_IIa * ATIII * IIa * 1e-9  # Convert to nM/s

    # ATIII inhibits Xa
    rate_ATIII_Xa = k_ATIII_Xa * ATIII * Xa * 1e-9

    # TFPI inhibits Xa
    rate_TFPI_Xa = k["k_TFPI_Xa"] * TFPI * Xa * 1e-9

    # Protein C activation by thrombin·thrombomodulin
    TM = p.thrombomodulin
    IIa_TM = IIa_eff * TM / (IIa_eff + TM + 10.0)
    PC = p.protein_C
    rate_APC_formation = k["kcat_IIa_TM_PC"] * IIa_TM * PC / (k["Km_IIa_TM_PC"] + PC)

    # APC inactivates Va and VIIIa
    rate_APC_Va = k["k_APC_Va"] * APC * Va * 1e-9
    rate_APC_VIIIa = k["k_APC_VIIIa"] * APC * VIIIa * 1e-9

    # ================================================================
    # SYNTHESIS (for VK-dependent factors with warfarin)
    # ================================================================

    # Degradation rate constants (ln(2)/half-life in hours → per second)
    k_deg_II = log(2) / (FACTOR_HALF_LIVES["II"] * 3600)
    k_deg_VII = log(2) / (FACTOR_HALF_LIVES["VII"] * 3600)
    k_deg_IX = log(2) / (FACTOR_HALF_LIVES["IX"] * 3600)
    k_deg_X = log(2) / (FACTOR_HALF_LIVES["X"] * 3600)

    # Synthesis rates (at steady state: k_syn = k_deg * [normal])
    k_syn_II = k_deg_II * NORMAL_FACTOR_CONCENTRATIONS["II"]
    k_syn_VII = k_deg_VII * NORMAL_FACTOR_CONCENTRATIONS["VII"]
    k_syn_IX = k_deg_IX * NORMAL_FACTOR_CONCENTRATIONS["IX"]
    k_syn_X = k_deg_X * NORMAL_FACTOR_CONCENTRATIONS["X"]

    # Net synthesis (reduced by warfarin via vk_synthesis)
    rate_syn_II = k_syn_II * vk_synthesis - k_deg_II * II
    rate_syn_VII = k_syn_VII * vk_synthesis - k_deg_VII * VII
    rate_syn_IX = k_syn_IX * vk_synthesis - k_deg_IX * IX
    rate_syn_X = k_syn_X * vk_synthesis - k_deg_X * X

    # ================================================================
    # ODEs
    # ================================================================

    # Zymogens
    du[1] = rate_syn_II - rate_prothrombinase  # dII/dt
    du[2] = -rate_IIa_Va  # dV/dt
    du[3] = rate_syn_VII  # dVII/dt (no consumption modeled)
    du[4] = -rate_IIa_VIIIa  # dVIII/dt
    du[5] = rate_syn_IX - rate_VIIa_IXa - rate_XIa_IXa  # dIX/dt
    du[6] = rate_syn_X - rate_VIIa_Xa - rate_tenase_Xa  # dX/dt
    du[7] = -rate_IIa_XIa  # dXI/dt
    du[8] = 0.0  # dXII/dt (contact activation not modeled in detail)
    du[9] = -rate_IIa_XIIIa  # dXIII/dt
    du[10] = -rate_fibrin  # dFg/dt

    # Activated factors
    du[11] = rate_prothrombinase - rate_ATIII_IIa  # dIIa/dt (Thrombin!)
    du[12] = rate_IIa_Va - rate_APC_Va  # dVa/dt
    du[13] = 0.0  # dVIIa/dt (assumed constant due to autoactivation)
    du[14] = rate_IIa_VIIIa - rate_APC_VIIIa  # dVIIIa/dt
    du[15] = rate_VIIa_IXa + rate_XIa_IXa  # dIXa/dt
    du[16] = rate_VIIa_Xa + rate_tenase_Xa - rate_ATIII_Xa - rate_TFPI_Xa  # dXa/dt
    du[17] = rate_IIa_XIa  # dXIa/dt
    du[18] = 0.0  # dXIIa/dt
    du[19] = rate_IIa_XIIIa  # dXIIIa/dt

    # Fibrin
    du[20] = rate_fibrin  # dFibrin/dt

    # Anticoagulants
    du[21] = -rate_ATIII_IIa - rate_ATIII_Xa  # dATIII/dt
    du[22] = -rate_TFPI_Xa  # dTFPI/dt
    du[23] = rate_APC_formation  # dAPC/dt

    # Complexes
    du[24] = rate_ATIII_IIa  # dTAT/dt
    du[25] = rate_ATIII_Xa  # dATIII_Xa/dt

    return nothing
end

# ============================================================================
# SOLVER CONFIGURATION AND SPARSE JACOBIAN
# ============================================================================

"""
CoagulationSolverConfig - Configuration for stiff ODE solvers

# Fields
- `algorithm`: ODE solver algorithm (:rodas4, :fbdf, :tsit5)
- `reltol`: Relative tolerance (default 1e-6)
- `abstol`: Absolute tolerance (default 1e-8)
- `maxiters`: Maximum iterations (default 100000)
- `use_sparse`: Use sparse Jacobian (default true)
- `save_everystep`: Save all solver steps (default false)
- `saveat`: Time points to save at (default Float64[])
"""
Base.@kwdef struct CoagulationSolverConfig
    algorithm::Symbol = :rodas4
    reltol::Float64 = 1e-6
    abstol::Float64 = 1e-8
    maxiters::Int = 100000
    use_sparse::Bool = true
    save_everystep::Bool = false
    saveat::Vector{Float64} = Float64[]
end

"""
create_coagulation_jacobian_prototype()

Create sparse Jacobian prototype for the coagulation ODE system.

The 25-state coagulation system has a sparse dependency structure:
- Zymogens (1-10): affected by their own consumption/synthesis
- Activated factors (11-19): depend on zymogens and inhibitors
- Fibrin (20): depends on thrombin and fibrinogen
- Anticoagulants (21-23): depend on activated factors
- Complexes (24-25): depend on inhibitors and activated factors

Returns SparseMatrixCSC with ~80 non-zeros (vs 625 dense).
"""
function create_coagulation_jacobian_prototype()::SparseMatrixCSC{Float64, Int64}
    n = 25

    # Build sparsity pattern based on ODE dependencies
    # (i, j) means ∂du[i]/∂u[j] ≠ 0
    I = Int64[]
    J = Int64[]

    # State indices:
    # 1-10: II, V, VII, VIII, IX, X, XI, XII, XIII, Fg (zymogens)
    # 11-19: IIa, Va, VIIa, VIIIa, IXa, Xa, XIa, XIIa, XIIIa (activated)
    # 20: Fibrin
    # 21-23: ATIII, TFPI, APC
    # 24-25: TAT, ATIII_Xa

    # du[1] = rate_syn_II - rate_prothrombinase
    # Depends on: II(1), Va(12), Xa(16)
    append!(I, [1, 1, 1])
    append!(J, [1, 12, 16])

    # du[2] = -rate_IIa_Va
    # Depends on: V(2), IIa(11)
    append!(I, [2, 2])
    append!(J, [2, 11])

    # du[3] = rate_syn_VII (just turnover)
    # Depends on: VII(3)
    push!(I, 3); push!(J, 3)

    # du[4] = -rate_IIa_VIIIa
    # Depends on: VIII(4), IIa(11)
    append!(I, [4, 4])
    append!(J, [4, 11])

    # du[5] = rate_syn_IX - rate_VIIa_IXa - rate_XIa_IXa
    # Depends on: IX(5), VIIa(13), XIa(17)
    append!(I, [5, 5, 5])
    append!(J, [5, 13, 17])

    # du[6] = rate_syn_X - rate_VIIa_Xa - rate_tenase_Xa
    # Depends on: X(6), VIIa(13), IXa(15), VIIIa(14)
    append!(I, [6, 6, 6, 6])
    append!(J, [6, 13, 15, 14])

    # du[7] = -rate_IIa_XIa
    # Depends on: XI(7), IIa(11)
    append!(I, [7, 7])
    append!(J, [7, 11])

    # du[8] = 0 (XII constant)
    push!(I, 8); push!(J, 8)

    # du[9] = -rate_IIa_XIIIa
    # Depends on: XIII(9), IIa(11)
    append!(I, [9, 9])
    append!(J, [9, 11])

    # du[10] = -rate_fibrin
    # Depends on: Fg(10), IIa(11)
    append!(I, [10, 10])
    append!(J, [10, 11])

    # du[11] = rate_prothrombinase - rate_ATIII_IIa (dIIa/dt)
    # Depends on: II(1), Va(12), Xa(16), IIa(11), ATIII(21)
    append!(I, [11, 11, 11, 11, 11])
    append!(J, [1, 12, 16, 11, 21])

    # du[12] = rate_IIa_Va - rate_APC_Va (dVa/dt)
    # Depends on: V(2), IIa(11), Va(12), APC(23)
    append!(I, [12, 12, 12, 12])
    append!(J, [2, 11, 12, 23])

    # du[13] = 0 (VIIa constant)
    push!(I, 13); push!(J, 13)

    # du[14] = rate_IIa_VIIIa - rate_APC_VIIIa (dVIIIa/dt)
    # Depends on: VIII(4), IIa(11), VIIIa(14), APC(23)
    append!(I, [14, 14, 14, 14])
    append!(J, [4, 11, 14, 23])

    # du[15] = rate_VIIa_IXa + rate_XIa_IXa (dIXa/dt)
    # Depends on: IX(5), VIIa(13), XIa(17), IXa(15)
    append!(I, [15, 15, 15, 15])
    append!(J, [5, 13, 17, 15])

    # du[16] = rate_VIIa_Xa + rate_tenase_Xa - rate_ATIII_Xa - rate_TFPI_Xa (dXa/dt)
    # Depends on: X(6), VIIa(13), IXa(15), VIIIa(14), Xa(16), ATIII(21), TFPI(22)
    append!(I, [16, 16, 16, 16, 16, 16, 16])
    append!(J, [6, 13, 15, 14, 16, 21, 22])

    # du[17] = rate_IIa_XIa (dXIa/dt)
    # Depends on: XI(7), IIa(11), XIa(17)
    append!(I, [17, 17, 17])
    append!(J, [7, 11, 17])

    # du[18] = 0 (XIIa constant)
    push!(I, 18); push!(J, 18)

    # du[19] = rate_IIa_XIIIa (dXIIIa/dt)
    # Depends on: XIII(9), IIa(11), XIIIa(19)
    append!(I, [19, 19, 19])
    append!(J, [9, 11, 19])

    # du[20] = rate_fibrin (dFibrin/dt)
    # Depends on: Fg(10), IIa(11), fibrin(20)
    append!(I, [20, 20, 20])
    append!(J, [10, 11, 20])

    # du[21] = -rate_ATIII_IIa - rate_ATIII_Xa (dATIII/dt)
    # Depends on: ATIII(21), IIa(11), Xa(16)
    append!(I, [21, 21, 21])
    append!(J, [21, 11, 16])

    # du[22] = -rate_TFPI_Xa (dTFPI/dt)
    # Depends on: TFPI(22), Xa(16)
    append!(I, [22, 22])
    append!(J, [22, 16])

    # du[23] = rate_APC_formation (dAPC/dt)
    # Depends on: IIa(11), APC(23), protein_C (parameter)
    append!(I, [23, 23])
    append!(J, [11, 23])

    # du[24] = rate_ATIII_IIa (dTAT/dt)
    # Depends on: ATIII(21), IIa(11), TAT(24)
    append!(I, [24, 24, 24])
    append!(J, [21, 11, 24])

    # du[25] = rate_ATIII_Xa (dATIII_Xa/dt)
    # Depends on: ATIII(21), Xa(16), ATIII_Xa(25)
    append!(I, [25, 25, 25])
    append!(J, [21, 16, 25])

    # Create sparse matrix with placeholder values
    V = ones(Float64, length(I))

    return sparse(I, J, V, n, n)
end

"""
simulate_coagulation_stiff!(system, tspan; config, tissue_factor_pulse)

Simulate coagulation cascade using stiff ODE solver (Rodas4).

This is the preferred method for coagulation models due to:
- Stiffness ratio ~1000:1 (fast reactions vs slow factor turnover)
- Adaptive time stepping (10-100x fewer function evaluations)
- Better accuracy with unconditional stability

# Arguments
- `system`: CoagulationSystem to simulate
- `tspan`: (t_start, t_end) in seconds
- `config`: CoagulationSolverConfig (optional)
- `tissue_factor_pulse`: Function(t) → TF concentration (nM)

# Returns
- times: Vector of time points
- results: Vector of state dictionaries
"""
function simulate_coagulation_stiff!(
    system::CoagulationSystem,
    tspan::Tuple{Float64, Float64};
    config::CoagulationSolverConfig=CoagulationSolverConfig(),
    tissue_factor_pulse::Union{Function, Nothing}=nothing
)
    t_start, t_end = tspan

    # Initialize state vector
    f = system.factors
    u0 = [
        f.factor_II, f.factor_V, f.factor_VII, f.factor_VIII, f.factor_IX,
        f.factor_X, f.factor_XI, f.factor_XII, f.factor_XIII, f.fibrinogen,
        f.factor_IIa, f.factor_Va, f.factor_VIIa, f.factor_VIIIa, f.factor_IXa,
        f.factor_Xa, f.factor_XIa, f.factor_XIIa, f.factor_XIIIa,
        f.fibrin,
        f.antithrombin, f.tfpi, f.protein_C_active,
        f.ATIII_IIa, f.ATIII_Xa
    ]

    # Build parameters (potentially time-varying TF)
    base_tf = system.tissue_factor
    tf_func = isnothing(tissue_factor_pulse) ? (t -> base_tf) : tissue_factor_pulse

    params = (
        tissue_factor_func = tf_func,
        thrombomodulin = system.thrombomodulin,
        protein_C = system.factors.protein_C,
        kinetics = KINETIC_PARAMS,
        xa_inhibition = system.anticoagulant.xa_inhibition,
        iia_inhibition = system.anticoagulant.iia_inhibition,
        atiii_potentiation = system.anticoagulant.atiii_potentiation,
        vk_synthesis_rate = system.anticoagulant.vk_synthesis_rate
    )

    # ODE function wrapper for time-varying TF
    function coagulation_ode_wrapper!(du, u, p, t)
        # Create params with current TF value
        params_t = merge(p, (tissue_factor = p.tissue_factor_func(t),))
        coagulation_ode!(du, u, params_t, t)
    end

    # Build ODE function with optional sparse Jacobian
    if config.use_sparse
        jac_prototype = create_coagulation_jacobian_prototype()
        odefunc = ODEFunction(coagulation_ode_wrapper!; jac_prototype=jac_prototype)
    else
        odefunc = ODEFunction(coagulation_ode_wrapper!)
    end

    # Create ODE problem
    prob = ODEProblem(odefunc, u0, tspan, params)

    # Select solver algorithm
    alg = if config.algorithm == :rodas4
        Rodas4()
    elseif config.algorithm == :fbdf
        FBDF()
    elseif config.algorithm == :tsit5
        Tsit5()
    else
        Rodas4()  # Default to Rodas4 for stiff systems
    end

    # Build saveat points
    saveat = if !isempty(config.saveat)
        config.saveat
    else
        # Default: save at 0.5s intervals like Euler version
        collect(t_start:0.5:t_end)
    end

    # Solve with non-negativity constraint via callback
    function condition(u, t, integrator)
        any(u .< 0)
    end
    function affect!(integrator)
        integrator.u .= max.(integrator.u, 0.0)
    end
    cb = DiscreteCallback(condition, affect!)

    sol = solve(
        prob, alg;
        reltol=config.reltol,
        abstol=config.abstol,
        maxiters=config.maxiters,
        save_everystep=config.save_everystep,
        saveat=saveat,
        callback=cb
    )

    # Convert solution to result format
    times = sol.t
    n_steps = length(times)
    results = Vector{Dict{String, Float64}}(undef, n_steps)

    for (i, t) in enumerate(times)
        u = sol.u[i]
        results[i] = Dict(
            "time" => t,
            "thrombin_nM" => max(0.0, u[11]),
            "fibrin_nM" => max(0.0, u[20]),
            "factor_Xa_nM" => max(0.0, u[16]),
            "factor_IXa_nM" => max(0.0, u[15]),
            "factor_Va_nM" => max(0.0, u[12]),
            "factor_VIIIa_nM" => max(0.0, u[14]),
            "prothrombin_nM" => max(0.0, u[1]),
            "fibrinogen_nM" => max(0.0, u[10]),
            "ATIII_nM" => max(0.0, u[21]),
            "TAT_nM" => max(0.0, u[24]),
            "APC_nM" => max(0.0, u[23])
        )
    end

    # Update system with final state
    final_u = max.(sol.u[end], 0.0)
    update_system_from_state!(system, final_u)

    # Calculate thrombin generation parameters
    calculate_tg_parameters!(system, collect(times), results)

    return times, results
end

"""
simulate_coagulation!(system, tspan, dt; tissue_factor_pulse)

Simulate coagulation cascade over time.

# Arguments
- `system`: CoagulationSystem to simulate
- `tspan`: (t_start, t_end) in seconds
- `dt`: Time step in seconds
- `tissue_factor_pulse`: Function(t) → TF concentration (nM)

# Returns
- times: Vector of time points
- results: Vector of state dictionaries
"""
function simulate_coagulation!(
    system::CoagulationSystem,
    tspan::Tuple{Float64, Float64},
    dt::Float64;
    tissue_factor_pulse::Union{Function, Nothing}=nothing
)
    t_start, t_end = tspan
    times = collect(t_start:dt:t_end)
    n_steps = length(times)

    # Initialize state vector
    f = system.factors
    u = [
        f.factor_II, f.factor_V, f.factor_VII, f.factor_VIII, f.factor_IX,
        f.factor_X, f.factor_XI, f.factor_XII, f.factor_XIII, f.fibrinogen,
        f.factor_IIa, f.factor_Va, f.factor_VIIa, f.factor_VIIIa, f.factor_IXa,
        f.factor_Xa, f.factor_XIa, f.factor_XIIa, f.factor_XIIIa,
        f.fibrin,
        f.antithrombin, f.tfpi, f.protein_C_active,
        f.ATIII_IIa, f.ATIII_Xa
    ]

    du = zeros(25)

    # Parameters
    params = (
        tissue_factor = system.tissue_factor,
        thrombomodulin = system.thrombomodulin,
        protein_C = system.factors.protein_C,
        kinetics = KINETIC_PARAMS,
        xa_inhibition = system.anticoagulant.xa_inhibition,
        iia_inhibition = system.anticoagulant.iia_inhibition,
        atiii_potentiation = system.anticoagulant.atiii_potentiation,
        vk_synthesis_rate = system.anticoagulant.vk_synthesis_rate
    )

    # Store results
    results = Vector{Dict{String, Float64}}(undef, n_steps)

    # Simple Euler integration
    for (i, t) in enumerate(times)
        # Update tissue factor if pulse function provided
        if !isnothing(tissue_factor_pulse)
            params = merge(params, (tissue_factor = tissue_factor_pulse(t),))
        end

        # Store current state
        results[i] = Dict(
            "time" => t,
            "thrombin_nM" => u[11],
            "fibrin_nM" => u[20],
            "factor_Xa_nM" => u[16],
            "factor_IXa_nM" => u[15],
            "factor_Va_nM" => u[12],
            "factor_VIIIa_nM" => u[14],
            "prothrombin_nM" => u[1],
            "fibrinogen_nM" => u[10],
            "ATIII_nM" => u[21],
            "TAT_nM" => u[24],
            "APC_nM" => u[23]
        )

        # Compute derivatives
        coagulation_ode!(du, u, params, t)

        # Euler step
        u .= u .+ du .* dt

        # Enforce non-negativity
        u .= max.(u, 0.0)
    end

    # Update system with final state
    update_system_from_state!(system, u)

    # Calculate thrombin generation parameters
    calculate_tg_parameters!(system, times, results)

    return times, results
end

"""
update_system_from_state!(system, u)

Update CoagulationSystem from ODE state vector.
"""
function update_system_from_state!(system::CoagulationSystem, u::Vector{Float64})
    f = system.factors

    f.factor_II = u[1]
    f.factor_V = u[2]
    f.factor_VII = u[3]
    f.factor_VIII = u[4]
    f.factor_IX = u[5]
    f.factor_X = u[6]
    f.factor_XI = u[7]
    f.factor_XII = u[8]
    f.factor_XIII = u[9]
    f.fibrinogen = u[10]

    f.factor_IIa = u[11]
    f.factor_Va = u[12]
    f.factor_VIIa = u[13]
    f.factor_VIIIa = u[14]
    f.factor_IXa = u[15]
    f.factor_Xa = u[16]
    f.factor_XIa = u[17]
    f.factor_XIIa = u[18]
    f.factor_XIIIa = u[19]

    f.fibrin = u[20]

    f.antithrombin = u[21]
    f.tfpi = u[22]
    f.protein_C_active = u[23]

    f.ATIII_IIa = u[24]
    f.ATIII_Xa = u[25]

    return nothing
end

"""
calculate_tg_parameters!(system, times, results)

Calculate thrombin generation (TG) curve parameters.
"""
function calculate_tg_parameters!(
    system::CoagulationSystem,
    times::Vector{Float64},
    results::Vector{Dict{String, Float64}}
)
    thrombin = [r["thrombin_nM"] for r in results]

    # Find peak
    peak_idx = argmax(thrombin)
    system.peak_thrombin = thrombin[peak_idx]
    system.time_to_peak = times[peak_idx] / 60.0  # Convert to minutes

    # Lag time (time to reach 10 nM thrombin)
    lag_idx = findfirst(t -> t > 10.0, thrombin)
    system.lag_time = isnothing(lag_idx) ? Inf : times[lag_idx] / 60.0

    # ETP (area under curve, trapezoidal rule)
    dt = times[2] - times[1]
    system.etp = sum(thrombin) * dt / 60.0  # nM·min

    return nothing
end

# ============================================================================
# ANTICOAGULANT DRUG EFFECTS
# ============================================================================

"""
apply_warfarin!(system, S_conc, R_conc; genotype_vkorc1, genotype_cyp2c9)

Apply warfarin effect to coagulation system.

# Arguments
- `system`: CoagulationSystem
- `S_conc`: S-warfarin concentration (μM)
- `R_conc`: R-warfarin concentration (μM)
- `genotype_vkorc1`: "AA", "AB", "BB" (affects IC50)
- `genotype_cyp2c9`: "*1/*1", "*1/*2", etc. (affects metabolism)
"""
function apply_warfarin!(
    system::CoagulationSystem,
    S_conc::Float64,
    R_conc::Float64;
    genotype_vkorc1::String="AB",
    genotype_cyp2c9::String="*1/*1"
)
    ac = system.anticoagulant

    ac.warfarin_S_conc = S_conc
    ac.warfarin_R_conc = R_conc
    ac.vkorc1_genotype = genotype_vkorc1
    ac.cyp2c9_genotype = genotype_cyp2c9

    # Adjust IC50 for VKORC1 genotype
    ic50_base = WARFARIN_PARAMS["IC50_VKORC1"]
    ic50 = if genotype_vkorc1 == "AA"
        ic50_base * 0.5  # High sensitivity
    elseif genotype_vkorc1 == "BB"
        ic50_base * 1.5  # Low sensitivity
    else
        ic50_base  # AB (wild-type)
    end

    # Calculate VKORC1 inhibition (S-warfarin is more potent)
    # S-warfarin: potency = 1.0, R-warfarin: potency = 0.3
    effective_conc = S_conc + 0.3 * R_conc

    Emax = WARFARIN_PARAMS["Emax"]
    gamma = WARFARIN_PARAMS["gamma"]

    ac.vkorc1_inhibition = Emax * effective_conc^gamma / (ic50^gamma + effective_conc^gamma)
    ac.vk_synthesis_rate = 1.0 - ac.vkorc1_inhibition

    # Update clinical endpoints
    update_clinical_endpoints!(system)

    return nothing
end

"""
apply_doac!(system, drug_name, concentration)

Apply direct oral anticoagulant effect.

# Arguments
- `system`: CoagulationSystem
- `drug_name`: "rivaroxaban", "apixaban", "edoxaban", "betrixaban", "dabigatran"
- `concentration`: Drug concentration (nM)
"""
function apply_doac!(
    system::CoagulationSystem,
    drug_name::String,
    concentration::Float64
)
    ac = system.anticoagulant
    drug = lowercase(drug_name)

    if !haskey(DOAC_PARAMS, drug)
        error("Unknown DOAC: $drug_name")
    end

    params = DOAC_PARAMS[drug]
    ac.doac_type = drug
    ac.doac_conc = concentration

    if drug == "dabigatran"
        # Direct thrombin inhibitor
        Ki = params["Ki_IIa"]
        ac.iia_inhibition = concentration / (Ki + concentration)
        ac.xa_inhibition = 0.0
    else
        # Factor Xa inhibitors
        Ki = params["Ki_Xa"]
        ac.xa_inhibition = concentration / (Ki + concentration)
        ac.iia_inhibition = 0.0
    end

    # Update clinical endpoints
    update_clinical_endpoints!(system)

    return nothing
end

"""
apply_heparin!(system, concentration, heparin_type)

Apply heparin effect (potentiates ATIII).

# Arguments
- `system`: CoagulationSystem
- `concentration`: Heparin concentration (IU/mL)
- `heparin_type`: "UFH" (unfractionated) or "LMWH" (low molecular weight)
"""
function apply_heparin!(
    system::CoagulationSystem,
    concentration::Float64,
    heparin_type::String="UFH"
)
    ac = system.anticoagulant

    ac.heparin_conc = concentration
    ac.heparin_type = heparin_type

    # ATIII potentiation factor
    if heparin_type == "UFH"
        # UFH: 1000-fold potentiation of ATIII, affects both Xa and IIa
        ac.atiii_potentiation = 1.0 + concentration * 100.0  # Simplified
    else  # LMWH
        # LMWH: preferentially inhibits Xa over IIa
        ac.atiii_potentiation = 1.0 + concentration * 50.0
        # Additional direct Xa effect
        ac.xa_inhibition = min(0.5, concentration * 0.1)
    end

    # Update clinical endpoints
    update_clinical_endpoints!(system)

    return nothing
end

# ============================================================================
# CLINICAL ENDPOINTS
# ============================================================================

"""
calculate_pt_inr(system)

Calculate Prothrombin Time and INR.

PT depends on extrinsic pathway: factors II, V, VII, X and fibrinogen.
"""
function calculate_pt_inr(system::CoagulationSystem)::Tuple{Float64, Float64}
    f = system.factors

    # Factor activity as fraction of normal
    act_II = f.factor_II / NORMAL_FACTOR_CONCENTRATIONS["II"]
    act_V = f.factor_V / NORMAL_FACTOR_CONCENTRATIONS["V"]
    act_VII = f.factor_VII / NORMAL_FACTOR_CONCENTRATIONS["VII"]
    act_X = f.factor_X / NORMAL_FACTOR_CONCENTRATIONS["X"]
    act_Fg = f.fibrinogen / NORMAL_FACTOR_CONCENTRATIONS["fibrinogen"]

    # Apply drug effects
    ac = system.anticoagulant
    if ac.xa_inhibition > 0
        act_X *= (1.0 - ac.xa_inhibition)
    end
    if ac.iia_inhibition > 0
        act_II *= (1.0 - ac.iia_inhibition)
    end

    # Combined activity (weighted)
    # Factor VII is most important for PT
    pca = 0.35 * act_VII + 0.25 * act_X + 0.20 * act_II + 0.10 * act_V + 0.10 * act_Fg
    pca = clamp(pca, 0.01, 1.0)

    # PT (seconds)
    # Normal PT ≈ 12-13 s
    pt_normal = 12.5
    pt = pt_normal / pca

    # INR calculation
    # INR = (PT_patient / PT_normal)^ISI
    # ISI (International Sensitivity Index) ≈ 1.0-1.4
    ISI = 1.1
    inr = (pt / pt_normal)^ISI

    return (pt, inr)
end

"""
calculate_aptt(system)

Calculate activated Partial Thromboplastin Time.

aPTT depends on intrinsic pathway: factors VIII, IX, XI, XII
plus common pathway: II, V, X, fibrinogen.
"""
function calculate_aptt(system::CoagulationSystem)::Float64
    f = system.factors

    # Factor activity as fraction of normal
    act_VIII = f.factor_VIII / NORMAL_FACTOR_CONCENTRATIONS["VIII"]
    act_IX = f.factor_IX / NORMAL_FACTOR_CONCENTRATIONS["IX"]
    act_XI = f.factor_XI / NORMAL_FACTOR_CONCENTRATIONS["XI"]
    act_XII = f.factor_XII / NORMAL_FACTOR_CONCENTRATIONS["XII"]
    act_II = f.factor_II / NORMAL_FACTOR_CONCENTRATIONS["II"]
    act_V = f.factor_V / NORMAL_FACTOR_CONCENTRATIONS["V"]
    act_X = f.factor_X / NORMAL_FACTOR_CONCENTRATIONS["X"]

    # Apply drug effects
    ac = system.anticoagulant
    if ac.xa_inhibition > 0
        act_X *= (1.0 - ac.xa_inhibition)
    end
    if ac.iia_inhibition > 0
        act_II *= (1.0 - ac.iia_inhibition)
    end
    if ac.atiii_potentiation > 1.0
        # Heparin effect
        heparin_factor = 1.0 / ac.atiii_potentiation
        act_II *= heparin_factor
        act_X *= heparin_factor
    end

    # Combined intrinsic pathway activity
    intrinsic = 0.30 * act_VIII + 0.25 * act_IX + 0.20 * act_XI + 0.10 * act_XII
    common = 0.10 * act_II + 0.05 * act_X
    pca = intrinsic + common
    pca = clamp(pca, 0.01, 1.0)

    # aPTT (seconds)
    # Normal aPTT ≈ 28-32 s
    aptt_normal = 30.0
    aptt = aptt_normal / pca

    return aptt
end

"""
calculate_anti_xa(system)

Calculate Anti-Xa activity (IU/mL).

Used for monitoring LMWH, fondaparinux, and DOACs.
"""
function calculate_anti_xa(system::CoagulationSystem)::Float64
    ac = system.anticoagulant

    # For DOACs: calibrated to drug concentration
    if ac.doac_type != "none" && ac.doac_type != "dabigatran"
        # Anti-Xa activity correlates with drug concentration
        # Typical calibration: 100 ng/mL ≈ 0.5-1.0 IU/mL
        return ac.doac_conc / 100.0 * 0.7  # IU/mL
    end

    # For heparin: based on ATIII potentiation
    if ac.heparin_type != "none"
        if ac.heparin_type == "LMWH"
            return ac.heparin_conc * 0.8  # LMWH has high anti-Xa/anti-IIa ratio
        else  # UFH
            return ac.heparin_conc * 0.4
        end
    end

    return 0.0
end

"""
update_clinical_endpoints!(system)

Update all clinical endpoint calculations.
"""
function update_clinical_endpoints!(system::CoagulationSystem)
    pt, inr = calculate_pt_inr(system)
    system.pt_seconds = pt
    system.inr = inr
    system.aptt_seconds = calculate_aptt(system)
    system.anti_xa_IU = calculate_anti_xa(system)

    return nothing
end

# ============================================================================
# THROMBIN GENERATION ASSAY
# ============================================================================

"""
thrombin_generation_assay(system; tf_conc, duration_min)

Perform calibrated automated thrombogram (CAT) simulation.

# Arguments
- `system`: CoagulationSystem
- `tf_conc`: Tissue factor concentration for assay (nM), default 5 pM
- `duration_min`: Assay duration (minutes), default 60

# Returns
- Dict with TG parameters: lag_time, peak, ttp, etp
"""
function thrombin_generation_assay(
    system::CoagulationSystem;
    tf_conc::Float64=0.005,  # 5 pM
    duration_min::Float64=60.0
)::Dict{String, Float64}

    # Create copy of system for assay
    assay_system = deepcopy(system)
    assay_system.tissue_factor = tf_conc

    # Simulate
    tspan = (0.0, duration_min * 60.0)  # Convert to seconds
    dt = 0.5  # 0.5 second steps

    times, results = simulate_coagulation!(assay_system, tspan, dt)

    return Dict(
        "lag_time_min" => assay_system.lag_time,
        "peak_thrombin_nM" => assay_system.peak_thrombin,
        "time_to_peak_min" => assay_system.time_to_peak,
        "etp_nM_min" => assay_system.etp
    )
end

# ============================================================================
# STATE REPORTING
# ============================================================================

"""
get_coagulation_state(system)

Get comprehensive state report for coagulation system.
"""
function get_coagulation_state(system::CoagulationSystem)::Dict{String, Any}
    f = system.factors
    ac = system.anticoagulant

    return Dict(
        "factors" => Dict(
            "prothrombin_nM" => f.factor_II,
            "factor_V_nM" => f.factor_V,
            "factor_VII_nM" => f.factor_VII,
            "factor_VIII_nM" => f.factor_VIII,
            "factor_IX_nM" => f.factor_IX,
            "factor_X_nM" => f.factor_X,
            "fibrinogen_nM" => f.fibrinogen,
            "antithrombin_nM" => f.antithrombin
        ),

        "activated" => Dict(
            "thrombin_nM" => f.factor_IIa,
            "factor_Xa_nM" => f.factor_Xa,
            "factor_IXa_nM" => f.factor_IXa,
            "fibrin_nM" => f.fibrin,
            "APC_nM" => f.protein_C_active
        ),

        "clinical_endpoints" => Dict(
            "PT_seconds" => system.pt_seconds,
            "INR" => system.inr,
            "aPTT_seconds" => system.aptt_seconds,
            "Anti_Xa_IU_mL" => system.anti_xa_IU
        ),

        "thrombin_generation" => Dict(
            "lag_time_min" => system.lag_time,
            "peak_thrombin_nM" => system.peak_thrombin,
            "time_to_peak_min" => system.time_to_peak,
            "ETP_nM_min" => system.etp
        ),

        "anticoagulants" => Dict(
            "warfarin_S_uM" => ac.warfarin_S_conc,
            "VKORC1_inhibition" => ac.vkorc1_inhibition,
            "VK_synthesis_rate" => ac.vk_synthesis_rate,
            "DOAC_type" => ac.doac_type,
            "DOAC_conc_nM" => ac.doac_conc,
            "Xa_inhibition" => ac.xa_inhibition,
            "IIa_inhibition" => ac.iia_inhibition,
            "heparin_type" => ac.heparin_type,
            "ATIII_potentiation" => ac.atiii_potentiation
        ),

        "genetics" => Dict(
            "VKORC1_genotype" => ac.vkorc1_genotype,
            "CYP2C9_genotype" => ac.cyp2c9_genotype
        )
    )
end

end  # module Coagulation
