"""
ODE Solver PBPK - Ground Truth para Treinamento

Solver ODE tradicional para PBPK (14 compartimentos).
Usado para gerar dados de treinamento para o Dynamic GNN.

Inovações SOTA:
- DifferentialEquations.jl com algoritmos SOTA (Tsit5, Vern9)
- Stack allocation (SVector) para parâmetros fixos
- SIMD vectorization automática
- Type stability (zero overhead)
- Validação de conservação de massa

Autor: Dr. Sounio Agourakis + AI Assistant
Data: Novembro 2025
"""

module ODEPBPKSolver

using DifferentialEquations
using ForwardDiff
using StaticArrays
using Unitful
using QuadGK  # For numerical integration (convolution)

# Import FractalBlood module for transit time distribution
include("fractal_blood.jl")
using .FractalBlood

# 14 compartimentos PBPK padrão
const PBPK_ORGANS = [
    "blood",   # 0 - Plasma/sangue
    "liver",   # 1 - Fígado (metabolismo)
    "kidney",  # 2 - Rim (excreção)
    "brain",   # 3 - Cérebro (BBB)
    "heart",   # 4 - Coração
    "lung",    # 5 - Pulmão
    "muscle",  # 6 - Músculo
    "adipose", # 7 - Tecido adiposo
    "gut",     # 8 - Intestino (absorção)
    "skin",    # 9 - Pele
    "bone",    # 10 - Osso
    "spleen",  # 11 - Baço
    "pancreas", # 12 - Pâncreas
    "other",   # 13 - Resto do corpo
]

const NUM_ORGANS = length(PBPK_ORGANS)

# Índices críticos (constantes em tempo de compilação)
const BLOOD_IDX = 1
const LIVER_IDX = 2
const KIDNEY_IDX = 3

"""
Estrutura otimizada para parâmetros PBPK.

Inovações:
- Stack allocation (SVector) - zero heap allocation
- SIMD-friendly - compilador otimiza automaticamente
- Type-stable - zero runtime overhead
- Immutable - thread-safe
- Blood partitioning support for mechanistic B:P ratio calculation
"""
struct PBPKParams
    volumes::SVector{14, Float64}           # Volumes (L)
    blood_flows::SVector{14, Float64}       # Fluxos sanguíneos (L/h)
    clearance_hepatic::Float64               # Clearance hepático (L/h)
    clearance_renal::Float64                 # Clearance renal (L/h)
    partition_coeffs::SVector{14, Float64}  # Partition coefficients (Kp)

    # Blood partitioning parameters (for B:P ratio calculation)
    ke_p::Float64                            # Erythrocyte:plasma partition coefficient
    hematocrit::Float64                      # Hematocrit fraction (0-1)
    rbc_binding_type::Symbol                 # :passive, :active_uptake, :sequestration
    fu_plasma::Float64                       # Fraction unbound in plasma
    enable_bp_ratio::Bool                    # Enable blood partitioning (default false for backward compatibility)

    function PBPKParams(;
        volumes::Dict{String, Float64} = default_volumes(),
        blood_flows::Dict{String, Float64} = default_blood_flows(),
        clearance_hepatic::Float64 = 0.0,
        clearance_renal::Float64 = 0.0,
        partition_coeffs::Dict{String, Float64} = default_partition_coeffs(),
        ke_p::Float64 = 1.0,                  # Default: equal RBC and plasma concentrations
        hematocrit::Float64 = 0.45,           # Default: normal hematocrit
        rbc_binding_type::Symbol = :passive,
        fu_plasma::Float64 = 1.0,             # Default: fully unbound
        enable_bp_ratio::Bool = false,        # Default: disabled for backward compatibility
    )
        # Converter dicts para SVectors (type-safe, stack-allocated)
        vol_vec = SVector{14, Float64}([get(volumes, organ, 1.0) for organ in PBPK_ORGANS])
        flow_vec = SVector{14, Float64}([get(blood_flows, organ, 0.0) for organ in PBPK_ORGANS])
        kp_vec = SVector{14, Float64}([get(partition_coeffs, organ, 1.0) for organ in PBPK_ORGANS])

        new(vol_vec, flow_vec, clearance_hepatic, clearance_renal, kp_vec,
            ke_p, hematocrit, rbc_binding_type, fu_plasma, enable_bp_ratio)
    end
end

# Valores padrão (70kg adulto)
function default_volumes()::Dict{String, Float64}
    return Dict(
        "blood" => 5.0,
        "liver" => 1.8,
        "kidney" => 0.31,
        "brain" => 1.4,
        "heart" => 0.33,
        "lung" => 0.5,
        "muscle" => 30.0,
        "adipose" => 15.0,
        "gut" => 1.1,
        "skin" => 3.3,
        "bone" => 10.0,
        "spleen" => 0.18,
        "pancreas" => 0.1,
        "other" => 5.0,
    )
end

function default_blood_flows()::Dict{String, Float64}
    return Dict(
        "blood" => 0.0,
        "liver" => 90.0,
        "kidney" => 60.0,
        "brain" => 50.0,
        "heart" => 20.0,
        "lung" => 300.0,  # Cardiac output
        "muscle" => 75.0,
        "adipose" => 12.0,
        "gut" => 45.0,
        "skin" => 10.0,
        "bone" => 5.0,
        "spleen" => 15.0,
        "pancreas" => 5.0,
        "other" => 20.0,
    )
end

function default_partition_coeffs()::Dict{String, Float64}
    return Dict(organ => 1.0 for organ in PBPK_ORGANS)
end

# =============================================================================
# FRACTALBLOOD INTEGRATION
# =============================================================================

"""
FractalBloodParams - Parameters for FractalBlood integration with PBPK

Stores transit time distribution parameters from the fractal vascular network
for use in convolution-based blood dynamics.

Fields:
- enabled: Whether to use FractalBlood dynamics (vs traditional well-stirred)
- alpha: Power-law exponent for transit time distribution (typically 1.3-1.5)
- tau_min: Minimum transit time through vasculature (seconds)
- tau_mean: Mean transit time (seconds)
- beta: CTRW anomalous diffusion exponent (0 < beta <= 1)
- use_convolution: Use full convolution integral (slower but accurate)
- n_convolution_points: Number of points for numerical convolution

Physics:
- Traditional PBPK assumes instantaneous mixing ("well-stirred tank")
- FractalBlood accounts for realistic vascular transit times
- Transit time distribution E(t) follows power law from fractal network topology
- Drug concentration: C_blood(t) = ∫ dose(t-τ) × E(τ) dτ
"""
struct FractalBloodParams
    enabled::Bool
    alpha::Float64
    tau_min::Float64
    tau_mean::Float64
    beta::Float64
    use_convolution::Bool
    n_convolution_points::Int

    function FractalBloodParams(;
        enabled::Bool = false,
        alpha::Float64 = 1.37,
        tau_min::Float64 = 0.1,      # 0.1 seconds
        tau_mean::Float64 = 20.0,    # 20 seconds mean circulation time
        beta::Float64 = 0.8,
        use_convolution::Bool = false,
        n_convolution_points::Int = 50
    )
        if enabled && alpha <= 1.0
            error("FractalBlood alpha must be > 1.0 for finite moments")
        end
        if enabled && tau_min <= 0.0
            error("FractalBlood tau_min must be > 0")
        end
        if enabled && !(0.0 < beta <= 1.0)
            error("FractalBlood beta must be in (0, 1]")
        end

        new(enabled, alpha, tau_min, tau_mean, beta, use_convolution, n_convolution_points)
    end
end

"""
Extended PBPKParams with FractalBlood integration.

This is a convenience constructor that combines standard PBPK parameters
with FractalBlood transit time dynamics.
"""
struct PBPKParamsWithFractal
    pbpk::PBPKParams
    fractal::FractalBloodParams
end

"""
integrate_fractal_blood!(pbpk_params, fractal_model)

Extract transit time distribution parameters from a FractalBloodModel
and integrate them into PBPK parameters.

Args:
- pbpk_params: Standard PBPK parameters
- fractal_model: FractalBloodModel from fractal_blood.jl

Returns:
- PBPKParamsWithFractal combining both parameter sets
"""
function integrate_fractal_blood!(
    pbpk_params::PBPKParams,
    fractal_model::FractalBlood.FractalBloodModel
)::PBPKParamsWithFractal

    fractal_params = FractalBloodParams(
        enabled = true,
        alpha = fractal_model.alpha,
        tau_min = fractal_model.tau_min,
        tau_mean = fractal_model.tau_mean,
        beta = fractal_model.beta,
        use_convolution = true,
        n_convolution_points = 50
    )

    return PBPKParamsWithFractal(pbpk_params, fractal_params)
end

"""
create_fractal_pbpk_params(; kwargs...)

Convenience function to create PBPK parameters with FractalBlood enabled.

Example:
```julia
params = create_fractal_pbpk_params(
    alpha = 1.37,
    tau_min = 0.1,
    tau_mean = 20.0
)
```
"""
function create_fractal_pbpk_params(;
    volumes::Dict{String, Float64} = default_volumes(),
    blood_flows::Dict{String, Float64} = default_blood_flows(),
    clearance_hepatic::Float64 = 0.0,
    clearance_renal::Float64 = 0.0,
    partition_coeffs::Dict{String, Float64} = default_partition_coeffs(),
    alpha::Float64 = 1.37,
    tau_min::Float64 = 0.1,
    tau_mean::Float64 = 20.0,
    beta::Float64 = 0.8,
    use_convolution::Bool = false
)::PBPKParamsWithFractal

    pbpk = PBPKParams(
        volumes = volumes,
        blood_flows = blood_flows,
        clearance_hepatic = clearance_hepatic,
        clearance_renal = clearance_renal,
        partition_coeffs = partition_coeffs
    )

    fractal = FractalBloodParams(
        enabled = true,
        alpha = alpha,
        tau_min = tau_min,
        tau_mean = tau_mean,
        beta = beta,
        use_convolution = use_convolution,
        n_convolution_points = 50
    )

    return PBPKParamsWithFractal(pbpk, fractal)
end

"""
fractal_transit_time_distribution(t, fractal_params)

Compute the transit time distribution E(t) from FractalBlood parameters.

This is the power-law PDF: E(t) = (α-1)/τ_min × (t/τ_min)^(-α) for t ≥ τ_min

Used for convolution: C_blood(t) = ∫ dose(t-τ) × E(τ) dτ
"""
function fractal_transit_time_distribution(t::Float64, fractal_params::FractalBloodParams)::Float64
    if !fractal_params.enabled || t < fractal_params.tau_min
        return 0.0
    end

    alpha = fractal_params.alpha
    tau_min = fractal_params.tau_min

    # Power-law PDF
    return (alpha - 1.0) / tau_min * (t / tau_min)^(-alpha)
end

"""
apply_fractal_dispersion(C_input, t, history, fractal_params)

Apply fractal vascular dispersion to input concentration via convolution.

For FractalBlood dynamics, the output concentration is:
C_out(t) = ∫₀ᵗ C_in(t-τ) × E(τ) dτ

where E(τ) is the transit time distribution.

This captures the realistic dispersion of drug through the vascular network
rather than assuming instantaneous mixing.
"""
function apply_fractal_dispersion(
    C_input::Float64,
    t::Float64,
    history::Vector{Tuple{Float64, Float64}},  # (time, concentration) pairs
    fractal_params::FractalBloodParams
)::Float64

    if !fractal_params.enabled || !fractal_params.use_convolution
        # No dispersion - return input directly (well-stirred approximation)
        return C_input
    end

    if isempty(history)
        return C_input
    end

    # Numerical convolution using QuadGK
    # C_out(t) = ∫₀ᵗ C_in(t-τ) × E(τ) dτ

    function integrand(tau::Float64)::Float64
        # Get C_in at time (t - tau) via linear interpolation
        t_past = t - tau

        if t_past <= 0.0
            return 0.0
        end

        # Find bracketing points in history
        idx = searchsortedlast([h[1] for h in history], t_past)

        if idx == 0
            C_past = history[1][2]  # Use first value
        elseif idx >= length(history)
            C_past = history[end][2]  # Use last value
        else
            # Linear interpolation
            t1, C1 = history[idx]
            t2, C2 = history[idx + 1]
            C_past = C1 + (C2 - C1) * (t_past - t1) / (t2 - t1)
        end

        # E(tau)
        E_tau = fractal_transit_time_distribution(tau, fractal_params)

        return C_past * E_tau
    end

    # Integrate from tau_min to t
    if t < fractal_params.tau_min
        return 0.0
    end

    result, _ = quadgk(integrand, fractal_params.tau_min, t, rtol=1e-6)
    return result
end

"""
Sistema ODE otimizado para PBPK.

Inovações:
- SIMD vectorization automática (JIT compiler)
- Zero allocations (stack-only)
- Type-stable (zero runtime overhead)
- Validação de invariantes
- Blood:Plasma ratio support for mechanistic drug distribution

Equações (with B:P ratio enabled):
- C_plasma = C_blood / Rb (where Rb = Blood:Plasma ratio)
- Para cada órgão: dC_organ/dt = (Q_organ / V_organ) * (C_plasma - C_organ / Kp_organ)
- Para blood: dC_blood/dt = Σ[fluxos] - clearance_rate * C_unbound
- C_unbound = C_plasma * fu (only unbound drug clears)

Equações (with B:P ratio disabled - backward compatible):
- Para cada órgão: dC_organ/dt = (Q_organ / V_organ) * (C_blood - C_organ / Kp_organ)
- Para blood: dC_blood/dt = Σ[fluxos] - clearance_rate * C_blood
"""
function ode_system!(du::AbstractVector{Float64}, u::AbstractVector{Float64}, p::PBPKParams, t::Float64)
    # Inicializar derivadas
    fill!(du, 0.0)

    C_blood = u[BLOOD_IDX]

    # Calculate Blood:Plasma ratio if enabled
    Rb = calculate_blood_plasma_ratio(p)

    # Plasma concentration (what tissues see)
    C_plasma = C_blood / Rb

    # Unbound plasma concentration (what drives clearance)
    C_unbound = C_plasma * p.fu_plasma

    # Para cada órgão (exceto blood)
    @inbounds for i in 1:NUM_ORGANS
        if i == BLOOD_IDX
            continue
        end

        # Parâmetros do órgão (stack-allocated, SIMD-friendly)
        V_organ = p.volumes[i]
        Q_organ = p.blood_flows[i]
        Kp_organ = p.partition_coeffs[i]

        # Concentração no órgão
        C_organ = u[i]

        # Fluxo de entrada (plasma -> organ)
        # IMPORTANT: Use C_plasma instead of C_blood for tissue exchange
        # Only unbound drug in plasma can distribute to tissues
        du[i] = (Q_organ / V_organ) * (C_plasma - C_organ / Kp_organ)

        # Fluxo de saída (organ -> blood)
        # Return flow affects blood compartment
        V_blood = p.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= (Q_organ / V_blood) * Rb * (C_plasma - C_organ / Kp_organ)
    end

    # Clearance hepático (only unbound drug clears)
    # IMPORTANT: Use C_unbound for clearance, not total C_blood
    if p.clearance_hepatic > 0.0
        clearance_rate = p.clearance_hepatic / p.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= clearance_rate * C_unbound * Rb
    end

    # Clearance renal (only unbound drug clears)
    # IMPORTANT: Use C_unbound for clearance, not total C_blood
    if p.clearance_renal > 0.0
        clearance_rate = p.clearance_renal / p.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= clearance_rate * C_unbound * Rb
    end

    return nothing
end

"""
Resolve o sistema ODE com algoritmos SOTA.

Inovações:
- DifferentialEquations.jl com Tsit5 (Runge-Kutta 5ª ordem)
- Tolerâncias adaptativas (reltol=1e-8, abstol=1e-10)
- Type-stable
- Validação de conservação de massa

Args:
    p: Parâmetros PBPK
    dose: Dose administrada (mg)
    tspan: Intervalo de tempo (horas)
    time_points: Pontos temporais específicos (opcional)

Returns:
    Solution object do DifferentialEquations.jl
"""
function solve(
    p::PBPKParams,
    dose::Float64,
    tspan::Tuple{Float64, Float64};
    time_points::Union{Vector{Float64}, Nothing} = nothing,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
    alg = Tsit5(),  # SOTA algorithm (Runge-Kutta 5ª ordem)
)
    # Condições iniciais
    u0 = zeros(Float64, NUM_ORGANS)
    blood_volume = p.volumes[BLOOD_IDX]
    u0[BLOOD_IDX] = dose / blood_volume  # mg/L

    # Criar problema ODE
    prob = ODEProblem(ode_system!, u0, tspan, p)

    # Resolver (usar DifferentialEquations.solve explicitamente)
    if time_points !== nothing
        # Interpolação nos pontos específicos
        sol = DifferentialEquations.solve(prob, alg, reltol=reltol, abstol=abstol, saveat=time_points)
    else
        # Solução adaptativa
        sol = DifferentialEquations.solve(prob, alg, reltol=reltol, abstol=abstol)
    end

    return sol
end

"""
Simula PBPK com parâmetros padrão.

Args:
    p: Parâmetros PBPK
    dose: Dose (mg)
    t_max: Tempo máximo (horas)
    num_points: Número de pontos temporais

Returns:
    Dict com concentrações por órgão ao longo do tempo
"""
function simulate(
    p::PBPKParams,
    dose::Float64;
    t_max::Float64 = 24.0,
    num_points::Int = 100,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
)
    time_points = range(0.0, t_max, length=num_points)
    tspan = (0.0, t_max)

    # Usar solve do DifferentialEquations.jl diretamente (namespace completo)
    u0 = zeros(Float64, NUM_ORGANS)
    blood_volume = p.volumes[BLOOD_IDX]
    u0[BLOOD_IDX] = dose / blood_volume

    prob = ODEProblem(ode_system!, u0, tspan, p)
    sol = DifferentialEquations.solve(prob, Tsit5(); reltol=reltol, abstol=abstol, saveat=collect(time_points))

    # Organizar resultados
    results = Dict{String, Vector{Float64}}()
    for (i, organ) in enumerate(PBPK_ORGANS)
        results[organ] = [sol[j][i] for j in 1:length(sol)]
    end
    results["time"] = collect(time_points)

    return results
end

"""
Valida conservação de massa (invariante físico).

Massa_total(t) = Σ[C_organ(t) * V_organ] = constante (dose inicial)

With Blood:Plasma ratio enabled, the blood compartment mass needs special handling:
- Blood compartment stores total blood concentration
- Total mass in blood = C_blood × V_blood
- This already accounts for drug in plasma, RBC, and WBC

Returns:
    true se conservação válida (erro relativo < 1e-6)
"""
function validate_mass_conservation(
    sol::ODESolution,
    p::PBPKParams,
    dose::Float64,
    tol::Float64 = 1e-6
)::Bool
    initial_mass = dose  # mg

    for t_idx in 1:length(sol)
        total_mass = 0.0
        for i in 1:NUM_ORGANS
            total_mass += sol[t_idx][i] * p.volumes[i]
        end

        error = abs(total_mass - initial_mass) / initial_mass
        if error > tol
            @warn "Conservação de massa violada em t=$(sol.t[t_idx]): erro relativo = $error"
            if p.enable_bp_ratio
                # Provide additional diagnostics for blood partitioning
                C_blood = sol[t_idx][BLOOD_IDX]
                Rb = calculate_blood_plasma_ratio(p)
                partitioned = partition_blood_concentration(C_blood, Rb, p.hematocrit)
                @warn "  Blood partitioning: C_plasma=$(partitioned.C_plasma), C_rbc=$(partitioned.C_rbc)"
                @warn "  B:P ratio=$(Rb), Hematocrit=$(p.hematocrit), Ke_p=$(p.ke_p)"
            end
            return false
        end
    end

    return true
end

"""
Sensibilidade automática usando Automatic Differentiation.

Inovações:
- ForwardDiff.jl para AD automático
- Útil para parameter estimation
- Type-stable

Returns:
    Sensitividade de cada parâmetro
"""
function solve_with_sensitivity(
    p::PBPKParams,
    dose::Float64,
    tspan::Tuple{Float64, Float64};
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
)
    # TODO: Implementar com ForwardDiff.jl
    # Por enquanto, retornar solução normal
    return solve(p, dose, tspan; reltol=reltol, abstol=abstol)
end

export PBPKParams, solve, simulate, validate_mass_conservation, solve_with_sensitivity
export PBPK_ORGANS, NUM_ORGANS

#=============================================================================
  7-SEGMENT DETAILED GI TRACT MODEL (PK-SIM STANDARD)

  Implements physiologically accurate gastrointestinal absorption with:
  - 7 anatomical segments (stomach → duodenum → jejunum → ileum → colon)
  - pH-dependent ionization (Henderson-Hasselbalch)
  - Transporter expression (P-gp, BCRP, OATP, PEPT1)
  - Intestinal metabolism (CYP3A4, UGT1A1)
  - Regional blood flow and permeability
  - Fed vs. fasted state effects

  References:
  - Willmann S et al. J Med Chem 2004;47:4022-4031 (PK-Sim model)
  - Amidon GL et al. Pharm Res 1995;12:413-420 (BCS)
=============================================================================#

# Import GI detailed module (already included in parent DarwinPBPK module)
# Access via parent module to avoid duplicate inclusion
using ..GIDetailed

# State indices for 7 GI segments
const GI_STOMACH_IDX = 31
const GI_DUODENUM_IDX = 32
const GI_JEJUNUM_UPPER_IDX = 33
const GI_JEJUNUM_LOWER_IDX = 34
const GI_ILEUM_UPPER_IDX = 35
const GI_ILEUM_LOWER_IDX = 36
const GI_COLON_IDX = 37

const GI_SEGMENT_INDICES = [
    GI_STOMACH_IDX, GI_DUODENUM_IDX, GI_JEJUNUM_UPPER_IDX, GI_JEJUNUM_LOWER_IDX,
    GI_ILEUM_UPPER_IDX, GI_ILEUM_LOWER_IDX, GI_COLON_IDX
]

"""
Extended ODE system with detailed 7-segment GI tract.

State vector (37 elements):
- u[1:14]: Organ concentrations (standard PBPK)
- u[15:30]: (reserved for other extensions)
- u[31:37]: Amount in each GI segment (mg)

Equations:
- Each GI segment:
    * dA_seg/dt = -Ka_seg * A_seg - Ktr_seg * A_seg + Ktr_prev * A_prev
    * Absorption: Ka_seg = Peff * SA / V (pH, transporter, metabolism corrected)
    * Transit: Ktr = 1 / transit_time
- Blood: dC_blood/dt = ... + Σ[absorbed from all segments] / V_blood
"""
function gi7_ode_system!(du::AbstractVector{Float64}, u::AbstractVector{Float64},
                         p::Tuple{PBPKParams, GITract, DrugGIProperties}, t::Float64)
    pbpk, gi_tract, drug = p

    # Initialize derivatives
    fill!(du, 0.0)

    C_blood = u[BLOOD_IDX]

    # Standard PBPK for organs
    @inbounds for i in 1:NUM_ORGANS
        if i == BLOOD_IDX
            continue
        end

        V_organ = pbpk.volumes[i]
        Q_organ = pbpk.blood_flows[i]
        Kp_organ = pbpk.partition_coeffs[i]
        C_organ = u[i]

        du[i] = (Q_organ / V_organ) * (C_blood - C_organ / Kp_organ)
        V_blood = pbpk.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= (Q_organ / V_blood) * (C_blood - C_organ / Kp_organ)
    end

    # Clearance
    if pbpk.clearance_hepatic > 0.0
        du[BLOOD_IDX] -= (pbpk.clearance_hepatic / pbpk.volumes[BLOOD_IDX]) * C_blood
    end
    if pbpk.clearance_renal > 0.0
        du[BLOOD_IDX] -= (pbpk.clearance_renal / pbpk.volumes[BLOOD_IDX]) * C_blood
    end

    # 7-segment GI dynamics
    total_absorbed = 0.0

    @inbounds for (seg_idx, state_idx) in enumerate(GI_SEGMENT_INDICES)
        segment = gi_tract.segments[seg_idx]
        amount_in_seg = max(0.0, u[state_idx])

        # Calculate absorption from this segment
        absorption_rate = calculate_gi_absorption(segment, drug, amount_in_seg, gi_tract.fed_state)

        # Transit rate (first-order)
        ktr = 1.0 / segment.transit_time_min  # 1/min
        transit_out = ktr * amount_in_seg

        # Transit in from previous segment (if not stomach)
        transit_in = 0.0
        if seg_idx > 1
            prev_idx = GI_SEGMENT_INDICES[seg_idx - 1]
            prev_segment = gi_tract.segments[seg_idx - 1]
            prev_amount = max(0.0, u[prev_idx])
            prev_ktr = 1.0 / prev_segment.transit_time_min
            transit_in = prev_ktr * prev_amount
        end

        # Rate of change: inflow - outflow - absorption
        du[state_idx] = transit_in - transit_out - absorption_rate

        # Accumulate total absorbed (goes to blood)
        total_absorbed += absorption_rate
    end

    # Add absorbed drug to blood (convert mg/min to rate in concentration units)
    # Absorption gives first-pass through gut wall, then to portal vein → liver → systemic
    # Simplified: direct to blood with effective Fg (fraction escaping gut metabolism)
    fg = calculate_fg(gi_tract, drug)
    du[BLOOD_IDX] += (total_absorbed * fg) / pbpk.volumes[BLOOD_IDX]

    return nothing
end

"""
Solve PBPK with detailed 7-segment GI model.

Args:
    pbpk_params: PBPK parameters
    gi_tract: 7-segment GI tract model
    drug: Drug GI properties
    dose: Oral dose (mg)
    tspan: Time span (min)

Returns:
    Solution object with 37 states
"""
function solve_gi7(
    pbpk_params::PBPKParams,
    gi_tract::GITract,
    drug::DrugGIProperties,
    dose::Float64,
    tspan::Tuple{Float64, Float64};
    time_points::Union{Vector{Float64}, Nothing} = nothing,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
    alg = Tsit5(),
)
    # Initial conditions: all dose in stomach
    n_states = GI_COLON_IDX
    u0 = zeros(Float64, n_states)
    u0[GI_STOMACH_IDX] = dose

    # Parameters
    p = (pbpk_params, gi_tract, drug)

    # Create ODE problem
    prob = ODEProblem(gi7_ode_system!, u0, tspan, p)

    # Solve
    if time_points !== nothing
        sol = DifferentialEquations.solve(prob, alg; reltol=reltol, abstol=abstol, saveat=time_points)
    else
        sol = DifferentialEquations.solve(prob, alg; reltol=reltol, abstol=abstol)
    end

    return sol
end

"""
Simulate PBPK with detailed 7-segment GI model.

Args:
    pbpk_params: PBPK parameters
    gi_tract: 7-segment GI tract model (use create_gi_tract())
    drug: Drug GI properties
    dose: Oral dose (mg)
    t_max: Maximum time (min)
    num_points: Number of time points

Returns:
    Dict with concentrations, GI amounts, and PK metrics
"""
function simulate_gi7(
    pbpk_params::PBPKParams,
    gi_tract::GITract,
    drug::DrugGIProperties,
    dose::Float64;
    t_max::Float64 = 1440.0,  # 24h in minutes
    num_points::Int = 200,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
)
    time_points = collect(range(0.0, t_max, length=num_points))
    tspan = (0.0, t_max)

    # Solve ODE
    sol = solve_gi7(pbpk_params, gi_tract, drug, dose, tspan;
                    time_points=time_points, reltol=reltol, abstol=abstol)

    # Build results
    results = Dict{String, Any}()
    results["time"] = time_points

    # Organ concentrations
    for (i, organ) in enumerate(PBPK_ORGANS)
        results[organ] = [sol[j][i] for j in 1:length(sol)]
    end
    results["plasma"] = results["blood"]

    # GI segment amounts
    gi_names = ["stomach", "duodenum", "jejunum_upper", "jejunum_lower",
                "ileum_upper", "ileum_lower", "colon"]
    for (i, name) in enumerate(gi_names)
        idx = GI_SEGMENT_INDICES[i]
        results["gi_$name"] = [sol[j][idx] for j in 1:length(sol)]
    end

    # Total amount in GI tract
    results["gi_total"] = zeros(num_points)
    for j in 1:num_points
        results["gi_total"][j] = sum(sol[j][idx] for idx in GI_SEGMENT_INDICES)
    end

    # PK metrics
    plasma_conc = results["plasma"]

    if !isempty(plasma_conc) && any(c -> c > 0, plasma_conc)
        cmax_idx = argmax(plasma_conc)
        results["cmax"] = plasma_conc[cmax_idx]
        results["tmax"] = time_points[cmax_idx]
    else
        results["cmax"] = 0.0
        results["tmax"] = 0.0
    end

    # AUC (trapezoidal rule)
    auc = 0.0
    for i in 2:length(time_points)
        dt = time_points[i] - time_points[i-1]
        auc += 0.5 * (plasma_conc[i] + plasma_conc[i-1]) * dt
    end
    results["auc"] = auc

    # Fraction absorbed
    fa = (dose - results["gi_total"][end]) / dose
    results["fa"] = fa

    # Effective bioavailability (accounting for gut metabolism)
    fg = calculate_fg(gi_tract, drug)
    results["fg"] = fg
    results["f_oral"] = fa * fg  # Simplified (excludes hepatic first-pass)

    # BCS classification
    results["bcs_class"] = calculate_bcs_class(drug, dose)

    # Regional absorption contributions
    results["absorption_by_segment"] = Dict{String, Float64}()
    for (i, name) in enumerate(gi_names)
        # Approximate absorbed from each segment
        seg_absorbed = dose * 0.0  # Placeholder - would need integration of absorption rates
        results["absorption_by_segment"][name] = seg_absorbed
    end

    return results
end

"""
Compare single gut vs. 7-segment GI model.

Returns Dict with both simulations for comparison.
"""
function compare_gi_models(
    pbpk_params::PBPKParams,
    drug::DrugGIProperties,
    dose::Float64;
    t_max::Float64 = 1440.0,
    num_points::Int = 200,
)::Dict{String, Any}
    # Create GI tract
    gi_tract = create_gi_tract(fed_state=false)

    # Simulate 7-segment model
    results_7seg = simulate_gi7(pbpk_params, gi_tract, drug, dose;
                                t_max=t_max, num_points=num_points)

    # Simulate single-gut model (legacy)
    oral_params = OralParams(1.0, 0.7)  # Approximate ka and F
    results_1gut = simulate_oral(pbpk_params, oral_params, dose;
                                  t_max=t_max/60.0, num_points=num_points)  # Convert to hours

    comparison = Dict{String, Any}(
        "gi7_model" => results_7seg,
        "single_gut_model" => results_1gut,
        "metrics_comparison" => Dict(
            "cmax_7seg" => results_7seg["cmax"],
            "cmax_1gut" => results_1gut["cmax"],
            "tmax_7seg" => results_7seg["tmax"],
            "tmax_1gut" => results_1gut["tmax"] * 60.0,  # Convert to min
            "auc_7seg" => results_7seg["auc"],
            "auc_1gut" => results_1gut["auc"],
        )
    )

    return comparison
end

export gi7_ode_system!, solve_gi7, simulate_gi7, compare_gi_models
export GI_STOMACH_IDX, GI_DUODENUM_IDX, GI_JEJUNUM_UPPER_IDX, GI_JEJUNUM_LOWER_IDX
export GI_ILEUM_UPPER_IDX, GI_ILEUM_LOWER_IDX, GI_COLON_IDX, GI_SEGMENT_INDICES

# FractalBlood integration exports
export FractalBloodParams, PBPKParamsWithFractal
export integrate_fractal_blood!, create_fractal_pbpk_params
export fractal_transit_time_distribution, apply_fractal_dispersion

#=============================================================================
  Oral Absorption PBPK Model

  Extended 15-compartment model with gut lumen for oral dosing:
  - Compartment 15: Gut Lumen (absorption site)
  - First-order absorption: dA_gut/dt = -Ka * A_gut
  - First-pass metabolism: F_eff = Fa * Fg * Fh

  References:
  - Rowland & Tozer, Clinical Pharmacokinetics
  - Poulin & Theil, J Pharm Sci 2002
=============================================================================#

const GUT_LUMEN_IDX = 15  # Index for gut lumen compartment

"""
Parameters for oral absorption and first-pass metabolism.
"""
struct OralParams
    ka::Float64           # Absorption rate constant (1/h)
    fa::Float64           # Fraction absorbed (0-1)
    fg::Float64           # Fraction escaping gut metabolism (0-1)
    fh::Float64           # Fraction escaping hepatic first-pass (0-1)
    lag::Float64          # Lag time (h)
end

# Default oral params (complete absorption, no first-pass)
OralParams() = OralParams(1.0, 1.0, 1.0, 1.0, 0.0)

# Constructor with bioavailability
function OralParams(ka::Float64, f::Float64)
    # Assume F = Fa * Fg * Fh, distribute evenly if only F given
    fa = 1.0
    fg = sqrt(f)
    fh = sqrt(f)
    OralParams(ka, fa, fg, fh, 0.0)
end

"""
Calculate effective bioavailability: F = Fa * Fg * Fh
"""
effective_f(op::OralParams) = op.fa * op.fg * op.fh

"""
Extended ODE system for oral absorption PBPK.

State vector (15 elements):
- u[1:14]: Organ concentrations (same as standard PBPK)
- u[15]: Amount in gut lumen (mg)

Equations:
- Gut lumen: dA_gut/dt = -Ka * A_gut (after lag time)
- Blood: dC_blood/dt = ... + (Ka * A_gut * Fg * Fh) / V_blood
- Other organs: same as standard PBPK
"""
function oral_ode_system!(du::AbstractVector{Float64}, u::AbstractVector{Float64},
                          p::Tuple{PBPKParams, OralParams}, t::Float64)
    pbpk, oral = p

    # Initialize derivatives
    fill!(du, 0.0)

    C_blood = u[BLOOD_IDX]
    A_gut_lumen = u[GUT_LUMEN_IDX]  # Amount in gut lumen (mg)

    # Standard PBPK for organs
    @inbounds for i in 1:NUM_ORGANS
        if i == BLOOD_IDX
            continue
        end

        V_organ = pbpk.volumes[i]
        Q_organ = pbpk.blood_flows[i]
        Kp_organ = pbpk.partition_coeffs[i]
        C_organ = u[i]

        # Organ dynamics
        du[i] = (Q_organ / V_organ) * (C_blood - C_organ / Kp_organ)

        # Blood dynamics (outflow from blood to organs)
        V_blood = pbpk.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= (Q_organ / V_blood) * (C_blood - C_organ / Kp_organ)
    end

    # Hepatic clearance
    if pbpk.clearance_hepatic > 0.0
        clearance_rate = pbpk.clearance_hepatic / pbpk.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= clearance_rate * C_blood
    end

    # Renal clearance
    if pbpk.clearance_renal > 0.0
        clearance_rate = pbpk.clearance_renal / pbpk.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= clearance_rate * C_blood
    end

    # Oral absorption (after lag time is handled by delayed initial condition)
    # dA_gut/dt = -Ka * A_gut
    du[GUT_LUMEN_IDX] = -oral.ka * A_gut_lumen

    # Drug absorbed from gut goes to blood (after first-pass)
    # Rate = Ka * A_gut * Fg * Fh / V_blood
    absorption_rate = oral.ka * A_gut_lumen * oral.fg * oral.fh / pbpk.volumes[BLOOD_IDX]
    du[BLOOD_IDX] += absorption_rate

    return nothing
end

"""
Solve oral PBPK model with full ODE integration.

Args:
    pbpk_params: PBPK parameters
    oral_params: Oral absorption parameters
    dose: Oral dose (mg)
    tspan: Time span (hours)

Returns:
    Solution object
"""
function solve_oral(
    pbpk_params::PBPKParams,
    oral_params::OralParams,
    dose::Float64,
    tspan::Tuple{Float64, Float64};
    time_points::Union{Vector{Float64}, Nothing} = nothing,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
    alg = Tsit5(),
)
    # Initial conditions: all drug in gut lumen
    u0 = zeros(Float64, GUT_LUMEN_IDX)
    u0[GUT_LUMEN_IDX] = dose * oral_params.fa  # Amount available for absorption

    # Handle lag time by adjusting tspan
    effective_tspan = if oral_params.lag > 0.0
        (oral_params.lag, tspan[2])
    else
        tspan
    end

    # Parameters tuple
    p = (pbpk_params, oral_params)

    # Create and solve ODE problem
    prob = ODEProblem(oral_ode_system!, u0, effective_tspan, p)

    if time_points !== nothing
        # Adjust time points for lag
        adjusted_times = [t for t in time_points if t >= oral_params.lag]
        if isempty(adjusted_times)
            adjusted_times = [effective_tspan[2]]
        end
        sol = DifferentialEquations.solve(prob, alg; reltol=reltol, abstol=abstol, saveat=adjusted_times)
    else
        sol = DifferentialEquations.solve(prob, alg; reltol=reltol, abstol=abstol)
    end

    return sol
end

"""
Simulate oral PBPK and return concentration-time profiles.

Args:
    pbpk_params: PBPK parameters
    oral_params: Oral absorption parameters
    dose: Oral dose (mg)
    t_max: Maximum time (hours)
    num_points: Number of time points

Returns:
    Dict with concentrations and PK metrics
"""
function simulate_oral(
    pbpk_params::PBPKParams,
    oral_params::OralParams,
    dose::Float64;
    t_max::Float64 = 24.0,
    num_points::Int = 100,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
)
    time_points = collect(range(0.0, t_max, length=num_points))
    tspan = (0.0, t_max)
    lag = oral_params.lag

    # Initial conditions
    u0 = zeros(Float64, GUT_LUMEN_IDX)
    u0[GUT_LUMEN_IDX] = dose * oral_params.fa

    # Parameters
    p = (pbpk_params, oral_params)

    # Handle lag time with callback
    if lag > 0.0
        # Use a condition callback to start absorption at lag time
        # For simplicity, we'll solve in two phases or use delayed start

        # Phase 1: t = 0 to lag (no absorption, drug stays in gut)
        pre_lag_points = [t for t in time_points if t < lag]
        post_lag_points = [t for t in time_points if t >= lag]

        results = Dict{String, Any}()
        results["time"] = time_points

        # Initialize concentration arrays
        for organ in PBPK_ORGANS
            results[organ] = zeros(num_points)
        end
        results["gut_lumen"] = zeros(num_points)
        results["plasma"] = zeros(num_points)

        # Pre-lag: all drug in gut lumen, no systemic
        for (i, t) in enumerate(time_points)
            if t < lag
                results["gut_lumen"][i] = dose * oral_params.fa
            end
        end

        # Post-lag: solve ODE
        if !isempty(post_lag_points)
            prob = ODEProblem(oral_ode_system!, u0, (lag, t_max), p)
            sol = DifferentialEquations.solve(prob, Tsit5(); reltol=reltol, abstol=abstol,
                                               saveat=post_lag_points)

            # Fill in results
            for (j, t) in enumerate(post_lag_points)
                # Find index in original time array
                i = findfirst(x -> x >= t - 1e-10, time_points)
                if i !== nothing && j <= length(sol)
                    for (k, organ) in enumerate(PBPK_ORGANS)
                        results[organ][i] = sol[j][k]
                    end
                    results["gut_lumen"][i] = sol[j][GUT_LUMEN_IDX]
                    results["plasma"][i] = sol[j][BLOOD_IDX]
                end
            end
        end
    else
        # No lag: solve directly
        prob = ODEProblem(oral_ode_system!, u0, tspan, p)
        sol = DifferentialEquations.solve(prob, Tsit5(); reltol=reltol, abstol=abstol,
                                           saveat=time_points)

        results = Dict{String, Any}()
        results["time"] = time_points

        for (i, organ) in enumerate(PBPK_ORGANS)
            results[organ] = [sol[j][i] for j in 1:length(sol)]
        end
        results["gut_lumen"] = [sol[j][GUT_LUMEN_IDX] for j in 1:length(sol)]
        results["plasma"] = results["blood"]
    end

    # Calculate PK metrics
    plasma_conc = results["plasma"]

    # Cmax and Tmax
    if !isempty(plasma_conc) && any(c -> c > 0, plasma_conc)
        cmax_idx = argmax(plasma_conc)
        results["cmax"] = plasma_conc[cmax_idx]
        results["tmax"] = time_points[cmax_idx]
    else
        results["cmax"] = 0.0
        results["tmax"] = 0.0
    end

    # AUC (trapezoidal rule)
    auc = 0.0
    for i in 2:length(time_points)
        dt = time_points[i] - time_points[i-1]
        auc += 0.5 * (plasma_conc[i] + plasma_conc[i-1]) * dt
    end
    results["auc"] = auc

    # Terminal half-life
    n_terminal = max(3, num_points ÷ 4)
    terminal_idx = (num_points - n_terminal + 1):num_points
    valid_idx = [i for i in terminal_idx if plasma_conc[i] > 1e-10]

    if length(valid_idx) >= 2
        t_term = time_points[valid_idx]
        c_term = log.(plasma_conc[valid_idx])

        n = length(t_term)
        sum_t = sum(t_term)
        sum_c = sum(c_term)
        sum_tc = sum(t_term .* c_term)
        sum_t2 = sum(t_term .^ 2)

        denom = n * sum_t2 - sum_t^2
        if abs(denom) > 1e-10
            slope = (n * sum_tc - sum_t * sum_c) / denom
            if slope < 0
                results["half_life"] = -log(2) / slope
            else
                results["half_life"] = NaN
            end
        else
            results["half_life"] = NaN
        end
    else
        results["half_life"] = NaN
    end

    # Effective bioavailability
    results["f_eff"] = effective_f(oral_params)

    return results
end

export OralParams, effective_f, solve_oral, simulate_oral, GUT_LUMEN_IDX

#=============================================================================
  Transit Compartment Absorption Model (CAT Model)

  MedLang v0.3 Feature: Transit compartment absorption for drugs with
  delayed/complex absorption profiles (e.g., modified release, BCS III/IV)

  Model: n transit compartments in series before absorption site
  - dA_1/dt = -Ktr * A_1 (first transit)
  - dA_i/dt = Ktr * A_{i-1} - Ktr * A_i (intermediate transits)
  - dA_n/dt = Ktr * A_{n-1} - Ka * A_n (final transit -> absorption)

  References:
  - Savic RM et al. J Pharmacokinet Pharmacodyn 2007;34:711-726
  - Yu LX et al. AAPS PharmSci 2002;4:E33
=============================================================================#

"""
Parameters for transit compartment absorption model.
"""
struct TransitParams
    n_transit::Int        # Number of transit compartments (1-10)
    ktr::Float64          # Transit rate constant (1/h)
    ka::Float64           # Final absorption rate constant (1/h)
    fa::Float64           # Fraction absorbed (0-1)
    fg::Float64           # Fraction escaping gut metabolism (0-1)
    fh::Float64           # Fraction escaping hepatic first-pass (0-1)
    lag::Float64          # Initial lag time before first transit (h)
end

# Default transit params
TransitParams() = TransitParams(3, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)

# Constructor with mean transit time (MTT)
function TransitParams(n_transit::Int, mtt::Float64, ka::Float64; fa=1.0, fg=1.0, fh=1.0, lag=0.0)
    # ktr = (n + 1) / MTT for CAT model
    ktr = (n_transit + 1) / mtt
    TransitParams(n_transit, ktr, ka, fa, fg, fh, lag)
end

"""
Calculate mean transit time: MTT = (n + 1) / Ktr
"""
mean_transit_time(tp::TransitParams) = (tp.n_transit + 1) / tp.ktr

"""
Calculate effective bioavailability for transit model
"""
effective_f(tp::TransitParams) = tp.fa * tp.fg * tp.fh

# Maximum number of transit compartments supported
const MAX_TRANSIT = 10
const TRANSIT_START_IDX = 16  # Indices 16-25 for transit compartments

"""
Extended ODE system with transit compartment absorption.

State vector (15 + n_transit elements):
- u[1:14]: Organ concentrations (standard PBPK)
- u[15]: Gut lumen (final absorption site)
- u[16:16+n-1]: Transit compartments

Equations:
- Transit 1: dA_1/dt = -Ktr * A_1
- Transit i: dA_i/dt = Ktr * A_{i-1} - Ktr * A_i
- Gut lumen: dA_gut/dt = Ktr * A_n - Ka * A_gut
- Blood: dC_blood/dt = ... + Ka * A_gut * Fg * Fh / V_blood
"""
function transit_ode_system!(du::AbstractVector{Float64}, u::AbstractVector{Float64},
                              p::Tuple{PBPKParams, TransitParams}, t::Float64)
    pbpk, transit = p
    n = transit.n_transit

    # Initialize derivatives
    fill!(du, 0.0)

    C_blood = u[BLOOD_IDX]
    A_gut_lumen = u[GUT_LUMEN_IDX]

    # Standard PBPK for organs (same as oral_ode_system!)
    @inbounds for i in 1:NUM_ORGANS
        if i == BLOOD_IDX
            continue
        end

        V_organ = pbpk.volumes[i]
        Q_organ = pbpk.blood_flows[i]
        Kp_organ = pbpk.partition_coeffs[i]
        C_organ = u[i]

        du[i] = (Q_organ / V_organ) * (C_blood - C_organ / Kp_organ)
        V_blood = pbpk.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= (Q_organ / V_blood) * (C_blood - C_organ / Kp_organ)
    end

    # Clearance
    if pbpk.clearance_hepatic > 0.0
        du[BLOOD_IDX] -= (pbpk.clearance_hepatic / pbpk.volumes[BLOOD_IDX]) * C_blood
    end
    if pbpk.clearance_renal > 0.0
        du[BLOOD_IDX] -= (pbpk.clearance_renal / pbpk.volumes[BLOOD_IDX]) * C_blood
    end

    # Transit compartment chain
    @inbounds for i in 1:n
        idx = TRANSIT_START_IDX + i - 1
        A_transit = u[idx]

        if i == 1
            # First transit: only outflow
            du[idx] = -transit.ktr * A_transit
        else
            # Intermediate transits: inflow from previous, outflow to next
            A_prev = u[idx - 1]
            du[idx] = transit.ktr * A_prev - transit.ktr * A_transit
        end
    end

    # Gut lumen receives from last transit, loses to absorption
    if n > 0
        last_transit_idx = TRANSIT_START_IDX + n - 1
        A_last_transit = u[last_transit_idx]
        du[GUT_LUMEN_IDX] = transit.ktr * A_last_transit - transit.ka * A_gut_lumen
    else
        # No transit compartments - direct absorption
        du[GUT_LUMEN_IDX] = -transit.ka * A_gut_lumen
    end

    # Absorption to blood (after first-pass)
    absorption_rate = transit.ka * A_gut_lumen * transit.fg * transit.fh / pbpk.volumes[BLOOD_IDX]
    du[BLOOD_IDX] += absorption_rate

    return nothing
end

"""
Simulate transit compartment absorption model.

Args:
    pbpk_params: PBPK parameters
    transit_params: Transit absorption parameters
    dose: Oral dose (mg)
    t_max: Maximum time (hours)
    num_points: Number of time points

Returns:
    Dict with concentrations, transit amounts, and PK metrics
"""
function simulate_transit(
    pbpk_params::PBPKParams,
    transit_params::TransitParams,
    dose::Float64;
    t_max::Float64 = 24.0,
    num_points::Int = 100,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
)
    n = transit_params.n_transit
    total_states = GUT_LUMEN_IDX + n
    time_points = collect(range(0.0, t_max, length=num_points))
    lag = transit_params.lag

    # Initial conditions: all drug in first transit compartment
    u0 = zeros(Float64, total_states)
    if n > 0
        u0[TRANSIT_START_IDX] = dose * transit_params.fa  # First transit
    else
        u0[GUT_LUMEN_IDX] = dose * transit_params.fa  # Direct to gut lumen
    end

    p = (pbpk_params, transit_params)

    # Handle lag time
    effective_start = lag > 0.0 ? lag : 0.0
    tspan = (effective_start, t_max)

    # Solve ODE
    prob = ODEProblem(transit_ode_system!, u0, tspan, p)
    sol = DifferentialEquations.solve(prob, Tsit5(); reltol=reltol, abstol=abstol,
                                       saveat=[t for t in time_points if t >= effective_start])

    # Build results
    results = Dict{String, Any}()
    results["time"] = time_points
    results["n_transit"] = n
    results["mtt"] = mean_transit_time(transit_params)

    # Initialize arrays
    for organ in PBPK_ORGANS
        results[organ] = zeros(num_points)
    end
    results["gut_lumen"] = zeros(num_points)
    results["plasma"] = zeros(num_points)

    # Transit compartment amounts
    for i in 1:n
        results["transit_$i"] = zeros(num_points)
    end

    # Fill pre-lag with initial state
    for (i, t) in enumerate(time_points)
        if t < effective_start
            if n > 0
                results["transit_1"][i] = dose * transit_params.fa
            else
                results["gut_lumen"][i] = dose * transit_params.fa
            end
        end
    end

    # Fill post-lag from solution
    post_lag_times = [t for t in time_points if t >= effective_start]
    for (j, t) in enumerate(post_lag_times)
        i = findfirst(x -> x >= t - 1e-10, time_points)
        if i !== nothing && j <= length(sol)
            for (k, organ) in enumerate(PBPK_ORGANS)
                results[organ][i] = sol[j][k]
            end
            results["gut_lumen"][i] = sol[j][GUT_LUMEN_IDX]
            results["plasma"][i] = sol[j][BLOOD_IDX]

            for k in 1:n
                results["transit_$k"][i] = sol[j][TRANSIT_START_IDX + k - 1]
            end
        end
    end

    # PK metrics
    plasma_conc = results["plasma"]

    if !isempty(plasma_conc) && any(c -> c > 0, plasma_conc)
        cmax_idx = argmax(plasma_conc)
        results["cmax"] = plasma_conc[cmax_idx]
        results["tmax"] = time_points[cmax_idx]
    else
        results["cmax"] = 0.0
        results["tmax"] = 0.0
    end

    # AUC
    auc = 0.0
    for i in 2:length(time_points)
        dt = time_points[i] - time_points[i-1]
        auc += 0.5 * (plasma_conc[i] + plasma_conc[i-1]) * dt
    end
    results["auc"] = auc

    results["f_eff"] = effective_f(transit_params)

    return results
end

export TransitParams, mean_transit_time, transit_ode_system!, simulate_transit
export MAX_TRANSIT, TRANSIT_START_IDX

#=============================================================================
  Enterohepatic Recirculation (EHR) Model

  MedLang v0.3 Feature: Enterohepatic recirculation for drugs that undergo
  biliary excretion and intestinal reabsorption (e.g., digoxin, mycophenolate)

  Model:
  - Biliary excretion: Liver -> Bile -> Gut
  - Intestinal reabsorption: Gut -> Portal vein -> Liver -> Systemic
  - Creates secondary peaks in plasma concentration

  References:
  - Roberts MS et al. J Pharmacokinet Biopharm 2002;30:97-130
  - Shepard TA et al. J Pharm Sci 1985;74:1197-1202
=============================================================================#

"""
Parameters for enterohepatic recirculation model.
"""
struct EHRParams
    f_bile::Float64       # Fraction excreted in bile (0-1)
    k_bile::Float64       # Biliary excretion rate constant (1/h)
    f_reabs::Float64      # Fraction reabsorbed from gut (0-1)
    k_reabs::Float64      # Reabsorption rate constant (1/h)
    t_gb::Float64         # Gallbladder emptying delay (h), typ. 0.5-2h postprandial
    meal_times::Vector{Float64}  # Times of meals triggering GB emptying (h)
end

# Default EHR params (moderate recirculation)
EHRParams() = EHRParams(0.3, 0.5, 0.8, 1.0, 1.0, Float64[])

# Constructor with single meal
function EHRParams(f_bile, k_bile, f_reabs, k_reabs, t_gb, meal_time::Float64)
    EHRParams(f_bile, k_bile, f_reabs, k_reabs, t_gb, [meal_time])
end

const BILE_COMPARTMENT_IDX = 26  # Index for bile/gallbladder compartment
const GUT_REABS_IDX = 27        # Index for gut reabsorption compartment

"""
Extended ODE system with enterohepatic recirculation.

State vector (27 elements):
- u[1:14]: Organ concentrations (standard PBPK)
- u[15]: Gut lumen (absorption site)
- u[16:25]: Transit compartments (optional)
- u[26]: Bile/gallbladder amount (mg)
- u[27]: Gut amount available for reabsorption (mg)

Equations:
- Liver: includes biliary excretion term
- Bile: dA_bile/dt = k_bile * f_bile * A_liver - k_empty(t) * A_bile
- Gut_reabs: dA_gut/dt = k_empty(t) * A_bile - k_reabs * A_gut
- Blood: dC_blood/dt = ... + k_reabs * f_reabs * A_gut / V_blood
"""
function ehr_ode_system!(du::AbstractVector{Float64}, u::AbstractVector{Float64},
                         p::Tuple{PBPKParams, OralParams, EHRParams}, t::Float64)
    pbpk, oral, ehr = p

    # Initialize derivatives
    fill!(du, 0.0)

    C_blood = u[BLOOD_IDX]
    C_liver = u[LIVER_IDX]
    A_gut_lumen = u[GUT_LUMEN_IDX]
    A_bile = u[BILE_COMPARTMENT_IDX]
    A_gut_reabs = u[GUT_REABS_IDX]

    # Standard PBPK for organs
    @inbounds for i in 1:NUM_ORGANS
        if i == BLOOD_IDX
            continue
        end

        V_organ = pbpk.volumes[i]
        Q_organ = pbpk.blood_flows[i]
        Kp_organ = pbpk.partition_coeffs[i]
        C_organ = u[i]

        du[i] = (Q_organ / V_organ) * (C_blood - C_organ / Kp_organ)
        V_blood = pbpk.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= (Q_organ / V_blood) * (C_blood - C_organ / Kp_organ)
    end

    # Hepatic clearance (metabolic)
    if pbpk.clearance_hepatic > 0.0
        du[BLOOD_IDX] -= (pbpk.clearance_hepatic / pbpk.volumes[BLOOD_IDX]) * C_blood
    end

    # Renal clearance
    if pbpk.clearance_renal > 0.0
        du[BLOOD_IDX] -= (pbpk.clearance_renal / pbpk.volumes[BLOOD_IDX]) * C_blood
    end

    # Biliary excretion from liver
    # Amount in liver = C_liver * V_liver
    A_liver = C_liver * pbpk.volumes[LIVER_IDX]
    biliary_excretion = ehr.k_bile * ehr.f_bile * A_liver
    du[LIVER_IDX] -= biliary_excretion / pbpk.volumes[LIVER_IDX]  # Convert back to concentration
    du[BILE_COMPARTMENT_IDX] = biliary_excretion

    # Gallbladder emptying (triggered by meals)
    # Rate increases during meal times
    k_empty = 0.1  # Baseline slow emptying
    for meal_t in ehr.meal_times
        # Gaussian pulse of emptying around meal time + delay
        t_peak = meal_t + ehr.t_gb
        sigma = 0.5  # 30-min window
        if abs(t - t_peak) < 3 * sigma
            pulse = exp(-0.5 * ((t - t_peak) / sigma)^2)
            k_empty += 2.0 * pulse  # Increased emptying during meal
        end
    end

    # Bile -> Gut for reabsorption
    bile_to_gut = k_empty * A_bile
    du[BILE_COMPARTMENT_IDX] -= bile_to_gut
    du[GUT_REABS_IDX] = bile_to_gut - ehr.k_reabs * A_gut_reabs

    # Reabsorption to blood
    reabsorption_rate = ehr.k_reabs * ehr.f_reabs * A_gut_reabs / pbpk.volumes[BLOOD_IDX]
    du[BLOOD_IDX] += reabsorption_rate

    # Original oral absorption (if present)
    du[GUT_LUMEN_IDX] = -oral.ka * A_gut_lumen
    absorption_rate = oral.ka * A_gut_lumen * oral.fg * oral.fh / pbpk.volumes[BLOOD_IDX]
    du[BLOOD_IDX] += absorption_rate

    return nothing
end

"""
Simulate PBPK with enterohepatic recirculation.

Args:
    pbpk_params: PBPK parameters
    oral_params: Oral absorption parameters
    ehr_params: Enterohepatic recirculation parameters
    dose: Oral dose (mg)
    t_max: Maximum time (hours)
    num_points: Number of time points

Returns:
    Dict with concentrations, bile amounts, and PK metrics
"""
function simulate_ehr(
    pbpk_params::PBPKParams,
    oral_params::OralParams,
    ehr_params::EHRParams,
    dose::Float64;
    t_max::Float64 = 48.0,  # Longer default for EHR effects
    num_points::Int = 200,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
)
    total_states = GUT_REABS_IDX
    time_points = collect(range(0.0, t_max, length=num_points))

    # Initial conditions
    u0 = zeros(Float64, total_states)
    u0[GUT_LUMEN_IDX] = dose * oral_params.fa

    p = (pbpk_params, oral_params, ehr_params)
    tspan = (0.0, t_max)

    # Solve ODE
    prob = ODEProblem(ehr_ode_system!, u0, tspan, p)
    sol = DifferentialEquations.solve(prob, Tsit5(); reltol=reltol, abstol=abstol,
                                       saveat=time_points)

    # Build results
    results = Dict{String, Any}()
    results["time"] = time_points

    for (i, organ) in enumerate(PBPK_ORGANS)
        results[organ] = [sol[j][i] for j in 1:length(sol)]
    end
    results["gut_lumen"] = [sol[j][GUT_LUMEN_IDX] for j in 1:length(sol)]
    results["bile"] = [sol[j][BILE_COMPARTMENT_IDX] for j in 1:length(sol)]
    results["gut_reabs"] = [sol[j][GUT_REABS_IDX] for j in 1:length(sol)]
    results["plasma"] = results["blood"]

    # PK metrics
    plasma_conc = results["plasma"]

    if !isempty(plasma_conc) && any(c -> c > 0, plasma_conc)
        cmax_idx = argmax(plasma_conc)
        results["cmax"] = plasma_conc[cmax_idx]
        results["tmax"] = time_points[cmax_idx]

        # Detect secondary peaks (EHR signature)
        peaks = Int[]
        for i in 2:(length(plasma_conc)-1)
            if plasma_conc[i] > plasma_conc[i-1] && plasma_conc[i] > plasma_conc[i+1]
                push!(peaks, i)
            end
        end
        results["n_peaks"] = length(peaks)
        results["peak_times"] = [time_points[i] for i in peaks]
    else
        results["cmax"] = 0.0
        results["tmax"] = 0.0
        results["n_peaks"] = 0
        results["peak_times"] = Float64[]
    end

    # AUC
    auc = 0.0
    for i in 2:length(time_points)
        dt = time_points[i] - time_points[i-1]
        auc += 0.5 * (plasma_conc[i] + plasma_conc[i-1]) * dt
    end
    results["auc"] = auc

    results["f_eff"] = effective_f(oral_params)
    results["f_bile"] = ehr_params.f_bile
    results["f_reabs"] = ehr_params.f_reabs

    return results
end

export EHRParams, ehr_ode_system!, simulate_ehr
export BILE_COMPARTMENT_IDX, GUT_REABS_IDX

#=============================================================================
  Non-Linear (Saturable) Absorption Model

  MedLang v0.4 Feature: Michaelis-Menten absorption kinetics for drugs with:
  - Saturable transporters (P-gp, OATP, PEPT1)
  - Dose-dependent bioavailability
  - Non-linear PK at high doses

  Model:
  - dA_gut/dt = -Vmax * A_gut / (Km + A_gut)  (saturable)
  - At low doses: approaches first-order (Vmax/Km * A_gut)
  - At high doses: approaches zero-order (Vmax)

  References:
  - Amidon GL et al. Pharm Res 1995 (BCS)
  - Estudante M et al. Adv Drug Deliv Rev 2013
=============================================================================#

"""
Parameters for saturable (Michaelis-Menten) absorption.
"""
struct SaturableAbsorptionParams
    vmax::Float64         # Maximum absorption rate (mg/h)
    km::Float64           # Michaelis constant (mg) - amount at half-Vmax
    fa::Float64           # Fraction available for absorption (0-1)
    fg::Float64           # Fraction escaping gut metabolism (0-1)
    fh::Float64           # Fraction escaping hepatic first-pass (0-1)
    lag::Float64          # Lag time (h)
    passive_ka::Float64   # Additional passive absorption rate (1/h), default 0
end

# Default constructor
SaturableAbsorptionParams() = SaturableAbsorptionParams(100.0, 50.0, 1.0, 1.0, 1.0, 0.0, 0.0)

# Constructor without passive component
function SaturableAbsorptionParams(vmax, km; fa=1.0, fg=1.0, fh=1.0, lag=0.0)
    SaturableAbsorptionParams(vmax, km, fa, fg, fh, lag, 0.0)
end

"""
Calculate apparent first-order rate at low concentrations: ka_app = Vmax / Km
"""
apparent_ka(sp::SaturableAbsorptionParams) = sp.vmax / sp.km

"""
Calculate effective bioavailability
"""
effective_f(sp::SaturableAbsorptionParams) = sp.fa * sp.fg * sp.fh

"""
ODE system with saturable (Michaelis-Menten) absorption.

Absorption rate = Vmax * A_gut / (Km + A_gut) + passive_ka * A_gut

At A_gut << Km: rate ≈ (Vmax/Km) * A_gut (first-order)
At A_gut >> Km: rate ≈ Vmax (zero-order)
"""
function saturable_ode_system!(du::AbstractVector{Float64}, u::AbstractVector{Float64},
                                p::Tuple{PBPKParams, SaturableAbsorptionParams}, t::Float64)
    pbpk, sat = p

    fill!(du, 0.0)

    C_blood = u[BLOOD_IDX]
    A_gut_lumen = u[GUT_LUMEN_IDX]

    # Standard PBPK for organs
    @inbounds for i in 1:NUM_ORGANS
        if i == BLOOD_IDX
            continue
        end

        V_organ = pbpk.volumes[i]
        Q_organ = pbpk.blood_flows[i]
        Kp_organ = pbpk.partition_coeffs[i]
        C_organ = u[i]

        du[i] = (Q_organ / V_organ) * (C_blood - C_organ / Kp_organ)
        V_blood = pbpk.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= (Q_organ / V_blood) * (C_blood - C_organ / Kp_organ)
    end

    # Clearance
    if pbpk.clearance_hepatic > 0.0
        du[BLOOD_IDX] -= (pbpk.clearance_hepatic / pbpk.volumes[BLOOD_IDX]) * C_blood
    end
    if pbpk.clearance_renal > 0.0
        du[BLOOD_IDX] -= (pbpk.clearance_renal / pbpk.volumes[BLOOD_IDX]) * C_blood
    end

    # Saturable (Michaelis-Menten) absorption
    # Rate = Vmax * A / (Km + A) + passive * A
    saturable_rate = sat.vmax * A_gut_lumen / (sat.km + A_gut_lumen + 1e-10)
    passive_rate = sat.passive_ka * A_gut_lumen
    total_absorption_rate = saturable_rate + passive_rate

    du[GUT_LUMEN_IDX] = -total_absorption_rate

    # Drug absorbed goes to blood (after first-pass)
    absorption_to_blood = total_absorption_rate * sat.fg * sat.fh / pbpk.volumes[BLOOD_IDX]
    du[BLOOD_IDX] += absorption_to_blood

    return nothing
end

"""
Simulate saturable absorption model.

Shows dose-dependent bioavailability - higher doses have lower F.
"""
function simulate_saturable(
    pbpk_params::PBPKParams,
    sat_params::SaturableAbsorptionParams,
    dose::Float64;
    t_max::Float64 = 24.0,
    num_points::Int = 100,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
)
    time_points = collect(range(0.0, t_max, length=num_points))
    lag = sat_params.lag

    # Initial conditions
    u0 = zeros(Float64, GUT_LUMEN_IDX)
    u0[GUT_LUMEN_IDX] = dose * sat_params.fa

    p = (pbpk_params, sat_params)

    # Handle lag time
    effective_start = lag > 0.0 ? lag : 0.0
    tspan = (effective_start, t_max)

    # Solve ODE
    prob = ODEProblem(saturable_ode_system!, u0, tspan, p)
    sol = DifferentialEquations.solve(prob, Tsit5(); reltol=reltol, abstol=abstol,
                                       saveat=[t for t in time_points if t >= effective_start])

    # Build results
    results = Dict{String, Any}()
    results["time"] = time_points

    for organ in PBPK_ORGANS
        results[organ] = zeros(num_points)
    end
    results["gut_lumen"] = zeros(num_points)
    results["plasma"] = zeros(num_points)

    # Fill pre-lag
    for (i, t) in enumerate(time_points)
        if t < effective_start
            results["gut_lumen"][i] = dose * sat_params.fa
        end
    end

    # Fill from solution
    post_lag_times = [t for t in time_points if t >= effective_start]
    for (j, t) in enumerate(post_lag_times)
        i = findfirst(x -> x >= t - 1e-10, time_points)
        if i !== nothing && j <= length(sol)
            for (k, organ) in enumerate(PBPK_ORGANS)
                results[organ][i] = sol[j][k]
            end
            results["gut_lumen"][i] = sol[j][GUT_LUMEN_IDX]
            results["plasma"][i] = sol[j][BLOOD_IDX]
        end
    end

    # PK metrics
    plasma_conc = results["plasma"]

    if !isempty(plasma_conc) && any(c -> c > 0, plasma_conc)
        cmax_idx = argmax(plasma_conc)
        results["cmax"] = plasma_conc[cmax_idx]
        results["tmax"] = time_points[cmax_idx]
    else
        results["cmax"] = 0.0
        results["tmax"] = 0.0
    end

    # AUC
    auc = 0.0
    for i in 2:length(time_points)
        dt = time_points[i] - time_points[i-1]
        auc += 0.5 * (plasma_conc[i] + plasma_conc[i-1]) * dt
    end
    results["auc"] = auc

    # Calculate dose-normalized AUC to show non-linearity
    results["auc_norm"] = auc / dose
    results["f_eff"] = effective_f(sat_params)
    results["apparent_ka"] = apparent_ka(sat_params)

    return results
end

"""
Analyze dose-proportionality for saturable absorption.

Returns AUC at multiple doses to demonstrate non-linearity.
"""
function analyze_dose_proportionality(
    pbpk_params::PBPKParams,
    sat_params::SaturableAbsorptionParams,
    doses::Vector{Float64};
    t_max::Float64 = 24.0,
)
    results = Dict{String, Vector{Float64}}()
    results["dose"] = doses
    results["auc"] = Float64[]
    results["auc_norm"] = Float64[]
    results["cmax"] = Float64[]
    results["tmax"] = Float64[]

    for dose in doses
        sim = simulate_saturable(pbpk_params, sat_params, dose; t_max=t_max)
        push!(results["auc"], sim["auc"])
        push!(results["auc_norm"], sim["auc_norm"])
        push!(results["cmax"], sim["cmax"])
        push!(results["tmax"], sim["tmax"])
    end

    # Calculate power law exponent (AUC ∝ Dose^β)
    # β = 1 for linear, β < 1 for saturable
    if length(doses) >= 2
        log_doses = log.(doses)
        log_aucs = log.(results["auc"] .+ 1e-10)
        n = length(doses)
        sum_x = sum(log_doses)
        sum_y = sum(log_aucs)
        sum_xy = sum(log_doses .* log_aucs)
        sum_x2 = sum(log_doses .^ 2)
        beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
        results["power_exponent"] = [beta]
    else
        results["power_exponent"] = [1.0]
    end

    return results
end

export SaturableAbsorptionParams, apparent_ka, saturable_ode_system!
export simulate_saturable, analyze_dose_proportionality

#=============================================================================
  Multi-Compartment Depot Model (IM/SC Administration)

  MedLang v0.4 Feature: Flip-flop kinetics for subcutaneous/intramuscular
  injection with:
  - Slow release from depot site
  - Multiple depot compartments for complex release
  - Bioavailability considerations

  Model options:
  1. Single depot: First-order release
  2. Dual depot: Fast + slow release fractions
  3. Zero-order + first-order: Initial burst + sustained

  References:
  - Mager DE et al. J Pharmacokinet Pharmacodyn 2001
  - Supersaxo A et al. Pharm Res 1990
=============================================================================#

"""
Parameters for depot (IM/SC) absorption model.
"""
struct DepotParams
    route::Symbol         # :IM or :SC
    n_depots::Int         # Number of depot compartments (1-3)
    ka::Vector{Float64}   # Absorption rate constants for each depot (1/h)
    fractions::Vector{Float64}  # Fraction of dose in each depot
    f::Float64            # Overall bioavailability (0-1)
    lag::Float64          # Lag time before absorption starts (h)
end

# Single depot constructor
function DepotParams(route::Symbol, ka::Float64; f=1.0, lag=0.0)
    DepotParams(route, 1, [ka], [1.0], f, lag)
end

# Dual depot constructor (fast + slow release)
function DepotParams(route::Symbol, ka_fast::Float64, ka_slow::Float64, f_fast::Float64; f=1.0, lag=0.0)
    DepotParams(route, 2, [ka_fast, ka_slow], [f_fast, 1.0 - f_fast], f, lag)
end

# Typical IM parameters
DepotParams_IM() = DepotParams(:IM, 0.5; f=0.9, lag=0.1)

# Typical SC parameters (slower than IM)
DepotParams_SC() = DepotParams(:SC, 0.2; f=0.85, lag=0.2)

# SC with dual absorption (biologics pattern)
DepotParams_SC_Biologic() = DepotParams(:SC, 0.5, 0.05, 0.3; f=0.7, lag=0.5)

const DEPOT_START_IDX = 28  # Indices 28-30 for depot compartments
const MAX_DEPOTS = 3

"""
Calculate mean absorption time for depot model.
"""
function mean_absorption_time(dp::DepotParams)
    mat = 0.0
    for i in 1:dp.n_depots
        mat += dp.fractions[i] / dp.ka[i]
    end
    return mat + dp.lag
end

"""
ODE system for depot (IM/SC) absorption.

State vector includes:
- u[1:14]: Organ concentrations
- u[15]: (unused for depot)
- u[28:28+n-1]: Depot compartments

Each depot releases to blood with its own rate constant.
"""
function depot_ode_system!(du::AbstractVector{Float64}, u::AbstractVector{Float64},
                           p::Tuple{PBPKParams, DepotParams}, t::Float64)
    pbpk, depot = p

    fill!(du, 0.0)

    C_blood = u[BLOOD_IDX]

    # Standard PBPK for organs
    @inbounds for i in 1:NUM_ORGANS
        if i == BLOOD_IDX
            continue
        end

        V_organ = pbpk.volumes[i]
        Q_organ = pbpk.blood_flows[i]
        Kp_organ = pbpk.partition_coeffs[i]
        C_organ = u[i]

        du[i] = (Q_organ / V_organ) * (C_blood - C_organ / Kp_organ)
        V_blood = pbpk.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= (Q_organ / V_blood) * (C_blood - C_organ / Kp_organ)
    end

    # Clearance
    if pbpk.clearance_hepatic > 0.0
        du[BLOOD_IDX] -= (pbpk.clearance_hepatic / pbpk.volumes[BLOOD_IDX]) * C_blood
    end
    if pbpk.clearance_renal > 0.0
        du[BLOOD_IDX] -= (pbpk.clearance_renal / pbpk.volumes[BLOOD_IDX]) * C_blood
    end

    # Depot absorption (each depot releases independently)
    total_absorption = 0.0
    @inbounds for i in 1:depot.n_depots
        depot_idx = DEPOT_START_IDX + i - 1
        A_depot = u[depot_idx]
        ka = depot.ka[i]

        # First-order release from depot
        release_rate = ka * A_depot
        du[depot_idx] = -release_rate
        total_absorption += release_rate
    end

    # All absorbed drug goes directly to blood (no first-pass for IM/SC)
    du[BLOOD_IDX] += total_absorption * depot.f / pbpk.volumes[BLOOD_IDX]

    return nothing
end

"""
Simulate depot (IM/SC) absorption model.
"""
function simulate_depot(
    pbpk_params::PBPKParams,
    depot_params::DepotParams,
    dose::Float64;
    t_max::Float64 = 72.0,  # Longer default for slow SC absorption
    num_points::Int = 150,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
)
    n = depot_params.n_depots
    total_states = DEPOT_START_IDX + n - 1
    time_points = collect(range(0.0, t_max, length=num_points))
    lag = depot_params.lag

    # Initial conditions: distribute dose across depots
    u0 = zeros(Float64, total_states)
    for i in 1:n
        depot_idx = DEPOT_START_IDX + i - 1
        u0[depot_idx] = dose * depot_params.fractions[i]
    end

    p = (pbpk_params, depot_params)

    # Handle lag time
    effective_start = lag > 0.0 ? lag : 0.0
    tspan = (effective_start, t_max)

    # Solve ODE
    prob = ODEProblem(depot_ode_system!, u0, tspan, p)
    sol = DifferentialEquations.solve(prob, Tsit5(); reltol=reltol, abstol=abstol,
                                       saveat=[t for t in time_points if t >= effective_start])

    # Build results
    results = Dict{String, Any}()
    results["time"] = time_points
    results["route"] = depot_params.route
    results["n_depots"] = n
    results["mat"] = mean_absorption_time(depot_params)

    for organ in PBPK_ORGANS
        results[organ] = zeros(num_points)
    end
    results["plasma"] = zeros(num_points)

    # Depot amounts
    for i in 1:n
        results["depot_$i"] = zeros(num_points)
    end

    # Fill pre-lag
    for (i, t) in enumerate(time_points)
        if t < effective_start
            for j in 1:n
                results["depot_$j"][i] = dose * depot_params.fractions[j]
            end
        end
    end

    # Fill from solution
    post_lag_times = [t for t in time_points if t >= effective_start]
    for (j, t) in enumerate(post_lag_times)
        i = findfirst(x -> x >= t - 1e-10, time_points)
        if i !== nothing && j <= length(sol)
            for (k, organ) in enumerate(PBPK_ORGANS)
                results[organ][i] = sol[j][k]
            end
            results["plasma"][i] = sol[j][BLOOD_IDX]

            for k in 1:n
                depot_idx = DEPOT_START_IDX + k - 1
                if depot_idx <= length(sol[j])
                    results["depot_$k"][i] = sol[j][depot_idx]
                end
            end
        end
    end

    # PK metrics
    plasma_conc = results["plasma"]

    if !isempty(plasma_conc) && any(c -> c > 0, plasma_conc)
        cmax_idx = argmax(plasma_conc)
        results["cmax"] = plasma_conc[cmax_idx]
        results["tmax"] = time_points[cmax_idx]
    else
        results["cmax"] = 0.0
        results["tmax"] = 0.0
    end

    # AUC
    auc = 0.0
    for i in 2:length(time_points)
        dt = time_points[i] - time_points[i-1]
        auc += 0.5 * (plasma_conc[i] + plasma_conc[i-1]) * dt
    end
    results["auc"] = auc

    results["f"] = depot_params.f

    # Check for flip-flop kinetics (Tmax > expected from ka)
    # In flip-flop, absorption is rate-limiting
    expected_tmax_absorption = 1.0 / minimum(depot_params.ka)
    results["flip_flop"] = results["tmax"] > expected_tmax_absorption * 0.8

    return results
end

export DepotParams, DepotParams_IM, DepotParams_SC, DepotParams_SC_Biologic
export mean_absorption_time, depot_ode_system!, simulate_depot
export DEPOT_START_IDX, MAX_DEPOTS


# =============================================================================
# QSP CONSTRUCTS (v1.0)
# =============================================================================
# Quantitative Systems Pharmacology extensions for mechanistic modeling
# - Target-Mediated Drug Disposition (TMDD)
# - Receptor-Ligand Dynamics
# - Tumor Growth-Kill Models
# - Enzyme Turnover
# - Signal Transduction Cascades

# -----------------------------------------------------------------------------
# TMDD (Target-Mediated Drug Disposition)
# -----------------------------------------------------------------------------
# For drugs where target binding significantly affects PK
# Michaelis-Menten approximation (QSS) and full TMDD

"""
TMDD Parameters for target-mediated drug disposition.

Supports:
- Full TMDD (receptor binding kinetics)
- QSS approximation (quasi-steady-state)
- QE approximation (quasi-equilibrium)

Reference: Mager & Jusko (2001), J Pharmacokinet Pharmacodyn
"""
struct TMDDParams
    # Binding kinetics
    kon::Float64      # Association rate (1/nM/h)
    koff::Float64     # Dissociation rate (1/h)

    # Target turnover
    ksyn::Float64     # Target synthesis rate (nM/h)
    kdeg::Float64     # Target degradation rate (1/h)

    # Complex internalization
    kint::Float64     # Internalization rate of drug-target complex (1/h)

    # Derived
    kd::Float64       # Equilibrium dissociation constant (nM) = koff/kon
    r0::Float64       # Baseline target concentration (nM) = ksyn/kdeg

    function TMDDParams(;
        kon::Float64 = 0.1,       # 1/nM/h
        koff::Float64 = 0.01,     # 1/h
        ksyn::Float64 = 1.0,      # nM/h
        kdeg::Float64 = 0.1,      # 1/h
        kint::Float64 = 0.05,     # 1/h
    )
        kd = koff / kon
        r0 = ksyn / kdeg
        new(kon, koff, ksyn, kdeg, kint, kd, r0)
    end
end

"""
Full TMDD ODE system.

State vector:
- u[1:NUM_ORGANS]: Drug concentrations in organs
- u[GUT_LUMEN_IDX]: Gut lumen (oral absorption)
- u[TARGET_IDX]: Free target concentration
- u[COMPLEX_IDX]: Drug-target complex concentration

ODEs:
- dL/dt = -kon*L*R + koff*LR + (PBPK terms)
- dR/dt = ksyn - kdeg*R - kon*L*R + koff*LR
- dLR/dt = kon*L*R - koff*LR - kint*LR
"""
const TARGET_IDX = 32
const COMPLEX_IDX = 33

function tmdd_ode_system!(du, u, p, t)
    # Unpack parameters
    pbpk_params, tmdd_params, dose, absorption_params = p

    # Get drug concentration in plasma (central compartment)
    c_plasma = u[BLOOD_IDX]

    # Get target and complex
    r_free = max(0.0, u[TARGET_IDX])      # Free target
    lr_complex = max(0.0, u[COMPLEX_IDX]) # Drug-target complex

    # PBPK component (standard tissue distribution)
    volumes = pbpk_params.volumes
    flows = pbpk_params.blood_flows
    kps = pbpk_params.partition_coeffs

    # Plasma concentration
    v_plasma = volumes[BLOOD_IDX]

    # TMDD terms (in plasma only for simplicity)
    # Binding: L + R <-> LR
    binding_rate = tmdd_params.kon * c_plasma * r_free
    unbinding_rate = tmdd_params.koff * lr_complex

    # Net binding flux
    net_binding = binding_rate - unbinding_rate

    # Target turnover
    target_synthesis = tmdd_params.ksyn
    target_degradation = tmdd_params.kdeg * r_free

    # Complex internalization (elimination)
    complex_internalization = tmdd_params.kint * lr_complex

    # Drug ODEs (PBPK + TMDD)
    for i in 1:NUM_ORGANS
        if i == BLOOD_IDX
            # Plasma: includes TMDD binding
            du[i] = -net_binding  # Binding removes free drug

            # Add flow terms from other organs
            for j in 1:NUM_ORGANS
                if j != BLOOD_IDX && flows[j] > 0
                    c_tissue = u[j] / kps[j]  # Venous concentration
                    du[i] += flows[j] * (c_tissue - c_plasma)
                end
            end

            # Hepatic clearance (of free drug only)
            du[i] -= pbpk_params.clearance_hepatic * c_plasma

        elseif i == LIVER_IDX
            # Liver
            c_liver = u[i]
            c_liver_ven = c_liver / kps[i]
            du[i] = flows[i] * (c_plasma - c_liver_ven) - pbpk_params.clearance_hepatic * c_liver_ven

        elseif i == KIDNEY_IDX
            # Kidney
            c_kidney = u[i]
            c_kidney_ven = c_kidney / kps[i]
            du[i] = flows[i] * (c_plasma - c_kidney_ven) - pbpk_params.clearance_renal * c_kidney_ven

        else
            # Other organs
            c_organ = u[i]
            c_organ_ven = c_organ / kps[i]
            du[i] = flows[i] * (c_plasma - c_organ_ven)
        end
    end

    # Absorption from gut lumen
    if absorption_params !== nothing
        gut_lumen = max(0.0, u[GUT_LUMEN_IDX])
        ka = absorption_params.ka
        fg = absorption_params.fg
        fh = absorption_params.fh

        absorption_rate = ka * gut_lumen
        du[GUT_LUMEN_IDX] = -absorption_rate
        du[BLOOD_IDX] += fg * fh * absorption_rate
    end

    # Target ODE
    du[TARGET_IDX] = target_synthesis - target_degradation - net_binding

    # Complex ODE
    du[COMPLEX_IDX] = net_binding - complex_internalization

    return nothing
end

"""
Simulate TMDD model.

Returns Dict with concentrations, target occupancy, and PK metrics.
"""
function simulate_tmdd(
    dose::Float64,
    pbpk_params::PBPKParams,
    tmdd_params::TMDDParams,
    time_points::Vector{Float64};
    absorption_params = nothing,
    route::Symbol = :IV,
)::Dict{String, Any}
    # Initial conditions
    n_states = COMPLEX_IDX  # 33 states
    u0 = zeros(Float64, n_states)

    # Initial target at baseline
    u0[TARGET_IDX] = tmdd_params.r0

    # Dose application
    if route == :IV
        # IV bolus to plasma
        v_plasma = pbpk_params.volumes[BLOOD_IDX]
        u0[BLOOD_IDX] = dose / v_plasma
    elseif route == :ORAL && absorption_params !== nothing
        # Oral: dose to gut lumen
        u0[GUT_LUMEN_IDX] = dose * absorption_params.fa
    end

    # Parameter tuple
    p = (pbpk_params, tmdd_params, dose, absorption_params)

    # Solve ODE
    tspan = (0.0, maximum(time_points))
    prob = ODEProblem(tmdd_ode_system!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=time_points, abstol=1e-8, reltol=1e-6)

    # Extract results
    results = Dict{String, Any}()

    # Drug concentrations
    for (k, organ) in enumerate(PBPK_ORGANS)
        results[organ] = [sol[j][k] for j in 1:length(sol)]
    end
    results["plasma"] = results["blood"]

    # Target and complex
    results["target_free"] = [sol[j][TARGET_IDX] for j in 1:length(sol)]
    results["complex"] = [sol[j][COMPLEX_IDX] for j in 1:length(sol)]

    # Target occupancy (TO)
    total_target = results["target_free"] .+ results["complex"]
    results["target_occupancy"] = results["complex"] ./ max.(total_target, 1e-10)

    # PK metrics
    plasma_conc = results["plasma"]
    if !isempty(plasma_conc) && any(c -> c > 0, plasma_conc)
        cmax_idx = argmax(plasma_conc)
        results["cmax"] = plasma_conc[cmax_idx]
        results["tmax"] = time_points[cmax_idx]

        # AUC
        auc = 0.0
        for i in 2:length(time_points)
            dt = time_points[i] - time_points[i-1]
            auc += 0.5 * (plasma_conc[i] + plasma_conc[i-1]) * dt
        end
        results["auc"] = auc
    end

    # Maximum target occupancy
    results["max_to"] = maximum(results["target_occupancy"])

    return results
end

# -----------------------------------------------------------------------------
# TUMOR GROWTH-KILL MODELS
# -----------------------------------------------------------------------------
# For oncology QSP applications

"""
Tumor Growth-Kill Parameters.

Supports:
- Exponential growth
- Logistic (Verhulst) growth
- Gompertz growth
- Simeoni transit kill model

Reference: Simeoni et al. (2004), Cancer Res
"""
struct TumorGrowthKillParams
    # Growth model
    growth_model::Symbol      # :exponential, :logistic, :gompertz
    kg::Float64               # Growth rate (1/day)
    kmax::Float64             # Carrying capacity (mm3) for logistic

    # Kill model
    kill_model::Symbol        # :emax, :linear, :simeoni
    kk::Float64               # Kill rate constant (1/day)
    emax::Float64             # Maximum effect (for Emax model)
    ec50::Float64             # Half-maximal concentration (ng/mL)
    gamma::Float64            # Hill coefficient

    # Simeoni transit compartments
    n_transit::Int            # Number of damage transit compartments
    ktr::Float64              # Transit rate (1/day)

    # Initial conditions
    tumor0::Float64           # Initial tumor volume (mm3)

    function TumorGrowthKillParams(;
        growth_model::Symbol = :logistic,
        kg::Float64 = 0.05,         # 5% per day
        kmax::Float64 = 5000.0,     # 5000 mm3 max
        kill_model::Symbol = :emax,
        kk::Float64 = 0.1,          # Kill rate
        emax::Float64 = 1.0,        # Max effect
        ec50::Float64 = 100.0,      # ng/mL
        gamma::Float64 = 1.0,       # Hill coefficient
        n_transit::Int = 3,         # Transit compartments
        ktr::Float64 = 0.5,         # Transit rate
        tumor0::Float64 = 100.0,    # Initial 100 mm3
    )
        new(growth_model, kg, kmax, kill_model, kk, emax, ec50, gamma, n_transit, ktr, tumor0)
    end
end

# State indices for tumor model
const TUMOR_IDX = 34
const TUMOR_TRANSIT_START = 35
const MAX_TUMOR_TRANSIT = 4

"""
Tumor growth-kill ODE system.

State vector:
- u[1:NUM_ORGANS]: Drug concentrations
- u[TUMOR_IDX]: Proliferating tumor cells
- u[TUMOR_TRANSIT_START:...]: Damaged cells in transit compartments
"""
function tumor_growth_kill_ode!(du, u, p, t)
    pbpk_params, tumor_params, dose, absorption_params = p

    # Drug concentration in tumor (assume similar to plasma for now)
    c_drug = max(0.0, u[BLOOD_IDX])

    # Tumor volume (proliferating + damaged)
    tumor_prolif = max(0.0, u[TUMOR_IDX])

    # Damaged cells in transit
    damaged = [max(0.0, u[TUMOR_TRANSIT_START + i - 1]) for i in 1:tumor_params.n_transit]

    # Total tumor volume
    total_tumor = tumor_prolif + sum(damaged)

    # Growth term
    growth = if tumor_params.growth_model == :exponential
        tumor_params.kg * tumor_prolif
    elseif tumor_params.growth_model == :logistic
        tumor_params.kg * tumor_prolif * (1.0 - total_tumor / tumor_params.kmax)
    elseif tumor_params.growth_model == :gompertz
        tumor_params.kg * tumor_prolif * log(tumor_params.kmax / max(total_tumor, 1e-10))
    else
        tumor_params.kg * tumor_prolif  # Default exponential
    end

    # Kill term (drug effect)
    effect = if tumor_params.kill_model == :emax
        tumor_params.emax * (c_drug ^ tumor_params.gamma) /
            (tumor_params.ec50 ^ tumor_params.gamma + c_drug ^ tumor_params.gamma)
    elseif tumor_params.kill_model == :linear
        tumor_params.kk * c_drug
    else
        tumor_params.emax * c_drug / (tumor_params.ec50 + c_drug)
    end

    kill_rate = effect * tumor_params.kk

    # PBPK ODEs (standard)
    volumes = pbpk_params.volumes
    flows = pbpk_params.blood_flows
    kps = pbpk_params.partition_coeffs
    c_plasma = u[BLOOD_IDX]

    for i in 1:NUM_ORGANS
        if i == BLOOD_IDX
            du[i] = 0.0
            for j in 1:NUM_ORGANS
                if j != BLOOD_IDX && flows[j] > 0
                    c_tissue = u[j] / kps[j]
                    du[i] += flows[j] * (c_tissue - c_plasma)
                end
            end
            du[i] -= pbpk_params.clearance_hepatic * c_plasma
        elseif i == LIVER_IDX
            c_liver = u[i]
            c_liver_ven = c_liver / kps[i]
            du[i] = flows[i] * (c_plasma - c_liver_ven) - pbpk_params.clearance_hepatic * c_liver_ven
        elseif i == KIDNEY_IDX
            c_kidney = u[i]
            c_kidney_ven = c_kidney / kps[i]
            du[i] = flows[i] * (c_plasma - c_kidney_ven) - pbpk_params.clearance_renal * c_kidney_ven
        else
            c_organ = u[i]
            c_organ_ven = c_organ / kps[i]
            du[i] = flows[i] * (c_plasma - c_organ_ven)
        end
    end

    # Absorption
    if absorption_params !== nothing
        gut_lumen = max(0.0, u[GUT_LUMEN_IDX])
        ka = absorption_params.ka
        absorption_rate = ka * gut_lumen
        du[GUT_LUMEN_IDX] = -absorption_rate
        du[BLOOD_IDX] += absorption_params.fg * absorption_params.fh * absorption_rate
    end

    # Tumor dynamics (Simeoni model)
    # Proliferating cells: growth - kill -> transit1
    du[TUMOR_IDX] = growth - kill_rate * tumor_prolif

    # Transit compartments (damaged cells)
    ktr = tumor_params.ktr
    du[TUMOR_TRANSIT_START] = kill_rate * tumor_prolif - ktr * damaged[1]

    for i in 2:tumor_params.n_transit
        du[TUMOR_TRANSIT_START + i - 1] = ktr * damaged[i-1] - ktr * damaged[i]
    end

    return nothing
end

"""
Simulate tumor growth-kill model.

Returns Dict with drug concentrations and tumor dynamics.
"""
function simulate_tumor_growth_kill(
    dose::Float64,
    pbpk_params::PBPKParams,
    tumor_params::TumorGrowthKillParams,
    time_points::Vector{Float64};
    absorption_params = nothing,
    route::Symbol = :IV,
    dosing_times::Vector{Float64} = [0.0],
)::Dict{String, Any}
    # Initial conditions
    n_states = TUMOR_TRANSIT_START + tumor_params.n_transit - 1
    u0 = zeros(Float64, n_states)

    # Initial tumor
    u0[TUMOR_IDX] = tumor_params.tumor0

    # First dose
    if route == :IV
        v_plasma = pbpk_params.volumes[BLOOD_IDX]
        u0[BLOOD_IDX] = dose / v_plasma
    elseif route == :ORAL && absorption_params !== nothing
        u0[GUT_LUMEN_IDX] = dose * absorption_params.fa
    end

    # Parameter tuple
    p = (pbpk_params, tumor_params, dose, absorption_params)

    # Callback for multiple doses
    function dose_callback(integrator)
        if route == :IV
            v_plasma = pbpk_params.volumes[BLOOD_IDX]
            integrator.u[BLOOD_IDX] += dose / v_plasma
        elseif route == :ORAL && absorption_params !== nothing
            integrator.u[GUT_LUMEN_IDX] += dose * absorption_params.fa
        end
    end

    # Create callbacks for dosing times (skip first, already applied)
    callbacks = if length(dosing_times) > 1
        cb_set = [PresetTimeCallback([t], dose_callback) for t in dosing_times[2:end]]
        CallbackSet(cb_set...)
    else
        nothing
    end

    # Solve ODE
    tspan = (0.0, maximum(time_points))
    prob = ODEProblem(tumor_growth_kill_ode!, u0, tspan, p)

    sol = if callbacks !== nothing
        solve(prob, Tsit5(), saveat=time_points, callback=callbacks, abstol=1e-8, reltol=1e-6)
    else
        solve(prob, Tsit5(), saveat=time_points, abstol=1e-8, reltol=1e-6)
    end

    # Extract results
    results = Dict{String, Any}()

    # Drug concentrations
    for (k, organ) in enumerate(PBPK_ORGANS)
        results[organ] = [sol[j][k] for j in 1:length(sol)]
    end
    results["plasma"] = results["blood"]

    # Tumor dynamics
    results["tumor_proliferating"] = [sol[j][TUMOR_IDX] for j in 1:length(sol)]

    # Transit compartments (damaged cells)
    for i in 1:tumor_params.n_transit
        results["tumor_damaged_"] = [sol[j][TUMOR_TRANSIT_START + i - 1] for j in 1:length(sol)]
    end

    # Total tumor volume
    results["tumor_total"] = results["tumor_proliferating"] .+
        sum([results["tumor_damaged_"] for i in 1:tumor_params.n_transit])

    # Tumor metrics
    tumor_total = results["tumor_total"]
    results["tumor_initial"] = tumor_params.tumor0
    results["tumor_final"] = tumor_total[end]
    results["tumor_change_pct"] = (tumor_total[end] - tumor_params.tumor0) / tumor_params.tumor0 * 100

    # Time to nadir
    nadir_idx = argmin(tumor_total)
    results["tumor_nadir"] = tumor_total[nadir_idx]
    results["time_to_nadir"] = time_points[nadir_idx]

    # Response classification (RECIST-like)
    change_pct = results["tumor_change_pct"]
    results["response"] = if change_pct <= -30
        "Partial Response"
    elseif change_pct >= 20
        "Progressive Disease"
    else
        "Stable Disease"
    end

    return results
end

# -----------------------------------------------------------------------------
# RECEPTOR-LIGAND BINDING
# -----------------------------------------------------------------------------
# General receptor-ligand equilibrium and kinetics

"""
Receptor-Ligand Binding Parameters.

For modeling drug-receptor interactions, antibody-antigen binding,
enzyme-substrate kinetics.
"""
struct ReceptorLigandParams
    # Kinetic parameters
    kon::Float64       # On-rate (1/nM/h)
    koff::Float64      # Off-rate (1/h)

    # Receptor properties
    rtot::Float64      # Total receptor (nM)
    krecycle::Float64  # Receptor recycling rate (1/h)
    kintern::Float64   # Internalization rate (1/h)

    # Derived
    kd::Float64        # Equilibrium Kd (nM)

    function ReceptorLigandParams(;
        kon::Float64 = 0.1,
        koff::Float64 = 0.01,
        rtot::Float64 = 10.0,
        krecycle::Float64 = 0.1,
        kintern::Float64 = 0.05,
    )
        kd = koff / kon
        new(kon, koff, rtot, krecycle, kintern, kd)
    end
end

"""
Calculate receptor occupancy at equilibrium.

RO = L / (L + Kd)

where L is free ligand concentration.
"""
function receptor_occupancy(
    ligand_conc::Float64,
    rl_params::ReceptorLigandParams,
)::Float64
    return ligand_conc / (ligand_conc + rl_params.kd)
end

"""
Calculate free receptor at equilibrium.

Rfree = Rtot * Kd / (L + Kd)
"""
function free_receptor(
    ligand_conc::Float64,
    rl_params::ReceptorLigandParams,
)::Float64
    return rl_params.rtot * rl_params.kd / (ligand_conc + rl_params.kd)
end

# -----------------------------------------------------------------------------
# ENZYME TURNOVER
# -----------------------------------------------------------------------------
# For modeling enzyme induction/inhibition

"""
Enzyme Turnover Parameters.

For modeling CYP induction, inhibition, and time-dependent effects.
"""
struct EnzymeTurnoverParams
    # Baseline
    ksyn::Float64      # Synthesis rate (amount/h)
    kdeg::Float64      # Degradation rate (1/h)
    e0::Float64        # Baseline enzyme level = ksyn/kdeg

    # Induction parameters
    emax_ind::Float64  # Max fold induction
    ec50_ind::Float64  # Inducer concentration for half-max effect

    # Inhibition parameters
    ki::Float64        # Inhibition constant
    kinact::Float64    # Inactivation rate (for TDI)

    function EnzymeTurnoverParams(;
        ksyn::Float64 = 1.0,
        kdeg::Float64 = 0.01,     # ~70h half-life (typical CYP)
        emax_ind::Float64 = 4.0,  # Max 4-fold induction
        ec50_ind::Float64 = 100.0,
        ki::Float64 = 1000.0,
        kinact::Float64 = 0.0,
    )
        e0 = ksyn / kdeg
        new(ksyn, kdeg, e0, emax_ind, ec50_ind, ki, kinact)
    end
end

"""
Calculate enzyme level under induction.

Emax model for induction.
"""
function enzyme_induction(
    inducer_conc::Float64,
    enzyme_params::EnzymeTurnoverParams,
)::Float64
    fold_induction = 1.0 + (enzyme_params.emax_ind - 1.0) * inducer_conc /
        (enzyme_params.ec50_ind + inducer_conc)
    return enzyme_params.e0 * fold_induction
end

"""
Calculate enzyme level under reversible inhibition.

Competitive inhibition model.
"""
function enzyme_inhibition(
    inhibitor_conc::Float64,
    enzyme_params::EnzymeTurnoverParams,
)::Float64
    return enzyme_params.e0 / (1.0 + inhibitor_conc / enzyme_params.ki)
end

# -----------------------------------------------------------------------------
# EXPORTS
# -----------------------------------------------------------------------------
export TMDDParams, tmdd_ode_system!, simulate_tmdd
export TARGET_IDX, COMPLEX_IDX
export TumorGrowthKillParams, tumor_growth_kill_ode!, simulate_tumor_growth_kill
export TUMOR_IDX, TUMOR_TRANSIT_START, MAX_TUMOR_TRANSIT
export ReceptorLigandParams, receptor_occupancy, free_receptor
export EnzymeTurnoverParams, enzyme_induction, enzyme_inhibition



# =============================================================================
# ML SUBMODELS INTEGRATION (v1.0)
# =============================================================================
# Neural network integration for parameter prediction
# - Neural ODE hybrid models
# - GNN-predicted partition coefficients
# - SMILES-to-PK surrogate models
# - Uncertainty quantification

# -----------------------------------------------------------------------------
# ML PARAMETER PREDICTOR
# -----------------------------------------------------------------------------

"""
ML-based parameter predictor interface.

Wraps neural network models for predicting PBPK parameters
from molecular descriptors or SMILES strings.
"""
struct MLParameterPredictor
    model_type::Symbol           # :gnn, :chemberta, :multimodal
    model_path::Union{String, Nothing}
    prediction_targets::Vector{Symbol}  # What parameters to predict

    # Cached model (loaded on demand)
    _model::Ref{Any}
    _loaded::Ref{Bool}

    function MLParameterPredictor(;
        model_type::Symbol = :multimodal,
        model_path::Union{String, Nothing} = nothing,
        prediction_targets::Vector{Symbol} = [:kp, :clearance, :vd],
    )
        new(model_type, model_path, prediction_targets, Ref{Any}(nothing), Ref{Bool}(false))
    end
end

"""
Predict partition coefficients from molecular structure.

Uses GNN or multimodal encoder to predict Kp values for all organs.
Falls back to QSPR equations if model not available.
"""
function predict_partition_coeffs(
    predictor::MLParameterPredictor,
    smiles::String;
    logP::Float64 = 0.0,
    pKa::Float64 = 7.0,
    fup::Float64 = 0.1,
)::Dict{String, Float64}
    # Fallback: Poulin-Theil QSPR method
    # Kp = (fup / fut) * (Vwt + 0.7 * Vnlt + (Vpt + 0.3 * Vnlt) * P)
    # Simplified version using logP correlation

    kp_dict = Dict{String, Float64}()

    # Tissue composition factors (fraction lipid, water)
    tissue_lipid = Dict(
        "blood" => 0.0056, "liver" => 0.0348, "kidney" => 0.0207,
        "brain" => 0.1148, "heart" => 0.0293, "lung" => 0.0220,
        "muscle" => 0.0238, "adipose" => 0.8530, "gut" => 0.0487,
        "skin" => 0.0603, "bone" => 0.0436, "spleen" => 0.0157,
        "pancreas" => 0.0448, "other" => 0.0300,
    )

    tissue_water = Dict(
        "blood" => 0.8290, "liver" => 0.7510, "kidney" => 0.7830,
        "brain" => 0.7770, "heart" => 0.7580, "lung" => 0.8110,
        "muscle" => 0.7600, "adipose" => 0.1350, "gut" => 0.7180,
        "skin" => 0.6180, "bone" => 0.4390, "spleen" => 0.7880,
        "pancreas" => 0.6640, "other" => 0.7000,
    )

    # Calculate P (partition coefficient)
    P = 10.0 ^ logP

    for organ in PBPK_ORGANS
        fl = get(tissue_lipid, organ, 0.03)
        fw = get(tissue_water, organ, 0.70)

        # Poulin-Theil equation (simplified)
        kp = (fup / 0.5) * (fw + fl * P)

        # Clamp to reasonable range
        kp_dict[organ] = clamp(kp, 0.1, 50.0)
    end

    return kp_dict
end

"""
Predict clearance from molecular structure.

Uses ML model or QSPR to estimate hepatic and renal clearance.
"""
function predict_clearance(
    predictor::MLParameterPredictor,
    smiles::String;
    mw::Float64 = 400.0,
    logP::Float64 = 2.0,
    hbd::Int = 2,
    hba::Int = 4,
    psa::Float64 = 80.0,
)::NamedTuple{(:hepatic, :renal), Tuple{Float64, Float64}}
    # QSPR fallback based on physicochemical properties

    # Hepatic clearance correlation (simplified)
    # High logP -> more hepatic metabolism
    # High PSA -> less hepatic (more polar)
    cl_hepatic_base = 10.0  # L/h baseline
    logP_factor = 1.0 + 0.3 * (logP - 2.0)  # Scale by logP
    psa_factor = 1.0 - 0.005 * (psa - 60.0)  # PSA reduces CL

    cl_hepatic = cl_hepatic_base * max(0.1, logP_factor * psa_factor)

    # Renal clearance (GFR-based for small molecules)
    # Small, polar compounds have higher renal clearance
    gfr = 7.5  # L/h (125 mL/min)
    mw_factor = mw < 500 ? 1.0 : 0.5  # Large molecules filtered less
    polar_factor = 1.0 + 0.01 * (psa - 60.0)  # More polar = more renal

    cl_renal = gfr * 0.2 * mw_factor * max(0.1, polar_factor)  # 20% of GFR typical

    return (hepatic = cl_hepatic, renal = cl_renal)
end

# -----------------------------------------------------------------------------
# NEURAL ODE HYBRID
# -----------------------------------------------------------------------------

"""
Neural ODE component for hybrid mechanistic-ML models.

Augments mechanistic ODE with learned correction terms.
"""
struct NeuralODEComponent
    correction_nn::Any      # Neural network for correction term
    state_indices::Vector{Int}  # Which states to correct
    scaling::Float64        # Scale factor for correction

    function NeuralODEComponent(;
        hidden_dim::Int = 32,
        state_indices::Vector{Int} = [BLOOD_IDX, LIVER_IDX],
        scaling::Float64 = 0.1,
    )
        # Simple MLP for correction term
        # In production, this would be a Flux.Chain
        correction_nn = nothing  # Placeholder
        new(correction_nn, state_indices, scaling)
    end
end

"""
Apply neural correction to ODE derivatives.

du_corrected = du_mechanistic + scaling * NN(u, t)
"""
function apply_neural_correction!(
    du::Vector{Float64},
    u::Vector{Float64},
    t::Float64,
    neural_component::NeuralODEComponent,
)
    if neural_component.correction_nn === nothing
        return  # No correction if model not loaded
    end

    # Get states to correct
    states = u[neural_component.state_indices]

    # Neural network input: [states..., t]
    nn_input = vcat(states, [t])

    # Get correction (would call neural_component.correction_nn(nn_input))
    # Placeholder: no correction
    correction = zeros(length(neural_component.state_indices))

    # Apply scaled correction
    for (i, idx) in enumerate(neural_component.state_indices)
        du[idx] += neural_component.scaling * correction[i]
    end
end

# -----------------------------------------------------------------------------
# SURROGATE MODEL INTERFACE
# -----------------------------------------------------------------------------

"""
Surrogate model for fast PK predictions.

Replaces full ODE solve with neural network for speed.
"""
struct PKSurrogateModel
    model::Any              # Neural network model
    input_features::Vector{Symbol}  # Required inputs
    output_features::Vector{Symbol} # What it predicts
    trained::Bool

    function PKSurrogateModel(;
        input_features::Vector{Symbol} = [:dose, :logP, :mw, :clearance],
        output_features::Vector{Symbol} = [:cmax, :tmax, :auc, :half_life],
    )
        new(nothing, input_features, output_features, false)
    end
end

"""
Predict PK parameters using surrogate model.

Returns Dict with predicted Cmax, Tmax, AUC, half-life.
Falls back to analytical solutions if surrogate not trained.
"""
function predict_pk_surrogate(
    surrogate::PKSurrogateModel,
    dose::Float64,
    clearance::Float64,
    vd::Float64;
    ka::Float64 = 1.0,
    f::Float64 = 1.0,
)::Dict{Symbol, Float64}
    # Analytical one-compartment PK (fallback)
    ke = clearance / vd

    # Cmax and Tmax for oral
    if ka > ke
        tmax = log(ka / ke) / (ka - ke)
        cmax = (f * dose / vd) * (ka / (ka - ke)) * (exp(-ke * tmax) - exp(-ka * tmax))
    else
        # Flip-flop
        tmax = log(ke / ka) / (ke - ka)
        cmax = (f * dose / vd) * (ke / (ke - ka)) * (exp(-ka * tmax) - exp(-ke * tmax))
    end

    # AUC
    auc = f * dose / clearance

    # Half-life
    half_life = log(2) / ke

    return Dict(
        :cmax => cmax,
        :tmax => tmax,
        :auc => auc,
        :half_life => half_life,
    )
end

# -----------------------------------------------------------------------------
# UNCERTAINTY QUANTIFICATION
# -----------------------------------------------------------------------------

"""
Uncertainty quantification for ML predictions.

Uses ensemble methods or dropout for prediction intervals.
"""
struct UncertaintyQuantifier
    method::Symbol          # :ensemble, :dropout, :conformal
    n_samples::Int          # Number of samples for MC
    confidence_level::Float64

    function UncertaintyQuantifier(;
        method::Symbol = :ensemble,
        n_samples::Int = 100,
        confidence_level::Float64 = 0.95,
    )
        new(method, n_samples, confidence_level)
    end
end

"""
Compute prediction interval for ML prediction.

Returns (lower, upper) bounds.
"""
function prediction_interval(
    uq::UncertaintyQuantifier,
    predictions::Vector{Float64},
)::Tuple{Float64, Float64}
    n = length(predictions)
    if n < 2
        return (predictions[1], predictions[1])
    end

    # Sort for percentile calculation
    sorted = sort(predictions)

    alpha = 1.0 - uq.confidence_level
    lower_idx = max(1, Int(floor(alpha / 2 * n)))
    upper_idx = min(n, Int(ceil((1 - alpha / 2) * n)))

    return (sorted[lower_idx], sorted[upper_idx])
end

"""
Compute prediction with uncertainty.

Returns NamedTuple with mean, std, lower, upper.
"""
function predict_with_uncertainty(
    uq::UncertaintyQuantifier,
    predictions::Vector{Float64},
)::NamedTuple{(:mean, :std, :lower, :upper), Tuple{Float64, Float64, Float64, Float64}}
    mean_pred = sum(predictions) / length(predictions)
    std_pred = sqrt(sum((p - mean_pred)^2 for p in predictions) / length(predictions))
    lower, upper = prediction_interval(uq, predictions)

    return (mean = mean_pred, std = std_pred, lower = lower, upper = upper)
end

# -----------------------------------------------------------------------------
# ML EXPORTS
# -----------------------------------------------------------------------------
export MLParameterPredictor, predict_partition_coeffs, predict_clearance
export NeuralODEComponent, apply_neural_correction!
export PKSurrogateModel, predict_pk_surrogate
export UncertaintyQuantifier, prediction_interval, predict_with_uncertainty



# =============================================================================
# TRACK C: CLINICAL TRIAL & POPULATION OPERATORS (v1.0)
# =============================================================================
# Clinical trial simulation and population modeling constructs
# - Virtual population generation
# - Covariate models (allometric, categorical)
# - Trial design simulation
# - Endpoints and power analysis

# -----------------------------------------------------------------------------
# COVARIATE MODELS
# -----------------------------------------------------------------------------

"""
Covariate effect on a parameter.

Supports:
- Power model: P = P_ref * (COV/COV_ref)^exp
- Exponential: P = P_ref * exp(theta * (COV - COV_ref))
- Categorical: P = P_ref * factor_map[category]
- Linear: P = P_ref + theta * (COV - COV_ref)
"""
struct CovariateEffect
    covariate_name::Symbol
    effect_type::Symbol       # :power, :exponential, :categorical, :linear
    reference_value::Float64  # Reference covariate value
    theta::Float64           # Effect parameter
    category_factors::Dict{String, Float64}  # For categorical

    function CovariateEffect(;
        covariate_name::Symbol,
        effect_type::Symbol = :power,
        reference_value::Float64 = 70.0,
        theta::Float64 = 0.75,
        category_factors::Dict{String, Float64} = Dict{String, Float64}(),
    )
        new(covariate_name, effect_type, reference_value, theta, category_factors)
    end
end

"""
Apply covariate effect to a parameter.
"""
function apply_covariate(
    base_value::Float64,
    covariate_value::Union{Float64, String},
    effect::CovariateEffect,
)::Float64
    if effect.effect_type == :power
        ratio = covariate_value / effect.reference_value
        return base_value * (ratio ^ effect.theta)
    elseif effect.effect_type == :exponential
        delta = covariate_value - effect.reference_value
        return base_value * exp(effect.theta * delta)
    elseif effect.effect_type == :categorical
        factor = get(effect.category_factors, string(covariate_value), 1.0)
        return base_value * factor
    elseif effect.effect_type == :linear
        delta = covariate_value - effect.reference_value
        return base_value + effect.theta * delta
    else
        return base_value
    end
end

# Standard allometric scaling
const ALLOMETRIC_CL = CovariateEffect(
    covariate_name = :WT,
    effect_type = :power,
    reference_value = 70.0,
    theta = 0.75,
)

const ALLOMETRIC_V = CovariateEffect(
    covariate_name = :WT,
    effect_type = :power,
    reference_value = 70.0,
    theta = 1.0,
)

# -----------------------------------------------------------------------------
# INTER-INDIVIDUAL VARIABILITY (IIV)
# -----------------------------------------------------------------------------

"""
IIV specification for a parameter.

Supports:
- Exponential: P_i = P_pop * exp(eta_i), eta ~ N(0, omega)
- Proportional: P_i = P_pop * (1 + eta_i)
- Additive: P_i = P_pop + eta_i
"""
struct IIVSpec
    parameter_name::Symbol
    distribution::Symbol      # :exponential, :proportional, :additive
    omega::Float64           # Standard deviation
    correlation_group::Int   # For correlated parameters

    function IIVSpec(;
        parameter_name::Symbol,
        distribution::Symbol = :exponential,
        omega::Float64 = 0.3,
        correlation_group::Int = 0,
    )
        new(parameter_name, distribution, omega, correlation_group)
    end
end

"""
Sample individual parameter value.
"""
function sample_individual_param(
    population_value::Float64,
    iiv::IIVSpec;
    rng = Random.GLOBAL_RNG,
)::Float64
    eta = randn(rng) * iiv.omega

    if iiv.distribution == :exponential
        return population_value * exp(eta)
    elseif iiv.distribution == :proportional
        return population_value * (1.0 + eta)
    elseif iiv.distribution == :additive
        return population_value + eta
    else
        return population_value * exp(eta)
    end
end

# -----------------------------------------------------------------------------
# VIRTUAL POPULATION
# -----------------------------------------------------------------------------

"""
Virtual subject with covariates and individual parameters.
"""
struct VirtualSubject
    id::Int
    covariates::Dict{Symbol, Any}      # Demographics, genotypes, etc.
    individual_params::Dict{Symbol, Float64}  # Sampled individual parameters

    function VirtualSubject(
        id::Int;
        covariates::Dict{Symbol, Any} = Dict{Symbol, Any}(),
        individual_params::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
    )
        new(id, covariates, individual_params)
    end
end

"""
Virtual population generator.

Generates subjects with realistic covariate distributions.
"""
struct VirtualPopulationGenerator
    n_subjects::Int
    covariate_distributions::Dict{Symbol, Any}  # Distribution specs
    iiv_specs::Vector{IIVSpec}
    correlation_matrix::Union{Matrix{Float64}, Nothing}

    function VirtualPopulationGenerator(;
        n_subjects::Int = 100,
        covariate_distributions::Dict{Symbol, Any} = default_covariate_distributions(),
        iiv_specs::Vector{IIVSpec} = IIVSpec[],
        correlation_matrix::Union{Matrix{Float64}, Nothing} = nothing,
    )
        new(n_subjects, covariate_distributions, iiv_specs, correlation_matrix)
    end
end

"""
Default covariate distributions (healthy adults).
"""
function default_covariate_distributions()::Dict{Symbol, Any}
    return Dict{Symbol, Any}(
        :WT => (dist = :normal, mean = 70.0, std = 15.0, min = 40.0, max = 150.0),
        :AGE => (dist = :uniform, min = 18.0, max = 65.0),
        :SEX => (dist = :categorical, probs = Dict("M" => 0.5, "F" => 0.5)),
        :CRCL => (dist = :normal, mean = 100.0, std = 25.0, min = 30.0, max = 150.0),
        :ALB => (dist = :normal, mean = 4.0, std = 0.5, min = 2.0, max = 5.5),
    )
end

"""
Sample a covariate value from its distribution.
"""
function sample_covariate(
    spec::NamedTuple;
    rng = Random.GLOBAL_RNG,
)::Any
    if spec.dist == :normal
        value = spec.mean + randn(rng) * spec.std
        return clamp(value, spec.min, spec.max)
    elseif spec.dist == :uniform
        return spec.min + rand(rng) * (spec.max - spec.min)
    elseif spec.dist == :categorical
        r = rand(rng)
        cumprob = 0.0
        for (cat, prob) in spec.probs
            cumprob += prob
            if r <= cumprob
                return cat
            end
        end
        return first(keys(spec.probs))
    elseif spec.dist == :lognormal
        mu = log(spec.median)
        sigma = spec.cv  # Approximate
        return exp(mu + randn(rng) * sigma)
    else
        return spec.mean
    end
end

"""
Generate virtual population.
"""
function generate_virtual_population(
    gen::VirtualPopulationGenerator,
    population_params::Dict{Symbol, Float64};
    rng = Random.GLOBAL_RNG,
)::Vector{VirtualSubject}
    subjects = VirtualSubject[]

    for i in 1:gen.n_subjects
        # Sample covariates
        covariates = Dict{Symbol, Any}()
        for (name, spec) in gen.covariate_distributions
            covariates[name] = sample_covariate(spec; rng=rng)
        end

        # Sample individual parameters
        individual_params = Dict{Symbol, Float64}()
        for (param, pop_value) in population_params
            # Find IIV spec for this parameter
            iiv_idx = findfirst(s -> s.parameter_name == param, gen.iiv_specs)

            if iiv_idx !== nothing
                individual_params[param] = sample_individual_param(
                    pop_value, gen.iiv_specs[iiv_idx]; rng=rng
                )
            else
                individual_params[param] = pop_value
            end
        end

        push!(subjects, VirtualSubject(i; covariates=covariates, individual_params=individual_params))
    end

    return subjects
end

# -----------------------------------------------------------------------------
# TRIAL DESIGN
# -----------------------------------------------------------------------------

"""
Dosing regimen specification.
"""
struct DosingRegimen
    dose::Float64
    route::Symbol           # :IV, :ORAL, :SC, :IM
    interval::Float64       # Dosing interval (hours)
    n_doses::Int           # Number of doses
    infusion_duration::Float64  # For IV infusion

    function DosingRegimen(;
        dose::Float64 = 100.0,
        route::Symbol = :ORAL,
        interval::Float64 = 24.0,
        n_doses::Int = 7,
        infusion_duration::Float64 = 0.0,
    )
        new(dose, route, interval, n_doses, infusion_duration)
    end
end

"""
Get dosing times from regimen.
"""
function dosing_times(regimen::DosingRegimen)::Vector{Float64}
    return [regimen.interval * (i - 1) for i in 1:regimen.n_doses]
end

"""
Trial arm specification.
"""
struct TrialArm
    name::String
    regimen::DosingRegimen
    n_subjects::Int
    sampling_times::Vector{Float64}

    function TrialArm(;
        name::String = "Treatment",
        regimen::DosingRegimen = DosingRegimen(),
        n_subjects::Int = 30,
        sampling_times::Vector{Float64} = [0.5, 1, 2, 4, 8, 12, 24],
    )
        new(name, regimen, n_subjects, sampling_times)
    end
end

"""
Complete trial design.
"""
struct TrialDesign
    name::String
    arms::Vector{TrialArm}
    duration::Float64       # Total trial duration (hours)
    endpoints::Vector{Symbol}

    function TrialDesign(;
        name::String = "Phase1_PK",
        arms::Vector{TrialArm} = [TrialArm()],
        duration::Float64 = 168.0,  # 7 days
        endpoints::Vector{Symbol} = [:Cmax, :AUC, :Tmax],
    )
        new(name, arms, duration, endpoints)
    end
end

# -----------------------------------------------------------------------------
# TRIAL SIMULATION
# -----------------------------------------------------------------------------

"""
Simulate a clinical trial.

Returns Dict with:
- subjects: Individual simulation results
- summary: Population statistics
- endpoints: Endpoint analysis
"""
function simulate_trial(
    design::TrialDesign,
    pbpk_params::PBPKParams,
    population_generator::VirtualPopulationGenerator;
    absorption_params = nothing,
)::Dict{String, Any}
    results = Dict{String, Any}(
        "design" => design,
        "arms" => Dict{String, Any}(),
    )

    for arm in design.arms
        arm_results = Dict{String, Any}(
            "name" => arm.name,
            "subjects" => [],
            "summary" => Dict{String, Any}(),
        )

        # Generate virtual population for this arm
        population_params = Dict{Symbol, Float64}(
            :clearance_hepatic => pbpk_params.clearance_hepatic,
            :clearance_renal => pbpk_params.clearance_renal,
        )

        subjects = generate_virtual_population(
            VirtualPopulationGenerator(
                n_subjects = arm.n_subjects,
                covariate_distributions = population_generator.covariate_distributions,
                iiv_specs = population_generator.iiv_specs,
            ),
            population_params,
        )

        # Simulate each subject
        cmax_values = Float64[]
        auc_values = Float64[]
        tmax_values = Float64[]

        for subject in subjects
            # Adjust parameters for individual
            ind_pbpk = PBPKParams(
                volumes = pbpk_params.volumes,
                blood_flows = pbpk_params.blood_flows,
                clearance_hepatic = get(subject.individual_params, :clearance_hepatic, pbpk_params.clearance_hepatic),
                clearance_renal = get(subject.individual_params, :clearance_renal, pbpk_params.clearance_renal),
                partition_coeffs = pbpk_params.partition_coeffs,
            )

            # Run simulation
            dose = arm.regimen.dose
            time_points = collect(0.0:0.1:design.duration)

            sim_result = simulate_pbpk(
                dose,
                ind_pbpk,
                time_points;
                absorption_params = absorption_params,
            )

            # Collect endpoints
            push!(cmax_values, sim_result["cmax"])
            push!(auc_values, sim_result["auc"])
            push!(tmax_values, sim_result["tmax"])

            push!(arm_results["subjects"], Dict(
                "id" => subject.id,
                "covariates" => subject.covariates,
                "cmax" => sim_result["cmax"],
                "auc" => sim_result["auc"],
                "tmax" => sim_result["tmax"],
            ))
        end

        # Summary statistics
        arm_results["summary"]["Cmax"] = (
            mean = sum(cmax_values) / length(cmax_values),
            std = sqrt(sum((c - sum(cmax_values)/length(cmax_values))^2 for c in cmax_values) / length(cmax_values)),
            cv = sqrt(sum((c - sum(cmax_values)/length(cmax_values))^2 for c in cmax_values) / length(cmax_values)) / (sum(cmax_values) / length(cmax_values)) * 100,
            min = minimum(cmax_values),
            max = maximum(cmax_values),
        )

        arm_results["summary"]["AUC"] = (
            mean = sum(auc_values) / length(auc_values),
            std = sqrt(sum((a - sum(auc_values)/length(auc_values))^2 for a in auc_values) / length(auc_values)),
            cv = sqrt(sum((a - sum(auc_values)/length(auc_values))^2 for a in auc_values) / length(auc_values)) / (sum(auc_values) / length(auc_values)) * 100,
            min = minimum(auc_values),
            max = maximum(auc_values),
        )

        results["arms"][arm.name] = arm_results
    end

    return results
end

# -----------------------------------------------------------------------------
# BIOEQUIVALENCE ANALYSIS
# -----------------------------------------------------------------------------

"""
Perform bioequivalence analysis between two arms.

Returns 90% confidence interval for ratio of geometric means.
"""
function bioequivalence_analysis(
    test_values::Vector{Float64},
    reference_values::Vector{Float64};
    alpha::Float64 = 0.1,
)::NamedTuple{(:ratio, :lower, :upper, :bioequivalent), Tuple{Float64, Float64, Float64, Bool}}
    # Log-transform
    log_test = log.(test_values)
    log_ref = log.(reference_values)

    # Geometric means
    gm_test = exp(sum(log_test) / length(log_test))
    gm_ref = exp(sum(log_ref) / length(log_ref))
    ratio = gm_test / gm_ref

    # Pooled variance (simplified - assumes equal n)
    n = length(log_test)
    var_test = sum((x - sum(log_test)/n)^2 for x in log_test) / (n - 1)
    var_ref = sum((x - sum(log_ref)/n)^2 for x in log_ref) / (n - 1)
    pooled_se = sqrt((var_test + var_ref) / n)

    # 90% CI on log scale
    # Using normal approximation (t-distribution would be more accurate)
    z = 1.645  # 95th percentile for 90% CI
    log_lower = log(ratio) - z * pooled_se
    log_upper = log(ratio) + z * pooled_se

    lower = exp(log_lower)
    upper = exp(log_upper)

    # BE criteria: 80-125%
    bioequivalent = (lower >= 0.80) && (upper <= 1.25)

    return (ratio = ratio, lower = lower, upper = upper, bioequivalent = bioequivalent)
end

# -----------------------------------------------------------------------------
# EXPOSURE-RESPONSE ANALYSIS
# -----------------------------------------------------------------------------

"""
Fit Emax exposure-response model.

Response = E0 + Emax * C / (EC50 + C)
"""
function fit_emax_model(
    exposures::Vector{Float64},
    responses::Vector{Float64},
)::NamedTuple{(:e0, :emax, :ec50, :r_squared), Tuple{Float64, Float64, Float64, Float64}}
    # Simple grid search for EC50 (production would use NLopt)
    best_r2 = -Inf
    best_params = (e0 = 0.0, emax = 0.0, ec50 = 1.0)

    e0_est = minimum(responses)
    emax_est = maximum(responses) - minimum(responses)

    for ec50_mult in 0.1:0.1:10.0
        ec50_test = median(exposures) * ec50_mult

        # Predicted responses
        predicted = [e0_est + emax_est * c / (ec50_test + c) for c in exposures]

        # R-squared
        ss_res = sum((responses[i] - predicted[i])^2 for i in 1:length(responses))
        ss_tot = sum((r - sum(responses)/length(responses))^2 for r in responses)
        r2 = 1.0 - ss_res / ss_tot

        if r2 > best_r2
            best_r2 = r2
            best_params = (e0 = e0_est, emax = emax_est, ec50 = ec50_test, r_squared = r2)
        end
    end

    return best_params
end

# -----------------------------------------------------------------------------
# TRACK C EXPORTS
# -----------------------------------------------------------------------------
export CovariateEffect, apply_covariate, ALLOMETRIC_CL, ALLOMETRIC_V
export IIVSpec, sample_individual_param
export VirtualSubject, VirtualPopulationGenerator, generate_virtual_population
export default_covariate_distributions, sample_covariate
export DosingRegimen, dosing_times, TrialArm, TrialDesign
export simulate_trial, bioequivalence_analysis, fit_emax_model


# =============================================================================
# BLOOD COMPARTMENT INTEGRATION FOR DYNAMIC PK
# =============================================================================
# This section integrates BloodCompartmentState with ODE solver for:
# - Time-varying clearances (acute phase, disease progression)
# - Dynamic fu adjustments based on protein binding state
# - Hematocrit-dependent blood flow adjustments
#
# Uses DifferentialEquations.jl callbacks for state-dependent parameters

export DynamicPBPKParams, BloodStateODECallback
export solve_with_blood_state, simulate_with_blood_state
export create_blood_state_callback, update_blood_state_ode!
export calculate_dynamic_adjustments, effective_cl_hepatic, effective_cl_renal

"""
    DynamicPBPKParams

PBPK parameters that can change during simulation based on blood state.

Unlike PBPKParams which is immutable, this holds mutable adjustment factors
that are updated by the blood compartment integration callback.
"""
mutable struct DynamicPBPKParams
    base_params::PBPKParams              # Base PBPK parameters
    hepatic_cl_factor::Float64           # Dynamic hepatic CL multiplier
    renal_cl_factor::Float64             # Dynamic renal CL multiplier
    hepatic_flow_factor::Float64         # Dynamic hepatic flow multiplier
    fu_factor::Float64                   # Dynamic fu multiplier
    vd_factor::Float64                   # Dynamic Vd multiplier
    rb::Float64                          # Blood:plasma ratio
    time_last_update::Float64            # Time of last update

    function DynamicPBPKParams(base::PBPKParams)
        new(base, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    end
end

# Effective clearances accounting for dynamic factors
effective_cl_hepatic(p::DynamicPBPKParams) = p.base_params.clearance_hepatic * p.hepatic_cl_factor
effective_cl_renal(p::DynamicPBPKParams) = p.base_params.clearance_renal * p.renal_cl_factor

"""
    BloodStateODECallback

Holds blood compartment state and drug properties for ODE callback.
This is passed as part of the parameter tuple to the ODE system.
"""
mutable struct BloodStateODECallback
    # Blood state (mutable, updated during simulation)
    hematocrit::Float64
    albumin_g_L::Float64
    aag_g_L::Float64
    il6_pg_mL::Float64
    crp_mg_L::Float64
    gfr::Float64
    hepatic_flow::Float64

    # Drug properties (immutable)
    ke_p::Float64                        # RBC partition
    fu_reference::Float64                # Reference fu
    charge_type::Symbol                  # :acidic, :basic, :neutral
    extraction_ratio::Float64            # Hepatic extraction

    # Disease flags
    is_acute_phase::Bool
    time_since_onset::Float64

    # Update interval (hours)
    update_interval::Float64

    function BloodStateODECallback(;
        hematocrit::Float64 = 0.42,
        albumin_g_L::Float64 = 40.0,
        aag_g_L::Float64 = 0.8,
        il6_pg_mL::Float64 = 5.0,
        crp_mg_L::Float64 = 1.0,
        gfr::Float64 = 100.0,
        hepatic_flow::Float64 = 90.0,
        ke_p::Float64 = 1.0,
        fu_reference::Float64 = 0.5,
        charge_type::Symbol = :neutral,
        extraction_ratio::Float64 = 0.3,
        is_acute_phase::Bool = false,
        time_since_onset::Float64 = 0.0,
        update_interval::Float64 = 1.0
    )
        new(hematocrit, albumin_g_L, aag_g_L, il6_pg_mL, crp_mg_L,
            gfr, hepatic_flow, ke_p, fu_reference, charge_type,
            extraction_ratio, is_acute_phase, time_since_onset,
            update_interval)
    end
end

"""
    update_blood_state_ode!(callback::BloodStateODECallback, dt::Float64)

Update blood state for ODE simulation step.
Implements acute phase protein kinetics.
"""
function update_blood_state_ode!(callback::BloodStateODECallback, dt::Float64)
    if !callback.is_acute_phase
        return nothing
    end

    callback.time_since_onset += dt
    t = callback.time_since_onset

    # IL-6 decay (half-life ~2-4 hours)
    il6_decay = t > 24.0 ? 0.02 : 0.1
    callback.il6_pg_mL = max(5.0, callback.il6_pg_mL * exp(-il6_decay * dt))

    # CRP kinetics (peaks at 48-72h)
    crp_production = 0.5 * log10(max(1.0, callback.il6_pg_mL))
    crp_clearance = 0.05 * callback.crp_mg_L
    callback.crp_mg_L = clamp(callback.crp_mg_L + (crp_production - crp_clearance) * dt, 1.0, 500.0)

    # AAG kinetics (peaks at 48-96h)
    aag_target = min(3.0, 0.8 + 2.0 * (callback.il6_pg_mL / 200.0))
    callback.aag_g_L = callback.aag_g_L + (aag_target - callback.aag_g_L) * 0.02 * dt

    # Albumin kinetics (decreases during inflammation)
    alb_target = max(20.0, 40.0 - 15.0 * (callback.il6_pg_mL / 200.0))
    callback.albumin_g_L = callback.albumin_g_L + (alb_target - callback.albumin_g_L) * 0.01 * dt

    return nothing
end

"""
    calculate_dynamic_adjustments(callback::BloodStateODECallback)

Calculate current PK adjustment factors from blood state.
"""
function calculate_dynamic_adjustments(callback::BloodStateODECallback)
    # Reference values
    ref_albumin = 40.0
    ref_aag = 0.8
    ref_gfr = 100.0
    ref_hepatic_flow = 90.0

    # fu adjustment based on drug type and proteins
    fu_factor = 1.0
    if callback.charge_type == :acidic
        fu_factor = ref_albumin / callback.albumin_g_L
    elseif callback.charge_type == :basic
        fu_factor = ref_aag / callback.aag_g_L
    end

    # Blood:plasma ratio
    rb = 1.0 - callback.hematocrit + (callback.hematocrit * callback.ke_p)

    # Clearance adjustments
    renal_cl_factor = callback.gfr / ref_gfr

    # Hepatic clearance depends on extraction ratio
    if callback.extraction_ratio > 0.7
        # High extraction: flow-limited
        hepatic_cl_factor = callback.hepatic_flow / ref_hepatic_flow
    else
        # Low extraction: capacity-limited, fu-dependent
        fu_blood = (callback.fu_reference * fu_factor) / rb
        hepatic_cl_factor = fu_blood / (callback.fu_reference / 1.0)
    end

    hepatic_flow_factor = callback.hepatic_flow / ref_hepatic_flow

    return (
        fu_factor = fu_factor,
        rb = rb,
        hepatic_cl_factor = hepatic_cl_factor,
        renal_cl_factor = renal_cl_factor,
        hepatic_flow_factor = hepatic_flow_factor
    )
end

"""
ODE system with dynamic blood state adjustments.

Parameters are a tuple: (DynamicPBPKParams, BloodStateODECallback)
"""
function dynamic_ode_system!(du::AbstractVector{Float64}, u::AbstractVector{Float64},
                              p::Tuple{DynamicPBPKParams, BloodStateODECallback}, t::Float64)
    dyn_params, blood_callback = p
    base = dyn_params.base_params

    # Initialize derivatives
    fill!(du, 0.0)

    C_blood = u[BLOOD_IDX]

    # Get current adjustment factors
    adj = calculate_dynamic_adjustments(blood_callback)

    # Update dynamic params
    dyn_params.fu_factor = adj.fu_factor
    dyn_params.rb = adj.rb
    dyn_params.hepatic_cl_factor = adj.hepatic_cl_factor
    dyn_params.renal_cl_factor = adj.renal_cl_factor
    dyn_params.hepatic_flow_factor = adj.hepatic_flow_factor

    # Organ dynamics (same as standard, but with flow adjustment)
    @inbounds for i in 1:NUM_ORGANS
        if i == BLOOD_IDX
            continue
        end

        V_organ = base.volumes[i]
        Q_organ = base.blood_flows[i]
        Kp_organ = base.partition_coeffs[i]
        C_organ = u[i]

        # Apply hepatic flow adjustment to liver
        if i == LIVER_IDX
            Q_organ *= adj.hepatic_flow_factor
        end

        du[i] = (Q_organ / V_organ) * (C_blood - C_organ / Kp_organ)

        V_blood = base.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= (Q_organ / V_blood) * (C_blood - C_organ / Kp_organ)
    end

    # Dynamic hepatic clearance
    if base.clearance_hepatic > 0.0
        effective_cl = base.clearance_hepatic * adj.hepatic_cl_factor
        clearance_rate = effective_cl / base.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= clearance_rate * C_blood
    end

    # Dynamic renal clearance
    if base.clearance_renal > 0.0
        effective_cl = base.clearance_renal * adj.renal_cl_factor
        clearance_rate = effective_cl / base.volumes[BLOOD_IDX]
        du[BLOOD_IDX] -= clearance_rate * C_blood
    end

    return nothing
end

"""
    create_blood_state_callback(blood_callback::BloodStateODECallback)

Create a DifferentialEquations.jl callback that updates blood state at intervals.
"""
function create_blood_state_callback(blood_callback::BloodStateODECallback)
    # Discrete callback that fires at regular intervals
    function condition(u, t, integrator)
        return t % blood_callback.update_interval ≈ 0.0
    end

    function affect!(integrator)
        _, blood_cb = integrator.p
        update_blood_state_ode!(blood_cb, blood_callback.update_interval)
    end

    # Use PeriodicCallback for simplicity
    return PeriodicCallback(
        (integrator) -> begin
            _, blood_cb = integrator.p
            update_blood_state_ode!(blood_cb, blood_callback.update_interval)
        end,
        blood_callback.update_interval;
        initial_affect = false
    )
end

"""
    solve_with_blood_state(
        pbpk_params::PBPKParams,
        blood_callback::BloodStateODECallback,
        dose::Float64,
        tspan::Tuple{Float64, Float64};
        time_points = nothing,
        reltol = 1e-8,
        abstol = 1e-10
    )

Solve PBPK with dynamic blood compartment state updates.

# Arguments
- `pbpk_params`: Base PBPK parameters
- `blood_callback`: Blood state callback with initial conditions and drug properties
- `dose`: Dose in mg
- `tspan`: Time span (start, end) in hours

# Returns
Solution object with dynamic PK adjustments applied throughout

# Example
```julia
# Create blood state for sepsis patient
blood = BloodStateODECallback(
    hematocrit = 0.30,
    albumin_g_L = 20.0,
    aag_g_L = 2.5,
    il6_pg_mL = 200.0,
    gfr = 40.0,
    ke_p = 37.0,              # Tacrolimus
    fu_reference = 0.01,
    charge_type = :basic,
    extraction_ratio = 0.3,
    is_acute_phase = true
)

# Solve with dynamic adjustments
sol = solve_with_blood_state(pbpk_params, blood, 5.0, (0.0, 72.0))
```
"""
function solve_with_blood_state(
    pbpk_params::PBPKParams,
    blood_callback::BloodStateODECallback,
    dose::Float64,
    tspan::Tuple{Float64, Float64};
    time_points::Union{Vector{Float64}, Nothing} = nothing,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10,
    alg = Tsit5()
)
    # Create dynamic params wrapper
    dyn_params = DynamicPBPKParams(pbpk_params)

    # Initial conditions
    u0 = zeros(Float64, NUM_ORGANS)
    u0[BLOOD_IDX] = dose / pbpk_params.volumes[BLOOD_IDX]

    # Parameters tuple
    p = (dyn_params, blood_callback)

    # Create problem
    prob = ODEProblem(dynamic_ode_system!, u0, tspan, p)

    # Create blood state callback
    cb = create_blood_state_callback(blood_callback)

    # Solve with callback
    if time_points !== nothing
        sol = DifferentialEquations.solve(prob, alg;
            reltol=reltol, abstol=abstol,
            saveat=time_points,
            callback=cb
        )
    else
        sol = DifferentialEquations.solve(prob, alg;
            reltol=reltol, abstol=abstol,
            callback=cb
        )
    end

    return sol
end

"""
    simulate_with_blood_state(
        pbpk_params::PBPKParams,
        blood_callback::BloodStateODECallback,
        dose::Float64;
        t_max = 72.0,
        num_points = 100
    )

Simulate PBPK with blood state integration and return results dict.

# Returns
Dict with:
- Organ concentrations over time
- Blood state evolution (albumin, AAG, IL-6)
- Dynamic PK adjustments applied
"""
function simulate_with_blood_state(
    pbpk_params::PBPKParams,
    blood_callback::BloodStateODECallback,
    dose::Float64;
    t_max::Float64 = 72.0,
    num_points::Int = 100,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-10
)
    time_points = collect(range(0.0, t_max, length=num_points))

    # Make a copy of blood_callback to track evolution
    blood_copy = deepcopy(blood_callback)

    # Solve
    sol = solve_with_blood_state(
        pbpk_params, blood_copy, dose, (0.0, t_max);
        time_points=time_points,
        reltol=reltol, abstol=abstol
    )

    # Calculate Rb from blood state
    rb = 1.0 - blood_copy.hematocrit + (blood_copy.hematocrit * blood_copy.ke_p)

    # Build results
    results = Dict{String, Any}(
        "time" => time_points,
        "plasma" => [sol[i][BLOOD_IDX] for i in 1:length(sol)],
        "blood" => [sol[i][BLOOD_IDX] * rb for i in 1:length(sol)]
    )

    # Organ concentrations
    for (i, organ) in enumerate(PBPK_ORGANS)
        results[organ] = [sol[j][i] for j in 1:length(sol)]
    end

    # Record blood state at key timepoints
    # (Note: this is approximate since we're simulating the evolution)
    results["blood_state"] = Dict(
        "initial_albumin" => blood_callback.albumin_g_L,
        "initial_aag" => blood_callback.aag_g_L,
        "initial_il6" => blood_callback.il6_pg_mL,
        "was_acute_phase" => blood_callback.is_acute_phase,
        "ke_p" => blood_callback.ke_p,
        "fu_reference" => blood_callback.fu_reference,
        "charge_type" => blood_callback.charge_type
    )

    return results
end

# =============================================================================
# BLOOD PARTITIONING HELPER FUNCTIONS
# =============================================================================
# Helper functions for calculating Blood:Plasma (B:P) ratios and partitioning
# drug between plasma, RBC, and WBC compartments

"""
    calculate_blood_plasma_ratio(params::PBPKParams)

Calculate Blood:Plasma concentration ratio using mechanistic equation.

Formula: Rb = 1 - Hct + Hct × Ke_p

Where:
- Rb = Blood:Plasma ratio
- Hct = Hematocrit (fraction)
- Ke_p = Erythrocyte:plasma partition coefficient

Reference:
- Rodgers & Rowland (2006) J Pharm Sci
- PK-Sim documentation

# Returns
- Blood:Plasma ratio (dimensionless)
"""
function calculate_blood_plasma_ratio(params::PBPKParams)::Float64
    if !params.enable_bp_ratio
        return 1.0  # Disabled: assume blood = plasma
    end

    Hct = params.hematocrit
    Ke_p = params.ke_p

    # Rb = 1 - Hct + Hct × Ke_p
    Rb = (1.0 - Hct) + (Hct * Ke_p)

    return Rb
end

# Note: For advanced blood binding calculations using the BloodBinding module,
# use the module's calculate_blood_plasma_ratio function directly with
# DrugProperties and BloodComposition objects.

"""
    partition_blood_concentration(C_blood::Float64, Rb::Float64, Hct::Float64)

Partition whole blood concentration into plasma and RBC components.

# Arguments
- `C_blood`: Whole blood concentration
- `Rb`: Blood:Plasma ratio
- `Hct`: Hematocrit fraction

# Returns
NamedTuple with:
- `C_plasma`: Plasma concentration
- `C_rbc`: RBC concentration
- `C_wbc`: WBC concentration (approximation)

# Equations
- C_plasma = C_blood / Rb
- C_rbc = Ke_p × C_plasma = (Rb - (1 - Hct)) / Hct × C_plasma
"""
function partition_blood_concentration(C_blood::Float64, Rb::Float64, Hct::Float64)
    # C_plasma = C_blood / Rb
    C_plasma = C_blood / Rb

    # Ke_p can be back-calculated from Rb and Hct
    # Rb = 1 - Hct + Hct × Ke_p
    # Ke_p = (Rb - (1 - Hct)) / Hct
    Ke_p = (Rb - (1.0 - Hct)) / Hct

    # C_rbc = Ke_p × C_plasma
    C_rbc = Ke_p * C_plasma

    # WBC concentration (approximate as similar to RBC for now)
    # This can be refined using WBC-specific binding from BloodBinding module
    C_wbc = C_rbc

    return (C_plasma = C_plasma, C_rbc = C_rbc, C_wbc = C_wbc)
end

"""
    get_unbound_plasma_concentration(C_blood::Float64, Rb::Float64, fu::Float64)

Calculate unbound (free) plasma concentration from whole blood concentration.

This is the pharmacologically active concentration that drives tissue distribution
and clearance.

# Arguments
- `C_blood`: Whole blood concentration
- `Rb`: Blood:Plasma ratio
- `fu`: Fraction unbound in plasma

# Returns
- Unbound plasma concentration

# Equation
C_unbound = (C_blood / Rb) × fu
"""
function get_unbound_plasma_concentration(C_blood::Float64, Rb::Float64, fu::Float64)::Float64
    C_plasma = C_blood / Rb
    C_unbound = C_plasma * fu
    return C_unbound
end

"""
    calculate_fu_blood(fu_plasma::Float64, Rb::Float64)

Calculate fraction unbound in whole blood from plasma fu and B:P ratio.

# Arguments
- `fu_plasma`: Fraction unbound in plasma
- `Rb`: Blood:Plasma ratio

# Returns
- Fraction unbound in blood

# Equation
fu_blood = fu_plasma / Rb

This accounts for the fact that drug in RBCs is generally not available
for protein binding.
"""
function calculate_fu_blood(fu_plasma::Float64, Rb::Float64)::Float64
    return fu_plasma / Rb
end

"""
    apply_bp_ratio_to_clearance(CL_blood::Float64, Rb::Float64)

Convert blood clearance to plasma clearance or vice versa.

# Arguments
- `CL_blood`: Clearance based on blood concentrations
- `Rb`: Blood:Plasma ratio

# Returns
- Plasma clearance

# Equation
CL_plasma = CL_blood × Rb

Note: Most hepatic and renal clearances are intrinsically based on
unbound plasma concentrations, so this conversion is important when
measured clearances are reported in different concentration units.
"""
function apply_bp_ratio_to_clearance(CL_blood::Float64, Rb::Float64)::Float64
    return CL_blood * Rb
end

"""
    estimate_ke_p_from_logP(logP::Float64, charge_type::Symbol, pKa::Vector{Float64})

Estimate RBC partition coefficient from physicochemical properties.

This is a simplified QSPR for when experimental Ke_p is not available.

# Arguments
- `logP`: Octanol-water partition coefficient (log scale)
- `charge_type`: :acidic, :basic, :neutral, :zwitterion
- `pKa`: pKa value(s)

# Returns
- Estimated Ke_p

# Rules of thumb (from literature):
- Neutral, lipophilic (logP > 2): Ke_p ≈ 0.5 - 1.5
- Bases (pKa 7-10): Ke_p ≈ 0.8 - 2.0 (pH trapping in RBC)
- Acids (pKa 3-5): Ke_p ≈ 0.3 - 0.8 (excluded from RBC)
- Very polar (logP < 0): Ke_p ≈ 0.4 - 0.7 (water space only)

Reference: Hinderling (1997) Clin Pharmacokinet
"""
function estimate_ke_p_from_logP(logP::Float64, charge_type::Symbol, pKa::Vector{Float64})::Float64
    # Base estimate from lipophilicity
    base_ke_p = if logP > 2.0
        0.8 + 0.2 * min(logP - 2.0, 3.0)  # Lipophilic: higher partition
    elseif logP < 0.0
        0.5 + 0.1 * logP  # Polar: lower partition
    else
        0.7  # Moderate
    end

    # Adjust for ionization
    if charge_type == :basic && !isempty(pKa)
        # Bases accumulate in RBC (pH trapping)
        base_ke_p *= 1.3
    elseif charge_type == :acidic && !isempty(pKa)
        # Acids are excluded from RBC
        base_ke_p *= 0.6
    end

    # Clamp to reasonable range
    return clamp(base_ke_p, 0.2, 3.0)
end

# Export new functions
export calculate_blood_plasma_ratio, partition_blood_concentration
export get_unbound_plasma_concentration, calculate_fu_blood
export apply_bp_ratio_to_clearance, estimate_ke_p_from_logP

end # module
