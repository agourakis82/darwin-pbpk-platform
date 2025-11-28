"""
Fractal Kinetics Extension for MedLang

This module extends MedLang to support fractional pharmacokinetics,
implementing the deep fractal theory of drug distribution.

Key concepts:
- Mittag-Leffler response functions (replaces exponential)
- Spectral dimension corrections (Alexander-Orbach, d_s ≈ 4/3)
- Molecular-tissue fractal coupling
- Memory-dependent transport

References:
- Alexander & Orbach (1982) - Spectral dimension conjecture
- Kopelman (1988) - Fractal reaction kinetics
- West, Brown, Enquist (1997) - Allometric scaling
- Dokoumetzidis & Macheras (2009) - Fractional kinetics in PK

Author: Darwin PBPK Platform Development Team
Version: 1.0.0
"""
module FractalKinetics

using SpecialFunctions: gamma

export mittag_leffler, mittag_leffler_derivative
export fractional_decay, fractional_accumulation
export FractalCompartment, FractalPBPKParams
export tissue_fractal_dim, tissue_alpha, molecular_fractal_dim
export fractal_coupling, spectral_correction
export fractal_oie_tozer, estimate_fut_fractal
export ALEXANDER_ORBACH_DS

#=============================================================================
  CONSTANTS
=============================================================================#

"""
Alexander-Orbach spectral dimension.

The Alexander-Orbach conjecture states that for percolation networks,
the spectral dimension d_s ≈ 4/3, independent of embedding dimension.

This has been proven for d > 6 (Kozma & Nachmias, 2009).
"""
const ALEXANDER_ORBACH_DS = 4/3

"""
Tissue fractal dimensions (Hausdorff dimension of vascular network).

Based on:
- Fractal analysis of vascular networks (Gazit et al., 1997)
- Tumor vascular architecture (Baish & Jain, 2000)
"""
const TISSUE_FRACTAL_DIM = Dict{Symbol, Float64}(
    :plasma => 3.0,      # Flowing compartment (Euclidean)
    :blood => 3.0,       # Well-mixed
    :liver => 2.85,      # Sinusoidal architecture
    :kidney => 2.88,     # Glomerular network
    :brain => 2.80,      # Tortuous ECS, BBB
    :lung => 2.97,       # Highly fractal alveolar surface
    :heart => 2.75,      # Coronary network
    :muscle => 2.70,     # Fibrous, moderate vessels
    :skin => 2.50,       # Dermal capillaries
    :adipose => 2.40,    # Sparse vascularization
    :bone => 2.30,       # Limited vascularization
    :gut => 2.70,        # Intestinal villi
    :spleen => 2.60,     # Red pulp
    :tumor => 2.45       # Chaotic, tortuous vasculature
)

"""
Tissue heterogeneity parameter (fractional order α).

α = 1: Homogeneous (classical first-order kinetics)
α < 1: Heterogeneous (anomalous kinetics with memory)

Based on:
- Fractional kinetics analysis (Dokoumetzidis & Macheras, 2009)
- Tissue microstructure studies
"""
const TISSUE_ALPHA = Dict{Symbol, Float64}(
    :plasma => 1.0,      # Well-mixed
    :blood => 0.98,      # Nearly homogeneous
    :liver => 0.90,      # Sinusoidal, some heterogeneity
    :kidney => 0.88,     # Glomerular filtering
    :brain => 0.60,      # Highly heterogeneous ECS
    :lung => 0.92,       # Regular alveolar structure
    :heart => 0.85,      # Myocardial fibers
    :muscle => 0.80,     # Fibrous structure
    :skin => 0.75,       # Layered structure
    :adipose => 0.70,    # Heterogeneous fat cells
    :bone => 0.65,       # Highly heterogeneous
    :gut => 0.78,        # Villi structure
    :spleen => 0.72,     # Pulp structure
    :tumor => 0.50       # Chaotic, highly heterogeneous
)

#=============================================================================
  MITTAG-LEFFLER FUNCTIONS
=============================================================================#

"""
    mittag_leffler(z, α; max_terms=200, tol=1e-15) -> Float64

Single-parameter Mittag-Leffler function E_α(z).

E_α(z) = Σ_{k=0}^∞ z^k / Γ(αk + 1)

Properties:
- E₁(z) = exp(z) (classical exponential)
- For 0 < α < 1:
  - Short times: ≈ exp(z/Γ(1+α)) (stretched exponential)
  - Long times: ≈ -1/(z × Γ(1-α)) (power law)

# Arguments
- `z::Float64`: Argument (typically -k×t^α for decay)
- `α::Float64`: Fractional order (0 < α ≤ 2)
- `max_terms::Int`: Maximum series terms
- `tol::Float64`: Convergence tolerance

# Returns
- `Float64`: E_α(z)

# Example
```julia
# Classical exponential decay
t = 1.0
k = 0.5
c_classical = mittag_leffler(-k*t, 1.0)  # = exp(-0.5)

# Fractional decay with memory
α = 0.8
c_fractal = mittag_leffler(-k*t^α, α)
```
"""
function mittag_leffler(z::Float64, α::Float64; max_terms::Int=200, tol::Float64=1e-15)::Float64
    if α <= 0 || α > 2
        throw(ArgumentError("α must be in (0, 2], got $α"))
    end

    # Special case: α = 1 is exponential
    if abs(α - 1.0) < 1e-10
        return exp(z)
    end

    result = 0.0

    for k in 0:max_terms
        term = z^k / gamma(α * k + 1)
        result += term

        # Convergence check
        if k > 10 && abs(term) < tol * abs(result)
            break
        end

        # Guard against overflow/underflow
        if !isfinite(term) || !isfinite(result)
            break
        end
    end

    return result
end

"""
    mittag_leffler(z, α, β; max_terms=200, tol=1e-15) -> Float64

Two-parameter Mittag-Leffler function E_{α,β}(z).

E_{α,β}(z) = Σ_{k=0}^∞ z^k / Γ(αk + β)

This form appears in solutions of fractional differential equations.

# Arguments
- `z::Float64`: Argument
- `α::Float64`: First parameter (0 < α ≤ 2)
- `β::Float64`: Second parameter (β > 0)

# Returns
- `Float64`: E_{α,β}(z)
"""
function mittag_leffler(z::Float64, α::Float64, β::Float64; max_terms::Int=200, tol::Float64=1e-15)::Float64
    if α <= 0 || α > 2
        throw(ArgumentError("α must be in (0, 2], got $α"))
    end
    if β <= 0
        throw(ArgumentError("β must be positive, got $β"))
    end

    result = 0.0

    for k in 0:max_terms
        term = z^k / gamma(α * k + β)
        result += term

        if k > 10 && abs(term) < tol * abs(result)
            break
        end

        if !isfinite(term) || !isfinite(result)
            break
        end
    end

    return result
end

"""
    mittag_leffler_derivative(z, α; max_terms=200) -> Float64

Derivative of the Mittag-Leffler function: dE_α(z)/dz.

Useful for computing fractional kinetics derivatives.
"""
function mittag_leffler_derivative(z::Float64, α::Float64; max_terms::Int=200)::Float64
    if α <= 0 || α > 2
        throw(ArgumentError("α must be in (0, 2], got $α"))
    end

    result = 0.0

    for k in 1:max_terms
        term = k * z^(k-1) / gamma(α * k + 1)
        result += term

        if k > 10 && abs(term) < 1e-15 * abs(result)
            break
        end
    end

    return result
end

#=============================================================================
  FRACTIONAL KINETICS
=============================================================================#

"""
    fractional_decay(t, k, α) -> Float64

Fractional exponential decay using Mittag-Leffler function.

C(t) = C₀ × E_α(-k × t^α)

This replaces classical exponential decay C(t) = C₀ × exp(-kt) when α < 1.

# Arguments
- `t::Float64`: Time
- `k::Float64`: Rate constant
- `α::Float64`: Fractional order (1 = classical, <1 = memory effects)

# Returns
- `Float64`: Fraction remaining at time t

# Properties
- t << 1: Stretched exponential behavior
- t >> 1: Power law t^(-α) tail (no true half-life!)

# Example
```julia
# Classical vs fractional elimination
t_half = 10.0
k = log(2) / t_half

# Classical: exactly 50% at t_half
c_classical = fractional_decay(t_half, k, 1.0)  # ≈ 0.5

# Fractional: more than 50% remains due to memory
c_fractal = fractional_decay(t_half, k, 0.8)  # > 0.5
```
"""
function fractional_decay(t::Float64, k::Float64, α::Float64)::Float64
    if t < 0
        return 1.0
    end
    if t == 0
        return 1.0
    end

    return mittag_leffler(-k * t^α, α)
end

"""
    fractional_accumulation(τ, k, α, n_doses) -> Float64

Accumulation factor after repeated dosing with interval τ.

Unlike classical kinetics where accumulation approaches steady state,
fractional kinetics shows continued slow accumulation (power law).

# Arguments
- `τ::Float64`: Dosing interval
- `k::Float64`: Rate constant
- `α::Float64`: Fractional order
- `n_doses::Int`: Number of doses

# Returns
- `Float64`: Accumulation factor (total/single dose)

# Clinical Implication
For α < 1, accumulation continues beyond 5 "half-lives".
Loading doses may need adjustment.
"""
function fractional_accumulation(τ::Float64, k::Float64, α::Float64, n_doses::Int)::Float64
    acc = 0.0
    for i in 0:n_doses-1
        t = i * τ
        acc += fractional_decay(t, k, α)
    end
    return acc
end

"""
    effective_half_life(k, α; target=0.5) -> Float64

Compute effective half-life for fractional kinetics.

Note: True half-life doesn't exist for α < 1 (power law tail).
This returns time to reach target fraction remaining.

# Arguments
- `k::Float64`: Rate constant
- `α::Float64`: Fractional order
- `target::Float64`: Target fraction (default 0.5)

# Returns
- `Float64`: Time to reach target fraction
"""
function effective_half_life(k::Float64, α::Float64; target::Float64=0.5)::Float64
    # Binary search
    t_low, t_high = 0.0, 1000.0

    for _ in 1:100
        t_mid = (t_low + t_high) / 2
        decay = fractional_decay(t_mid, k, α)

        if decay > target
            t_low = t_mid
        else
            t_high = t_mid
        end

        if abs(decay - target) < 0.001
            break
        end
    end

    return (t_low + t_high) / 2
end

#=============================================================================
  FRACTAL TISSUE PARAMETERS
=============================================================================#

"""
    tissue_fractal_dim(tissue::Symbol) -> Float64

Get fractal dimension for a tissue.

The Hausdorff dimension of the vascular network.
d_f = 3 for Euclidean (normal), d_f < 3 for fractal.
"""
function tissue_fractal_dim(tissue::Symbol)::Float64
    return get(TISSUE_FRACTAL_DIM, tissue, 2.70)
end

"""
    tissue_alpha(tissue::Symbol) -> Float64

Get fractional order (heterogeneity) for a tissue.

α = 1: homogeneous (classical kinetics)
α < 1: heterogeneous (memory effects)
"""
function tissue_alpha(tissue::Symbol)::Float64
    return get(TISSUE_ALPHA, tissue, 0.80)
end

"""
    molecular_fractal_dim(MW, RB, TPSA, HBD, HBA) -> Float64

Estimate molecular fractal dimension from descriptors.

Based on the observation that molecular fragments show self-similarity.

# Arguments
- `MW::Float64`: Molecular weight
- `RB::Float64`: Rotatable bonds
- `TPSA::Float64`: Topological polar surface area
- `HBD::Float64`: H-bond donors
- `HBA::Float64`: H-bond acceptors

# Returns
- `Float64`: Estimated molecular fractal dimension (2.0-2.6)
"""
function molecular_fractal_dim(MW::Float64, RB::Float64, TPSA::Float64, HBD::Float64, HBA::Float64)::Float64
    # Surface area / volume scaling for fractals
    # For fractals: S ∝ V^(d_f/3) instead of V^(2/3)
    surface_vol_ratio = (TPSA + 1) / (MW^(2/3) + 1)

    # Branching contributes to fractal complexity
    branching_factor = 1 + 0.1 * RB / 10

    # Base estimate from scaling
    d_f_base = 2.0 + 0.3 * log(1 + surface_vol_ratio)
    d_f = d_f_base * branching_factor

    return clamp(d_f, 2.0, 2.6)
end

#=============================================================================
  FRACTAL COUPLING
=============================================================================#

"""
    fractal_coupling(d_f_mol, d_f_tissue; σ=0.3) -> Float64

Compute molecular-tissue fractal coupling efficiency.

η = exp(-|d_f(molecule) - d_f(tissue)|² / σ²)

Molecules with fractal dimensions matching tissue architecture
distribute more efficiently. This is "like dissolves like" in fractal space.

# Arguments
- `d_f_mol::Float64`: Molecular fractal dimension
- `d_f_tissue::Float64`: Tissue fractal dimension
- `σ::Float64`: Coupling width parameter

# Returns
- `Float64`: Coupling efficiency (0 to 1)
"""
function fractal_coupling(d_f_mol::Float64, d_f_tissue::Float64; σ::Float64=0.3)::Float64
    mismatch = abs(d_f_mol - d_f_tissue)
    return exp(-mismatch^2 / σ^2)
end

"""
    spectral_correction(d_f) -> Float64

Compute spectral dimension correction for subdiffusive transport.

Based on Alexander-Orbach conjecture: d_s ≈ 4/3 for fractal networks.

The correction factor accounts for slower-than-Fickian diffusion.
"""
function spectral_correction(d_f::Float64)::Float64
    # Walk dimension: d_w = 2*d_f/d_s
    d_w = 2 * d_f / ALEXANDER_ORBACH_DS

    # Correction factor for subdiffusion
    # Normal: d_w = 2, factor = 1
    # Subdiffusion: d_w > 2, factor < 1
    return sqrt(2 / d_w)
end

#=============================================================================
  FRACTAL VOLUME OF DISTRIBUTION
=============================================================================#

"""
    estimate_fut_fractal(logD, tissue; P_scale=0.1) -> Float64

Estimate fraction unbound in tissue using fractal transport model.

For fractal systems, effective tissue binding depends on:
- Partition coefficient (lipophilicity)
- Tissue fractional order (heterogeneity)
- Transport limitations (spectral dimension)

# Arguments
- `logD::Float64`: Log distribution coefficient at pH 7.4
- `tissue::Symbol`: Target tissue
- `P_scale::Float64`: Partitioning scale factor

# Returns
- `Float64`: Estimated fut (0.001 to 1.0)
"""
function estimate_fut_fractal(logD::Float64, tissue::Symbol=:muscle; P_scale::Float64=0.1)::Float64
    P = 10^logD
    α = tissue_alpha(tissue)
    d_f = tissue_fractal_dim(tissue)

    # Classical tissue binding
    fut_classical = 1 / (1 + P_scale * P)

    # Fractal correction: heterogeneous tissues have more binding sites
    # accessible through fractal network
    fractal_factor = (d_f / 3)^α

    fut = fut_classical * fractal_factor
    return clamp(fut, 0.001, 1.0)
end

"""
    fractal_oie_tozer(fup, fut, d_f_mol; body_weight=70.0) -> Float64

Fractal-corrected Øie-Tozer volume of distribution.

Vdss = Vp + Ve × (fup/fut)^(d_s/2) × η + Vr × (fup/fut) × (d_f/3)^α × η

This extends classical Øie-Tozer to incorporate:
- Spectral dimension (d_s) correction for subdiffusion
- Fractal geometry (d_f) with tissue heterogeneity (α)
- Molecular-tissue coupling (η)

# Arguments
- `fup::Float64`: Fraction unbound in plasma
- `fut::Float64`: Fraction unbound in tissue
- `d_f_mol::Float64`: Molecular fractal dimension

# Returns
- `Float64`: Vdss in L/kg
"""
function fractal_oie_tozer(fup::Float64, fut::Float64, d_f_mol::Float64; body_weight::Float64=70.0)::Float64
    # Standard volumes (L) for 70 kg human
    Vp = 3.0    # Plasma
    Ve = 12.0   # Extracellular fluid
    Vr = 27.0   # Remaining tissue volume

    # Effective tissue parameters (volume-weighted average)
    tissues = [:muscle, :adipose, :liver, :brain, :skin, :kidney]
    weights = [0.40, 0.20, 0.03, 0.02, 0.10, 0.005]

    d_f_tissue = sum(tissue_fractal_dim(t) * w for (t, w) in zip(tissues, weights)) / sum(weights)
    α_tissue = sum(tissue_alpha(t) * w for (t, w) in zip(tissues, weights)) / sum(weights)

    # Fractal corrections
    spectral_corr = sqrt(ALEXANDER_ORBACH_DS / 2)
    fractal_geom = (d_f_tissue / 3)^α_tissue
    η = fractal_coupling(d_f_mol, d_f_tissue)

    # Fractional Øie-Tozer
    fup_fut = fup / (fut + 1e-6)

    Vdss = Vp +
           Ve * fup_fut^spectral_corr * η +
           Vr * fup_fut * fractal_geom * η

    # Scale to body weight and convert to L/kg
    scale = body_weight / 70.0
    return (Vdss * scale) / body_weight
end

#=============================================================================
  FRACTAL COMPARTMENT TYPE
=============================================================================#

"""
    FractalCompartment

A compartment with fractal kinetics.

Unlike classical compartments with exponential kinetics,
fractal compartments use Mittag-Leffler response.

# Fields
- `name::String`: Compartment name
- `volume::Float64`: Volume (L)
- `alpha::Float64`: Fractional order (1 = classical)
- `d_f::Float64`: Hausdorff dimension of vascular network
- `initial::Float64`: Initial amount/concentration
"""
struct FractalCompartment
    name::String
    volume::Float64
    alpha::Float64
    d_f::Float64
    initial::Float64

    function FractalCompartment(name::String, volume::Float64;
                                alpha::Float64=1.0, d_f::Float64=3.0, initial::Float64=0.0)
        if alpha <= 0 || alpha > 1
            throw(ArgumentError("alpha must be in (0, 1], got $alpha"))
        end
        if d_f < 2 || d_f > 3
            throw(ArgumentError("d_f must be in [2, 3], got $d_f"))
        end
        new(name, volume, alpha, d_f, initial)
    end
end

"""
    FractalPBPKParams

PBPK parameters with fractal kinetics support.

Extends standard PBPKParams to include:
- Per-tissue fractional orders
- Per-tissue fractal dimensions
- Molecular fractal dimension
- Global spectral dimension
"""
struct FractalPBPKParams
    # Standard PBPK
    volumes::Dict{String, Float64}
    blood_flows::Dict{String, Float64}
    partition_coeffs::Dict{String, Float64}
    clearances::Dict{String, Float64}

    # Fractal extensions
    tissue_alpha::Dict{String, Float64}
    tissue_d_f::Dict{String, Float64}
    mol_d_f::Float64
    d_s::Float64  # Spectral dimension (default 4/3)
end

"""
Constructor for FractalPBPKParams with defaults from tissue tables.
"""
function FractalPBPKParams(;
    volumes::Dict{String, Float64},
    blood_flows::Dict{String, Float64},
    partition_coeffs::Dict{String, Float64},
    clearances::Dict{String, Float64},
    mol_d_f::Float64=2.3
)
    # Initialize tissue parameters from tables
    tissue_alpha = Dict{String, Float64}()
    tissue_d_f = Dict{String, Float64}()

    for organ in keys(volumes)
        sym = Symbol(lowercase(organ))
        tissue_alpha[organ] = get(TISSUE_ALPHA, sym, 0.80)
        tissue_d_f[organ] = get(TISSUE_FRACTAL_DIM, sym, 2.70)
    end

    return FractalPBPKParams(
        volumes, blood_flows, partition_coeffs, clearances,
        tissue_alpha, tissue_d_f, mol_d_f, ALEXANDER_ORBACH_DS
    )
end

#=============================================================================
  MEDLANG INTEGRATION
=============================================================================#

"""
Reserved keywords for fractal kinetics in MedLang.

Example MedLang syntax:
```
model FractalDrug {
    kinetics fractional {
        alpha: 0.8
        d_f_mol: 2.3
    }

    compartment plasma {
        V: 3.0_L
        alpha: 1.0  # Well-mixed
    }

    compartment muscle {
        V: 28.0_L
        alpha: 0.8   # Heterogeneous
        d_f: 2.7     # Vascular fractal dim
    }
}
```
"""
const FRACTAL_KEYWORDS = Set([
    "fractional", "alpha", "d_f", "d_s",
    "mittag_leffler", "spectral_dim",
    "fractal_coupling", "memory"
])

export FRACTAL_KEYWORDS

end # module
